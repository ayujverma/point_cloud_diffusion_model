"""
Rectified Flow Matcher with Sinkhorn Alignment
================================================

This module implements the training objective for a Rotation-Equivariant
Rectified Flow model that establishes dense point correspondence.

Core ideas
----------
1. **Rectified Flow**: We train a velocity network v_θ(x_t, t) so that
   the ODE  dx/dt = v_θ(x, t)  transports the canonical template x_0
   to the target shape x_1 along *straight lines*:

       x_t = (1 − t) x_0  +  t x_1          (interpolation)
       v*(x_t, t) = x_1 − x_0                (ground-truth velocity)

   The training loss is simply  ‖v_θ − v*‖².  No ODE solver is needed
   during training (simulation-free), yielding ~10× faster convergence
   than the CNF approach in PointFlow.

2. **Sinkhorn Alignment**: Because the target point cloud is *unordered*,
   there is no canonical pairing between template point i and target
   point j.  We solve a soft Optimal Transport problem (Sinkhorn) to
   find the best permutation π before computing the velocity:

       v* = π(x_1) − x_0

   We use the `geomloss` library SamplesLoss for a differentiable
   Sinkhorn divergence, and additionally extract a soft assignment
   matrix via a custom Sinkhorn iteration for permuting the target.

3. **Total Loss**:
       L = L_velocity  +  λ_ot · L_sinkhorn  +  λ_reg · L_template_reg

   - L_velocity:  MSE between predicted and Sinkhorn-aligned GT velocity
   - L_sinkhorn:  Sinkhorn divergence between flowed template and target
                   (ensures global distributional match)
   - L_template_reg:  mild regulariser keeping the template near the unit
                       sphere (prevents collapse / explosion)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from geomloss import SamplesLoss
from core.point_ops import farthest_point_sample, fps_gather


# ---------------------------------------------------------------------------
# Sinkhorn utilities
# ---------------------------------------------------------------------------

def sinkhorn_assignment(
    cost: torch.Tensor,
    n_iters: int = 20,
    reg: float = 0.05,
) -> torch.Tensor:
    """
    Compute a soft doubly-stochastic assignment matrix via Sinkhorn iteration.

    Parameters
    ----------
    cost : [B, N, M]
        Pairwise cost matrix (e.g., squared Euclidean distance).
    n_iters : int
        Number of Sinkhorn iterations.
    reg : float
        Entropic regularisation (lower → sharper assignment, but less stable).

    Returns
    -------
    P : [B, N, M]
        Soft assignment (doubly-stochastic up to normalisation).
    """
    # Log-domain Sinkhorn for numerical stability
    log_K = -cost / reg                                          # [B, N, M]
    log_u = torch.zeros_like(log_K[:, :, 0])                    # [B, N]
    log_v = torch.zeros_like(log_K[:, 0, :])                    # [B, M]

    for _ in range(n_iters):
        # Row normalisation
        log_u = -torch.logsumexp(log_K + log_v[:, None, :], dim=2)
        # Column normalisation
        log_v = -torch.logsumexp(log_K + log_u[:, :, None], dim=1)

    # Recover assignment matrix
    log_P = log_K + log_u[:, :, None] + log_v[:, None, :]
    return log_P.exp()


def pairwise_dist_sq(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Squared Euclidean distance between two point sets.

    a : [B, N, 3]
    b : [B, M, 3]
    returns : [B, N, M]
    """
    # ||a_i - b_j||² = ||a_i||² + ||b_j||² - 2 a_i · b_j
    a_sq = (a * a).sum(dim=-1, keepdim=True)                     # [B, N, 1]
    b_sq = (b * b).sum(dim=-1, keepdim=True).transpose(1, 2)     # [B, 1, M]
    ab = torch.bmm(a, b.transpose(1, 2))                         # [B, N, M]
    return (a_sq + b_sq - 2 * ab).clamp(min=0.0)


def permute_target(
    target: torch.Tensor,
    template: torch.Tensor,
    sinkhorn_iters: int = 20,
    sinkhorn_reg: float = 0.05,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use Sinkhorn OT to find the best soft permutation of `target` that
    aligns it to `template`, then apply it.

    Parameters
    ----------
    target   : [B, N, 3]  — unordered target point cloud
    template : [B, N, 3]  — ordered canonical template

    Returns
    -------
    target_perm : [B, N, 3]  — target points reordered to match template
    P           : [B, N, N]  — the soft assignment matrix
    """
    cost = pairwise_dist_sq(template, target)                    # [B, N, N]
    P = sinkhorn_assignment(cost, n_iters=sinkhorn_iters, reg=sinkhorn_reg)
    # Apply soft permutation: target_perm_i = Σ_j  P_ij · target_j
    target_perm = torch.bmm(P, target)                           # [B, N, 3]
    return target_perm, P


# ---------------------------------------------------------------------------
# Flow Matcher
# ---------------------------------------------------------------------------

class RectifiedFlowMatcher(nn.Module):
    """
    Training-time module that:
    1. Samples a random time t ~ U(0, 1).
    2. Computes Sinkhorn-aligned target permutation.
    3. Constructs the straight-line interpolant x_t.
    4. Computes the velocity matching loss + Sinkhorn divergence.

    Parameters
    ----------
    model : nn.Module
        The FlowTransformer (or any module with the same forward signature).
    lambda_ot : float
        Weight for the Sinkhorn divergence loss.
    lambda_reg : float
        Weight for template regularisation.
    sinkhorn_iters : int
        Iterations for the assignment Sinkhorn.
    sinkhorn_reg : float
        Entropic regularisation for the assignment.
    ot_blur : float
        Blur (length-scale) for geomloss SamplesLoss.
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ot: float = 0.1,
        lambda_reg: float = 0.01,
        sinkhorn_iters: int = 20,
        sinkhorn_reg: float = 0.05,
        ot_blur: float = 0.05,
        train_n_points: int = 0,
    ):
        super().__init__()
        self.model = model
        self.lambda_ot = lambda_ot
        self.lambda_reg = lambda_reg
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_reg = sinkhorn_reg
        # If > 0, FPS-subsample template & target to this many points during
        # training.  Keeps memory at O(train_n_points²) even when the raw data
        # has 15k points.  Set to 0 to disable (use all points).
        self.train_n_points = train_n_points

        # Geomloss Sinkhorn divergence  (used for distributional loss, NOT for
        # the assignment — those are separate Sinkhorn calls).
        self.ot_loss_fn = SamplesLoss(
            loss="sinkhorn",
            p=2,
            blur=ot_blur,
            scaling=0.9,
            backend="tensorized",
        )

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def forward(self, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for one training batch.

        If ``self.train_n_points > 0``, both the template and the target are
        FPS-subsampled to ``train_n_points`` before computing the flow.  This
        is the key trick that lets us train on 15k-point datasets without
        running out of VRAM — the *shape* is captured at 2k points, and the
        velocity field generalises to any resolution at test time.

        Parameters
        ----------
        target : [B, N, 3]  — target point cloud batch (may be 15k points)

        Returns
        -------
        dict with keys:
            loss          — total scalar loss
            loss_velocity — velocity MSE
            loss_ot       — Sinkhorn divergence
            loss_reg      — template regularisation
        """
        B, N, _ = target.shape
        device = target.device

        # 1. Sample random time  t ∈ (0, 1)  (avoid exact endpoints)
        t = torch.rand(B, device=device).clamp(1e-5, 1 - 1e-5)  # [B]

        # 2. Get template and optionally subsample
        n_sub = self.train_n_points
        if n_sub > 0 and n_sub < N:
            # --- FPS subsample target ---
            fps_idx_tgt = farthest_point_sample(target, n_sub)   # [B, n_sub]
            target_sub = fps_gather(target, fps_idx_tgt)         # [B, n_sub, 3]

            # --- FPS subsample template ---
            template_full = self.model.get_template(B)           # [B, N_tmpl, 3]
            N_tmpl = template_full.shape[1]
            if n_sub < N_tmpl:
                fps_idx_tmpl = farthest_point_sample(template_full, n_sub)
                template_sub = fps_gather(template_full, fps_idx_tmpl)
            else:
                template_sub = template_full
        else:
            # No subsampling — use everything
            target_sub = target
            template_sub = self.model.get_template(B)
            template_full = template_sub

        # 3. Sinkhorn alignment: reorder target_sub to match template_sub ordering
        target_perm, P = permute_target(
            target_sub, template_sub,
            sinkhorn_iters=self.sinkhorn_iters,
            sinkhorn_reg=self.sinkhorn_reg,
        )

        # 4. Ground-truth velocity (straight-line flow)
        v_gt = target_perm - template_sub                        # [B, n_sub, 3]

        # 5. Interpolant  x_t = (1 − t) template + t target_perm
        t_ = t[:, None, None]                                    # [B,1,1]
        x_t = (1 - t_) * template_sub + t_ * target_perm

        # 6. Encode target (can use the subsampled version for the latent)
        _, z = self.model.encode(target_sub)

        # 7. Predict velocity
        v_pred = self.model.predict_velocity(x_t, t, z)

        # ===== Losses =====

        # (a) Velocity matching MSE
        loss_velocity = F.mse_loss(v_pred, v_gt)

        # (b) Sinkhorn divergence between the flowed template at t=1 and target
        #     This is an auxiliary distributional loss that encourages coverage.
        with torch.no_grad():
            # Approximate endpoint: template + v_pred  (single Euler step)
            x_1_approx = template_sub + v_pred
        loss_ot = self.ot_loss_fn(x_1_approx, target_sub)

        # (c) Template regularisation — keep ALL template points near the
        #     unit sphere (not just the subsample) so the full template
        #     stays well-conditioned.
        tmpl_for_reg = self.model.get_template(1)                # [1, N_full, 3]
        template_norms = tmpl_for_reg.norm(dim=-1)               # [1, N_full]
        loss_reg = ((template_norms - 1.0) ** 2).mean()

        # Total
        loss = loss_velocity + self.lambda_ot * loss_ot + self.lambda_reg * loss_reg

        return {
            "loss": loss,
            "loss_velocity": loss_velocity,
            "loss_ot": loss_ot,
            "loss_reg": loss_reg,
        }

    # ------------------------------------------------------------------
    # Inference: integrate the flow ODE  (Euler or midpoint)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        target: torch.Tensor,
        n_steps: int = 50,
        method: str = "euler",
        use_full_template: bool = False,
    ) -> torch.Tensor:
        """
        Integrate the learned flow ODE from template (t=0) to shape (t=1).

        The model was trained on subsampled (e.g. 2 048) point clouds, but
        the learned velocity field is *point-wise* conditioned on (x_t, t, z).
        By setting ``use_full_template=True`` you can flow the **entire**
        15 000-point template through the field — the decoder processes each
        point independently (aside from local KNN attention), so no retraining
        is required.

        Parameters
        ----------
        target             : [B, M, 3]  — target shape (needed to compute z).
                              M can be any number of points; the encoder will
                              handle it.
        n_steps            : int         — number of integration steps
        method             : str         — 'euler' or 'midpoint'
        use_full_template  : bool        — if True, flow the full-resolution
                              template (all points stored in the model).  If
                              False, flow a template of size M (for
                              visualisation with matched sizes).

        Returns
        -------
        trajectory : [n_steps+1, B, N_tmpl, 3]  — full flow trajectory
        """
        B = target.shape[0]
        _, z = self.model.encode(target)

        if use_full_template:
            x = self.model.get_template(B)                       # [B, N_full, 3]
        else:
            # Subsample template to match target size, for easy comparison
            N_target = target.shape[1]
            full_tmpl = self.model.get_template(B)
            if full_tmpl.shape[1] > N_target:
                fps_idx = farthest_point_sample(full_tmpl, N_target)
                x = fps_gather(full_tmpl, fps_idx)
            else:
                x = full_tmpl

        dt = 1.0 / n_steps
        trajectory = [x.clone()]

        for step in range(n_steps):
            t_val = step * dt
            t = torch.full((B,), t_val, device=x.device)

            if method == "euler":
                v = self.model.predict_velocity(x, t, z)
                x = x + dt * v

            elif method == "midpoint":
                # Evaluate at current point
                v1 = self.model.predict_velocity(x, t, z)
                # Midpoint
                x_mid = x + 0.5 * dt * v1
                t_mid = torch.full((B,), t_val + 0.5 * dt, device=x.device)
                v_mid = self.model.predict_velocity(x_mid, t_mid, z)
                x = x + dt * v_mid
            else:
                raise ValueError(f"Unknown integration method: {method}")

            trajectory.append(x.clone())

        return torch.stack(trajectory, dim=0)                    # [S+1, B, N, 3]
