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
         + λ_chamfer · L_chamfer  +  λ_repulsion · L_repulsion

   - L_velocity:  MSE between predicted and Sinkhorn-aligned GT velocity
   - L_sinkhorn:  Sinkhorn divergence between flowed template and target
                   (ensures global distributional match)
   - L_template_reg:  mild regulariser keeping the template near a target
                       radius (prevents collapse / explosion)
   - L_chamfer:  bidirectional Chamfer distance between predicted endpoint
                  and target (direct surface matching)
   - L_repulsion: penalises output point clustering (uniform spacing)
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
    n_iters: int = 50,
    reg: float = 0.01,
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
    sinkhorn_iters: int = 50,
    sinkhorn_reg: float = 0.01,
    use_hard_assignment: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use Sinkhorn OT to find the best permutation of `target` that
    aligns it to `template`, then apply it.

    Parameters
    ----------
    target   : [B, N, 3]  — unordered target point cloud
    template : [B, N, 3]  — ordered canonical template
    use_hard_assignment : bool
        If True, convert soft assignment to hard 1-to-1 via argmax.
        This prevents centroid-averaging that kills shape structure.

    Returns
    -------
    target_perm : [B, N, 3]  — target points reordered to match template
    P           : [B, N, N]  — the assignment matrix (soft or hard)
    """
    cost = pairwise_dist_sq(template, target)                    # [B, N, N]
    P = sinkhorn_assignment(cost, n_iters=sinkhorn_iters, reg=sinkhorn_reg)

    if use_hard_assignment:
        # Hard 1-to-1: for each template point, pick the target point with
        # highest assignment weight → crisp pairing, no centroid averaging
        hard_idx = P.argmax(dim=-1)                              # [B, N]
        target_perm = torch.gather(
            target, 1, hard_idx.unsqueeze(-1).expand(-1, -1, 3)
        )                                                        # [B, N, 3]
    else:
        # Soft permutation: target_perm_i = Σ_j  P_ij · target_j
        target_perm = torch.bmm(P, target)                       # [B, N, 3]

    return target_perm, P


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------

def chamfer_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Bidirectional Chamfer distance between point sets.

    a : [B, N, 3]
    b : [B, M, 3]
    returns : scalar
    """
    dist = pairwise_dist_sq(a, b)                                # [B, N, M]
    # a→b: for each point in a, distance to closest in b
    loss_ab = dist.min(dim=2).values.mean()
    # b→a: for each point in b, distance to closest in a
    loss_ba = dist.min(dim=1).values.mean()
    return loss_ab + loss_ba


def repulsion_loss(points: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    Repulsion loss: penalise nearby output points to encourage uniform spacing.

    points : [B, N, 3]
    k : int — number of nearest neighbours to consider
    returns : scalar
    """
    dist = pairwise_dist_sq(points, points)                      # [B, N, N]
    # Get k+1 nearest (includes self at dist=0), take indices 1..k
    knn_dists = dist.topk(k + 1, dim=-1, largest=False).values[:, :, 1:]  # [B, N, k]
    # Inverse distance penalty (closer points → higher penalty)
    return (1.0 / (knn_dists + 1e-8)).mean()


# ---------------------------------------------------------------------------
# Flow Matcher
# ---------------------------------------------------------------------------

class RectifiedFlowMatcher(nn.Module):
    """
    Training-time module that:
    1. Samples a random time t ~ U(0, 1).
    2. FPS-subsamples both template and target to train_n_points.
    3. Computes Sinkhorn-aligned target permutation.
    4. Constructs the straight-line interpolant x_t.
    5. Computes the velocity matching loss + all auxiliary losses.

    Parameters
    ----------
    model : nn.Module
        The FlowTransformer (or any module with the same forward signature).
    lambda_ot : float
        Weight for the Sinkhorn divergence loss.
    lambda_reg : float
        Weight for template regularisation.
    lambda_chamfer : float
        Weight for chamfer distance loss.
    lambda_repulsion : float
        Weight for repulsion loss.
    sinkhorn_iters : int
        Iterations for the assignment Sinkhorn.
    sinkhorn_reg : float
        Entropic regularisation for the assignment.
    ot_blur : float
        Blur (length-scale) for geomloss SamplesLoss.
    train_n_points : int
        FPS subsample size during training.  Set to 0 to disable.
    use_hard_assignment : bool
        If True, use hard 1-to-1 assignment after Sinkhorn.
    template_reg_radius : float
        Target radius for template regularisation (default 1.5).
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ot: float = 0.1,
        lambda_reg: float = 0.01,
        lambda_chamfer: float = 0.1,
        lambda_repulsion: float = 0.01,
        sinkhorn_iters: int = 50,
        sinkhorn_reg: float = 0.01,
        ot_blur: float = 0.05,
        train_n_points: int = 2048,
        use_hard_assignment: bool = True,
        template_reg_radius: float = 1.5,
    ):
        super().__init__()
        self.model = model
        self.lambda_ot = lambda_ot
        self.lambda_reg = lambda_reg
        self.lambda_chamfer = lambda_chamfer
        self.lambda_repulsion = lambda_repulsion
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_reg = sinkhorn_reg
        self.use_hard_assignment = use_hard_assignment
        self.template_reg_radius = template_reg_radius
        # NOTE: FPS subsampling from 15k→2048 now happens in the dataset
        # (CPU-side, in __getitem__).  This field is kept as a safety fallback:
        # if data arrives with more points than train_n_points, the FPS code
        # in forward() will still activate.  Under normal usage, data arrives
        # at n_points=2048 and no further subsampling is needed.
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
            loss           — total scalar loss
            loss_velocity  — velocity MSE
            loss_ot        — Sinkhorn divergence
            loss_reg       — template regularisation
            loss_chamfer   — Chamfer distance
            loss_repulsion — repulsion loss
        """
        B, N, _ = target.shape
        device = target.device

        # 1. Sample random time  t ∈ (0, 1)  (avoid exact endpoints)
        t = torch.rand(B, device=device).clamp(1e-5, 1 - 1e-5)  # [B]

        # 2. Get template and FPS-subsample both template and target
        n_sub = self.train_n_points
        template_full = self.model.get_template(B)               # [B, N_tmpl, 3]

        if n_sub > 0 and n_sub < N:
            # --- FPS subsample target (from 15k → 2048) ---
            fps_idx_tgt = farthest_point_sample(target, n_sub)   # [B, n_sub]
            target_sub = fps_gather(target, fps_idx_tgt)         # [B, n_sub, 3]

            # --- FPS subsample template (if template is larger than n_sub) ---
            N_tmpl = template_full.shape[1]
            if n_sub < N_tmpl:
                fps_idx_tmpl = farthest_point_sample(template_full, n_sub)
                template_sub = fps_gather(template_full, fps_idx_tmpl)
            else:
                template_sub = template_full
        else:
            # No subsampling — use everything
            target_sub = target
            template_sub = template_full

        # 3. Sinkhorn alignment: reorder target_sub to match template_sub ordering
        target_perm, P = permute_target(
            target_sub, template_sub,
            sinkhorn_iters=self.sinkhorn_iters,
            sinkhorn_reg=self.sinkhorn_reg,
            use_hard_assignment=self.use_hard_assignment,
        )

        # 4. Ground-truth velocity (straight-line flow)
        v_gt = target_perm - template_sub                        # [B, n_sub, 3]

        # 5. Interpolant  x_t = (1 − t) template + t target_perm
        t_ = t[:, None, None]                                    # [B,1,1]
        x_t = (1 - t_) * template_sub + t_ * target_perm

        # 6. Encode target (use the subsampled version for the latent)
        _, z = self.model.encode(target_sub)

        # 7. Predict velocity
        v_pred = self.model.predict_velocity(x_t, t, z)

        # ===== Losses =====

        # (a) Velocity matching MSE
        loss_velocity = F.mse_loss(v_pred, v_gt)

        # (b) Sinkhorn divergence between the flowed template and target
        #     Gradients flow through v_pred → model gets OT signal.
        #     Target is detached (ground truth should not be pulled).
        x_1_approx = template_sub + v_pred                       # predicted endpoint
        loss_ot = self.ot_loss_fn(x_1_approx, target_sub.detach()).mean()

        # (c) Chamfer distance: direct surface matching
        loss_chamfer = chamfer_loss(x_1_approx, target_sub.detach())

        # (d) Repulsion loss: uniform point spacing
        loss_repulsion = repulsion_loss(x_1_approx, k=8)

        # (e) Template regularisation — keep ALL template points near target
        #     radius (not just the subsample) so the template stays
        #     well-conditioned.
        tmpl_for_reg = self.model.get_template(1)                # [1, N_full, 3]
        template_norms = tmpl_for_reg.norm(dim=-1)               # [1, N_full]
        loss_reg = ((template_norms - self.template_reg_radius) ** 2).mean()

        # Total
        loss = (
            loss_velocity
            + self.lambda_ot * loss_ot
            + self.lambda_reg * loss_reg
            + self.lambda_chamfer * loss_chamfer
            + self.lambda_repulsion * loss_repulsion
        )

        return {
            "loss": loss,
            "loss_velocity": loss_velocity,
            "loss_ot": loss_ot,
            "loss_reg": loss_reg,
            "loss_chamfer": loss_chamfer,
            "loss_repulsion": loss_repulsion,
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
    ) -> torch.Tensor:
        """
        Integrate the learned flow ODE from template (t=0) to shape (t=1).

        Both target and template are at the same resolution (e.g. 2048).
        The target is used only to compute the shape latent z.

        Parameters
        ----------
        target  : [B, M, 3]  — target shape (at inference resolution).
        n_steps : int         — number of integration steps
        method  : str         — 'euler' or 'midpoint'

        Returns
        -------
        trajectory : [n_steps+1, B, N_tmpl, 3]  — full flow trajectory
        """
        B = target.shape[0]
        _, z = self.model.encode(target)

        # Start from the full template (should be 2048 at this point)
        x = self.model.get_template(B)                           # [B, N_tmpl, 3]

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
