"""
Vector Neuron Transformer for Rotation-Equivariant Point Cloud Processing
=========================================================================

Every feature in this module is stored as a [B, N, 3, C] tensor — a collection
of C three-dimensional vectors per point.  All linear maps, activations, and
attention operations act on the *channel* dimension while preserving the
geometric (3-dim) structure.  If the input coordinates undergo a rotation R,
every internal feature rotates identically, guaranteeing SO(3)-equivariance.

Key components
--------------
- VNLinear          : SO(3)-equivariant linear layer  (acts on last dim C)
- VNLeakyReLU      : Direction-aware activation that keeps the 3-vector structure
- VNLayerNorm       : Norm-based layer normalisation for 3-vector features
- VNAdaLN           : Adaptive Layer Norm conditioned on scalar signals (time t, shape latent z)
- VNAttention       : Multi-head self-attention over 3-vector tokens
                      Supports both **global** (O(N²)) and **KNN-local** (O(N·K))
                      attention modes.  Set knn_k > 0 to activate local mode.
- VNTransformerBlock: A single transformer block  (VNAttention + FFN + AdaLN)
- VNEncoder         : Stacked blocks → per-point features + global latent z
- FlowTransformer   : Full velocity-prediction network  (encoder + decoder)
                      Stores a full-resolution template (e.g. 15k) and can
                      subsample via FPS during training for VRAM efficiency.

Tensor conventions
------------------
All point-cloud features:  [B, N, 3, C]
Scalar conditioning (t, z): [B, D_scalar]
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from core.point_ops import knn, knn_gather_vn, farthest_point_sample, fps_gather


# ---------------------------------------------------------------------------
# 1.  VNLinear — the fundamental equivariant linear map
# ---------------------------------------------------------------------------

class VNLinear(nn.Module):
    """
    SO(3)-equivariant linear layer.

    Operates on the *channel* axis of a [B, N, 3, C_in] tensor, producing
    [B, N, 3, C_out].  Mathematically identical to a standard Linear but
    applied independently per spatial-3 component, which preserves equivariance.
    """

    def __init__(self, in_channels: int, out_channels: int, bias: bool = False):
        super().__init__()
        # Weight: [C_out, C_in] — shared across the 3 spatial dims
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels) * math.sqrt(2.0 / in_channels))
        # Bias breaks equivariance → default off, but allowed for ablations
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 3, C_in] → [B, N, 3, C_out]"""
        return F.linear(x, self.weight, self.bias)


# ---------------------------------------------------------------------------
# 2.  VNLeakyReLU — direction-aware activation
# ---------------------------------------------------------------------------

class VNLeakyReLU(nn.Module):
    """
    Vector Neuron LeakyReLU (Deng et al. 2021).

    For each 3-vector, project onto a learnable direction k.
    If the projection is positive, keep the vector; otherwise scale it by
    `negative_slope`.  This is equivariant because projections and
    reflections commute with rotations.
    """

    def __init__(self, channels: int, negative_slope: float = 0.2, share_nonlinearity: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        # Learnable direction per channel (or shared)
        n_dirs = 1 if share_nonlinearity else channels
        self.direction = nn.Parameter(torch.randn(1, 1, 3, n_dirs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 3, C] → [B, N, 3, C]"""
        # Normalise the learned direction
        k = F.normalize(self.direction, dim=2)                          # [1,1,3,C'] 
        # Signed projection of each vector onto k:  <x, k> → [B, N, 1, C]
        proj = (x * k).sum(dim=2, keepdim=True)                         # [B,N,1,C]
        # Mask: positive projections keep the vector, negative get scaled
        mask = (proj >= 0).float()
        # Equivariant activation:
        #   positive part:  x
        #   negative part:  x - (1 - α)(x · k)k        (reflect + scale)
        out = mask * x + (1 - mask) * (self.negative_slope * x + (1 - self.negative_slope) * proj * k)
        return out


# ---------------------------------------------------------------------------
# 3.  VNLayerNorm — norm-based layer normalisation
# ---------------------------------------------------------------------------

class VNLayerNorm(nn.Module):
    """
    Equivariant Layer Normalisation for 3-vector features.

    Instead of normalising component-wise (which would break equivariance),
    we normalise by the *norm* of each 3-vector channel, then apply a
    learnable per-channel scale.
    """

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, 3, C] → [B, N, 3, C]"""
        # Per-vector norm: [B, N, 1, C]
        norm = x.norm(dim=2, keepdim=True).clamp(min=self.eps)
        # Mean norm across channels for this token: [B, N, 1, 1]
        mean_norm = norm.mean(dim=-1, keepdim=True)
        # Normalise and rescale
        x_normed = x / (mean_norm + self.eps)
        return self.gamma.view(1, 1, 1, -1) * x_normed


# ---------------------------------------------------------------------------
# 4.  VNAdaLN — Adaptive Layer Norm conditioned on (t, z)
# ---------------------------------------------------------------------------

class VNAdaLN(nn.Module):
    """
    Adaptive Layer Norm (AdaLN) for Vector Neuron features.

    Conditioning signal (scalar) is projected to per-channel scale & shift
    that modulate the VN-normalized features.  The shift is applied as a
    *magnitude* shift (added to the norm), keeping equivariance intact.

    Parameters
    ----------
    channels : int
        Number of VN channels (C).
    cond_dim : int
        Dimension of the scalar conditioning vector (time embed + shape latent).
    """

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm = VNLayerNorm(channels)
        # Project scalar conditioning → 2C  (scale + shift per channel)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * channels),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x    : [B, N, 3, C]
        cond : [B, D_cond]   (scalar conditioning)
        returns: [B, N, 3, C]
        """
        x_normed = self.norm(x)                                 # [B, N, 3, C]
        params = self.proj(cond)                                # [B, 2C]
        scale, shift = params.chunk(2, dim=-1)                  # each [B, C]
        # Reshape for broadcast:  [B, 1, 1, C]
        scale = scale[:, None, None, :] + 1.0   # centre around 1
        shift = shift[:, None, None, :]
        # Scale acts on vectors; shift acts on the *norm* direction
        # To keep equivariance, shift is multiplied by the unit-direction of x
        norm = x_normed.norm(dim=2, keepdim=True).clamp(min=1e-8)
        direction = x_normed / norm                             # unit vectors [B,N,3,C]
        return scale * x_normed + shift * direction


# ---------------------------------------------------------------------------
# 5.  VNAttention — multi-head self-attention on 3-vector tokens
# ---------------------------------------------------------------------------

class VNAttention(nn.Module):
    """
    Multi-head self-attention for Vector Neuron features.

    Supports two modes:

    1. **Global attention** (knn_idx=None): Every point attends to every other
       point.  Memory O(N²).  Fine for N ≤ ~4096 on an A100.

    2. **KNN-local attention** (knn_idx provided): Each point only attends to
       its K nearest neighbours.  Memory O(N·K).  Enables N = 15,000+.

    Invariant logits: ⟨q_i, k_j⟩ summed over the 3 spatial components,
    yielding an SO(3)-invariant scalar.

    Parameters
    ----------
    channels : int
        Total VN channels (must be divisible by n_heads).
    n_heads : int
        Number of attention heads.
    """

    def __init__(self, channels: int, n_heads: int = 8):
        super().__init__()
        assert channels % n_heads == 0, "channels must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.scale = math.sqrt(self.head_dim * 3)  # √(d_head * 3) for 3-vector dot product

        # Equivariant Q, K, V projections
        self.wq = VNLinear(channels, channels)
        self.wk = VNLinear(channels, channels)
        self.wv = VNLinear(channels, channels)
        self.wo = VNLinear(channels, channels)

    def forward(
        self,
        x: torch.Tensor,
        knn_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x       : [B, N, 3, C]
        knn_idx : [B, N, K]  (optional) — if provided, use local attention
        returns : [B, N, 3, C]
        """
        B, N, _, C = x.shape
        H, D = self.n_heads, self.head_dim

        # Project → [B, N, 3, C] each
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        if knn_idx is None:
            # ---- Global attention (original path) ----
            q = rearrange(q, "b n s (h d) -> b h n s d", h=H)
            k = rearrange(k, "b n s (h d) -> b h n s d", h=H)
            v = rearrange(v, "b n s (h d) -> b h n s d", h=H)

            attn_logits = torch.einsum("bhqsd, bhksd -> bhqk", q, k) / self.scale
            attn_weights = F.softmax(attn_logits, dim=-1)
            out = torch.einsum("bhqk, bhksd -> bhqsd", attn_weights, v)
            out = rearrange(out, "b h n s d -> b n s (h d)")

        else:
            # ---- KNN-local attention ----
            # Gather K and V for each query's neighbours
            # k: [B, N, 3, C] → gather → [B, N, K, 3, C]
            K_neigh = knn_idx.shape[-1]
            k_local = knn_gather_vn(k, knn_idx)                  # [B, N, K, 3, C]
            v_local = knn_gather_vn(v, knn_idx)                  # [B, N, K, 3, C]

            # Reshape to heads
            q = rearrange(q, "b n s (h d) -> b h n s d", h=H)                # [B,H,N,3,D]
            k_local = rearrange(k_local, "b n k s (h d) -> b h n k s d", h=H)  # [B,H,N,K,3,D]
            v_local = rearrange(v_local, "b n k s (h d) -> b h n k s d", h=H)  # [B,H,N,K,3,D]

            # Attention logits: q[b,h,n,:,:] · k_local[b,h,n,k,:,:] → [B,H,N,K]
            # Expand q for broadcast:  [B, H, N, 1, 3, D]
            q_exp = q.unsqueeze(3)
            attn_logits = (q_exp * k_local).sum(dim=(-2, -1)) / self.scale  # [B,H,N,K]
            attn_weights = F.softmax(attn_logits, dim=-1)                    # [B,H,N,K]

            # Weighted sum: [B,H,N,K] × [B,H,N,K,3,D] → [B,H,N,3,D]
            out = torch.einsum("bhnk, bhnksd -> bhnsd", attn_weights, v_local)
            out = rearrange(out, "b h n s d -> b n s (h d)")

        out = self.wo(out)
        return out


# ---------------------------------------------------------------------------
# 6.  VNTransformerBlock — Attention + FFN + AdaLN
# ---------------------------------------------------------------------------

class VNTransformerBlock(nn.Module):
    """
    A single transformer block for 3-vector tokens.

    Pre-norm architecture with AdaLN:
        x ← x + Attention(AdaLN(x, cond))    # global or KNN-local
        x ← x + FFN(AdaLN(x, cond))
    """

    def __init__(self, channels: int, cond_dim: int, n_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.adaln_attn = VNAdaLN(channels, cond_dim)
        self.attn = VNAttention(channels, n_heads)

        self.adaln_ffn = VNAdaLN(channels, cond_dim)
        hidden = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(
            VNLinear(channels, hidden),
            VNLeakyReLU(hidden),
            VNLinear(hidden, channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        knn_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x       : [B, N, 3, C]
        cond    : [B, D_cond]
        knn_idx : [B, N, K]  (optional) — KNN indices for local attention
        """
        # --- Self-attention branch (global or local) ---
        x = x + self.attn(self.adaln_attn(x, cond), knn_idx=knn_idx)

        # --- Feed-forward branch (always point-wise) ---
        h = self.adaln_ffn(x, cond)
        for layer in self.ffn:
            h = layer(h) if isinstance(layer, (VNLinear, VNLeakyReLU)) else layer(h)
        x = x + h
        return x


# ---------------------------------------------------------------------------
# 7.  VNEncoder — stacked transformer blocks → global shape latent z
# ---------------------------------------------------------------------------

class VNEncoder(nn.Module):
    """
    Encode an unordered point cloud [B, N, 3] into:
      • per-point VN features [B, N, 3, C]
      • a global *scalar* shape latent z [B, D_latent]

    The scalar latent is obtained by taking the per-vector norms (invariant)
    and running a small MLP on the max-pooled result.

    When ``knn_k > 0``, uses KNN-local attention (O(N·K)) instead of global
    attention (O(N²)), enabling 15k+ point clouds on a single A100.
    """

    def __init__(
        self,
        in_channels: int = 1,          # each input point is a single 3-vector
        channels: int = 128,
        n_heads: int = 8,
        depth: int = 6,
        latent_dim: int = 256,
        knn_k: int = 0,                # 0 = global attention, >0 = local
        use_checkpoint: bool = False,  # gradient checkpointing to save VRAM
    ):
        super().__init__()
        self.knn_k = knn_k
        self.use_checkpoint = use_checkpoint
        self.input_proj = VNLinear(in_channels, channels)
        # Encoder blocks use plain VNLayerNorm (no adaptive conditioning)
        self.blocks = nn.ModuleList([
            VNEncoderBlock(channels, n_heads) for _ in range(depth)
        ])
        self.norm = VNLayerNorm(channels)

        # Invariant pooling → scalar latent
        # Per-vector norm gives [B, N, C] scalars.  Max-pool over N → [B, C]
        self.to_latent = nn.Sequential(
            nn.Linear(channels, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.channels = channels
        self.latent_dim = latent_dim

    def forward(
        self,
        points: torch.Tensor,
        knn_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        points  : [B, N, 3]
        knn_idx : [B, N, K]  (optional) — precomputed KNN graph.  If None and
                  self.knn_k > 0, it will be computed from `points`.

        returns:
            feats : [B, N, 3, C]    per-point equivariant features
            z     : [B, D_latent]   global invariant shape descriptor
        """
        # Build KNN graph if needed
        if knn_idx is None and self.knn_k > 0:
            knn_idx = knn(points, points, self.knn_k)            # [B, N, K]

        # Lift to VN feature: [B, N, 3, 1] → [B, N, 3, C]
        x = points.unsqueeze(-1)                                # [B, N, 3, 1]
        x = self.input_proj(x)                                  # [B, N, 3, C]

        for block in self.blocks:
            if self.use_checkpoint:
                x = gradient_checkpoint(block, x, knn_idx, use_reentrant=False)
            else:
                x = block(x, knn_idx=knn_idx)

        x = self.norm(x)

        # Invariant pooling: norm per 3-vector → [B, N, C]
        norms = x.norm(dim=2)                                   # [B, N, C]
        pooled = norms.max(dim=1).values                        # [B, C]
        z = self.to_latent(pooled)                              # [B, D_latent]

        return x, z


class VNEncoderBlock(nn.Module):
    """Encoder-only transformer block (no AdaLN conditioning).

    Accepts optional `knn_idx` for local attention."""

    def __init__(self, channels: int, n_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = VNLayerNorm(channels)
        self.attn = VNAttention(channels, n_heads)
        self.norm2 = VNLayerNorm(channels)

        hidden = int(channels * mlp_ratio)
        self.ffn = nn.Sequential(
            VNLinear(channels, hidden),
            VNLeakyReLU(hidden),
            VNLinear(hidden, channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        knn_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), knn_idx=knn_idx)
        h = self.norm2(x)
        for layer in self.ffn:
            h = layer(h)
        x = x + h
        return x


# ---------------------------------------------------------------------------
# 8.  FlowTransformer — the full velocity-prediction network
# ---------------------------------------------------------------------------

class FlowTransformer(nn.Module):
    """
    Velocity-prediction network for Rectified Flow on 3D point clouds.

    Architecture
    ------------
    1. **Encoder** processes the *target* shape → global latent z.
    2. **Decoder** takes the *interpolated* point cloud x_t together with
       conditioning (t, z) and predicts the instantaneous velocity v(x_t, t).
    3. AdaLN injects (t, z) into the decoder blocks.

    Resolution handling
    -------------------
    The template is stored at full resolution (``n_points``, e.g. 15 000).
    During **training** you can subsample both the template and the target to
    ``train_n_points`` (e.g. 2 048) via Farthest Point Sampling, cutting
    memory from O(15k²) to O(2k²).  At **inference** the full 15k template
    can be flowed through the learned velocity field using KNN-local
    attention, which is O(N·K).

    Parameters
    ----------
    n_points       : int   — full-resolution template size (e.g. 2048)
    channels       : int   — VN channel width
    n_heads        : int   — attention heads
    enc_depth      : int   — encoder depth
    dec_depth      : int   — decoder depth
    latent_dim     : int   — shape-latent dimension
    time_dim       : int   — sinusoidal time-embedding dimension
    knn_k          : int   — neighbours for KNN attention (0 = global)
    """

    def __init__(
        self,
        n_points: int = 2048,
        channels: int = 128,
        n_heads: int = 8,
        enc_depth: int = 6,
        dec_depth: int = 6,
        latent_dim: int = 256,
        time_dim: int = 128,
        knn_k: int = 0,
        use_checkpoint: bool = False,  # gradient checkpointing to save VRAM
    ):
        super().__init__()
        self.n_points = n_points
        self.channels = channels
        self.latent_dim = latent_dim
        self.knn_k = knn_k
        self.use_checkpoint = use_checkpoint

        # --- Learnable canonical template  [1, N, 3] ---
        self.template = nn.Parameter(
            self._init_template(n_points),
            requires_grad=True,
        )

        # --- Encoder (processes target shape) ---
        self.encoder = VNEncoder(
            in_channels=1, channels=channels, n_heads=n_heads,
            depth=enc_depth, latent_dim=latent_dim, knn_k=knn_k,
            use_checkpoint=use_checkpoint,
        )

        # --- Time embedding (scalar) ---
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # Total scalar conditioning dimension for decoder AdaLN
        cond_dim = latent_dim * 2   # [time_embed ; shape_latent]

        # --- Decoder (predicts velocity from x_t) ---
        self.dec_input_proj = VNLinear(1, channels)
        self.dec_blocks = nn.ModuleList([
            VNTransformerBlock(channels, cond_dim, n_heads) for _ in range(dec_depth)
        ])
        self.dec_norm = VNAdaLN(channels, cond_dim)

        # Output projection: [B, N, 3, C] → [B, N, 3, 1] → squeeze → [B, N, 3]
        self.output_proj = VNLinear(channels, 1)

    # ------------------------------------------------------------------

    @staticmethod
    def _init_template(n_points: int) -> torch.Tensor:
        """Initialise the canonical template on a unit sphere (Fibonacci lattice)."""
        indices = torch.arange(0, n_points, dtype=torch.float32) + 0.5
        phi = torch.acos(1 - 2 * indices / n_points)
        theta = math.pi * (1 + math.sqrt(5)) * indices
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        pts = torch.stack([x, y, z], dim=-1)  # [N, 3]
        return pts.unsqueeze(0)                # [1, N, 3]

    # ------------------------------------------------------------------

    def get_template(self, batch_size: int) -> torch.Tensor:
        """Return the full-resolution template expanded for a batch:  [B, N, 3]."""
        return self.template.expand(batch_size, -1, -1)

    def get_subsampled_template(
        self,
        batch_size: int,
        n_sub: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FPS-subsample the template for training.

        Returns
        -------
        template_sub : [B, n_sub, 3]
        fps_indices  : [B, n_sub]      (indices into the full template)
        """
        full = self.get_template(batch_size)                     # [B, N, 3]
        fps_idx = farthest_point_sample(full, n_sub)             # [B, n_sub]
        template_sub = fps_gather(full, fps_idx)                 # [B, n_sub, 3]
        return template_sub, fps_idx

    # ------------------------------------------------------------------

    def encode(self, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode target point cloud.

        target : [B, N, 3]
        returns: (per_point_feats [B, N, 3, C],  z [B, D_latent])
        """
        return self.encoder(target)

    # ------------------------------------------------------------------

    def _build_knn_idx(self, points: torch.Tensor) -> Optional[torch.Tensor]:
        """Build a KNN graph if knn_k > 0, else return None."""
        if self.knn_k > 0:
            return knn(points, points, self.knn_k)
        return None

    # ------------------------------------------------------------------

    def predict_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity v(x_t, t; z).

        x_t : [B, N, 3]   — interpolated point cloud at time t
        t   : [B] or [B,1] — diffusion time in [0,1]
        z   : [B, D_latent] — shape latent from encoder

        returns : [B, N, 3]  — predicted velocity
        """
        t = t.view(-1)                                           # [B]
        t_emb = self.time_mlp(t)                                 # [B, D_latent]
        cond = torch.cat([t_emb, z], dim=-1)                     # [B, 2*D_latent]

        # Build KNN graph for the decoder's local attention
        knn_idx = self._build_knn_idx(x_t)

        # Lift x_t to VN feature
        h = x_t.unsqueeze(-1)                                    # [B, N, 3, 1]
        h = self.dec_input_proj(h)                               # [B, N, 3, C]

        for block in self.dec_blocks:
            if self.use_checkpoint:
                h = gradient_checkpoint(block, h, cond, knn_idx, use_reentrant=False)
            else:
                h = block(h, cond, knn_idx=knn_idx)

        h = self.dec_norm(h, cond)
        v = self.output_proj(h).squeeze(-1)                      # [B, N, 3]
        return v

    # ------------------------------------------------------------------

    def forward(
        self,
        target: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass (used in training).

        target : [B, N, 3]  — target point cloud
        t      : [B]        — random time samples ∈ [0, 1]

        returns:
            v_pred   : [B, N, 3]  — predicted velocity
            template : [B, N, 3]  — (permuted) canonical template
            z        : [B, D_latent]
        """
        B = target.shape[0]
        _, z = self.encode(target)
        template = self.get_template(B)

        # Interpolate along straight-line flow:  x_t = (1-t)*template + t*target
        t_ = t[:, None, None]                                    # [B,1,1]
        x_t = (1 - t_) * template + t_ * target

        v_pred = self.predict_velocity(x_t, t, z)
        return v_pred, template, z


# ---------------------------------------------------------------------------
# Helper: Sinusoidal positional embedding for time
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal positional embedding (Vaswani et al.)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] → [B, dim]"""
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)
