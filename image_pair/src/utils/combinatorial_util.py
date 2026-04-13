"""Utility for creating combinations of exemplar interpolations."""

import itertools
import torch
from torch import Tensor


# ---------------------------------------------------------------------
# helper: reshape a [n_pts, k] weight matrix onto exemplar latents
# ---------------------------------------------------------------------
def _mix_and_rescale(
    weights: Tensor,
    latents: Tensor,
    rescale_norm: bool = False,
    eps: float = 1e-8
) -> Tensor:
    """
    weights      : [n_pts, k]      (rows sum to 1)
    latents      : [k, *shape]     (exemplar tensors)
    rescale_norm : if True, scale each mixed latent so that
                   ||z_mix||₂ == Σ_i λ_i · ||z_i||₂
    
    returns      : [n_pts, *shape]
    """
    w = weights.to(device=latents.device, dtype=latents.dtype)
    
    # Convex combination (shape‑agnostic einsum)
    mixed = torch.einsum('nk,k...->n...', w, latents)  # [n_pts, *shape]
    
    if not rescale_norm:
        return mixed
    
    # ------------------------------------------------------------
    # L2‑norm matching (per sample)
    # ------------------------------------------------------------
    k = latents.shape[0]
    norms = latents.view(k, -1).norm(dim=1)    # [k]
    target_norm = w @ norms     # [n_pts]
    
    flat_mix = mixed.view(mixed.shape[0], -1)  # [n_pts, D]
    mix_norm  = flat_mix.norm(dim=1) + eps     # avoid div0
    
    scale = (target_norm + eps) / mix_norm     # [n_pts]
    flat_mix = flat_mix * scale.unsqueeze(1)
    
    return flat_mix.view_as(mixed)


# ---------------------------------------------------------------------
# deterministic stars‑and‑bars weight lattice
# ---------------------------------------------------------------------
def simplex_grid(k: int, resolution: int) -> Tensor:
    """
    Return all weight vectors (λ₁ ... λ_k) on a regular grid of step 1/resolution
    such that Σ λ = 1 and λ ≥ 0.
    - k          : number of exemplars / vertices
    - resolution : denominator R  (grid spacing = 1/R)
    Result shape : [C(R+k-1, k-1), k]
    """
    
    R = resolution
    N = R + k - 1 # total discrete slots
    
    grid_int = []
    for comb in itertools.combinations(range(N), k - 1):
        bars = (-1,) + comb + (N,)    # pad with sentinels
        counts = [bars[i+1] - bars[i] - 1 for i in range(k)]
        grid_int.append(counts)
    
    # integer counts to weights in [0,1]
    return torch.tensor(grid_int, dtype=torch.float32) / float(R)


# ---------------------------------------------------------------------
# interpolate actual latent tensors on that grid
# ---------------------------------------------------------------------
def grid_interpolations(
        latents: Tensor,
        resolution: int = 5,
        rescale_norm: bool = False
) -> Tensor:
    """
    latents      : [k, *shape]
    resolution   : denominator R (λ-step = 1/R)
    rescale_norm : flag for L2-norm matching
    returns      : [n_pts, *shape]
    """
    weights = simplex_grid(latents.shape[0], resolution)        # [n_pts, k]
    return _mix_and_rescale(weights, latents, rescale_norm)




if __name__ == "__main__":
    
    # python -m src.utils.combinatorial_util
    
    k, lc, lh, lw = 7, 4, 8, 8
    latents = torch.randn(k, lc, lh, lw)
    
    resolution = 4
    out_raw    = grid_interpolations(latents, resolution, rescale_norm=False)
    out_scaled = grid_interpolations(latents, resolution, rescale_norm=True)
    
    print("Grid points:", out_raw.shape[0])
    # Should equal  C(resolution + k - 1, k - 1)
    # For k=7, R=4  ->  C(10,6) = 210
    
    # quick numeric check for the first point
    sample_idx = 2
    w0   = simplex_grid(k, resolution)[sample_idx]
    raw0 = out_raw[sample_idx].view(-1)
    sc0  = out_scaled[sample_idx].view(-1)
    
    print("Norm raw    :", raw0.norm().item())
    print("Target norm :", (w0 * latents.view(k, -1).norm(dim=1)).sum().item())
    print("Norm scaled :", sc0.norm().item())

