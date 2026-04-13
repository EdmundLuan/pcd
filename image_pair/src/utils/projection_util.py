"""Utility functions for performing projections."""

import os
import time
import torch
import hashlib
import numpy as np
import cvxpy as cp
from typing import List
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, linprog
from concurrent.futures import (
    ProcessPoolExecutor, 
    as_completed
)
try:
    from IPython import get_ipython
    if 'ipykernel' in str(type(get_ipython())):
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
except ImportError:
    from tqdm import tqdm



def parse_projection_timesteps(projection_timesteps: str, T:int) -> List[int]:
    """
    Parses timesteps from `x:y` to a list of [x, ...y].
    
    1. :, all, '' -> [0, ..., T]
    2. x:         -> [x, ..., T]
    3. :x         -> [0, ..., x]
    4. x:y        -> [x, ..., y]
    5. x          -> [x]
    """
    
    def convert_to_int(int_str):
        try:
            int_ = int(int_str)
        except ValueError:
            raise ValueError(f"Invalid timestep specification: '{projection_timesteps}'")
        if not (0 <= int_ <= T):
            raise ValueError(f"Timestep {int_} out of range [0, {T}]")
        return int_
    
    s = projection_timesteps.strip().lower()
    if s in ("", ":", "all"):
        return list(range(0, T + 1))
    
    # if it’s a single integer (no colon), treat as [n]
    if ":" not in s:
        n = convert_to_int(s)
        return [n]
    
    # has a colon: split into l/r
    parts = s.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid timestep range: '{projection_timesteps}'")
    
    # parse start & end
    l,r = parts
    start = 0 if (l == "") else convert_to_int(l)
    end   = T if (r == "") else convert_to_int(r)
    
    if start > end:
        raise ValueError(f"Start timestep ({start}) must be ≤ end timestep ({end})")
    
    return list(range(start, end + 1))






# ------- Start of code for projection using Mirror Descent ---------------------------------------------

def batch_project_onto_convex_hull_mirror_descent(
    ext_points: torch.Tensor,   # [bs, .., n, n]
    hull_points: torch.Tensor,  # [ k, .., n, n]
    learning_rate: float = 0.01,
    num_iter: int = 10000,
    verbose: bool = False,
    return_lambdas: bool = False
) -> torch.Tensor:  # [bs, .., n, n]
    """
    Master function for running projection on GPUs via Mirror Descent.
    
    Args:
        ext_points    : Torch tensor of shape [bs, .., n, n] representing the diffused images at each timestep t.
        hull_points   : Torch tensor of shape [ k, .., n, n] representing the k exemplars used.
        learning_rate : Learning rate for exponentiated gradient updates.
        num_iter      : Number of iterations for optimization.
    
    Returns:
        projected_points : Torch tensor of shape [bs, .., n, n] representing the images that have been projected into the convex hull of exemplars.
    """
    
    original_shape = ext_points.shape
    
    # flatten
    ext_points = ext_points.view(len(ext_points), -1)
    hull_points = hull_points.view(len(hull_points), -1)
    
    # =====================================================
    bs, flat_n = ext_points.shape
    k,  flat_m = hull_points.shape
    
    assert flat_n == flat_m, f"Different flattened shapes to work with! flat_n={flat_n}, flat_m={flat_m}"
    
    # Initial guess for lambdas, uniform distribution
    lambd = torch.full((bs, k), 1.0 / k, device=ext_points.device)
    
    for it in tqdm(range(num_iter), desc="MD Steps"):
        
        # Compute convex combination for each batch: [B, n]
        combo = torch.matmul(lambd, hull_points)  # [B, k] @ [k, n] = [B, n]
        
        # Compute gradient: grad[b] = 2 * points @ (combo[b] - ext_points[b])
        diff = combo - ext_points  # [B, n]
        grad = 2 * torch.matmul(diff, hull_points.T)  # [B, n] @ [n, k] = [B, k]
        
        # log‑space update
        # ------------------------------
        # # Update lambdas by exponentiated gradient
        # lambd = lambd * torch.exp(-learning_rate * grad)  # [B, k]
        log_lambd = torch.log(lambd) - learning_rate * grad  # [B, k]
        
        # Normalize lambdas to ensure they sum to 1
        # lambd = lambd / (lambd.sum(dim=1, keepdim=True) + 1e-8)
        log_lambd = log_lambd - torch.logsumexp(log_lambd, dim=1, keepdim=True)
        
        # back to probability space
        lambd = torch.exp(log_lambd)  # [B, k]
        # ------------------------------
        
        # Report loss
        if verbose and ((it % 100 == 0) or (it == num_iter-1)):
            # loss = (diff ** 2).sum(dim=1).mean()
            loss = torch.norm(diff, dim=1).mean()
            print(f"iter {it}/{num_iter}  loss={loss.item():.6f}")
    
    
    projected_points = torch.matmul(lambd, hull_points)  # [B, k] @ [k, n] = [B, n]
    # =====================================================
    
    # reshape to original shape
    projected_points = projected_points.view(*original_shape)
    
    if return_lambdas:
        return projected_points, lambd
    else:
        return projected_points

# ------- End of code for projection using Mirror Descent ---------------------------------------------

