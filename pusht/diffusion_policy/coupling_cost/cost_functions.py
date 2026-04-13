import torch
import numpy as np
import math


def sum_log_l2_cost(
    x: torch.Tensor,
    y: torch.Tensor, 
    alpha: float,  # A small constant to avoid singularity in log
    **kwargs,
) -> torch.Tensor:
    """
    Computes a summed, negative-logarithmic penalty on squared L2 distances
    for batches of trajectories. At each time-step, the cost is
        // -log( sum_d (x_hd - y_hd) + epsilon )
        -log( sum_d (x_hd - y_hd) + 1 )
    which strongly penalizes small distances and grows more gently as
    distance increases.

    Args:
        x (torch.Tensor): Tensor of shape (B, H, d).
        y (torch.Tensor): Tensor of shape (B, H, d). Must match x's shape.
        epsilon (float): Small constant added inside the log to avoid
                         singularity when distance is zero.

    Returns:
        torch.Tensor: A tensor of shape (B,), where each element is
                      the total log-penalty cost for one trajectory pair.
    """
    if x.shape != y.shape or x.dim() < 3:
        raise ValueError("Expected x,y of shape (..., H, d).")

    # L2 norm per time-step -> (B, H)
    # l2_sqr = ((x - y)**2).sum(dim=-1)
    l2 = torch.linalg.norm(x - y, dim=-1)  # L2 norm

    # -2 * log of clamped norm
    # per_step = -0.5 * torch.log(torch.clamp(l2_sqr, epsilon))
    per_step = -torch.log(alpha + l2)

    # sum over time -> (B,)
    return per_step.sum(dim=-1)


def dpp_cost(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-7, **kwargs) -> torch.Tensor:
    """
    Computes a diversity score based on the Determinantal Point Process (DPP)
    for pairs of trajectories (x_i, y_i). The score is the log-determinant
    of the kernel matrix formed by each pair. A higher score corresponds to
    more diverse pairs of trajectories.

    This function handles tensors with 3 or more dimensions, applying the DPP
    calculation to the last two dimensions while preserving all leading dimensions.

    The group size is fixed to 2, with the groups being (x_i, y_i) for each
    item i in the batch dimensions.

    Args:
        x (torch.Tensor): The first batch of trajectories, shape (..., H, d).
        y (torch.Tensor): The second batch of trajectories, shape (..., H, d).
        eps (float): A small epsilon for numerical stability, used to clamp
                    the determinant before taking the logarithm.

    Returns:
        torch.Tensor: A tensor of costs with shape (...), where each element is the
                    log-determinant for the corresponding pair in the batch.
                    This cost should be minimized for diversity.
    """
    if x.shape != y.shape:
        raise ValueError("Input tensors x and y must have the same shape.")
    if x.dim() < 3:
        raise ValueError(f"Input tensors must be at least 3D (..., H, d). Got {x.dim()}-D.")

    # Flatten the last two dimensions (H, d) into a single feature vector.
    # For an input of shape (..., H, d), this results in shape (..., H*d).
    phi_x = torch.flatten(x, start_dim=-2)
    phi_y = torch.flatten(y, start_dim=-2)

    # Stack to form groups of 2 for each item in the batch dimensions.
    # phi will have a shape of (..., 2, H*d).
    phi = torch.stack([phi_x, phi_y], dim=-2)

    # Normalize feature vectors along the feature dimension (the last dimension).
    phi = torch.nn.functional.normalize(phi, p=2, dim=-1)

    # Compute the kernel matrix for each item. `torch.matmul` (@) handles
    # batching over all leading dimensions.
    # For phi of shape (..., 2, H*d), the resulting kernel matrix S
    # will have a shape of (..., 2, 2).
    S = phi @ phi.transpose(-1, -2)
    assert S.shape[-1] == S.shape[-2] == 2, "Kernel matrix S should be of shape (..., 2, 2)."

    # Compute the determinant for each 2x2 kernel matrix.
    # The result will have a shape of (...).
    det = torch.linalg.det(S)

    # Clamp the determinant to avoid values.
    det = torch.clamp(torch.abs(det), min=eps)

    # The cost is the log determinant. Minimizing this cost encourages
    # diversity between the x and y trajectories.
    cost = -torch.log(det)

    return cost


# Registry for cost functions
cost_registry = {
    "sum_log_l2": sum_log_l2_cost,
    "dpp": dpp_cost,
}
