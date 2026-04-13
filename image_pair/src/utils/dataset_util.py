"""Dataset utility functions."""

from typing import Tuple

import torch
import torch.nn.functional as F
from src.utils.logging_util import LoggingUtils

logger = LoggingUtils.configure_logger(log_name=__name__)



DEFAULT_DATA_PATH = './data/'


def map_tensor_range(
    tensor: torch.Tensor,
    in_range: Tuple[float, float],
    out_range: Tuple[float, float]
) -> torch.Tensor:
    """
    Linearly map a tensor from in_range [a, b] to out_range [c, d].
    """
    a, b = in_range
    c, d = out_range
    a, b, c, d = float(a), float(b), float(c), float(d)
    return ((tensor - a) / (b - a)) * (d - c) + c



def resize_imgs(x, m, mode='bicubic'):
    """
    x:  [B, 3, H, W]
    m:  target size (int)
    mode: 'bicubic' (good general choice), 'area' (best for big downscales)
    """
    if x.shape[-1] == m:
        return x
    # antialias only matters when downscaling; safe to always set True on >= PyTorch 2.0
    return F.interpolate(
        x, size=(m, m), mode=mode,
        align_corners=False if mode in ('bilinear', 'bicubic', 'trilinear') else None,
        antialias=True
    )
