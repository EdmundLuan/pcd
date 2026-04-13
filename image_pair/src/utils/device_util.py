"""Utility functions for torch devices."""

import torch
from src.utils.logging_util import LoggingUtils

logger = LoggingUtils.configure_logger(log_name=__name__)



DTYPE_MAP = {
    "float"   : torch.float,
    "float32" : torch.float32,
    "float16" : torch.float16,
    "bfloat16": torch.bfloat16,
}


def get_device(device_str: str):
    """Set device for pytorch operations."""
    
    logger.info(f"PyTorch version: {torch.__version__}")

    if "cuda" in device_str and torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        device= torch.device(device_str)
    else:
        logger.warning("CUDA not available, resorting to CPU.")
        device= torch.device("cpu")

    logger.info(f"Device: {device}")
    return device



def set_all_seed(seed: int = 42, deterministic: bool = True):
    """Set random seed for reproducibility."""
    
    import os
    import random
    import numpy as np
    import torch
    
    if seed == -1:
        logger.warning("Set seed disabled.")
    else:
        # Python built‑ins
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch (CPU & CUDA)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # multiple GPUs
    
    # CuDNN backends
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



def recursive_to(obj, to_device="cpu", to_dtype=torch.float):
    """
    Recursively traverse any data structure and convert all torch tensors
    to specified dtype and move to given device.
    
    Args:
        obj: nested structure (tensor, dict, list, etc.)
        to_device (str or torch.device): target device
        to_dtype (torch.dtype): target tensor dtype
    """
    if isinstance(to_device, str):
        device = torch.device(to_device)
    elif isinstance(to_device, torch.device):
        device = to_device
    else:
        raise ValueError(f"Invalid device: {to_device}")

    if isinstance(obj, torch.Tensor):
        return obj.to(device=device, dtype=to_dtype)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, to_device, to_dtype) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_to(v, to_device, to_dtype) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(v, to_device, to_dtype) for v in obj)
    elif isinstance(obj, set):
        return {recursive_to(v, to_device, to_dtype) for v in obj}
    else:
        return obj