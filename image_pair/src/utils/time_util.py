"""Time benchmarking utility functions."""

import time
import numpy as np
from src.utils.logging_util import LoggingUtils

logger = LoggingUtils.configure_logger(log_name=__name__)



def compute_time_stats(time_stats: dict, percentiles: list = [90, 95, 99]) -> dict:
    """Compute time statistics from a dictionary of time measurements."""
    
    # compute time stats
    for k in time_stats.keys():
        time_list = time_stats[k]["time"]
        time_stats[k]["average"] = np.mean(time_list)
        for p in percentiles:
            time_stats[k][f"percentile{p}"] = np.percentile(time_list, p)
    
    return time_stats
