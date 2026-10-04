from __future__ import annotations
import time
from typing import Callable, Any, Dict, Tuple


def profile_function(fn: Callable, *args, num_runs: int = 5, **kwargs) -> Tuple[Any, float]:
    """Profiles a function's execution time over multiple runs.
    
    Returns:
        (result, avg_time_sec)
    """
    times = []
    result = None
    for _ in range(num_runs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        end = time.perf_counter()
        times.append(end - start)
    avg_time = sum(times) / len(times)
    return result, avg_time
