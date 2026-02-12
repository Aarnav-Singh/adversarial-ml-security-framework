import numpy as np
import torch
import random
import logging
import time
from functools import wraps

def set_seed(seed=42):
    """Ensures deterministic results across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure CUDNN is deterministic (slightly slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(level=logging.INFO):
    """Configures professional logging."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )

def benchmark_latency(func):
    """Decorator to measure execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        ms = (end - start) * 1000
        # Store last latency in a global or pass it back if needed
        # For now, we just log it
        logging.getLogger(func.__name__).debug(f"Execution time: {ms:.2f}ms")
        return result, ms
    return wrapper
