"""
Runner wrapper for SciPy Differential Evolution.
"""

import numpy as np
from scipy.optimize import differential_evolution as _de


def run(fn, dim, bounds, seed, max_evals, **kwargs) -> np.ndarray:
    trace = []
    best_so_far = float('inf')

    def wrapped_fn(x):
        nonlocal best_so_far
        val = fn(x)
        best_so_far = min(best_so_far, val)
        trace.append(best_so_far)
        if len(trace) >= max_evals:
            raise StopIteration()
        return val

    bound_list = [(bounds[0], bounds[1])] * dim
    try:
        _de(wrapped_fn, bound_list, seed=seed, maxiter=10000,
            tol=0, atol=0, init='latinhypercube', polish=False)
    except StopIteration:
        pass

    arr = np.array(trace[:max_evals])
    if len(arr) < max_evals:
        arr = np.pad(arr, (0, max_evals - len(arr)), constant_values=arr[-1])
    return arr
