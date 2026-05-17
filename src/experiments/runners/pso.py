"""
Runner wrapper for Particle Swarm Optimization (pyswarms).
"""

import numpy as np


def run(fn, dim, bounds, seed, max_evals, *, n_particles=50, **kwargs) -> np.ndarray:
    np.random.seed(seed)
    import pyswarms as ps

    max_iters = max_evals // n_particles
    bounds_low = np.full(dim, bounds[0])
    bounds_high = np.full(dim, bounds[1])
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    optimizer = ps.single.GlobalBestPSO(
        n_particles=n_particles, dimensions=dim,
        options=options, bounds=(bounds_low, bounds_high)
    )

    trace = []
    best_so_far = float('inf')
    for _ in range(max_iters):
        optimizer.optimize(lambda x: np.array([fn(xi) for xi in x]), iters=1, verbose=False)
        for cost in optimizer.cost_history[-1:]:
            best_so_far = min(best_so_far, cost)
        for _ in range(n_particles):
            trace.append(best_so_far)
            if len(trace) >= max_evals:
                break
        if len(trace) >= max_evals:
            break

    arr = np.array(trace[:max_evals])
    if len(arr) < max_evals:
        arr = np.pad(arr, (0, max_evals - len(arr)), constant_values=arr[-1])
    return arr
