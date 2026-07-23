"""
Runner wrapper for sep-CMA-ES (Separable CMA-ES).
Implemented using the pycma library with the flag CMA_diagonal=True.
"""

import numpy as np
import cma

def run(fn, dim, bounds, seed, max_evals, **kwargs) -> np.ndarray:
    rng = np.random.default_rng(seed)
    
    # Initialize guess uniformly within bounds
    x0 = rng.uniform(bounds[0], bounds[1], size=dim)
    
    # Initial step-size sigma0: 0.3 * (bounds[1] - bounds[0])
    sigma0 = 0.3 * (bounds[1] - bounds[0])
    
    opts = {
        'seed': int(seed),
        'bounds': [bounds[0], bounds[1]],
        'verbose': -9,
        'CMA_diagonal': True,
    }
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    trace = []
    best_so_far = float('inf')
    
    while len(trace) < max_evals and not es.stop():
        solutions = es.ask()
        costs = []
        for sol in solutions:
            sol_clipped = np.clip(sol, bounds[0], bounds[1])
            cost = fn(sol_clipped)
            costs.append(cost)
            best_so_far = min(best_so_far, cost)
            trace.append(best_so_far)
            if len(trace) >= max_evals:
                break
        
        if len(trace) >= max_evals:
            break
            
        es.tell(solutions, costs)
        
    # Pad trace if it ended early
    arr = np.array(trace[:max_evals])
    if len(arr) < max_evals:
        arr = np.pad(arr, (0, max_evals - len(arr)), constant_values=arr[-1] if len(arr) > 0 else float('inf'))
    return arr
