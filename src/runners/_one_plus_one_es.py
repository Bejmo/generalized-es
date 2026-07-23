"""
Runner wrapper for (1+1)-ES with 1/5-rule step-size adaptation.
"""

import numpy as np


def run(fn, dim, bounds, seed, max_evals, **kwargs) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = 0.5 * (bounds[1] - bounds[0])
    x = rng.uniform(bounds[0], bounds[1], size=dim)
    fx = fn(x)
    trace = [fx]
    successes = 0
    window = 10 * dim

    for i in range(1, max_evals):
        x_new = x + rng.standard_normal(dim) * sigma
        x_new = np.clip(x_new, bounds[0], bounds[1])
        fx_new = fn(x_new)
        if fx_new <= fx:
            x, fx = x_new, fx_new
            successes += 1
        trace.append(fx)
        if (i % window == 0) and i > 0:
            success_rate = successes / window
            if success_rate > 0.2:
                sigma *= 1.2
            elif success_rate < 0.2:
                sigma *= 0.82
            successes = 0

    return np.array(trace[:max_evals])
