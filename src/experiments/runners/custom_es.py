"""
Runner wrapper for the custom heterogeneous (mu, lambda)-ES.
"""

import numpy as np
from src.es_model import config as cfg
from src.es_model import kernels
from src.es_model.engine import (
    create_random_individual,
    clone_individual,
    mutate_individual,
    select_survivors,
)


def run(fn, dim, bounds, seed, max_evals, *, mu=15, lam=100, **kwargs) -> np.ndarray:
    orig = (cfg.N_REAL, cfg.N_COUNT, cfg.N_BINARY, cfg.N_CATEGORICAL, cfg.N_FUZZY,
            cfg.REAL_BOUNDS, cfg.N_TOTAL, cfg.MU, cfg.LAMBDA, cfg.CATEGORICAL_K)
    cfg.N_REAL = dim
    cfg.N_COUNT = 0
    cfg.N_BINARY = 0
    cfg.N_CATEGORICAL = 0
    cfg.N_FUZZY = 0
    cfg.CATEGORICAL_K = np.array([], dtype=np.int64)
    cfg.REAL_BOUNDS = np.column_stack([
        np.full(dim, bounds[0], dtype=np.float64),
        np.full(dim, bounds[1], dtype=np.float64),
    ])
    cfg.N_TOTAL = dim
    cfg.MU = mu
    cfg.LAMBDA = lam
    kernels.TAU_PRIME, kernels.TAU = kernels.compute_learning_rates(dim)

    rng = np.random.default_rng(seed)
    trace = []
    best_so_far = float('inf')
    parents = [create_random_individual(rng) for _ in range(mu)]
    for ind in parents:
        ind.fitness = fn(ind.real)
        best_so_far = min(best_so_far, ind.fitness)
        trace.append(best_so_far)
        if len(trace) >= max_evals:
            break
    while len(trace) < max_evals:
        offspring = []
        for _ in range(lam):
            parent_idx = rng.integers(0, mu)
            child = clone_individual(parents[parent_idx])
            mutate_individual(child, rng)
            child.fitness = fn(child.real)
            best_so_far = min(best_so_far, child.fitness)
            trace.append(best_so_far)
            offspring.append(child)
            if len(trace) >= max_evals:
                break
        parents = select_survivors(parents, offspring)

    (cfg.N_REAL, cfg.N_COUNT, cfg.N_BINARY, cfg.N_CATEGORICAL, cfg.N_FUZZY,
     cfg.REAL_BOUNDS, cfg.N_TOTAL, cfg.MU, cfg.LAMBDA, cfg.CATEGORICAL_K) = orig
    kernels.TAU_PRIME, kernels.TAU = kernels.compute_learning_rates(cfg.N_TOTAL)
    return np.array(trace[:max_evals])
