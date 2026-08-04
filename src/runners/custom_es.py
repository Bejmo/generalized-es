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
            cfg.REAL_BOUNDS, cfg.COUNT_BOUNDS, cfg.N_TOTAL, cfg.MU, cfg.LAMBDA, cfg.CATEGORICAL_K, cfg.INIT_SIGMA_REAL)
    # Only real variables (first prototype)
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
    cfg.COUNT_BOUNDS = np.empty((0, 2), dtype=np.int64)
    cfg.INIT_SIGMA_REAL = (bounds[1] - bounds[0]) / 3.0
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
        # Precompute the global geometric mean of all parents' sigmas
        # to implement global geometric intermediate recombination.
        all_sigmas = np.array([p.sigma_real for p in parents])
        global_geom_sigma = np.exp(np.mean(np.log(all_sigmas), axis=0))

        childs = []
        for _ in range(lam):
            idx_a, idx_b = rng.choice(mu, size=2, replace=False)
            parent_a = parents[idx_a]
            parent_b = parents[idx_b]

            child = clone_individual(parent_a)
            child.sigma_real = global_geom_sigma.copy()

            # @TODO Test different recombinations (discrete/intermediate)
            use_discrete_recombination = False

            if use_discrete_recombination:
                # Dual discrete recombination of object variables
                mask_a = rng.random(dim) < 0.5
                child.real = np.where(mask_a, parent_a.real, parent_b.real)
            else:
                # Dual intermediate recombination of object variables
                child.real = (parent_a.real + parent_b.real) / 2.0

            mutate_individual(child, rng)
            child.fitness = fn(child.real)
            best_so_far = min(best_so_far, child.fitness)
            trace.append(best_so_far)
            childs.append(child)
            if len(trace) >= max_evals:
                break

        if len(childs) > 0:
            parents = select_survivors(parents, childs)

    (cfg.N_REAL, cfg.N_COUNT, cfg.N_BINARY, cfg.N_CATEGORICAL, cfg.N_FUZZY,
     cfg.REAL_BOUNDS, cfg.COUNT_BOUNDS, cfg.N_TOTAL, cfg.MU, cfg.LAMBDA, cfg.CATEGORICAL_K, cfg.INIT_SIGMA_REAL) = orig
    kernels.TAU_PRIME, kernels.TAU = kernels.compute_learning_rates(cfg.N_TOTAL)
    return np.array(trace[:max_evals])
