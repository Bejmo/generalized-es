"""
Runner wrapper for the custom heterogeneous (mu, lambda)-ES.
"""

import numpy as np
from src.es_model.config import ESConfig
from src.es_model.engine import (
    clone_individual,
    create_random_individual,
    mutate_individual,
    select_survivors,
)


def run(fn, dim, bounds, seed, max_evals, *, mu=15, lam=100, config: ESConfig | None = None, **kwargs) -> np.ndarray:
    """
    Benchmark runner interface for the Heterogeneous ES.
    """
    if config is None:
        cfg = ESConfig(
            mu=mu,
            lam=lam,
            n_real=dim,
            n_count=0,
            n_binary=0,
            n_categorical=0,
            n_fuzzy=0,
            lower_bound_real=float(bounds[0]),
            upper_bound_real=float(bounds[1]),
            seed=seed,
        )
    else:
        cfg = config
        cfg.mu = mu
        cfg.lam = lam
        cfg.seed = seed

    rng = np.random.default_rng(seed)
    trace = []
    best_so_far = float("inf")
    parents = [create_random_individual(rng, cfg) for _ in range(cfg.mu)]
    for ind in parents:
        eval_input = ind.real if (cfg.n_real > 0 and cfg.n_total == cfg.n_real) else ind.genome
        ind.fitness = fn(eval_input)
        best_so_far = min(best_so_far, ind.fitness)
        trace.append(best_so_far)
        if len(trace) >= max_evals:
            break

    while len(trace) < max_evals:
        if cfg.n_real > 0:
            all_sigmas = np.array([p.sigma_real for p in parents])
            global_geom_sigma = np.exp(np.mean(np.log(all_sigmas), axis=0))
        else:
            global_geom_sigma = None

        childs = []
        for _ in range(cfg.lam):
            idx_a, idx_b = rng.choice(cfg.mu, size=2, replace=False)
            parent_a = parents[idx_a]
            parent_b = parents[idx_b]

            child = clone_individual(parent_a)
            if global_geom_sigma is not None:
                child.sigma_real = global_geom_sigma.copy()

            use_discrete_recombination = False

            if use_discrete_recombination:
                if cfg.n_real > 0:
                    mask_a = rng.random(cfg.n_real) < 0.5
                    child.real = np.where(mask_a, parent_a.real, parent_b.real)
            else:
                if cfg.n_real > 0:
                    child.real = (parent_a.real + parent_b.real) / 2.0

            mutate_individual(child, rng, cfg)
            eval_input = child.real if (cfg.n_real > 0 and cfg.n_total == cfg.n_real) else child.genome
            child.fitness = fn(eval_input)
            best_so_far = min(best_so_far, child.fitness)
            trace.append(best_so_far)
            childs.append(child)
            if len(trace) >= max_evals:
                break

        if len(childs) > 0:
            parents = select_survivors(parents, childs, mode=cfg.selection_mode, mu=cfg.mu)

    return np.array(trace[:max_evals])
