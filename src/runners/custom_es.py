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


def _global_intermediate(
    base_params: np.ndarray,
    parents: list,
    attr_name: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Global intermediate recombination for a strategy parameter vector.

    For each gene, a random second parent is chosen from the population
    and its value is averaged with the base parent's value.
    """
    n = base_params.shape[0]
    if n == 0:
        return base_params.copy()
    result = base_params.copy()
    mu = len(parents)
    for j in range(n):
        second_idx = rng.integers(0, mu)
        second_val = getattr(parents[second_idx], attr_name)[j]
        result[j] = (result[j] + second_val) / 2.0
    return result


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
        childs = []
        for _ in range(cfg.lam):
            # Select two distinct parents for recombination.
            idx_a, idx_b = rng.choice(cfg.mu, size=2, replace=False)
            parent_a = parents[idx_a]
            parent_b = parents[idx_b]

            child = clone_individual(parent_a)

            # -----------------------------------------------------------------
            # Dual discrete recombination for objective variables:
            # For each gene, randomly pick from parent_a or parent_b (50/50).
            # -----------------------------------------------------------------
            if cfg.n_real > 0:
                mask = rng.random(cfg.n_real) < 0.5
                child.real = np.where(mask, parent_a.real, parent_b.real)
            if cfg.n_count > 0:
                mask = rng.random(cfg.n_count) < 0.5
                child.count = np.where(mask, parent_a.count, parent_b.count)
            if cfg.n_binary > 0:
                mask = rng.random(cfg.n_binary) < 0.5
                child.binary = np.where(mask, parent_a.binary, parent_b.binary)
            if cfg.n_categorical > 0:
                mask = rng.random(cfg.n_categorical) < 0.5
                child.categorical = np.where(mask, parent_a.categorical, parent_b.categorical)
            if cfg.n_fuzzy > 0:
                mask = rng.random(cfg.n_fuzzy * 3) < 0.5
                child.fuzzy = np.where(mask, parent_a.fuzzy, parent_b.fuzzy)
            if cfg.n_circular > 0:
                mask = rng.random(cfg.n_circular) < 0.5
                child.circular = np.where(mask, parent_a.circular, parent_b.circular)

            # -----------------------------------------------------------------
            # Global intermediate recombination for strategy parameters:
            # For each gene, pick a random second parent and average.
            # -----------------------------------------------------------------
            child.sigma_real = _global_intermediate(parent_a.sigma_real, parents, "sigma_real", rng)
            child.lambda_count = _global_intermediate(parent_a.lambda_count, parents, "lambda_count", rng)
            child.sigma_fuzzy = _global_intermediate(parent_a.sigma_fuzzy, parents, "sigma_fuzzy", rng)
            child.logit_binary = _global_intermediate(parent_a.logit_binary, parents, "logit_binary", rng)
            child.logit_cat = _global_intermediate(parent_a.logit_cat, parents, "logit_cat", rng)
            child.logit_circular = _global_intermediate(parent_a.logit_circular, parents, "logit_circular", rng)

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
