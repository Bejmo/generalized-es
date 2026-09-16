"""
Runner wrapper for the custom Generalized (mu, lambda)-ES.
"""

from copy import copy

import numpy as np
from src.es_model.config import ESConfig
from src.es_model.engine import HeterogeneousES


def run(fn, dim, bounds, seed, max_evals, *, mu=15, lam=100, config: ESConfig | None = None, **kwargs) -> np.ndarray:
    """
    Benchmark runner interface for the Generalized ES.
    Delegates optimization directly to the unified HeterogeneousES engine.
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
        cfg = copy(config)
        cfg.seed = seed

    es = HeterogeneousES(config=cfg, seed=seed)
    return es.optimize(fn, max_evals)

