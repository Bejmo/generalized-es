"""
generalized_es.py: Generalized Evolution Strategy optimizer and population survivor selection.
"""

from __future__ import annotations

from typing import Callable
import numpy as np
from numpy.typing import NDArray

from .config import DEFAULT_CONFIG, ESConfig
from .individual import Individual


# -----------------------------------------------------------------------------
# 1. Deterministic Rank-Based Survivor Selection
# -----------------------------------------------------------------------------

def select_survivors(
    parents: list[Individual],
    child: list[Individual],
    mode: str = "comma",
    mu: int = 15,
) -> list[Individual]:
    """Deterministic rank-based selection of the top MU individuals."""
    if mode == "comma":
        pool = child
    elif mode == "plus":
        pool = parents + child
    else:
        raise ValueError(f"Unknown selection mode: {mode!r}. Use 'comma' or 'plus'.")

    ranked = sorted(pool, key=lambda ind: ind.fitness)
    return ranked[:mu]


# -----------------------------------------------------------------------------
# 2. GeneralizedES Optimizer Class
# -----------------------------------------------------------------------------

class GeneralizedES:
    """
    Stateful Generalized Evolution Strategy optimizer for mixed-variable optimization.

    Operates with dual discrete recombination on decision variables, global intermediate
    recombination on strategy parameters, and diagonal log-normal / logit-normal self-adaptation.
    """

    def __init__(self, config: ESConfig | None = None, seed: int | None = None) -> None:
        self.config = config if config is not None else DEFAULT_CONFIG
        actual_seed = seed if seed is not None else self.config.seed
        self.rng = np.random.default_rng(actual_seed)
        self.parents: list[Individual] = []
        self.best_so_far: float = float("inf")
        self.best_individual: Individual | None = None
        self.evaluations: int = 0
        self.generation: int = 0

    def _get_eval_input(self, ind: Individual) -> NDArray[np.float64]:
        """Format individual representation for objective evaluation function."""
        if self.config.n_real > 0 and self.config.n_total == self.config.n_real and "real" in ind.variables:
            return ind.variables["real"].values
        return ind.genome

    def init_population(self, fn: Callable[[NDArray[np.float64]], float]) -> None:
        """Initialize population and evaluate fitness of initial parents."""
        self.parents = [
            Individual.random(self.rng, self.config)
            for _ in range(self.config.mu)
        ]
        self.evaluations = 0
        self.generation = 0
        self.best_so_far = float("inf")

        for ind in self.parents:
            eval_input = self._get_eval_input(ind)
            ind.fitness = fn(eval_input)
            self.evaluations += 1
            if ind.fitness < self.best_so_far:
                self.best_so_far = ind.fitness
                self.best_individual = ind.clone()

    def step(self, fn: Callable[[NDArray[np.float64]], float]) -> float:
        """Execute one generational step."""
        if not self.parents:
            self.init_population(fn)

        self.generation += 1
        childs: list[Individual] = []

        for _ in range(self.config.lam):
            parent_idx = int(self.rng.integers(0, self.config.mu))
            parent = self.parents[parent_idx]
            child = parent.recombine(self.parents, self.rng, self.config)
            child.mutate(self.rng, self.config)
            eval_input = self._get_eval_input(child)
            child.fitness = fn(eval_input)
            self.evaluations += 1

            if child.fitness < self.best_so_far:
                self.best_so_far = child.fitness
                self.best_individual = child.clone()

            childs.append(child)

        self.parents = select_survivors(
            self.parents,
            childs,
            mode=self.config.selection_mode,
            mu=self.config.mu,
        )
        return self.best_so_far

    def optimize(
        self,
        fn: Callable[[NDArray[np.float64]], float],
        max_evals: int,
    ) -> NDArray[np.float64]:
        """Full optimization loop returning 1-D trace of best fitness per evaluation."""
        trace: list[float] = []
        self.init_population(fn)

        for _ in self.parents:
            trace.append(self.best_so_far)
            if len(trace) >= max_evals:
                return np.array(trace[:max_evals], dtype=np.float64)

        while len(trace) < max_evals:
            self.generation += 1
            childs: list[Individual] = []

            for _ in range(self.config.lam):
                parent_idx = int(self.rng.integers(0, self.config.mu))
                parent = self.parents[parent_idx]
                child = parent.recombine(self.parents, self.rng, self.config)
                child.mutate(self.rng, self.config)
                eval_input = self._get_eval_input(child)
                child.fitness = fn(eval_input)
                self.evaluations += 1

                if child.fitness < self.best_so_far:
                    self.best_so_far = child.fitness
                    self.best_individual = child.clone()

                trace.append(self.best_so_far)
                childs.append(child)
                if len(trace) >= max_evals:
                    break

            self.parents = select_survivors(
                self.parents,
                childs,
                mode=self.config.selection_mode,
                mu=self.config.mu,
            )

        return np.array(trace[:max_evals], dtype=np.float64)


# Backward-compatibility alias
HeterogeneousES = GeneralizedES
