"""
engine.py — Evolutionary algorithm engine, Individual representation, and selection.

Defines the Individual class (encapsulating VariableBlock instances), child cloning,
recombination, mutation pipelines, deterministic rank-based survivor selection,
and the HeterogeneousES optimizer class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable
import numpy as np
from numpy.typing import NDArray

from .config import DEFAULT_CONFIG, ESConfig
from .blocks import (
    VariableBlock,
    RealBlock,
    CountBlock,
    BinaryBlock,
    CategoricalBlock,
    FuzzyBlock,
    CircularBlock,
)


# -----------------------------------------------------------------------------
# 1. Individual Data Structure
# -----------------------------------------------------------------------------

@dataclass
class Individual:
    """
    A single candidate solution in the heterogeneous ES.

    Decision variables and co-evolving strategy parameters are encapsulated
    in modular, vectorized VariableBlock instances inside the `blocks` dictionary.
    """

    blocks: dict[str, VariableBlock] = field(default_factory=dict)
    fitness: float = float("inf")

    @property
    def genome(self) -> NDArray[np.float64]:
        """
        Return a single flattened scalar array of all decision variable blocks.

        Canonical Order: real -> count -> binary -> categorical -> fuzzy -> circular.
        Excludes all strategy parameters.
        Used for logging, plotting, and scalar evaluation functions.
        """
        canonical_order = ["real", "count", "binary", "categorical", "fuzzy", "circular"]
        parts = [self.blocks[k].flatten() for k in canonical_order if k in self.blocks]
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(parts)

    # Convenience properties for backward compatibility
    @property
    def real(self) -> NDArray[np.float64]:
        return self.blocks["real"].values.astype(np.float64) if "real" in self.blocks else np.zeros(0, dtype=np.float64)

    @property
    def count(self) -> NDArray[np.int64]:
        return self.blocks["count"].values.astype(np.int64) if "count" in self.blocks else np.zeros(0, dtype=np.int64)

    @property
    def binary(self) -> NDArray[np.uint8]:
        return self.blocks["binary"].values.astype(np.uint8) if "binary" in self.blocks else np.zeros(0, dtype=np.uint8)

    @property
    def categorical(self) -> NDArray[np.int64]:
        return self.blocks["categorical"].values.astype(np.int64) if "categorical" in self.blocks else np.zeros(0, dtype=np.int64)

    @property
    def fuzzy(self) -> NDArray[np.float64]:
        return self.blocks["fuzzy"].values.astype(np.float64) if "fuzzy" in self.blocks else np.zeros(0, dtype=np.float64)

    @property
    def circular(self) -> NDArray[np.int64]:
        return self.blocks["circular"].values.astype(np.int64) if "circular" in self.blocks else np.zeros(0, dtype=np.int64)

    @property
    def sigma_real(self) -> NDArray[np.float64]:
        return self.blocks["real"].strategy if "real" in self.blocks else np.zeros(0, dtype=np.float64)

    @property
    def lambda_count(self) -> NDArray[np.float64]:
        return self.blocks["count"].strategy if "count" in self.blocks else np.zeros(0, dtype=np.float64)

    @property
    def logit_binary(self) -> NDArray[np.float64]:
        return self.blocks["binary"].strategy if "binary" in self.blocks else np.zeros(0, dtype=np.float64)

    @property
    def logit_cat(self) -> NDArray[np.float64]:
        return self.blocks["categorical"].strategy if "categorical" in self.blocks else np.zeros(0, dtype=np.float64)

    @property
    def sigma_fuzzy(self) -> NDArray[np.float64]:
        return self.blocks["fuzzy"].strategy if "fuzzy" in self.blocks else np.zeros(0, dtype=np.float64)

    @property
    def logit_circular(self) -> NDArray[np.float64]:
        return self.blocks["circular"].strategy if "circular" in self.blocks else np.zeros(0, dtype=np.float64)


@dataclass
class UnpackedGenome:
    """Convenience container for un-concatenated decision variable blocks."""

    real: NDArray[np.float64]
    count: NDArray[np.int64]
    binary: NDArray[np.uint8]
    categorical: NDArray[np.int64]
    fuzzy: NDArray[np.float64]
    circular: NDArray[np.int64]


def unpack_genome(
    x_or_ind: Individual | NDArray[np.float64],
    config: ESConfig | None = None,
) -> UnpackedGenome:
    """
    Safely extract all 6 decision variable blocks from either an Individual object
    or a 1-D flattened genome NumPy array.
    """
    cfg = config if config is not None else DEFAULT_CONFIG

    if isinstance(x_or_ind, Individual):
        return UnpackedGenome(
            real=(
                x_or_ind.blocks["real"].values.astype(np.float64)
                if "real" in x_or_ind.blocks
                else np.zeros(0, dtype=np.float64)
            ),
            count=(
                x_or_ind.blocks["count"].values.astype(np.int64)
                if "count" in x_or_ind.blocks
                else np.zeros(0, dtype=np.int64)
            ),
            binary=(
                x_or_ind.blocks["binary"].values.astype(np.uint8)
                if "binary" in x_or_ind.blocks
                else np.zeros(0, dtype=np.uint8)
            ),
            categorical=(
                x_or_ind.blocks["categorical"].values.astype(np.int64)
                if "categorical" in x_or_ind.blocks
                else np.zeros(0, dtype=np.int64)
            ),
            fuzzy=(
                x_or_ind.blocks["fuzzy"].values.astype(np.float64)
                if "fuzzy" in x_or_ind.blocks
                else np.zeros(0, dtype=np.float64)
            ),
            circular=(
                x_or_ind.blocks["circular"].values.astype(np.int64)
                if "circular" in x_or_ind.blocks
                else np.zeros(0, dtype=np.int64)
            ),
        )

    vec = np.asarray(x_or_ind, dtype=np.float64)
    offset = 0

    # 1. Real
    real = vec[offset : offset + cfg.n_real]
    offset += cfg.n_real

    # 2. Count
    count = np.round(vec[offset : offset + cfg.n_count]).astype(np.int64)
    offset += cfg.n_count

    # 3. Binary
    binary = (vec[offset : offset + cfg.n_binary] >= 0.5).astype(np.uint8)
    offset += cfg.n_binary

    # 4. Categorical
    cat = np.round(vec[offset : offset + cfg.n_categorical]).astype(np.int64)
    offset += cfg.n_categorical

    # 5. Fuzzy (triplets)
    fuzzy = vec[offset : offset + cfg.n_fuzzy * 3]
    offset += cfg.n_fuzzy * 3

    # 6. Circular
    circ = np.round(vec[offset : offset + cfg.n_circular]).astype(np.int64)

    return UnpackedGenome(
        real=real,
        count=count,
        binary=binary,
        categorical=cat,
        fuzzy=fuzzy,
        circular=circ,
    )


# -----------------------------------------------------------------------------
# 2. Random Initialization
# -----------------------------------------------------------------------------

def create_random_individual(
    rng: np.random.Generator,
    config: ESConfig | None = None,
) -> Individual:
    """Create a randomly initialized individual within configured bounds."""
    cfg = config if config is not None else DEFAULT_CONFIG
    blocks: dict[str, VariableBlock] = {}

    if cfg.n_real > 0:
        blocks["real"] = RealBlock.random(cfg, rng)
    if cfg.n_count > 0:
        blocks["count"] = CountBlock.random(cfg, rng)
    if cfg.n_binary > 0:
        blocks["binary"] = BinaryBlock.random(cfg, rng)
    if cfg.n_categorical > 0:
        blocks["categorical"] = CategoricalBlock.random(cfg, rng)
    if cfg.n_fuzzy > 0:
        blocks["fuzzy"] = FuzzyBlock.random(cfg, rng)
    if cfg.n_circular > 0:
        blocks["circular"] = CircularBlock.random(cfg, rng)

    return Individual(blocks=blocks)


# -----------------------------------------------------------------------------
# 3. Deep-Copy-Safe Child Cloning
# -----------------------------------------------------------------------------

def clone_individual(parent: Individual) -> Individual:
    """
    Create a fully independent deep copy of a parent individual.

    All NumPy arrays are copied inside each block so that mutations on the child
    do not affect the parent's state.
    """
    return Individual(
        blocks={name: block.copy() for name, block in parent.blocks.items()},
        fitness=parent.fitness,
    )


# -----------------------------------------------------------------------------
# 4. Child Recombination & Mutation Pipeline
# -----------------------------------------------------------------------------

def recombine_individual(
    parent_a: Individual,
    parents: list[Individual],
    rng: np.random.Generator,
    config: ESConfig | None = None,
) -> Individual:
    """
    Create a child individual via recombination.

    - Objective variables: Dual discrete recombination (two distinct parents
      chosen at random for each offspring, 50% probability per gene).
    - Strategy parameters: Global intermediate recombination (for each gene,
      averages base parent value with a random second parent's value).
    """
    mu = len(parents)
    if mu <= 1:
        return clone_individual(parent_a)

    # Select a second distinct parent for discrete recombination
    idx_b = int(rng.integers(0, mu))
    while parents[idx_b] is parent_a and mu > 1:
        idx_b = int(rng.integers(0, mu))
    parent_b = parents[idx_b]

    child_blocks: dict[str, VariableBlock] = {
        name: block.copy() for name, block in parent_a.blocks.items()
    }

    # 1. Dual discrete recombination for decision variables across all blocks
    for name, block_a in parent_a.blocks.items():
        if name in parent_b.blocks:
            child_blocks[name].values = block_a.recombine_values(parent_b.blocks[name], rng)

    # 2. Global intermediate recombination for strategy parameters across all blocks
    for name, block_a in parent_a.blocks.items():
        donor_blocks = [p.blocks[name] for p in parents if name in p.blocks]
        child_blocks[name].strategy = block_a.recombine_strategy(donor_blocks, rng)

    return Individual(blocks=child_blocks)


def mutate_individual(
    ind: Individual,
    rng: np.random.Generator,
    config: ESConfig | None = None,
) -> None:
    """
    Apply self-adaptive mutation.

    Steps:
    1. Adapt strategy parameters across all blocks.
    2. Mutate decision variables using the adapted strategies.
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    tau_prime, tau = cfg.tau_prime_and_tau

    # Adapt strategy parameters first
    for block in ind.blocks.values():
        block.adapt_strategy(rng, tau_prime, tau)

    # Mutate decision variables using updated strategies
    for block in ind.blocks.values():
        block.mutate(rng)


# -----------------------------------------------------------------------------
# 5. Deterministic Rank-Based Survivor Selection
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
# 6. HeterogeneousES Optimizer Class
# -----------------------------------------------------------------------------

class HeterogeneousES:
    """
    Stateful Evolution Strategy optimizer for heterogeneous mixed-variable optimization.
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
        if self.config.n_real > 0 and self.config.n_total == self.config.n_real and "real" in ind.blocks:
            return ind.blocks["real"].values
        return ind.genome

    def init_population(self, fn: Callable[[NDArray[np.float64]], float]) -> None:
        """Initialize population and evaluate fitness of initial parents."""
        self.parents = [
            create_random_individual(self.rng, self.config)
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
                self.best_individual = clone_individual(ind)

    def step(self, fn: Callable[[NDArray[np.float64]], float]) -> float:
        """Execute one generational step."""
        if not self.parents:
            self.init_population(fn)

        self.generation += 1
        childs: list[Individual] = []

        for _ in range(self.config.lam):
            parent_idx = int(self.rng.integers(0, self.config.mu))
            child = recombine_individual(
                self.parents[parent_idx],
                self.parents,
                self.rng,
                self.config,
            )
            mutate_individual(child, self.rng, self.config)
            eval_input = self._get_eval_input(child)
            child.fitness = fn(eval_input)
            self.evaluations += 1

            if child.fitness < self.best_so_far:
                self.best_so_far = child.fitness
                self.best_individual = clone_individual(child)

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
                child = recombine_individual(
                    self.parents[parent_idx],
                    self.parents,
                    self.rng,
                    self.config,
                )
                mutate_individual(child, self.rng, self.config)
                eval_input = self._get_eval_input(child)
                child.fitness = fn(eval_input)
                self.evaluations += 1

                if child.fitness < self.best_so_far:
                    self.best_so_far = child.fitness
                    self.best_individual = clone_individual(child)

                trace.append(self.best_so_far)
                childs.append(child)
                if len(trace) >= max_evals:
                    break

            if len(childs) > 0:
                self.parents = select_survivors(
                    self.parents,
                    childs,
                    mode=self.config.selection_mode,
                    mu=self.config.mu,
                )

        return np.array(trace[:max_evals], dtype=np.float64)
