"""
Defines the Individual data class, child cloning, deterministic rank-based
survivor selection for both (mu, lambda) and (mu + lambda) modes, and the
HeterogeneousES optimizer class.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .config import DEFAULT_CONFIG, ESConfig
from . import kernels


# -----------------------------------------------------------------------------
# 1. Individual Data Structure
# -----------------------------------------------------------------------------

@dataclass
class Individual:
    """
    A single candidate solution in the heterogeneous ES.
    Strategy parameters co-evolve with the genome via self-adaptation.
    """

    # Decision Variable Blocks
    real: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    count: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    binary: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))
    categorical: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    fuzzy: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    circular: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    # Strategy Parameters
    sigma_real: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    lambda_count: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    sigma_fuzzy: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    logit_binary: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    logit_cat: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    logit_circular: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))

    # Fitness
    fitness: float = float("inf")

    @property
    def genome(self) -> NDArray[np.float64]:
        """
        Return a single flattened scalar array of all decision variable blocks.

        Order: real → count → binary → categorical → fuzzy control points → circular.
        Excludes all strategy parameters (sigmas/logits).
        Used exclusively for logging and plotting.
        """
        return np.concatenate([
            self.real.astype(np.float64),
            self.count.astype(np.float64),
            self.binary.astype(np.float64),
            self.categorical.astype(np.float64),
            self.fuzzy.astype(np.float64),
            self.circular.astype(np.float64),
        ])


@dataclass
class UnpackedGenome:
    """Convenience container for un-concatenated decision variable blocks."""
    real: NDArray[np.float64]
    count: NDArray[np.int64]
    binary: NDArray[np.uint8]
    categorical: NDArray[np.int64]
    fuzzy: NDArray[np.float64]
    circular: NDArray[np.int64]


def unpack_genome(x_or_ind: Individual | NDArray[np.float64], config: ESConfig | None = None) -> UnpackedGenome:
    """
    Safely extract all 6 decision variable blocks from either an Individual object
    or a 1-D flattened genome NumPy array.
    """
    cfg = config if config is not None else DEFAULT_CONFIG

    if isinstance(x_or_ind, Individual):
        return UnpackedGenome(
            real=x_or_ind.real,
            count=x_or_ind.count,
            binary=x_or_ind.binary,
            categorical=x_or_ind.categorical,
            fuzzy=x_or_ind.fuzzy,
            circular=x_or_ind.circular,
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

    real_b = cfg.real_bounds
    count_b = cfg.count_bounds

    real_vars = (
        rng.uniform(real_b[:, 0], real_b[:, 1]).astype(np.float64)
        if cfg.n_real > 0
        else np.zeros(0, dtype=np.float64)
    )

    count_vars = (
        rng.integers(count_b[:, 0], count_b[:, 1] + 1, dtype=np.int64)
        if cfg.n_count > 0
        else np.zeros(0, dtype=np.int64)
    )

    binary_vars = (
        rng.integers(0, 2, size=cfg.n_binary, dtype=np.uint8)
        if cfg.n_binary > 0
        else np.zeros(0, dtype=np.uint8)
    )

    cat_vars = (
        np.array([rng.integers(0, int(k)) for k in cfg.categorical_k], dtype=np.int64)
        if cfg.n_categorical > 0
        else np.zeros(0, dtype=np.int64)
    )

    fuzzy_vars = _random_fuzzy_points(rng, cfg.n_fuzzy) if cfg.n_fuzzy > 0 else np.zeros(0, dtype=np.float64)

    circular_vars = (
        np.array([rng.integers(0, int(k)) for k in cfg.circular_k], dtype=np.int64)
        if cfg.n_circular > 0
        else np.zeros(0, dtype=np.int64)
    )

    ind = Individual(
        real=real_vars,
        count=count_vars,
        binary=binary_vars,
        categorical=cat_vars,
        fuzzy=fuzzy_vars,
        circular=circular_vars,
        sigma_real=np.full(cfg.n_real, cfg.init_sigma_real, dtype=np.float64),
        lambda_count=np.full(cfg.n_count, cfg.init_lambda_count, dtype=np.float64),
        sigma_fuzzy=np.full(cfg.n_fuzzy * 3, cfg.init_sigma_fuzzy, dtype=np.float64),
        logit_binary=np.full(cfg.n_binary, cfg.init_logit_binary, dtype=np.float64),
        logit_cat=np.full(cfg.total_cat_logits, cfg.init_logit_cat, dtype=np.float64),
        logit_circular=np.full(cfg.n_circular, cfg.init_logit_circular, dtype=np.float64),
    )
    return ind


def _random_fuzzy_points(rng: np.random.Generator, n_fuzzy: int) -> NDArray[np.float64]:
    """Generate random fuzzy triples (a, b, c) satisfying a <= b <= c."""
    if n_fuzzy == 0:
        return np.zeros(0, dtype=np.float64)
    points = np.empty(n_fuzzy * 3, dtype=np.float64)
    for i in range(n_fuzzy):
        abc = np.sort(rng.random(3))
        points[i * 3: i * 3 + 3] = abc
    return points


# -----------------------------------------------------------------------------
# 3. Deep-Copy-Safe Child Cloning
# -----------------------------------------------------------------------------

def clone_individual(parent: Individual) -> Individual:
    """
    Create a fully independent deep copy of a parent individual.

    All NumPy arrays are copied so that mutations on the child
    do not affect the parent's state.
    """
    return Individual(
        real=parent.real.copy(),
        count=parent.count.copy(),
        binary=parent.binary.copy(),
        categorical=parent.categorical.copy(),
        fuzzy=parent.fuzzy.copy(),
        circular=parent.circular.copy(),
        sigma_real=parent.sigma_real.copy(),
        lambda_count=parent.lambda_count.copy(),
        sigma_fuzzy=parent.sigma_fuzzy.copy(),
        logit_binary=parent.logit_binary.copy(),
        logit_cat=parent.logit_cat.copy(),
        logit_circular=parent.logit_circular.copy(),
        fitness=parent.fitness,
    )


# -----------------------------------------------------------------------------
# 4. Child Mutation Pipeline
# -----------------------------------------------------------------------------

def mutate_individual(
    ind: Individual,
    rng: np.random.Generator,
    config: ESConfig | None = None,
) -> None:
    """
    Apply self-adaptive mutation.

    Steps:
    1. Adapt strategy parameters (log-normal for scales, logit-normal for probs).
    2. Mutate decision variables using adapted strategies.
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    tau_prime, tau = cfg.tau_prime_and_tau

    # Adapt Strategies
    ind.sigma_real = kernels.adapt_scales(ind.sigma_real, rng, tau_prime, tau, cfg.sigma_floor)
    ind.lambda_count = kernels.adapt_scales(ind.lambda_count, rng, tau_prime, tau, cfg.sigma_floor)
    ind.sigma_fuzzy = kernels.adapt_scales(ind.sigma_fuzzy, rng, tau_prime, tau, cfg.sigma_floor)
    ind.logit_binary = kernels.adapt_logits(ind.logit_binary, rng, tau)
    ind.logit_cat = kernels.adapt_logits(ind.logit_cat, rng, tau)
    ind.logit_circular = kernels.adapt_logits(ind.logit_circular, rng, tau)

    # Mutate Decision Variables
    ind.real = kernels.mutate_real(ind.real, ind.sigma_real, rng, cfg.real_bounds)
    ind.count = kernels.mutate_count(ind.count, ind.lambda_count, rng, cfg.count_bounds)
    ind.binary = kernels.mutate_binary(ind.binary, ind.logit_binary, rng)
    ind.categorical = kernels.mutate_categorical(ind.categorical, ind.logit_cat, rng, cfg.categorical_k)
    ind.fuzzy = kernels.mutate_fuzzy(ind.fuzzy, ind.sigma_fuzzy, rng, cfg.n_fuzzy, cfg.fuzzy_epsilon)
    ind.circular = kernels.mutate_circular(ind.circular, ind.logit_circular, rng, cfg.circular_k)


# -----------------------------------------------------------------------------
# 5. Deterministic Rank-Based Survivor Selection
# -----------------------------------------------------------------------------

def select_survivors(
    parents: list[Individual],
    child: list[Individual],
    mode: str = "comma",
    mu: int = 15,
) -> list[Individual]:
    """
    Deterministic rank-based selection of the top MU individuals.
    """
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

    def __init__(self, config: ESConfig | None = None, seed: int | None = None):
        self.config = config if config is not None else DEFAULT_CONFIG
        actual_seed = seed if seed is not None else self.config.seed
        self.rng = np.random.default_rng(actual_seed)
        self.parents: list[Individual] = []
        self.best_so_far: float = float("inf")
        self.best_individual: Individual | None = None
        self.evaluations: int = 0
        self.generation: int = 0

    def init_population(self, fn: Callable) -> None:
        """Initialize population and evaluate fitness of initial parents."""
        self.parents = [create_random_individual(self.rng, self.config) for _ in range(self.config.mu)]
        self.evaluations = 0
        self.generation = 0
        self.best_so_far = float("inf")

        for ind in self.parents:
            eval_input = ind.real if (self.config.n_real > 0 and self.config.n_total == self.config.n_real) else ind.genome
            ind.fitness = fn(eval_input)
            self.evaluations += 1
            if ind.fitness < self.best_so_far:
                self.best_so_far = ind.fitness
                self.best_individual = clone_individual(ind)

    def step(self, fn: Callable) -> float:
        """Execute one generational step."""
        self.generation += 1
        childs: list[Individual] = []

        for _ in range(self.config.lam):
            parent_idx = self.rng.integers(0, self.config.mu)
            child = clone_individual(self.parents[parent_idx])
            mutate_individual(child, self.rng, self.config)
            eval_input = child.real if (self.config.n_real > 0 and self.config.n_total == self.config.n_real) else child.genome
            child.fitness = fn(eval_input)
            self.evaluations += 1

            if child.fitness < self.best_so_far:
                self.best_so_far = child.fitness
                self.best_individual = clone_individual(child)

            childs.append(child)

        self.parents = select_survivors(self.parents, childs, mode=self.config.selection_mode, mu=self.config.mu)
        return self.best_so_far

    def optimize(self, fn: Callable, max_evals: int) -> NDArray[np.float64]:
        """Full optimization loop returning 1-D trace of best fitness per evaluation."""
        trace: list[float] = []
        self.parents = [create_random_individual(self.rng, self.config) for _ in range(self.config.mu)]
        self.evaluations = 0
        self.generation = 0
        self.best_so_far = float("inf")

        for ind in self.parents:
            eval_input = ind.real if (self.config.n_real > 0 and self.config.n_total == self.config.n_real) else ind.genome
            ind.fitness = fn(eval_input)
            if ind.fitness < self.best_so_far:
                self.best_so_far = ind.fitness
                self.best_individual = clone_individual(ind)
            trace.append(self.best_so_far)
            if len(trace) >= max_evals:
                return np.array(trace[:max_evals], dtype=np.float64)

        while len(trace) < max_evals:
            self.generation += 1
            childs: list[Individual] = []

            for _ in range(self.config.lam):
                parent_idx = self.rng.integers(0, self.config.mu)
                child = clone_individual(self.parents[parent_idx])
                mutate_individual(child, self.rng, self.config)
                eval_input = child.real if (self.config.n_real > 0 and self.config.n_total == self.config.n_real) else child.genome
                child.fitness = fn(eval_input)
                if child.fitness < self.best_so_far:
                    self.best_so_far = child.fitness
                    self.best_individual = clone_individual(child)
                trace.append(self.best_so_far)
                childs.append(child)
                if len(trace) >= max_evals:
                    break

            if len(childs) > 0:
                self.parents = select_survivors(self.parents, childs, mode=self.config.selection_mode, mu=self.config.mu)

        return np.array(trace[:max_evals], dtype=np.float64)
