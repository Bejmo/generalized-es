"""
individual.py: Candidate solution representation, variable encapsulation, and genetic operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from numpy.typing import NDArray

from .config import DEFAULT_CONFIG, ESConfig
from .variables import (
    Variable,
    RealVariable,
    CountVariable,
    BinaryVariable,
    CategoricalVariable,
    FuzzyVariable,
    CircularVariable,
)


# -----------------------------------------------------------------------------
# 1. Individual Data Structure
# -----------------------------------------------------------------------------

@dataclass
class Individual:
    """
    A single candidate solution in the Generalized ES.

    Decision variables and co-evolving strategy parameters are encapsulated
    in modular, vectorized Variable instances inside the `variables` dictionary.
    """

    variables: dict[str, Variable] = field(default_factory=dict)
    fitness: float = float("inf")

    @property
    def blocks(self) -> dict[str, Variable]:
        """Compatibility property mapping 'blocks' to 'variables'."""
        return self.variables

    @blocks.setter
    def blocks(self, val: dict[str, Variable]) -> None:
        self.variables = val

    @property
    def genome(self) -> NDArray[np.float64]:
        """
        Return a single flattened scalar array of all decision variable vectors.

        Canonical Order: real -> count -> binary -> categorical -> fuzzy -> circular.
        Excludes all strategy parameters.
        Used for logging, plotting, and scalar evaluation functions.
        """
        canonical_order = ["real", "count", "binary", "categorical", "fuzzy", "circular"]
        parts = [self.variables[k].flatten() for k in canonical_order if k in self.variables]
        if not parts:
            return np.zeros(0, dtype=np.float64)
        return np.concatenate(parts)

    # Direct variable accessors
    @property
    def real(self) -> NDArray[np.float64]:
        return self.variables["real"].values.astype(np.float64) if "real" in self.variables else np.zeros(0, dtype=np.float64)

    @property
    def count(self) -> NDArray[np.int64]:
        return self.variables["count"].values.astype(np.int64) if "count" in self.variables else np.zeros(0, dtype=np.int64)

    @property
    def binary(self) -> NDArray[np.uint8]:
        return self.variables["binary"].values.astype(np.uint8) if "binary" in self.variables else np.zeros(0, dtype=np.uint8)

    @property
    def categorical(self) -> NDArray[np.int64]:
        return self.variables["categorical"].values.astype(np.int64) if "categorical" in self.variables else np.zeros(0, dtype=np.int64)

    @property
    def fuzzy(self) -> NDArray[np.float64]:
        return self.variables["fuzzy"].values.astype(np.float64) if "fuzzy" in self.variables else np.zeros(0, dtype=np.float64)

    @property
    def circular(self) -> NDArray[np.int64]:
        return self.variables["circular"].values.astype(np.int64) if "circular" in self.variables else np.zeros(0, dtype=np.int64)

    @property
    def sigma_real(self) -> NDArray[np.float64]:
        return self.variables["real"].strategy if "real" in self.variables else np.zeros(0, dtype=np.float64)

    @property
    def lambda_count(self) -> NDArray[np.float64]:
        return self.variables["count"].strategy if "count" in self.variables else np.zeros(0, dtype=np.float64)

    @property
    def logit_binary(self) -> NDArray[np.float64]:
        return self.variables["binary"].strategy if "binary" in self.variables else np.zeros(0, dtype=np.float64)

    @property
    def logit_cat(self) -> NDArray[np.float64]:
        return self.variables["categorical"].strategy if "categorical" in self.variables else np.zeros(0, dtype=np.float64)

    @property
    def sigma_fuzzy(self) -> NDArray[np.float64]:
        return self.variables["fuzzy"].strategy if "fuzzy" in self.variables else np.zeros(0, dtype=np.float64)

    @property
    def lambda_circular(self) -> NDArray[np.float64]:
        return self.variables["circular"].strategy if "circular" in self.variables else np.zeros(0, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Lifecycle & Genetic Operator Methods
    # -------------------------------------------------------------------------

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        config: ESConfig | None = None,
    ) -> Individual:
        """Create a randomly initialized individual within configured bounds."""
        cfg = config if config is not None else DEFAULT_CONFIG
        variables: dict[str, Variable] = {}

        if cfg.n_real > 0:
            variables["real"] = RealVariable.random(cfg, rng)
        if cfg.n_count > 0:
            variables["count"] = CountVariable.random(cfg, rng)
        if cfg.n_binary > 0:
            variables["binary"] = BinaryVariable.random(cfg, rng)
        if cfg.n_categorical > 0:
            variables["categorical"] = CategoricalVariable.random(cfg, rng)
        if cfg.n_fuzzy > 0:
            variables["fuzzy"] = FuzzyVariable.random(cfg, rng)
        if cfg.n_circular > 0:
            variables["circular"] = CircularVariable.random(cfg, rng)

        return cls(variables=variables)

    def clone(self) -> Individual:
        """
        Create a fully independent deep copy of this individual.

        NumPy arrays are copied inside each variable block so that subsequent mutations
        do not affect this individual's state.
        """
        return Individual(
            variables={name: var.copy() for name, var in self.variables.items()},
            fitness=self.fitness,
        )

    def recombine(
        self,
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
            return self.clone()

        # Select a second distinct parent for discrete recombination
        idx_b = int(rng.integers(0, mu))
        while parents[idx_b] is self and mu > 1:
            idx_b = int(rng.integers(0, mu))
        parent_b = parents[idx_b]

        child_vars: dict[str, Variable] = {
            name: var.copy() for name, var in self.variables.items()
        }

        # 1. Dual discrete recombination for decision variables
        for name, var_a in self.variables.items():
            if name in parent_b.variables:
                child_vars[name].values = var_a.recombine_values(parent_b.variables[name], rng)

        # 2. Global intermediate recombination for strategy parameters
        for name, var_a in self.variables.items():
            donor_vars = [p.variables[name] for p in parents if name in p.variables]
            child_vars[name].strategy = var_a.recombine_strategy(donor_vars, rng)

        return Individual(variables=child_vars)

    def mutate(
        self,
        rng: np.random.Generator,
        config: ESConfig | None = None,
    ) -> None:
        """
        Apply self-adaptive mutation.

        Steps:
        1. Adapt strategy parameters across all variable blocks.
        2. Mutate decision variables using the adapted strategies.
        """
        cfg = config if config is not None else DEFAULT_CONFIG
        tau_prime, tau = cfg.tau_prime_and_tau

        # Adapt strategy parameters first
        for var in self.variables.values():
            var.adapt_strategy(rng, tau_prime, tau)

        # Mutate decision variables using updated strategies
        for var in self.variables.values():
            var.mutate(rng)


# -----------------------------------------------------------------------------
# 2. Unpacked Genome Container
# -----------------------------------------------------------------------------

@dataclass
class UnpackedGenome:
    """Convenience container for un-concatenated decision variable vectors."""

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
    Safely extract all 6 decision variable vectors from either an Individual object
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
# 3. Functional Wrappers for Compatibility
# -----------------------------------------------------------------------------

def create_random_individual(
    rng: np.random.Generator,
    config: ESConfig | None = None,
) -> Individual:
    """Create a randomly initialized individual (functional wrapper)."""
    return Individual.random(rng, config)


def clone_individual(parent: Individual) -> Individual:
    """Create a deep copy of an individual (functional wrapper)."""
    return parent.clone()


def recombine_individual(
    parent_a: Individual,
    parents: list[Individual],
    rng: np.random.Generator,
    config: ESConfig | None = None,
) -> Individual:
    """Recombine parents into an offspring individual (functional wrapper)."""
    return parent_a.recombine(parents, rng, config)


def mutate_individual(
    ind: Individual,
    rng: np.random.Generator,
    config: ESConfig | None = None,
) -> None:
    """Mutate an individual in-place (functional wrapper)."""
    ind.mutate(rng, config)
