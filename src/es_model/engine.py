"""
engine.py — Population dynamics, selection policies, and structural invariants.

Defines the Individual data class, offspring cloning, deterministic rank-based
survivor selection for both (mu, lambda) and (mu + lambda) modes, and the
flattened genome output contract.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from . import config as cfg
from . import kernels


# =============================================================================
# 1. Individual Data Structure
# =============================================================================

@dataclass
class Individual:
    """
    A single candidate solution in the heterogeneous ES.

    Contains typed genome blocks and their associated strategy parameters.
    Strategy parameters co-evolve with the genome via self-adaptation.
    """

    # ── Decision Variable Blocks ──────────────────────────────────────────
    real: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    count: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    binary: NDArray[np.uint8] = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))
    categorical: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    fuzzy: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))

    # ── Strategy Parameters ───────────────────────────────────────────────
    sigma_real: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    lambda_count: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    sigma_fuzzy: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    logit_binary: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    logit_cat: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))

    # ── Fitness ───────────────────────────────────────────────────────────
    fitness: float = float("inf")

    # ─────────────────────────────────────────────────────────────────────
    @property
    def genome(self) -> NDArray[np.float64]:
        """
        Return a single flattened scalar array of all decision variable blocks.

        Order: real → count → binary → categorical → fuzzy control points.
        Excludes all strategy parameters (sigmas/logits).
        Used exclusively for logging and plotting.
        """
        return np.concatenate([
            self.real.astype(np.float64),
            self.count.astype(np.float64),
            self.binary.astype(np.float64),
            self.categorical.astype(np.float64),
            self.fuzzy.astype(np.float64),
        ])


# =============================================================================
# 2. Random Initialization
# =============================================================================

def create_random_individual(rng: np.random.Generator) -> Individual:
    """Create a randomly initialized individual within configured bounds."""
    ind = Individual(
        # Decision variables — uniform random within bounds.
        real=rng.uniform(cfg.REAL_BOUNDS[:, 0], cfg.REAL_BOUNDS[:, 1]).astype(np.float64),
        count=rng.integers(
            cfg.COUNT_BOUNDS[:, 0], cfg.COUNT_BOUNDS[:, 1] + 1, dtype=np.int64
        ),
        binary=rng.integers(0, 2, size=cfg.N_BINARY, dtype=np.uint8),
        categorical=np.array(
            [rng.integers(0, int(k)) for k in cfg.CATEGORICAL_K], dtype=np.int64
        ),
        fuzzy=_random_fuzzy_points(rng),

        # Strategy parameters — initialized to configured defaults.
        sigma_real=np.full(cfg.N_REAL, cfg.INIT_SIGMA_REAL, dtype=np.float64),
        lambda_count=np.full(cfg.N_COUNT, cfg.INIT_LAMBDA_COUNT, dtype=np.float64),
        sigma_fuzzy=np.full(cfg.N_FUZZY * 3, cfg.INIT_SIGMA_FUZZY, dtype=np.float64),
        logit_binary=np.full(cfg.N_BINARY, cfg.INIT_LOGIT_BINARY, dtype=np.float64),
        logit_cat=np.full(cfg.N_CATEGORICAL, cfg.INIT_LOGIT_CAT, dtype=np.float64),
    )
    return ind


def _random_fuzzy_points(rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate random fuzzy triples (a, b, c) satisfying a <= b <= c."""
    points = np.empty(cfg.N_FUZZY * 3, dtype=np.float64)
    for i in range(cfg.N_FUZZY):
        abc = np.sort(rng.random(3))
        points[i * 3: i * 3 + 3] = abc
    return points


# =============================================================================
# 3. Deep-Copy-Safe Offspring Cloning
# =============================================================================

def clone_individual(parent: Individual) -> Individual:
    """
    Create a fully independent deep copy of a parent individual.

    All NumPy arrays are copied so that mutations on the offspring
    do not affect the parent's state.
    """
    return Individual(
        real=parent.real.copy(),
        count=parent.count.copy(),
        binary=parent.binary.copy(),
        categorical=parent.categorical.copy(),
        fuzzy=parent.fuzzy.copy(),
        sigma_real=parent.sigma_real.copy(),
        lambda_count=parent.lambda_count.copy(),
        sigma_fuzzy=parent.sigma_fuzzy.copy(),
        logit_binary=parent.logit_binary.copy(),
        logit_cat=parent.logit_cat.copy(),
        fitness=parent.fitness,
    )


# =============================================================================
# 4. Offspring Mutation Pipeline
# =============================================================================

def mutate_individual(ind: Individual, rng: np.random.Generator) -> None:
    """
    Apply the full self-adaptive mutation pipeline to an individual IN-PLACE.

    Steps:
    1. Generate sparse masks per block.
    2. Adapt strategy parameters (log-normal for scales, logit-normal for probs).
    3. Mutate decision variables using adapted strategies.
    """
    # ── Sparse Masks ──────────────────────────────────────────────────────
    mask_real = kernels.generate_sparse_mask(cfg.N_REAL, rng)
    mask_count = kernels.generate_sparse_mask(cfg.N_COUNT, rng)
    mask_binary = kernels.generate_sparse_mask(cfg.N_BINARY, rng)
    mask_cat = kernels.generate_sparse_mask(cfg.N_CATEGORICAL, rng)
    mask_fuzzy = kernels.generate_sparse_mask(cfg.N_FUZZY, rng)

    # ── Adapt Strategies ──────────────────────────────────────────────────
    ind.sigma_real = kernels.adapt_scales(ind.sigma_real, rng, mask_real)
    ind.lambda_count = kernels.adapt_scales(ind.lambda_count, rng, mask_count)
    ind.sigma_fuzzy = kernels.adapt_scales(ind.sigma_fuzzy, rng, _expand_fuzzy_mask(mask_fuzzy))
    ind.logit_binary = kernels.adapt_logits(ind.logit_binary, rng, mask_binary)
    ind.logit_cat = kernels.adapt_logits(ind.logit_cat, rng, mask_cat)

    # ── Mutate Decision Variables ─────────────────────────────────────────
    ind.real = kernels.mutate_real(ind.real, ind.sigma_real, rng, mask_real)
    ind.count = kernels.mutate_count(ind.count, ind.lambda_count, rng, mask_count)
    ind.binary = kernels.mutate_binary(ind.binary, ind.logit_binary, rng, mask_binary)
    ind.categorical = kernels.mutate_categorical(ind.categorical, ind.logit_cat, rng, mask_cat)
    ind.fuzzy = kernels.mutate_fuzzy(ind.fuzzy, ind.sigma_fuzzy, rng, mask_fuzzy)


def _expand_fuzzy_mask(mask_per_var: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Expand per-variable fuzzy mask (N_FUZZY,) to per-point mask (N_FUZZY*3,)."""
    return np.repeat(mask_per_var, 3)


# =============================================================================
# 5. Deterministic Rank-Based Survivor Selection
# =============================================================================

def select_survivors(
    parents: list[Individual],
    offspring: list[Individual],
    mode: str = cfg.SELECTION_MODE,
) -> list[Individual]:
    """
    Deterministic rank-based selection of the top MU individuals.

    Parameters
    ----------
    parents : list of Individual
        Current parent population.
    offspring : list of Individual
        Newly generated offspring.
    mode : str
        "comma" — select from offspring only.
        "plus"  — select from parents + offspring combined.

    Returns
    -------
    list of Individual
        The top MU survivors sorted by fitness (ascending = better).

    Notes
    -----
    Ties are broken by stable positional order (earlier index wins),
    guaranteeing reproducible selection across runs.
    """
    if mode == "comma":
        pool = offspring
    elif mode == "plus":
        pool = parents + offspring
    else:
        raise ValueError(f"Unknown selection mode: {mode!r}. Use 'comma' or 'plus'.")

    # Stable sort: Python's sorted is stable, so equal-fitness candidates
    # retain their insertion order (deterministic tie-breaking).
    ranked = sorted(pool, key=lambda ind: ind.fitness)
    return ranked[: cfg.MU]
