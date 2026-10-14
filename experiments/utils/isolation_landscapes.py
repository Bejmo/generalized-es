"""
isolation_landscapes.py: Fitness landscapes for Experiment 3 (Variable Isolation and Operator Validation).

Provides purpose-built landscapes for testing each of the 6 variable types in isolation:
1. DTSLandscape: Dynamic Target Shift wrapper (Arnold & Beyer, 2006).
2. integer_staircase: Multi-scale staircase with varying step sizes.
3. binary_hamming: Hamming distance on binary bitstrings.
4. categorical_deceptive: Deceptive landscape with local suboptimum.
5. CategoricalCycling: Dynamic cycling target across categories.
6. fuzzy_jaccard: Triangular fuzzy number matching via Jaccard similarity.
7. circular_geodesic: Shortest geodesic distance on discrete ring Z_k.
8. circular_clamped_baseline: Non-modular clamped distance baseline.
9. real_boundary_target: Sphere with target adjacent to upper bound.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from numpy.typing import NDArray

from src.es_model.config import ESConfig, DEFAULT_CONFIG
from src.es_model.individual import Individual, unpack_genome


class DTSLandscape:
    """
    Dynamic Target Shift (DTS) landscape wrapper (Arnold & Beyer, 2006).

    Optimizes toward target t1 for generations g < g_shift, then instantaneously
    shifts the target to t2 for generations g >= g_shift.

    Parameters
    ----------
    config : ESConfig
        Evolution Strategy configuration.
    var_type : str
        Variable type under test: 'real', 'count', 'binary', 'categorical', 'fuzzy', or 'circular'.
    t1 : Any
        Target vector or scalar for Phase A (g < g_shift).
    t2 : Any
        Target vector or scalar for Phase C (g >= g_shift).
    g_shift : int, default=500
        Generation at which the target shifts from t1 to t2.
    """

    def __init__(
        self,
        config: ESConfig,
        var_type: str,
        t1: Any,
        t2: Any,
        g_shift: int = 500,
    ) -> None:
        self.config = config
        self.var_type = var_type
        self.g_shift = g_shift
        self.current_gen: int = 0
        self.t1 = self._format_target(t1)
        self.t2 = self._format_target(t2)

    def set_generation(self, g: int) -> None:
        """Update current generation counter."""
        self.current_gen = g

    def _format_target(self, val: Any) -> np.ndarray:
        if self.var_type == "real":
            return np.asarray(val, dtype=np.float64) if np.ndim(val) > 0 else np.full(self.config.n_real, val, dtype=np.float64)
        elif self.var_type == "count":
            return np.asarray(val, dtype=np.int64) if np.ndim(val) > 0 else np.full(self.config.n_count, val, dtype=np.int64)
        elif self.var_type == "binary":
            return np.asarray(val, dtype=np.uint8) if np.ndim(val) > 0 else np.full(self.config.n_binary, val, dtype=np.uint8)
        elif self.var_type == "categorical":
            return np.asarray(val, dtype=np.int64) if np.ndim(val) > 0 else np.full(self.config.n_categorical, val, dtype=np.int64)
        elif self.var_type == "fuzzy":
            # Target fuzzy parameters (m, d), shape (2 * n_fuzzy,)
            arr = np.asarray(val, dtype=np.float64)
            if arr.ndim == 1 and len(arr) == 2 and self.config.n_fuzzy > 1:
                return np.tile(arr, self.config.n_fuzzy)
            return arr
        elif self.var_type == "circular":
            return np.asarray(val, dtype=np.int64) if np.ndim(val) > 0 else np.full(self.config.n_circular, val, dtype=np.int64)
        else:
            raise ValueError(f"Unknown var_type: {self.var_type!r}")

    @property
    def current_target(self) -> np.ndarray:
        """Active target for the current generation."""
        return self.t1 if self.current_gen < self.g_shift else self.t2

    def __call__(self, x_or_ind: Individual | NDArray[np.float64]) -> float:
        """Evaluate fitness of candidate solution against active target."""
        u = unpack_genome(x_or_ind, self.config)
        target = self.current_target

        if self.var_type == "real":
            return float(np.sum((u.real - target) ** 2))
        elif self.var_type == "count":
            return float(np.sum((u.count - target) ** 2))
        elif self.var_type == "binary":
            return float(np.sum(u.binary != target))
        elif self.var_type == "categorical":
            return float(np.sum(u.categorical != target))
        elif self.var_type == "fuzzy":
            m1, d1 = u.fuzzy[0::2], u.fuzzy[1::2]
            m2, d2 = target[0::2], target[1::2]
            return float(3.0 * np.sum((m1 - m2) ** 2) + 2.0 * np.sum((d1 - d2) ** 2))
        elif self.var_type == "circular":
            k = self.config.circular_k if len(self.config.circular_k) > 0 else np.full(self.config.n_circular, 12, dtype=np.int64)
            diff = np.abs(u.circular - target) % k
            return float(np.sum(np.minimum(diff, k - diff) ** 2))
        else:
            raise ValueError(f"Unknown var_type: {self.var_type!r}")


# -----------------------------------------------------------------------------
# Type-Specific Invariant Test Landscapes
# -----------------------------------------------------------------------------

def real_boundary_target(
    x_or_ind: Individual | NDArray[np.float64],
    config: ESConfig | None = None,
    targets: Any = None,
) -> float:
    """
    Real boundary stress test: target placed at epsilon=0.01 inside upper boundary.

    f(x) = sum (x_i - (upper_i - 0.01))^2
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    u = unpack_genome(x_or_ind, cfg)
    ub = cfg.real_bounds[:, 1]
    target = ub - 0.01
    return float(np.sum((u.real - target) ** 2))


def integer_staircase(
    x_or_ind: Individual | NDArray[np.float64],
    config: ESConfig | None = None,
    targets: Any = None,
) -> float:
    """
    Multi-scale integer staircase landscape: f(x) = sum |x_i - t_i|.

    Stresses Skellam Poisson rate adaptation across vastly different jump scales.
    Default targets: [0, 1, 5, 10, 20] (or repeating if n_count differs).
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    u = unpack_genome(x_or_ind, cfg)
    if targets is None:
        ladder = np.array([0, 1, 5, 10, 20], dtype=np.int64)
        targets = np.resize(ladder, cfg.n_count)
    else:
        targets = np.asarray(targets, dtype=np.int64)
    return float(np.sum(np.abs(u.count - targets)))


def binary_hamming(
    x_or_ind: Individual | NDArray[np.float64],
    config: ESConfig | None = None,
    targets: Any = None,
) -> float:
    """
    Binary Hamming distance landscape: f(x) = sum(x_i != t_i).
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    u = unpack_genome(x_or_ind, cfg)
    if targets is None:
        targets = np.ones(cfg.n_binary, dtype=np.uint8)
    else:
        targets = np.asarray(targets, dtype=np.uint8)
    return float(np.sum(u.binary != targets))


def categorical_deceptive(
    x_or_ind: Individual | NDArray[np.float64],
    config: ESConfig | None = None,
    targets: Any = None,
) -> float:
    """
    Categorical deceptive landscape.

    Category 0 is global optimum (cost=0.0).
    Category 1 is deceptive local attractor (cost=0.1).
    All other categories cost 1.0.
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    u = unpack_genome(x_or_ind, cfg)
    cost = 0.0
    for val in u.categorical:
        if val == 0:
            cost += 0.0
        elif val == 1:
            cost += 0.1
        else:
            cost += 1.0
    return float(cost)


class CategoricalCycling:
    """
    Categorical cycling target that rotates the optimal category uniformly
    across all k categories over g_max generations.

    Forces sustained exploration and checks that softmax entropy does not collapse.
    """

    def __init__(self, config: ESConfig, k: int = 5, g_max: int = 500) -> None:
        self.config = config
        self.k = k
        self.g_max = g_max
        self.current_gen: int = 0

    def set_generation(self, g: int) -> None:
        self.current_gen = g

    def __call__(self, x_or_ind: Individual | NDArray[np.float64]) -> float:
        u = unpack_genome(x_or_ind, self.config)
        step = max(1, self.g_max // self.k)
        active_cat = (self.current_gen // step) % self.k
        return float(np.sum(u.categorical != active_cat))


def fuzzy_jaccard(
    x_or_ind: Individual | NDArray[np.float64],
    config: ESConfig | None = None,
    targets: Any = None,
) -> float:
    """
    Triangular fuzzy number matching via Jaccard similarity.

    f(x) = sum(1 - J(F_i, T_i))

    where J(F, T) is the intersection area divided by union area of two
    triangular membership functions over [0, 1].
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    u = unpack_genome(x_or_ind, cfg)

    if targets is None:
        # Default target: center=0.5, dispersion=0.3 per variable
        targets = np.tile([0.5, 0.3], cfg.n_fuzzy)
    else:
        arr = np.asarray(targets, dtype=np.float64)
        if arr.ndim == 1 and len(arr) == 2 and cfg.n_fuzzy > 1:
            targets = np.tile(arr, cfg.n_fuzzy)
        else:
            targets = arr

    total_cost = 0.0
    grid = np.linspace(0.0, 1.0, 200)

    for i in range(cfg.n_fuzzy):
        m1, d1 = u.fuzzy[i * 2 : i * 2 + 2]
        m2, d2 = targets[i * 2 : i * 2 + 2]

        # Evaluate triangular membership functions over [0, 1]
        y1 = np.maximum(0.0, 1.0 - np.abs(grid - m1) / max(d1, 1e-9))
        y2 = np.maximum(0.0, 1.0 - np.abs(grid - m2) / max(d2, 1e-9))

        intersection = np.trapezoid(np.minimum(y1, y2), grid) if hasattr(np, "trapezoid") else np.trapz(np.minimum(y1, y2), grid)
        area1 = np.trapezoid(y1, grid) if hasattr(np, "trapezoid") else np.trapz(y1, grid)
        area2 = np.trapezoid(y2, grid) if hasattr(np, "trapezoid") else np.trapz(y2, grid)
        union = area1 + area2 - intersection

        if union <= 1e-12:
            jaccard = 1.0 if intersection <= 1e-12 else 0.0
        else:
            jaccard = float(intersection / union)

        total_cost += (1.0 - jaccard)

    return float(total_cost)


def circular_geodesic(
    x_or_ind: Individual | NDArray[np.float64],
    config: ESConfig | None = None,
    targets: Any = None,
) -> float:
    """
    Circular geodesic landscape: sum min(|x_i - t_i|, k_i - |x_i - t_i|)^2.
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    u = unpack_genome(x_or_ind, cfg)
    k = cfg.circular_k if len(cfg.circular_k) > 0 else np.full(cfg.n_circular, 12, dtype=np.int64)
    if targets is None:
        targets = np.full(cfg.n_circular, k // 2, dtype=np.int64)
    else:
        targets = np.asarray(targets, dtype=np.int64)

    diff = np.abs(u.circular - targets) % k
    geo_dist = np.minimum(diff, k - diff)
    return float(np.sum(geo_dist ** 2))


def circular_clamped_baseline(
    x_or_ind: Individual | NDArray[np.float64],
    config: ESConfig | None = None,
    targets: Any = None,
) -> float:
    """
    Naïve non-modular linear baseline on circular variables: sum (x_i - t_i)^2 without wrap-around.
    Used in INV-S to benchmark the geodesic shortcut advantage.
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    u = unpack_genome(x_or_ind, cfg)
    k = cfg.circular_k if len(cfg.circular_k) > 0 else np.full(cfg.n_circular, 12, dtype=np.int64)
    if targets is None:
        targets = np.full(cfg.n_circular, k // 2, dtype=np.int64)
    else:
        targets = np.asarray(targets, dtype=np.int64)

    diff = u.circular - targets
    return float(np.sum(diff ** 2))
