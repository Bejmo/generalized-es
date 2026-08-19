"""
heterogeneous_functions.py — Synthetic mixed-variable benchmark functions.

Provides synthetic benchmark landscapes (Sphere, Rastrigin, Coupled) that test
all 6 decision variable types co-evolving in the Heterogeneous ES:
- Real (continuous)
- Count (integer)
- Binary (boolean/logit)
- Categorical (discrete categories)
- Fuzzy (triangular membership control points a <= b <= c)
- Circular (closed-order positions 0..k-1)
"""

from typing import Any
import numpy as np
from numpy.typing import NDArray

from src.es_model.config import ESConfig, DEFAULT_CONFIG
from src.es_model.engine import Individual, unpack_genome



# -----------------------------------------------------------------------------
# Default Target Optima Generator
# -----------------------------------------------------------------------------

def get_default_targets(config: ESConfig) -> dict[str, Any]:
    """Generate deterministic target optima for synthetic benchmarks."""
    circular_k = config.circular_k if len(config.circular_k) > 0 else np.full(config.n_circular, 7, dtype=np.int64)
    
    # Target fuzzy control points: [0.2, 0.5, 0.8] repeated per fuzzy variable
    fuzzy_target = np.tile([0.2, 0.5, 0.8], config.n_fuzzy) if config.n_fuzzy > 0 else np.zeros(0)

    return {
        "real": np.full(config.n_real, 1.5, dtype=np.float64),
        "count": np.full(config.n_count, 5, dtype=np.int64),
        "binary": np.ones(config.n_binary, dtype=np.uint8),
        "categorical": np.zeros(config.n_categorical, dtype=np.int64),
        "fuzzy": fuzzy_target,
        "circular": np.full(config.n_circular, 2, dtype=np.int64),
        "circular_k": circular_k,
    }


# -----------------------------------------------------------------------------
# 1. Heterogeneous Sphere Benchmark
# -----------------------------------------------------------------------------

def heterogeneous_sphere(
    x_or_ind: Individual | NDArray[np.float64],
    config: ESConfig | None = None,
    targets: dict[str, Any] | None = None,
) -> float:
    """
    Convex, separable mixed-variable Sphere landscape across all 6 variable types.
    Global minimum is 0.0 when all variables match target optima.
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    t = targets if targets is not None else get_default_targets(cfg)
    u = unpack_genome(x_or_ind, cfg)

    # 1. Real: sum of squared errors
    cost_real = float(np.sum((u.real - t["real"]) ** 2)) if cfg.n_real > 0 else 0.0

    # 2. Count: sum of squared integer differences
    cost_count = float(np.sum((u.count - t["count"]) ** 2)) if cfg.n_count > 0 else 0.0

    # 3. Binary: Hamming distance
    cost_binary = float(np.sum(u.binary != t["binary"])) if cfg.n_binary > 0 else 0.0

    # 4. Categorical: Misclassification count
    cost_cat = float(np.sum(u.categorical != t["categorical"])) if cfg.n_categorical > 0 else 0.0

    # 5. Fuzzy: Control points Euclidean distance
    cost_fuzzy = float(np.sum((u.fuzzy - t["fuzzy"]) ** 2)) if cfg.n_fuzzy > 0 else 0.0

    # 6. Circular: Shortest modular distance on circle
    cost_circ = 0.0
    if cfg.n_circular > 0:
        k = t["circular_k"]
        diff = np.abs(u.circular - t["circular"]) % k
        circular_dist = np.minimum(diff, k - diff)
        cost_circ = float(np.sum(circular_dist ** 2))

    return cost_real + cost_count + cost_binary + cost_cat + cost_fuzzy + cost_circ


# -----------------------------------------------------------------------------
# 2. Heterogeneous Rastrigin Benchmark
# -----------------------------------------------------------------------------

def heterogeneous_rastrigin(
    x_or_ind: Individual | NDArray[np.float64],
    config: ESConfig | None = None,
    targets: dict[str, Any] | None = None,
) -> float:
    """
    Multi-modal non-convex mixed-variable Rastrigin landscape.
    Combines cosine oscillations across continuous and discrete dimensions.
    Global minimum is 0.0.
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    t = targets if targets is not None else get_default_targets(cfg)
    u = unpack_genome(x_or_ind, cfg)

    # 1. Real Rastrigin
    cost_real = 0.0
    if cfg.n_real > 0:
        diff_r = u.real - t["real"]
        cost_real = float(10.0 * cfg.n_real + np.sum(diff_r ** 2 - 10.0 * np.cos(2.0 * np.pi * diff_r)))

    # 2. Count Rastrigin
    cost_count = 0.0
    if cfg.n_count > 0:
        diff_c = (u.count - t["count"]).astype(np.float64)
        cost_count = float(2.0 * cfg.n_count + np.sum(diff_c ** 2 - 2.0 * np.cos(np.pi * diff_c)))

    # 3. Binary Hamming Penalty
    cost_binary = float(5.0 * np.sum(u.binary != t["binary"])) if cfg.n_binary > 0 else 0.0

    # 4. Categorical Misclassification Penalty
    cost_cat = float(10.0 * np.sum(u.categorical != t["categorical"])) if cfg.n_categorical > 0 else 0.0

    # 5. Fuzzy Control Points Rastrigin
    cost_fuzzy = 0.0
    if cfg.n_fuzzy > 0:
        diff_f = u.fuzzy - t["fuzzy"]
        n_pts = cfg.n_fuzzy * 3
        cost_fuzzy = float(5.0 * n_pts + np.sum(diff_f ** 2 - 5.0 * np.cos(2.0 * np.pi * diff_f)))

    # 6. Circular Phase Cosine Distance
    cost_circ = 0.0
    if cfg.n_circular > 0:
        k = t["circular_k"]
        phase = 2.0 * np.pi * (u.circular - t["circular"]) / k
        cost_circ = float(5.0 * np.sum(1.0 - np.cos(phase)))

    return cost_real + cost_count + cost_binary + cost_cat + cost_fuzzy + cost_circ


# -----------------------------------------------------------------------------
# 3. Heterogeneous Coupled Benchmark
# -----------------------------------------------------------------------------

def heterogeneous_coupled(
    x_or_ind: Individual | NDArray[np.float64],
    config: ESConfig | None = None,
    targets: dict[str, Any] | None = None,
) -> float:
    """
    Coupled mixed-variable landscape where discrete choices (categorical, binary)
    dynamically alter the active target regime and scale factors for real and fuzzy variables.
    Global minimum is 0.0.
    """
    cfg = config if config is not None else DEFAULT_CONFIG
    t = targets if targets is not None else get_default_targets(cfg)
    u = unpack_genome(x_or_ind, cfg)

    # Categorical 0 selects active regime offset for real variables
    regime_offset = float(u.categorical[0]) if cfg.n_categorical > 0 else 0.0
    target_real_shifted = t["real"] + regime_offset * 0.5

    # Binary mask: inactive bits (0) incur penalty
    binary_penalty = float(np.sum(u.binary != 1)) * 3.0 if cfg.n_binary > 0 else 0.0

    # Count variable scales fuzzy penalty multiplier
    count_scale = float(1.0 + 0.1 * np.abs(u.count[0] - t["count"][0])) if cfg.n_count > 0 else 1.0

    # Real component under active regime
    cost_real = float(np.sum((u.real - target_real_shifted) ** 2)) if cfg.n_real > 0 else 0.0

    # Categorical penalty (category 0 is optimal)
    cost_cat = float(5.0 * np.sum(u.categorical != 0)) if cfg.n_categorical > 0 else 0.0

    # Count component
    cost_count = float(np.sum((u.count - t["count"]) ** 2)) if cfg.n_count > 0 else 0.0

    # Fuzzy component scaled by count error
    cost_fuzzy = float(count_scale * np.sum((u.fuzzy - t["fuzzy"]) ** 2)) if cfg.n_fuzzy > 0 else 0.0

    # Circular component
    cost_circ = 0.0
    if cfg.n_circular > 0:
        k = t["circular_k"]
        diff = np.abs(u.circular - t["circular"]) % k
        cost_circ = float(np.sum(np.minimum(diff, k - diff) ** 2))

    return cost_real + cost_count + cost_cat + binary_penalty + cost_fuzzy + cost_circ


# Dictionary exposing benchmarks
HETEROGENEOUS_BENCHMARKS = {
    "Heterogeneous_Sphere": {
        "fn": heterogeneous_sphere,
        "description": "Convex separable mixed-variable benchmark across all 6 variable types",
    },
    "Heterogeneous_Rastrigin": {
        "fn": heterogeneous_rastrigin,
        "description": "Multi-modal non-convex mixed-variable benchmark with cosine oscillations",
    },
    "Heterogeneous_Coupled": {
        "fn": heterogeneous_coupled,
        "description": "Coupled mixed-variable benchmark with discrete regime switches",
    },
}
