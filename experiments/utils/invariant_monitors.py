"""
invariant_monitors.py: Invariant monitoring functions for Experiment 3.

Checks domain constraints and structural invariants during the evolutionary cycle:
1. Real bounds: Reflection correctness, reflection counts, and [lower, upper] validity.
2. Count bounds and integrality: integer clamping and natural step-zero preservation.
3. Binary validity and ceiling contact: uint8 {0, 1} validity and logit ceiling events.
4. Categorical gauge residual and entropy: sum(z) = 0 invariance, softmax entropy, and logit norm.
5. Fuzzy monotonicity and support width: a <= b <= c ordering, [0, 1] unit bounding, and epsilon spacing.
6. Circular range and geodesics: discrete range [0, k_i) and modular distance on Z_k.
7. Aggregate monitor: runs all applicable checks and returns a summary dict.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from src.es_model.config import ESConfig, DEFAULT_CONFIG
from src.es_model.individual import Individual


# 1. Real Variable Monitor

def check_real_bounds(individual: Individual, config: ESConfig) -> int:
    """
    Check if any real value is outside domain bounds.

    Returns
    -------
    int
        Number of real bounds violations.
    """
    vals = individual.real
    if len(vals) == 0:
        return 0
    bounds = config.real_bounds
    violations = np.sum((vals < bounds[:, 0]) | (vals > bounds[:, 1]))
    return int(violations)


def count_reflections(values_before: np.ndarray, values_after: np.ndarray, bounds: np.ndarray) -> int:
    """
    Count how many real values changed direction due to boundary reflection.
    """
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    reflections = np.sum((values_before < lb) | (values_before > ub))
    return int(reflections)


# 2. Integer / Count Variable Monitor

def check_count_bounds(individual: Individual, config: ESConfig) -> int:
    """
    Check if any count value is outside domain bounds.
    """
    vals = individual.count
    if len(vals) == 0:
        return 0
    bounds = config.count_bounds
    violations = np.sum((vals < bounds[:, 0]) | (vals > bounds[:, 1]))
    return int(violations)


def check_count_integrality(individual: Individual) -> int:
    """
    Check if count values are exact integers without rounding artifacts.
    """
    vals = individual.count
    if len(vals) == 0:
        return 0
    # Values are stored as np.int64, so violations are inherently 0, but check for any non-integer representation
    violations = np.sum(np.abs(vals - np.round(vals)) > 1e-9)
    return int(violations)


# 3. Binary Variable Monitor

def check_binary_validity(individual: Individual) -> int:
    """
    Check that all binary values are strictly in {0, 1}.
    """
    vals = individual.binary
    if len(vals) == 0:
        return 0
    violations = np.sum(~np.isin(vals, [0, 1]))
    return int(violations)


def count_logit_ceiling_contacts(individual: Individual, config: ESConfig) -> int:
    """
    Count how many binary logits are at or above max_logit_binary.
    """
    logits = individual.logit_binary
    if len(logits) == 0:
        return 0
    contacts = np.sum(logits >= config.max_logit_binary - 1e-7)
    return int(contacts)


def compute_flip_rate(old_binary: np.ndarray, new_binary: np.ndarray) -> float:
    """
    Compute fraction of bits that flipped between two generations.
    """
    if len(old_binary) == 0:
        return 0.0
    flips = np.sum(old_binary != new_binary)
    return float(flips / len(old_binary))


# 4. Categorical Variable Monitor

def check_gauge_residual(individual: Individual, config: ESConfig) -> np.ndarray:
    """
    Compute sum of logits per categorical variable slice.
    Must be close to 0 (< 1e-12) if zero-mean centering is applied.
    """
    logits = individual.logit_cat
    if len(logits) == 0:
        return np.array([], dtype=np.float64)

    residuals = []
    offset = 0
    for k in config.categorical_k:
        k_int = int(k)
        slice_logits = logits[offset : offset + k_int]
        residuals.append(float(np.sum(slice_logits)))
        offset += k_int
    return np.array(residuals, dtype=np.float64)


def compute_softmax_entropy(individual: Individual, config: ESConfig) -> np.ndarray:
    """
    Compute normalized Shannon entropy of softmax probability distribution per categorical variable.

    H(p) = -sum p_c * log(p_c) / log(k)
    """
    logits = individual.logit_cat
    if len(logits) == 0:
        return np.array([], dtype=np.float64)

    entropies = []
    offset = 0
    for k in config.categorical_k:
        k_int = int(k)
        slice_logits = logits[offset : offset + k_int]
        # Numerically stable softmax
        exp_z = np.exp(slice_logits - np.max(slice_logits))
        probs = exp_z / np.sum(exp_z)
        # Shannon entropy
        h = -np.sum(probs * np.log(probs + 1e-15))
        h_norm = h / np.log(k_int)  # Normalized to [0, 1]
        entropies.append(float(h_norm))
        offset += k_int
    return np.array(entropies, dtype=np.float64)


def compute_logit_norm(individual: Individual, config: ESConfig) -> np.ndarray:
    """
    Compute L2 norm of logit vector per categorical variable.
    Detects gauge drift when zero-mean centering is ablated.
    """
    logits = individual.logit_cat
    if len(logits) == 0:
        return np.array([], dtype=np.float64)

    norms = []
    offset = 0
    for k in config.categorical_k:
        k_int = int(k)
        slice_logits = logits[offset : offset + k_int]
        norms.append(float(np.linalg.norm(slice_logits)))
        offset += k_int
    return np.array(norms, dtype=np.float64)


# 5. Fuzzy Variable Monitor

def check_fuzzy_monotonicity(individual: Individual, config: ESConfig) -> int:
    """
    Check strict monotonic ordering a <= b <= c for decoded fuzzy triplets.
    """
    pts = individual.fuzzy
    if len(pts) == 0:
        return 0

    violations = 0
    for i in range(config.n_fuzzy):
        m, d = pts[i * 2 : i * 2 + 2]
        a = max(0.0, m - d)
        b = m
        c = min(1.0, m + d)
        if not (a <= b <= c):
            violations += 1
    return violations


def check_fuzzy_unit_bounds(individual: Individual, config: ESConfig) -> int:
    """
    Check that fuzzy centers remain within [0, 1] and dispersions within [fuzzy_epsilon, d_max].
    """
    pts = individual.fuzzy
    if len(pts) == 0:
        return 0
    m = pts[0::2]
    d = pts[1::2]
    m_violations = np.sum((m < 0.0) | (m > 1.0))
    d_max = getattr(config, "d_max_fuzzy", 1.0)
    d_violations = np.sum((d < config.fuzzy_epsilon - 1e-9) | (d > d_max + 1e-7))
    return int(m_violations + d_violations)


def compute_support_width(individual: Individual, config: ESConfig) -> np.ndarray:
    """
    Compute total support width 2 * d for each fuzzy variable.
    """
    pts = individual.fuzzy
    if len(pts) == 0:
        return np.array([], dtype=np.float64)
    d = pts[1::2]
    return (2.0 * d).astype(np.float64)


def check_fuzzy_epsilon(individual: Individual, config: ESConfig) -> int:
    """
    Check minimum dispersion floor d >= fuzzy_epsilon.
    """
    pts = individual.fuzzy
    if len(pts) == 0:
        return 0

    eps = config.fuzzy_epsilon
    d = pts[1::2]
    violations = np.sum(d < (eps - 1e-9))
    return int(violations)


# 6. Circular Variable Monitor

def check_circular_range(individual: Individual, config: ESConfig) -> int:
    """
    Check that all circular values remain strictly within [0, k_i).
    """
    vals = individual.circular
    if len(vals) == 0:
        return 0

    violations = 0
    k_arr = config.circular_k
    for i in range(config.n_circular):
        k = k_arr[i] if len(k_arr) > i else 12
        if vals[i] < 0 or vals[i] >= k:
            violations += 1
    return violations


def compute_geodesic_distance(x: int | np.ndarray, target: int | np.ndarray, k: int | np.ndarray) -> np.ndarray:
    """
    Compute shortest modular geodesic distance on circle Z_k.
    """
    diff = np.abs(x - target) % k
    return np.minimum(diff, k - diff)


# 7. Aggregate Invariant Monitor

def check_all_invariants(individual: Individual, config: ESConfig) -> dict[str, int]:
    """
    Run all applicable invariant checks on an individual.
    Returns a dictionary of violation counts (all must be 0 for a valid execution).
    """
    violations: dict[str, int] = {}

    if config.n_real > 0:
        violations["real_bounds"] = check_real_bounds(individual, config)

    if config.n_count > 0:
        violations["count_bounds"] = check_count_bounds(individual, config)
        violations["count_integrality"] = check_count_integrality(individual)

    if config.n_binary > 0:
        violations["binary_validity"] = check_binary_validity(individual)

    if config.n_categorical > 0:
        res = check_gauge_residual(individual, config)
        violations["categorical_gauge"] = int(np.sum(np.abs(res) > 1e-10))

    if config.n_fuzzy > 0:
        violations["fuzzy_monotonicity"] = check_fuzzy_monotonicity(individual, config)
        violations["fuzzy_unit_bounds"] = check_fuzzy_unit_bounds(individual, config)
        violations["fuzzy_epsilon"] = check_fuzzy_epsilon(individual, config)

    if config.n_circular > 0:
        violations["circular_range"] = check_circular_range(individual, config)

    return violations
