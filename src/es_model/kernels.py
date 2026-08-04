"""
kernels.py — Pure mathematical mutation and self-adaptation kernels (vectorized).

This module contains stateless functions that operate on NumPy arrays representing
genome blocks and their associated strategy parameters. All heavy numerical paths
are fully vectorized; structural blocks (categorical, fuzzy) use minimal loops.
"""

import numpy as np
from numpy.typing import NDArray


# -----------------------------------------------------------------------------
# 1. Global Learning-Rate Computation
# -----------------------------------------------------------------------------

def compute_learning_rates(n_total: int) -> tuple[float, float]:
    """
    Compute the global and local learning rates for log-normal self-adaptation.

    tau'  = 1 / sqrt(2 * n_total)       — global component (shared noise)
    tau   = 1 / sqrt(2 * sqrt(n_total)) — local component (per-dimension noise)

    Parameters
    ----------
    n_total : int
        Total number of active genome dimensions across all blocks.

    Returns
    -------
    (tau_prime, tau) : tuple of float
    """
    if n_total == 0:
        return 0.0, 0.0
    tau_prime: float = 1.0 / np.sqrt(2.0 * n_total)
    tau: float = 1.0 / np.sqrt(2.0 * np.sqrt(n_total))
    return tau_prime, tau


# -----------------------------------------------------------------------------
# 2. Log-Normal Adaptation for Scale-Like Strategy Parameters
# -----------------------------------------------------------------------------

def adapt_scales(
    sigmas: NDArray[np.float64],
    rng: np.random.Generator,
    tau_prime: float,
    tau: float,
    sigma_floor: float = 1e-6,
) -> NDArray[np.float64]:
    """
    Apply diagonal log-normal self-adaptation to scale-like parameters.

    Formula per dimension i:
        sigma_i' = sigma_i * exp(tau' * N(0,1) + tau * N_i(0,1))

    After adaptation, values are clamped to sigma_floor to prevent freezing.

    Parameters
    ----------
    sigmas : ndarray of float64
        Current strategy parameter vector (positive).
    rng : numpy Generator
        Random number generator instance.
    tau_prime : float
        Global adaptation learning rate.
    tau : float
        Local (per-dimension) adaptation learning rate.
    sigma_floor : float
        Minimum floor to prevent dimensional freezing.

    Returns
    -------
    ndarray of float64
        Updated strategy vector.
    """
    n = sigmas.shape[0]
    if n == 0:
        return sigmas.copy()
    # Global noise term shared across all dimensions of this individual.
    global_noise: float = rng.standard_normal() * tau_prime
    # Per-dimension local noise.
    local_noise: NDArray[np.float64] = rng.standard_normal(n) * tau

    # Compute multiplicative update factor.
    factor: NDArray[np.float64] = np.exp(global_noise + local_noise)
    updated = sigmas * factor

    # Enforce stability floor — prevent complete dimensional convergence.
    np.maximum(updated, sigma_floor, out=updated)
    return updated


# -----------------------------------------------------------------------------
# 3. Logit-Normal Adaptation for Probability-Like Strategy Parameters
# -----------------------------------------------------------------------------

def adapt_logits(
    logits: NDArray[np.float64],
    rng: np.random.Generator,
    tau: float,
) -> NDArray[np.float64]:
    """
    Apply additive Gaussian noise to logit-space strategy parameters.

    The logits are NOT bounded; the mapping to probability happens via sigmoid
    at mutation time. This ensures probabilities remain strictly in (0, 1).

    Parameters
    ----------
    logits : ndarray of float64
        Current logit values (real line).
    rng : numpy Generator
        Random number generator instance.
    tau : float
        Local learning rate.

    Returns
    -------
    ndarray of float64
        Updated logit values.
    """
    n = logits.shape[0]
    if n == 0:
        return logits.copy()
    noise: NDArray[np.float64] = rng.standard_normal(n) * tau
    return logits + noise


def sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map logits to probabilities in (0, 1) via the sigmoid function."""
    return 1.0 / (1.0 + np.exp(-logits))


# -----------------------------------------------------------------------------
# 4. Vectorized Mutation Kernels for Numerical Blocks
# -----------------------------------------------------------------------------

def _reflect_bounds(
    x: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Iteratively reflect values into [lower, upper].

    Mirroring algorithm: values below lower become 2*lower - x,
    values above upper become 2*upper - x, repeated until all are in bounds.
    """
    for _ in range(10):  # Safety cap on iterations.
        below = x < lower
        above = x > upper
        if not (below.any() or above.any()):
            break
        x = np.where(below, 2.0 * lower - x, x)
        x = np.where(above, 2.0 * upper - x, x)
    return np.clip(x, lower, upper)  # Final safety clamp.


def mutate_real(
    values: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    rng: np.random.Generator,
    real_bounds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Gaussian mutation for the real-valued block.

    x' = x + N(0, sigma^2 * scale^2)  →  simplified as x' = x + sigma * N(0,1)
    since the scale factor is absorbed into sigma through adaptation.

    After perturbation, values are reflected into domain bounds.

    Parameters
    ----------
    values : ndarray of float64
        Current real genes.
    sigmas : ndarray of float64
        Adapted step-size vector for this block.
    rng : numpy Generator
    real_bounds : ndarray of shape (n_real, 2)
        Bound constraints [lower, upper] per dimension.

    Returns
    -------
    ndarray of float64
        Mutated and bounded values via reflection.
    """
    if values.shape[0] == 0:
        return values.copy()
    perturbation: NDArray[np.float64] = rng.standard_normal(values.shape[0]) * sigmas
    mutated = values + perturbation

    # Reflect to domain bounds (prevents boundary accumulation).
    mutated = _reflect_bounds(mutated, real_bounds[:, 0], real_bounds[:, 1])
    return mutated


def mutate_count(
    values: NDArray[np.int64],
    lambdas: NDArray[np.float64],
    rng: np.random.Generator,
    count_bounds: NDArray[np.int64],
) -> NDArray[np.int64]:
    """
    Skellam mutation for the integer count block.

    Delta = Poisson(lambda) - Poisson(lambda)  (difference of two independent Poissons).
    Result is clipped to configured integer bounds.

    Parameters
    ----------
    values : ndarray of int64
        Current integer genes.
    lambdas : ndarray of float64
        Adapted Poisson rate vector.
    rng : numpy Generator
    count_bounds : ndarray of shape (n_count, 2)

    Returns
    -------
    ndarray of int64
        Mutated and bounded integer values.
    """
    if values.shape[0] == 0:
        return values.copy()
    pos: NDArray[np.int64] = rng.poisson(lambdas).astype(np.int64)
    neg: NDArray[np.int64] = rng.poisson(lambdas).astype(np.int64)
    delta: NDArray[np.int64] = pos - neg
    mutated = values + delta

    # Clip to domain bounds.
    np.clip(mutated, count_bounds[:, 0], count_bounds[:, 1], out=mutated)
    return mutated


def mutate_binary(
    values: NDArray[np.uint8],
    logits: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    """
    Independent bit-flip mutation parameterized by per-dimension probabilities.

    Each bit flips with probability p = sigmoid(logit).

    Parameters
    ----------
    values : ndarray of uint8
        Current binary genes (0 or 1).
    logits : ndarray of float64
        Adapted logit parameters for flip probability.
    rng : numpy Generator

    Returns
    -------
    ndarray of uint8
        Mutated binary vector.
    """
    if values.shape[0] == 0:
        return values.copy()
    probs: NDArray[np.float64] = sigmoid(logits)
    flip: NDArray[np.bool_] = rng.random(values.shape[0]) < probs

    # XOR to flip selected bits.
    mutated: NDArray[np.uint8] = values ^ flip.astype(np.uint8)
    return mutated


# -----------------------------------------------------------------------------
# 5. Categorical Re-Sampling (Excluding Current Index)
# -----------------------------------------------------------------------------

def mutate_categorical(
    values: NDArray[np.int64],
    logits: NDArray[np.float64],
    rng: np.random.Generator,
    categorical_k: NDArray[np.int64],
) -> NDArray[np.int64]:
    """
    Categorical mutation: Bernoulli trial triggers uniform re-sample.

    For each variable, if the trial fires (prob from logit), the gene is
    re-sampled uniformly from all categories EXCEPT the current value.

    Parameters
    ----------
    values : ndarray of int64
        Current categorical indices.
    logits : ndarray of float64
        Adapted logit parameters for mutation probability.
    rng : numpy Generator
    categorical_k : ndarray of int64
        Number of categories per categorical variable.

    Returns
    -------
    ndarray of int64
        Mutated categorical indices.
    """
    if values.shape[0] == 0:
        return values.copy()
    probs: NDArray[np.float64] = sigmoid(logits)
    trigger: NDArray[np.bool_] = rng.random(values.shape[0]) < probs

    mutated = values.copy()
    for i in np.where(trigger)[0]:
        k = int(categorical_k[i])
        current = int(values[i])
        # Uniform sample from alternatives (exclude current).
        alternatives = np.delete(np.arange(k, dtype=np.int64), current)
        mutated[i] = rng.choice(alternatives)

    return mutated


# -----------------------------------------------------------------------------
# 6. Fuzzy Perturbation with Reflection and Monotonic Enforcement
# -----------------------------------------------------------------------------

def _reflect_unit(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Iteratively reflect values into [0, 1].

    Mirroring algorithm: values below 0 become |x|, values above 1 become 2-x,
    repeated until all are in [0, 1]. In practice ≤3 iterations suffice.
    """
    for _ in range(10):  # Safety cap on iterations.
        below = x < 0.0
        above = x > 1.0
        if not (below.any() or above.any()):
            break
        x = np.where(below, -x, x)
        x = np.where(above, 2.0 - x, x)
    return np.clip(x, 0.0, 1.0)  # Final safety clamp.


def _enforce_monotonic(a: float, b: float, c: float, fuzzy_epsilon: float = 1e-4) -> tuple[float, float, float]:
    """
    Guarantee a <= b <= c with epsilon margin.

    Uses a non-collapsing structural push so triangles do not flatten.
    """
    eps = fuzzy_epsilon
    # Sort to start from a feasible ordering.
    a, b, c = float(np.minimum(a, b)), float(np.median([a, b, c])), float(np.maximum(b, c))
    # Push b away from a and c away from b if too close.
    if b < a + eps:
        b = a + eps
    if c < b + eps:
        c = b + eps
    # Final reflection back into [0,1] if pushes exceeded bounds.
    a = float(np.clip(a, 0.0, 1.0 - 2 * eps))
    b = float(np.clip(b, a + eps, 1.0 - eps))
    c = float(np.clip(c, b + eps, 1.0))
    return a, b, c


def mutate_fuzzy(
    points: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    rng: np.random.Generator,
    n_fuzzy: int,
    fuzzy_epsilon: float = 1e-4,
) -> NDArray[np.float64]:
    """
    Gaussian perturbation on fuzzy control points with structural enforcement.

    Each fuzzy variable has 3 control points (a, b, c) stored contiguously:
    points shape = (n_fuzzy * 3,) and sigmas shape = (n_fuzzy * 3,).

    Steps:
    1. Perturb each point: x' = x + N(0, sigma^2)
    2. Reflect into [0, 1] using mirroring.
    3. Enforce monotonicity a <= b <= c with epsilon margin.

    Parameters
    ----------
    points : ndarray of float64
        Flattened control points [a0, b0, c0, a1, b1, c1, ...].
    sigmas : ndarray of float64
        Corresponding sigma vector.
    rng : numpy Generator
    n_fuzzy : int
        Number of fuzzy variables.
    fuzzy_epsilon : float
        Minimum epsilon margin.

    Returns
    -------
    ndarray of float64
        Mutated control points with structural guarantees.
    """
    if n_fuzzy == 0 or points.shape[0] == 0:
        return points.copy()
    mutated = points.copy()
    perturbation: NDArray[np.float64] = rng.standard_normal(points.shape[0]) * sigmas

    for i in range(n_fuzzy):
        idx = i * 3
        # Perturb.
        abc = points[idx:idx + 3] + perturbation[idx:idx + 3]
        # Reflect into [0, 1].
        abc = _reflect_unit(abc)
        # Enforce monotonic constraint.
        a, b, c = _enforce_monotonic(float(abc[0]), float(abc[1]), float(abc[2]), fuzzy_epsilon=fuzzy_epsilon)
        mutated[idx] = a
        mutated[idx + 1] = b
        mutated[idx + 2] = c

    return mutated
