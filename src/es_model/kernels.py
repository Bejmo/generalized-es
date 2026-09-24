"""
kernels.py: Pure mathematical primitives, activation functions, and self-adaptation kernels.

This module contains stateless, vectorized numerical primitives that operate on
NumPy arrays representing genome strategy parameters or probabilistic mappings.
Variable-specific mutation operators reside in their respective block modules.
"""

import numpy as np
from numpy.typing import NDArray


# -----------------------------------------------------------------------------
# 1. Log-Normal Adaptation for Scale-Like Strategy Parameters
# -----------------------------------------------------------------------------

def apply_lognormal_selfadaptation(
    sigmas: NDArray[np.float64],
    rng: np.random.Generator,
    tau_prime: float,
    tau: float,
    sigma_floor: float = 1e-6,
    sigma_ceiling: float | NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """
    Apply log-normal self-adaptation to scale-like strategy parameters.

    Formula per dimension i:
        sigma_i' = sigma_i * exp(tau' * N(0,1) + tau * N_i(0,1))

    After adaptation, values are clamped to [sigma_floor, sigma_ceiling] to prevent
    freezing and runaway divergence.

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
    sigma_ceiling : float or ndarray of float64, optional
        Maximum ceiling to prevent runaway dimensional divergence.

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

    # Enforce stability floor and optional ceiling.
    np.maximum(updated, sigma_floor, out=updated)
    if sigma_ceiling is not None:
        np.minimum(updated, sigma_ceiling, out=updated)
    return updated


# -----------------------------------------------------------------------------
# 3. Logit-Normal Adaptation for Probability-Like Strategy Parameters
# -----------------------------------------------------------------------------

def adapt_logits(
    logits: NDArray[np.float64],
    rng: np.random.Generator,
    tau: float,
    max_logit: float | None = None,
) -> NDArray[np.float64]:
    """
    Apply additive Gaussian noise to logit-space strategy parameters.

    The mapping to probability happens via sigmoid at mutation time.
    Optional max_logit upper bound prevents hypermutation (e.g., max_logit=0 gives p <= 0.5).

    Parameters
    ----------
    logits : ndarray of float64
        Current logit values (real line).
    rng : numpy Generator
        Random number generator instance.
    tau : float
        Local learning rate.
    max_logit : float, optional
        Upper bound on logits.

    Returns
    -------
    ndarray of float64
        Updated logit values.
    """
    n = logits.shape[0]
    if n == 0:
        return logits.copy()
    noise: NDArray[np.float64] = rng.standard_normal(n) * tau
    updated = logits + noise
    if max_logit is not None:
        np.minimum(updated, max_logit, out=updated)
    return updated


def sigmoid(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Map logits to probabilities in (0, 1) via the sigmoid function.

    Logits are temporarily clipped to [-700, 700] to prevent float64 overflow
    (exp(700) ~ 1e304), while the underlying genotype retains its full value.
    """
    clipped = np.clip(logits, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable softmax over a 1-D logit vector.

    Subtracts the max before exponentiating to avoid overflow.
    Returns a valid probability distribution (sums to 1).
    """
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / exps.sum()
