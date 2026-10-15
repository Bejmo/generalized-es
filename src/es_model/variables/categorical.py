"""
categorical.py: Categorical variable with self-adaptive mutation rates.

Each categorical variable stores its current discrete integer state c_i in {0, ..., K_i - 1}.
The co-evolving strategy parameter is a single real-valued logit z_i in [-4.6, 0.0] representing
the mutation rate p_mut = sigmoid(z_i) in [0.01, 0.5].
When mutation occurs, the variable jumps uniformly to one of the remaining (K_i - 1) categories:
c_i' = (c_i + delta) mod K_i, with delta in {1, ..., K_i - 1}.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
import numpy as np
from numpy.typing import NDArray

from .. import kernels
from .variable import Variable

if TYPE_CHECKING:
    from ..config import ESConfig


# -----------------------------------------------------------------------------
# Module-Level Mathematical Mutation Functions
# -----------------------------------------------------------------------------

def center_categorical_logits(
    cat_logits: NDArray[np.float64],
    categorical_k: NDArray[np.int64],
    max_logit: float | None = None,
) -> NDArray[np.float64]:
    """
    Backwards-compatible passthrough utility.
    Under the unified 1-logit-per-variable model, logits are clamped to [-4.6, 0.0] directly.
    """
    if max_logit is not None:
        return np.clip(cat_logits, -4.6, max_logit)
    return cat_logits.copy()


def mutate_categorical(
    values: NDArray[np.int64],
    logits: NDArray[np.float64],
    rng: np.random.Generator,
    categorical_k: NDArray[np.int64],
) -> NDArray[np.int64]:
    """
    Categorical jump mutation governed by sigmoid-mapped probabilities.

    Each variable mutates with probability p = sigmoid(logit) in [0.01, 0.5].
    When mutating, it transitions uniformly to any of the remaining (K_i - 1) states:
        c_i' = (c_i + delta) mod K_i,  delta in {1, ..., K_i - 1}

    Parameters
    ----------
    values : ndarray of int64
        Current categorical state indices (one per variable).
    logits : ndarray of float64
        Adapted logit parameters for mutation probability (shape: (n_categorical,)).
    rng : numpy Generator
    categorical_k : ndarray of int64
        Number of categories per categorical variable.

    Returns
    -------
    ndarray of int64
        Mutated categorical indices.
    """
    n = values.shape[0]
    if n == 0:
        return values.copy()

    probs: NDArray[np.float64] = kernels.sigmoid(logits)
    mut_mask: NDArray[np.bool_] = rng.random(n) < probs
    mutated: NDArray[np.int64] = values.copy()

    for i in range(n):
        if mut_mask[i]:
            k = int(categorical_k[i])
            if k > 1:
                offset = int(rng.integers(1, k))
                mutated[i] = (mutated[i] + offset) % k

    return mutated


class CategoricalVariable(Variable):
    """
    Categorical decision variable representing discrete unordered choices in {0, ..., K_i - 1}.

    Mutation uses independent categorical transitions governed by sigmoid-mapped probabilities.
    Strategy parameters represent unconstrained logits evolving via additive Gaussian noise,
    clamped to [-4.6, 0.0] to prevent freezing and hypermutation.
    """

    def __init__(
        self,
        values: NDArray[np.int64],
        strategy: NDArray[np.float64],
        categorical_k: NDArray[np.int64],
        max_logit_categorical: float = 0.0,
        min_logit_categorical: float = -4.6,
    ) -> None:
        super().__init__(
            name="categorical",
            values=values,
            strategy=strategy,
            strategy_recomb="arithmetic",
            adaptation_type="logit",
            max_logit=max_logit_categorical,
            min_logit=min_logit_categorical,
        )
        self.categorical_k: NDArray[np.int64] = np.asarray(categorical_k, dtype=np.int64)
        self.max_logit_categorical: float = max_logit_categorical
        self.min_logit_categorical: float = min_logit_categorical

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> CategoricalVariable:
        """Create a randomly initialized CategoricalVariable."""
        values = (
            np.array([rng.integers(0, int(k)) for k in config.categorical_k], dtype=np.int64)
            if config.n_categorical > 0
            else np.zeros(0, dtype=np.int64)
        )
        strategy = np.full(config.n_categorical, config.init_logit_cat, dtype=np.float64)
        max_logit = getattr(config, "max_logit_categorical", 0.0)
        min_logit = getattr(config, "min_logit_categorical", -4.6)
        return cls(
            values=values,
            strategy=strategy,
            categorical_k=config.categorical_k,
            max_logit_categorical=max_logit,
            min_logit_categorical=min_logit,
        )

    def copy(self) -> CategoricalVariable:
        """Create an independent deep copy of this CategoricalVariable."""
        return CategoricalVariable(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            categorical_k=self.categorical_k.copy(),
            max_logit_categorical=self.max_logit_categorical,
            min_logit_categorical=self.min_logit_categorical,
        )

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply jump mutation according to sigmoid-mapped probabilities."""
        self.values = mutate_categorical(
            self.values,
            self.strategy,
            rng,
            self.categorical_k,
        )

    def distance_squared(self, other: Variable | NDArray[Any]) -> float:
        """Compute 0-1 misclassification count across categories."""
        other_vals = other.values if isinstance(other, Variable) else np.asarray(other, dtype=np.int64)
        if self.dim == 0 or other_vals.shape[0] == 0:
            return 0.0
        return float(np.sum(self.values != other_vals))
