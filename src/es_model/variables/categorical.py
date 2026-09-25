"""
categorical.py: Categorical variable with softmax logit distributions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
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
) -> NDArray[np.float64]:
    """
    Center each categorical variable's logit slice around 0 to eliminate gauge freedom in softmax.
    """
    if cat_logits.shape[0] == 0:
        return cat_logits.copy()
    centered = cat_logits.copy()
    offset = 0
    for k in categorical_k:
        k_int = int(k)
        slice_vals = centered[offset : offset + k_int]
        centered[offset : offset + k_int] = slice_vals - np.mean(slice_vals)
        offset += k_int
    return centered


def mutate_categorical(
    values: NDArray[np.int64],
    cat_logits: NDArray[np.float64],
    rng: np.random.Generator,
    categorical_k: NDArray[np.int64],
) -> NDArray[np.int64]:
    """
    Categorical mutation using per-variable softmax logit vectors.

    Each categorical variable has a logit vector of size C (its cardinality).
    The genotype consists of C independent real values that evolve via
    self-adaptive Gaussian perturbation. The phenotype (selected category)
    is obtained by applying softmax over the logit vector to produce a
    probability distribution, then sampling from it.

    This guarantees a valid probability distribution (sum = 1) and preserves
    evolutionary inertia through stochastic sampling.

    Parameters
    ----------
    values : ndarray of int64
        Current categorical indices (one per variable).
    cat_logits : ndarray of float64
        Flattened logit vectors, shape (sum(categorical_k),).
        Sliced per variable using categorical_k.
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

    mutated = values.copy()
    offset = 0
    for i, k in enumerate(categorical_k):
        k_int = int(k)
        logit_slice = cat_logits[offset : offset + k_int]
        probs = kernels.softmax(logit_slice)
        r = float(rng.random())
        acc = 0.0
        choice = k_int - 1
        for idx in range(k_int):
            acc += probs[idx]
            if r < acc:
                choice = idx
                break
        mutated[i] = choice
        offset += k_int

    return mutated


class CategoricalVariable(Variable):
    """
    Categorical variable with per-variable softmax logit distributions.

    Each categorical variable has cardinality k_i.
    The strategy vector stores concatenated logits of length sum(k_i).
    Mutation evaluates softmax over each variable's logit slice and samples a category.
    """

    def __init__(
        self,
        values: NDArray[np.int64],
        strategy: NDArray[np.float64],
        categorical_k: NDArray[np.int64],
    ) -> None:
        super().__init__(
            name="categorical",
            values=values,
            strategy=strategy,
            strategy_recomb="arithmetic",
            adaptation_type="logit",
        )
        self.categorical_k: NDArray[np.int64] = np.asarray(categorical_k, dtype=np.int64)

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> CategoricalVariable:
        """Create a randomly initialized CategoricalVariable."""
        values = (
            np.array([rng.integers(0, int(k)) for k in config.categorical_k], dtype=np.int64)
            if config.n_categorical > 0
            else np.zeros(0, dtype=np.int64)
        )
        strategy = np.full(config.total_cat_logits, config.init_logit_cat, dtype=np.float64)
        return cls(
            values=values,
            strategy=strategy,
            categorical_k=config.categorical_k,
        )

    def copy(self) -> CategoricalVariable:
        """Create an independent deep copy of this CategoricalVariable."""
        return CategoricalVariable(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            categorical_k=self.categorical_k.copy(),
        )

    def _post_adapt_strategy(
        self,
        rng: np.random.Generator,
        tau_prime: float,
        tau: float,
    ) -> None:
        """Center logits around zero to eliminate gauge freedom in softmax."""
        self.strategy = center_categorical_logits(self.strategy, self.categorical_k)

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply softmax sampling per categorical variable slice."""
        self.values = mutate_categorical(
            self.values,
            self.strategy,
            rng,
            self.categorical_k,
        )
