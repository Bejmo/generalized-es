"""
categorical.py — Categorical variable block with softmax logit vectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .. import kernels
from .base import VariableBlock

if TYPE_CHECKING:
    from ..config import ESConfig


class CategoricalBlock(VariableBlock):
    """
    Categorical variable block with per-variable softmax logit distributions.

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
        super().__init__(name="categorical", values=values, strategy=strategy, strategy_recomb="arithmetic")
        self.categorical_k: NDArray[np.int64] = np.asarray(categorical_k, dtype=np.int64)

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> CategoricalBlock:
        """Create a randomly initialized CategoricalBlock."""
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

    def copy(self) -> CategoricalBlock:
        """Create an independent deep copy of this CategoricalBlock."""
        return CategoricalBlock(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            categorical_k=self.categorical_k.copy(),
        )

    def adapt_strategy(
        self,
        rng: np.random.Generator,
        tau_prime: float,
        tau: float,
    ) -> None:
        """Adapt categorical logits using additive Gaussian noise and zero-mean centering."""
        adapted = kernels.adapt_logits(
            self.strategy,
            rng,
            tau,
        )
        self.strategy = kernels.center_categorical_logits(adapted, self.categorical_k)

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply softmax sampling per categorical variable slice."""
        self.values = kernels.mutate_categorical(
            self.values,
            self.strategy,
            rng,
            self.categorical_k,
        )
