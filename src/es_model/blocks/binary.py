"""
binary.py — Binary variable block with logit-normal self-adaptation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .. import kernels
from .base import VariableBlock

if TYPE_CHECKING:
    from ..config import ESConfig


class BinaryBlock(VariableBlock):
    """
    Binary variable block representing boolean decision variables {0, 1}.

    Mutation uses independent bit-flips governed by sigmoid-mapped probabilities.
    Strategy parameters represent unconstrained logits evolving via additive Gaussian noise.
    """

    def __init__(
        self,
        values: NDArray[np.uint8],
        strategy: NDArray[np.float64],
    ) -> None:
        super().__init__(name="binary", values=values, strategy=strategy, strategy_recomb="arithmetic")

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> BinaryBlock:
        """Create a randomly initialized BinaryBlock."""
        values = (
            rng.integers(0, 2, size=config.n_binary, dtype=np.uint8)
            if config.n_binary > 0
            else np.zeros(0, dtype=np.uint8)
        )
        strategy = np.full(config.n_binary, config.init_logit_binary, dtype=np.float64)
        return cls(
            values=values,
            strategy=strategy,
        )

    def copy(self) -> BinaryBlock:
        """Create an independent deep copy of this BinaryBlock."""
        return BinaryBlock(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
        )

    def adapt_strategy(
        self,
        rng: np.random.Generator,
        tau_prime: float,
        tau: float,
    ) -> None:
        """Adapt flip logits using additive Gaussian noise."""
        self.strategy = kernels.adapt_logits(
            self.strategy,
            rng,
            tau,
        )

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply bit-flip mutation according to sigmoid-mapped probabilities."""
        self.values = kernels.mutate_binary(
            self.values,
            self.strategy,
            rng,
        )
