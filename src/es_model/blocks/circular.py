"""
circular.py — Closed-order circular variable block with modular step mutation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .. import kernels
from .base import VariableBlock

if TYPE_CHECKING:
    from ..config import ESConfig


class CircularBlock(VariableBlock):
    """
    Circular variable block representing closed-order discrete positions (e.g., days of week).

    Each circular variable has cardinality k_i (positions 0..k_i-1).
    Mutation steps forward (+1 mod k) with probability sigmoid(logit) or backward (-1 mod k).
    Strategy parameters represent forward-probability logits adapting via Gaussian noise.
    """

    def __init__(
        self,
        values: NDArray[np.int64],
        strategy: NDArray[np.float64],
        circular_k: NDArray[np.int64],
    ) -> None:
        super().__init__(name="circular", values=values, strategy=strategy)
        self.circular_k: NDArray[np.int64] = np.asarray(circular_k, dtype=np.int64)

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> CircularBlock:
        """Create a randomly initialized CircularBlock."""
        values = (
            np.array([rng.integers(0, int(k)) for k in config.circular_k], dtype=np.int64)
            if config.n_circular > 0
            else np.zeros(0, dtype=np.int64)
        )
        strategy = np.full(config.n_circular, config.init_logit_circular, dtype=np.float64)
        return cls(
            values=values,
            strategy=strategy,
            circular_k=config.circular_k,
        )

    def copy(self) -> CircularBlock:
        """Create an independent deep copy of this CircularBlock."""
        return CircularBlock(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            circular_k=self.circular_k.copy(),
        )

    def adapt_strategy(
        self,
        rng: np.random.Generator,
        tau_prime: float,
        tau: float,
    ) -> None:
        """Adapt forward-step logits using additive Gaussian noise."""
        self.strategy = kernels.adapt_logits(
            self.strategy,
            rng,
            tau,
        )

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply modular step (+1 or -1 mod k) governed by forward logit."""
        self.values = kernels.mutate_circular(
            self.values,
            self.strategy,
            rng,
            self.circular_k,
        )
