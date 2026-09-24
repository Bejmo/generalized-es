"""
binary.py: Binary variable with logit-normal self-adaptation.
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
    probs: NDArray[np.float64] = kernels.sigmoid(logits)
    flip: NDArray[np.bool_] = rng.random(values.shape[0]) < probs

    # XOR to flip selected bits.
    mutated: NDArray[np.uint8] = values ^ flip.astype(np.uint8)
    return mutated


class BinaryVariable(Variable):
    """
    Binary variable representing boolean decision variables {0, 1}.

    Mutation uses independent bit-flips governed by sigmoid-mapped probabilities.
    Strategy parameters represent unconstrained logits evolving via additive Gaussian noise.
    """

    def __init__(
        self,
        values: NDArray[np.uint8],
        strategy: NDArray[np.float64],
        max_logit_binary: float = 0.0,
    ) -> None:
        super().__init__(
            name="binary",
            values=values,
            strategy=strategy,
            strategy_recomb="arithmetic",
            adaptation_type="logit",
            max_logit=max_logit_binary,
        )
        self.max_logit_binary: float = max_logit_binary

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> BinaryVariable:
        """Create a randomly initialized BinaryVariable."""
        values = (
            rng.integers(0, 2, size=config.n_binary, dtype=np.uint8)
            if config.n_binary > 0
            else np.zeros(0, dtype=np.uint8)
        )
        strategy = np.full(config.n_binary, config.init_logit_binary, dtype=np.float64)
        max_logit = getattr(config, "max_logit_binary", 0.0)
        return cls(
            values=values,
            strategy=strategy,
            max_logit_binary=max_logit,
        )

    def copy(self) -> BinaryVariable:
        """Create an independent deep copy of this BinaryVariable."""
        return BinaryVariable(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            max_logit_binary=self.max_logit_binary,
        )

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply bit-flip mutation according to sigmoid-mapped probabilities."""
        self.values = mutate_binary(
            self.values,
            self.strategy,
            rng,
        )
