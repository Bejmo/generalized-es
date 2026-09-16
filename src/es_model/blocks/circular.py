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
    Mutation follows a modular (wrapped) Skellam distribution (difference of two Poissons modulo k).
    Strategy parameters represent Poisson rates (lambdas) with log-normal self-adaptation.
    """

    def __init__(
        self,
        values: NDArray[np.int64],
        strategy: NDArray[np.float64],
        circular_k: NDArray[np.int64],
        sigma_floor: float = 1e-6,
    ) -> None:
        super().__init__(name="circular", values=values, strategy=strategy)
        self.circular_k: NDArray[np.int64] = np.asarray(circular_k, dtype=np.int64)
        self.sigma_floor: float = sigma_floor

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> CircularBlock:
        """Create a randomly initialized CircularBlock."""
        values = (
            np.array([rng.integers(0, int(k)) for k in config.circular_k], dtype=np.int64)
            if config.n_circular > 0
            else np.zeros(0, dtype=np.int64)
        )
        init_lambda = getattr(config, "init_lambda_circular", 1.0)
        strategy = np.full(config.n_circular, init_lambda, dtype=np.float64)
        return cls(
            values=values,
            strategy=strategy,
            circular_k=config.circular_k,
            sigma_floor=config.sigma_floor,
        )

    def copy(self) -> CircularBlock:
        """Create an independent deep copy of this CircularBlock."""
        return CircularBlock(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            circular_k=self.circular_k.copy(),
            sigma_floor=self.sigma_floor,
        )

    def adapt_strategy(
        self,
        rng: np.random.Generator,
        tau_prime: float,
        tau: float,
    ) -> None:
        """Adapt circular Poisson rates using diagonal log-normal adaptation."""
        self.strategy = kernels.apply_lognormal_selfadaptation(
            self.strategy,
            rng,
            tau_prime,
            tau,
            self.sigma_floor,
        )

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply modular Skellam discrete perturbation."""
        self.values = kernels.mutate_circular(
            self.values,
            self.strategy,
            rng,
            self.circular_k,
        )
