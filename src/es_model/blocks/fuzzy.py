"""
fuzzy.py — Fuzzy triangular variable block with monotonic structural constraints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .. import kernels
from .base import VariableBlock

if TYPE_CHECKING:
    from ..config import ESConfig


class FuzzyBlock(VariableBlock):
    """
    Fuzzy triangular variable block parameterized by triplets (a, b, c).

    Each fuzzy variable defines a membership function on [0, 1] satisfying a <= b <= c.
    The values array has shape (n_fuzzy * 3,).
    Mutation applies Gaussian perturbation, unit interval reflection, and structural push.
    Strategy parameters represent standard deviations with log-normal adaptation and a ceiling.
    """

    def __init__(
        self,
        values: NDArray[np.float64],
        strategy: NDArray[np.float64],
        n_fuzzy: int,
        fuzzy_epsilon: float = 1e-4,
        sigma_ceiling: float = 0.5,
        sigma_floor: float = 1e-6,
    ) -> None:
        super().__init__(name="fuzzy", values=values, strategy=strategy)
        self.n_fuzzy: int = n_fuzzy
        self.fuzzy_epsilon: float = fuzzy_epsilon
        self.sigma_ceiling: float = sigma_ceiling
        self.sigma_floor: float = sigma_floor

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> FuzzyBlock:
        """Create a randomly initialized FuzzyBlock with valid monotonic triplets."""
        n_fuzzy = config.n_fuzzy
        if n_fuzzy == 0:
            values = np.zeros(0, dtype=np.float64)
        else:
            points = np.empty(n_fuzzy * 3, dtype=np.float64)
            for i in range(n_fuzzy):
                points[i * 3 : i * 3 + 3] = np.sort(rng.random(3))
            values = points
        strategy = np.full(n_fuzzy * 3, config.init_sigma_fuzzy, dtype=np.float64)
        return cls(
            values=values,
            strategy=strategy,
            n_fuzzy=n_fuzzy,
            fuzzy_epsilon=config.fuzzy_epsilon,
            sigma_ceiling=config.sigma_ceiling_fuzzy,
            sigma_floor=config.sigma_floor,
        )

    @property
    def num_fuzzy_vars(self) -> int:
        """Number of distinct fuzzy variables (triplets)."""
        return self.n_fuzzy

    def copy(self) -> FuzzyBlock:
        """Create an independent deep copy of this FuzzyBlock."""
        return FuzzyBlock(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            n_fuzzy=self.n_fuzzy,
            fuzzy_epsilon=self.fuzzy_epsilon,
            sigma_ceiling=self.sigma_ceiling,
            sigma_floor=self.sigma_floor,
        )

    def adapt_strategy(
        self,
        rng: np.random.Generator,
        tau_prime: float,
        tau: float,
    ) -> None:
        """Adapt sigmas using log-normal dynamics and clamp to stability ceiling."""
        self.strategy = kernels.adapt_scales(
            self.strategy,
            rng,
            tau_prime,
            tau,
            self.sigma_floor,
        )
        np.minimum(self.strategy, self.sigma_ceiling, out=self.strategy)

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply Gaussian perturbation, unit reflection, and monotonic enforcement."""
        self.values = kernels.mutate_fuzzy(
            self.values,
            self.strategy,
            rng,
            self.n_fuzzy,
            self.fuzzy_epsilon,
        )
