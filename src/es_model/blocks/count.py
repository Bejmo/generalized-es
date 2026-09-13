"""
count.py — Discrete integer count variable block with Skellam mutation.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .. import kernels
from .base import VariableBlock, resolve_bounds

if TYPE_CHECKING:
    from ..config import ESConfig


class CountBlock(VariableBlock):
    """
    Discrete integer count variable block.

    Mutation follows a Skellam distribution (difference of two independent Poisson variates).
    Strategy parameters represent Poisson rate parameters (lambdas) with log-normal adaptation.
    Bounds are strictly integers (np.int64).
    """

    def __init__(
        self,
        values: NDArray[np.int64],
        strategy: NDArray[np.float64],
        bounds: Any,
        sigma_floor: float = 1e-6,
    ) -> None:
        super().__init__(name="count", values=values, strategy=strategy)
        self.bounds: NDArray[np.int64] = resolve_bounds(bounds, self.dim, dtype=np.int64)
        self.sigma_floor: float = sigma_floor

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> CountBlock:
        """Create a randomly initialized CountBlock within configured bounds."""
        count_b = config.count_bounds
        values = (
            rng.integers(count_b[:, 0], count_b[:, 1] + 1, dtype=np.int64)
            if config.n_count > 0
            else np.zeros(0, dtype=np.int64)
        )
        strategy = np.full(config.n_count, config.init_lambda_count, dtype=np.float64)
        return cls(
            values=values,
            strategy=strategy,
            bounds=count_b,
            sigma_floor=config.sigma_floor,
        )

    def copy(self) -> CountBlock:
        """Create an independent deep copy of this CountBlock."""
        return CountBlock(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            bounds=self.bounds.copy(),
            sigma_floor=self.sigma_floor,
        )

    def adapt_strategy(
        self,
        rng: np.random.Generator,
        tau_prime: float,
        tau: float,
    ) -> None:
        """Adapt Poisson rates using diagonal log-normal adaptation."""
        self.strategy = kernels.adapt_scales(
            self.strategy,
            rng,
            tau_prime,
            tau,
            self.sigma_floor,
        )

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply Skellam discrete perturbation and integer domain clamping."""
        self.values = kernels.mutate_count(
            self.values,
            self.strategy,
            rng,
            self.bounds,
        )
