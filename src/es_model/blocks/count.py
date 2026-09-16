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
        lambda_ceiling: float | NDArray[np.float64] | None = None,
    ) -> None:
        super().__init__(name="count", values=values, strategy=strategy, strategy_recomb="geometric")
        self.bounds: NDArray[np.int64] = resolve_bounds(bounds, self.dim, dtype=np.int64)
        self.sigma_floor: float = sigma_floor
        if lambda_ceiling is not None:
            self.lambda_ceiling = np.asarray(lambda_ceiling, dtype=np.float64)
        elif self.dim > 0:
            self.lambda_ceiling = 0.3 * (self.bounds[:, 1] - self.bounds[:, 0])
        else:
            self.lambda_ceiling = None

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
        ceiling_factor = getattr(config, "sigma_ceiling_factor", 0.3)
        lambda_ceiling = (
            ceiling_factor * (count_b[:, 1] - count_b[:, 0])
            if config.n_count > 0
            else None
        )
        return cls(
            values=values,
            strategy=strategy,
            bounds=count_b,
            sigma_floor=config.sigma_floor,
            lambda_ceiling=lambda_ceiling,
        )

    def copy(self) -> CountBlock:
        """Create an independent deep copy of this CountBlock."""
        return CountBlock(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            bounds=self.bounds.copy(),
            sigma_floor=self.sigma_floor,
            lambda_ceiling=self.lambda_ceiling.copy() if self.lambda_ceiling is not None else None,
        )

    def adapt_strategy(
        self,
        rng: np.random.Generator,
        tau_prime: float,
        tau: float,
    ) -> None:
        """Adapt Poisson rates using diagonal log-normal adaptation with domain ceiling."""
        self.strategy = kernels.adapt_scales(
            self.strategy,
            rng,
            tau_prime,
            tau,
            self.sigma_floor,
            self.lambda_ceiling,
        )

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply Skellam discrete perturbation and integer domain clamping."""
        self.values = kernels.mutate_count(
            self.values,
            self.strategy,
            rng,
            self.bounds,
        )
