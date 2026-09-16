"""
real.py — Continuous real-valued variable block with log-normal self-adaptation.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .. import kernels
from .base import VariableBlock, resolve_bounds

if TYPE_CHECKING:
    from ..config import ESConfig


class RealBlock(VariableBlock):
    """
    Continuous real-valued variable block.

    Mutation applies Gaussian perturbation with boundary reflection.
    Strategy parameters represent standard deviations (sigmas) with log-normal adaptation.
    """

    def __init__(
        self,
        values: NDArray[np.float64],
        strategy: NDArray[np.float64],
        bounds: Any,
        sigma_floor: float = 1e-6,
        sigma_ceiling: float | NDArray[np.float64] | None = None,
    ) -> None:
        super().__init__(name="real", values=values, strategy=strategy, strategy_recomb="geometric")
        self.bounds: NDArray[np.float64] = resolve_bounds(bounds, self.dim, dtype=np.float64)
        self.sigma_floor: float = sigma_floor
        if sigma_ceiling is not None:
            self.sigma_ceiling = np.asarray(sigma_ceiling, dtype=np.float64)
        elif self.dim > 0:
            self.sigma_ceiling = 0.3 * (self.bounds[:, 1] - self.bounds[:, 0])
        else:
            self.sigma_ceiling = None

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> RealBlock:
        """Create a randomly initialized RealBlock within configured bounds."""
        real_b = config.real_bounds
        values = (
            rng.uniform(real_b[:, 0], real_b[:, 1]).astype(np.float64)
            if config.n_real > 0
            else np.zeros(0, dtype=np.float64)
        )
        strategy = np.full(config.n_real, config.init_sigma_real, dtype=np.float64)
        ceiling_factor = getattr(config, "sigma_ceiling_factor", 0.3)
        sigma_ceiling = (
            ceiling_factor * (real_b[:, 1] - real_b[:, 0])
            if config.n_real > 0
            else None
        )
        return cls(
            values=values,
            strategy=strategy,
            bounds=real_b,
            sigma_floor=config.sigma_floor,
            sigma_ceiling=sigma_ceiling,
        )

    def copy(self) -> RealBlock:
        """Create an independent deep copy of this RealBlock."""
        return RealBlock(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            bounds=self.bounds.copy(),
            sigma_floor=self.sigma_floor,
            sigma_ceiling=self.sigma_ceiling.copy() if self.sigma_ceiling is not None else None,
        )

    def adapt_strategy(
        self,
        rng: np.random.Generator,
        tau_prime: float,
        tau: float,
    ) -> None:
        """Adapt step-size vector using diagonal log-normal adaptation with domain ceiling."""
        self.strategy = kernels.adapt_scales(
            self.strategy,
            rng,
            tau_prime,
            tau,
            self.sigma_floor,
            self.sigma_ceiling,
        )

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply Gaussian perturbation and boundary reflection."""
        self.values = kernels.mutate_real(
            self.values,
            self.strategy,
            rng,
            self.bounds,
        )
