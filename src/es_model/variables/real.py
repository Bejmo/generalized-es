"""
real.py: Continuous real-valued variable with log-normal self-adaptation.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .variable import Variable, resolve_bounds

if TYPE_CHECKING:
    from ..config import ESConfig


# -----------------------------------------------------------------------------
# Module-Level Mathematical Mutation Functions
# -----------------------------------------------------------------------------

def _reflect_bounds(
    x: NDArray[np.float64],
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Iteratively reflect values into [lower, upper].

    Mirroring algorithm: values below lower become 2*lower - x,
    values above upper become 2*upper - x, repeated until all are in bounds.
    """
    for _ in range(10):  # Safety cap on iterations.
        below = x < lower
        above = x > upper
        if not (below.any() or above.any()):
            break
        x = np.where(below, 2.0 * lower - x, x)
        x = np.where(above, 2.0 * upper - x, x)
    return np.clip(x, lower, upper)  # Final safety clamp.


def mutate_real(
    values: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    rng: np.random.Generator,
    real_bounds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Gaussian mutation for the real-valued variable vector.

    x' = x + N(0, sigma^2 * scale^2), simplified as x' = x + sigma * N(0,1)
    since the scale factor is absorbed into sigma through adaptation.
    After perturbation, values are reflected into domain bounds.

    Parameters
    ----------
    values : ndarray of float64
        Current real genes.
    sigmas : ndarray of float64
        Adapted step-size vector for this variable.
    rng : numpy Generator
    real_bounds : ndarray of shape (n_real, 2)
        Bound constraints [lower, upper] per dimension.

    Returns
    -------
    ndarray of float64
        Mutated and bounded values via reflection.
    """
    if values.shape[0] == 0:
        return values.copy()
    perturbation: NDArray[np.float64] = rng.standard_normal(values.shape[0]) * sigmas
    mutated = values + perturbation

    # Reflect to domain bounds (prevents boundary accumulation).
    mutated = _reflect_bounds(mutated, real_bounds[:, 0], real_bounds[:, 1])
    return mutated


class RealVariable(Variable):
    """
    Continuous real-valued variable.

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
        strategy_recomb: str = "geometric",
    ) -> None:
        resolved_bounds: NDArray[np.float64] = resolve_bounds(bounds, int(values.shape[0]), dtype=np.float64)
        if sigma_ceiling is not None:
            ceiling = np.asarray(sigma_ceiling, dtype=np.float64)
        elif int(values.shape[0]) > 0:
            ceiling = 0.3 * (resolved_bounds[:, 1] - resolved_bounds[:, 0])
        else:
            ceiling = None

        super().__init__(
            name="real",
            values=values,
            strategy=strategy,
            strategy_recomb=strategy_recomb,
            adaptation_type="lognormal",
            sigma_floor=sigma_floor,
            strategy_ceiling=ceiling,
        )
        self.bounds: NDArray[np.float64] = resolved_bounds
        self.sigma_ceiling = ceiling

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> RealVariable:
        """Create a randomly initialized RealVariable within configured bounds."""
        real_b = config.real_bounds
        values = (
            rng.uniform(real_b[:, 0], real_b[:, 1]).astype(np.float64)
            if config.n_real > 0
            else np.zeros(0, dtype=np.float64)
        )
        strategy = (
            (0.3 * (real_b[:, 1] - real_b[:, 0])).astype(np.float64)
            if config.n_real > 0
            else np.zeros(0, dtype=np.float64)
        )
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
            strategy_recomb=getattr(config, "strategy_recomb_scale", "geometric"),
        )

    def copy(self) -> RealVariable:
        """Create an independent deep copy of this RealVariable."""
        return RealVariable(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            bounds=self.bounds.copy(),
            sigma_floor=self.sigma_floor,
            sigma_ceiling=self.sigma_ceiling.copy() if self.sigma_ceiling is not None else None,
            strategy_recomb=self.strategy_recomb,
        )

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply Gaussian perturbation and boundary reflection."""
        self.values = mutate_real(
            self.values,
            self.strategy,
            rng,
            self.bounds,
        )

    def distance_squared(self, other: Variable | NDArray[Any]) -> float:
        """Compute Euclidean squared distance between real variables."""
        other_vals = other.values if isinstance(other, Variable) else np.asarray(other, dtype=np.float64)
        if self.dim == 0 or other_vals.shape[0] == 0:
            return 0.0
        return float(np.sum((self.values - other_vals) ** 2))
