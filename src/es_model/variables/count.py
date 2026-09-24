"""
count.py: Discrete integer count variable with Skellam mutation.
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

def mutate_count(
    values: NDArray[np.int64],
    lambdas: NDArray[np.float64],
    rng: np.random.Generator,
    count_bounds: NDArray[np.int64],
) -> NDArray[np.int64]:
    """
    Skellam mutation for the integer count variable vector.

    Delta = Poisson(lambda) - Poisson(lambda) (difference of two independent Poissons).
    Result is clipped to configured integer bounds.

    Parameters
    ----------
    values : ndarray of int64
        Current integer genes.
    lambdas : ndarray of float64
        Adapted Poisson rate vector.
    rng : numpy Generator
    count_bounds : ndarray of shape (n_count, 2)

    Returns
    -------
    ndarray of int64
        Mutated and bounded integer values.
    """
    if values.shape[0] == 0:
        return values.copy()
    pos: NDArray[np.int64] = rng.poisson(lambdas).astype(np.int64)
    neg: NDArray[np.int64] = rng.poisson(lambdas).astype(np.int64)
    delta: NDArray[np.int64] = pos - neg
    mutated = values + delta

    # Clip to domain bounds.
    np.clip(mutated, count_bounds[:, 0], count_bounds[:, 1], out=mutated)
    return mutated


class CountVariable(Variable):
    """
    Discrete integer count variable.

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
        resolved_bounds: NDArray[np.int64] = resolve_bounds(bounds, int(values.shape[0]), dtype=np.int64)
        if lambda_ceiling is not None:
            ceiling = np.asarray(lambda_ceiling, dtype=np.float64)
        elif int(values.shape[0]) > 0:
            ceiling = 0.3 * (resolved_bounds[:, 1] - resolved_bounds[:, 0])
        else:
            ceiling = None

        super().__init__(
            name="count",
            values=values,
            strategy=strategy,
            strategy_recomb="geometric",
            adaptation_type="lognormal",
            sigma_floor=sigma_floor,
            strategy_ceiling=ceiling,
        )
        self.bounds: NDArray[np.int64] = resolved_bounds
        self.lambda_ceiling = ceiling

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> CountVariable:
        """Create a randomly initialized CountVariable within configured bounds."""
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

    def copy(self) -> CountVariable:
        """Create an independent deep copy of this CountVariable."""
        return CountVariable(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            bounds=self.bounds.copy(),
            sigma_floor=self.sigma_floor,
            lambda_ceiling=self.lambda_ceiling.copy() if self.lambda_ceiling is not None else None,
        )

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply Skellam discrete perturbation and integer domain clamping."""
        self.values = mutate_count(
            self.values,
            self.strategy,
            rng,
            self.bounds,
        )
