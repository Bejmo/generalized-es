"""
circular.py: Closed-order circular variable with modular step mutation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .variable import Variable

if TYPE_CHECKING:
    from ..config import ESConfig


# -----------------------------------------------------------------------------
# Module-Level Mathematical Mutation Functions
# -----------------------------------------------------------------------------

def mutate_circular(
    values: NDArray[np.int64],
    lambdas: NDArray[np.float64],
    rng: np.random.Generator,
    circular_k: NDArray[np.int64],
) -> NDArray[np.int64]:
    """
    Circular mutation using a modular (wrapped) Skellam distribution.

    Delta = Poisson(lambda) - Poisson(lambda) (difference of two independent Poissons).
    The step wraps around modular cardinality: (values + delta) % circular_k.
    When lambda -> 0, Delta = 0 with probability -> 1 (natural step zero / inertia).

    Parameters
    ----------
    values : ndarray of int64
        Current circular indices.
    lambdas : ndarray of float64
        Adapted Poisson rate parameters for circular diffusion.
    rng : numpy Generator
    circular_k : ndarray of int64
        Cardinality (number of discrete positions) per circular variable.

    Returns
    -------
    ndarray of int64
        Mutated circular indices wrapped modulo k.
    """
    if values.shape[0] == 0:
        return values.copy()

    pos: NDArray[np.int64] = rng.poisson(lambdas).astype(np.int64)
    neg: NDArray[np.int64] = rng.poisson(lambdas).astype(np.int64)
    delta: NDArray[np.int64] = pos - neg
    mutated = (values + delta) % circular_k

    return mutated


def mutate_circular_forced_step(
    values: NDArray[np.int64],
    rng: np.random.Generator,
    circular_k: NDArray[np.int64],
) -> NDArray[np.int64]:
    """
    Circular mutation using forced step (+1 or -1 with probability 0.5, never 0).
    The step wraps around modular cardinality: (values + delta) % circular_k.
    """
    if values.shape[0] == 0:
        return values.copy()

    step: NDArray[np.int64] = np.where(rng.random(values.shape[0]) < 0.5, np.int64(1), np.int64(-1))
    mutated = (values + step) % circular_k
    return mutated


class CircularVariable(Variable):
    """
    Circular variable representing closed-order discrete positions (e.g., days of week).

    Each circular variable has cardinality k_i (positions 0..k_i-1).
    Mutation follows a modular (wrapped) Skellam distribution (difference of two Poissons modulo k)
    or a forced-step operator (+1/-1 with equal probability).
    Strategy parameters represent Poisson rates (lambdas) with log-normal self-adaptation.
    """

    def __init__(
        self,
        values: NDArray[np.int64],
        strategy: NDArray[np.float64],
        circular_k: NDArray[np.int64],
        sigma_floor: float = 1e-6,
        lambda_ceiling: float | NDArray[np.float64] | None = None,
        strategy_recomb: str = "geometric",
        mutation_operator: str = "skellam",
    ) -> None:
        c_k = np.asarray(circular_k, dtype=np.int64)
        if lambda_ceiling is not None:
            ceiling = np.asarray(lambda_ceiling, dtype=np.float64)
        elif int(values.shape[0]) > 0:
            ceiling = 0.3 * c_k.astype(np.float64)
        else:
            ceiling = None

        super().__init__(
            name="circular",
            values=values,
            strategy=strategy,
            strategy_recomb=strategy_recomb,
            adaptation_type="lognormal",
            sigma_floor=sigma_floor,
            strategy_ceiling=ceiling,
        )
        self.circular_k: NDArray[np.int64] = c_k
        self.lambda_ceiling = ceiling
        self.mutation_operator = mutation_operator

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> CircularVariable:
        """Create a randomly initialized CircularVariable."""
        values = (
            np.array([rng.integers(0, int(k)) for k in config.circular_k], dtype=np.int64)
            if config.n_circular > 0
            else np.zeros(0, dtype=np.int64)
        )
        init_lambda = getattr(config, "init_lambda_circular", 1.0)
        strategy = np.full(config.n_circular, init_lambda, dtype=np.float64)
        ceiling_factor = getattr(config, "sigma_ceiling_factor", 0.3)
        lambda_ceiling = (
            ceiling_factor * config.circular_k.astype(np.float64)
            if config.n_circular > 0
            else None
        )
        return cls(
            values=values,
            strategy=strategy,
            circular_k=config.circular_k,
            sigma_floor=getattr(config, "lambda_floor_circular", 1e-2),
            lambda_ceiling=lambda_ceiling,
            strategy_recomb=getattr(config, "strategy_recomb_scale", "geometric"),
            mutation_operator=getattr(config, "circular_mutation_operator", "skellam"),
        )

    def copy(self) -> CircularVariable:
        """Create an independent deep copy of this CircularVariable."""
        return CircularVariable(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            circular_k=self.circular_k.copy(),
            sigma_floor=self.sigma_floor,
            lambda_ceiling=self.lambda_ceiling.copy() if self.lambda_ceiling is not None else None,
            strategy_recomb=self.strategy_recomb,
            mutation_operator=self.mutation_operator,
        )

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply modular discrete perturbation."""
        if self.mutation_operator == "forced_step":
            self.values = mutate_circular_forced_step(
                self.values,
                rng,
                self.circular_k,
            )
        else:
            self.values = mutate_circular(
                self.values,
                self.strategy,
                rng,
                self.circular_k,
            )

    def distance_squared(self, other: Variable | NDArray[Any]) -> float:
        """Compute shortest modular geodesic squared distance on the circle."""
        other_vals = other.values if isinstance(other, Variable) else np.asarray(other, dtype=np.int64)
        if self.dim == 0 or other_vals.shape[0] == 0:
            return 0.0
        k = self.circular_k
        diff = np.abs(self.values - other_vals) % k
        geodesic = np.minimum(diff, k - diff)
        return float(np.sum(geodesic ** 2))
