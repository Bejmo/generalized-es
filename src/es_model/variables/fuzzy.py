"""
fuzzy.py: Fuzzy triangular variable with monotonic structural constraints.
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

def _reflect_unit(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Iteratively reflect values into [0, 1].

    Mirroring algorithm: values below 0 become |x|, values above 1 become 2-x,
    repeated until all are in [0, 1]. In practice <= 3 iterations suffice.
    """
    for _ in range(10):  # Safety cap on iterations.
        below = x < 0.0
        above = x > 1.0
        if not (below.any() or above.any()):
            break
        x = np.where(below, -x, x)
        x = np.where(above, 2.0 - x, x)
    return np.clip(x, 0.0, 1.0)  # Final safety clamp.


def _enforce_monotonic(a: float, b: float, c: float, fuzzy_epsilon: float = 1e-4) -> tuple[float, float, float]:
    """
    Guarantee a <= b <= c with epsilon margin.

    Uses a non-collapsing structural push so triangles do not flatten.
    """
    eps = fuzzy_epsilon
    # Sort to start from a feasible ordering.
    a, b, c = float(np.minimum(a, b)), float(np.median([a, b, c])), float(np.maximum(b, c))
    # Push b away from a and c away from b if too close.
    if b < a + eps:
        b = a + eps
    if c < b + eps:
        c = b + eps
    # Final reflection back into [0,1] if pushes exceeded bounds.
    a = float(np.clip(a, 0.0, 1.0 - 2 * eps))
    b = float(np.clip(b, a + eps, 1.0 - eps))
    c = float(np.clip(c, b + eps, 1.0))
    return a, b, c


def mutate_fuzzy(
    points: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    rng: np.random.Generator,
    n_fuzzy: int,
    fuzzy_epsilon: float = 1e-4,
) -> NDArray[np.float64]:
    """
    Gaussian perturbation on fuzzy control points with structural enforcement.

    Each fuzzy variable has 3 control points (a, b, c) stored contiguously:
    points shape = (n_fuzzy * 3,) and sigmas shape = (n_fuzzy * 3,).

    Steps:
    1. Perturb each point: x' = x + N(0, sigma^2)
    2. Reflect into [0, 1] using mirroring.
    3. Enforce monotonicity a <= b <= c with epsilon margin.

    Parameters
    ----------
    points : ndarray of float64
        Flattened control points [a0, b0, c0, a1, b1, c1, ...].
    sigmas : ndarray of float64
        Corresponding sigma vector.
    rng : numpy Generator
    n_fuzzy : int
        Number of fuzzy variables.
    fuzzy_epsilon : float
        Minimum epsilon margin.

    Returns
    -------
    ndarray of float64
        Mutated control points with structural guarantees.
    """
    if n_fuzzy == 0 or points.shape[0] == 0:
        return points.copy()
    mutated = points.copy()
    perturbation: NDArray[np.float64] = rng.standard_normal(points.shape[0]) * sigmas

    for i in range(n_fuzzy):
        idx = i * 3
        # Perturb.
        abc = points[idx : idx + 3] + perturbation[idx : idx + 3]
        # Reflect into [0, 1].
        abc = _reflect_unit(abc)
        # Enforce monotonic constraint.
        a, b, c = _enforce_monotonic(float(abc[0]), float(abc[1]), float(abc[2]), fuzzy_epsilon=fuzzy_epsilon)
        mutated[idx] = a
        mutated[idx + 1] = b
        mutated[idx + 2] = c

    return mutated


class FuzzyVariable(Variable):
    """
    Fuzzy triangular variable parameterized by triplets (a, b, c).

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
        super().__init__(
            name="fuzzy",
            values=values,
            strategy=strategy,
            strategy_recomb="geometric",
            adaptation_type="lognormal",
            sigma_floor=sigma_floor,
            strategy_ceiling=sigma_ceiling,
        )
        self.n_fuzzy: int = n_fuzzy
        self.fuzzy_epsilon: float = fuzzy_epsilon
        self.sigma_ceiling: float = sigma_ceiling
        self.sigma_floor: float = sigma_floor

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> FuzzyVariable:
        """Create a randomly initialized FuzzyVariable with valid monotonic triplets."""
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

    def copy(self) -> FuzzyVariable:
        """Create an independent deep copy of this FuzzyVariable."""
        return FuzzyVariable(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            n_fuzzy=self.n_fuzzy,
            fuzzy_epsilon=self.fuzzy_epsilon,
            sigma_ceiling=self.sigma_ceiling,
            sigma_floor=self.sigma_floor,
        )

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply Gaussian perturbation, unit reflection, and monotonic enforcement."""
        self.values = mutate_fuzzy(
            self.values,
            self.strategy,
            rng,
            self.n_fuzzy,
            self.fuzzy_epsilon,
        )
