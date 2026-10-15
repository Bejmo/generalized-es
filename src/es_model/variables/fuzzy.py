"""
fuzzy.py: Fuzzy triangular variable with (center, dispersion) representation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .variable import Variable

if TYPE_CHECKING:
    from ..config import ESConfig


# -----------------------------------------------------------------------------
# Module-Level Mathematical Functions
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


def membership(
    x: float | NDArray[np.float64],
    m: float | NDArray[np.float64],
    d: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """
    Evaluate triangular fuzzy membership degree: mu(x) = max(0, 1 - |x - m| / d).
    """
    return np.maximum(0.0, 1.0 - np.abs(x - m) / np.maximum(d, 1e-12))


def mutate_fuzzy(
    points: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    rng: np.random.Generator,
    n_fuzzy: int,
    fuzzy_epsilon: float = 1e-4,
    d_max: float = 1.0,
) -> NDArray[np.float64]:
    """
    Gaussian perturbation on fuzzy center and dispersion parameters.

    Each fuzzy variable has 2 parameters (m, d) stored contiguously:
    points shape = (n_fuzzy * 2,) and sigmas shape = (n_fuzzy * 2,).

    Mutation steps:
    1. Perturb center: m' = reflect_unit(m + N(0, sigma_m^2))
    2. Perturb dispersion: d' = |d + N(0, sigma_d^2)| (reflection at 0)
    3. Boundary guard: clamp d' in [fuzzy_epsilon, d_max] to avoid degenerate zero.
    """
    if n_fuzzy == 0 or points.shape[0] == 0:
        return points.copy()

    mutated = points.copy()
    m = points[0::2]
    d = points[1::2]
    sigma_m = sigmas[0::2]
    sigma_d = sigmas[1::2]

    # 1. Mutate center m with unit reflection
    pert_m = rng.standard_normal(n_fuzzy) * sigma_m
    m_new = _reflect_unit(m + pert_m)

    # 2. Mutate dispersion d with absolute-value reflection at 0 and upper ceiling reflection
    pert_d = rng.standard_normal(n_fuzzy) * sigma_d
    d_raw = np.abs(d + pert_d)
    # Upper boundary reflection if exceeding d_max
    d_reflected = np.where(d_raw > d_max, 2.0 * d_max - d_raw, d_raw)
    d_new = np.clip(d_reflected, fuzzy_epsilon, d_max)

    mutated[0::2] = m_new
    mutated[1::2] = d_new
    return mutated


class FuzzyVariable(Variable):
    """
    Fuzzy triangular variable parameterized by (center, dispersion) pairs (m, d).

    Each fuzzy variable defines a membership function on [0, 1] with peak at m and
    half-width d. The values array has shape (n_fuzzy * 2,).
    Mutation applies Gaussian perturbation with boundary reflection.
    """

    def __init__(
        self,
        values: NDArray[np.float64],
        strategy: NDArray[np.float64],
        n_fuzzy: int,
        fuzzy_epsilon: float = 1e-4,
        d_max: float = 1.0,
        sigma_ceiling: float = 0.5,
        sigma_floor: float = 1e-6,
        strategy_recomb: str = "geometric",
    ) -> None:
        super().__init__(
            name="fuzzy",
            values=values,
            strategy=strategy,
            strategy_recomb=strategy_recomb,
            adaptation_type="lognormal",
            sigma_floor=sigma_floor,
            strategy_ceiling=sigma_ceiling,
        )
        self.n_fuzzy: int = n_fuzzy
        self.fuzzy_epsilon: float = fuzzy_epsilon
        self.d_max: float = d_max
        self.sigma_ceiling: float = sigma_ceiling
        self.sigma_floor: float = sigma_floor

    @property
    def dims_per_var(self) -> int:
        """Each fuzzy variable defines 2 scalar genome parameters: center and dispersion."""
        return 2

    @property
    def num_variables(self) -> int:
        """Number of distinct fuzzy variables."""
        return self.n_fuzzy

    @property
    def num_fuzzy_vars(self) -> int:
        """Backward-compatible alias for num_variables."""
        return self.n_fuzzy

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> FuzzyVariable:
        """Create a randomly initialized FuzzyVariable with valid (m, d) pairs."""
        n_fuzzy = config.n_fuzzy
        if n_fuzzy == 0:
            values = np.zeros(0, dtype=np.float64)
        else:
            values = np.empty(n_fuzzy * 2, dtype=np.float64)
            # m uniform in [0, 1]
            values[0::2] = rng.random(n_fuzzy)
            # d uniform in [0.05, 0.4] to establish realistic initial support
            values[1::2] = rng.uniform(0.05, 0.4, size=n_fuzzy)

        strategy = np.full(n_fuzzy * 2, config.init_sigma_fuzzy, dtype=np.float64)
        d_max = getattr(config, "d_max_fuzzy", 1.0)
        return cls(
            values=values,
            strategy=strategy,
            n_fuzzy=n_fuzzy,
            fuzzy_epsilon=config.fuzzy_epsilon,
            d_max=d_max,
            sigma_ceiling=config.sigma_ceiling_fuzzy,
            sigma_floor=config.sigma_floor,
            strategy_recomb=getattr(config, "strategy_recomb_scale", "geometric"),
        )

    def copy(self) -> FuzzyVariable:
        """Create an independent deep copy of this FuzzyVariable."""
        return FuzzyVariable(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            n_fuzzy=self.n_fuzzy,
            fuzzy_epsilon=self.fuzzy_epsilon,
            d_max=self.d_max,
            sigma_ceiling=self.sigma_ceiling,
            sigma_floor=self.sigma_floor,
            strategy_recomb=self.strategy_recomb,
        )

    def mutate(self, rng: np.random.Generator) -> None:
        """Apply Gaussian perturbation with boundary reflection."""
        self.values = mutate_fuzzy(
            self.values,
            self.strategy,
            rng,
            self.n_fuzzy,
            self.fuzzy_epsilon,
            self.d_max,
        )

    def recombine_values(
        self,
        other: Variable,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """
        Atomic discrete recombination for fuzzy variables.

        Per fuzzy variable, inherits the complete (m, d) pair from one parent
        to preserve physical coupling between center and spread.
        """
        if self.dim == 0:
            return self.values.copy()
        mask_pairs: NDArray[np.bool_] = rng.random(self.n_fuzzy) < 0.5
        mask_expanded: NDArray[np.bool_] = np.repeat(mask_pairs, 2)
        return np.where(mask_expanded, self.values, other.values)

    def decode_triplets(self) -> NDArray[np.float64]:
        """
        Decode (m, d) pairs to (a, b, c) triangular control points clipped to [0, 1].
        """
        if self.n_fuzzy == 0:
            return np.zeros(0, dtype=np.float64)
        triplets = np.empty(self.n_fuzzy * 3, dtype=np.float64)
        m = self.values[0::2]
        d = self.values[1::2]
        triplets[0::3] = np.maximum(0.0, m - d)
        triplets[1::3] = m
        triplets[2::3] = np.minimum(1.0, m + d)
        return triplets

    def distance_squared(self, other: Variable | NDArray[Any]) -> float:
        """
        Compute canonical squared metric distance between fuzzy variables.

        Exact isometric projection of the vertex Euclidean squared distance:
        D^2 = 3 * sum((m1 - m2)^2) + 2 * sum((d1 - d2)^2).
        """
        other_vals = other.values if isinstance(other, Variable) else np.asarray(other, dtype=np.float64)
        if self.dim == 0 or other_vals.shape[0] == 0:
            return 0.0
        m1 = self.values[0::2]
        d1 = self.values[1::2]
        m2 = other_vals[0::2]
        d2 = other_vals[1::2]
        return float(3.0 * np.sum((m1 - m2) ** 2) + 2.0 * np.sum((d1 - d2) ** 2))
