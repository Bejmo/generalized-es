"""
variable.py: Abstract base class for decision variables and bounds utilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence
import numpy as np
from numpy.typing import NDArray

from .. import kernels


def resolve_bounds(
    bounds_spec: Any,
    dim: int,
    dtype: type = np.float64,
) -> NDArray[Any]:
    """
    Universal bounds resolver supporting scalar pairs, grouped bounds, and matrices.

    Parameters
    ----------
    bounds_spec : tuple, list, or ndarray
        - Scalar pair: (lower, upper) applied across all dimensions.
        - Grouped: [(count, (lower, upper)), ...] grouping dimensions with identical bounds.
        - Matrix: (dim, 2) explicit bounds per dimension.
    dim : int
        Expected total number of dimensions.
    dtype : type, default=np.float64
        Target NumPy datatype (np.float64 for reals, np.int64 for integer counts).

    Returns
    -------
    ndarray of shape (dim, 2)
        Validated bounds matrix with column 0 as lower bounds and column 1 as upper bounds.
    """
    if dim == 0:
        return np.empty((0, 2), dtype=dtype)

    # 1. Simple scalar pair (lower, upper)
    if (
        isinstance(bounds_spec, (tuple, list))
        and len(bounds_spec) == 2
        and isinstance(bounds_spec[0], (int, float, np.integer, np.floating))
        and isinstance(bounds_spec[1], (int, float, np.integer, np.floating))
    ):
        low, high = bounds_spec
        return np.column_stack([
            np.full(dim, low, dtype=dtype),
            np.full(dim, high, dtype=dtype),
        ])

    # 2. Grouped bounds: [(count_1, (low_1, high_1)), (count_2, (low_2, high_2)), ...]
    if isinstance(bounds_spec, (list, tuple)) and len(bounds_spec) > 0:
        first_item = bounds_spec[0]
        if (
            isinstance(first_item, (list, tuple))
            and len(first_item) == 2
            and isinstance(first_item[0], (int, np.integer))
        ):
            parts: list[NDArray[Any]] = []
            total_count = 0
            for item in bounds_spec:
                count, (low, high) = item
                parts.append(np.column_stack([
                    np.full(count, low, dtype=dtype),
                    np.full(count, high, dtype=dtype),
                ]))
                total_count += count
            if total_count != dim:
                raise ValueError(
                    f"Sum of grouped bounds dimensions ({total_count}) must equal dim ({dim})"
                )
            return np.vstack(parts).astype(dtype)

    # 3. Explicit matrix or array-like of shape (dim, 2)
    arr = np.asarray(bounds_spec, dtype=dtype)
    if arr.ndim != 2 or arr.shape != (dim, 2):
        raise ValueError(f"Bounds array must have shape ({dim}, 2), got {arr.shape}")
    return arr


class Variable(ABC):
    """
    Abstract base class representing a vector of variables of the same mathematical type
    co-evolving with their corresponding strategy parameters.
    """

    def __init__(
        self,
        name: str,
        values: NDArray[Any],
        strategy: NDArray[np.float64],
        strategy_recomb: str = "geometric",
        adaptation_type: str = "lognormal",
        sigma_floor: float = 1e-6,
        strategy_ceiling: float | NDArray[np.float64] | None = None,
        max_logit: float | None = None,
        min_logit: float | None = None,
    ) -> None:
        self.name = name
        self.values = values
        self.strategy = strategy
        self.strategy_recomb = strategy_recomb
        self.adaptation_type = adaptation_type
        self.sigma_floor = sigma_floor
        self.strategy_ceiling = strategy_ceiling
        self.max_logit = max_logit
        self.min_logit = min_logit

    @property
    def dims_per_var(self) -> int:
        """Number of scalar decision dimensions per logical variable (default: 1)."""
        return 1

    @property
    def num_variables(self) -> int:
        """Number of logical decision variables represented by this block."""
        return self.dim // self.dims_per_var if self.dims_per_var > 0 else self.dim

    @property
    def dim(self) -> int:
        """Total scalar decision dimensions in this block."""
        return int(self.values.shape[0])

    @property
    def strategy_dim(self) -> int:
        """Number of strategy parameters associated with this block."""
        return int(self.strategy.shape[0])

    @classmethod
    @abstractmethod
    def random(cls, config: Any, rng: np.random.Generator) -> Variable:
        """Instantiate a randomly initialized variable block using configuration parameters."""
        ...

    @abstractmethod
    def copy(self) -> Variable:
        """Create a deep copy of this variable block with independent NumPy arrays."""
        ...

    @abstractmethod
    def distance_squared(self, other: Variable | NDArray[Any]) -> float:
        """
        Compute the canonical squared metric distance between this block and another.
        """
        ...

    def adapt_strategy(
        self,
        rng: np.random.Generator,
        tau_prime: float,
        tau: float,
    ) -> None:
        """Adapt strategy parameters via self-adaptation rules."""
        if self.adaptation_type == "lognormal":
            self.strategy = kernels.apply_lognormal_selfadaptation(
                self.strategy,
                rng,
                tau_prime,
                tau,
                self.sigma_floor,
                self.strategy_ceiling,
            )
        elif self.adaptation_type == "logit":
            self.strategy = kernels.adapt_logits(
                self.strategy,
                rng,
                tau,
                max_logit=self.max_logit,
                min_logit=self.min_logit,
            )
        else:
            raise ValueError(f"Unknown adaptation_type: {self.adaptation_type}")

        self._post_adapt_strategy(rng, tau_prime, tau)

    def _post_adapt_strategy(
        self,
        rng: np.random.Generator,
        tau_prime: float,
        tau: float,
    ) -> None:
        """Optional hook for post-adaptation adjustments (e.g. centering categorical logits)."""
        pass

    @abstractmethod
    def mutate(self, rng: np.random.Generator) -> None:
        """Mutate decision variables using currently adapted strategy parameters."""
        ...

    def recombine_values(
        self,
        other: Variable,
        rng: np.random.Generator,
    ) -> NDArray[Any]:
        """
        Dual discrete recombination for decision variables.

        Per gene, picks between self and other with 50% probability.
        """
        if self.dim == 0:
            return self.values.copy()
        mask: NDArray[np.bool_] = rng.random(self.dim) < 0.5
        return np.where(mask, self.values, other.values)

    def recombine_strategy(
        self,
        others: Sequence[Variable],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """
        Global intermediate recombination for strategy parameters.

        For each parameter index j, picks a random donor block from others.
        Applies geometric mean for scale parameters (preventing inflationary drift),
        and arithmetic mean for logit parameters (log-odds space).
        """
        n = self.strategy_dim
        if n == 0:
            return self.strategy.copy()
        result = self.strategy.copy()
        num_parents = len(others)
        for j in range(n):
            second_idx = int(rng.integers(0, num_parents))
            donor_val = others[second_idx].strategy[j]
            if self.strategy_recomb == "geometric":
                result[j] = float(np.sqrt(np.maximum(result[j] * donor_val, 1e-12)))
            else:
                result[j] = (result[j] + donor_val) / 2.0
        return result

    def flatten(self) -> NDArray[np.float64]:
        """
        Flatten values to a 1-D float64 array for concatenating into the global genome vector.
        """
        return self.values.astype(np.float64)
