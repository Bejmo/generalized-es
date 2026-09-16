from dataclasses import dataclass, field
from typing import Any
import numpy as np
from numpy.typing import NDArray

from .blocks.base import resolve_bounds


@dataclass
class ESConfig:
    """
    Configuration dataclass for the Generalized Evolution Strategy.
    Encapsulates all hyperparameters, genome dimensions, variable bounds,
    and self-adaptation defaults.
    """

    # -----------------------------------------------------------------------------
    # 1. Population Dynamics
    # -----------------------------------------------------------------------------
    mu: int = 15                   # Number of survivors (parents)
    lam: int = 100                 # Number of children per generation
    selection_mode: str = "comma"  # "comma" = (mu, lambda) | "plus" = (mu + lambda)
    max_generations: int = 500

    # -----------------------------------------------------------------------------
    # 2. Genome Dimensions
    # -----------------------------------------------------------------------------
    n_real: int = 10               # Number of real variables
    n_count: int = 5               # Number of count variables
    n_binary: int = 8              # Number of binary variables
    n_categorical: int = 3         # Number of categorical variables
    n_fuzzy: int = 4               # Number of fuzzy variables
    n_circular: int = 0             # Number of circular variables (opt-in)

    # Categorical cardinalities: number of categories per categorical variable
    categorical_k: NDArray[np.int64] = field(
        default_factory=lambda: np.array([3, 4, 5], dtype=np.int64)
    )

    # Circular cardinalities: number of discrete positions per circular variable
    circular_k: NDArray[np.int64] = field(
        default_factory=lambda: np.array([], dtype=np.int64)
    )

    # -----------------------------------------------------------------------------
    # 3. Bounds for variables
    # -----------------------------------------------------------------------------
    lower_bound_real: float = -5.12
    upper_bound_real: float = 5.12

    lower_bound_count: int = 0
    upper_bound_count: int = 20

    # Custom real/count bounds override: supports scalar pair, grouped list, or (N, 2) matrix.
    custom_real_bounds: Any = None
    custom_count_bounds: Any = None

    # -----------------------------------------------------------------------------
    # 4. Self-Adaptation Constants
    # -----------------------------------------------------------------------------
    sigma_floor: float = 1e-6      # Stability floor for strategy parameters
    sigma_ceiling_fuzzy: float = 0.5  # Maximum sigma for fuzzy [0,1] domain (prevents drift)

    # -----------------------------------------------------------------------------
    # 5. Initial Strategy Parameter Defaults
    # -----------------------------------------------------------------------------
    init_lambda_count: float = 1.0  # Initial Poisson rate for counts
    init_sigma_fuzzy: float = 0.05  # Initial sigma for fuzzy control points
    init_logit_binary: float = 0.0  # Initial logit for binary flip probability (p=0.5)
    init_logit_cat: float = 0.0     # Initial logit for categorical mutation prob (p=0.5)
    init_logit_circular: float = 0.0 # Initial forward logit for circular variables (p_f=0.5)

    # -----------------------------------------------------------------------------
    # 6. Fuzzy Structural Constants
    # -----------------------------------------------------------------------------
    fuzzy_epsilon: float = 1e-4    # Minimum margin between a, b, c to avoid collapse

    # -----------------------------------------------------------------------------
    # 7. Deterministic Seed and Runtime Toggles
    # -----------------------------------------------------------------------------
    seed: int = 42                 # Deterministic RNG seed for reproducibility
    enable_trace_log: bool = False # Whether to log flattened genome vectors per gen
    log_interval: int = 10         # Print convergence info every N generations

    # -----------------------------------------------------------------------------
    # Dynamic Computed Properties
    # -----------------------------------------------------------------------------
    @property
    def n_total(self) -> int:
        """Total active genome dimensionality used for learning-rate computation."""
        return (
            self.n_real
            + self.n_count
            + self.n_binary
            + self.n_categorical
            + self.n_fuzzy * 3
            + self.n_circular
        )

    @property
    def total_cat_logits(self) -> int:
        """Total number of logit dimensions for all categorical variables."""
        return int(np.sum(self.categorical_k)) if self.n_categorical > 0 else 0

    @property
    def real_bounds(self) -> NDArray[np.float64]:
        """Real variable bounds matrix of shape (n_real, 2)."""
        if self.custom_real_bounds is not None:
            return resolve_bounds(self.custom_real_bounds, self.n_real, dtype=np.float64)
        if self.n_real == 0:
            return np.empty((0, 2), dtype=np.float64)
        return resolve_bounds(
            (self.lower_bound_real, self.upper_bound_real),
            self.n_real,
            dtype=np.float64,
        )

    @property
    def count_bounds(self) -> NDArray[np.int64]:
        """Count variable bounds matrix of shape (n_count, 2)."""
        if self.custom_count_bounds is not None:
            return resolve_bounds(self.custom_count_bounds, self.n_count, dtype=np.int64)
        if self.n_count == 0:
            return np.empty((0, 2), dtype=np.int64)
        return resolve_bounds(
            (self.lower_bound_count, self.upper_bound_count),
            self.n_count,
            dtype=np.int64,
        )

    @property
    def init_sigma_real(self) -> float:
        """Initial mutation step-size for reals (~1/3 of domain range)."""
        return (self.upper_bound_real - self.lower_bound_real) / 3.0

    @property
    def tau_prime_and_tau(self) -> tuple[float, float]:
        """Compute global (tau_prime) and local (tau) learning rates."""
        if self.n_total == 0:
            return 0.0, 0.0
        tau_prime: float = 1.0 / np.sqrt(2.0 * self.n_total)
        tau: float = 1.0 / np.sqrt(2.0 * np.sqrt(self.n_total))
        return tau_prime, tau


# Default configuration instance for quick use / backward compatibility
DEFAULT_CONFIG = ESConfig()
