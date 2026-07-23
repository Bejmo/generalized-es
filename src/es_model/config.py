"""
Configuration file: hyperparameters, bounds, dimensions, adaptation constants, and runtime flags.
NOTE: By note, this file only contains static configuration constants.
"""

import numpy as np

# -----------------------------------------------------------------------------
# 1. Population Dynamics
# -----------------------------------------------------------------------------
MU: int = 15              # Number of survivors
LAMBDA: int = 100         # Number of childs per generation
SELECTION_MODE: str = "comma"  # "comma" = (mu, lambda) | "plus" = (mu + lambda)
MAX_GENERATIONS: int = 500

# -----------------------------------------------------------------------------
# 2. Genome Dimensions
# -----------------------------------------------------------------------------
N_REAL: int = 10          # Number of real variables
N_COUNT: int = 5          # Number of count variables
N_BINARY: int = 8         # Number of binary variables
N_CATEGORICAL: int = 3    # Number of categorical variables
N_FUZZY: int = 4          # Number of fuzzy variables

# Categorical cardinalities: number of categories per categorical variable
CATEGORICAL_K: np.ndarray = np.array([3, 4, 5], dtype=np.int64)

# -----------------------------------------------------------------------------
# 3. Bounds for variables
# -----------------------------------------------------------------------------
# Real bounds: shape (N_REAL, 2) — [lower, upper] per dimension
LOWER_BOUND_REAL: float = -5.12
UPPER_BOUND_REAL: float = 5.12
REAL_BOUNDS: np.ndarray = np.column_stack([
    np.full(N_REAL, LOWER_BOUND_REAL, dtype=np.float64), # Lower bound for real variables
    np.full(N_REAL, UPPER_BOUND_REAL, dtype=np.float64),  # Upper bound for real variables
])

# Count bounds: shape (N_COUNT, 2) — [lower, upper] per dimension
LOWER_BOUND_COUNT: int = 0
UPPER_BOUND_COUNT: int = 20
COUNT_BOUNDS: np.ndarray = np.column_stack([
    np.full(N_COUNT, LOWER_BOUND_COUNT, dtype=np.int64), # Lower bound for count variables
    np.full(N_COUNT, UPPER_BOUND_COUNT, dtype=np.int64), # Upper bound for count variables
])

# -----------------------------------------------------------------------------
# 4. Self-Adaptation Constants
# -----------------------------------------------------------------------------
# Total active genome dimensionality used for learning-rate computation.
N_TOTAL: int = N_REAL + N_COUNT + N_BINARY + N_CATEGORICAL + N_FUZZY * 3 # Each fuzzy variable has 3 control points (a,b,c)

# Stability floor for strategy parameters to prevent numerical issues (e.g., zero or negative mutation step-sizes)
SIGMA_FLOOR: float = 1e-6

# -----------------------------------------------------------------------------
# 5. Initial Strategy Parameter Defaults
# -----------------------------------------------------------------------------
INIT_SIGMA_REAL: float = (UPPER_BOUND_REAL - LOWER_BOUND_REAL) / 3.0       # Initial mutation step-size for reals (~1/3 of domain range)
INIT_LAMBDA_COUNT: float = 1.0     # Initial Poisson rate for counts
INIT_SIGMA_FUZZY: float = 0.05     # Initial sigma for fuzzy control points
INIT_LOGIT_BINARY: float = 0.0     # Initial logit for binary flip probability (p=0.5)
INIT_LOGIT_CAT: float = 0.0        # Initial logit for categorical mutation prob (p=0.5)

# -----------------------------------------------------------------------------
# 6. Fuzzy Structural Constants
# -----------------------------------------------------------------------------
FUZZY_EPSILON: float = 1e-4  # Minimum margin between a, b, c to avoid collapse

# -----------------------------------------------------------------------------
# 7. Deterministic Seed and Runtime Toggles
# -----------------------------------------------------------------------------
SEED: int = 42                # Deterministic RNG seed for reproducibility
ENABLE_TRACE_LOG: bool = False  # Whether to log flattened genome vectors per gen
LOG_INTERVAL: int = 10        # Print convergence info every N generations
