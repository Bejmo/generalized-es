"""
config.py — Centralized configuration for the Heterogeneous (mu, lambda)-ES prototype.

All hyperparameters, bounds, dimensions, adaptation constants, and runtime flags
are declared here. Other modules import from this single source of truth.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. Population Dynamics
# ─────────────────────────────────────────────────────────────────────────────
MU: int = 15              # Number of survivors (parents for next generation)
LAMBDA: int = 100         # Number of offspring generated per generation
SELECTION_MODE: str = "comma"  # "comma" = (mu, lambda) | "plus" = (mu + lambda)
MAX_GENERATIONS: int = 500

# ─────────────────────────────────────────────────────────────────────────────
# 2. Genome Block Dimensions
# ─────────────────────────────────────────────────────────────────────────────
N_REAL: int = 10          # Number of real-valued decision variables
N_COUNT: int = 5          # Number of integer count variables
N_BINARY: int = 8         # Number of binary variables
N_CATEGORICAL: int = 3    # Number of categorical variables
N_FUZZY: int = 4          # Number of fuzzy triangular membership variables

# Categorical cardinalities: number of categories per categorical variable
CATEGORICAL_K: np.ndarray = np.array([5, 4, 3], dtype=np.int64)

# ─────────────────────────────────────────────────────────────────────────────
# 3. Bounds for Numerical Blocks
# ─────────────────────────────────────────────────────────────────────────────
# Real bounds: shape (N_REAL, 2) — [lower, upper] per dimension
REAL_BOUNDS: np.ndarray = np.column_stack([
    np.full(N_REAL, -5.0, dtype=np.float64),
    np.full(N_REAL, 5.0, dtype=np.float64),
])

# Count bounds: shape (N_COUNT, 2) — [lower, upper] per dimension
COUNT_BOUNDS: np.ndarray = np.column_stack([
    np.zeros(N_COUNT, dtype=np.int64),
    np.full(N_COUNT, 20, dtype=np.int64),
])

# ─────────────────────────────────────────────────────────────────────────────
# 4. Self-Adaptation Constants
# ─────────────────────────────────────────────────────────────────────────────
# Total active genome dimensionality used for learning-rate computation.
# Computed as sum of all variable dimensions (real + count + binary + cat + fuzzy*3).
N_TOTAL: int = N_REAL + N_COUNT + N_BINARY + N_CATEGORICAL + N_FUZZY * 3

# Stability floor: minimum allowed value for scale-like strategy parameters.
SIGMA_FLOOR: float = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# 5. Initial Strategy Parameter Defaults
# ─────────────────────────────────────────────────────────────────────────────
INIT_SIGMA_REAL: float = 0.5       # Initial mutation step-size for reals
INIT_LAMBDA_COUNT: float = 1.0     # Initial Poisson rate for counts
INIT_SIGMA_FUZZY: float = 0.05     # Initial sigma for fuzzy control points
INIT_LOGIT_BINARY: float = 0.0     # Initial logit for binary flip probability (p=0.5)
INIT_LOGIT_CAT: float = 0.0        # Initial logit for categorical mutation prob (p=0.5)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Sparse Mutation Gating
# ─────────────────────────────────────────────────────────────────────────────
SPARSE_RATE: float = 0.7  # Probability that each dimension is ACTIVE per offspring

# ─────────────────────────────────────────────────────────────────────────────
# 7. Fuzzy Structural Constants
# ─────────────────────────────────────────────────────────────────────────────
FUZZY_EPSILON: float = 1e-4  # Minimum margin between a, b, c to avoid collapse

# ─────────────────────────────────────────────────────────────────────────────
# 8. Deterministic Seed and Runtime Toggles
# ─────────────────────────────────────────────────────────────────────────────
SEED: int = 42                # Deterministic RNG seed for reproducibility
ENABLE_TRACE_LOG: bool = False  # Whether to log flattened genome vectors per gen
LOG_INTERVAL: int = 10        # Print convergence info every N generations
