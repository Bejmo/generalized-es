# Heterogeneous Evolution Strategies for Mixed-Variable Optimization

## ES Architecture

## Overview

The engine implements a **(μ, λ) Evolution Strategy** (or **(μ + λ) Evolution Strategy**) with a heterogeneous genome and log-normal self-adaptation.

---

## Engine Modules (`src/es_model/`)

### `config.py` - Centralized Configuration

Single source of truth for all hyperparameters:

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `MU` | 15 | Survivors per generation |
| `LAMBDA` | 100 | Child generated per generation |
| `SELECTION_MODE` | `"comma"` | `"comma"` = (μ,λ) / `"plus"` = (μ+λ) |
| `MAX_GENERATIONS` | 500 | Maximum generations |
| `SIGMA_FLOOR` | 1e-6 | Stability floor for scale-like parameters |

**Genome blocks:**
- `N_REAL = 10` - Continuous real-valued variables
- `N_COUNT = 5` - Integer count variables
- `N_BINARY = 8` - Binary variables
- `N_CATEGORICAL = 3` - Categorical variables (cardinalities [5, 4, 3])
- `N_FUZZY = 4` - Fuzzy variables (triangular triplets with a <= b <= c)

---

### `kernels.py` - Vectorized Mutation Kernels

Pure, stateless functions operating on NumPy arrays. Each kernel implements a mutation operator for a specific block type.

#### Learning Rates (Log-Normal Self-Adaptation)

```
tau' = 1 / sqrt(2·n)       - global component (shared noise)
tau  = 1 / sqrt(2·sqrt(n)) - local component (per-dimension noise)
```

They are precomputed at import time based on `N_TOTAL`.

#### Available Kernels

| Kernel | Block | Mechanism |
|--------|-------|-----------|
| `adapt_scales()` | Real, Count, Fuzzy | Log-normal: sigma' = sigma · exp(tau'·N(0,1) + tau·N_i(0,1)) |
| `adapt_logits()` | Binary, Categorical | Additive Gaussian noise in logit space |
| `mutate_real()` | Real | Gaussian: x' = x + sigma·N(0,1), clipped to bounds |
| `mutate_count()` | Count | Skellam: Delta = Poisson(lambda) - Poisson(lambda), clipped |
| `mutate_binary()` | Binary | Bit-flip with p = sigmoid(logit) |
| `mutate_categorical()` | Categorical | Resampling excluding current index |
| `mutate_fuzzy()` | Fuzzy | Gaussian perturbation on triplets with a <= b <= c constraints |

---

### `engine.py` - Population Dynamics

#### `Individual` Structure

Dataclass with typed blocks:

```python
@dataclass
class Individual:
    # Decision variables
    real: NDArray[np.float64]
    count: NDArray[np.int64]
    binary: NDArray[np.uint8]
    categorical: NDArray[np.int64]
    fuzzy: NDArray[np.float64]

    # Strategy parameters (co-evolving)
    sigma_real: NDArray[np.float64]
    lambda_count: NDArray[np.float64]
    sigma_fuzzy: NDArray[np.float64]
    logit_binary: NDArray[np.float64]
    logit_cat: NDArray[np.float64]

    fitness: float = inf
```

#### Core Functions

- **`create_random_individual(rng)`** - Uniform initialization within configured bounds.
- **`clone_individual(parent)`** - Safe deep copy (all arrays are copied).
- **`mutate_individual(ind, rng)`** - Full pipeline: adapt strategy -> mutate genes.
- **`select_survivors(pool, mu, mode)`** - Deterministic rank-based selection. Returns the top μ individuals from the corresponding pool.

#### Selection

- **Comma mode (μ, λ):** Survivors are selected only from λ child.
- **Plus mode (μ + λ):** Survivors are selected from the combined parents + child pool.

---

### `main.py` - Executable Loop

Implements the complete generational cycle:

```
1. Initialize μ random parents
2. Evaluate fitness
3. For each generation:
    a. Generate λ child (clone random parent + mutate)
    b. Evaluate child
   c. Select μ survivors
   d. Log convergence every LOG_INTERVAL generations
```

The default objective function is **Sphere** (sum of squares on the real block). It can be replaced with any callable `Individual -> float`.

---

## Data Flow

```
config.py (hyperparameters)
    ↓
kernels.py (pure functions: adapt sigma, mutate genes)
    ↓
engine.py (Individual + clone + mutate + select)
    ↓
main.py (generational loop + evaluation)
```

All dependencies are unidirectional: `main -> engine -> kernels -> config`.
