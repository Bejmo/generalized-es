# Heterogeneous Evolution Strategies for Mixed-Variable Optimization

## Benchmark Framework

## Objective

Compare the custom heterogeneous ES against reference optimizers of similar complexity, using standard benchmark functions with equal evaluation budgets and multiple seeds for statistical significance.

---

## Objective Functions (`src/benchmarks/functions.py`)

| Function | Domain | Optimum | Characteristics |
|---------|---------|--------|-----------------|
| **Sphere** | [-5.12, 5.12]^n | f(0) = 0 | Unimodal, convex, separable |
| **Rastrigin** | [-5.12, 5.12]^n | f(0) = 0 | Multimodal (many local optima) |
| **Rosenbrock** | [-5.12, 5.12]^n | f(1) = 0 | Narrow valley, non-separable |
| **Ackley** | [-5.12, 5.12]^n | f(0) = 0 | Multimodal with global structure |

All functions accept an `n`-dimensional vector `x` and return a scalar value (lower is better).

---

## Compared Algorithms (`src/runners/`)

### 1. Prototype 1 - `custom_es.py`

Our custom Heterogeneous Evolution Strategy model. On these benchmarks, it is configured to optimize the real-valued block variables:
- $\mu = 15$, $\lambda = 100$ (default)
- Full coordinate-wise log-normal self-adaptation (independent $\sigma_i$ step-size for each dimension)

### 2. Isotropic ES (Simple Self-Adaptive) - `isotropic_es.py`

A classic self-adaptive baseline algorithm:
- $\mu = 15$, $\lambda = 100$
- Uses a single global step-size $\sigma$ for the entire individual ($n_\sigma = 1, n_\alpha = 0$)
- Log-normal self-adaptation on the global step-size with learning rate $\tau_0 = 1 / \sqrt{2 \cdot \text{dim}}$

### 3. sep-CMA-ES (Separable CMA-ES) - `sep_cma_es.py`

Diagonal Covariance Matrix Adaptation Evolution Strategy, wrapped around the `pycma` library with `CMA_diagonal=True`:
- Forces the covariance matrix to remain diagonal for the entire run
- Restricts the model to coordinate-wise variance learning, making it a direct baseline competitor to Prototype 1 (since both are restricted to diagonal variance/step-size models)

### 4. Standard CMA-ES (Upper Bound) - `standard_cma_es.py`

Full covariance matrix adaptation ES from the `pycma` library, used strictly as an **Upper Bound (performance ceiling)**:
- Learns the full covariance matrix (interactions and correlations between variables)
- Serves as the theoretical limit of what a full-covariance method can achieve, especially on rotated/non-separable functions where diagonal methods face a theoretical bottleneck


---

## Experimental Protocol

### Default Configuration

| Parameter | Value |
|-----------|-------|
| Dimensionality | 10 |
| Evaluation budget | 10,000 |
| Seeds | 10 (1..10) |

### Comparison Metrics

- **Convergence trace**: best-so-far fitness up to evaluation t
- **Reported result**: median final fitness across all seeds

### Execution

```bash
# Default configuration
python -m src.es_model.main benchmarks

# Customized
python -m src.es_model.main benchmarks --dim 20 --max-evals 50000 --seeds 30
```

---

## Runner Interface

Each runner implements the same signature:

```python
def run(fn, dim, bounds, seed, max_evals, **kwargs) -> np.ndarray:
    """
    Executes the optimizer and returns the convergence trace.

    Parameters
    ----------
    fn : callable
        Objective function f(x) -> float.
    dim : int
        Problem dimensionality.
    bounds : tuple(float, float)
        Bounds per dimension (lower, upper).
    seed : int
        Seed for reproducibility.
    max_evals : int
        Maximum objective-function evaluation budget.

    Returns
    -------
    np.ndarray of shape (max_evals,)
        Convergence trace: best-so-far at each evaluation.
    """
```

The trace always has length `max_evals`. If the algorithm finishes early, it is padded with the last best value found.

---

## Analysis Notebook

`experiments/notebooks/experiment_1.ipynb` provides:

1. Interactive execution of all runners
2. Convergence plots (matplotlib) per benchmark function
3. Visual comparison of medians with percentile bands
