# Generalized Evolution Strategies for Mixed-Variable Optimization

## Benchmark Framework

## Objective

Compare the custom heterogeneous ES against reference optimizers of similar complexity, using standard benchmark functions with equal evaluation budgets and multiple seeds for statistical significance.

---

## Objective Functions (`src/experiments/benchmarks/functions.py`)

| Function | Domain | Optimum | Characteristics |
|---------|---------|--------|-----------------|
| **Sphere** | [-5.12, 5.12]^n | f(0) = 0 | Unimodal, convex, separable |
| **Rastrigin** | [-5.12, 5.12]^n | f(0) = 0 | Multimodal (many local optima) |
| **Rosenbrock** | [-5.12, 5.12]^n | f(1) = 0 | Narrow valley, non-separable |
| **Ackley** | [-5.12, 5.12]^n | f(0) = 0 | Multimodal with global structure |

All functions accept an `n`-dimensional vector `x` and return a scalar value (lower is better).

---

## Compared Algorithms (`src/experiments/runners/`)

### 1. Custom ES (μ, λ) - `custom_es.py`

Wrapper around the main engine. It temporarily configures the engine to use only the real-valued block with the benchmark dimension, then restores the original configuration after execution.

- μ = 15, λ = 100 (default)
- Full log-normal self-adaptation
- Sparse mutation gating

### 2. Differential Evolution - `differential_evolution.py`

Wrapper around `scipy.optimize.differential_evolution`:
- Latin Hypercube initialization
- Runs until the evaluation budget is exhausted
- No final polishing

### 3. PSO (Particle Swarm Optimization) - `pso.py`

Wrapper around `pyswarms.single.GlobalBestPSO`:
- 50 particles by default
- Cognitive/social/inertia parameters: c1=1.5, c2=1.5, w=0.7
- Iterations = max_evals / n_particles

### 4. (1+1)-ES with the 1/5 Rule - `one_plus_one_es.py`

Classic baseline:
- One parent, one offspring per generation
- Step-size (sigma) adaptation via the 1/5 success rule
- Observation window: 10·dim evaluations
- If success rate > 0.2 -> sigma × 1.2; if < 0.2 -> sigma × 0.82

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
python -m src.cli.runner benchmarks

# Customized
python -m src.cli.runner benchmarks --dim 20 --max-evals 50000 --seeds 30
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

`experiments/notebooks/benchmark_comparison.ipynb` provides:

1. Interactive execution of all runners
2. Convergence plots (matplotlib) per benchmark function
3. Visual comparison of medians with percentile bands
