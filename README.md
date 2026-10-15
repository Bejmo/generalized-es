# Generalized Evolution Strategies for Mixed-Variable Optimization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.12](https://img.shields.io/badge/Python-3.12+-blue.svg)](pyproject.toml)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://docs.astral.sh/uv/)

An experimental Bachelor's Thesis (TFG) in Computer Science at the **Facultat d'Informàtica de Barcelona (FIB - UPC)**.

This framework formalizes and implements a **Generalized $(\mu, \lambda)$-Evolution Strategy** designed for black-box optimization over heterogeneous search spaces. It co-evolves decision variables and internal strategy parameters with linear $\mathcal{O}(n)$ per-generation complexity, featuring native support for **6 distinct variable domains**, dual discrete/intermediate recombination, and modular benchmark suites.

---

## Decision Variable Genome & Mutation Kernels

The framework natively co-evolves 6 decision variable domains alongside their respective strategy parameters within a unified genome representation:

| Variable Type | Domain | Strategy Parameter | Mutation Operator | Mathematical Dynamics |
| :--- | :--- | :--- | :--- | :--- |
| **Real ($\mathbb{R}$)** | $`\mathbb{R}^{N_\text{real}}`$ | $`\boldsymbol{\sigma}_\text{real} \in \mathbb{R}_{>0}^{N_\text{real}}`$ | Gaussian | Specular reflection: $`x' = \text{reflect}(x + \sigma \cdot \mathcal{N}(0,1), [L, U])`$ |
| **Count ($\mathbb{Z}$)** | $`\mathbb{Z}_{\ge 0}^{N_\text{count}}`$ | $`\boldsymbol{\lambda}_\text{count} \in \mathbb{R}_{>0}^{N_\text{count}}`$ | Symmetric Skellam | Difference of Poissons: $`x' = \text{clip}(x + \text{Pois}_1(\lambda) - \text{Pois}_2(\lambda), [L, U])`$ |
| **Binary ($\mathbb{B}$)** | $`\{0, 1\}^{N_\text{binary}}`$ | $`\mathbf{z}_\text{bin} \in \mathbb{R}^{N_\text{binary}}`$ | Bit-Flip (XOR) | Guided by Bernoulli($`p`$), where $`p = \sigma(z) \in (0, 1)`$ |
| **Categorical ($\mathbb{C}$)** | $`\prod_k \{0, \dots, K_k - 1\}`$ | $`\mathbf{z}_\text{cat} \in \mathbb{R}^{\sum K_k}`$ | Softmax Resampling | Sample from categorical distribution: $`p_k = \text{softmax}(\mathbf{z}_k)`$ |
| **Fuzzy ($\mathbb{F}$)** | $`([0, 1] \times [\epsilon, d_\text{max}])^{N_\text{fuzzy}}`$ | $`\boldsymbol{\sigma}_\text{fuzzy} \in \mathbb{R}_{>0}^{2 \cdot N_\text{fuzzy}}`$ | Dual Gaussian Reflection | Perturb $`(m, d)`$: reflect $`m' \in [0, 1]`$, reflect & clamp $`d' \in [\epsilon, d_\text{max}]`$ |
| **Circular ($\mathbb{O}$)** | $`\prod_k \mathbb{Z}_{K_k}`$ | $`\boldsymbol{\lambda}_\text{circ} \in \mathbb{R}_{>0}^{N_\text{circ}}`$ | Modular Skellam | Ring offset: $`x' = (x + \text{Pois}_1(\lambda) - \text{Pois}_2(\lambda)) \pmod{K_k}`$ |

### Self-Adaptation Dynamics

**Global & Coordinate Learning Rates:**

$$
\tau' = \frac{1}{\sqrt{2 \cdot N_\text{total}}}, \quad \tau = \frac{1}{\sqrt{2 \cdot \sqrt{N_\text{total}}}}
$$

**Log-Normal Scale Adaptation** ($`\boldsymbol{\sigma}_\text{real}, \boldsymbol{\lambda}_\text{count}, \boldsymbol{\sigma}_\text{fuzzy}, \boldsymbol{\lambda}_\text{circ}`$):

$$
\sigma_i' = \max\left(\sigma_\text{floor}, \, \sigma_i \cdot \exp\left(\tau' \mathcal{N}(0, 1) + \tau \mathcal{N}_i(0, 1)\right)\right), \quad \sigma_\text{floor} = 10^{-6}
$$

**Logit-Normal Probability Adaptation** ($`\mathbf{z}_\text{bin}, \mathbf{z}_\text{cat}`$):

$$
\mathbf{z}_i' = \mathbf{z}_i + \tau \mathcal{N}_i(0, \mathbf{I})
$$

---

## Quick Start with `uv`

The project is fully managed with [uv](https://docs.astral.sh/uv/) for fast, reproducible, and lockfile-pinned builds.

### 1. Install `uv` (if needed)

```bash
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Synchronize Environment

From the project root:

```bash
uv sync
```

This single command automatically resolves Python 3.12, creates `.venv`, installs all dependencies (`numpy`, `matplotlib`, `cma`, `pymoo`, `pytest`, `ipykernel`), and registers the console scripts `es-model` and `generalized-es`.

---

## Running the Optimizer

Execute commands from the project root using `uv run`:

### Interactive Menu (Recommended)

```bash
uv run es-model
# or: uv run generalized-es
```

Launches an interactive terminal menu with options:
1. `model`: Runs the Generalized ES optimization loop on the default mixed configuration.
2. `benchmarks`: Evaluates baseline optimizers (*Isotropic ES*, *sep-CMA-ES*, *Standard CMA-ES*).
3. `smoke`: Fast sanity verification with minimal budget (`dim=5`, `evals=200`).
4. `all`: Executes `model` followed by `benchmarks`.

### Direct CLI Workflows

```bash
# Fast smoke validation
uv run es-model smoke

# Model execution with custom generations and logging frequency
uv run es-model model --generations 200 --log-interval 20

# Run continuous benchmarks (dim=20, 50,000 evaluations across 5 seeds)
uv run es-model benchmarks --dim 20 --max-evals 50000 --seeds 5
```

---

## Experiments & Notebooks

In-depth empirical notebooks are available under [`experiments/`](experiments):

* [`experiment_1.ipynb`](experiments/experiment_1.ipynb): Equal-budget continuous benchmark comparisons against CMA-ES and Isotropic ES.
* [`experiment_2.ipynb`](experiments/experiment_2.ipynb): Dynamic Target Shift (DTS) tracking and mutation self-adaptation across all 6 variable types in isolation.
* [`experiment_3.ipynb`](experiments/experiment_3.ipynb): Simultaneous co-evolution across all 6 variable types on synthetic heterogeneous landscapes.

To launch a Jupyter session:

```bash
uv run jupyter lab
```

---

## Testing & Verification

Execute the test suite with `pytest`:

```bash
uv run pytest
```

---

## Architecture & Technical Documentation

For detailed architectural specifications, execution flowcharts, candidate genome diagrams, the full directory structure, and the complete module dependency matrix, see:

👉 **[Architecture & Technical Documentation](docs/architecture.md)**

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
