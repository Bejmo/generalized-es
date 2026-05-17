# Generalized Evolution Strategies for Mixed-Variable Optimization

Prototype of a **Heterogeneous Evolution Strategy** with (μ, λ) / (μ + λ) selection and log-normal self-adaptation. It includes a comparison framework against reference optimizers (Differential Evolution, PSO, and (1+1)-ES with the 1/5 rule).

---

## Prerequisites

- **Python 3.12+** (tested with CPython 3.14)
- Operating system: Windows / Linux / macOS

---

## Quick Setup

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate the environment
# Windows (PowerShell)
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running The Project

Run all commands from the **project root** with the virtual environment activated.

### Unified CLI

| Command | Description |
|---------|-------------|
| `python -m src.cli.runner model` | Runs the heterogeneous ES optimization loop |
| `python -m src.cli.runner benchmarks` | Runs algorithm comparison on benchmark functions |
| `python -m src.cli.runner smoke` | Runs a quick smoke validation |
| `python -m src.cli.runner all` | Runs model + benchmarks sequentially |

### Common Options

```bash
# Model run with custom generations
python -m src.cli.runner model --generations 200 --log-interval 20

# Benchmarks with custom dimension and evaluation budget
python -m src.cli.runner benchmarks --dim 20 --max-evals 50000 --seeds 5

# Fast smoke test
python -m src.cli.runner smoke --dim 5 --max-evals 200 --seeds 2
```

### Interactive Notebook

Open `experiments/notebooks/benchmark_comparison.ipynb` in VS Code or Jupyter and select the `.venv` kernel.

---

## Project Structure

```
TFG/
├── src/                        # Main source code
│   ├── es_model/               #   Heterogeneous ES engine
│   │   ├── config.py           #     Centralized hyperparameters
│   │   ├── kernels.py          #     Vectorized mutation kernels
│   │   ├── engine.py           #     Population dynamics and selection
│   │   └── main.py             #     Executable optimization loop
│   ├── cli/                    #   Command-line interface
│   │   ├── runner.py           #     Unified CLI entry point
│   │   ├── model_runner.py     #     ES model workflow
│   │   └── benchmark_runner.py #     Benchmark workflow
│   └── experiments/            #   Experiment modules and helper runners
│       ├── benchmarks/         #     Objective functions (Sphere, Rastrigin...)
│       └── runners/            #     Reference algorithm wrappers
├── experiments/                # Experimental notebooks and artifacts
│   └── notebooks/              #   Analysis Jupyter notebooks
├── docs/                       # Detailed technical documentation
├── openspec/                   # OpenSpec design specifications
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## Dependencies

| Package | Usage |
|---------|-----|
| `numpy` | ES vectorized operations and typed arrays |
| `scipy` | Differential Evolution (reference algorithm) |
| `pyswarms` | Particle Swarm Optimization (reference algorithm) |
| `matplotlib` | Convergence plots in notebooks |

---

## Technical Documentation

For implementation details, architecture, and algorithms, see [docs/](docs/):

- [ES Architecture](docs/arquitectura-es.md) - Evolutionary engine design
- [Benchmarks](docs/benchmarks.md) - Objective functions and comparison framework
- [Configuration](docs/configuracion.md) - Hyperparameters and tuning

---

## License

Academic project - Generalized Evolution Strategies for Mixed-Variable Optimization.
