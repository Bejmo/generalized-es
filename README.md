# Heterogeneous Evolution Strategies for Mixed-Variable Optimization

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

### Recommended Usage

The interactive menu is the recommended way to run the project. It is the simplest option for normal use and lets you choose the workflow directly from `main.py`.

```bash
python -m src.es_model.main
```

Available menu options:

- `model`: runs the heterogeneous ES optimization loop with the model configuration.
- `benchmarks`: runs the benchmark comparison against the reference optimizers.
- `smoke`: runs a short validation of model + benchmarks with small default values.
- `all`: runs the full model workflow first and then the benchmark workflow.
- `exit`: closes the program without running anything.

When you choose `model`, `benchmarks`, `smoke`, or `all`, `main.py` asks for the most relevant parameters and applies defaults if you just press Enter.

### Direct Command Alternatives

If you want to skip the menu or launch a run with fixed parameters, you can still use direct commands:

```bash
# Direct model run with custom generations
python -m src.es_model.main model --generations 200 --log-interval 20

# Benchmarks with custom dimension and evaluation budget
python -m src.es_model.main benchmarks --dim 20 --max-evals 50000 --seeds 5

# Fast smoke test
python -m src.es_model.main smoke --dim 5 --max-evals 200 --seeds 2
```

### Interactive Notebook

Open `experiments/notebooks/experiment_1_equal_budget_benchmark_comparison.ipynb` in VS Code or Jupyter and select the `.venv` kernel.

### Named Experiments

- **Experiment 1: Equal-Budget Benchmark Comparison**
	- Notebook: `experiments/notebooks/experiment_1_equal_budget_benchmark_comparison.ipynb`
	- Goal: compare Custom ES against Differential Evolution, PSO, and (1+1)-ES under the same evaluation budget.
	- Protocol: real-valued benchmark block, 10 seeds, fixed `MAX_EVALS` per runner.
	- Output: convergence plots and final median fitness summary per benchmark.

---

## Project Structure

```
TFG/
├── src/                        # Main source code
│   ├── es_model/               #   Heterogeneous ES engine
│   │   ├── config.py           #     Centralized hyperparameters
│   │   ├── kernels.py          #     Vectorized mutation kernels
│   │   ├── engine.py           #     Population dynamics and selection
│   │   └── main.py             #     Single terminal entrypoint + model loop
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

## Migration Note

Old commands using `python -m src.cli.runner ...` were retired.

Use:

- `python -m src.es_model.main` (interactive menu)
- `python -m src.es_model.main <mode>` for direct mode execution

---

## License

Academic project - Heterogeneous Evolution Strategies for Mixed-Variable Optimization.
