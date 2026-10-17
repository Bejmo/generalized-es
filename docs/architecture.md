# Architecture & Technical Documentation

This document provides in-depth technical documentation for the **Generalized Evolution Strategies (ES)** framework, covering module design, execution control flows, genome data structures, and the complete dependency matrix.

---

## 1. System Execution & Control Flow

### CLI Dispatcher & Workflows

The framework provides an interactive terminal menu and direct command-line arguments orchestrated by `src.es_model.main`:

```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': { 'fontSize': '12px', 'nodePadding': '6' }}}%%
flowchart TD
    CLI["CLI Entrypoint<br/><code>es-model</code> / <code>generalized-es</code> / <code>uv run python -m src.es_model.main</code>"] --> CheckMode{"Mode Provided?"}
    CheckMode -- "No" --> Menu["Interactive Terminal Menu"]
    CheckMode -- "Yes" --> Dispatch["CLI Argument Dispatcher"]
    
    Menu -->|1. model| ModeModel["Workflow: model"]
    Menu -->|2. benchmarks| ModeBench["Workflow: benchmarks"]
    Menu -->|3. smoke| ModeSmoke["Workflow: smoke"]
    Menu -->|4. all| ModeAll["Workflow: all (model + benchmarks)"]
    Menu -->|5. exit| ModeExit(["Exit"])

    Dispatch -->|model| ModeModel
    Dispatch -->|benchmarks| ModeBench
    Dispatch -->|smoke| ModeSmoke
    Dispatch -->|all| ModeAll

    ModeModel --> RunModel["run_model()<br/>Execute GeneralizedES optimization loop"]
    ModeBench --> RunBench["run_benchmarks()<br/>Evaluate against baseline optimizers"]
    ModeSmoke --> RunSmoke["run_model() + run_benchmarks()<br/>Fast sanity validation (dim=5, evals=200)"]
    ModeAll --> RunAll["run_model() then run_benchmarks()"]
```

---

## 2. Genome & Candidate Representation

Each candidate solution in the population is encapsulated by an `Individual` instance containing a polymorphic dictionary of variables co-evolving their internal strategy parameters:

```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': { 'fontSize': '12px', 'nodePadding': '6' }}}%%
flowchart TD
    Ind["Individual<br/>• fitness: float<br/>• variables: dict[str, Variable]"]
    
    Ind --> RV["RealVariable<br/>• values: float64[N_real]<br/>• strategy: sigma_real[N_real]<br/>• bounds: [lower, upper]"]
    Ind --> CV["CountVariable<br/>• values: int64[N_count]<br/>• strategy: lambda_count[N_count]<br/>• bounds: [lower, upper]"]
    Ind --> BV["BinaryVariable<br/>• values: uint8[N_binary]<br/>• strategy: logit_binary[N_binary]<br/>• bounds: [0, 1]"]
    Ind --> CatV["CategoricalVariable<br/>• values: int64[N_cat]<br/>• strategy: logit_cat[N_cat] (p_mut = sigmoid(z))<br/>• cardinalities: K"]
    Ind --> FV["FuzzyVariable<br/>• values: float64[2 * N_fuzzy] (center m, dispersion d)<br/>• strategy: sigma_fuzzy[2 * N_fuzzy]<br/>• bounds: [0, 1] x [epsilon, d_max]"]
    Ind --> CircV["CircularVariable<br/>• values: int64[N_circ]<br/>• strategy: lambda_circ[N_circ]<br/>• cardinalities: K"]

    Ind --> Genome["genome property<br/>Canonical flattened 1-D vector: [real, count, binary, categorical, fuzzy, circular]"]
```

---

## 3. Algorithmic Lifecycle Loop

The generational lifecycle adheres to canonical $(\mu, \lambda)$-ES and $(\mu + \lambda)$-ES dynamics:

```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': { 'fontSize': '12px', 'nodePadding': '6' }}}%%
flowchart TD
    Init["Initialize Parent Population (MU)"] --> EvalParents["Evaluate Parent Fitness"]
    EvalParents --> LoopStart["Start Generational Loop"]
    
    subgraph OffspringLoop ["Generate LAMBDA Offspring"]
        direction TB
        Recombine["1. Recombine Parents (Dual Discrete & Global Intermediate)"]
        AdaptS["2. Adapt Strategy Parameters (Log-Normal & Logit-Normal)"]
        MutateV["3. Mutate Decision Variables (Gaussian, Skellam, Modular Jump, Ring)"]
        EvalC["4. Evaluate Offspring Fitness"]
        
        Recombine --> AdaptS --> MutateV --> EvalC
    end

    LoopStart --> OffspringLoop
    OffspringLoop --> CheckLambda{"Completed LAMBDA Offspring?"}
    CheckLambda -- No --> OffspringLoop
    CheckLambda -- Yes --> Selection["Survivor Selection<br/>• Comma (μ, λ): Top μ offspring<br/>• Plus (μ + λ): Top μ parents + offspring"]
    
    Selection --> CheckGen{"Generation < MAX_GENERATIONS?"}
    CheckGen -- Yes --> LoopStart
    CheckGen -- No --> End(["Optimization Complete<br/>Return Best Individual"])
```

---

## 4. Project Structure

```text
TFG/
├── README.md                                               # Main project overview and quick start
├── pyproject.toml                                          # Project metadata, dependencies & CLI scripts (PEP 621)
├── uv.lock                                                 # Deterministic cross-platform lockfile
├── .python-version                                         # Pinned Python version (3.12)
├── docs/                                                   # In-depth architectural & technical documentation
│   └── architecture.md                                     # System design, module matrix, and control flow
├── src/                                                    # Core source code
│   ├── es_model/                                           #   Generalized ES Engine
│   │   ├── config.py                                       #     Problem settings, bounds, dimensions & learning rates
│   │   ├── kernels.py                                      #     Pure math functions for activation & self-adaptation
│   │   ├── individual.py                                   #     Candidate solution representation & genetic operators
│   │   ├── generalized_es.py                               #     GeneralizedES optimizer class & survivor selection
│   │   ├── main.py                                         #     CLI entrypoint, interactive menu & execution workflows
│   │   └── variables/                                      #     Modular variable representations & mutation operators
│   │       ├── variable.py                                 #       Variable ABC & resolve_bounds utility
│   │       ├── real.py                                     #       RealVariable (Gaussian mutation with reflection)
│   │       ├── count.py                                    #       CountVariable (Skellam mutation with clipping)
│   │       ├── binary.py                                   #       BinaryVariable (Bit-flip mutation via sigmoid)
│   │       ├── categorical.py                              #       CategoricalVariable (Modular jump with phenotypic inertia)
│   │       ├── fuzzy.py                                    #       FuzzyVariable (Center-dispersion parameterization with reflection)
│   │       └── circular.py                                 #       CircularVariable (Modular Skellam ring mutation)
│   ├── benchmarks/                                         #   Objective Benchmark Landscapes
│   │   ├── functions.py                                    #     Standard continuous landscapes (Sphere, Rastrigin, Ackley)
│   │   └── generalized_functions.py                        #     Synthetic mixed-variable landscapes testing all 6 types
│   └── runners/                                            #   Experimental Benchmark Runners
│       ├── custom_es.py                                    #     Generalized ES runner adapter
│       ├── isotropic_es.py                                 #     Isotropic ES baseline (single scalar step-size)
│       ├── sep_cma_es.py                                   #     Separable CMA-ES baseline wrapper
│       └── standard_cma_es.py                              #     Standard CMA-ES upper-bound runner
├── tests/                                                  # Unit & integration test suites
│   ├── test_variables.py                                   #     Variable representations & mutation tests
│   ├── test_individual_and_es.py                           #     Candidate individual & ES optimization cycle tests
│   ├── test_benchmarks.py                                  #     Continuous & generalized benchmark landscape tests
│   └── test_catcmawm_baseline.py                           #     CatCMAwM baseline adapter & domain invariant tests
└── experiments/                                            # Experimental suites & analysis
    ├── experiment_1/                                       #     Continuous benchmarks vs. isotropic/sep/standard CMA-ES
    │   ├── experiment_1.ipynb                              #       Analysis notebook & figures
    │   └── runners/run_experiment_1.py                     #       Execution & statistical testing runner
    ├── experiment_2/                                       #     Dynamic Target Shift (DTS) transversal dynamic tracking
    │   ├── experiment_2.ipynb                              #       Analysis notebook & transversal dynamic tracking
    │   └── runners/                                        #       Execution and plotting scripts
    ├── experiment_3/                                       #     Heterogeneous mixed-variable benchmarks (4-way comparison)
    │   ├── experiment_3.ipynb                              #       Interactive analysis notebook
    │   ├── runners/                                        #       Execution scripts & statistical testing
    │   │   ├── run_experiment_3.py                         #         Main runner & paired Wilcoxon hypothesis testing
    │   │   ├── run_exp3_baselines.py                       #         External baselines runner (RS, CMA-ES, CatCMAwM)
    │   │   ├── runners.py                                  #         Generalized ES benchmark suite runners
    │   │   └── plotting.py                                 #         Convergence curves (Median/IQR) & parameter dynamics
    │   └── data/                                           #       Raw traces (.npz) & baseline results (.json)
    └── extra/                                              #     Supplementary validation studies (Appendix E)
        ├── categorical/                                    #       Categorical operator validation & binary equivalence
        └── circular_and_geometric_mean/                    #       Modular circular & geometric mean validation
```

---

## 5. Module Dependency Graph

```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': { 'fontSize': '12px', 'nodePadding': '6' }}}%%
flowchart TD
    Kernels["src.es_model.kernels"]
    Config["src.es_model.config"]
    VarABC["src.es_model.variables.variable"]
    Vars["src.es_model.variables.*<br/>(real, count, binary, ...)"]
    Ind["src.es_model.individual"]
    ES["src.es_model.generalized_es"]
    Bench["src.benchmarks.*<br/>(functions, generalized_functions)"]
    Runners["src.runners.*<br/>(custom_es, isotropic, sep_cma, standard_cma)"]
    Main["src.es_model.main"]

    VarABC --> Config
    Kernels --> VarABC
    VarABC --> Vars
    Vars --> Ind
    Config --> Ind
    Ind --> ES
    Config --> ES
    ES --> Runners
    Bench --> Runners
    Runners --> Main
    ES --> Main
    Config --> Main
    Bench --> Main
```

---

## 6. Module Dependency Matrix

| Module | Primary Responsibility | Key Symbols / Classes | Internal Dependencies | Imported By |
| :--- | :--- | :--- | :--- | :--- |
| `src.es_model.config` | Configuration dataclass, hyper-parameters, bounds & learning rates | `ESConfig`, `DEFAULT_CONFIG` | `variables.variable` | `individual`, `generalized_es`, `main`, `runners`, `tests` |
| `src.es_model.kernels` | Stateless numerical primitives, log-normal/logit adaptation, activation | `apply_lognormal_selfadaptation`, `adapt_logits`, `sigmoid`, `softmax` | None | `variables.variable`, all variable modules |
| `src.es_model.variables.variable` | Abstract base class for variable types and universal bounds resolver | `Variable`, `resolve_bounds` | `kernels` | `config`, all concrete variable modules |
| `src.es_model.variables.*` | Concrete variable implementations with type-specific mutations | `RealVariable`, `CountVariable`, `BinaryVariable`, `CategoricalVariable`, `FuzzyVariable`, `CircularVariable` | `variable.Variable`, `kernels` | `individual`, `tests` |
| `src.es_model.individual` | Candidate solution representation, discrete recombination & genome extraction | `Individual`, `UnpackedGenome`, `unpack_genome` | `config`, `variables` | `generalized_es`, `main`, `benchmarks`, `tests` |
| `src.es_model.generalized_es` | Stateful optimization engine and $(\mu, \lambda) / (\mu + \lambda)$ survivor selection | `GeneralizedES`, `select_survivors` | `config`, `individual` | `main`, `runners.custom_es`, `tests` |
| `src.benchmarks.functions` | Standard continuous optimization landscapes | `BENCHMARKS` (`Sphere`, `Rastrigin`, `Rosenbrock`, `Ackley`) | None | `main`, `runners`, `tests` |
| `src.benchmarks.generalized_functions` | Synthetic mixed-variable test landscapes across all 6 domain types | `sphere_generalized`, `rastrigin_generalized`, `coupled_generalized` | `config`, `individual` | `experiments`, `tests` |
| `src.runners.*` | Optimizer execution adapters and baseline wrappers | `custom_es.run`, `isotropic_es.run`, `sep_cma_es.run`, `standard_cma_es.run` | `generalized_es`, `config` | `main` |
| `src.es_model.main` | Unified CLI parser, interactive menu & workflow orchestrator | `main`, `run_model`, `run_benchmarks` | `config`, `generalized_es`, `benchmarks`, `runners` | CLI entrypoints (`es-model`, `generalized-es`) |
| `experiments.experiment_3.runners.run_exp3_baselines` | External heterogeneous baselines runner (RS, CMA-ES with Rounding, CatCMAwM) | `RandomSearchRunner`, `CMARoundingRunner`, `CatCMAwMRunner`, `run_all_baselines` | `benchmarks.generalized_functions`, `config` | `run_experiment_3`, `tests.test_catcmawm_baseline` |
| `experiments.experiment_3.runners.run_experiment_3` | Heterogeneous benchmark orchestration, plotting & Wilcoxon hypothesis testing | `compute_wilcoxon_table_exp3`, `holm_bonferroni`, `main` | `benchmarks.generalized_functions`, `config`, `run_exp3_baselines`, `plotting` | `experiments/experiment_3.ipynb`, CLI |
