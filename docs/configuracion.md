# Heterogeneous Evolution Strategies for Mixed-Variable Optimization

## Configuration and Hyperparameters

## Configuration File: `src/es_model/config.py`

All ES parameters are centralized in a single module. Other modules import directly from `config`, which allows behavior changes without touching the algorithmic logic.

---

## Population Dynamics Parameters

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MU` | int | 15 | Parent population size (survivors) |
| `LAMBDA` | int | 100 | Number of child per generation |
| `SELECTION_MODE` | str | `"comma"` | Selection policy: `"comma"` or `"plus"` |
| `MAX_GENERATIONS` | int | 500 | Generation-based stopping criterion |

### μ/λ Ratio

The ratio μ/λ = 15/100 = 0.15 provides strong selective pressure. Typical values in the literature are between 1/5 and 1/7.

### Selection Mode

- **Comma (μ, λ):** Higher evolutionary pressure. Parents die each generation, reducing stagnation risk. Requires λ > μ.
- **Plus (μ + λ):** More conservative. Parents compete with child. Higher risk of local stagnation but guarantees monotonic best fitness.

---

## Genome Dimensions

| Variable | Default | Description |
|----------|---------|-------------|
| `N_REAL` | 10 | Continuous real-valued variables |
| `N_COUNT` | 5 | Integer variables |
| `N_BINARY` | 8 | Binary variables |
| `N_CATEGORICAL` | 3 | Categorical variables |
| `N_FUZZY` | 4 | Fuzzy variables (triplets) |
| `N_TOTAL` | 38 | Total dimensionality (auto-computed) |

`N_TOTAL = N_REAL + N_COUNT + N_BINARY + N_CATEGORICAL + N_FUZZY × 3`

Fuzzy triplets count ×3 because each variable has three control points (a, b, c).

---

## Bounds

### Real Variables

```python
REAL_BOUNDS = [[-5.0, 5.0]] × N_REAL   # shape (N_REAL, 2)
```

### Integer Variables

```python
COUNT_BOUNDS = [[0, 20]] × N_COUNT      # shape (N_COUNT, 2)
```

### Categorical Variables

```python
CATEGORICAL_K = [5, 4, 3]  # Cardinalities per variable
```

---

## Self-Adaptation

| Variable | Default | Description |
|----------|---------|-------------|
| `SIGMA_FLOOR` | 1e-6 | Minimum floor for sigma (prevents premature convergence) |
| `INIT_SIGMA_REAL` | 0.5 | Initial sigma for the real block |
| `INIT_LAMBDA_COUNT` | 1.0 | Initial Poisson rate for the count block |
| `INIT_SIGMA_FUZZY` | 0.05 | Initial sigma for fuzzy control points |
| `INIT_LOGIT_BINARY` | 0.0 | Initial logit for flip probability (-> p=0.5) |
| `INIT_LOGIT_CAT` | 0.0 | Initial logit for categorical mutation probability (-> p=0.5) |

### Derived Learning Rates

```
tau' = 1/sqrt(2·N_TOTAL) ≈ 0.115   (for N_TOTAL=38)
tau  = 1/sqrt(2·sqrt(N_TOTAL)) ≈ 0.284  (for N_TOTAL=38)
```

They are automatically recomputed when `N_TOTAL` changes.

---

## Runtime

| Variable | Default | Description |
|----------|---------|-------------|
| `SEED` | 42 | Deterministic seed for reproducibility |
| `ENABLE_TRACE_LOG` | False | Logging of full genome vectors |
| `LOG_INTERVAL` | 10 | Print progress every N generations |

---

## CLI Customization

The most common parameters can be overridden from the command line without modifying `config.py`:

```bash
# Change generations
python -m src.es_model.main model --generations 1000

# Change log interval
python -m src.es_model.main model --log-interval 50
```

For deeper changes (μ, λ, bounds, and so on), edit `src/es_model/config.py` directly.

---

## Tuning Considerations

| Scenario | Recommendation |
|-----------|---------------|
| Multimodal problems | Use `"comma"` mode, increase λ |
| Large search space | Increase `INIT_SIGMA_REAL` |
| Dominant discrete variables | Shift `INIT_LOGIT_*` toward negative values (less mutation) |
