# Extra Experiments: Categorical Operator Validation & Binary Equivalence

This directory isolates the auxiliary empirical validation studies supporting **Appendix E** of the Bachelor's Thesis (*Treball de Fi de Grau*).

## Contents

- `categorical_analysis.ipynb`: Interactive Jupyter Notebook executing, visualizing, and reporting the statistical tests for both studies.
- `run_experiments.py`: Headless CLI runner that executes the full battery of tests, generating persistent results in dual format (`.npz` and `.json`) and figures.
- `results/`: Raw numerical data in dual format:
  - `categorical_comparison_data.npz` / `categorical_comparison_results.json`: 15-seed data across 5 benchmark suites (Legacy Softmax vs. Unified Jump).
  - `binary_vs_categorical_data.npz` / `binary_vs_categorical_k2.json`: 50-seed data comparing Binary Variable (XOR) with Categorical Variable ($K=2$).
- `figures/`: High-resolution figures generated for Appendix E:
  - `categorical_convergence.png`: Multi-panel convergence dynamics across the 5 benchmark landscapes.
  - `binary_vs_cat_k2.png`: Loss distribution boxplot confirming statistical equivalence ($p > 0.05$).

## Reproducibility

To run the full suite from the terminal:
```bash
python experiments/extra/categorical/run_experiments.py
```
Or open and execute `categorical_analysis.ipynb` in your Jupyter environment.
