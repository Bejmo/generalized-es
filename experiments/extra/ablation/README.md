# Factorial Ablation and External Baselines (Appendix D)

This directory isolates the external baselines and operator ablation studies supporting Chapter 5 (Table 5.3) and Appendix D (Table D.1) of the Bachelor's Thesis (*Treball de Fi de Grau*).

## Directory Structure

```
experiments/extra/ablation/
├── ablation_analysis.ipynb          # Interactive Jupyter Notebook visualizing ablation and Wilcoxon tests
├── runners/
│   ├── run_random_search_exp3.py    # Runs Uniform Random Search and Continuous Relaxation CMA-ES with Rounding
│   └── run_factorial_ablation.py    # Runs the 2x2 Factorial Ablation (Circular Kernel x Scale Recombination)
└── results/
    ├── random_search_results.json   # 10-seed results for Random Search & CMA-ES (Rounding)
    ├── factorial_ablation_results.json # 10-seed results for Skellam_Arithmetic & ForcedStep_Geometric
    ├── ablation_history.json        # Step progression logs
    ├── step2_convergence.png        # Ablation convergence plots
    └── step3_convergence_log.png    # Log-scale convergence progression
```

## Reproducibility Commands

To execute and reproduce the external baselines:
```bash
python experiments/extra/ablation/runners/run_random_search_exp3.py
```

To execute and reproduce the 2x2 factorial ablation:
```bash
python experiments/extra/ablation/runners/run_factorial_ablation.py
```

Or open and run `ablation_analysis.ipynb` in your Jupyter environment.
