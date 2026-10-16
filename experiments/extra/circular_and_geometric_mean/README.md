# Factorial Study: Circular Mutation & Scale Parameter Recombination (Appendix D)

This directory isolates the auxiliary empirical study evaluating the $2 \times 2$ factorial combinations of circular mutation operators and scale parameter recombination schemes, directly supporting **Appendix D** (*Operator Ablation and Architectural Progression*) of the Bachelor's Thesis (*Treball de Fi de Grau*).

## Evaluated Architectural Variants

1. **Circular Mutation Kernel:**
   - *Forced-Step:* Discrete step restricted to $\pm 1 \pmod K$ without zero-step inertia ($\Delta x \neq 0$).
   - *Modular Skellam:* Symmetric difference of Poissons with non-zero probability of remaining stationary ($\Delta x = 0$).
2. **Scale Parameter Recombination:**
   - *Arithmetic:* Standard linear mean $(\sigma_1 + \sigma_2) / 2$ (subject to Jensen's inflation bias).
   - *Geometric:* Multiplicative root mean $\sqrt{\sigma_1 \cdot \sigma_2}$ (scale-invariant in log-normal space).

## Directory Structure

```
experiments/extra/circular_and_geometric_mean/
├── circular_and_geometric_mean_analysis.ipynb  # Interactive Jupyter Notebook visualizing metrics and plots
├── runners/
│   └── run_factorial_study.py                  # CLI runner executing the 2x2 factorial study
└── results/
    ├── factorial_study_data.npz                # Compact NumPy binary archive with per-seed losses
    ├── factorial_study_results.json            # Structured statistical summary (min, mean, std, max)
    ├── ablation_history.json                   # Architectural progression history
    ├── step2_convergence.png                   # Linear-scale convergence trajectories
    └── step3_convergence_log.png               # Log-scale convergence trajectories (7-order improvement)
```

## Reproducibility

To re-run the study from scratch:
```bash
python experiments/extra/circular_and_geometric_mean/runners/run_factorial_study.py
```
Or open and execute `circular_and_geometric_mean_analysis.ipynb` in your Jupyter environment.
