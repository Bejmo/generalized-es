"""
run_experiment_2.py: CLI runner for Experiment 2 (Target Shift - TS).
Executes TS tracking protocol across all 6 heterogeneous variable types.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path for direct script invocation
for parent in [Path.cwd().resolve()] + list(Path(__file__).resolve().parents):
    if (parent / "src").is_dir():
        if str(parent) not in sys.path:
            sys.path.insert(0, str(parent))
        break


import argparse
import os
from pathlib import Path
import numpy as np

from experiments.experiment_2.runners.runners import (
    run_dts_experiment,
    compute_dts_summary_metrics,
    save_dts_results,
    load_dts_results,
)
from experiments.experiment_2.runners.plotting import (
    plot_dts_fitness,
    plot_dts_strategy,
    plot_dts_individual_summaries,
    plot_dts_unified_grid,
)


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 2: Target Shift Experiment")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of random seeds")
    parser.add_argument("--n-gens", type=int, default=1000, help="Number of generations")
    parser.add_argument("--g-shift", type=int, default=500, help="Generation of target shock")
    parser.add_argument("--output-dir", type=str, default="experiments/experiment_2", help="Experiment root directory")
    args = parser.parse_args()

    seeds = list(range(1, args.n_seeds + 1))
    data_dir = os.path.join(args.output_dir, "data")
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Starting Experiment 2: {args.n_seeds} seeds, {args.n_gens} gens, shift at {args.g_shift}")

    results = {}
    # Real
    results["real"] = run_dts_experiment("real", 10, +3.0, -3.0, seeds, args.n_gens, args.g_shift, bounds=(-5.0, 5.0))
    # Count
    results["count"] = run_dts_experiment("count", 10, 15, 5, seeds, args.n_gens, args.g_shift, bounds=(0, 20))
    # Binary
    rng_bin = np.random.default_rng(12345)
    t1_b = rng_bin.integers(0, 2, size=32, dtype=np.uint8)
    t2_b = 1 - t1_b
    results["binary"] = run_dts_experiment("binary", 32, t1_b, t2_b, seeds, args.n_gens, args.g_shift)
    # Categorical
    results["categorical"] = run_dts_experiment("categorical", 5, 0, 4, seeds, args.n_gens, args.g_shift, categorical_k=np.full(5, 5, dtype=np.int64))
    # Fuzzy
    t1_f = np.tile([0.2, 0.1], 4)
    t2_f = np.tile([0.8, 0.1], 4)
    results["fuzzy"] = run_dts_experiment("fuzzy", 4, t1_f, t2_f, seeds, args.n_gens, args.g_shift)
    # Circular
    results["circular"] = run_dts_experiment("circular", 6, 0, 6, seeds, args.n_gens, args.g_shift, circular_k=np.full(6, 12, dtype=np.int64))

    # Save summary table and raw traces
    save_dts_results(results, data_dir)
    metrics_res = compute_dts_summary_metrics(list(results.values()), g_shift=args.g_shift, n_gens=args.n_gens)
    csv_path = os.path.join(data_dir, "dts_summary_table.csv")
    if hasattr(metrics_res, "to_csv"):
        metrics_res.to_csv(csv_path, index=False)
    else:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metrics_res[0].keys())
            writer.writeheader()
            writer.writerows(metrics_res)
    print(f"Summary table saved to: {csv_path}")

    # Generate individual plots
    plot_dts_individual_summaries(results, g_shift=args.g_shift, save_dir=fig_dir, show=False)
    print(f"All figures saved to: {fig_dir}")


if __name__ == "__main__":
    main()
