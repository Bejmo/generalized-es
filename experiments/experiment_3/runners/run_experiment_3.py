"""
run_experiment_3.py: Execution and hypothesis testing for Experiment 3 (Heterogeneous Benchmarks).
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
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
from scipy import stats

from src.benchmarks.generalized_functions import GENERALIZED_BENCHMARKS
from src.es_model.config import ESConfig
from experiments.experiment_3.runners.runners import (
    run_heterogeneous_benchmark_suite,
    run_heterogeneous_dynamics_tracking,
    build_heterogeneous_summary_table,
)
from experiments.experiment_3.runners.plotting import (
    plot_heterogeneous_convergence,
    plot_heterogeneous_dynamics,
)


def holm_bonferroni(p_values: np.ndarray | List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Step-down Holm-Bonferroni correction at alpha=0.05."""
    p_vals = np.asarray(p_values, dtype=float)
    m = len(p_vals)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)
    sort_idx = np.argsort(p_vals)
    inv_idx = np.empty(m, dtype=int)
    inv_idx[sort_idx] = np.arange(m)
    sorted_p = p_vals[sort_idx]
    factors = np.arange(m, 0, -1)
    adjusted_sorted = np.clip(np.maximum.accumulate(sorted_p * factors), 0.0, 1.0)
    adjusted_p = adjusted_sorted[inv_idx]
    return adjusted_p, adjusted_p < 0.05


def compute_wilcoxon_table_exp3(
    exp3_traces_path: str,
    baseline_json_path: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Computes Wilcoxon signed-rank tests for Experiment 3 (Table 5.3).
    Generalized ES vs [Uniform Random Search, Continuous CMA-ES Rounding]
    across 3 heterogeneous benchmarks: Sphere, Coupled, Rastrigin.
    """
    data3 = np.load(exp3_traces_path, allow_pickle=True)
    with open(baseline_json_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    benchmarks = [
        ("Generalized_Sphere", "sphere", "Heterogeneous Sphere"),
        ("Generalized_Coupled", "coupled", "Heterogeneous Coupled"),
        ("Generalized_Rastrigin", "rastrigin", "Heterogeneous Rastrigin"),
    ]

    seeds = [42, 143, 244, 345, 446, 547, 648, 749, 850, 951]
    results = {}

    for bench_key, json_key, bench_name in benchmarks:
        prop_traces = data3[f"{bench_key}__traces"]
        prop_final = prop_traces[:, -1]

        rs_dict = base_data["random_search"][json_key]["per_seed"]
        cma_dict = base_data["cma_es_relaxation"][json_key]["per_seed"]

        rs_final = np.array([rs_dict[str(s)] for s in seeds])
        cma_final = np.array([cma_dict[str(s)] for s in seeds])

        opponents = [
            ("Random Search", rs_final),
            ("CMA-ES Rounding", cma_final),
        ]

        bench_records = []
        raw_p_list = []

        for opp_label, opp_final in opponents:
            diff = prop_final - opp_final
            wins = int(np.sum(diff < 0))
            losses = int(np.sum(diff > 0))
            ties = int(np.sum(diff == 0))

            if np.all(diff == 0):
                w_stat, p_val = 0.0, 1.0
            else:
                try:
                    res = stats.wilcoxon(prop_final, opp_final, alternative="two-sided")
                    w_stat = float(res.statistic)
                    p_val = float(res.pvalue)
                except Exception:
                    w_stat, p_val = float("nan"), 1.0

            raw_p_list.append(p_val)
            bench_records.append({
                "benchmark": bench_name,
                "opponent": opp_label,
                "prop_mean": float(np.mean(prop_final)),
                "opp_mean": float(np.mean(opp_final)),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "w_stat": w_stat,
                "raw_p": p_val,
            })

        adj_p, sig = holm_bonferroni(raw_p_list)
        for i, rec in enumerate(bench_records):
            rec["adj_p"] = float(adj_p[i])
            rec["significant"] = bool(sig[i])
            if sig[i]:
                rec["decision"] = "Generalized ES Wins" if rec["wins"] > rec["losses"] else "Baseline Wins"
            else:
                rec["decision"] = "Statistical Tie"

        results[bench_name] = bench_records

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 3: Heterogeneous Benchmarks")
    parser.add_argument("--max-evals", type=int, default=20000)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="experiments/experiment_3")
    args = parser.parse_args()

    seeds = [42 + i * 101 for i in range(args.n_seeds)]
    data_dir = os.path.join(args.output_dir, "data")
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    config = ESConfig(
        n_real=10,
        n_count=5,
        n_binary=16,
        n_categorical=3,
        n_fuzzy=2,
        n_circular=2,
        categorical_k=np.array([4, 6, 8], dtype=np.int64),
        circular_k=np.array([12, 24], dtype=np.int64),
    )

    print("Running heterogeneous benchmark suite...")
    results = run_heterogeneous_benchmark_suite(
        benchmarks=GENERALIZED_BENCHMARKS,
        config=config,
        max_evals=args.max_evals,
        seeds=seeds,
        data_dir=data_dir,
    )

    plot_heterogeneous_convergence(results=results, save_dir=fig_dir)
    print(f"Results and figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
