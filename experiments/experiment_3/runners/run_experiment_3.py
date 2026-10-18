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
    load_heterogeneous_benchmark_suite,
    run_heterogeneous_dynamics_tracking,
    load_heterogeneous_dynamics_tracking,
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
    Computes Wilcoxon signed-rank tests for Experiment 3.
    Generalized ES vs [Uniform Random Search, Continuous CMA-ES Rounding, CatCMAwM (2025)]
    across 4 heterogeneous benchmarks: Sphere, Coupled, Rastrigin, Multiplicative.

    Enforces strict seed-to-result dictionary matching and global Holm-Bonferroni
    family-wise correction over all m=12 comparisons.
    """
    data3 = np.load(exp3_traces_path, allow_pickle=True)
    with open(baseline_json_path, "r", encoding="utf-8") as f:
        base_data = json.load(f)

    benchmarks = [
        ("Generalized_Sphere", "sphere", "Heterogeneous Sphere"),
        ("Generalized_Coupled", "coupled", "Heterogeneous Coupled"),
        ("Generalized_Rastrigin", "rastrigin", "Heterogeneous Rastrigin"),
        ("Generalized_Multiplicative", "multiplicative", "Heterogeneous Multiplicative"),
    ]

    if "seeds" in data3:
        seeds = [int(s) for s in data3["seeds"]]
    else:
        seeds = [int(s) for s in base_data["random_search"]["sphere"]["per_seed"].keys()]

    all_comparison_records: List[Dict[str, Any]] = []

    for bench_key, json_key, bench_name in benchmarks:
        if f"{bench_key}__traces" not in data3:
            continue
        prop_traces = data3[f"{bench_key}__traces"]
        prop_final = prop_traces[:, -1]
        prop_dict = {int(s): float(prop_final[i]) for i, s in enumerate(seeds)}

        opponents = [
            ("Random Search", base_data.get("random_search", {}).get(json_key, {}).get("per_seed", {})),
            ("CMA-ES Rounding", base_data.get("cma_es_relaxation", {}).get(json_key, {}).get("per_seed", {})),
            ("CatCMAwM (2025)", base_data.get("catcmawm", {}).get(json_key, {}).get("per_seed", {})),
        ]

        for opp_label, raw_opp_dict in opponents:
            if not raw_opp_dict:
                continue
            opp_dict = {int(s): float(v) for s, v in raw_opp_dict.items()}
            # Strict seed pairing via dictionary lookup
            shared_seeds = [s for s in seeds if s in prop_dict and s in opp_dict]
            if len(shared_seeds) < 3:
                continue

            p_sub = np.array([prop_dict[s] for s in shared_seeds], dtype=np.float64)
            o_sub = np.array([opp_dict[s] for s in shared_seeds], dtype=np.float64)
            diff = p_sub - o_sub

            wins = int(np.sum(diff < 0))
            ties = int(np.sum(diff == 0))
            losses = int(np.sum(diff > 0))

            if np.all(diff == 0):
                w_stat, p_val = 0.0, 1.0
                r_biserial = 0.0
            else:
                try:
                    res = stats.wilcoxon(p_sub, o_sub, alternative="two-sided")
                    w_stat = float(res.statistic)
                    p_val = float(res.pvalue)
                except Exception:
                    w_stat, p_val = float("nan"), 1.0

                # Matched-pairs rank-biserial correlation r = (W_minus - W_plus) / total_ranks
                non_zero = diff != 0
                d_nz = diff[non_zero]
                if len(d_nz) > 0:
                    ranks = stats.rankdata(np.abs(d_nz))
                    w_plus = np.sum(ranks[d_nz > 0])   # opponent wins ranks
                    w_minus = np.sum(ranks[d_nz < 0])  # generalized es wins ranks
                    total_s = np.sum(ranks)
                    r_biserial = float((w_minus - w_plus) / total_s) if total_s > 0 else 0.0
                else:
                    r_biserial = 0.0

            all_comparison_records.append({
                "benchmark": bench_name,
                "opponent": opp_label,
                "n_seeds": len(shared_seeds),
                "prop_median": float(np.median(p_sub)),
                "opp_median": float(np.median(o_sub)),
                "prop_mean": float(np.mean(p_sub)),
                "opp_mean": float(np.mean(o_sub)),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "w_stat": w_stat,
                "r_biserial": r_biserial,
                "raw_p": p_val,
            })

    # Global Holm-Bonferroni correction over all m comparisons across benchmarks
    raw_p_list = [rec["raw_p"] for rec in all_comparison_records]
    adj_p_list, sig_list = holm_bonferroni(raw_p_list)

    results: Dict[str, List[Dict[str, Any]]] = {}
    for i, rec in enumerate(all_comparison_records):
        rec["adj_p"] = float(adj_p_list[i])
        rec["significant"] = bool(sig_list[i])
        if sig_list[i]:
            rec["decision"] = "Generalized ES Wins" if rec["r_biserial"] > 0 else "Baseline Wins"
        else:
            rec["decision"] = "Statistical Tie"

        b_name = rec["benchmark"]
        if b_name not in results:
            results[b_name] = []
        results[b_name].append(rec)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 3: Heterogeneous Benchmarks")
    parser.add_argument("--max-evals", type=int, default=20000)
    parser.add_argument("--n-seeds", type=int, default=40)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="experiments/experiment_3")
    parser.add_argument("--force-all", action="store_true", help="Force recomputation of all benchmarks even if present in cache.")
    parser.add_argument("--load-precomputed", action="store_true", help="Load precomputed traces and dynamics from data/ instead of re-running simulations.")
    args = parser.parse_args()

    seeds = [42 + i * 101 for i in range(args.n_seeds)]
    data_dir = os.path.join(args.output_dir, "data")
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    config = ESConfig(
        mu=15,
        lam=100,
        n_real=5,
        n_count=3,
        n_binary=4,
        n_categorical=2,
        categorical_k=np.array([3, 4], dtype=np.int64),
        n_fuzzy=2,
        n_circular=2,
        circular_k=np.array([7, 12], dtype=np.int64),
        seed=42,
    )

    traces_file = os.path.join(data_dir, "exp3_traces.npz")
    dynamics_file = os.path.join(data_dir, "exp3_strategy_dynamics.npz")

    if args.load_precomputed:
        print(f"Loading precomputed benchmark suite and dynamics from {data_dir}...")
        results = load_heterogeneous_benchmark_suite(data_dir, GENERALIZED_BENCHMARKS)
        if results is None:
            raise FileNotFoundError(f"Missing traces at {traces_file}")
        all_saved_data, bench_dynamics = load_heterogeneous_dynamics_tracking(data_dir, GENERALIZED_BENCHMARKS)
    else:
        # Determine which benchmarks to run
        benchmarks_to_run_traces = dict(GENERALIZED_BENCHMARKS)
        benchmarks_to_run_dynamics = dict(GENERALIZED_BENCHMARKS)

        if not args.force_all and os.path.exists(traces_file):
            existing_traces = np.load(traces_file, allow_pickle=True)
            for b_name in list(benchmarks_to_run_traces.keys()):
                key = f"{b_name}__traces"
                if key in existing_traces and existing_traces[key].shape[0] >= args.n_seeds:
                    print(f"Skipping {b_name} traces - already present in {traces_file}.")
                    del benchmarks_to_run_traces[b_name]

        if not args.force_all and os.path.exists(dynamics_file):
            existing_dyn = np.load(dynamics_file, allow_pickle=True)
            for b_name in list(benchmarks_to_run_dynamics.keys()):
                key = f"{b_name}__cost_real_mean"
                if key in existing_dyn:
                    print(f"Skipping {b_name} dynamics - already present in {dynamics_file}.")
                    del benchmarks_to_run_dynamics[b_name]

        if benchmarks_to_run_traces:
            print(f"Running heterogeneous benchmark suite for {list(benchmarks_to_run_traces.keys())} ({args.n_seeds} seeds, max_evals={args.max_evals})...")
            run_heterogeneous_benchmark_suite(
                benchmarks=benchmarks_to_run_traces,
                config=config,
                max_evals=args.max_evals,
                seeds=seeds,
                data_dir=data_dir,
                max_workers=args.max_workers,
            )

        if benchmarks_to_run_dynamics:
            print(f"\nRunning heterogeneous strategy dynamics tracking for {list(benchmarks_to_run_dynamics.keys())} ({args.n_seeds} seeds)...")
            run_heterogeneous_dynamics_tracking(
                benchmarks=benchmarks_to_run_dynamics,
                config=config,
                seeds=seeds,
                max_evals=args.max_evals,
                data_dir=data_dir,
                max_workers=args.max_workers,
            )

        # Load complete benchmark suite and dynamics for all benchmarks to plot
        results = load_heterogeneous_benchmark_suite(data_dir, GENERALIZED_BENCHMARKS)
        all_saved_data, bench_dynamics = load_heterogeneous_dynamics_tracking(data_dir, GENERALIZED_BENCHMARKS)

    # Plot convergence curves comparing all 4 methods
    baselines_npz_path = os.path.join(data_dir, "random_search_baselines.npz")
    plot_heterogeneous_convergence(
        results=results,
        max_evals=args.max_evals,
        config=config,
        n_seeds=args.n_seeds,
        save_dir=fig_dir,
        baselines_path=baselines_npz_path,
        show=False,
    )

    for bench_name, b_agg in bench_dynamics.items():
        plot_heterogeneous_dynamics(
            bench_name=bench_name,
            bench_agg=b_agg,
            n_generations=args.max_evals // config.lam,
            n_seeds=args.n_seeds,
            save_dir=fig_dir,
            show=False,
        )

    print(f"\nAll results and figures saved to {args.output_dir}")

    # Print Wilcoxon tests if baselines exist
    baseline_file = os.path.join(data_dir, "random_search_results.json")
    traces_file = os.path.join(data_dir, "exp3_traces.npz")
    if os.path.exists(baseline_file) and os.path.exists(traces_file):
        tests_exp3 = compute_wilcoxon_table_exp3(traces_file, baseline_file)
        m_tests = sum(len(records) for records in tests_exp3.values())
        print(f"\n" + "=" * 115)
        print(f" STATISTICAL HYPOTHESIS TESTING VS. EXTERNAL BASELINES ({args.n_seeds} SEEDS, GLOBAL HOLM m={m_tests})")
        print("=" * 115)
        print(f"{'Benchmark':<24} | {'Opponent':<18} | {'W/T/L':<8} | {'W-stat':<8} | {'r_rb':<7} | {'p_raw':<10} | {'p_adj':<10} | {'Decision'}")
        print("-" * 115)
        for bench_name, records in tests_exp3.items():
            for r in records:
                wtl = f"{r['wins']}/{r['ties']}/{r['losses']}"
                w_str = f"{r['w_stat']:.1f}" if not np.isnan(r['w_stat']) else "NaN"
                r_str = f"{r['r_biserial']:+.3f}"
                p_r = f"{r['raw_p']:.3e}"
                p_a = f"{r['adj_p']:.3e}"
                print(f"{bench_name:<24} | {r['opponent']:<18} | {wtl:<8} | {w_str:<8} | {r_str:<7} | {p_r:<10} | {p_a:<10} | {r['decision']}")
        print("-" * 115)


if __name__ == "__main__":
    main()
