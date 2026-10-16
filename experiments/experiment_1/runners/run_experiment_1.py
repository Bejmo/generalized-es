"""
run_experiment_1.py: Execution and statistical analysis routines for Experiment 1.

Evaluates Generalized ES (continuous mode) against canonical continuous baselines:
- Isotropic ES (single scalar self-adaptation)
- sep-CMA-ES (separable diagonal covariance matrix adaptation)
- Full CMA-ES (full rotational covariance matrix adaptation)
across 4 standard continuous benchmarks: Sphere, Rastrigin, Rosenbrock, Ackley.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from src.benchmarks.functions import BENCHMARKS
from src.es_model.config import ESConfig
from src.runners import custom_es, isotropic_es, sep_cma_es, standard_cma_es

# Baseline optimizers registry
RUNNERS = {
    'Prototype 1': custom_es.run,
    'Isotropic ES (Simple Self-Adaptive)': isotropic_es.run,
    'sep-CMA-ES (Separable CMA-ES)': sep_cma_es.run,
    'Standard CMA-ES (Upper Bound)': standard_cma_es.run,
}

CLEAN_NAMES = {
    'Prototype 1': 'Prototype_1',
    'Isotropic ES (Simple Self-Adaptive)': 'Isotropic_ES_Simple_Self_Adaptive',
    'sep-CMA-ES (Separable CMA-ES)': 'sep_CMA_ES_Separable_CMA_ES',
    'Standard CMA-ES (Upper Bound)': 'Standard_CMA_ES_Upper_Bound',
}


def run_all_experiments(
    benchmarks: Dict[str, Any] = BENCHMARKS,
    runners: Dict[str, Any] = RUNNERS,
    dim: int = 10,
    max_evals: int = 10_000,
    seeds: Optional[List[int]] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Execute continuous benchmarks across all runners and random seeds."""
    if seeds is None:
        seeds = list(range(1, 11))

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for bname, binfo in benchmarks.items():
        results[bname] = {}
        b_dim = binfo.get("dim", dim)
        for rname, runner in runners.items():
            traces = []
            for seed in seeds:
                if rname == "Prototype 1":
                    proto_config = ESConfig(
                        n_real=b_dim,
                        n_count=0,
                        n_binary=0,
                        n_categorical=0,
                        n_fuzzy=0,
                        lower_bound_real=float(binfo["bounds"][0]),
                        upper_bound_real=float(binfo["bounds"][1]),
                        seed=seed,
                    )
                    trace = runner(binfo["fn"], b_dim, binfo["bounds"], seed, max_evals, config=proto_config)
                else:
                    trace = runner(binfo["fn"], b_dim, binfo["bounds"], seed, max_evals)
                traces.append(trace)
            results[bname][rname] = np.array(traces)
            median_final = np.median(results[bname][rname][:, -1])
            if verbose:
                print(f"  {bname:12s} | {rname:36s} | median final: {median_final:.4e}")
    return results


def save_experiment_1_traces(
    results: Dict[str, Dict[str, np.ndarray]],
    seeds: List[int],
    output_path: str,
) -> None:
    """Save optimization traces into a compressed .npz archive."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_dict: Dict[str, np.ndarray] = {"seeds": np.array(seeds, dtype=np.int64)}
    for bname, r_dict in results.items():
        for rname, arr in r_dict.items():
            clean = CLEAN_NAMES.get(rname, rname.replace(" ", "_"))
            save_dict[f"{bname}__{clean}"] = arr
    np.savez_compressed(output_path, **save_dict)
    print(f"Results successfully saved to: {output_path}")


def load_experiment_1_traces(
    traces_path: str,
    benchmarks: Dict[str, Any] = BENCHMARKS,
    runners: Dict[str, Any] = RUNNERS,
) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
    """Load traces from .npz archive if it exists."""
    if not os.path.exists(traces_path):
        return None
    loaded = np.load(traces_path)
    results: Dict[str, Dict[str, np.ndarray]] = {}
    for bname in benchmarks.keys():
        results[bname] = {}
        for rname in runners.keys():
            clean = CLEAN_NAMES.get(rname, rname.replace(" ", "_"))
            key = f"{bname}__{clean}"
            if key in loaded:
                results[bname][rname] = loaded[key]
    return results


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


def compute_wilcoxon_table_exp1(
    traces_path_or_results: str | Dict[str, Dict[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    """
    Compute paired two-sided Wilcoxon signed-rank tests for Table 5.1 of the TFG.
    Proposed ES (Prototype 1) vs. [Isotropic ES, sep-CMA-ES, Full CMA-ES]
    across the 4 benchmarks: Sphere, Rastrigin, Rosenbrock, Ackley.
    """
    if isinstance(traces_path_or_results, str):
        data = np.load(traces_path_or_results, allow_pickle=True)
    else:
        data = {}
        for bname, r_dict in traces_path_or_results.items():
            for rname, arr in r_dict.items():
                clean = CLEAN_NAMES.get(rname, rname.replace(" ", "_"))
                data[f"{bname}__{clean}"] = arr

    benchmarks = ["Sphere", "Rastrigin", "Rosenbrock", "Ackley"]
    opponents = [
        ("Isotropic ES", "Isotropic_ES_Simple_Self_Adaptive"),
        ("sep-CMA-ES", "sep_CMA_ES_Separable_CMA_ES"),
        ("Full CMA-ES", "Standard_CMA_ES_Upper_Bound"),
    ]

    all_tests = []
    for bench in benchmarks:
        prop_key = f"{bench}__Prototype_1"
        if prop_key not in data:
            continue
        prop_finals = data[prop_key][:, -1]

        bench_p_vals = []
        bench_entries = []
        for opp_label, opp_suffix in opponents:
            opp_key = f"{bench}__{opp_suffix}"
            opp_finals = data[opp_key][:, -1]

            diff = prop_finals - opp_finals
            n_wins = int(np.sum(diff < 0))
            n_ties = int(np.sum(diff == 0))
            n_losses = int(np.sum(diff > 0))

            if np.all(diff == 0):
                w_stat, p_val = 0.0, 1.0
            else:
                try:
                    res = stats.wilcoxon(prop_finals, opp_finals, alternative="two-sided", zero_method="wilcox")
                    w_stat = float(res.statistic)
                    p_val = float(res.pvalue)
                except Exception:
                    w_stat, p_val = float("nan"), 1.0

            bench_p_vals.append(p_val)
            bench_entries.append({
                "benchmark": bench,
                "opponent": opp_label,
                "wtl": f"{n_wins}/{n_ties}/{n_losses}",
                "statistic": w_stat,
                "p_raw": p_val,
                "prop_mean": float(np.mean(prop_finals)),
                "opp_mean": float(np.mean(opp_finals)),
            })

        p_adj, sig = holm_bonferroni(bench_p_vals)
        for i, entry in enumerate(bench_entries):
            entry["p_adj"] = float(p_adj[i])
            entry["significant"] = bool(sig[i])
            if entry["significant"]:
                entry["decision"] = "Proposed Wins" if entry["prop_mean"] < entry["opp_mean"] else "Opponent Wins"
            else:
                entry["decision"] = "Statistical Tie"
            all_tests.append(entry)

    return all_tests


def main():
    parser = argparse.ArgumentParser(description="Run Experiment 1: Continuous Benchmarks")
    parser.add_argument("--dim", type=int, default=10, help="Dimensionality")
    parser.add_argument("--max-evals", type=int, default=10_000, help="Max evaluations")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--output-dir", type=str, default="experiments/experiment_1/data", help="Output directory")
    args = parser.parse_args()

    seeds = list(range(1, args.n_seeds + 1))
    print(f"Starting Experiment 1: DIM={args.dim}, MAX_EVALS={args.max_evals}, N_SEEDS={args.n_seeds}")
    results = run_all_experiments(BENCHMARKS, RUNNERS, args.dim, args.max_evals, seeds)

    out_file = os.path.join(args.output_dir, "exp1_traces.npz")
    save_experiment_1_traces(results, seeds, out_file)

    tests = compute_wilcoxon_table_exp1(results)
    print("\n--- Statistical Hypothesis Testing (Wilcoxon + Holm-Bonferroni) ---")
    for t in tests:
        print(f"{t['benchmark']:12s} vs {t['opponent']:14s} | W/T/L: {t['wtl']} | W={t['statistic']:.1f} | p_raw={t['p_raw']:.4e} | p_adj={t['p_adj']:.4e} | {t['decision']}")


if __name__ == "__main__":
    main()
