"""
runners.py: Execution and orchestration routines for Experiments 2 and 3.

Centralizes simulation loops, hyperparameter configuration, and trace collection for:
1. Target Shift (TS) across the 6 variable types (Real, Count, Binary, Categorical, Fuzzy, Circular).
2. Domain invariant and singularity stress tests (INV-R to INV-S).
3. Architectural ablation studies (Binary Ceiling and Categorical Gauge Drift).
4. Experiment 3 synthetic mixed-variable benchmark suites and heterogeneous dynamics tracking.
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


import os
import time
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None

from src.es_model.config import ESConfig
from src.es_model.individual import Individual, unpack_genome
from src.es_model.generalized_es import GeneralizedES
from src.runners.custom_es import run as run_generalized_es
from src.benchmarks.generalized_functions import get_default_targets
from experiments.experiment_2.runners.isolation_landscapes import (
    DTSLandscape,
    real_boundary_target,
    integer_staircase,
    binary_hamming,
    categorical_deceptive,
    fuzzy_jaccard,
    circular_geodesic,
)
from experiments.experiment_2.runners.invariant_monitors import (
    check_real_bounds,
    check_count_bounds,
    check_binary_validity,
    check_categorical_validity,
    check_gauge_residual,
    compute_softmax_entropy,
    check_fuzzy_monotonicity,
    compute_support_width,
    check_circular_range,
    compute_logit_norm,
)


def run_dts_experiment(
    var_type: str,
    n_vars: int,
    t1: Any,
    t2: Any,
    seeds: List[int],
    n_gens: int = 1000,
    g_shift: int = 500,
    bounds: Tuple[float, float] | None = None,
    categorical_k: List[int] | np.ndarray | None = None,
    circular_k: List[int] | np.ndarray | None = None,
    mu: int = 15,
    lam: int = 100,
) -> Dict[str, Any]:
    """
    Execute the Target Shift (TS) protocol on a specific variable type in isolation.

    Parameters
    ----------
    var_type : str
        Variable type under test ('real', 'count', 'binary', 'categorical', 'fuzzy', 'circular').
    n_vars : int
        Number of active variables of this type in the genome (other 5 types set to dimension 0).
    t1 : Any
        Initial target (Phase A: gen < g_shift).
    t2 : Any
        Displaced target (Phase C: gen >= g_shift).
    seeds : List[int]
        Random seeds for statistical replication.
    n_gens : int, default=1000
        Total number of generations.
    g_shift : int, default=500
        Generation at which the target shifts instantaneously from t1 to t2.
    bounds : Tuple[float, float], optional
        Domain bounds (lower, upper) for real or count variables.
    categorical_k : array-like, optional
        Cardinality per categorical dimension.
    circular_k : array-like, optional
        Cardinality per circular dimension.
    mu : int, default=15
        Parent population size.
    lam : int, default=100
        Offspring population size.

    Returns
    -------
    dict with NumPy arrays of traces (fitness, adaptive strategy parameter, and invariants).
    """
    print(f"Running DTS for variable '{var_type}' ({len(seeds)} seeds, {n_gens} generations)...")
    t0 = time.time()

    fitness_runs: List[List[float]] = []
    strat_runs: List[List[float]] = []
    inv_runs: List[List[int | float]] = []

    for s_idx, seed in enumerate(seeds):
        # 1. Configure the evolutionary engine, isolating the active variable type
        cfg_kwargs: Dict[str, Any] = {
            "mu": mu,
            "lam": lam,
            "max_generations": n_gens,
            "seed": seed,
            "n_real": n_vars if var_type == "real" else 0,
            "n_count": n_vars if var_type == "count" else 0,
            "n_binary": n_vars if var_type == "binary" else 0,
            "n_categorical": n_vars if var_type == "categorical" else 0,
            "n_fuzzy": n_vars if var_type == "fuzzy" else 0,
            "n_circular": n_vars if var_type == "circular" else 0,
            "categorical_k": np.asarray(categorical_k, dtype=np.int64) if categorical_k is not None else [],
            "circular_k": np.asarray(circular_k, dtype=np.int64) if circular_k is not None else [],
        }

        if var_type == "real" and bounds is not None:
            cfg_kwargs["lower_bound_real"], cfg_kwargs["upper_bound_real"] = bounds
        elif var_type == "count" and bounds is not None:
            cfg_kwargs["lower_bound_count"], cfg_kwargs["upper_bound_count"] = int(bounds[0]), int(bounds[1])

        cfg = ESConfig(**cfg_kwargs)

        # 2. Instantiate dynamic DTS landscape objective and ES optimizer
        dts_landscape = DTSLandscape(cfg, var_type=var_type, t1=t1, t2=t2, g_shift=g_shift)
        es = GeneralizedES(config=cfg, seed=seed)
        es.init_population(dts_landscape)

        f_trace: List[float] = []
        strat_trace: List[float] = []
        inv_trace: List[int | float] = []

        # 3. Generational loop
        for g in range(n_gens):
            dts_landscape.set_generation(g)
            es.step(dts_landscape)
            best_p = es.parents[0]

            f_trace.append(float(best_p.fitness))

            # Extract the corresponding adaptive strategy parameter
            if var_type == "real":
                strat_trace.append(float(np.median(best_p.sigma_real)))
                inv_trace.append(check_real_bounds(best_p, cfg))
            elif var_type == "count":
                strat_trace.append(float(np.median(best_p.lambda_count)))
                inv_trace.append(check_count_bounds(best_p, cfg))
            elif var_type == "binary":
                strat_trace.append(float(np.median(best_p.logit_binary)))
                inv_trace.append(check_binary_validity(best_p))
            elif var_type == "categorical":
                strat_trace.append(float(np.median(best_p.logit_cat)))
                inv_trace.append(check_categorical_validity(best_p, cfg))
            elif var_type == "fuzzy":
                strat_trace.append(float(np.median(best_p.sigma_fuzzy)))
                inv_trace.append(check_fuzzy_monotonicity(best_p, cfg))
            elif var_type == "circular":
                strat_trace.append(float(np.median(best_p.lambda_circular)))
                inv_trace.append(check_circular_range(best_p, cfg))

        fitness_runs.append(f_trace)
        strat_runs.append(strat_trace)
        inv_runs.append(inv_trace)

        # Periodic progress logging
        if (s_idx + 1) % max(1, len(seeds) // 5) == 0:
            print(f"  Seed {seed:5d} ({s_idx + 1:2d}/{len(seeds)}) | "
                  f"Phase A min fit: {min(f_trace[:g_shift]):.2e} | "
                  f"Post-shift final fit: {f_trace[-1]:.2e}")

    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.2f}s.\n")

    return {
        "var_type": var_type,
        "fitness": np.array(fitness_runs, dtype=np.float64),
        "strategy": np.array(strat_runs, dtype=np.float64),
        "invariants": np.array(inv_runs, dtype=np.float64),
        "seeds": seeds,
        "n_gens": n_gens,
        "g_shift": g_shift,
    }


def compute_dts_summary_metrics(results_list: List[Dict[str, Any]], g_shift: int, n_gens: int) -> pd.DataFrame:
    """
    Compute standard dynamic benchmark metrics for the Target Shift (TS) protocol:
    - Shock Severity (Delta f): Median fitness immediately after the target displacement.
    - Recovery Time (G_rec): Median generations post-shift to attain resolution threshold
      (0.0 for discrete/combinatorial domains, 1e-4 for continuous/fuzzy spaces).
    - Final Fitness: Median best residual fitness at the final generation (G=n_gens-1).
    - Speed Ratio (SR_C/A): Ratio of convergence velocity post-shock vs cold start.
    """
    label_map = {
        "real": "Continuous (R)",
        "count": "Count (Z)",
        "binary": "Binary (B)",
        "categorical": "Categorical (C)",
        "fuzzy": "Fuzzy (F)",
        "circular": "Circular (S1)",
    }

    metrics: List[Dict[str, Any]] = []
    for res in results_list:
        vtype = res["var_type"]
        fit_arr = res["fitness"]  # shape: (n_seeds, n_gens)

        # 1. Shock Severity (Delta f): median fitness immediately following the shock at g_shift
        post_shift_init = fit_arr[:, g_shift]
        shock_severity = float(np.median(post_shift_init))

        # 2. Recovery Time (G_rec): median generations post-shock to reach resolution threshold
        threshold = 1e-4 if vtype in ("real", "fuzzy") else 0.0
        rec_gens_list: List[int] = []
        for s in range(fit_arr.shape[0]):
            post_trace = fit_arr[s, g_shift:]
            rec_idx = np.where(post_trace <= threshold)[0]
            if len(rec_idx) > 0:
                rec_gens_list.append(int(rec_idx[0]))
            else:
                rec_gens_list.append(int(len(post_trace)))
        recovery_time = int(np.median(rec_gens_list))

        # 3. Final Fitness at the last generation
        final_fit = float(np.median(fit_arr[:, -1]))

        # 4. Relative convergence speed ratio (Phase C slope vs Phase A slope)
        fit_a_start = np.median(fit_arr[:, 0])
        fit_a_end = np.median(fit_arr[:, g_shift - 1])
        slope_a = (np.log10(max(1e-12, fit_a_start)) - np.log10(max(1e-12, fit_a_end))) / g_shift

        fit_c_start = np.median(fit_arr[:, g_shift])
        fit_c_end = np.median(fit_arr[:, -1])
        slope_c = (np.log10(max(1e-12, fit_c_start)) - np.log10(max(1e-12, fit_c_end))) / (n_gens - g_shift)
        speed_ratio = float(slope_c / max(1e-6, slope_a))

        metrics.append({
            "Variable Domain": label_map.get(vtype, vtype),
            "Shock Severity (Delta f)": f"{shock_severity:.2f}" if shock_severity >= 1.0 else f"{shock_severity:.2e}",
            "Recovery Time (G_rec)": f"{recovery_time:d}",
            "Final Fitness": f"{final_fit:.2e}" if final_fit > 0.0 else "0.00",
            "Speed Ratio (SR_C/A)": f"{speed_ratio:.2f}",
        })

    if pd is not None:
        return pd.DataFrame(metrics)
    return metrics


# =============================================================================
# Experiment 3: Heterogeneous Benchmark Suite & Dynamics Tracking
# =============================================================================


def save_dts_results(results: Dict[str, Dict[str, Any]], data_dir: str) -> str:
    """Save multi-variable DTS traces into a single compressed .npz archive."""
    os.makedirs(data_dir, exist_ok=True)
    save_dict: Dict[str, Any] = {}
    for var_type, res in results.items():
        save_dict[f"{var_type}__fitness"] = res["fitness"]
        save_dict[f"{var_type}__strategy"] = res["strategy"]
        save_dict[f"{var_type}__invariants"] = res["invariants"]
        save_dict[f"{var_type}__seeds"] = np.array(res["seeds"])
        save_dict[f"{var_type}__meta"] = np.array([res["n_gens"], res["g_shift"]])
    out_path = os.path.join(data_dir, "dts_traces.npz")
    np.savez_compressed(out_path, **save_dict)
    print(f"DTS traces successfully saved to: {out_path}")
    return out_path


def load_dts_results(data_dir: str) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load precomputed DTS traces from data_dir/dts_traces.npz."""
    in_path = os.path.join(data_dir, "dts_traces.npz")
    if not os.path.exists(in_path):
        return None
    loaded = np.load(in_path, allow_pickle=True)
    results: Dict[str, Dict[str, Any]] = {}
    var_types = ["real", "count", "binary", "categorical", "fuzzy", "circular"]
    for vt in var_types:
        fit_key = f"{vt}__fitness"
        if fit_key in loaded:
            meta = loaded[f"{vt}__meta"]
            results[vt] = {
                "var_type": vt,
                "fitness": loaded[f"{vt}__fitness"],
                "strategy": loaded[f"{vt}__strategy"],
                "invariants": loaded[f"{vt}__invariants"],
                "seeds": list(loaded[f"{vt}__seeds"]),
                "n_gens": int(meta[0]),
                "g_shift": int(meta[1]),
            }
    return results
