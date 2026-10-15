"""
runners.py: Execution and orchestration routines for Experiments 2 and 3.

Centralizes simulation loops, hyperparameter configuration, and trace collection for:
1. Dynamic Target Shift (DTS) across the 6 variable types (Real, Count, Binary, Categorical, Fuzzy, Circular).
2. Domain invariant and singularity stress tests (INV-R to INV-S).
3. Architectural ablation studies (Binary Ceiling and Categorical Gauge Drift).
4. Experiment 3 synthetic mixed-variable benchmark suites and heterogeneous dynamics tracking.
"""

from __future__ import annotations

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
from experiments.utils.isolation_landscapes import (
    DTSLandscape,
    real_boundary_target,
    integer_staircase,
    binary_hamming,
    categorical_deceptive,
    fuzzy_jaccard,
    circular_geodesic,
)
from experiments.utils.invariant_monitors import (
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
    Execute the Dynamic Target Shift (DTS) protocol on a specific variable type in isolation.

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
    Compute consolidated quantitative metrics for the DTS protocol:
    - Cooling Ratio (Phase A)
    - Re-inflation Lag (generations to peak parameter following target shift)
    - Phase C / Phase A convergence speed ratio
    - Invariant violations count
    """
    label_map = {
        "real": "Real (R)",
        "count": "Count (Z)",
        "binary": "Binary (B)",
        "categorical": "Categorical (C)",
        "fuzzy": "Fuzzy (F)",
        "circular": "Circular (S1)",
    }

    metrics: List[Dict[str, Any]] = []
    for res in results_list:
        vtype = res["var_type"]
        fit_arr = res["fitness"]
        strat_arr = res["strategy"]
        inv_arr = res["invariants"]

        # 1. Phase A cooling ratio
        init_strat = np.median(strat_arr[:, 0])
        min_strat_a = np.median(strat_arr[:, g_shift - 1])
        cooling_ratio = init_strat / max(1e-12, min_strat_a)

        # 2. Re-inflation lag (generations to strategy peak post-shift)
        post_shift_strat = strat_arr[:, g_shift:]
        peak_gens = [np.argmax(post_shift_strat[i]) for i in range(len(post_shift_strat))]
        re_inflation_lag = int(np.median(peak_gens))

        # 3. Relative convergence speed ratio (Phase C slope vs Phase A slope)
        fit_a_start = np.median(fit_arr[:, 0])
        fit_a_end = np.median(fit_arr[:, g_shift - 1])
        slope_a = (np.log10(max(1e-12, fit_a_start)) - np.log10(max(1e-12, fit_a_end))) / g_shift

        fit_c_start = np.median(fit_arr[:, g_shift])
        fit_c_end = np.median(fit_arr[:, -1])
        slope_c = (np.log10(max(1e-12, fit_c_start)) - np.log10(max(1e-12, fit_c_end))) / (n_gens - g_shift)
        speed_ratio = slope_c / max(1e-6, slope_a)

        # 4. Invariant violations count (all monitors record discrete violation counts >= 0)
        tot_violations = int(np.sum(inv_arr > 0))

        metrics.append({
            "Variable Type": label_map.get(vtype, vtype),
            "Cooling Ratio (Phase A)": f"{cooling_ratio:.2e}",
            "Re-inflation Lag (gens)": f"{re_inflation_lag:d}",
            "Phase C/A Speed Ratio": f"{speed_ratio:.2f}",
            "Invariant Violations": tot_violations,
        })

    if pd is not None:
        return pd.DataFrame(metrics)
    return metrics


# =============================================================================
# Experiment 3: Heterogeneous Benchmark Suite & Dynamics Tracking
# =============================================================================

def compute_heterogeneous_decomposition(
    ind: Individual,
    bench_name: str,
    cfg: ESConfig,
) -> Dict[str, float]:
    """
    Compute partial fitness error decomposition across the 6 variable domains
    for an individual on a specific synthetic benchmark.
    """
    t = get_default_targets(cfg)
    u = unpack_genome(ind, cfg)
    if bench_name == "Generalized_Sphere":
        c_real = float(np.sum((u.real - t["real"]) ** 2)) if cfg.n_real > 0 else 0.0
        c_count = float(np.sum((u.count - t["count"]) ** 2)) if cfg.n_count > 0 else 0.0
        c_bin = float(np.sum(u.binary != t["binary"])) if cfg.n_binary > 0 else 0.0
        c_cat = float(np.sum(u.categorical != t["categorical"])) if cfg.n_categorical > 0 else 0.0
        c_fuzz = float(np.sum((u.fuzzy - t["fuzzy"]) ** 2)) if cfg.n_fuzzy > 0 else 0.0
        k = t["circular_k"]
        diff = np.abs(u.circular - t["circular"]) % k
        c_circ = float(np.sum(np.minimum(diff, k - diff) ** 2)) if cfg.n_circular > 0 else 0.0
    elif bench_name == "Generalized_Rastrigin":
        diff_r = u.real - t["real"]
        c_real = float(10.0 * cfg.n_real + np.sum(diff_r ** 2 - 10.0 * np.cos(2.0 * np.pi * diff_r))) if cfg.n_real > 0 else 0.0
        diff_c = (u.count - t["count"]).astype(np.float64)
        c_count = float(2.0 * cfg.n_count + np.sum(diff_c ** 2 - 2.0 * np.cos(np.pi * diff_c))) if cfg.n_count > 0 else 0.0
        c_bin = float(5.0 * np.sum(u.binary != t["binary"])) if cfg.n_binary > 0 else 0.0
        c_cat = float(10.0 * np.sum(u.categorical != t["categorical"])) if cfg.n_categorical > 0 else 0.0
        diff_f = u.fuzzy - t["fuzzy"]
        n_pts = cfg.n_fuzzy * 2
        c_fuzz = float(5.0 * n_pts + np.sum(diff_f ** 2 - 5.0 * np.cos(2.0 * np.pi * diff_f))) if cfg.n_fuzzy > 0 else 0.0
        k = t["circular_k"]
        phase = 2.0 * np.pi * (u.circular - t["circular"]) / k
        c_circ = float(5.0 * np.sum(1.0 - np.cos(phase))) if cfg.n_circular > 0 else 0.0
    elif bench_name == "Generalized_Coupled":
        regime_offset = float(u.categorical[0]) if cfg.n_categorical > 0 else 0.0
        target_real_shifted = t["real"] + regime_offset * 0.5
        c_bin = float(np.sum(u.binary != 1)) * 3.0 if cfg.n_binary > 0 else 0.0
        count_scale = float(1.0 + 0.1 * np.abs(u.count[0] - t["count"][0])) if cfg.n_count > 0 else 1.0
        c_real = float(np.sum((u.real - target_real_shifted) ** 2)) if cfg.n_real > 0 else 0.0
        c_cat = float(5.0 * np.sum(u.categorical != 0)) if cfg.n_categorical > 0 else 0.0
        c_count = float(np.sum((u.count - t["count"]) ** 2)) if cfg.n_count > 0 else 0.0
        c_fuzz = float(count_scale * np.sum((u.fuzzy - t["fuzzy"]) ** 2)) if cfg.n_fuzzy > 0 else 0.0
        k = t["circular_k"]
        diff = np.abs(u.circular - t["circular"]) % k
        c_circ = float(np.sum(np.minimum(diff, k - diff) ** 2)) if cfg.n_circular > 0 else 0.0
    else:
        raise ValueError(f"Unknown benchmark: {bench_name}")
    return {
        "cost_real": c_real,
        "cost_count": c_count,
        "cost_binary": c_bin,
        "cost_cat": c_cat,
        "cost_fuzzy": c_fuzz,
        "cost_circular": c_circ,
    }


def run_heterogeneous_benchmark_suite(
    benchmarks: Dict[str, Dict[str, Any]],
    config: ESConfig,
    seeds: List[int],
    max_evals: int,
    data_dir: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run multi-seed optimization across all synthetic generalized benchmarks.

    Parameters
    ----------
    benchmarks : dict
        Mapping of benchmark names to benchmark dictionary with 'fn' and 'description'.
    config : ESConfig
        Genome configuration and hyperparameter settings.
    seeds : list of int
        Random seeds for statistical replication.
    max_evals : int
        Maximum fitness evaluations per optimization run.
    data_dir : str, optional
        Path to save 'exp3_traces.npz' if provided.

    Returns
    -------
    dict
        Aggregated results per benchmark containing mean, std, min, max, and all traces.
    """
    results: Dict[str, Dict[str, Any]] = {}
    n_seeds = len(seeds)

    for bench_name, bench_info in benchmarks.items():
        print(f"\n==========================================")
        print(f" Running Benchmark: {bench_name}")
        print(f" {bench_info.get('description', '')}")
        print(f"==========================================")

        fn = bench_info["fn"]
        eval_fn = lambda ind_or_x, _fn=fn: _fn(ind_or_x, config=config)

        seed_traces = []
        for idx, seed in enumerate(seeds):
            trace = run_generalized_es(
                fn=eval_fn,
                dim=config.n_real,
                bounds=(config.lower_bound_real, config.upper_bound_real),
                seed=seed,
                max_evals=max_evals,
                config=config,
            )
            seed_traces.append(trace)
            if (idx + 1) % max(1, n_seeds // 5) == 0 or idx == n_seeds - 1:
                print(f"  Seed {seed:5d} ({idx+1:2d}/{n_seeds}) | Final Best Fitness: {trace[-1]:.6e}")

        traces_arr = np.array(seed_traces)
        results[bench_name] = {
            "mean": np.mean(traces_arr, axis=0),
            "std": np.std(traces_arr, axis=0),
            "min": np.min(traces_arr, axis=0),
            "max": np.max(traces_arr, axis=0),
            "traces": traces_arr,
        }

    if data_dir is not None:
        os.makedirs(data_dir, exist_ok=True)
        save_dict = {
            f"{bench_name}__traces": res["traces"]
            for bench_name, res in results.items()
        }
        save_dict["seeds"] = np.array(seeds, dtype=np.int64)
        traces_path = os.path.join(data_dir, "exp3_traces.npz")
        np.savez_compressed(traces_path, **save_dict)
        print(f"\nSuccessfully saved experiment 3 traces to: {traces_path}")

    return results


def run_heterogeneous_dynamics_tracking(
    benchmarks: Dict[str, Dict[str, Any]],
    config: ESConfig,
    seeds: List[int],
    max_evals: int,
    data_dir: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, np.ndarray]]]:
    """
    Execute generation-by-generation tracking of strategy parameters, mutation intensities,
    categorical certainty metrics, and partial fitness decompositions across multiple seeds.

    Parameters
    ----------
    benchmarks : dict
        Mapping of benchmark names to benchmark dictionary.
    config : ESConfig
        Genome configuration.
    seeds : list of int
        Random seeds for evaluation.
    max_evals : int
        Maximum evaluations per run.
    data_dir : str, optional
        Path to save 'exp3_strategy_dynamics.npz'.

    Returns
    -------
    tuple of (all_saved_data, bench_dynamics)
        all_saved_data: Dict with all means, stds, raw arrays, generations, and seeds.
        bench_dynamics: Dict mapping benchmark name to aggregated mean and std metrics.
    """
    n_generations = max_evals // config.lam
    all_saved_data: Dict[str, Any] = {}
    bench_dynamics: Dict[str, Dict[str, np.ndarray]] = {}

    for bench_name, bench_info in benchmarks.items():
        print(f">>> Tracking Heterogeneous Dynamics: {bench_name} ({len(seeds)} Seeds)")
        fn = bench_info["fn"]
        eval_fn = lambda ind_or_x, _fn=fn: _fn(ind_or_x, config=config)
        seed_data: List[Dict[str, np.ndarray]] = []

        for seed in seeds:
            es = GeneralizedES(config=config, seed=seed)
            es.init_population(eval_fn)
            history: Dict[str, List[float]] = {
                "sigma_real": [],
                "lambda_count": [],
                "sigma_fuzzy": [],
                "lambda_circular": [],
                "p_flip_binary": [],
                "cat_confidence": [],
                "cat_entropy_norm": [],
                "cost_real": [],
                "cost_count": [],
                "cost_binary": [],
                "cost_cat": [],
                "cost_fuzzy": [],
                "cost_circular": [],
            }

            for _ in range(n_generations):
                es.step(eval_fn)
                best_p = es.parents[0]
                history["sigma_real"].append(float(np.mean(best_p.sigma_real)))
                history["lambda_count"].append(float(np.mean(best_p.lambda_count)))
                history["sigma_fuzzy"].append(float(np.mean(best_p.sigma_fuzzy)))
                history["lambda_circular"].append(float(np.mean(best_p.lambda_circular)))

                p_flips = 1.0 / (1.0 + np.exp(-np.clip(best_p.logit_binary, -700.0, 700.0)))
                history["p_flip_binary"].append(float(np.mean(p_flips)))

                offset = 0
                confs = []
                entropies = []
                for k in config.categorical_k:
                    k_int = int(k)
                    sl = best_p.logit_cat[offset : offset + k_int]
                    ez = np.exp(sl - np.max(sl))
                    probs = ez / np.sum(ez)
                    confs.append(float(np.max(probs)))
                    h = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12)))) / np.log(k_int)
                    entropies.append(h)
                    offset += k_int
                history["cat_confidence"].append(float(np.mean(confs)))
                history["cat_entropy_norm"].append(float(np.mean(entropies)))

                decomp = compute_heterogeneous_decomposition(best_p, bench_name, config)
                for k_cost, v_cost in decomp.items():
                    history[k_cost].append(v_cost)

            seed_data.append({k: np.array(v) for k, v in history.items()})

        bench_agg: Dict[str, np.ndarray] = {}
        for m in seed_data[0].keys():
            stacked = np.array([sd[m] for sd in seed_data])
            bench_agg[f"{m}_mean"] = np.mean(stacked, axis=0)
            bench_agg[f"{m}_std"] = np.std(stacked, axis=0)
            all_saved_data[f"{bench_name}__{m}_mean"] = bench_agg[f"{m}_mean"]
            all_saved_data[f"{bench_name}__{m}_std"] = bench_agg[f"{m}_std"]
            all_saved_data[f"{bench_name}__{m}_raw"] = stacked

        bench_dynamics[bench_name] = bench_agg

    all_saved_data["generations"] = np.arange(n_generations)
    all_saved_data["seeds"] = np.array(seeds)

    if data_dir is not None:
        os.makedirs(data_dir, exist_ok=True)
        dynamics_path = os.path.join(data_dir, "exp3_strategy_dynamics.npz")
        np.savez_compressed(dynamics_path, **all_saved_data)
        print(f"\nSuccessfully saved complete heterogeneous dynamics to: {dynamics_path}")

    return all_saved_data, bench_dynamics


def build_heterogeneous_summary_table(
    results: Dict[str, Dict[str, Any]],
    seeds: List[int],
    max_evals: int,
) -> Any:
    """
    Format statistical convergence summary metrics across benchmarks.
    """
    rows = []
    for bench_name, res in results.items():
        final_mean = res["mean"][-1]
        final_std = res["std"][-1]
        final_min = res["min"][-1]
        rows.append({
            "Benchmark Function": bench_name,
            "Mean Best Fitness": f"{final_mean:.6e}",
            "Std Dev": f"{final_std:.6e}",
            "Min (Best Run)": f"{final_min:.6e}",
        })

    if pd is not None:
        return pd.DataFrame(rows)
    return rows


