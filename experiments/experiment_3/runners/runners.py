"""
runners.py: Execution and dynamics tracking routines for Experiment 3 (Heterogeneous Benchmarks).
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

import concurrent.futures
from src.es_model.config import ESConfig
from src.es_model.individual import Individual, unpack_genome
from src.es_model.generalized_es import GeneralizedES
from src.runners.custom_es import run as run_generalized_es
from src.benchmarks.generalized_functions import get_default_targets, GENERALIZED_BENCHMARKS


def _eval_bench_seed_worker(args: Tuple[str, ESConfig, int, int]) -> Tuple[int, np.ndarray]:
    """Top-level picklable worker for multi-seed benchmark traces."""
    bench_name, config, seed, max_evals = args
    fn = GENERALIZED_BENCHMARKS[bench_name]["fn"]
    eval_fn = lambda ind_or_x, _fn=fn: _fn(ind_or_x, config=config)
    trace = run_generalized_es(
        fn=eval_fn,
        dim=config.n_real,
        bounds=(config.lower_bound_real, config.upper_bound_real),
        seed=seed,
        max_evals=max_evals,
        config=config,
    )
    return seed, trace


def _dynamics_seed_worker(args: Tuple[str, ESConfig, int, int]) -> Tuple[int, Dict[str, np.ndarray]]:
    """Top-level picklable worker for multi-seed strategy dynamics tracking."""
    bench_name, config, seed, max_evals = args
    fn = GENERALIZED_BENCHMARKS[bench_name]["fn"]
    eval_fn = lambda ind_or_x, _fn=fn: _fn(ind_or_x, config=config)
    n_generations = max_evals // config.lam

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

        p_mut_cat = 1.0 / (1.0 + np.exp(-np.clip(best_p.logit_cat, -700.0, 700.0)))
        history["cat_confidence"].append(float(np.mean(1.0 - p_mut_cat)) if len(p_mut_cat) > 0 else 0.0)
        if len(p_mut_cat) > 0:
            p_c = np.clip(p_mut_cat, 1e-12, 1.0 - 1e-12)
            h_cat = -(p_c * np.log2(p_c) + (1.0 - p_c) * np.log2(1.0 - p_c))
            history["cat_entropy_norm"].append(float(np.mean(h_cat)))
        else:
            history["cat_entropy_norm"].append(0.0)

        decomp = compute_heterogeneous_decomposition(best_p, bench_name, config)
        for k_cost, v_cost in decomp.items():
            history[k_cost].append(v_cost)

    return seed, {k: np.array(v) for k, v in history.items()}


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
    max_workers: int = 8,
) -> Dict[str, Dict[str, Any]]:
    """
    Run multi-seed optimization across all synthetic generalized benchmarks.
    """
    results: Dict[str, Dict[str, Any]] = {}
    n_seeds = len(seeds)

    for bench_name, bench_info in benchmarks.items():
        print(f"\n==========================================")
        print(f" Running Benchmark: {bench_name} ({n_seeds} Seeds)")
        print(f" {bench_info.get('description', '')}")
        print(f"==========================================")

        if max_workers > 1 and n_seeds > 1:
            worker_args = [(bench_name, config, seed, max_evals) for seed in seeds]
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                raw_results = list(executor.map(_eval_bench_seed_worker, worker_args))
            res_dict = dict(raw_results)
            seed_traces = [res_dict[s] for s in seeds]
        else:
            fn = bench_info["fn"]
            eval_fn = lambda ind_or_x, _fn=fn: _fn(ind_or_x, config=config)
            seed_traces = []
            for seed in seeds:
                trace = run_generalized_es(
                    fn=eval_fn,
                    dim=config.n_real,
                    bounds=(config.lower_bound_real, config.upper_bound_real),
                    seed=seed,
                    max_evals=max_evals,
                    config=config,
                )
                seed_traces.append(trace)

        for idx, (s, tr) in enumerate(zip(seeds, seed_traces)):
            if (idx + 1) % max(1, n_seeds // 5) == 0 or idx == n_seeds - 1:
                print(f"  Seed {s:5d} ({idx+1:2d}/{n_seeds}) | Final Best Fitness: {tr[-1]:.6e}")

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
    max_workers: int = 8,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, np.ndarray]]]:
    """
    Execute generation-by-generation tracking of strategy parameters, mutation intensities,
    categorical certainty metrics, and partial fitness decompositions across multiple seeds.
    """
    n_generations = max_evals // config.lam
    all_saved_data: Dict[str, Any] = {}
    bench_dynamics: Dict[str, Dict[str, np.ndarray]] = {}
    n_seeds = len(seeds)

    for bench_name, bench_info in benchmarks.items():
        print(f">>> Tracking Heterogeneous Dynamics: {bench_name} ({n_seeds} Seeds)")

        if max_workers > 1 and n_seeds > 1:
            worker_args = [(bench_name, config, seed, max_evals) for seed in seeds]
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                raw_results = list(executor.map(_dynamics_seed_worker, worker_args))
            res_dict = dict(raw_results)
            seed_data = [res_dict[s] for s in seeds]
        else:
            fn = bench_info["fn"]
            eval_fn = lambda ind_or_x, _fn=fn: _fn(ind_or_x, config=config)
            seed_data = []
            for seed in seeds:
                _, s_dict = _dynamics_seed_worker((bench_name, config, seed, max_evals))
                seed_data.append(s_dict)

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
    all_saved_data["seeds"] = np.array(seeds, dtype=np.int64)

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


def load_heterogeneous_benchmark_suite(
    data_dir: str,
    benchmarks: Dict[str, Any],
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load precomputed heterogeneous benchmark traces from exp3_traces.npz."""
    traces_path = os.path.join(data_dir, "exp3_traces.npz")
    if not os.path.exists(traces_path):
        return None
    loaded = np.load(traces_path, allow_pickle=True)
    results: Dict[str, Dict[str, Any]] = {}
    for bench_name in benchmarks.keys():
        key = f"{bench_name}__traces"
        if key in loaded:
            traces_arr = loaded[key]
            results[bench_name] = {
                "mean": np.mean(traces_arr, axis=0),
                "std": np.std(traces_arr, axis=0),
                "min": np.min(traces_arr, axis=0),
                "max": np.max(traces_arr, axis=0),
                "traces": traces_arr,
            }
    return results


def load_heterogeneous_dynamics_tracking(
    data_dir: str,
    benchmarks: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Dict[str, np.ndarray]]]]:
    """Load precomputed heterogeneous dynamics from exp3_strategy_dynamics.npz."""
    dynamics_path = os.path.join(data_dir, "exp3_strategy_dynamics.npz")
    if not os.path.exists(dynamics_path):
        return None, None
    loaded = np.load(dynamics_path, allow_pickle=True)
    all_saved_data = {k: loaded[k] for k in loaded.files}
    bench_dynamics: Dict[str, Dict[str, np.ndarray]] = {}
    for bench_name in benchmarks.keys():
        prefix = f"{bench_name}__"
        bench_agg: Dict[str, np.ndarray] = {}
        for k in loaded.files:
            if k.startswith(prefix) and not k.endswith("_raw"):
                metric_name = k[len(prefix):]
                bench_agg[metric_name] = loaded[k]
        bench_dynamics[bench_name] = bench_agg
    return all_saved_data, bench_dynamics
