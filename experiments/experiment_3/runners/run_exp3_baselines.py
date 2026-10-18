"""
run_exp3_baselines.py: External Baselines for Experiment 3 (Heterogeneous Benchmarks).

Executes:
1. Uniform Random Search (20,000 evaluations per seed, 40 seeds).
2. Standard CMA-ES with Continuous Relaxation & Rounding/Clipping (20,000 evaluations per seed, 40 seeds).
3. CatCMA with Margin (CatCMAwM, 2025) with native continuous/integer/categorical spaces (20,000 evaluations per seed, 40 seeds).

Evaluates on:
- Generalized Sphere (convex, separable)
- Generalized Coupled (discrete regime switches)
- Generalized Rastrigin (multimodal, non-convex)
- Generalized Multiplicative (non-separable multiplicative epistasis)

Saves results to:
- experiments/experiment_3/data/random_search_results.json
- experiments/experiment_3/data/random_search_baselines.npz
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
import concurrent.futures
import json
import os
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src").is_dir())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.es_model.config import ESConfig
from src.benchmarks.generalized_functions import (
    generalized_sphere,
    generalized_rastrigin,
    generalized_coupled,
    generalized_multiplicative,
    get_default_targets,
)

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False

try:
    from cmaes import CatCMAwM
    HAS_CATCMAWM = True
except ImportError:
    HAS_CATCMAWM = False


# Exact Experiment 3 Configuration
CONFIG = ESConfig(
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

SEEDS = [42 + i * 101 for i in range(40)]
MAX_EVALS = 20000

# Benchmark mappings
BENCHMARKS = [
    ("sphere", "Generalized Sphere", generalized_sphere),
    ("coupled", "Generalized Coupled", generalized_coupled),
    ("rastrigin", "Generalized Rastrigin", generalized_rastrigin),
    ("multiplicative", "Generalized Multiplicative", generalized_multiplicative),
]

# Continuous relaxation bounds for CMA-ES (20 dimensions)
LOWER_BOUNDS = np.array(
    [-5.12] * 5                      # Real
    + [0.0] * 3                      # Count
    + [0.0] * 4                      # Binary
    + [0.0, 0.0]                     # Categorical
    + [0.0, 1e-4, 0.0, 1e-4]         # Fuzzy (m1, d1, m2, d2)
    + [0.0, 0.0],                    # Circular
    dtype=np.float64,
)

UPPER_BOUNDS = np.array(
    [5.12] * 5                       # Real
    + [20.0] * 3                     # Count
    + [1.0] * 4                      # Binary
    + [2.0, 3.0]                     # Categorical (k-1: 3-1=2, 4-1=3)
    + [1.0, 1.0, 1.0, 1.0]           # Fuzzy (m1, d1, m2, d2)
    + [6.0, 11.0],                   # Circular (k-1: 7-1=6, 12-1=11)
    dtype=np.float64,
)


def sample_uniform_candidates(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    """
    Generate n_samples uniform random candidate phenotypes respecting domain bounds:
    - real: uniform in [-5.12, 5.12]^5
    - count: discrete uniform in integers [0, 20]^3
    - binary: discrete uniform in {0, 1}^4
    - categorical: uniform categorical choice in {0..k-1} with k=[3, 4]
    - fuzzy: uniform m in [0, 1], uniform d in [1e-4, 1.0] for the 2 fuzzy variables
    - circular: uniform integer in {0..k-1} with k=[7, 12]
    """
    vecs = np.empty((n_samples, 20), dtype=np.float64)
    # Real
    vecs[:, 0:5] = rng.uniform(-5.12, 5.12, size=(n_samples, 5))
    # Count
    vecs[:, 5:8] = rng.integers(0, 21, size=(n_samples, 3))
    # Binary
    vecs[:, 8:12] = rng.integers(0, 2, size=(n_samples, 4))
    # Categorical
    vecs[:, 12] = rng.integers(0, 3, size=n_samples)
    vecs[:, 13] = rng.integers(0, 4, size=n_samples)
    # Fuzzy: m1, d1, m2, d2
    vecs[:, 14] = rng.uniform(0.0, 1.0, size=n_samples)
    vecs[:, 15] = rng.uniform(1e-4, 1.0, size=n_samples)
    vecs[:, 16] = rng.uniform(0.0, 1.0, size=n_samples)
    vecs[:, 17] = rng.uniform(1e-4, 1.0, size=n_samples)
    # Circular
    vecs[:, 18] = rng.integers(0, 7, size=n_samples)
    vecs[:, 19] = rng.integers(0, 12, size=n_samples)
    return vecs


def run_random_search_seed(
    fn: Callable,
    seed: int,
    max_evals: int,
    targets: dict,
) -> Tuple[float, np.ndarray]:
    """Execute Random Search for a single seed and return the best fitness and evaluation trace."""
    rng = np.random.default_rng(seed)
    candidates = sample_uniform_candidates(rng, max_evals)
    trace = np.empty(max_evals, dtype=np.float64)
    best_fitness = float("inf")
    for i, vec in enumerate(candidates):
        fitness = float(fn(vec, config=CONFIG, targets=targets))
        if fitness < best_fitness:
            best_fitness = fitness
        trace[i] = best_fitness
    return best_fitness, trace


def evaluate_cma_candidate(x: np.ndarray, fn: Callable, targets: dict) -> float:
    """Discretize continuous CMA solution via clipping and rounding."""
    vec = np.clip(x, LOWER_BOUNDS, UPPER_BOUNDS).copy()
    # Count rounding
    vec[5:8] = np.round(vec[5:8])
    # Binary thresholding at 0.5
    vec[8:12] = (vec[8:12] >= 0.5).astype(np.float64)
    # Categorical rounding
    vec[12] = np.round(vec[12])
    vec[13] = np.round(vec[13])
    # Circular rounding
    vec[18] = np.round(vec[18])
    vec[19] = np.round(vec[19])
    return float(fn(vec, config=CONFIG, targets=targets))


def run_cma_es_seed(
    fn: Callable,
    seed: int,
    max_evals: int,
    targets: dict,
) -> Tuple[float, np.ndarray]:
    """Execute CMA-ES with continuous relaxation and rounding for a single seed."""
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(LOWER_BOUNDS, UPPER_BOUNDS)
    sigma0 = 0.3 * float(np.mean(UPPER_BOUNDS - LOWER_BOUNDS))
    
    opts = {
        "seed": int(seed),
        "bounds": [LOWER_BOUNDS.tolist(), UPPER_BOUNDS.tolist()],
        "verbose": -9,
        "maxfevals": max_evals,
    }
    
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    eval_count = 0
    best_fitness = float("inf")
    trace = np.empty(max_evals, dtype=np.float64)
    
    while eval_count < max_evals and not es.stop():
        solutions = es.ask()
        costs = []
        for sol in solutions:
            cost = evaluate_cma_candidate(sol, fn, targets)
            costs.append(cost)
            if cost < best_fitness:
                best_fitness = cost
            trace[eval_count] = best_fitness
            eval_count += 1
            if eval_count >= max_evals:
                break
        es.tell(solutions[:len(costs)], costs)
        
    while eval_count < max_evals:
        trace[eval_count] = best_fitness
        eval_count += 1

    return best_fitness, trace


# -----------------------------------------------------------------------------
# CatCMA with Margin (CatCMAwM, 2025) Adapter & Runner
# -----------------------------------------------------------------------------

def build_catcmawm_spaces(config: ESConfig) -> Dict[str, Any]:
    """
    Construct explicit search spaces, bounds, and initial parameters for CatCMAwM:
    - x_space: 5 real [-5.12, 5.12] + 4 fuzzy parameters [0, 1]x[1e-4, 1] (9 continuous dimensions)
    - z_space: 3 count variables with domain [0..20] (3 integer dimensions)
    - c_space: 4 binary (cardinality 2) + 2 categorical (k=[3, 4]) + 2 circular (k=[7, 12]) (8 categorical dimensions)
    - mean: explicit midpoint of continuous and integer bounds (length 12)
    - sigma: 1.0 (standard initial step size)
    - cat_param: explicit uniform categorical distribution (8 x 12)
    """
    x_space = (
        [[-5.12, 5.12]] * 5
        + [[0.0, 1.0], [1e-4, 1.0], [0.0, 1.0], [1e-4, 1.0]]
    )
    z_space = [list(range(21)) for _ in range(3)]
    c_space = [2, 2, 2, 2, 3, 4, 7, 12]

    # Explicit initial mean: midpoint of each Gaussian variable
    # 5 real (center 0.0), 4 fuzzy (center 0.5, 0.50005, 0.5, 0.50005), 3 count (center 10.0)
    mean = np.array(
        [0.0] * 5 + [0.5, 0.50005, 0.5, 0.50005] + [10.0] * 3,
        dtype=np.float64,
    )
    sigma = 1.0

    # Explicit categorical initial probabilities (uniform 1/K_i for each variable)
    k_max = max(c_space)
    cat_param = np.zeros((len(c_space), k_max), dtype=np.float64)
    for i, k in enumerate(c_space):
        cat_param[i, :k] = 1.0 / k

    return {
        "x_space": x_space,
        "z_space": z_space,
        "c_space": c_space,
        "mean": mean,
        "sigma": sigma,
        "cat_param": cat_param,
    }


def decode_catcmawm_solution(sol: Any, config: ESConfig) -> np.ndarray:
    """
    Decode CatCMAwM Solution object into canonical 20D phenotype vector:
    [0:5]   real (5)
    [5:8]   count (3)
    [8:12]  binary (4)
    [12:14] categorical (2)
    [14:18] fuzzy (4: m1, d1, m2, d2)
    [18:20] circular (2)
    """
    vec = np.empty(20, dtype=np.float64)
    # Real
    vec[0:5] = sol.x[0:5]
    # Count (integer)
    vec[5:8] = sol.z[0:3]
    # Categorical: one-hot rows decoded via argmax
    cat_decoded = np.argmax(sol.c, axis=1)
    # Binary
    vec[8:12] = cat_decoded[0:4]
    # Categorical
    vec[12:14] = cat_decoded[4:6]
    # Fuzzy (m1, d1, m2, d2)
    vec[14:18] = sol.x[5:9]
    # Circular
    vec[18:20] = cat_decoded[6:8]
    return vec


def run_catcmawm_seed(
    fn: Callable,
    seed: int,
    max_evals: int,
    targets: dict,
) -> Tuple[float, np.ndarray]:
    """Execute CatCMAwM for a single seed and return the best fitness and evaluation trace."""
    spaces = build_catcmawm_spaces(CONFIG)
    optimizer = CatCMAwM(
        x_space=spaces["x_space"],
        z_space=spaces["z_space"],
        c_space=spaces["c_space"],
        mean=spaces["mean"],
        sigma=spaces["sigma"],
        cat_param=spaces["cat_param"],
        seed=int(seed),
    )
    trace = np.empty(max_evals, dtype=np.float64)
    eval_count = 0
    best_fitness = float("inf")

    while eval_count < max_evals:
        solutions = []
        for _ in range(optimizer.population_size):
            sol = optimizer.ask()
            vec = decode_catcmawm_solution(sol, CONFIG)
            cost = float(fn(vec, config=CONFIG, targets=targets))
            if cost < best_fitness:
                best_fitness = cost
            trace[eval_count] = best_fitness
            solutions.append((sol, cost))
            eval_count += 1
            if eval_count >= max_evals:
                break
        if len(solutions) == optimizer.population_size:
            optimizer.tell(solutions)

    return best_fitness, trace


# -----------------------------------------------------------------------------
# Multiprocessing Worker Callbacks
# -----------------------------------------------------------------------------

def _worker_rs_seed(args: Tuple[str, int, int]) -> Tuple[int, float, np.ndarray]:
    bench_key, seed, max_evals = args
    bench_dict = {k: fn for k, _, fn in BENCHMARKS}
    fn = bench_dict[bench_key]
    targets = get_default_targets(CONFIG)
    f_best, trace = run_random_search_seed(fn, seed, max_evals, targets)
    return seed, f_best, trace


def _worker_cma_seed(args: Tuple[str, int, int]) -> Tuple[int, float, np.ndarray]:
    bench_key, seed, max_evals = args
    bench_dict = {k: fn for k, _, fn in BENCHMARKS}
    fn = bench_dict[bench_key]
    targets = get_default_targets(CONFIG)
    f_best, trace = run_cma_es_seed(fn, seed, max_evals, targets)
    return seed, f_best, trace


def _worker_catcmawm_seed(args: Tuple[str, int, int]) -> Tuple[int, float, np.ndarray]:
    bench_key, seed, max_evals = args
    bench_dict = {k: fn for k, _, fn in BENCHMARKS}
    fn = bench_dict[bench_key]
    targets = get_default_targets(CONFIG)
    f_best, trace = run_catcmawm_seed(fn, seed, max_evals, targets)
    return seed, f_best, trace


def _calc_stats(vals: np.ndarray, per_seed_dict: Dict[str, float]) -> Dict[str, Any]:
    q25 = float(np.percentile(vals, 25))
    q75 = float(np.percentile(vals, 75))
    return {
        "min": float(np.min(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "max": float(np.max(vals)),
        "median": float(np.median(vals)),
        "iqr": float(q75 - q25),
        "q25": q25,
        "q75": q75,
        "per_seed": per_seed_dict,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 3 External Baselines Runner")
    parser.add_argument("--force-all", action="store_true", help="Force recomputation of all benchmarks")
    parser.add_argument("--max-evals", type=int, default=MAX_EVALS, help="Budget per seed")
    parser.add_argument("--n-seeds", type=int, default=len(SEEDS), help="Number of seeds")
    args = parser.parse_args()

    max_evals = args.max_evals
    seeds = SEEDS[:args.n_seeds]

    print("=" * 70)
    print(" EXPERIMENT 3 EXTERNAL BASELINE RUNNER (4 METHODS)")
    print(f" Config: 20D Heterogeneous Phenotype (5R, 3C, 4B, 2Cat, 2Fuz, 2Circ)")
    print(f" Budget: {max_evals:,} evaluations | Seeds: {len(seeds)} seeds")
    print(f" CMA-ES available: {HAS_CMA} | CatCMAwM available: {HAS_CATCMAWM}")
    print("=" * 70)

    output_dir = PROJECT_ROOT / "experiments" / "experiment_3" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "random_search_results.json"
    npz_path = output_dir / "random_search_baselines.npz"

    rs_results: Dict[str, Any] = {}
    cma_results: Dict[str, Any] = {}
    catcmawm_results: Dict[str, Any] = {}

    rs_traces: Dict[str, np.ndarray] = {}
    cma_traces: Dict[str, np.ndarray] = {}
    catcmawm_traces: Dict[str, np.ndarray] = {}

    # Load existing if available and not force-all
    if not args.force_all and npz_path.exists() and results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                prev_json = json.load(f)
            prev_npz = np.load(npz_path, allow_pickle=True)
            rs_results = prev_json.get("random_search", {})
            cma_results = prev_json.get("cma_es_relaxation", {}) or {}
            catcmawm_results = prev_json.get("catcmawm", {}) or {}
            for k, _, _ in BENCHMARKS:
                if f"{k}__rs_traces" in prev_npz:
                    rs_traces[k] = prev_npz[f"{k}__rs_traces"]
                if f"{k}__cma_traces" in prev_npz:
                    cma_traces[k] = prev_npz[f"{k}__cma_traces"]
                if f"{k}__catcmawm_traces" in prev_npz:
                    catcmawm_traces[k] = prev_npz[f"{k}__catcmawm_traces"]
            print(f"Loaded existing precomputed baselines for: {list(rs_traces.keys())}")
        except Exception as e:
            print(f"Notice: Could not load previous baseline cache ({e}), starting fresh.")

    # 1. Random Search Evaluation
    print("\n[1/3] RUNNING UNIFORM RANDOM SEARCH...")
    t0 = time.time()
    for key, name, fn in BENCHMARKS:
        if key in rs_traces and not args.force_all:
            print(f"Skipping {name} ({key}) - already present in precomputed baselines.")
            continue
        print(f"\n>>> Benchmark: {name} ({key})")
        worker_args = [(key, s, max_evals) for s in seeds]
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            raw_rs = list(executor.map(_worker_rs_seed, worker_args))
        # raw_rs is list of (seed, f_best, trace)
        raw_rs.sort(key=lambda item: item[0])
        res_dict = {item[0]: item[1] for item in raw_rs}
        trace_mat = np.array([item[2] for item in raw_rs], dtype=np.float64)
        rs_traces[key] = trace_mat

        values = [res_dict[s] for s in seeds]
        per_seed = {str(s): res_dict[s] for s in seeds}
        vals_arr = np.array(values, dtype=np.float64)
        rs_results[key] = _calc_stats(vals_arr, per_seed)
        print(f"  -> Min: {np.min(vals_arr):.6f} | Median: {np.median(vals_arr):.6f} | Mean: {np.mean(vals_arr):.6f} | Std: {np.std(vals_arr):.6f}")

    print(f"\nRandom Search step finished in {time.time() - t0:.2f} seconds.")

    # 2. Standard CMA-ES Continuous Relaxation Evaluation
    if HAS_CMA:
        print("\n[2/3] RUNNING STANDARD CMA-ES CONTINUOUS RELAXATION...")
        t1 = time.time()
        for key, name, fn in BENCHMARKS:
            if key in cma_traces and not args.force_all:
                print(f"Skipping {name} ({key}) - already present in precomputed baselines.")
                continue
            print(f"\n>>> Benchmark: {name} ({key})")
            worker_args = [(key, s, max_evals) for s in seeds]
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                raw_cma = list(executor.map(_worker_cma_seed, worker_args))
            raw_cma.sort(key=lambda item: item[0])
            res_dict = {item[0]: item[1] for item in raw_cma}
            trace_mat = np.array([item[2] for item in raw_cma], dtype=np.float64)
            cma_traces[key] = trace_mat

            values = [res_dict[s] for s in seeds]
            per_seed = {str(s): res_dict[s] for s in seeds}
            vals_arr = np.array(values, dtype=np.float64)
            cma_results[key] = _calc_stats(vals_arr, per_seed)
            print(f"  -> Min: {np.min(vals_arr):.6f} | Median: {np.median(vals_arr):.6f} | Mean: {np.mean(vals_arr):.6f} | Std: {np.std(vals_arr):.6f}")
        print(f"\nCMA-ES Relaxation step finished in {time.time() - t1:.2f} seconds.")

    # 3. CatCMA with Margin (CatCMAwM, 2025) Evaluation
    if HAS_CATCMAWM:
        print("\n[3/3] RUNNING CATCMA WITH MARGIN (CatCMAwM, 2025)...")
        t2 = time.time()
        for key, name, fn in BENCHMARKS:
            if key in catcmawm_traces and not args.force_all:
                print(f"Skipping {name} ({key}) - already present in precomputed baselines.")
                continue
            print(f"\n>>> Benchmark: {name} ({key})")
            worker_args = [(key, s, max_evals) for s in seeds]
            with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
                raw_cat = list(executor.map(_worker_catcmawm_seed, worker_args))
            raw_cat.sort(key=lambda item: item[0])
            res_dict = {item[0]: item[1] for item in raw_cat}
            trace_mat = np.array([item[2] for item in raw_cat], dtype=np.float64)
            catcmawm_traces[key] = trace_mat

            values = [res_dict[s] for s in seeds]
            per_seed = {str(s): res_dict[s] for s in seeds}
            vals_arr = np.array(values, dtype=np.float64)
            catcmawm_results[key] = _calc_stats(vals_arr, per_seed)
            print(f"  -> Min: {np.min(vals_arr):.6f} | Median: {np.median(vals_arr):.6f} | Mean: {np.mean(vals_arr):.6f} | Std: {np.std(vals_arr):.6f}")
        print(f"\nCatCMAwM step finished in {time.time() - t2:.2f} seconds.")

    # 4. Save to JSON
    combined_results = {
        "random_search": rs_results,
        "cma_es_relaxation": cma_results if HAS_CMA else None,
        "catcmawm": catcmawm_results if HAS_CATCMAWM else None,
        "metadata": {
            "max_evals": max_evals,
            "seeds": seeds,
            "config": {
                "n_real": CONFIG.n_real,
                "n_count": CONFIG.n_count,
                "n_binary": CONFIG.n_binary,
                "n_categorical": CONFIG.n_categorical,
                "categorical_k": CONFIG.categorical_k.tolist(),
                "n_fuzzy": CONFIG.n_fuzzy,
                "n_circular": CONFIG.n_circular,
                "circular_k": CONFIG.circular_k.tolist(),
            },
        },
    }
    for key, _, _ in BENCHMARKS:
        if key in rs_results:
            combined_results[key] = rs_results[key]

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    # 5. Save full traces to .npz
    save_dict = {"seeds": np.array(seeds, dtype=np.int64)}
    for key, _, _ in BENCHMARKS:
        if key in rs_results and key in rs_traces:
            save_dict[f"random_search_{key}"] = np.array([rs_results[key]["per_seed"][str(s)] for s in seeds], dtype=np.float64)
            save_dict[f"{key}__rs_traces"] = rs_traces[key]
        if HAS_CMA and key in cma_results and key in cma_traces:
            save_dict[f"cma_es_relaxation_{key}"] = np.array([cma_results[key]["per_seed"][str(s)] for s in seeds], dtype=np.float64)
            save_dict[f"{key}__cma_traces"] = cma_traces[key]
        if HAS_CATCMAWM and key in catcmawm_results and key in catcmawm_traces:
            save_dict[f"catcmawm_{key}"] = np.array([catcmawm_results[key]["per_seed"][str(s)] for s in seeds], dtype=np.float64)
            save_dict[f"{key}__catcmawm_traces"] = catcmawm_traces[key]

    np.savez_compressed(npz_path, **save_dict)

    print(f"\n[OK] Results successfully saved to: {results_path}")
    print(f"[OK] NumPy arrays successfully saved to: {npz_path}")

    # 6. Summary Table Output
    print("\n" + "=" * 95)
    print(f" SUMMARY TABLE: EXTERNAL BASELINES ({len(SEEDS)} SEEDS, {MAX_EVALS:,} EVALS)")
    print("=" * 95)
    print(f"{'Benchmark':<24} | {'Metric':<10} | {'Random Search':<16} | {'CMA-ES (Relaxed)':<16} | {'CatCMAwM (2025)':<16}")
    print("-" * 95)
    for key, name, _ in BENCHMARKS:
        rs_min = f"{rs_results[key]['min']:.4e}"
        rs_med = f"{rs_results[key]['median']:.4e}"
        rs_mean = f"{rs_results[key]['mean']:.4e}"
        rs_iqr = f"{rs_results[key]['iqr']:.4e}"

        cma_min = f"{cma_results[key]['min']:.4e}" if HAS_CMA else "N/A"
        cma_med = f"{cma_results[key]['median']:.4e}" if HAS_CMA else "N/A"
        cma_mean = f"{cma_results[key]['mean']:.4e}" if HAS_CMA else "N/A"
        cma_iqr = f"{cma_results[key]['iqr']:.4e}" if HAS_CMA else "N/A"

        cat_min = f"{catcmawm_results[key]['min']:.4e}" if HAS_CATCMAWM else "N/A"
        cat_med = f"{catcmawm_results[key]['median']:.4e}" if HAS_CATCMAWM else "N/A"
        cat_mean = f"{catcmawm_results[key]['mean']:.4e}" if HAS_CATCMAWM else "N/A"
        cat_iqr = f"{catcmawm_results[key]['iqr']:.4e}" if HAS_CATCMAWM else "N/A"

        print(f"{name:<24} | {'Best (Min)':<10} | {rs_min:<16} | {cma_min:<16} | {cat_min:<16}")
        print(f"{'':<24} | {'Median':<10} | {rs_med:<16} | {cma_med:<16} | {cat_med:<16}")
        print(f"{'':<24} | {'IQR':<10} | {rs_iqr:<16} | {cma_iqr:<16} | {cat_iqr:<16}")
        print(f"{'':<24} | {'Mean':<10} | {rs_mean:<16} | {cma_mean:<16} | {cat_mean:<16}")
        print("-" * 95)


if __name__ == "__main__":
    main()
