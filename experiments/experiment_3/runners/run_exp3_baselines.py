"""
run_random_search_exp3.py: External Baselines for Experiment 3 (Heterogeneous Benchmarks).

Executes:
1. Uniform Random Search (20,000 evaluations per seed, 10 seeds).
2. Standard CMA-ES with Continuous Relaxation & Rounding/Clipping (20,000 evaluations per seed, 10 seeds).

Evaluates on:
- Generalized Sphere (convex, separable)
- Generalized Coupled (discrete regime switches)
- Generalized Rastrigin (multimodal, non-convex)

Saves results to:
experiments/ablation_progress/random_search_results.json
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


import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

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
    get_default_targets,
)

try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False


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

SEEDS = [42, 143, 244, 345, 446, 547, 648, 749, 850, 951]
MAX_EVALS = 20000

# Benchmark mappings
BENCHMARKS = [
    ("sphere", "Generalized Sphere", generalized_sphere),
    ("coupled", "Generalized Coupled", generalized_coupled),
    ("rastrigin", "Generalized Rastrigin", generalized_rastrigin),
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
) -> float:
    """Execute Random Search for a single seed and return the best fitness."""
    rng = np.random.default_rng(seed)
    candidates = sample_uniform_candidates(rng, max_evals)
    
    best_fitness = float("inf")
    for vec in candidates:
        fitness = fn(vec, config=CONFIG, targets=targets)
        if fitness < best_fitness:
            best_fitness = float(fitness)
            
    return best_fitness


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
) -> float:
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
    
    while eval_count < max_evals and not es.stop():
        solutions = es.ask()
        costs = []
        for sol in solutions:
            cost = evaluate_cma_candidate(sol, fn, targets)
            costs.append(cost)
            if cost < best_fitness:
                best_fitness = cost
            eval_count += 1
            if eval_count >= max_evals:
                break
        es.tell(solutions[:len(costs)], costs)
        
    return best_fitness


def main():
    print("=" * 70)
    print(" EXPERIMENT 3 (A7) EXTERNAL BASELINE RUNNER")
    print(f" Config: 20D Heterogeneous Phenotype (5R, 3C, 4B, 2Cat, 2Fuz, 2Circ)")
    print(f" Budget: {MAX_EVALS:,} evaluations | Seeds: {len(SEEDS)} seeds")
    print(f" CMA-ES available: {HAS_CMA}")
    print("=" * 70)

    targets = get_default_targets(CONFIG)
    output_dir = PROJECT_ROOT / "experiments" / "data" / "experiment_3"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "random_search_results.json"
    npz_path = output_dir / "random_search_baselines.npz"

    rs_results: Dict[str, Any] = {}
    cma_results: Dict[str, Any] = {}

    # 1. Random Search Evaluation
    print("\n[1/2] RUNNING UNIFORM RANDOM SEARCH...")
    t0 = time.time()
    for key, name, fn in BENCHMARKS:
        print(f"\n>>> Benchmark: {name} ({key})")
        per_seed = {}
        values = []
        for s in SEEDS:
            f_best = run_random_search_seed(fn, s, MAX_EVALS, targets)
            per_seed[str(s)] = f_best
            values.append(f_best)
            print(f"  Seed {s:4d} | Best Fitness: {f_best:12.6f}")

        vals_arr = np.array(values, dtype=np.float64)
        rs_results[key] = {
            "min": float(np.min(vals_arr)),
            "mean": float(np.mean(vals_arr)),
            "std": float(np.std(vals_arr)),
            "max": float(np.max(vals_arr)),
            "per_seed": per_seed,
        }
        print(f"  -> Min: {np.min(vals_arr):.6f} | Mean: {np.mean(vals_arr):.6f} | Std: {np.std(vals_arr):.6f}")

    print(f"\nRandom Search completed in {time.time() - t0:.2f} seconds.")

    # 2. Standard CMA-ES Continuous Relaxation Evaluation
    if HAS_CMA:
        print("\n[2/2] RUNNING STANDARD CMA-ES CONTINUOUS RELAXATION...")
        t1 = time.time()
        for key, name, fn in BENCHMARKS:
            print(f"\n>>> Benchmark: {name} ({key})")
            per_seed = {}
            values = []
            for s in SEEDS:
                f_best = run_cma_es_seed(fn, s, MAX_EVALS, targets)
                per_seed[str(s)] = f_best
                values.append(f_best)
                print(f"  Seed {s:4d} | Best Fitness: {f_best:12.6f}")

            vals_arr = np.array(values, dtype=np.float64)
            cma_results[key] = {
                "min": float(np.min(vals_arr)),
                "mean": float(np.mean(vals_arr)),
                "std": float(np.std(vals_arr)),
                "max": float(np.max(vals_arr)),
                "per_seed": per_seed,
            }
            print(f"  -> Min: {np.min(vals_arr):.6f} | Mean: {np.mean(vals_arr):.6f} | Std: {np.std(vals_arr):.6f}")
        print(f"\nCMA-ES Relaxation completed in {time.time() - t1:.2f} seconds.")

    # 3. Save to JSON
    combined_results = {
        "random_search": rs_results,
        "cma_es_relaxation": cma_results if HAS_CMA else None,
        # Direct aliases for standard access:
        "sphere": rs_results["sphere"],
        "coupled": rs_results["coupled"],
        "rastrigin": rs_results["rastrigin"],
        "metadata": {
            "max_evals": MAX_EVALS,
            "seeds": SEEDS,
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

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    # Save compact .npz arrays
    save_dict = {"seeds": np.array(SEEDS)}
    for key, _, _ in BENCHMARKS:
        save_dict[f"random_search_{key}"] = np.array([rs_results[key]["per_seed"][s] for s in SEEDS], dtype=np.float64)
        if HAS_CMA:
            save_dict[f"cma_es_relaxation_{key}"] = np.array([cma_results[key]["per_seed"][s] for s in SEEDS], dtype=np.float64)
    np.savez_compressed(npz_path, **save_dict)

    print(f"\n[OK] Results successfully saved to: {results_path}")
    print(f"[OK] NumPy arrays successfully saved to: {npz_path}")

    # 4. Summary Table Output
    print("\n" + "=" * 70)
    print(" SUMMARY TABLE: RANDOM SEARCH & CMA-ES RELAXATION (10 SEEDS, 20,000 EVALS)")
    print("=" * 70)
    print(f"{'Benchmark':<24} | {'Metric':<10} | {'Random Search':<16} | {'CMA-ES (Relaxed)':<16}")
    print("-" * 70)
    for key, name, _ in BENCHMARKS:
        rs_min = f"{rs_results[key]['min']:.4e}"
        rs_mean = f"{rs_results[key]['mean']:.4e}"
        rs_std = f"{rs_results[key]['std']:.4e}"
        if HAS_CMA:
            cma_min = f"{cma_results[key]['min']:.4e}"
            cma_mean = f"{cma_results[key]['mean']:.4e}"
            cma_std = f"{cma_results[key]['std']:.4e}"
        else:
            cma_min = cma_mean = cma_std = "N/A"
            
        print(f"{name:<24} | {'Best (Min)':<10} | {rs_min:<16} | {cma_min:<16}")
        print(f"{'':<24} | {'Mean':<10} | {rs_mean:<16} | {cma_mean:<16}")
        print(f"{'':<24} | {'Std':<10} | {rs_std:<16} | {cma_std:<16}")
        print("-" * 70)


if __name__ == "__main__":
    main()
