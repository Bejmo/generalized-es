"""
run_factorial_ablation.py: Factorial ablation study on Circular Mutation and Scale Parameter Recombination.

Evaluates two specific variants across 3 benchmarks (Sphere, Coupled, Rastrigin)
and 10 canonical seeds (42, 143, 244, 345, 446, 547, 648, 749, 850, 951) with MAX_EVALS = 20000:
1. Variant 1: Skellam_Arithmetic
   - Modular Skellam mutation for circular variables
   - Arithmetic intermediate recombination for scale parameters (real, count, fuzzy, circular)
2. Variant 2: ForcedStep_Geometric
   - Geometric intermediate recombination for scale parameters
   - Forced-step mutation for circular variables (step in {-1, +1}, never 0)
"""

from __future__ import annotations

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.es_model.config import ESConfig
from src.es_model.generalized_es import GeneralizedES
from src.benchmarks.generalized_functions import (
    generalized_sphere,
    generalized_coupled,
    generalized_rastrigin,
)

SEEDS = [42, 143, 244, 345, 446, 547, 648, 749, 850, 951]
MAX_EVALS = 20000

BENCHMARKS = {
    "sphere": {
        "fn": generalized_sphere,
        "name": "Generalized_Sphere",
    },
    "coupled": {
        "fn": generalized_coupled,
        "name": "Generalized_Coupled",
    },
    "rastrigin": {
        "fn": generalized_rastrigin,
        "name": "Generalized_Rastrigin",
    },
}

VARIANTS = {
    "Skellam_Arithmetic": {
        "strategy_recomb_scale": "arithmetic",
        "circular_mutation_operator": "skellam",
        "description": "Modular Skellam circular mutation with arithmetic scale parameter recombination",
    },
    "ForcedStep_Geometric": {
        "strategy_recomb_scale": "geometric",
        "circular_mutation_operator": "forced_step",
        "description": "Forced-step circular mutation (+-1, never 0) with geometric scale parameter recombination",
    },
}


def run_single_experiment(
    variant_name: str,
    bench_key: str,
    seed: int,
    strategy_recomb_scale: str,
    circular_mutation_operator: str,
) -> Tuple[str, str, int, float]:
    bench_fn = BENCHMARKS[bench_key]["fn"]
    cfg = ESConfig(
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
        seed=seed,
        strategy_recomb_scale=strategy_recomb_scale,
        circular_mutation_operator=circular_mutation_operator,
    )
    eval_fn = lambda x: bench_fn(x, config=cfg)
    es = GeneralizedES(config=cfg, seed=seed)
    trace = es.optimize(eval_fn, max_evals=MAX_EVALS)
    final_fitness = float(trace[-1])
    return variant_name, bench_key, seed, final_fitness


def main() -> None:
    t_start = time.time()
    print("=" * 70)
    print("STARTING FACTORIAL ABLATION EXPERIMENTS")
    print(f"Variants: {list(VARIANTS.keys())}")
    print(f"Benchmarks: {list(BENCHMARKS.keys())}")
    print(f"Seeds ({len(SEEDS)}): {SEEDS}")
    print(f"Max Evals per run: {MAX_EVALS}")
    print("=" * 70)

    tasks = []
    for var_name, var_info in VARIANTS.items():
        for b_key in BENCHMARKS:
            for seed in SEEDS:
                tasks.append((
                    var_name,
                    b_key,
                    seed,
                    var_info["strategy_recomb_scale"],
                    var_info["circular_mutation_operator"],
                ))

    total_tasks = len(tasks)
    print(f"Total experiment runs to execute: {total_tasks}")

    raw_results: Dict[str, Dict[str, Dict[int, float]]] = {
        v: {b: {} for b in BENCHMARKS} for v in VARIANTS
    }

    n_workers = min(8, os.cpu_count() or 4)
    print(f"Dispatching with ProcessPoolExecutor (workers={n_workers})...\n")

    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_single_experiment, *task) for task in tasks]
        for f in as_completed(futures):
            var_name, b_key, seed, fitness = f.result()
            raw_results[var_name][b_key][seed] = fitness
            completed += 1
            print(
                f"[{completed:2d}/{total_tasks:2d}] {var_name:<20} | {b_key:<10} | "
                f"Seed {seed:5d} -> Best: {fitness:.6e}"
            )

    # Format structured results
    structured_results: Dict[str, Any] = {}
    for var_name, var_info in VARIANTS.items():
        structured_results[var_name] = {
            "description": var_info["description"],
            "strategy_recomb_scale": var_info["strategy_recomb_scale"],
            "circular_mutation_operator": var_info["circular_mutation_operator"],
        }
        for b_key in BENCHMARKS:
            seed_map = raw_results[var_name][b_key]
            vals = [seed_map[s] for s in SEEDS]
            structured_results[var_name][b_key] = {
                "min": float(np.min(vals)),
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "max": float(np.max(vals)),
                "per_seed": {str(s): seed_map[s] for s in SEEDS},
            }

    # Save to local worktree
    out_dir = PROJECT_ROOT / "experiments" / "ablation_progress"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "factorial_ablation_results.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(structured_results, f, indent=2)
    print(f"\n[OK] Results successfully saved to: {out_path}")

    # Also copy to main repo if available
    main_repo_path = Path(r"c:\Users\bejmo\Documents\documents\GitHub\TFG\experiments\ablation_progress\factorial_ablation_results.json")
    try:
        main_repo_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_path, main_repo_path)
        print(f"[OK] Results successfully copied to main repo: {main_repo_path}")
    except Exception as e:
        print(f"[WARNING] Could not copy to main repo: {e}")

    # Summary table output
    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"ABLATION EXECUTION COMPLETED IN {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print("=" * 70)
    for var_name in VARIANTS:
        print(f"\nVariant: {var_name}")
        print(f"  Description: {VARIANTS[var_name]['description']}")
        print(f"  {'Benchmark':<12} | {'Min':<14} | {'Mean':<14} | {'Std':<14} | {'Max':<14}")
        print("  " + "-" * 66)
        for b_key in ["sphere", "coupled", "rastrigin"]:
            stats = structured_results[var_name][b_key]
            print(
                f"  {b_key:<12} | {stats['min']:<14.6e} | {stats['mean']:<14.6e} | "
                f"{stats['std']:<14.6e} | {stats['max']:<14.6e}"
            )
    print("=" * 70)


if __name__ == "__main__":
    main()
