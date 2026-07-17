"""
Main project entrypoint.

This module exposes:
- The heterogeneous ES optimization loop.
- A simple terminal interface to run model, benchmarks, smoke, or all.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np

from src.experiments.benchmarks.functions import BENCHMARKS
from src.experiments.runners import custom_es, isotropic_es, sep_cma_es, standard_cma_es

from . import config as cfg
from .engine import Individual, clone_individual, create_random_individual, mutate_individual, select_survivors

RUNNERS = {
    "Prototype 1": custom_es.run,
    "Isotropic ES (Simple Self-Adaptive)": isotropic_es.run,
    "sep-CMA-ES (Separable CMA-ES)": sep_cma_es.run,
    "Standard CMA-ES (Upper Bound)": standard_cma_es.run,
}

DEFAULT_MODEL_GENERATIONS = 500
DEFAULT_MODEL_LOG_INTERVAL = 10
DEFAULT_BENCHMARK_DIM = 10
DEFAULT_BENCHMARK_MAX_EVALS = 10_000
DEFAULT_BENCHMARK_SEEDS = 10
DEFAULT_SMOKE_MODEL_GENERATIONS = 5
DEFAULT_SMOKE_DIM = 5
DEFAULT_SMOKE_MAX_EVALS = 200
DEFAULT_SMOKE_SEEDS = 2


@dataclass
class ModelSummary:
    workflow: str
    status: str
    elapsed_seconds: float
    generations: int


@dataclass
class BenchmarkSummary:
    workflow: str
    status: str
    elapsed_seconds: float
    dimensions: int
    max_evals: int
    seeds: int


def evaluate(ind: Individual) -> float:
    """
    Default fitness function: Sphere.
    """
    return float(np.sum(ind.real ** 2))


def run() -> None:
    """Execute the ES generational loop."""
    rng = np.random.default_rng(cfg.SEED)

    parents: list[Individual] = [create_random_individual(rng) for _ in range(cfg.MU)]
    for ind in parents:
        ind.fitness = evaluate(ind)

    print(f"[ES] Mode={cfg.SELECTION_MODE} | mu={cfg.MU}, lambda={cfg.LAMBDA}")
    print(
        f"[ES] Dimensions: real={cfg.N_REAL}, count={cfg.N_COUNT}, "
        f"binary={cfg.N_BINARY}, cat={cfg.N_CATEGORICAL}, fuzzy={cfg.N_FUZZY}"
    )
    print(f"[ES] Starting optimization for {cfg.MAX_GENERATIONS} generations...\n")

    for gen in range(1, cfg.MAX_GENERATIONS + 1):
        childs: list[Individual] = []
        for _ in range(cfg.LAMBDA):
            parent_idx = rng.integers(0, cfg.MU)
            child = clone_individual(parents[parent_idx])
            mutate_individual(child, rng)
            childs.append(child)

        for child in childs:
            child.fitness = evaluate(child)

        parents = select_survivors(parents, childs)

        best_fitness: float = parents[0].fitness
        if gen % cfg.LOG_INTERVAL == 0 or gen == 1:
            print(f"  Gen {gen:>4d} | Best fitness: {best_fitness:.6e}")

        if cfg.ENABLE_TRACE_LOG:
            _log_trace(gen, parents[0])

    print("\n[ES] Optimization complete.")
    print(f"[ES] Final best fitness: {parents[0].fitness:.6e}")
    print(f"[ES] Best genome (flattened): {parents[0].genome}")


def run_model(generations: int | None = None, log_interval: int | None = None) -> ModelSummary:
    start = time.perf_counter()

    old_generations = cfg.MAX_GENERATIONS
    old_log_interval = cfg.LOG_INTERVAL

    if generations is not None:
        cfg.MAX_GENERATIONS = generations
    if log_interval is not None:
        cfg.LOG_INTERVAL = log_interval

    try:
        print("[WORKFLOW] model")
        print("[STATUS] running")
        print(f"[CONFIG] generations={cfg.MAX_GENERATIONS} | log_interval={cfg.LOG_INTERVAL}")
        run()
        status = "success"
    finally:
        cfg.MAX_GENERATIONS = old_generations
        cfg.LOG_INTERVAL = old_log_interval

    elapsed = time.perf_counter() - start
    print("[WORKFLOW] model")
    print(f"[STATUS] {status}")
    print(f"[ELAPSED] {elapsed:.2f}s")

    return ModelSummary(
        workflow="model",
        status=status,
        elapsed_seconds=elapsed,
        generations=cfg.MAX_GENERATIONS,
    )


def run_benchmarks(dim: int = 10, max_evals: int = 10_000, n_seeds: int = 10) -> BenchmarkSummary:
    start = time.perf_counter()
    seeds = list(range(1, n_seeds + 1))

    print("[WORKFLOW] benchmarks")
    print("[STATUS] running")
    print(f"[CONFIG] dim={dim} | max_evals={max_evals} | seeds={n_seeds}")

    results = {}
    for bname, binfo in BENCHMARKS.items():
        results[bname] = {}
        for rname, runner in RUNNERS.items():
            traces = []
            for seed in seeds:
                trace = runner(binfo["fn"], dim, binfo["bounds"], seed, max_evals)
                traces.append(trace)
            results[bname][rname] = np.array(traces)
            median_final = np.median(traces, axis=0)[-1]
            print(f"  {bname:12s} | {rname:36s} | median final: {median_final:.4e}")

    print("\n" + "=" * 170)
    print("RESULTS SUMMARY - Median final fitness")
    print("=" * 170)
    header = f"{'Benchmark':<14}" + "".join(f"{r:<38}" for r in RUNNERS.keys())
    print(header)
    print("-" * len(header))
    for bname in BENCHMARKS:
        row = f"{bname:<14}"
        for rname in RUNNERS:
            med = np.median(results[bname][rname][:, -1])
            row += f"{med:<38.4e}"
        print(row)

    elapsed = time.perf_counter() - start
    print("\n[WORKFLOW] benchmarks")
    print("[STATUS] success")
    print(f"[ELAPSED] {elapsed:.2f}s")

    return BenchmarkSummary(
        workflow="benchmarks",
        status="success",
        elapsed_seconds=elapsed,
        dimensions=dim,
        max_evals=max_evals,
        seeds=n_seeds,
    )


def _log_trace(gen: int, best: Individual) -> None:
    genome_str = np.array2string(best.genome, precision=4, separator=", ")
    print(f"  [TRACE] Gen {gen:>4d} genome: {genome_str}")


def _prompt_int(label: str, default: int, minimum: int = 1) -> int:
    while True:
        raw = input(f"{label} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Invalid number. Please enter an integer.")
            continue
        if value < minimum:
            print(f"Value must be >= {minimum}.")
            continue
        return value


def _prompt_model_options(default_generations: int, default_log_interval: int) -> dict[str, int]:
    print("\n----------------------------------------------------------------------------")
    print("Model options (press Enter to use defaults)")
    print("----------------------------------------------------------------------------")
    generations = _prompt_int("Generations", default_generations)
    log_interval = _prompt_int("Log interval", default_log_interval)
    return {"generations": generations, "log_interval": log_interval}


def _prompt_benchmark_options(default_dim: int, default_max_evals: int, default_seeds: int) -> dict[str, int]:
    print("\n----------------------------------------------------------------------------")
    print("Benchmark options (press Enter to use defaults)")
    print("----------------------------------------------------------------------------")
    dim = _prompt_int("Dimension", default_dim)
    max_evals = _prompt_int("Max evals", default_max_evals)
    seeds = _prompt_int("Seeds", default_seeds)
    return {"dim": dim, "max_evals": max_evals, "seeds": seeds}


def _select_interactive_mode() -> tuple[str, dict[str, int]]:
    mode_map = {
        "1": "model",
        "2": "benchmarks",
        "3": "smoke",
        "4": "all",
        "5": "exit",
        "model": "model",
        "benchmarks": "benchmarks",
        "smoke": "smoke",
        "all": "all",
        "exit": "exit",
    }

    while True:
        print("\n----------------------------------------------------------------------------")
        print("Select workflow")
        print("----------------------------------------------------------------------------")
        print("  1) model")
        print("  2) benchmarks")
        print("  3) smoke")
        print("  4) all")
        print("  5) exit")
        choice = input("Choice: ").strip().lower()
        mode = mode_map.get(choice)
        if mode is None:
            print("Invalid choice. Try again.")
            continue
        if mode == "exit":
            return "exit", {}
        if mode == "model":
            opts = _prompt_model_options(DEFAULT_MODEL_GENERATIONS, DEFAULT_MODEL_LOG_INTERVAL)
            return mode, opts
        if mode == "benchmarks":
            opts = _prompt_benchmark_options(
                DEFAULT_BENCHMARK_DIM,
                DEFAULT_BENCHMARK_MAX_EVALS,
                DEFAULT_BENCHMARK_SEEDS,
            )
            return mode, opts
        if mode == "smoke":
            model_opts = _prompt_model_options(DEFAULT_SMOKE_MODEL_GENERATIONS, max(1, DEFAULT_SMOKE_MODEL_GENERATIONS // 2))
            bench_opts = _prompt_benchmark_options(DEFAULT_SMOKE_DIM, DEFAULT_SMOKE_MAX_EVALS, DEFAULT_SMOKE_SEEDS)
            return mode, {
                "model_generations": model_opts["generations"],
                "dim": bench_opts["dim"],
                "max_evals": bench_opts["max_evals"],
                "seeds": bench_opts["seeds"],
            }
        model_opts = _prompt_model_options(DEFAULT_MODEL_GENERATIONS, DEFAULT_MODEL_LOG_INTERVAL)
        bench_opts = _prompt_benchmark_options(
            DEFAULT_BENCHMARK_DIM,
            DEFAULT_BENCHMARK_MAX_EVALS,
            DEFAULT_BENCHMARK_SEEDS,
        )
        return mode, {
            "generations": model_opts["generations"],
            "log_interval": model_opts["log_interval"],
            "dim": bench_opts["dim"],
            "max_evals": bench_opts["max_evals"],
            "seeds": bench_opts["seeds"],
        }


def _resolve_int(value: int | None, default: int) -> int:
    if value is None:
        return default
    return value


def _run_selected_mode(mode: str, options: dict[str, int | None]) -> None:
    if mode == "model":
        run_model(generations=options.get("generations"), log_interval=options.get("log_interval"))
        return

    if mode == "benchmarks":
        run_benchmarks(
            dim=_resolve_int(options.get("dim"), DEFAULT_BENCHMARK_DIM),
            max_evals=_resolve_int(options.get("max_evals"), DEFAULT_BENCHMARK_MAX_EVALS),
            n_seeds=_resolve_int(options.get("seeds"), DEFAULT_BENCHMARK_SEEDS),
        )
        return

    if mode == "smoke":
        model_generations = _resolve_int(options.get("model_generations"), DEFAULT_SMOKE_MODEL_GENERATIONS)
        dim = _resolve_int(options.get("dim"), DEFAULT_SMOKE_DIM)
        max_evals = _resolve_int(options.get("max_evals"), DEFAULT_SMOKE_MAX_EVALS)
        seeds = _resolve_int(options.get("seeds"), DEFAULT_SMOKE_SEEDS)

        print("[WORKFLOW] smoke")
        print("[STATUS] running")
        run_model(generations=model_generations, log_interval=max(1, model_generations // 2))
        run_benchmarks(dim=dim, max_evals=max_evals, n_seeds=seeds)
        print("[WORKFLOW] smoke")
        print("[STATUS] success")
        return

    if mode == "all":
        generations = options.get("generations")
        log_interval = options.get("log_interval")
        dim = _resolve_int(options.get("dim"), DEFAULT_BENCHMARK_DIM)
        max_evals = _resolve_int(options.get("max_evals"), DEFAULT_BENCHMARK_MAX_EVALS)
        seeds = _resolve_int(options.get("seeds"), DEFAULT_BENCHMARK_SEEDS)

        print("[WORKFLOW] all")
        print("[STATUS] running")
        run_model(generations=generations, log_interval=log_interval)
        run_benchmarks(dim=dim, max_evals=max_evals, n_seeds=seeds)
        print("[WORKFLOW] all")
        print("[STATUS] success")
        return

    raise ValueError(f"Unsupported mode: {mode}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.es_model.main",
        description="Run model, benchmarks, smoke, or all from a single entrypoint.",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=["model", "benchmarks", "smoke", "all"],
        help="Optional direct mode. If omitted, interactive menu is shown.",
    )
    parser.add_argument("--generations", type=int, default=None, help="Model generations")
    parser.add_argument("--log-interval", type=int, default=None, help="Model log interval")
    parser.add_argument("--model-generations", type=int, default=None, help="Smoke model generations")
    parser.add_argument("--dim", type=int, default=None, help="Benchmark dimensionality")
    parser.add_argument("--max-evals", type=int, default=None, help="Benchmark evaluation budget")
    parser.add_argument("--seeds", type=int, default=None, help="Number of benchmark seeds")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    start = time.perf_counter()

    if args.mode is None:
        mode, options = _select_interactive_mode()
        if mode == "exit":
            print("Exiting.")
            return
        _run_selected_mode(mode, options)
    else:
        _run_selected_mode(
            args.mode,
            {
                "generations": args.generations,
                "log_interval": args.log_interval,
                "model_generations": args.model_generations,
                "dim": args.dim,
                "max_evals": args.max_evals,
                "seeds": args.seeds,
            },
        )

    elapsed = time.perf_counter() - start
    print(f"[TOTAL ELAPSED] {elapsed:.2f}s")


if __name__ == "__main__":
    main()
