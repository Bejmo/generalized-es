"""Benchmark workflow runner used by the unified CLI."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.experiments.benchmarks.functions import BENCHMARKS
from src.experiments.runners import custom_es, differential_evolution, one_plus_one_es, pso


RUNNERS = {
    "Custom ES (mu,lam)": custom_es.run,
    "Diff. Evolution": differential_evolution.run,
    "PSO": pso.run,
    "(1+1)-ES 1/5": one_plus_one_es.run,
}


@dataclass
class BenchmarkSummary:
    workflow: str
    status: str
    elapsed_seconds: float
    dimensions: int
    max_evals: int
    seeds: int


def run_benchmarks(dim: int = 10, max_evals: int = 10_000, n_seeds: int = 10) -> BenchmarkSummary:
    start = time.perf_counter()
    seeds = list(range(1, n_seeds + 1))

    print(f"[WORKFLOW] benchmarks")
    print(f"[STATUS] running")
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
            print(f"  {bname:12s} | {rname:20s} | median final: {median_final:.4e}")

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY - Median final fitness")
    print("=" * 80)
    header = f"{'Benchmark':<14}" + "".join(f"{r:<22}" for r in RUNNERS.keys())
    print(header)
    print("-" * len(header))
    for bname in BENCHMARKS:
        row = f"{bname:<14}"
        for rname in RUNNERS:
            med = np.median(results[bname][rname][:, -1])
            row += f"{med:<22.4e}"
        print(row)

    elapsed = time.perf_counter() - start
    print(f"\n[WORKFLOW] benchmarks")
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
