"""Unified terminal runner for model and experiment workflows."""

from __future__ import annotations

import argparse
import time

from src.cli.benchmark_runner import run_benchmarks
from src.cli.model_runner import run_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.runner",
        description="Unified workflow runner for model and benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    model = subparsers.add_parser("model", help="Run model optimization workflow")
    model.add_argument("--generations", type=int, default=None, help="Override max generations")
    model.add_argument("--log-interval", type=int, default=None, help="Override logging interval")

    benchmarks = subparsers.add_parser("benchmarks", help="Run benchmark comparison workflow")
    benchmarks.add_argument("--dim", type=int, default=10, help="Problem dimensionality")
    benchmarks.add_argument("--max-evals", type=int, default=10000, help="Evaluation budget")
    benchmarks.add_argument("--seeds", type=int, default=10, help="Number of random seeds")

    smoke = subparsers.add_parser("smoke", help="Run fast smoke validation")
    smoke.add_argument("--model-generations", type=int, default=5, help="Fast model generations for smoke run")
    smoke.add_argument("--dim", type=int, default=5, help="Smoke benchmark dimensionality")
    smoke.add_argument("--max-evals", type=int, default=200, help="Smoke benchmark eval budget")
    smoke.add_argument("--seeds", type=int, default=2, help="Smoke benchmark seeds")

    all_cmd = subparsers.add_parser("all", help="Run model then benchmarks")
    all_cmd.add_argument("--generations", type=int, default=None, help="Override model max generations")
    all_cmd.add_argument("--log-interval", type=int, default=None, help="Override model logging interval")
    all_cmd.add_argument("--dim", type=int, default=10, help="Benchmark dimensionality")
    all_cmd.add_argument("--max-evals", type=int, default=10000, help="Benchmark eval budget")
    all_cmd.add_argument("--seeds", type=int, default=10, help="Benchmark seed count")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    start = time.perf_counter()

    if args.command == "model":
        run_model(generations=args.generations, log_interval=args.log_interval)
    elif args.command == "benchmarks":
        run_benchmarks(dim=args.dim, max_evals=args.max_evals, n_seeds=args.seeds)
    elif args.command == "smoke":
        print("[WORKFLOW] smoke")
        print("[STATUS] running")
        run_model(generations=args.model_generations, log_interval=max(1, args.model_generations // 2))
        run_benchmarks(dim=args.dim, max_evals=args.max_evals, n_seeds=args.seeds)
        print("[WORKFLOW] smoke")
        print("[STATUS] success")
    elif args.command == "all":
        print("[WORKFLOW] all")
        print("[STATUS] running")
        run_model(generations=args.generations, log_interval=args.log_interval)
        run_benchmarks(dim=args.dim, max_evals=args.max_evals, n_seeds=args.seeds)
        print("[WORKFLOW] all")
        print("[STATUS] success")

    elapsed = time.perf_counter() - start
    print(f"[TOTAL ELAPSED] {elapsed:.2f}s")


if __name__ == "__main__":
    main()
