"""Model workflow runner used by the unified CLI."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.es_model import config as cfg
from src.es_model.main import run as run_model_loop


@dataclass
class ModelSummary:
    workflow: str
    status: str
    elapsed_seconds: float
    generations: int


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
        run_model_loop()
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
