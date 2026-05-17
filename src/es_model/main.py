"""
main.py — Project entry point: execution loop, fitness evaluation, and logging.

Runs the heterogeneous (mu, lambda)-ES optimization loop end-to-end.
A default benchmark objective (Sphere) is provided; replace `evaluate`
with any callable accepting an Individual and returning a float (lower=better).
"""

import numpy as np

from . import config as cfg
from .engine import Individual, clone_individual, create_random_individual, mutate_individual, select_survivors


# =============================================================================
# 1. Fitness Hook / Default Benchmark Objective
# =============================================================================

def evaluate(ind: Individual) -> float:
    """
    Default fitness function: Sphere (sum of squared real components).

    Replace this with any domain-specific objective. Lower values are better.
    The genome property gives a full flattened view for multi-block objectives.
    """
    # Sphere on the real block — simple, smooth, unimodal baseline.
    return float(np.sum(ind.real ** 2))


# =============================================================================
# 2. Executable Optimization Loop
# =============================================================================

def run() -> None:
    """
    Execute the ES generational loop:
      1. Initialize population.
      2. Evaluate fitness.
      3. Generate offspring (clone + mutate).
      4. Evaluate offspring.
      5. Select survivors.
      6. Log convergence info.
      7. Repeat until MAX_GENERATIONS.
    """
    rng = np.random.default_rng(cfg.SEED)

    # ── Initialize parent population ──────────────────────────────────────
    parents: list[Individual] = [
        create_random_individual(rng) for _ in range(cfg.MU)
    ]
    for ind in parents:
        ind.fitness = evaluate(ind)

    print(f"[ES] Mode={cfg.SELECTION_MODE} | mu={cfg.MU}, lambda={cfg.LAMBDA}")
    print(f"[ES] Dimensions: real={cfg.N_REAL}, count={cfg.N_COUNT}, "
          f"binary={cfg.N_BINARY}, cat={cfg.N_CATEGORICAL}, fuzzy={cfg.N_FUZZY}")
    print(f"[ES] Starting optimization for {cfg.MAX_GENERATIONS} generations...\n")

    # ── Generational Loop ─────────────────────────────────────────────────
    for gen in range(1, cfg.MAX_GENERATIONS + 1):
        # 3. Generate offspring by cloning random parents and mutating.
        offspring: list[Individual] = []
        for _ in range(cfg.LAMBDA):
            parent_idx = rng.integers(0, cfg.MU)
            child = clone_individual(parents[parent_idx])
            mutate_individual(child, rng)
            offspring.append(child)

        # 4. Evaluate offspring fitness.
        for child in offspring:
            child.fitness = evaluate(child)

        # 5. Select survivors for next generation.
        parents = select_survivors(parents, offspring)

        # 6. Convergence logging.
        best_fitness: float = parents[0].fitness
        if gen % cfg.LOG_INTERVAL == 0 or gen == 1:
            print(f"  Gen {gen:>4d} | Best fitness: {best_fitness:.6e}")

        # Optional genome trace logging.
        if cfg.ENABLE_TRACE_LOG:
            _log_trace(gen, parents[0])

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n[ES] Optimization complete.")
    print(f"[ES] Final best fitness: {parents[0].fitness:.6e}")
    print(f"[ES] Best genome (flattened): {parents[0].genome}")


# =============================================================================
# 3. Trace Logging Helper
# =============================================================================

def _log_trace(gen: int, best: Individual) -> None:
    """Print flattened genome vector of the best individual for trace analysis."""
    genome_str = np.array2string(best.genome, precision=4, separator=", ")
    print(f"  [TRACE] Gen {gen:>4d} genome: {genome_str}")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    run()
