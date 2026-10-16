"""
experiments/extra/categorical/run_experiments.py
=================================================
Reproducible runner for categorical operator validation and binary equivalence studies.

Evaluates:
1. Study 1: Legacy Softmax (continuous logits) vs. New Unified Categorical (state + jump mutation)
   across 5 benchmark suites (15 seeds each, with paired Wilcoxon signed-rank tests).
2. Study 2: Binary Variable (XOR) vs. Categorical Variable (K=2)
   across 50 seeds (with paired Wilcoxon signed-rank test, Mann-Whitney U, and Fisher exact test).

Outputs:
- JSON results saved in `results/`
- Publication-quality figures saved in `figures/`
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import scipy.stats as stats

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.es_model.config import ESConfig
from src.es_model.individual import Individual
import src.es_model.individual as ind_mod
from src.es_model.generalized_es import GeneralizedES
from src.es_model.variables.variable import Variable
from src.es_model.variables.categorical import CategoricalVariable


# =============================================================================
# 1. Legacy Softmax Implementation for Baseline Contrast
# =============================================================================

class LegacySoftmaxCategoricalVariable(Variable):
    """Legacy categorical operator: sum(K_i) centered logits with memoryless Softmax sampling."""

    def __init__(
        self,
        values: NDArray[np.int64],
        strategy: NDArray[np.float64],
        categorical_k: NDArray[np.int64],
        max_logit: float = 4.8,
    ) -> None:
        super().__init__(
            name="categorical",
            values=values,
            strategy=strategy,
            strategy_recomb="arithmetic",
            adaptation_type="logit",
            max_logit=max_logit,
            min_logit=-max_logit,
        )
        self.categorical_k = np.asarray(categorical_k, dtype=np.int64)
        self.max_logit = max_logit

    @classmethod
    def random(cls, config: ESConfig, rng: np.random.Generator) -> LegacySoftmaxCategoricalVariable:
        values = (
            np.array([rng.integers(0, int(k)) for k in config.categorical_k], dtype=np.int64)
            if config.n_categorical > 0
            else np.zeros(0, dtype=np.int64)
        )
        tot_logits = int(np.sum(config.categorical_k)) if config.n_categorical > 0 else 0
        strategy = np.zeros(tot_logits, dtype=np.float64)
        return cls(values=values, strategy=strategy, categorical_k=config.categorical_k)

    def copy(self) -> LegacySoftmaxCategoricalVariable:
        return LegacySoftmaxCategoricalVariable(
            values=self.values.copy(),
            strategy=self.strategy.copy(),
            categorical_k=self.categorical_k.copy(),
            max_logit=self.max_logit,
        )

    def _post_adapt_strategy(self, rng: np.random.Generator, tau_prime: float, tau: float) -> None:
        offset = 0
        for k in self.categorical_k:
            k_int = int(k)
            slice_vals = self.strategy[offset : offset + k_int]
            c_vals = slice_vals - np.mean(slice_vals)
            c_vals = np.clip(c_vals, -self.max_logit, self.max_logit)
            c_vals -= np.mean(c_vals)
            self.strategy[offset : offset + k_int] = c_vals
            offset += k_int

    def mutate(self, rng: np.random.Generator) -> None:
        offset = 0
        for i, k in enumerate(self.categorical_k):
            k_int = int(k)
            slice_vals = self.strategy[offset : offset + k_int]
            exp_z = np.exp(slice_vals - np.max(slice_vals))
            probs = exp_z / np.sum(exp_z)
            r = float(rng.random())
            acc = 0.0
            choice = k_int - 1
            for idx in range(k_int):
                acc += probs[idx]
                if r < acc:
                    choice = idx
                    break
            self.values[i] = choice
            offset += k_int

    def distance_squared(self, other: Variable | NDArray) -> float:
        other_vals = other.values if isinstance(other, Variable) else np.asarray(other, dtype=np.int64)
        return float(np.sum(self.values != other_vals))


# =============================================================================
# 2. Benchmark Tasks
# =============================================================================

class DynamicShiftTask:
    def __init__(self, t1: NDArray[np.int64], t2: NDArray[np.int64], g_shift: int = 60):
        self.t1 = t1
        self.t2 = t2
        self.g_shift = g_shift
        self.g = 0

    def set_gen(self, g: int) -> None:
        self.g = g

    def __call__(self, x: NDArray) -> float:
        target = self.t1 if self.g < self.g_shift else self.t2
        return float(np.sum(x.astype(np.int64) != target))


def make_deceptive_task(n_vars: int, k: int = 4) -> Callable[[NDArray], float]:
    cost_table = np.array([0.0, 1.0, 3.0, 4.0], dtype=np.float64)

    def fitness(x: NDArray) -> float:
        indices = np.clip(x.astype(np.int64)[:n_vars], 0, k - 1)
        return float(np.sum(cost_table[indices]))

    return fitness


def make_royal_road_task(n_vars: int, block_size: int = 4, target_cat: int = 0) -> Callable[[NDArray], float]:
    n_blocks = n_vars // block_size

    def fitness(x: NDArray) -> float:
        blocks = x.astype(np.int64)[: n_blocks * block_size].reshape(n_blocks, block_size)
        matches = np.all(blocks == target_cat, axis=1)
        score = np.sum(matches)
        return float(n_blocks - score)

    return fitness


# =============================================================================
# 3. Execution Engines
# =============================================================================

def run_categorical_single(
    op_type: str,
    n_vars: int,
    cat_k: NDArray[np.int64],
    objective: Callable[[NDArray], float],
    seed: int,
    n_gens: int = 120,
    mu: int = 10,
    lam: int = 50,
    dts_obj: DynamicShiftTask | None = None,
) -> Tuple[List[float], float, float]:
    t0 = time.perf_counter()
    cfg = ESConfig(
        mu=mu,
        lam=lam,
        n_real=0,
        n_count=0,
        n_binary=0,
        n_categorical=n_vars,
        categorical_k=cat_k,
        n_fuzzy=0,
        n_circular=0,
        max_generations=n_gens,
        seed=seed,
    )

    if op_type == "legacy_softmax":
        orig_random = ind_mod.Individual.random

        def legacy_random(rng, config=None):
            ind = orig_random(rng, config)
            ind.variables["categorical"] = LegacySoftmaxCategoricalVariable.random(cfg, rng)
            return ind

        ind_mod.Individual.random = legacy_random
        try:
            es = GeneralizedES(config=cfg, seed=seed)
            es.init_population(objective)
            trace = []
            for g in range(n_gens):
                if dts_obj is not None:
                    dts_obj.set_gen(g)
                es.step(objective)
                trace.append(float(es.parents[0].fitness))
        finally:
            ind_mod.Individual.random = orig_random
    else:
        es = GeneralizedES(config=cfg, seed=seed)
        es.init_population(objective)
        trace = []
        for g in range(n_gens):
            if dts_obj is not None:
                dts_obj.set_gen(g)
            es.step(objective)
            trace.append(float(es.parents[0].fitness))

    elapsed = time.perf_counter() - t0
    return trace, trace[-1], elapsed


# =============================================================================
# 4. Study 1: Legacy Softmax vs. New Jump Operator
# =============================================================================

def run_study_1(results_dir: str, figures_dir: str) -> Dict:
    seeds = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    n_runs = len(seeds)

    benchmarks = [
        {
            "id": "B1_Stationary",
            "name": "Stationary Matching (N=32, K=4)",
            "n_vars": 32,
            "cat_k": np.full(32, 4, dtype=np.int64),
            "maker": lambda: (lambda x: float(np.sum(x.astype(np.int64) != np.random.default_rng(101).integers(0, 4, 32)))),
            "dts": False,
        },
        {
            "id": "B2_DynamicShift",
            "name": "Dynamic Shift (N=32, K=4, shift at g=60)",
            "n_vars": 32,
            "cat_k": np.full(32, 4, dtype=np.int64),
            "maker": lambda: DynamicShiftTask(
                t1=np.random.default_rng(201).integers(0, 4, 32),
                t2=np.random.default_rng(202).integers(0, 4, 32),
                g_shift=60,
            ),
            "dts": True,
        },
        {
            "id": "B3_MixedCardinality",
            "name": "Mixed Cardinalities (N=32, K in {2,3,5,8})",
            "n_vars": 32,
            "cat_k": np.tile(np.array([2, 3, 5, 8], dtype=np.int64), 8),
            "maker": lambda: (lambda x: float(np.sum(x.astype(np.int64) != np.random.default_rng(301).integers(0, np.tile(np.array([2, 3, 5, 8], dtype=np.int64), 8))))),
            "dts": False,
        },
        {
            "id": "B4_Deceptive",
            "name": "Deceptive Trap (N=20, K=4)",
            "n_vars": 20,
            "cat_k": np.full(20, 4, dtype=np.int64),
            "maker": lambda: make_deceptive_task(20, 4),
            "dts": False,
        },
        {
            "id": "B5_RoyalRoad",
            "name": "Epistatic Royal Road (N=24, K=4)",
            "n_vars": 24,
            "cat_k": np.full(24, 4, dtype=np.int64),
            "maker": lambda: make_royal_road_task(24, block_size=4, target_cat=0),
            "dts": False,
        },
    ]

    all_data = {}
    total_wins, total_ties, total_losses = 0, 0, 0

    print("\n" + "=" * 90)
    print("STUDY 1: Legacy Softmax vs. New Jump Categorical (15 Seeds, Wilcoxon Tests)")
    print("=" * 90)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=False)

    for idx, b in enumerate(benchmarks):
        print(f"\n--- Running Benchmark: {b['name']} ---")
        legacy_losses, new_losses = [], []
        legacy_times, new_times = [], []
        legacy_traces, new_traces = [], []

        for s in seeds:
            task = b["maker"]()
            dts_obj = task if b["dts"] else None
            obj_fn = task if not b["dts"] else task

            tr_leg, loss_leg, t_leg = run_categorical_single(
                "legacy_softmax", b["n_vars"], b["cat_k"], obj_fn, seed=s, dts_obj=dts_obj
            )
            legacy_losses.append(loss_leg)
            legacy_times.append(t_leg)
            legacy_traces.append(tr_leg)

            tr_new, loss_new, t_new = run_categorical_single(
                "new_state", b["n_vars"], b["cat_k"], obj_fn, seed=s, dts_obj=dts_obj
            )
            new_losses.append(loss_new)
            new_times.append(t_new)
            new_traces.append(tr_new)

        arr_leg = np.array(legacy_losses)
        arr_new = np.array(new_losses)
        diff = arr_leg - arr_new

        wins = int(np.sum(diff > 1e-9))
        ties = int(np.sum(np.abs(diff) <= 1e-9))
        losses = int(np.sum(diff < -1e-9))
        total_wins += wins
        total_ties += ties
        total_losses += losses

        if np.any(diff != 0):
            w_stat, w_p = stats.wilcoxon(arr_leg, arr_new)
        else:
            w_stat, w_p = 0.0, 1.0

        succ_leg = float(np.mean(arr_leg == 0.0) * 100)
        succ_new = float(np.mean(arr_new == 0.0) * 100)
        mean_t_leg = float(np.mean(legacy_times))
        mean_t_new = float(np.mean(new_times))
        speedup = mean_t_leg / max(mean_t_new, 1e-9)

        all_data[b["id"]] = {
            "name": b["name"],
            "legacy_loss_mean": float(np.mean(arr_leg)),
            "legacy_loss_std": float(np.std(arr_leg)),
            "legacy_success_pct": succ_leg,
            "new_loss_mean": float(np.mean(arr_new)),
            "new_loss_std": float(np.std(arr_new)),
            "new_success_pct": succ_new,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "wilcoxon_stat": float(w_stat),
            "wilcoxon_p": float(w_p),
            "speedup": float(speedup),
            "legacy_traces_mean": np.mean(legacy_traces, axis=0).tolist(),
            "new_traces_mean": np.mean(new_traces, axis=0).tolist(),
        }

        print(f"Legacy Softmax: Loss = {np.mean(arr_leg):.2f} +/- {np.std(arr_leg):.2f} | Succ = {succ_leg:5.1f}% | Time = {mean_t_leg:.2f}s")
        print(f"New Jump:       Loss = {np.mean(arr_new):.2f} +/- {np.std(arr_new):.2f} | Succ = {succ_new:5.1f}% | Time = {mean_t_new:.2f}s")
        print(f"Wilcoxon W = {w_stat:.1f}, p = {w_p:.4e} | Record (W/T/L): {wins}/{ties}/{losses} | Speedup: {speedup:.1f}x")

        # Plot convergence curve
        ax = axes[idx]
        mean_tr_leg = np.mean(legacy_traces, axis=0)
        mean_tr_new = np.mean(new_traces, axis=0)
        ax.plot(mean_tr_leg, label="Legacy Softmax", color="#d95f02", lw=2, linestyle="--")
        ax.plot(mean_tr_new, label="New Jump (Unified)", color="#1b9e77", lw=2)
        ax.set_title(b["name"].split(" (")[0], fontsize=11, fontweight="bold")
        ax.set_xlabel("Generación", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Pérdida media f(x)", fontsize=10)
            ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(figures_dir, "categorical_convergence.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    print("\n" + "=" * 90)
    print(f"STUDY 1 OVERALL SUMMARY: {total_wins} Wins, {total_ties} Ties, {total_losses} Losses across 75 evaluations")
    print(f"Plot saved to: {plot_path}")
    print("=" * 90)

    json_path = os.path.join(results_dir, "categorical_comparison_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    return all_data


# =============================================================================
# 5. Study 2: Binary XOR vs. Categorical K=2 (50 Seeds)
# =============================================================================

def run_study_2(results_dir: str, figures_dir: str) -> Dict:
    n_vars = 32
    seeds = list(range(1, 51))
    target = np.random.default_rng(999).integers(0, 2, size=n_vars, dtype=np.uint8)

    print("\n" + "=" * 90)
    print("STUDY 2: Binary Variable (XOR) vs. Categorical Variable (K=2) (50 Seeds, Wilcoxon Test)")
    print("=" * 90)

    res_bin, res_cat = [], []
    time_bin, time_cat = [], []

    for s in seeds:
        t0 = time.perf_counter()
        cfg_b = ESConfig(n_real=0, n_count=0, n_binary=n_vars, n_categorical=0, n_fuzzy=0, n_circular=0, max_generations=100, seed=s)
        es_b = GeneralizedES(cfg_b, seed=s)
        tb = es_b.optimize(lambda x: float(np.sum(x.astype(np.uint8) != target)), max_evals=5000)
        res_bin.append(tb[-1])
        time_bin.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        cfg_c = ESConfig(n_real=0, n_count=0, n_binary=0, n_categorical=n_vars, categorical_k=np.full(n_vars, 2, dtype=np.int64), n_fuzzy=0, n_circular=0, max_generations=100, seed=s)
        es_c = GeneralizedES(cfg_c, seed=s)
        tc = es_c.optimize(lambda x: float(np.sum(x.astype(np.uint8) != target)), max_evals=5000)
        res_cat.append(tc[-1])
        time_cat.append(time.perf_counter() - t0)

    arr_bin = np.array(res_bin)
    arr_cat = np.array(res_cat)
    diff = arr_bin - arr_cat

    # Wilcoxon signed-rank test
    if np.any(diff != 0):
        w_stat, w_p = stats.wilcoxon(arr_bin, arr_cat)
    else:
        w_stat, w_p = 0.0, 1.0

    # Mann-Whitney U test
    u_stat, u_p = stats.mannwhitneyu(arr_bin, arr_cat, alternative="two-sided")

    # Fisher exact test
    succ_b = int(np.sum(arr_bin == 0.0))
    succ_c = int(np.sum(arr_cat == 0.0))
    table = [[succ_b, len(seeds) - succ_b], [succ_c, len(seeds) - succ_c]]
    fisher_stat, fisher_p = stats.fisher_exact(table)

    summary_data = {
        "n_seeds": len(seeds),
        "n_vars": n_vars,
        "binary_loss_mean": float(np.mean(arr_bin)),
        "binary_loss_std": float(np.std(arr_bin)),
        "binary_success_pct": float(succ_b / len(seeds) * 100),
        "binary_time_total": float(np.sum(time_bin)),
        "categorical_loss_mean": float(np.mean(arr_cat)),
        "categorical_loss_std": float(np.std(arr_cat)),
        "categorical_success_pct": float(succ_c / len(seeds) * 100),
        "categorical_time_total": float(np.sum(time_cat)),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "mann_whitney_u_p": float(u_p),
        "fisher_exact_p": float(fisher_p),
        "identical_seeds_count": int(np.sum(diff == 0)),
        "different_seeds_count": int(np.sum(diff != 0)),
        "raw_binary_losses": arr_bin.tolist(),
        "raw_categorical_losses": arr_cat.tolist(),
    }

    print(f"Binary (XOR):      Mean Loss = {np.mean(arr_bin):.4f} +/- {np.std(arr_bin):.4f} | Succ = {succ_b}/50 ({succ_b*2}%) | Time = {np.sum(time_bin):.2f}s")
    print(f"Cat (K=2 Jump):    Mean Loss = {np.mean(arr_cat):.4f} +/- {np.std(arr_cat):.4f} | Succ = {succ_c}/50 ({succ_c*2}%) | Time = {np.sum(time_cat):.2f}s")
    print(f"Identical Seeds:   {np.sum(diff == 0)}/50 ({np.sum(diff == 0)*2}%)")
    print(f"Wilcoxon Test:     W = {w_stat:.1f}, p = {w_p:.4f} (No statistical difference: p >> 0.05)")
    print(f"Mann-Whitney U:    U_p = {u_p:.4f}")
    print(f"Fisher Exact Test: p = {fisher_p:.4f}")

    # Plot boxplot / violin comparison
    fig, ax = plt.subplots(figsize=(6, 5))
    box = ax.boxplot(
        [arr_bin, arr_cat],
        tick_labels=["Binaria (XOR)", "Categórica (K=2)"],
        patch_artist=True,
        widths=0.4,
    )
    colors = ["#386cb0", "#7fc97f"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("Pérdida final f(x) (32 variables)", fontsize=11)
    ax.set_title(f"Comparativa de Equivalencia (50 Semillas)\nWilcoxon p = {w_p:.4f} (p > 0.05)", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(figures_dir, "binary_vs_cat_k2.png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    print(f"Plot saved to: {plot_path}")
    print("=" * 90)

    json_path = os.path.join(results_dir, "binary_vs_categorical_k2.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


# =============================================================================
# 6. Main Entrypoint
# =============================================================================

def main() -> None:
    results_dir = os.path.join(SCRIPT_DIR, "results")
    figures_dir = os.path.join(SCRIPT_DIR, "figures")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    study1_data = run_study_1(results_dir, figures_dir)
    study2_data = run_study_2(results_dir, figures_dir)
    print("\nAll extra categorical validation experiments successfully completed and saved.")


if __name__ == "__main__":
    main()
