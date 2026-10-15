"""
Statistical Hypothesis Testing Module for Generalized ES TFG Experiments.
Computes paired two-sided Wilcoxon signed-rank tests with Holm-Bonferroni
family-wise error correction across 10 random seeds for Experiments 1 and 3.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats


def holm_bonferroni(p_values):
    """
    Applies the step-down Holm-Bonferroni correction to an array/list of p-values.
    Returns:
        adjusted_p_values: np.ndarray of same shape with adjusted p-values
        significant: np.ndarray of boolean decisions at alpha=0.05
    """
    p_vals = np.asarray(p_values, dtype=float)
    m = len(p_vals)
    if m == 0:
        return np.array([]), np.array([], dtype=bool)

    # Sort indices
    sort_idx = np.argsort(p_vals)
    inv_idx = np.empty(m, dtype=int)
    inv_idx[sort_idx] = np.arange(m)

    sorted_p = p_vals[sort_idx]
    # Step-down adjustment: p_adj_i = (m - i) * p_i
    factors = np.arange(m, 0, -1)
    adjusted_sorted = sorted_p * factors

    # Enforce monotonicity: cummax
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    # Cap at 1.0
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    # Restore original order
    adjusted_p = adjusted_sorted[inv_idx]
    significant = adjusted_p < 0.05
    return adjusted_p, significant


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_experiment_1():
    """
    Computes Wilcoxon signed-rank tests for Experiment 1 (Table 5.1).
    Proposed ES vs [Isotropic ES, sep-CMA-ES, Full CMA-ES]
    across 4 benchmarks: Sphere, Rastrigin, Rosenbrock, Ackley.
    """
    exp1_file = REPO_ROOT / "experiments/data/experiment_1/exp1_traces.npz"
    if not exp1_file.exists():
        raise FileNotFoundError(f"Missing {exp1_file}")

    data = np.load(exp1_file, allow_pickle=True)
    benchmarks = ["Sphere", "Rastrigin", "Rosenbrock", "Ackley"]
    opponents = [
        ("Isotropic ES", "Isotropic_ES_Simple_Self_Adaptive"),
        ("sep-CMA-ES", "sep_CMA_ES_Separable_CMA_ES"),
        ("Full CMA-ES", "Standard_CMA_ES_Upper_Bound"),
    ]

    results = {}
    print("=" * 80)
    print("EXPERIMENT 1: Continuous Baseline Validation (10 Seeds, 10,000 Evaluations)")
    print("=" * 80)

    for bench in benchmarks:
        prop_key = f"{bench}__Prototype_1"
        prop_final = data[prop_key][:, -1]  # (10,)
        
        bench_records = []
        raw_p_list = []

        for opp_label, opp_suffix in opponents:
            opp_key = f"{bench}__{opp_suffix}"
            opp_final = data[opp_key][:, -1]

            # Differences: proposed - opponent
            # Note: lower fitness is better (minimization)
            diff = prop_final - opp_final
            wins = np.sum(diff < 0)
            losses = np.sum(diff > 0)
            ties = np.sum(diff == 0)

            # Two-sided Wilcoxon signed-rank test
            if np.all(diff == 0):
                w_stat, p_val = 0.0, 1.0
            else:
                try:
                    res = stats.wilcoxon(prop_final, opp_final, alternative="two-sided")
                    w_stat = res.statistic
                    p_val = res.pvalue
                except Exception as e:
                    w_stat, p_val = np.nan, 1.0

            raw_p_list.append(p_val)
            bench_records.append({
                "benchmark": bench,
                "opponent": opp_label,
                "prop_mean": np.mean(prop_final),
                "opp_mean": np.mean(opp_final),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "w_stat": w_stat,
                "raw_p": p_val,
            })

        # Apply Holm-Bonferroni correction within the benchmark family (m=3)
        adj_p, sig = holm_bonferroni(raw_p_list)

        for i, rec in enumerate(bench_records):
            rec["adj_p"] = adj_p[i]
            rec["significant"] = sig[i]
            if sig[i]:
                if rec["wins"] > rec["losses"]:
                    verdict = "Proposed Wins (dagger)"
                    symbol = "\\dagger"
                else:
                    verdict = "Baseline Wins (ddagger)"
                    symbol = "\\ddagger"
            else:
                verdict = "Statistical Tie (approx)"
                symbol = "\\approx"
            rec["verdict"] = verdict
            rec["symbol"] = symbol

            print(f"[{bench}] vs {rec['opponent']:<12} | Wins/Ties/Losses: {rec['wins']}/{rec['ties']}/{rec['losses']} "
                  f"| W={rec['w_stat']:<5.1f} | p_raw={rec['raw_p']:<9.4e} | p_adj={rec['adj_p']:<9.4e} | {verdict}")

        results[bench] = bench_records
        print("-" * 80)

    return results


def run_experiment_3():
    """
    Computes Wilcoxon signed-rank tests for Experiment 3 (Table 5.3).
    Generalized ES vs [Uniform Random Search, Continuous CMA-ES Rounding]
    across 3 heterogeneous benchmarks: Sphere, Coupled, Rastrigin.
    """
    exp3_file = REPO_ROOT / "experiments/data/experiment_3/exp3_traces.npz"
    baseline_file = REPO_ROOT / "experiments/baselines_and_ablations/results/random_search_results.json"

    data3 = np.load(exp3_file, allow_pickle=True)
    with open(baseline_file, "r") as f:
        base_data = json.load(f)

    benchmarks = [
        ("Generalized_Sphere", "sphere", "Heterogeneous Sphere"),
        ("Generalized_Coupled", "coupled", "Heterogeneous Coupled"),
        ("Generalized_Rastrigin", "rastrigin", "Heterogeneous Rastrigin"),
    ]

    seeds = [42, 143, 244, 345, 446, 547, 648, 749, 850, 951]
    results = {}

    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Heterogeneous Benchmarks Validation (10 Seeds, 20,000 Evaluations)")
    print("=" * 80)

    for bench_key, json_key, bench_name in benchmarks:
        prop_traces = data3[f"{bench_key}__traces"]
        prop_final = prop_traces[:, -1]

        rs_dict = base_data["random_search"][json_key]["per_seed"]
        cma_dict = base_data["cma_es_relaxation"][json_key]["per_seed"]

        rs_final = np.array([rs_dict[str(s)] for s in seeds])
        cma_final = np.array([cma_dict[str(s)] for s in seeds])

        opponents = [
            ("Random Search", rs_final),
            ("CMA-ES Rounding", cma_final),
        ]

        bench_records = []
        raw_p_list = []

        for opp_label, opp_final in opponents:
            diff = prop_final - opp_final
            wins = np.sum(diff < 0)
            losses = np.sum(diff > 0)
            ties = np.sum(diff == 0)

            if np.all(diff == 0):
                w_stat, p_val = 0.0, 1.0
            else:
                try:
                    res = stats.wilcoxon(prop_final, opp_final, alternative="two-sided")
                    w_stat = res.statistic
                    p_val = res.pvalue
                except Exception as e:
                    w_stat, p_val = np.nan, 1.0

            raw_p_list.append(p_val)
            bench_records.append({
                "benchmark": bench_name,
                "opponent": opp_label,
                "prop_mean": np.mean(prop_final),
                "opp_mean": np.mean(opp_final),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "w_stat": w_stat,
                "raw_p": p_val,
            })

        # Apply Holm correction for family of 2 comparisons per benchmark
        adj_p, sig = holm_bonferroni(raw_p_list)

        for i, rec in enumerate(bench_records):
            rec["adj_p"] = adj_p[i]
            rec["significant"] = sig[i]
            if sig[i]:
                if rec["wins"] > rec["losses"]:
                    verdict = "Generalized ES Wins (dagger)"
                    symbol = "\\dagger"
                else:
                    verdict = "Baseline Wins (ddagger)"
                    symbol = "\\ddagger"
            else:
                verdict = "Statistical Tie (approx)"
                symbol = "\\approx"
            rec["verdict"] = verdict
            rec["symbol"] = symbol

            print(f"[{bench_name}] vs {rec['opponent']:<16} | Wins/Ties/Losses: {rec['wins']}/{rec['ties']}/{rec['losses']} "
                  f"| W={rec['w_stat']:<5.1f} | p_raw={rec['raw_p']:<9.4e} | p_adj={rec['adj_p']:<9.4e} | {verdict}")

        results[bench_name] = bench_records
        print("-" * 80)

    return results


def main():
    run_experiment_1()
    run_experiment_3()


if __name__ == "__main__":
    main()
