"""
plotting.py: Decoupled and reusable visualization utilities for experiments.

Generates consistent, publication-grade figures with Matplotlib for:
1. Dynamic Target Shock (DTS) profiles per variable type and unified summary grid.
2. Domain-specific operator validation and ablation studies.
3. Experiment 3 heterogeneous convergence curves and strategy parameter dynamics.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from src.es_model.config import ESConfig



def plot_dts_fitness(
    results: Dict[str, Any],
    title: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Render standalone fitness convergence profile under DTS:
    - Median + IQR (25-75%) Fitness trajectory (log/linear scale as appropriate).
    - Shock marker vertical dashed line at g_shift.
    """
    fit_arr = results["fitness"]
    n_gens = results["n_gens"]
    g_shift = results["g_shift"]
    var_type = results["var_type"]
    gens = np.arange(n_gens)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    med_f = np.median(fit_arr, axis=0)
    q25_f = np.percentile(fit_arr, 25, axis=0)
    q75_f = np.percentile(fit_arr, 75, axis=0)

    ax.plot(gens, med_f, color="#1f77b4", lw=2.0, label="Median Best Fitness")
    if var_type in ("real", "count", "fuzzy", "circular"):
        ax.fill_between(gens, np.maximum(1e-15, q25_f), q75_f, color="#1f77b4", alpha=0.25, label="IQR (25-75%)")
        ax.set_yscale("log")
        ylabel_f = "Fitness / Distance (Log Scale)"
    else:
        ax.fill_between(gens, q25_f, q75_f, color="#1f77b4", alpha=0.25, label="IQR (25-75%)")
        ylabel_f = "Target Error"

    ax.axvline(g_shift, color="red", ls="--", lw=1.5, label=f"Shift (g={g_shift})")
    ax.set_title(f"{title}: Fitness Convergence", fontsize=12, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.set_ylabel(ylabel_f)
    ax.legend(loc="upper right")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_dts_strategy(
    results: Dict[str, Any],
    param_name: str,
    title: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Render standalone adaptive strategy parameter profile under DTS:
    - Median + IQR trajectory (sigma, lambda, logit, etc.).
    - For binary: Twin-axis showing logit z (left) and mutation flip probability (right).
    - For categorical: Normalized Softmax Entropy [0, 1].
    - Shock marker vertical dashed line at g_shift.
    """
    strat_arr = results["strategy"]
    n_gens = results["n_gens"]
    g_shift = results["g_shift"]
    var_type = results["var_type"]
    gens = np.arange(n_gens)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    med_s = np.median(strat_arr, axis=0)
    q25_s = np.percentile(strat_arr, 25, axis=0)
    q75_s = np.percentile(strat_arr, 75, axis=0)

    ax_twin = None
    if var_type in ("binary", "categorical"):
        color = "#9467bd" if var_type == "binary" else "#e377c2"
        ax.plot(gens, med_s, color=color, lw=2.2, label=r"Median Logit $z$ (Left Axis)")
        ax.set_ylabel(r"Logit Value $z$ (Log-odds scale)", color=color, fontweight="bold", fontsize=11)
        ax.tick_params(axis="y", labelcolor=color, labelsize=10)
        ax.spines["left"].set_color(color)
        ax.spines["left"].set_linewidth(1.8)

        p_mut = 1.0 / (1.0 + np.exp(-med_s))
        ax_twin = ax.twinx()
        ax_twin.plot(gens, p_mut, color="#8c564b", ls="--", lw=2.0, label=r"Mutation Prob. $p_{\mathrm{mut}} = \sigma(z)$ (Right Axis)")
        ax_twin.set_ylabel(r"Mutation Probability $p_{\mathrm{mut}}$ (Probability scale)", color="#8c564b", fontweight="bold", fontsize=11)
        ax_twin.tick_params(axis="y", labelcolor="#8c564b", labelsize=10)
        ax_twin.spines["right"].set_color("#8c564b")
        ax_twin.spines["right"].set_linewidth(1.8)
        ax_twin.grid(False)
    else:
        q25_s = np.percentile(strat_arr, 25, axis=0)
        ax.plot(gens, med_s, color="#2ca02c", lw=2.0, label=f"Median {param_name}")
        ax.fill_between(gens, np.maximum(1e-10, q25_s), q75_s, color="#2ca02c", alpha=0.25, label="IQR (25-75%)")
        ax.set_yscale("log")
        ax.set_ylabel(f"{param_name} (Log Scale)")

    ax.axvline(g_shift, color="red", ls="--", lw=1.5, label=f"Shift (g={g_shift})")
    ax.set_title(f"{title}: Self-Adaptation ({param_name})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Generation")

    if ax_twin is not None:
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower left", framealpha=0.9)
    else:
        ax.legend(loc="upper right" if var_type != "binary" else "lower left")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_dts_3panel(
    results: Dict[str, Any],
    param_name: str,
    title: str,
    save_path: Optional[str] = None,
) -> None:
    """
    Render standard 3-panel dynamic profile for a specific variable type under DTS:
    - Panel 1: Median + IQR (25-75%) Fitness trajectory (log/linear scale as appropriate).
    - Panel 2: Median + IQR adaptive strategy parameter trajectory (sigma, lambda, logit, etc.).
    - Panel 3: Invariant violations count or gauge residual (flat at 0 throughout).
    """
    fit_arr = results["fitness"]
    strat_arr = results["strategy"]
    inv_arr = results["invariants"]
    n_gens = results["n_gens"]
    g_shift = results["g_shift"]
    var_type = results["var_type"]
    gens = np.arange(n_gens)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    # --- PANEL 1: Fitness Convergence ---
    med_f = np.median(fit_arr, axis=0)
    q25_f = np.percentile(fit_arr, 25, axis=0)
    q75_f = np.percentile(fit_arr, 75, axis=0)

    axes[0].plot(gens, med_f, color="#1f77b4", lw=2.0, label="Median Best Fitness")
    if var_type in ("real", "count", "fuzzy", "circular"):
        axes[0].fill_between(gens, np.maximum(1e-15, q25_f), q75_f, color="#1f77b4", alpha=0.25, label="IQR (25-75%)")
        axes[0].set_yscale("log")
        ylabel_f = "Fitness / Distance (Log Scale)"
    else:
        axes[0].fill_between(gens, q25_f, q75_f, color="#1f77b4", alpha=0.25, label="IQR (25-75%)")
        ylabel_f = "Target Error"

    axes[0].axvline(g_shift, color="red", ls="--", lw=1.5, label=f"Shift (g={g_shift})")
    axes[0].set_title(f"{title}: Fitness Convergence", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel(ylabel_f)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)

    # --- PANEL 2: Adaptive Strategy Parameter ---
    med_s = np.median(strat_arr, axis=0)
    q25_s = np.percentile(strat_arr, 25, axis=0)
    q75_s = np.percentile(strat_arr, 75, axis=0)

    ax_twin = None
    if var_type in ("binary", "categorical"):
        color = "#9467bd" if var_type == "binary" else "#e377c2"
        axes[1].plot(gens, med_s, color=color, lw=2.2, label=r"Median Logit $z$ (Left Axis)")
        axes[1].set_ylabel(r"Logit Value $z$ (Log-odds scale)", color=color, fontweight="bold", fontsize=11)
        axes[1].tick_params(axis="y", labelcolor=color, labelsize=10)
        axes[1].spines["left"].set_color(color)
        axes[1].spines["left"].set_linewidth(1.8)

        p_mut = 1.0 / (1.0 + np.exp(-med_s))
        ax_twin = axes[1].twinx()
        ax_twin.plot(gens, p_mut, color="#8c564b", ls="--", lw=2.0, label=r"Mutation Prob. $p_{\mathrm{mut}} = \sigma(z)$ (Right Axis)")
        ax_twin.set_ylabel(r"Mutation Probability $p_{\mathrm{mut}}$ (Probability scale)", color="#8c564b", fontweight="bold", fontsize=11)
        ax_twin.tick_params(axis="y", labelcolor="#8c564b", labelsize=10)
        ax_twin.spines["right"].set_color("#8c564b")
        ax_twin.spines["right"].set_linewidth(1.8)
        ax_twin.grid(False)
    else:
        axes[1].plot(gens, med_s, color="#2ca02c", lw=2.0, label=f"Median {param_name}")
        axes[1].fill_between(gens, np.maximum(1e-10, q25_s), q75_s, color="#2ca02c", alpha=0.25, label="IQR (25-75%)")
        axes[1].set_yscale("log")
        axes[1].set_ylabel(f"{param_name} (Log Scale)")

    axes[1].axvline(g_shift, color="red", ls="--", lw=1.5, label=f"Shift (g={g_shift})")
    axes[1].set_title(f"{title}: Self-Adaptation ({param_name})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Generation")

    if ax_twin is not None:
        lines_1, labels_1 = axes[1].get_legend_handles_labels()
        lines_2, labels_2 = ax_twin.get_legend_handles_labels()
        axes[1].legend(lines_1 + lines_2, labels_1 + labels_2, loc="lower left", framealpha=0.9)
    else:
        axes[1].legend(loc="upper right" if var_type not in ("binary", "categorical") else "lower left")
    axes[1].grid(True, which="both", ls="--", alpha=0.5)

    # --- PANEL 3: Invariant Violations ---
    tot_inv = np.sum(inv_arr, axis=0)
    axes[2].plot(gens, tot_inv, color="#d62728", lw=1.8, label="Domain Violations")
    axes[2].set_title(f"{title}: Domain Invariant Check", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Violation Count (Strictly 0)")
    axes[2].set_ylim(-0.5, 5)

    axes[2].set_xlabel("Generation")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, ls="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_dts_unified_grid(
    results_list: List[Dict[str, Any]],
    g_shift: int,
    save_path: Optional[str] = None,
) -> None:
    """
    Unified 6-panel summary grid (2x3) comparing strategy parameter dynamics
    across all 6 variable types under the exact same target shock protocol.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)

    configs = [
        (0, 0, "real", "#2ca02c", r"Real: $\sigma_{\mathrm{real}}$ (Log-normal)", r"Step Size $\sigma$", True),
        (0, 1, "count", "#ff7f0e", r"Count: $\lambda_{\mathrm{count}}$ (Skellam)", r"Rate Parameter $\lambda$", True),
        (0, 2, "binary", "#9467bd", r"Binary: Logit $z$ (Logit-normal)", "Logit $z$", False),
        (1, 0, "categorical", "#e377c2", r"Categorical: Logit $z$ (Logit-normal)", "Logit $z$", False),
        (1, 1, "fuzzy", "#17becf", r"Fuzzy: $\sigma_{\mathrm{fuzzy}}$ (Center/Disp)", r"Step Size $\sigma$", True),
        (1, 2, "circular", "#bcbd22", r"Circular: $\lambda_{\mathrm{circular}}$ (Modular)", r"Rate Parameter $\lambda$", True),
    ]

    res_by_type = {r["var_type"]: r for r in results_list}

    for row, col, vtype, color, title, ylabel, is_log in configs:
        ax = axes[row, col]
        res = res_by_type[vtype]
        gens = np.arange(res["n_gens"])
        med_strat = np.median(res["strategy"], axis=0)

        ax.plot(gens, med_strat, color=color, lw=2.0)
        ax.axvline(g_shift, color="red", ls="--", lw=1.2)
        if is_log:
            ax.set_yscale("log")

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_ylabel(ylabel)
        if row == 1:
            ax.set_xlabel("Generation")
        ax.grid(True, which="both" if is_log else "major", ls="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_dts_individual_summaries(
    results_list: List[Dict[str, Any]],
    g_shift: int,
    save_dir: Optional[str] = None,
) -> None:
    """
    Generate and save separate standalone figures for each of the 6 variable representations
    under the Dynamic Target Shift (DTS) protocol.
    """
    configs = [
        ("real", "#2ca02c", r"Real ($\mathbb{R}$): $\sigma_{\mathrm{real}}$ (Log-normal)", r"Step Size $\sigma$", True),
        ("count", "#ff7f0e", r"Count ($\mathbb{N}_0$): $\lambda_{\mathrm{count}}$ (Skellam)", r"Rate Parameter $\lambda$", True),
        ("binary", "#9467bd", r"Binary ({0, 1}): Logit $z$ (Logit-normal)", "Logit $z$", False),
        ("categorical", "#e377c2", r"Categorical: Logit $z$ (Logit-normal)", "Logit $z$", False),
        ("fuzzy", "#17becf", r"Fuzzy: $\sigma_{\mathrm{fuzzy}}$ (Center/Dispersion)", r"Step Size $\sigma$", True),
        ("circular", "#bcbd22", r"Circular: $\lambda_{\mathrm{circular}}$ (Modular Step)", r"Rate Parameter $\lambda$", True),
    ]

    res_by_type = {r["var_type"]: r for r in results_list}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for vtype, color, title, ylabel, is_log in configs:
        res = res_by_type[vtype]
        gens = np.arange(res["n_gens"])
        med_strat = np.median(res["strategy"], axis=0)
        q25_strat = np.percentile(res["strategy"], 25, axis=0)
        q75_strat = np.percentile(res["strategy"], 75, axis=0)

        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot(gens, med_strat, color=color, lw=2.2, label=f"Median Strategy ({vtype.capitalize()})")
        if is_log:
            ax.fill_between(gens, np.maximum(1e-12, q25_strat), q75_strat, color=color, alpha=0.25, label="IQR (25-75%)")
            ax.set_yscale("log")
        else:
            ax.fill_between(gens, q25_strat, q75_strat, color=color, alpha=0.25, label="IQR (25-75%)")

        ax.axvline(g_shift, color="red", ls="--", lw=1.5, label=f"Target Shift (g={g_shift})")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("Generation", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc="upper right" if vtype != "binary" else "lower left", frameon=True)
        ax.grid(True, which="both" if is_log else "major", ls="--", alpha=0.5)

        plt.tight_layout()
        if save_dir:
            fig_path = os.path.join(save_dir, f"dts_{vtype}_summary.png")
            fig.savefig(fig_path, dpi=200, bbox_inches="tight")
            print(f"Saved separate figure: {fig_path}")
        plt.show()


# =============================================================================
# Experiment 3: Heterogeneous Convergence and Strategy Dynamics Plotting
# =============================================================================

def plot_heterogeneous_convergence(
    results: Dict[str, Dict[str, Any]],
    max_evals: int,
    config: ESConfig,
    n_seeds: int,
    data_dir: Optional[str] = None,
) -> None:
    """
    Plot and save convergence curves across synthetic benchmarks:
    1. Standalone publication-grade figures per benchmark saved to data_dir.
    2. Side-by-side comparative overview across all benchmarks.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    bench_names = list(results.keys())
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    evals = np.arange(1, max_evals + 1)

    # 1. Save standalone convergence curves separately per benchmark
    for i, bench_name in enumerate(bench_names):
        fig, ax = plt.subplots(figsize=(7, 4.8))
        res = results[bench_name]
        mean_trace = res["mean"]
        std_trace = res["std"]
        col = colors[i % len(colors)]

        ax.plot(evals, mean_trace, color=col, lw=2.2, label=f"Generalized ES (μ={config.mu}, λ={config.lam})")
        ax.fill_between(evals, np.maximum(1e-12, mean_trace - std_trace), mean_trace + std_trace, color=col, alpha=0.25, label="±1 Std Dev")

        ax.set_title(f"{bench_name}: Fitness Convergence ({n_seeds} Seeds)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Function Evaluations", fontsize=11)
        ax.set_ylabel("Best Fitness (Log Scale)", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)

        plt.tight_layout()
        if data_dir is not None:
            os.makedirs(data_dir, exist_ok=True)
            fig_path = os.path.join(data_dir, f"exp3_{bench_name.lower()}_convergence.png")
            fig.savefig(fig_path, dpi=200, bbox_inches="tight")
            print(f"Saved separate convergence figure: {fig_path}")
        plt.show()

    # 2. Interactive comparative overview in notebook
    n_b = len(bench_names)
    fig, axes = plt.subplots(1, n_b, figsize=(6 * n_b, 5), sharey=False)
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]
    for i, bench_name in enumerate(bench_names):
        ax = axes[i]
        res = results[bench_name]
        mean_trace = res["mean"]
        std_trace = res["std"]
        col = colors[i % len(colors)]

        ax.plot(evals, mean_trace, color=col, lw=2.0, label=f"Generalized ES (μ={config.mu}, λ={config.lam})")
        ax.fill_between(evals, np.maximum(1e-12, mean_trace - std_trace), mean_trace + std_trace, color=col, alpha=0.25, label="±1 Std Dev")
        ax.set_title(f"{bench_name}\n({n_seeds} Seeds)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Function Evaluations", fontsize=11)
        ax.set_ylabel("Best Fitness (Log Scale)", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.show()


def plot_heterogeneous_dynamics(
    bench_name: str,
    bench_agg: Dict[str, np.ndarray],
    n_generations: int,
    n_seeds: int,
    data_dir: Optional[str] = None,
) -> None:
    """
    Plot and save heterogeneous strategy dynamics and cost decompositions for a benchmark:
    1. Standalone Panel A: Mutation step sizes / intensities.
    2. Standalone Panel B: Probabilities & categorical certainty metrics.
    3. Standalone Panel C: Partial fitness error decomposition across the 6 variable types.
    4. Composite 3-panel diagnostics figure.
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    gens = np.arange(n_generations)

    m_keys_1 = [
        ("sigma_real", r"Real $\sigma_{real}$", "#1f77b4", "-"),
        ("lambda_count", r"Count $\lambda_{count}$", "#ff7f0e", "-"),
        ("sigma_fuzzy", r"Fuzzy $\sigma_{fuzzy}$", "#8c564b", "-"),
        ("lambda_circular", r"Circular $\lambda_{circ}$", "#e377c2", "-"),
    ]

    m_keys_2 = [
        ("cat_confidence", r"Cat. Confidence $\max(p)$", "#2ca02c", "-", 2.2),
        ("cat_entropy_norm", r"Cat. Shannon Entropy $H/\ln(K)$", "#9467bd", "--", 1.8),
        ("p_flip_binary", r"Binary Flip Prob. $p_{flip}$", "#d62728", "-", 2.0),
    ]

    m_keys_3 = [
        ("cost_real", "Real Cost", "#1f77b4"),
        ("cost_count", "Count Cost", "#ff7f0e"),
        ("cost_binary", "Binary Cost", "#d62728"),
        ("cost_cat", "Categorical Cost", "#2ca02c"),
        ("cost_fuzzy", "Fuzzy Cost", "#8c564b"),
        ("cost_circular", "Circular Cost", "#e377c2"),
    ]

    if data_dir is not None:
        os.makedirs(data_dir, exist_ok=True)

    # 1. Standalone: Mutation Step Sizes & Rates (Log scale)
    fig_a, ax_a = plt.subplots(figsize=(7.5, 4.8))
    for key, lbl, col, ls in m_keys_1:
        mean_v = bench_agg[f"{key}_mean"]
        std_v = bench_agg[f"{key}_std"]
        ax_a.plot(gens, mean_v, label=lbl, color=col, ls=ls, lw=2.0)
        ax_a.fill_between(gens, np.maximum(mean_v - std_v, 1e-7), mean_v + std_v, color=col, alpha=0.15)
    ax_a.set_title(f"{bench_name}: Mutation Intensities & Step Sizes", fontsize=12, fontweight="bold")
    ax_a.set_xlabel("Generation", fontsize=11)
    ax_a.set_ylabel("Parameter Value (Log Scale)", fontsize=11)
    ax_a.set_yscale("log")
    ax_a.grid(True, which="both", ls="--", alpha=0.5)
    ax_a.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    if data_dir is not None:
        path_a = os.path.join(data_dir, f"exp3_{bench_name.lower()}_step_sizes.png")
        fig_a.savefig(path_a, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path_a}")
    plt.show()

    # 2. Standalone: Probabilities & Certainty [0, 1]
    fig_b, ax_b = plt.subplots(figsize=(7.5, 4.8))
    for key, lbl, col, ls, lw in m_keys_2:
        mean_v = bench_agg[f"{key}_mean"]
        std_v = bench_agg[f"{key}_std"]
        ax_b.plot(gens, mean_v, label=lbl, color=col, ls=ls, lw=lw)
        ax_b.fill_between(gens, np.clip(mean_v - std_v, 0.0, 1.0), np.clip(mean_v + std_v, 0.0, 1.0), color=col, alpha=0.15)
    ax_b.set_title(f"{bench_name}: Probabilities & Information Certainty", fontsize=12, fontweight="bold")
    ax_b.set_xlabel("Generation", fontsize=11)
    ax_b.set_ylabel("Probability / Normalized Metric [0, 1]", fontsize=11)
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.grid(True, which="both", ls="--", alpha=0.5)
    ax_b.legend(loc="center right", frameon=True, fontsize=10)
    plt.tight_layout()
    if data_dir is not None:
        path_b = os.path.join(data_dir, f"exp3_{bench_name.lower()}_probabilities.png")
        fig_b.savefig(path_b, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path_b}")
    plt.show()

    # 3. Standalone: Fitness Error Decomposition per Variable Type
    fig_c, ax_c = plt.subplots(figsize=(7.5, 4.8))
    for key, lbl, col in m_keys_3:
        mean_v = np.maximum(bench_agg[f"{key}_mean"], 1e-12)
        ax_c.plot(gens, mean_v, label=lbl, color=col, lw=1.8)
    ax_c.set_title(f"{bench_name}: Fitness Error Decomposition", fontsize=12, fontweight="bold")
    ax_c.set_xlabel("Generation", fontsize=11)
    ax_c.set_ylabel("Partial Fitness Contribution (Log Scale)", fontsize=11)
    ax_c.set_yscale("log")
    ax_c.set_ylim(bottom=1e-8)
    ax_c.grid(True, which="both", ls="--", alpha=0.5)
    ax_c.legend(loc="upper right", frameon=True, fontsize=9.5)
    plt.tight_layout()
    if data_dir is not None:
        path_c = os.path.join(data_dir, f"exp3_{bench_name.lower()}_decomposition.png")
        fig_c.savefig(path_c, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path_c}")
    plt.show()

    # 4. Interactive 3-Panel Diagnostics Overview in notebook
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    for key, lbl, col, ls in m_keys_1:
        axes[0].plot(gens, bench_agg[f"{key}_mean"], label=lbl, color=col, ls=ls, lw=2.0)
        axes[0].fill_between(gens, np.maximum(bench_agg[f"{key}_mean"] - bench_agg[f"{key}_std"], 1e-7), bench_agg[f"{key}_mean"] + bench_agg[f"{key}_std"], color=col, alpha=0.15)
    axes[0].set_title("A) Mutation Intensities & Step Sizes", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Generation", fontsize=11)
    axes[0].set_ylabel("Parameter Value (Log Scale)", fontsize=11)
    axes[0].set_yscale("log")
    axes[0].grid(True, which="both", ls="--", alpha=0.5)
    axes[0].legend(loc="upper right", frameon=True, fontsize=10)

    for key, lbl, col, ls, lw in m_keys_2:
        axes[1].plot(gens, bench_agg[f"{key}_mean"], label=lbl, color=col, ls=ls, lw=lw)
        axes[1].fill_between(gens, np.clip(bench_agg[f"{key}_mean"] - bench_agg[f"{key}_std"], 0.0, 1.0), np.clip(bench_agg[f"{key}_mean"] + bench_agg[f"{key}_std"], 0.0, 1.0), color=col, alpha=0.15)
    axes[1].set_title("B) Probabilities & Information Certainty", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Generation", fontsize=11)
    axes[1].set_ylabel("Probability / Normalized Metric [0, 1]", fontsize=11)
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].grid(True, which="both", ls="--", alpha=0.5)
    axes[1].legend(loc="center right", frameon=True, fontsize=10)

    for key, lbl, col in m_keys_3:
        axes[2].plot(gens, np.maximum(bench_agg[f"{key}_mean"], 1e-12), label=lbl, color=col, lw=1.8)
    axes[2].set_title("C) Fitness Error Decomposition", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Generation", fontsize=11)
    axes[2].set_ylabel("Partial Fitness Contribution (Log Scale)", fontsize=11)
    axes[2].set_yscale("log")
    axes[2].set_ylim(bottom=1e-8)
    axes[2].grid(True, which="both", ls="--", alpha=0.5)
    axes[2].legend(loc="upper right", frameon=True, fontsize=9.5)

    fig.suptitle(f"Heterogeneous Variable & Strategy Dynamics: {bench_name} (Mean $\\pm 1$ Std, {n_seeds} Seeds)", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.show()



