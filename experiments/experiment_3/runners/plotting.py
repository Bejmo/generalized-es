"""
plotting.py: Visualization routines for Experiment 3 (Heterogeneous Benchmarks).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from src.es_model.config import ESConfig


def plot_heterogeneous_convergence(
    results: Dict[str, Dict[str, Any]],
    max_evals: Optional[int] = None,
    config: Optional[ESConfig] = None,
    n_seeds: Optional[int] = None,
    save_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    target_dir = save_dir or data_dir
    bench_names = list(results.keys())
    if not bench_names:
        return
    if max_evals is None:
        max_evals = len(results[bench_names[0]]["mean"])
    if n_seeds is None:
        traces = results[bench_names[0]].get("traces")
        n_seeds = traces.shape[0] if traces is not None else 10
    cfg_label = f" (μ={config.mu}, λ={config.lam})" if config else ""
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

        ax.plot(evals, mean_trace, color=col, lw=2.2, label=f"Generalized ES{cfg_label}")
        ax.fill_between(evals, np.maximum(1e-12, mean_trace - std_trace), mean_trace + std_trace, color=col, alpha=0.25, label="±1 Std Dev")

        ax.set_title(f"{bench_name}: Fitness Convergence ({n_seeds} Seeds)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Function Evaluations", fontsize=11)
        ax.set_ylabel("Best Fitness (Log Scale)", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)

        plt.tight_layout()
        if target_dir is not None:
            os.makedirs(target_dir, exist_ok=True)
            fig_path = os.path.join(target_dir, f"exp3_{bench_name.lower()}_convergence.png")
            fig.savefig(fig_path, dpi=200, bbox_inches="tight")
            print(f"Saved separate convergence figure: {fig_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    # 2. Interactive comparative overview in notebook
    if not show:
        return
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

        ax.plot(evals, mean_trace, color=col, lw=2.0, label=f"Generalized ES{cfg_label}")
        ax.fill_between(evals, np.maximum(1e-12, mean_trace - std_trace), mean_trace + std_trace, color=col, alpha=0.25, label="±1 Std Dev")
        ax.set_title(f"{bench_name}\n({n_seeds} Seeds)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Function Evaluations", fontsize=11)
        ax.set_ylabel("Best Fitness (Log Scale)", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_heterogeneous_dynamics(
    bench_name: str,
    bench_agg: Dict[str, np.ndarray],
    n_generations: Optional[int] = None,
    n_seeds: Optional[int] = None,
    save_dir: Optional[str] = None,
    data_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    target_dir = save_dir or data_dir
    if n_generations is None:
        first_arr = next(iter(bench_agg.values()))
        n_generations = len(first_arr)
    if n_seeds is None:
        n_seeds = 10
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

    if target_dir is not None:
        os.makedirs(target_dir, exist_ok=True)

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
        path_a = os.path.join(target_dir, f"exp3_{bench_name.lower()}_step_sizes.png")
        fig_a.savefig(path_a, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path_a}")
    if show:
        plt.show()
    else:
        plt.close(fig_a)

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
        path_b = os.path.join(target_dir, f"exp3_{bench_name.lower()}_probabilities.png")
        fig_b.savefig(path_b, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path_b}")
    if show:
        plt.show()
    else:
        plt.close(fig_b)

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
        path_c = os.path.join(target_dir, f"exp3_{bench_name.lower()}_decomposition.png")
        fig_c.savefig(path_c, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path_c}")
    if show:
        plt.show()
    else:
        plt.close(fig_c)

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
