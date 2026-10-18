"""
plotting.py: Decoupled and reusable visualization utilities for experiments.

Generates consistent, publication-grade figures with Matplotlib for:
1. Dynamic Target Shock (DTS) profiles per variable type and unified summary grid.
2. Domain-specific operator validation and ablation studies.
3. Experiment 3 heterogeneous convergence curves and strategy parameter dynamics.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from src.es_model.config import ESConfig


def _safe_savefig(fig: plt.Figure, path: str, **kwargs) -> None:
    """Save figure with retries and graceful fallback to handle Windows file locks."""
    import time
    for attempt in range(5):
        try:
            fig.savefig(path, **kwargs)
            print(f"Saved figure: {path}")
            return
        except OSError as e:
            if attempt == 4:
                print(f"Warning: Could not overwrite '{path}' due to Windows file lock ({e}). Figure is displayed in notebook.")
                return
            time.sleep(0.3)


def plot_dts_fitness(
    results: Dict[str, Any],
    title: str,
    save_path: Optional[str] = None,
    show: bool = True,
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
        _safe_savefig(fig, save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_dts_strategy(
    results: Dict[str, Any],
    param_name: str,
    title: str,
    save_path: Optional[str] = None,
    show: bool = True,
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
        p_arr = 1.0 / (1.0 + np.exp(-strat_arr))
        med_p = np.median(p_arr, axis=0)
        q25_p = np.percentile(p_arr, 25, axis=0)
        q75_p = np.percentile(p_arr, 75, axis=0)

        color = "#9467bd" if var_type == "binary" else "#e377c2"
        ax.plot(gens, med_p, color=color, lw=2.2, label=r"Median Mutation Rate $p$")
        ax.fill_between(gens, q25_p, q75_p, color=color, alpha=0.25, label="IQR (25-75%)")
        ax.set_ylabel(r"Mutation Rate $p = \sigma(z)$", color=color, fontweight="bold", fontsize=11)
        ax.tick_params(axis="y", labelcolor=color, labelsize=10)
        ax.set_ylim(0.0, max(0.15, float(np.max(q75_p)) * 1.15))
    else:
        q25_s = np.percentile(strat_arr, 25, axis=0)
        ax.plot(gens, med_s, color="#2ca02c", lw=2.0, label=f"Median {param_name}")
        ax.fill_between(gens, np.maximum(1e-10, q25_s), q75_s, color="#2ca02c", alpha=0.25, label="IQR (25-75%)")
        ax.set_yscale("log")
        ax.set_ylabel(f"{param_name} (Log Scale)")

    ax.axvline(g_shift, color="red", ls="--", lw=1.5, label=f"Shift (g={g_shift})")
    ax.set_title(f"{title}: Self-Adaptation ({param_name})", fontsize=12, fontweight="bold")
    ax.set_xlabel("Generation")
    ax.legend(loc="upper right" if var_type != "binary" else "lower left")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _safe_savefig(fig, save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_dts_3panel(
    results: Dict[str, Any],
    param_name: str,
    title: str,
    save_path: Optional[str] = None,
    show: bool = True,
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

    if var_type in ("binary", "categorical"):
        p_arr = 1.0 / (1.0 + np.exp(-strat_arr))
        med_p = np.median(p_arr, axis=0)
        q25_p = np.percentile(p_arr, 25, axis=0)
        q75_p = np.percentile(p_arr, 75, axis=0)

        color = "#9467bd" if var_type == "binary" else "#e377c2"
        axes[1].plot(gens, med_p, color=color, lw=2.2, label=r"Median Mutation Rate $p$")
        axes[1].fill_between(gens, q25_p, q75_p, color=color, alpha=0.25, label="IQR (25-75%)")
        axes[1].set_ylabel(r"Mutation Rate $p = \sigma(z)$", color=color, fontweight="bold", fontsize=11)
        axes[1].tick_params(axis="y", labelcolor=color, labelsize=10)
        axes[1].set_ylim(0.0, max(0.15, float(np.max(q75_p)) * 1.15))
    else:
        axes[1].plot(gens, med_s, color="#2ca02c", lw=2.0, label=f"Median {param_name}")
        axes[1].fill_between(gens, np.maximum(1e-10, q25_s), q75_s, color="#2ca02c", alpha=0.25, label="IQR (25-75%)")
        axes[1].set_yscale("log")
        axes[1].set_ylabel(f"{param_name} (Log Scale)")

    axes[1].axvline(g_shift, color="red", ls="--", lw=1.5, label=f"Shift (g={g_shift})")
    axes[1].set_title(f"{title}: Self-Adaptation ({param_name})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Generation")
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
        _safe_savefig(fig, save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_dts_unified_grid(
    results_list: Union[List[Dict[str, Any]], Dict[str, Any]],
    g_shift: int,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Unified 6-panel summary grid (2x3) comparing strategy parameter dynamics
    across all 6 variable types under the exact same target shock protocol.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharex=True)

    configs = [
        (0, 0, "real", "#2ca02c", r"Real: $\sigma_{\mathrm{real}}$ (Log-normal)", r"Step Size $\sigma$", True),
        (0, 1, "count", "#ff7f0e", r"Count: $\lambda_{\mathrm{count}}$ (Skellam)", r"Rate Parameter $\lambda$", True),
        (0, 2, "binary", "#9467bd", r"Binary: Mutation Rate $p$", r"Mutation Rate $p$", False),
        (1, 0, "categorical", "#e377c2", r"Categorical: Mutation Rate $p$", r"Mutation Rate $p$", False),
        (1, 1, "fuzzy", "#17becf", r"Fuzzy: $\sigma_{\mathrm{fuzzy}}$ (Center/Disp)", r"Step Size $\sigma$", True),
        (1, 2, "circular", "#bcbd22", r"Circular: $\lambda_{\mathrm{circular}}$ (Modular)", r"Rate Parameter $\lambda$", True),
    ]

    if isinstance(results_list, dict):
        res_by_type = results_list
    else:
        res_by_type = {r["var_type"]: r for r in results_list}

    for row, col, vtype, color, title, ylabel, is_log in configs:
        ax = axes[row, col]
        res = res_by_type[vtype]
        gens = np.arange(res["n_gens"])
        if vtype in ("binary", "categorical"):
            p_arr = 1.0 / (1.0 + np.exp(-res["strategy"]))
            med_strat = np.median(p_arr, axis=0)
            ax.set_ylim(0.0, max(0.15, float(np.max(med_strat)) * 1.15))
        else:
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
        _safe_savefig(fig, save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_dts_individual_summaries(
    results_list: Union[List[Dict[str, Any]], Dict[str, Any]],
    g_shift: int,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Generate and save separate standalone figures for each of the 6 variable representations
    under the Target Shift (TS) protocol.
    """
    configs = [
        ("real", "#2ca02c", r"Real ($\mathbb{R}$): $\sigma_{\mathrm{real}}$ (Log-normal)", r"Step Size $\sigma$", True),
        ("count", "#ff7f0e", r"Count ($\mathbb{N}_0$): $\lambda_{\mathrm{count}}$ (Skellam)", r"Rate Parameter $\lambda$", True),
        ("binary", "#9467bd", r"Binary ({0, 1}): Mutation Rate $p$", r"Mutation Rate $p$", False),
        ("categorical", "#e377c2", r"Categorical: Mutation Rate $p$", r"Mutation Rate $p$", False),
        ("fuzzy", "#17becf", r"Fuzzy: $\sigma_{\mathrm{fuzzy}}$ (Center/Dispersion)", r"Step Size $\sigma$", True),
        ("circular", "#bcbd22", r"Circular: $\lambda_{\mathrm{circular}}$ (Modular Step)", r"Rate Parameter $\lambda$", True),
    ]

    if isinstance(results_list, dict):
        res_by_type = results_list
    else:
        res_by_type = {r["var_type"]: r for r in results_list}

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for vtype, color, title, ylabel, is_log in configs:
        res = res_by_type[vtype]
        gens = np.arange(res["n_gens"])
        if vtype in ("binary", "categorical"):
            p_arr = 1.0 / (1.0 + np.exp(-res["strategy"]))
            med_strat = np.median(p_arr, axis=0)
            q25_strat = np.percentile(p_arr, 25, axis=0)
            q75_strat = np.percentile(p_arr, 75, axis=0)
        else:
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
            if vtype in ("binary", "categorical"):
                ax.set_ylim(0.0, max(0.15, float(np.max(q75_strat)) * 1.15))

        ax.axvline(g_shift, color="red", ls="--", lw=1.5, label=f"Target Shift (g={g_shift})")
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel("Generation", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc="upper right" if vtype != "binary" else "lower left", frameon=True)
        ax.grid(True, which="both" if is_log else "major", ls="--", alpha=0.5)

        plt.tight_layout()
        if save_dir:
            fig_path = os.path.join(save_dir, f"dts_{vtype}_summary.png")
            _safe_savefig(fig, fig_path, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
