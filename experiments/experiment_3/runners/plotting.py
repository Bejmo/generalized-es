"""
plotting.py: Visualization routines for Experiment 3 (Heterogeneous Benchmarks).
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt

from src.es_model.config import ESConfig


def _safe_savefig(fig: plt.Figure, path: str, **kwargs) -> None:
    """Save figure with retries to handle transient Windows file locks."""
    import time
    for attempt in range(5):
        try:
            fig.savefig(path, **kwargs)
            return
        except OSError as e:
            if attempt == 4:
                print(f"Warning: Could not overwrite '{path}' due to Windows file lock ({e}). Figure will still be displayed.")
                return
            time.sleep(0.3)


def plot_heterogeneous_convergence(
    results: Dict[str, Dict[str, Any]],
    max_evals: Optional[int] = None,
    config: Optional[ESConfig] = None,
    n_seeds: Optional[int] = None,
    save_dir: Optional[str] = None,
    baselines_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot and save convergence curves across synthetic benchmarks:
    Compares 4 methods side-by-side using Median and Interquartile Range (IQR [Q1, Q3]):
    1. Generalized ES (ours)
    2. CatCMA with Margin (CatCMAwM 2025)
    3. CMA-ES with Continuous Relaxation & Rounding
    4. Uniform Random Search
    """
    bench_names = list(results.keys())
    if not bench_names:
        return
    if max_evals is None:
        max_evals = len(results[bench_names[0]]["mean"])
    if n_seeds is None:
        traces = results[bench_names[0]].get("traces")
        n_seeds = traces.shape[0] if traces is not None else 10
    cfg_label = f" (μ={config.mu}, λ={config.lam})" if config else ""

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    evals = np.arange(1, max_evals + 1)

    # Attempt to load external baseline traces if available
    base_data = None
    if baselines_path is not None and os.path.exists(baselines_path):
        base_data = np.load(baselines_path, allow_pickle=True)
    elif save_dir is not None:
        cand_path = os.path.abspath(os.path.join(save_dir, "..", "data", "random_search_baselines.npz"))
        if os.path.exists(cand_path):
            base_data = np.load(cand_path, allow_pickle=True)

    bench_key_map = {
        "generalized_sphere": "sphere",
        "sphere": "sphere",
        "generalized_coupled": "coupled",
        "coupled": "coupled",
        "generalized_rastrigin": "rastrigin",
        "rastrigin": "rastrigin",
        "generalized_multiplicative": "multiplicative",
        "multiplicative": "multiplicative",
    }

    # 1. Standalone publication-grade figures per benchmark
    for bench_name in bench_names:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        b_key = bench_key_map.get(bench_name.lower(), bench_name.lower().replace("generalized_", ""))

        # A. Generalized ES (ours)
        res = results[bench_name]
        tr_prop = res.get("traces")
        if tr_prop is not None:
            med_prop = np.median(tr_prop, axis=0)
            q25_prop = np.percentile(tr_prop, 25, axis=0)
            q75_prop = np.percentile(tr_prop, 75, axis=0)
        else:
            med_prop = res["mean"]
            q25_prop = np.maximum(1e-12, med_prop - res["std"])
            q75_prop = med_prop + res["std"]

        ax.plot(evals, med_prop, color="#1f77b4", lw=2.4, label=f"Generalized ES{cfg_label}")
        ax.fill_between(evals, np.maximum(1e-12, q25_prop), q75_prop, color="#1f77b4", alpha=0.22)

        # B. External baselines from base_data
        if base_data is not None:
            # 1. CatCMAwM (2025)
            cat_key = f"{b_key}__catcmawm_traces"
            if cat_key in base_data:
                tr_cat = base_data[cat_key]
                med_cat = np.median(tr_cat, axis=0)
                q25_cat = np.percentile(tr_cat, 25, axis=0)
                q75_cat = np.percentile(tr_cat, 75, axis=0)
                ax.plot(evals, med_cat, color="#2ca02c", lw=2.0, label="CatCMAwM (2025)")
                ax.fill_between(evals, np.maximum(1e-12, q25_cat), q75_cat, color="#2ca02c", alpha=0.18)

            # 2. CMA-ES Rounding
            cma_key = f"{b_key}__cma_traces"
            if cma_key in base_data:
                tr_cma = base_data[cma_key]
                med_cma = np.median(tr_cma, axis=0)
                q25_cma = np.percentile(tr_cma, 25, axis=0)
                q75_cma = np.percentile(tr_cma, 75, axis=0)
                ax.plot(evals, med_cma, color="#d62728", lw=1.8, ls="-.", label="CMA-ES Rounding")
                ax.fill_between(evals, np.maximum(1e-12, q25_cma), q75_cma, color="#d62728", alpha=0.14)

            # 3. Random Search
            rs_key = f"{b_key}__rs_traces"
            if rs_key in base_data:
                tr_rs = base_data[rs_key]
                med_rs = np.median(tr_rs, axis=0)
                ax.plot(evals, med_rs, color="#7f7f7f", lw=1.5, ls="--", label="Random Search")

        ax.set_title(f"{bench_name}: Convergence Comparison ({n_seeds} Seeds)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Function Evaluations", fontsize=11)
        ax.set_ylabel("Best Fitness (Log Scale)", fontsize=11)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(loc="upper right", frameon=True, fontsize=9)

        plt.tight_layout()
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            fig_path = os.path.join(save_dir, f"exp3_{bench_name.lower()}_convergence.png")
            _safe_savefig(fig, fig_path, dpi=200, bbox_inches="tight")
            print(f"Saved separate convergence figure: {fig_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    return


def plot_heterogeneous_dynamics(
    bench_name: str,
    bench_agg: Dict[str, np.ndarray],
    n_generations: Optional[int] = None,
    n_seeds: Optional[int] = None,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot and save heterogeneous strategy dynamics and cost decompositions for a benchmark:
    1. Standalone Panel A: Mutation step sizes / intensities.
    2. Standalone Panel B: Probabilities & categorical certainty metrics.
    3. Standalone Panel C: Partial fitness error decomposition across the 6 variable types.
    4. Composite 3-panel diagnostics figure.
    """
    if n_generations is None:
        first_arr = next(iter(bench_agg.values()))
        n_generations = len(first_arr)
    if n_seeds is None:
        n_seeds = 40

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    gens = np.arange(n_generations)

    m_keys_1 = [
        ("sigma_real", r"Real $\sigma_{real}$", "#1f77b4", "-"),
        ("lambda_count", r"Count $\lambda_{count}$", "#ff7f0e", "-"),
        ("sigma_fuzzy", r"Fuzzy $\sigma_{fuzzy}$", "#8c564b", "-"),
        ("lambda_circular", r"Circular $\lambda_{circ}$", "#e377c2", "-"),
    ]

    m_keys_2 = [
        ("p_flip_binary", r"Binary Flip Prob. $p_{flip}$", "#d62728", "-", 2.0),
    ]
    if "p_mut_categorical_mean" in bench_agg:
        m_keys_2.append(("p_mut_categorical", r"Cat. Mutation Prob. $p_{mut}$", "#2ca02c", "-", 2.0))
    elif "cat_confidence_mean" in bench_agg:
        m_keys_2.append(("cat_confidence", r"Cat. Confidence $\max(p)$", "#2ca02c", "-", 2.2))
        m_keys_2.append(("cat_entropy_norm", r"Cat. Shannon Entropy $H/\ln(K)$", "#9467bd", "--", 1.8))

    m_keys_3 = [
        ("cost_real", "Real Cost", "#1f77b4"),
        ("cost_count", "Count Cost", "#ff7f0e"),
        ("cost_binary", "Binary Cost", "#d62728"),
        ("cost_cat", "Categorical Cost", "#2ca02c"),
        ("cost_fuzzy", "Fuzzy Cost", "#8c564b"),
        ("cost_circular", "Circular Cost", "#e377c2"),
    ]

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

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
    if save_dir is not None:
        path_a = os.path.join(save_dir, f"exp3_{bench_name.lower()}_step_sizes.png")
        _safe_savefig(fig_a, path_a, dpi=200, bbox_inches="tight")
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
    if save_dir is not None:
        path_b = os.path.join(save_dir, f"exp3_{bench_name.lower()}_probabilities.png")
        _safe_savefig(fig_b, path_b, dpi=200, bbox_inches="tight")
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
    if save_dir is not None:
        path_c = os.path.join(save_dir, f"exp3_{bench_name.lower()}_decomposition.png")
        _safe_savefig(fig_c, path_c, dpi=200, bbox_inches="tight")
        print(f"  Saved: {path_c}")
    if show:
        plt.show()
    else:
        plt.close(fig_c)

    return
