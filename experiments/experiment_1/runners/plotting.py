"""
plotting.py: Visualization routines for Experiment 1.
Renders 3D loss surface alongside convergence trajectories.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

warnings.filterwarnings("ignore", message=".*Axes that are not compatible with tight_layout.*")

COLORS = {
    'Generalized ES': '#E63946',
    'Generalized ES (GES)': '#E63946',
    'Isotropic ES (Simple Self-Adaptive)': '#E9C46A',
    'sep-CMA-ES (Separable CMA-ES)': '#2A9D8F',
    'Standard CMA-ES (Upper Bound)': '#457B9D',
}

LABEL_DISPLAY_MAP = {
    'Generalized_ES': 'Generalized ES',
}


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


def plot_benchmark_summary(
    bname: str,
    binfo: Dict[str, Any],
    data: Dict[str, np.ndarray],
    max_evals: int = 10_000,
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """
    Plot side-by-side with asymmetric width ratios:
      Left:  Compact 3D Function Landscape (Surface Plot).
      Right: Main Convergence Trajectories curve.
    """
    fig = plt.figure(figsize=(11.5, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.8], wspace=0.25)
    ax_surf = fig.add_subplot(gs[0, 0], projection='3d')
    ax_conv = fig.add_subplot(gs[0, 1])

    # 1. Left Panel: 3D Surface Plot
    bounds = binfo['bounds']
    if bname == "Ackley":
        x = np.linspace(-5, 5, 80)
        y = np.linspace(-5, 5, 80)
    elif bname == "Rosenbrock":
        x = np.linspace(-2.0, 2.0, 80)
        y = np.linspace(-1.0, 3.0, 80)
    else:
        x = np.linspace(bounds[0], bounds[1], 80)
        y = np.linspace(bounds[0], bounds[1], 80)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = binfo['fn'](np.array([X[i, j], Y[i, j]]))

    surf = ax_surf.plot_surface(
        X, Y, Z,
        cmap='viridis',
        edgecolor='none',
        alpha=0.9,
        antialiased=True,
    )
    cbar = fig.colorbar(surf, ax=ax_surf, pad=0.1, shrink=0.7, aspect=12)
    cbar.set_label(r'$f(\mathbf{x})$', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    opt = binfo['optimum'](2)
    opt_val = binfo['fn'](opt)
    ax_surf.scatter(
        [opt[0]], [opt[1]], [opt_val],
        color='red', marker='*', s=120, label='Optimum',
        depthshade=False, zorder=10
    )

    ax_surf.set_title(f'{bname} (3D Landscape)', fontsize=9.5, fontweight='bold', pad=8)
    ax_surf.set_xlabel(r'$x_1$', fontsize=8, labelpad=2)
    ax_surf.set_ylabel(r'$x_2$', fontsize=8, labelpad=2)
    ax_surf.set_zlabel(r'$f(\mathbf{x})$', fontsize=8, labelpad=2)
    ax_surf.tick_params(labelsize=7, pad=1)
    ax_surf.view_init(elev=28, azim=-120)
    ax_surf.legend(loc='upper right', fontsize=7.5)

    # 2. Right Panel: Convergence Trajectories
    evals = np.arange(1, max_evals + 1)
    for rname, traces in data.items():
        disp_name = LABEL_DISPLAY_MAP.get(rname, rname)
        median = np.median(traces, axis=0)
        q25 = np.percentile(traces, 25, axis=0)
        q75 = np.percentile(traces, 75, axis=0)

        color = COLORS.get(disp_name, COLORS.get(rname, '#888888'))
        ls = '--' if 'Upper Bound' in rname else '-'
        ax_conv.plot(evals, median, label=disp_name, color=color, linewidth=1.5, linestyle=ls)
        ax_conv.fill_between(evals, q25, q75, alpha=0.15, color=color)

    ax_conv.set_yscale('log')
    ax_conv.set_xlabel('Evaluations', fontsize=9)
    ax_conv.set_ylabel('Best Fitness (log)', fontsize=9)
    ax_conv.set_title(f'Convergence Comparison - {bname} (D=10)', fontsize=10.5, fontweight='bold')
    ax_conv.legend(loc='upper right', fontsize=8)
    ax_conv.grid(True, alpha=0.3)
    ax_conv.tick_params(labelsize=8)

    try:
        plt.tight_layout()
    except Exception:
        pass
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        _safe_savefig(fig, save_path, dpi=120, bbox_inches='tight')
    if show:
        plt.show()
    return fig
