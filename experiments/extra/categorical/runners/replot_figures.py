"""Re-generate Appendix E figures with clean English typography."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def main():
    root_dir = Path(__file__).resolve().parents[4]
    cat_dir = root_dir / "experiments" / "extra" / "categorical"
    doc_fig_dir = Path(r"C:\Users\bejmo\Documents\documents\Universidad\4.2\TFG\documentacion\figures")

    # 1. Study 1: Convergence
    d1_path = cat_dir / "data" / "categorical_comparison_data.npz"
    d1 = np.load(d1_path)

    benchmarks = [
        ('B1_Stationary', 'Stationary Matching'),
        ('B2_DynamicShift', 'Dynamic Shift'),
        ('B3_MixedCardinality', 'Mixed Cardinalities'),
        ('B4_Deceptive', 'Deceptive Trap'),
        ('B5_RoyalRoad', 'Epistatic Royal Road'),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=False)
    for idx, (b_id, b_title) in enumerate(benchmarks):
        ax = axes[idx]
        mean_tr_leg = d1[f'{b_id}_legacy_traces_mean']
        mean_tr_new = d1[f'{b_id}_new_traces_mean']
        ax.plot(mean_tr_leg, label='Legacy Softmax', color='#d95f02', lw=2, linestyle='--')
        ax.plot(mean_tr_new, label='New Jump (Unified)', color='#1b9e77', lw=2)
        ax.set_title(b_title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Generation', fontsize=10)
        if idx == 0:
            ax.set_ylabel(r'Mean Loss $f(x)$', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig1_local = cat_dir / "figures" / "categorical_convergence.png"
    fig1_doc = doc_fig_dir / "appendix_e_categorical_convergence.png"
    fig.savefig(fig1_local, dpi=300)
    fig.savefig(fig1_doc, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig1_local} and {fig1_doc}")

    # 2. Study 2: Equivalence
    d2_path = cat_dir / "data" / "binary_vs_categorical_data.npz"
    d2 = np.load(d2_path)
    arr_bin = d2['binary_losses']
    arr_cat = d2['categorical_losses']
    diff = arr_bin - arr_cat
    if np.any(diff != 0):
        w_stat, w_p = stats.wilcoxon(arr_bin, arr_cat)
    else:
        w_stat, w_p = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    box = ax.boxplot(
        [arr_bin, arr_cat],
        tick_labels=['Binary (XOR)', 'Categorical ($K=2$)'],
        patch_artist=True,
        widths=0.4,
    )
    colors = ['#386cb0', '#7fc97f']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(r'Final Loss $f(x)$ (32 variables)', fontsize=11)
    ax.set_title(f"Equivalence Comparison (50 Seeds)\nWilcoxon $p = {w_p:.4f}$ ($p > 0.05$)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    fig2_local = cat_dir / "figures" / "binary_vs_cat_k2.png"
    fig2_doc = doc_fig_dir / "appendix_e_binary_vs_cat_k2.png"
    fig.savefig(fig2_local, dpi=300)
    fig.savefig(fig2_doc, dpi=300)
    plt.close(fig)
    print(f"Saved: {fig2_local} and {fig2_doc}")

if __name__ == "__main__":
    main()
