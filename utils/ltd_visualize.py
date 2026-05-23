"""LifeTopoDict atom visualization tools (Agent I).

Provides plotting functions for atom state distribution, node reconstruction
residuals, and atom-usage heatmaps.  Falls back to plain-text output when
matplotlib is not available.
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy matplotlib import
_MPL_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL_AVAILABLE = True
except ImportError:
    pass


def plot_atom_state_distribution(clf, save_path: str) -> None:
    """Plot a pie chart of atom lifecycle states (protected / plastic / inactive).

    Falls back to text output when matplotlib is unavailable.

    Args:
        clf: HCSOINNClassifier instance with ``atom_states``.
        save_path: File path for the output image (e.g. ``'atom_states.png'``).
    """
    atom_states = getattr(clf, 'atom_states', [])
    if not atom_states:
        logger.warning("[Viz] No atom_states to plot.")
        return

    # Count each state
    from collections import Counter
    counts = Counter(atom_states)
    labels = list(counts.keys())
    sizes = list(counts.values())

    if _MPL_AVAILABLE:
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = {
            'protected': '#4CAF50',
            'plastic': '#2196F3',
            'inactive': '#9E9E9E',
        }
        pie_colors = [colors.get(l, '#FF9800') for l in labels]

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=90,
            textprops={'fontsize': 12},
        )
        for autotext in autotexts:
            autotext.set_fontsize(11)

        ax.set_title('Atom State Distribution', fontsize=14, fontweight='bold')

        # Add total count annotation
        ax.annotate(
            f'Total atoms: {len(atom_states)}',
            xy=(0, 0), fontsize=10,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8),
        )

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"[Viz] Atom state distribution saved to {save_path}")
    else:
        # Text fallback
        lines = ["Atom State Distribution:", "-" * 30]
        for label, size in zip(labels, sizes):
            pct = size / len(atom_states) * 100
            bar = '#' * int(pct / 2)
            lines.append(f"  {label:12s}: {size:4d} ({pct:5.1f}%) {bar}")
        lines.append(f"  {'Total':12s}: {len(atom_states)}")

        text_path = save_path.rsplit('.', 1)[0] + '.txt'
        with open(text_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        logger.info(f"[Viz] Atom state distribution saved to {text_path} (text fallback)")


def plot_residual_histogram(clf, save_path: str, bins: int = 30) -> None:
    """Plot a histogram of node reconstruction residuals.

    Falls back to text output when matplotlib is unavailable.

    Args:
        clf: HCSOINNClassifier instance.
        save_path: File path for the output image.
        bins: Number of histogram bins.
    """
    residuals: List[float] = []
    for cls, clusters in getattr(clf, 'class_clusters', {}).items():
        for node in clusters:
            if node.node_state != 'inactive' and node.coeff is not None:
                residuals.append(node.residual)

    if not residuals:
        logger.warning("[Viz] No residuals to plot.")
        return

    residuals_arr = np.array(residuals)

    if _MPL_AVAILABLE:
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.hist(residuals_arr, bins=bins, color='#2196F3', edgecolor='white',
                alpha=0.85, density=False)

        # Add statistics lines
        mean_r = float(np.mean(residuals_arr))
        median_r = float(np.median(residuals_arr))

        ax.axvline(mean_r, color='#F44336', linestyle='--', linewidth=1.5,
                   label=f'Mean: {mean_r:.4f}')
        ax.axvline(median_r, color='#FF9800', linestyle=':', linewidth=1.5,
                   label=f'Median: {median_r:.4f}')

        ax.set_xlabel('Reconstruction Residual (1 - cos_sim)', fontsize=12)
        ax.set_ylabel('Number of Nodes', fontsize=12)
        ax.set_title('Node Reconstruction Residual Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        # Add text box with stats
        stats_text = (
            f'N = {len(residuals)}\n'
            f'Mean = {mean_r:.4f}\n'
            f'Std  = {float(np.std(residuals_arr)):.4f}\n'
            f'Min  = {float(np.min(residuals_arr)):.4f}\n'
            f'Max  = {float(np.max(residuals_arr)):.4f}'
        )
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
        ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=props)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"[Viz] Residual histogram saved to {save_path}")
    else:
        # Text fallback: ASCII histogram
        hist, bin_edges = np.histogram(residuals_arr, bins=bins)
        max_count = max(hist) if max(hist) > 0 else 1

        lines = ["Reconstruction Residual Histogram:", "-" * 50]
        for i in range(len(hist)):
            bar_len = int(hist[i] / max_count * 40)
            bar = '#' * bar_len
            lines.append(f"  [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}) | {bar} ({hist[i]})")

        lines.append("-" * 50)
        lines.append(f"  N={len(residuals)}, Mean={float(np.mean(residuals_arr)):.4f}, "
                     f"Std={float(np.std(residuals_arr)):.4f}")

        text_path = save_path.rsplit('.', 1)[0] + '.txt'
        with open(text_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        logger.info(f"[Viz] Residual histogram saved to {text_path} (text fallback)")


def plot_atom_usage_heatmap(clf, save_path: str) -> None:
    """Plot a heatmap of per-class mean |a_{c,m}| across atoms.

    X-axis: atom index.
    Y-axis: class ID.
    Cell value: mean |a_{c,m}| across all active nodes of class c.

    Falls back to text output when matplotlib is unavailable.

    Args:
        clf: HCSOINNClassifier instance.
        save_path: File path for the output image.
    """
    class_clusters = getattr(clf, 'class_clusters', {})
    if not class_clusters:
        logger.warning("[Viz] No class_clusters to plot.")
        return

    # Collect classes and M dimension
    classes_sorted = sorted(class_clusters.keys())
    if not classes_sorted:
        return

    # Determine M from the first available coefficient
    M = 0
    for cls in classes_sorted:
        for node in class_clusters[cls]:
            if node.coeff is not None:
                M = max(M, node.coeff.shape[-1])

    if M == 0:
        logger.warning("[Viz] No coefficient data found.")
        return

    # Build heatmap matrix [C, M]
    heatmap = np.zeros((len(classes_sorted), M), dtype=np.float32)
    counts = np.zeros(len(classes_sorted), dtype=np.float32)

    for ci, cls in enumerate(classes_sorted):
        for node in class_clusters[cls]:
            if node.node_state == 'inactive' or node.coeff is None:
                continue
            coeff_abs = np.abs(node.coeff.ravel())
            if len(coeff_abs) >= M:
                heatmap[ci, :] += coeff_abs[:M]
            else:
                heatmap[ci, :len(coeff_abs)] += coeff_abs
            counts[ci] += 1.0

    # Normalize by node count per class
    for ci in range(len(classes_sorted)):
        if counts[ci] > 0:
            heatmap[ci, :] /= counts[ci]

    if _MPL_AVAILABLE:
        fig, ax = plt.subplots(figsize=(max(8, M * 0.15 + 2), max(5, len(classes_sorted) * 0.3 + 2)))

        im = ax.imshow(heatmap, aspect='auto', cmap='YlOrRd', interpolation='nearest')

        ax.set_xlabel('Atom Index', fontsize=12)
        ax.set_ylabel('Class ID', fontsize=12)
        ax.set_title('Atom Usage Heatmap (mean |a_{c,m}| per class)', fontsize=14, fontweight='bold')

        # Tick labels
        if M <= 50:
            ax.set_xticks(range(M))
        else:
            step = max(1, M // 20)
            ax.set_xticks(range(0, M, step))

        if len(classes_sorted) <= 30:
            ax.set_yticks(range(len(classes_sorted)))
            ax.set_yticklabels([str(c) for c in classes_sorted], fontsize=8)
        else:
            step = max(1, len(classes_sorted) // 20)
            ax.set_yticks(range(0, len(classes_sorted), step))
            ax.set_yticklabels([str(classes_sorted[i]) for i in range(0, len(classes_sorted), step)],
                               fontsize=8)

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Mean |coefficient|', fontsize=10)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"[Viz] Atom usage heatmap saved to {save_path}")
    else:
        # Text fallback: compact table
        lines = ["Atom Usage Heatmap (text fallback):", "-" * 60]

        # Show top-10 atoms by total usage
        col_sums = heatmap.sum(axis=0)
        top_atoms = np.argsort(col_sums)[::-1][:10]
        top_atoms = sorted(top_atoms)

        header = f"{'Class':>6s}"
        for m in top_atoms:
            header += f"  a_{m:>3d}"
        lines.append(header)
        lines.append("-" * len(header))

        for ci, cls in enumerate(classes_sorted):
            row = f"{cls:>6d}"
            for m in top_atoms:
                row += f"  {heatmap[ci, m]:.3f}"
            lines.append(row)

        text_path = save_path.rsplit('.', 1)[0] + '.txt'
        with open(text_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        logger.info(f"[Viz] Atom usage heatmap saved to {text_path} (text fallback)")
