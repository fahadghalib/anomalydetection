"""Legible confusion-matrix figure for the corrected protocol.

The reviewer noted that the manuscript's confusion matrix is too small to
read. A 20x20 matrix of raw counts printed at column width is unreadable by
construction, so this renders it differently:

  * cells carry ROW-NORMALISED percentages (what fraction of each true class
    went where), which is the quantity a reader actually wants and which
    keeps every cell on one 0-100 scale;
  * only cells above a visibility threshold are annotated, so the diagonal
    and the real confusions stand out instead of being lost among zeros;
  * the figure is rendered large with readable type, and off-diagonal mass
    is summarised in a caption line naming the worst confusions explicitly.

Run:  python -m src.plot_confusion
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

INK = "#1A1D23"
MUTED = "#606A78"
ANNOT_MIN = 1.0          # percent; below this a cell is left blank


def plot_one(cm: np.ndarray, title: str, out_path):
    row_tot = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm * 100.0, np.maximum(row_tot, 1), dtype="float64")

    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)

    labels = [f"{c}  {config.CLASS_NAMES[c]}" for c in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n), fontsize=11, color=INK)
    ax.set_yticklabels(labels, fontsize=10.5, color=INK)
    ax.set_xlabel("Predicted class", fontsize=13, color=INK, labelpad=10)
    ax.set_ylabel("True class", fontsize=13, color=INK, labelpad=10)
    ax.set_title(title, fontsize=15, color=INK, pad=16)

    for i in range(n):
        for j in range(n):
            v = pct[i, j]
            if v < ANNOT_MIN:
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=9.5, weight="bold" if i == j else "normal",
                    color="white" if v > 55 else INK)

    cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label("% of true class", fontsize=11, color=INK)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", length=0)

    # Name the worst confusions rather than leaving them to be hunted for.
    off = pct.copy()
    np.fill_diagonal(off, 0.0)
    worst = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1], off.shape))[0][:4]
    items = [f"{config.CLASS_NAMES[i]} → {config.CLASS_NAMES[j]}  ({off[i, j]:.0f}%)"
             for i, j in worst if off[i, j] >= 5]
    if items:
        # One per line: a single joined line overflows the figure width.
        text = "Largest confusions\n" + "\n".join("   • " + s for s in items)
        fig.text(0.5, 0.012, text, ha="center", va="bottom",
                 fontsize=10.5, color=MUTED, linespacing=1.45)

    fig.tight_layout(rect=(0, 0.035 + 0.017 * len(items), 1, 1))
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    found = False
    for path in sorted(config.RESULTS_DIR.glob("confusion_matrix_corrected_*.npy")):
        key = path.stem.replace("confusion_matrix_corrected_", "")
        cm = np.load(path)
        plot_one(cm,
                 f"Row-normalised confusion matrix — {key}, "
                 f"sender-disjoint held-out test set",
                 config.FIGURES_DIR / f"confusion_corrected_{key}.png")
        found = True
    if not found:
        print("  no corrected-protocol confusion matrices found yet; run "
              "`python -m src.run_corrected --models ... --eval-test` first")


if __name__ == "__main__":
    main()
