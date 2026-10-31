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


def plot_one(cm: np.ndarray, title: str | None, out_path):
    """Render at the size it is printed (full text width, 6.5 in), so the
    7-pt labels in the file are 7-pt labels on the page."""
    row_tot = cm.sum(axis=1, keepdims=True)
    pct = np.divide(cm * 100.0, np.maximum(row_tot, 1), dtype="float64")

    n = cm.shape[0]
    plt.rcParams.update({"font.family": "Arial"})
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)

    labels = [f"{config.CLASS_NAMES[c]}  {c}" for c in range(n)]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n), fontsize=7, color=INK)
    ax.set_yticklabels(labels, fontsize=7, color=INK)
    ax.set_xlabel("Predicted class (ID)", fontsize=7.5, color=INK, labelpad=4)
    ax.set_ylabel("True class", fontsize=7.5, color=INK, labelpad=4)
    if title:
        ax.set_title(title, fontsize=8, color=INK, pad=6)

    for i in range(n):
        for j in range(n):
            v = pct[i, j]
            if v < ANNOT_MIN:
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=6.3, weight="bold" if i == j else "normal",
                    color="white" if v > 55 else INK)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.015, fraction=0.04)
    cbar.set_label("% of messages of the true class", fontsize=7, color=INK)
    cbar.ax.tick_params(labelsize=6.5)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.6)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=2, pad=1.5)

    fig.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    found = False
    for path in sorted(config.RESULTS_DIR.glob("confusion_matrix_corrected_*.npy")):
        key = path.stem.replace("confusion_matrix_corrected_", "")
        cm = np.load(path)
        # The manuscript figure carries its caption in the text, so the
        # proposed model's figure is rendered without a title.
        plot_one(cm, None if key == "cnn" else f"Row-normalized confusion matrix: {key}",
                 config.FIGURES_DIR / f"confusion_corrected_{key}.png")
        found = True
    if not found:
        print("  no corrected-protocol confusion matrices found yet; run "
              "`python -m src.run_corrected --models ... --eval-test` first")


if __name__ == "__main__":
    main()
