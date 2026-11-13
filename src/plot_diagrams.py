"""Schematic figures of the manuscript: the evaluation protocol and the model.

Both are drawn from the numbers the pipeline produces, so the figures cannot
drift from the code: partition sizes come from data/cache via
src/corrected_protocol.py when available, otherwise from the values recorded
in outputs/results.

Run:  python -m src.plot_diagrams  ->  outputs/figures/fig_protocol.png
                                      outputs/figures/fig_architecture.png
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from . import config

INK = "#1A1D23"
MUTED = "#4A5361"
FILL_DATA = "#E8EEF7"
FILL_STEP = "#F4F4F4"
FILL_DEV = "#E6F2EA"
FILL_TEST = "#FBEDE7"
EDGE = "#5B6573"

plt.rcParams.update({"font.family": "Arial", "font.size": 7.2})


def _box(ax, x, y, w, h, text, fill=FILL_STEP, bold_first=True, fs=7.2, lh=0.036):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                linewidth=0.8, edgecolor=EDGE, facecolor=fill))
    lines = text.split("\n")
    n = len(lines)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h / 2 + (n - 1) * 0.5 * lh - i * lh, ln,
                ha="center", va="center", fontsize=fs, color=INK,
                weight="bold" if (bold_first and i == 0) else "normal")


def _arrow(ax, x0, y0, x1, y1, text=None):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=EDGE, shrinkA=0, shrinkB=0,
                                mutation_scale=7))
    if text:
        ax.text((x0 + x1) / 2 + 0.006, (y0 + y1) / 2, text, fontsize=6.4, color=MUTED,
                ha="left", va="center")


def _elbow(ax, pts, text=None, text_at=None):
    """Orthogonal connector through the given points, arrowhead at the end."""
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=EDGE, lw=0.8, solid_capstyle="butt")
    _arrow(ax, xs[-2], ys[-2], xs[-1], ys[-1])
    if text:
        ax.text(*text_at, text, fontsize=6.2, color=MUTED, ha="left", va="bottom")


def protocol_figure(path):
    fig = plt.figure(figsize=(6.5, 3.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fs = 6.6

    # row 1: inputs, pooling, de-identification, sender-grouped split
    _box(ax, 0.01, 0.865, 0.14, 0.115, "Training file (70%)\n2,236,367 messages", FILL_DATA, fs=fs)
    _box(ax, 0.01, 0.705, 0.14, 0.115, "Test file (30%)\n958,441 messages", FILL_DATA, fs=fs)
    _box(ax, 0.19, 0.72, 0.19, 0.25, "Pooled corpus\n= released file\n3,194,808 messages\n24,663 senders; 99.92%\nof test-file senders also\nin the training file", FILL_DATA, fs=fs)
    _arrow(ax, 0.15, 0.92, 0.19, 0.88)
    _arrow(ax, 0.15, 0.76, 0.19, 0.80)
    _box(ax, 0.42, 0.72, 0.25, 0.25, "De-identification\nremove sender, senderPseudo,\nmessageID and type; add four\nper-sender temporal features;\ndrop sendTime → 28 features", FILL_STEP, fs=fs)
    _arrow(ax, 0.38, 0.845, 0.42, 0.845)
    _box(ax, 0.71, 0.72, 0.24, 0.25, "Sender-grouped split\nStratifiedGroupKFold, 3 groups;\nno sender in two partitions\n(checked by assertion)", FILL_STEP, fs=fs)
    _arrow(ax, 0.67, 0.845, 0.71, 0.845)

    # row 2: development branch
    _box(ax, 0.01, 0.40, 0.18, 0.22, "Development\npartition\n2,129,867 messages\n16,441 senders", FILL_DEV, fs=fs)
    _box(ax, 0.23, 0.40, 0.23, 0.22, "5 sender-disjoint folds\ntraining ≈ 1.70 M messages\nvalidation ≈ 426 K messages\n≈ 3,290 senders per fold", FILL_DEV, fs=fs)
    _box(ax, 0.50, 0.40, 0.22, 0.22, "Per fold, training part only\nscaler fit + ADASYN,\ncomputed once and cached;\nidentical arrays for all models", FILL_DEV, fs=fs)
    _box(ax, 0.76, 0.40, 0.19, 0.22, "Model comparison\nsame folds, budget\nand seeds; mean ± SD;\ncorrected t-test", FILL_DEV, fs=fs)
    _arrow(ax, 0.19, 0.51, 0.23, 0.51)
    _arrow(ax, 0.46, 0.51, 0.50, 0.51)
    _arrow(ax, 0.72, 0.51, 0.76, 0.51)
    _elbow(ax, [(0.80, 0.72), (0.80, 0.665), (0.10, 0.665), (0.10, 0.62)])

    # row 3: final refit and single evaluation on the held-out partition
    _box(ax, 0.01, 0.03, 0.32, 0.25, "Final refit\non the full development partition;\nscaler + ADASYN on development data;\nfixed epochs = median best CV epoch;\nno validation data, no early stopping", FILL_STEP, fs=fs)
    _box(ax, 0.38, 0.03, 0.27, 0.25, "Single evaluation\nof each final model: per-class\nmetrics, confusion matrix and\nfalse alarm rate", FILL_TEST, fs=fs)
    _box(ax, 0.70, 0.03, 0.25, 0.25, "Held-out partition\n1,064,941 messages\n8,222 senders never seen\nduring development", FILL_TEST, fs=fs)
    _arrow(ax, 0.10, 0.40, 0.10, 0.28)
    _elbow(ax, [(0.855, 0.40), (0.855, 0.335), (0.25, 0.335), (0.25, 0.28)],
           text="epoch budget", text_at=(0.40, 0.341))
    _arrow(ax, 0.33, 0.155, 0.38, 0.155)
    _arrow(ax, 0.70, 0.155, 0.65, 0.155)
    _elbow(ax, [(0.95, 0.80), (0.975, 0.80), (0.975, 0.30), (0.90, 0.30), (0.90, 0.28)])

    fig.savefig(path, dpi=400)
    plt.close(fig)
    print(f"  wrote {path}")


def architecture_figure(path):
    fig = plt.figure(figsize=(3.0, 3.9))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fs = 6.8

    _box(ax, 0.18, 0.905, 0.64, 0.075, "Input: 28 standardized features\nreshaped to 28 × 1", FILL_DATA, fs=fs)
    xs = [0.03, 0.355, 0.68]
    for x, k in zip(xs, (1, 3, 5)):
        _box(ax, x, 0.745, 0.29, 0.10, f"Conv1D, k = {k}\n96 filters, ReLU\n+ batch norm.", FILL_STEP, fs=fs)
        _arrow(ax, 0.5, 0.905, x + 0.145, 0.845)
        _arrow(ax, x + 0.145, 0.745, 0.5, 0.685)
    _box(ax, 0.18, 0.62, 0.64, 0.065, "Concatenate → 28 × 288", FILL_STEP, fs=fs)
    _box(ax, 0.10, 0.46, 0.80, 0.115, "Bidirectional GRU\n64 + 64 units, all positions returned\nrecurrent dropout 0.2 → 28 × 128", FILL_DEV, fs=fs)
    _arrow(ax, 0.5, 0.62, 0.5, 0.575)
    _box(ax, 0.18, 0.37, 0.64, 0.055, "Dropout 0.3", FILL_STEP, fs=fs)
    _arrow(ax, 0.5, 0.46, 0.5, 0.425)
    _box(ax, 0.06, 0.18, 0.88, 0.15, "Additive attention\n$e_t = \\tanh(\\mathbf{w}_a^{\\top}\\mathbf{h}_t + b_a)$,  "
         "$\\alpha_t = \\mathrm{softmax}_t(e_t)$\n$\\mathbf{u} = \\Sigma_t\\, \\alpha_t \\mathbf{h}_t$  → 128",
         FILL_DEV, fs=fs, lh=0.042)
    _arrow(ax, 0.5, 0.37, 0.5, 0.33)
    _box(ax, 0.12, 0.03, 0.76, 0.10, "Dense, softmax (L2 = $10^{-4}$)\n20 class probabilities", FILL_TEST, fs=fs)
    _arrow(ax, 0.5, 0.18, 0.5, 0.13)

    fig.savefig(path, dpi=400)
    plt.close(fig)
    print(f"  wrote {path}")


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    protocol_figure(config.FIGURES_DIR / "fig_protocol.png")
    architecture_figure(config.FIGURES_DIR / "fig_architecture.png")


if __name__ == "__main__":
    main()
