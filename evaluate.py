"""Ensemble (soft-voting) evaluation on the held-out VeReMi test set.

Produces, from REAL predictions on REAL data (no numbers copied from the
paper):
  - outputs/results/table2_per_class_metrics.csv / .json  (paper's Table 2)
  - outputs/results/summary_metrics.json                  (accuracy, macro/
    weighted P/R/F1, FAR, param count, inference latency)
  - outputs/figures/confusion_matrix.png
  - outputs/figures/per_class_metrics.png
  - outputs/figures/far_per_class.png
"""

from __future__ import annotations

import json
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from . import config
from .data_pipeline import load_test
from .model import AdditiveAttention  # noqa: F401 (needed for model deserialization)


def _load_fold_artifacts(n_folds: int = config.N_FOLDS):
    models_, scalers_ = [], []
    for i in range(1, n_folds + 1):
        model_path = config.MODELS_DIR / f"fold{i}.keras"
        scaler_path = config.MODELS_DIR / f"scaler_fold{i}.pkl"
        # compile=False: we only need forward inference here, and the custom
        # focal-loss closure (baked with each fold's own class weights) is
        # not meant to be deserialized generically.
        models_.append(tf.keras.models.load_model(model_path, compile=False))
        with open(scaler_path, "rb") as f:
            scalers_.append(pickle.load(f))
    return models_, scalers_


def ensemble_predict_proba(X: np.ndarray, models_, scalers_) -> np.ndarray:
    """Average softmax probabilities across folds; each fold uses its OWN
    scaler (fit only on that fold's training data) -- Eq. (soft voting)."""
    probs_sum = np.zeros((X.shape[0], config.N_CLASSES), dtype="float64")
    for model, scaler in zip(models_, scalers_):
        Xs = scaler.transform(X).reshape(-1, config.N_FEATURES, 1).astype("float32")
        probs_sum += model.predict(Xs, batch_size=config.BATCH_SIZE, verbose=0)
    return (probs_sum / len(models_)).astype("float32")


def compute_far(cm: np.ndarray) -> dict:
    """False Alarm Rate, Eq. (13): FAR_benign = FP_benign / (TN_benign + FP_benign).

    Framed binary (attack=positive, benign=negative): FP_benign = actual-benign
    rows predicted as any attack class; TN_benign = actual-benign rows
    correctly predicted as benign. FP+TN for the benign row = the total count
    of actual benign samples, so FAR_benign reduces to 1 - recall(benign).
    """
    benign = 0
    row_total = cm[benign, :].sum()
    tp_benign = cm[benign, benign]
    fp_benign = row_total - tp_benign
    far_benign = fp_benign / row_total if row_total > 0 else 0.0
    return {"far_benign_pct": float(far_benign * 100)}


def benchmark_latency(models_, scalers_, X_sample: np.ndarray, n_repeats: int = 5) -> dict:
    """Average per-sample inference latency of the ensemble (ms/sample)."""
    Xs_list = [
        scaler.transform(X_sample).reshape(-1, config.N_FEATURES, 1).astype("float32")
        for scaler in scalers_
    ]
    # warmup
    for model, Xs in zip(models_, Xs_list):
        model.predict(Xs, batch_size=config.BATCH_SIZE, verbose=0)

    times = []
    for _ in range(n_repeats):
        t0 = time.time()
        for model, Xs in zip(models_, Xs_list):
            model.predict(Xs, batch_size=config.BATCH_SIZE, verbose=0)
        times.append(time.time() - t0)
    avg_total = sum(times) / len(times)
    ms_per_sample = (avg_total / X_sample.shape[0]) * 1000
    return {"ms_per_sample_ensemble": float(ms_per_sample), "n_sample": int(X_sample.shape[0])}


def total_params(models_) -> int:
    return int(sum(m.count_params() for m in models_))


# ---------------------------------------------------------------------------
# Figures (dataviz principles: single sequential hue for magnitude, direct
# value labels, recessive gridlines, no dual axes, no rainbow categorical use
# where there is only one series).
# ---------------------------------------------------------------------------
SEQ_BLUE = "#2E5FA3"
INK = "#1F2430"
MUTED = "#6B7280"
GRID = "#E5E7EB"


def plot_confusion_matrix(cm: np.ndarray, class_ids, out_path):
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_ids)))
    ax.set_yticks(range(len(class_ids)))
    ax.set_xticklabels(class_ids, fontsize=8)
    ax.set_yticklabels(class_ids, fontsize=8)
    ax.set_xlabel("Predicted Class ID", color=INK)
    ax.set_ylabel("Actual Class ID", color=INK)
    ax.set_title("Ensemble Confusion Matrix — VeReMi Extension Test Set (real run)", color=INK)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(labelsize=8)
    vmax = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = cm[i, j]
            if v == 0:
                continue
            color = "white" if v > vmax * 0.5 else INK
            ax.text(j, i, str(v), ha="center", va="center", fontsize=5.5, color=color)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_class_bar(values, labels, title, xlabel, out_path, pct=True):
    order = np.argsort(values)
    values_sorted = np.array(values)[order]
    labels_sorted = np.array(labels)[order]
    fig, ax = plt.subplots(figsize=(9, 8))
    y_pos = np.arange(len(values_sorted))
    ax.barh(y_pos, values_sorted, color=SEQ_BLUE, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels_sorted, fontsize=8, color=INK)
    ax.set_xlabel(xlabel, color=INK)
    ax.set_title(title, color=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    for y, v in zip(y_pos, values_sorted):
        suffix = "%" if pct else ""
        ax.text(v, y, f" {v:.1f}{suffix}", va="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_evaluation(n_folds: int = config.N_FOLDS):
    print("Loading fold models + scalers...", flush=True)
    models_, scalers_ = _load_fold_artifacts(n_folds)

    print("Loading REAL held-out test set...", flush=True)
    X_test, y_test = load_test()

    print("Running ensemble soft-voting inference on the test set...", flush=True)
    probs = ensemble_predict_proba(X_test, models_, scalers_)
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0,
        labels=list(range(config.N_CLASSES)),
        target_names=[config.CLASS_NAMES[c] for c in range(config.N_CLASSES)],
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(config.N_CLASSES)))
    far = compute_far(cm)

    sample_idx = np.random.RandomState(config.RANDOM_STATE).choice(
        len(X_test), size=min(2000, len(X_test)), replace=False
    )
    latency = benchmark_latency(models_, scalers_, X_test[sample_idx])
    n_params = total_params(models_)

    # --- Table 2 ---
    rows = []
    for c in range(config.N_CLASSES):
        name = config.CLASS_NAMES[c]
        r = report[name]
        rows.append({
            "class_id": c,
            "class_name": name,
            "precision_pct": round(r["precision"] * 100, 1),
            "recall_pct": round(r["recall"] * 100, 1),
            "f1_pct": round(r["f1-score"] * 100, 1),
            "support": int(r["support"]),
        })
    table2 = pd.DataFrame(rows)
    table2.to_csv(config.RESULTS_DIR / "table2_per_class_metrics.csv", index=False)
    with open(config.RESULTS_DIR / "table2_per_class_metrics.json", "w") as f:
        json.dump(rows, f, indent=2)

    summary = {
        "accuracy_pct": round(acc * 100, 2),
        "macro_precision_pct": round(report["macro avg"]["precision"] * 100, 2),
        "macro_recall_pct": round(report["macro avg"]["recall"] * 100, 2),
        "macro_f1_pct": round(report["macro avg"]["f1-score"] * 100, 2),
        "weighted_precision_pct": round(report["weighted avg"]["precision"] * 100, 2),
        "weighted_recall_pct": round(report["weighted avg"]["recall"] * 100, 2),
        "weighted_f1_pct": round(report["weighted avg"]["f1-score"] * 100, 2),
        "false_alarm_rate_benign_pct": round(far["far_benign_pct"], 3),
        "n_test_samples": int(len(y_test)),
        "n_folds_in_ensemble": n_folds,
        "total_trainable_params_all_folds": n_params,
        "inference_latency_ms_per_sample_ensemble": round(latency["ms_per_sample_ensemble"], 4),
    }
    with open(config.RESULTS_DIR / "summary_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.save(config.RESULTS_DIR / "confusion_matrix.npy", cm)
    pd.DataFrame(
        cm, index=[f"actual_{c}" for c in range(config.N_CLASSES)],
        columns=[f"pred_{c}" for c in range(config.N_CLASSES)],
    ).to_csv(config.RESULTS_DIR / "confusion_matrix.csv")

    # --- figures ---
    plot_confusion_matrix(cm, list(range(config.N_CLASSES)), config.FIGURES_DIR / "confusion_matrix.png")
    plot_per_class_bar(
        table2["recall_pct"].tolist(), table2["class_name"].tolist(),
        "Per-Class Recall — Real Ensemble Result on VeReMi Test Set",
        "Recall (%)", config.FIGURES_DIR / "per_class_recall.png",
    )
    plot_per_class_bar(
        table2["precision_pct"].tolist(), table2["class_name"].tolist(),
        "Per-Class Precision — Real Ensemble Result on VeReMi Test Set",
        "Precision (%)", config.FIGURES_DIR / "per_class_precision.png",
    )
    plot_per_class_bar(
        table2["f1_pct"].tolist(), table2["class_name"].tolist(),
        "Per-Class F1-Score — Real Ensemble Result on VeReMi Test Set",
        "F1-Score (%)", config.FIGURES_DIR / "per_class_f1.png",
    )

    print("\n=== SUMMARY (real numbers, this run) ===")
    print(json.dumps(summary, indent=2))

    return summary, table2


if __name__ == "__main__":
    run_evaluation()
