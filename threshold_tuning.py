"""Post-hoc per-class decision-threshold tuning (no retraining).

Diagnosis: after fixing Benign undersampling, 4 classes (Disruptive Attack,
Disruptive Position Attack, Data Replay Attack, DDoS) collapsed to near-zero
recall even though their precision-when-predicted stayed reasonable -- the
model has real signal for them, it just never wins the argmax against more
confident classes (chiefly Benign).

This module:
  1. Reconstructs the exact same 5 StratifiedKFold splits used in training
     (fixed random_state) and, for each fold, uses THAT fold's own model+
     scaler to predict on its held-out validation slice -- stitching
     together out-of-fold (OOF) predictions covering the entire training
     set, with zero leakage (every row is scored by a model that never saw
     it in training or resampling).
  2. Searches a per-class additive probability boost for the collapsed
     classes that maximizes accuracy on the OOF set.
  3. Applies the SAME fixed boosts to the real ensemble test-set
     probabilities (already computed by evaluate.py) and reports the
     adjusted metrics -- the test set is only ever touched once, at the end,
     with a rule chosen entirely from training-side OOF data.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from . import config
from .data_pipeline import load_train, load_test
from .model import AdditiveAttention  # noqa: F401


def compute_oof_probs():
    X, y = load_train()
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    oof_probs = np.zeros((X.shape[0], config.N_CLASSES), dtype="float32")
    oof_y = y.copy()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        model = tf.keras.models.load_model(config.MODELS_DIR / f"fold{fold_idx}.keras", compile=False)
        with open(config.MODELS_DIR / f"scaler_fold{fold_idx}.pkl", "rb") as f:
            scaler = pickle.load(f)
        Xv = scaler.transform(X[val_idx]).reshape(-1, config.N_FEATURES, 1).astype("float32")
        probs = model.predict(Xv, batch_size=config.BATCH_SIZE, verbose=0)
        oof_probs[val_idx] = probs
        print(f"[oof] fold {fold_idx} done ({len(val_idx)} rows)", flush=True)

    return oof_probs, oof_y


def search_boosts(oof_probs: np.ndarray, oof_y: np.ndarray, target_classes: list[int],
                   grid=(0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6)):
    """Greedy coordinate search: for each target class in turn, pick the
    additive boost (added to that class's probability before argmax) that
    maximizes overall accuracy on the OOF set, holding other boosts fixed.
    """
    boosts = {c: 0.0 for c in target_classes}
    base_pred = oof_probs.argmax(axis=1)
    best_acc = accuracy_score(oof_y, base_pred)
    print(f"[search] baseline OOF accuracy: {best_acc:.4f}", flush=True)

    improved = True
    n_rounds = 0
    while improved and n_rounds < 3:
        improved = False
        n_rounds += 1
        for c in target_classes:
            best_b = boosts[c]
            for b in grid:
                trial = dict(boosts)
                trial[c] = b
                adj = oof_probs.copy()
                for cc, bb in trial.items():
                    adj[:, cc] += bb
                pred = adj.argmax(axis=1)
                acc = accuracy_score(oof_y, pred)
                if acc > best_acc:
                    best_acc = acc
                    best_b = b
                    improved = True
            boosts[c] = best_b
            print(f"[search] round {n_rounds}, class {c}: boost={best_b}, OOF acc={best_acc:.4f}", flush=True)

    return boosts, best_acc


def apply_boosts_and_report(test_probs: np.ndarray, y_test: np.ndarray, boosts: dict):
    adj = test_probs.copy()
    for c, b in boosts.items():
        adj[:, c] += b
    pred = adj.argmax(axis=1)

    acc = accuracy_score(y_test, pred)
    report = classification_report(
        y_test, pred, output_dict=True, zero_division=0,
        labels=list(range(config.N_CLASSES)),
        target_names=[config.CLASS_NAMES[c] for c in range(config.N_CLASSES)],
    )
    cm = confusion_matrix(y_test, pred, labels=list(range(config.N_CLASSES)))
    return acc, report, cm, pred


if __name__ == "__main__":
    COLLAPSED = [10, 11, 13, 15]  # Disruptive, Disruptive Position, Data Replay, DDoS

    print("Computing out-of-fold predictions on the training set...", flush=True)
    oof_probs, oof_y = compute_oof_probs()
    np.save(config.RESULTS_DIR / "oof_probs.npy", oof_probs)
    np.save(config.RESULTS_DIR / "oof_y.npy", oof_y)

    print("\nSearching per-class boosts on OOF data...", flush=True)
    boosts, oof_acc = search_boosts(oof_probs, oof_y, COLLAPSED)
    print(f"\nFinal boosts: {boosts}")
    print(f"Final OOF accuracy: {oof_acc:.4f}")

    with open(config.RESULTS_DIR / "threshold_boosts.json", "w") as f:
        json.dump({"boosts": {config.CLASS_NAMES[c]: b for c, b in boosts.items()},
                    "boosts_raw": boosts, "oof_accuracy": oof_acc}, f, indent=2)

    print("\nApplying boosts to the REAL test set (first and only touch)...", flush=True)
    X_test, y_test = load_test()
    models_, scalers_ = [], []
    for i in range(1, config.N_FOLDS + 1):
        models_.append(tf.keras.models.load_model(config.MODELS_DIR / f"fold{i}.keras", compile=False))
        with open(config.MODELS_DIR / f"scaler_fold{i}.pkl", "rb") as f:
            scalers_.append(pickle.load(f))

    probs_sum = np.zeros((X_test.shape[0], config.N_CLASSES), dtype="float64")
    for model, scaler in zip(models_, scalers_):
        Xs = scaler.transform(X_test).reshape(-1, config.N_FEATURES, 1).astype("float32")
        probs_sum += model.predict(Xs, batch_size=config.BATCH_SIZE, verbose=0)
    test_probs = (probs_sum / len(models_)).astype("float32")
    np.save(config.RESULTS_DIR / "test_probs.npy", test_probs)

    acc, report, cm, pred = apply_boosts_and_report(test_probs, y_test, boosts)
    print(f"\n=== FINAL (threshold-adjusted) TEST ACCURACY: {acc*100:.2f}% ===")

    rows = []
    for c in range(config.N_CLASSES):
        name = config.CLASS_NAMES[c]
        r = report[name]
        rows.append({
            "class_id": c, "class_name": name,
            "precision_pct": round(r["precision"] * 100, 1),
            "recall_pct": round(r["recall"] * 100, 1),
            "f1_pct": round(r["f1-score"] * 100, 1),
            "support": int(r["support"]),
        })
    with open(config.RESULTS_DIR / "table2_threshold_adjusted.json", "w") as f:
        json.dump(rows, f, indent=2)

    summary = {
        "accuracy_pct": round(acc * 100, 2),
        "macro_precision_pct": round(report["macro avg"]["precision"] * 100, 2),
        "macro_recall_pct": round(report["macro avg"]["recall"] * 100, 2),
        "macro_f1_pct": round(report["macro avg"]["f1-score"] * 100, 2),
        "weighted_precision_pct": round(report["weighted avg"]["precision"] * 100, 2),
        "weighted_recall_pct": round(report["weighted avg"]["recall"] * 100, 2),
        "weighted_f1_pct": round(report["weighted avg"]["f1-score"] * 100, 2),
        "boosts": {config.CLASS_NAMES[c]: b for c, b in boosts.items()},
    }
    with open(config.RESULTS_DIR / "summary_threshold_adjusted.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.save(config.RESULTS_DIR / "confusion_matrix_threshold_adjusted.npy", cm)

    print(json.dumps(summary, indent=2))
