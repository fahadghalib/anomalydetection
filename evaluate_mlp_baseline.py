"""Per-class ensemble evaluation of the M8 strong-MLP baseline on the REAL
held-out test set -- mirrors evaluate.py's Table 2, but for the flat-input
MLP models (outputs/models/mlp_baseline_fold{1..5}.keras) instead of the
CNN-BiGRU-Attention models, so M7 and M8 can be compared class-by-class on
the exact same test set, not just on aggregate macro metrics.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from . import config
from .data_pipeline import load_test


def _load_mlp_fold_artifacts(n_folds: int = config.N_FOLDS):
    models_, scalers_ = [], []
    for i in range(1, n_folds + 1):
        model_path = config.MODELS_DIR / f"mlp_baseline_fold{i}.keras"
        scaler_path = config.MODELS_DIR / f"scaler_mlp_baseline_fold{i}.pkl"
        models_.append(tf.keras.models.load_model(model_path, compile=False))
        with open(scaler_path, "rb") as f:
            scalers_.append(pickle.load(f))
    return models_, scalers_


def ensemble_predict_proba_flat(X: np.ndarray, models_, scalers_) -> np.ndarray:
    probs_sum = np.zeros((X.shape[0], config.N_CLASSES), dtype="float64")
    for model, scaler in zip(models_, scalers_):
        Xs = scaler.transform(X).astype("float32")
        probs_sum += model.predict(Xs, batch_size=config.BATCH_SIZE, verbose=0)
    return (probs_sum / len(models_)).astype("float32")


def run_evaluation_mlp(n_folds: int = config.N_FOLDS):
    print("Loading M8 MLP fold models + scalers...", flush=True)
    models_, scalers_ = _load_mlp_fold_artifacts(n_folds)

    print("Loading REAL held-out test set...", flush=True)
    X_test, y_test = load_test()

    print("Running M8 ensemble soft-voting inference on the test set...", flush=True)
    probs = ensemble_predict_proba_flat(X_test, models_, scalers_)
    y_pred = probs.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0,
        labels=list(range(config.N_CLASSES)),
        target_names=[config.CLASS_NAMES[c] for c in range(config.N_CLASSES)],
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(config.N_CLASSES)))

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
    table2.to_csv(config.RESULTS_DIR / "table2_mlp_baseline_per_class_metrics.csv", index=False)

    summary = {
        "accuracy_pct": round(acc * 100, 2),
        "macro_precision_pct": round(report["macro avg"]["precision"] * 100, 2),
        "macro_recall_pct": round(report["macro avg"]["recall"] * 100, 2),
        "macro_f1_pct": round(report["macro avg"]["f1-score"] * 100, 2),
        "weighted_precision_pct": round(report["weighted avg"]["precision"] * 100, 2),
        "weighted_recall_pct": round(report["weighted avg"]["recall"] * 100, 2),
        "weighted_f1_pct": round(report["weighted avg"]["f1-score"] * 100, 2),
        "n_test_samples": int(len(y_test)),
        "n_folds_in_ensemble": n_folds,
    }
    with open(config.RESULTS_DIR / "summary_mlp_baseline_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.save(config.RESULTS_DIR / "confusion_matrix_mlp_baseline.npy", cm)

    print("\n=== M8 MLP baseline -- REAL ensemble test-set summary ===")
    print(json.dumps(summary, indent=2))
    return summary, table2


if __name__ == "__main__":
    run_evaluation_mlp()
