"""Per-class ensemble evaluation of M9 (MLP baseline WITHOUT temporal
features) on the REAL held-out test set -- mirrors evaluate_mlp_baseline.py,
but slices to the first 28 columns (temporal features are always appended
last in data_pipeline.py) and loads the mlp_no_temporal_fold{1..5} models.
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

N_FEATURES_NO_TEMPORAL = 28
MODEL_PREFIX = "mlp_no_temporal_fold"


def _load_artifacts(n_folds: int = config.N_FOLDS):
    models_, scalers_ = [], []
    for i in range(1, n_folds + 1):
        model_path = config.MODELS_DIR / f"{MODEL_PREFIX}{i}.keras"
        scaler_path = config.MODELS_DIR / f"scaler_{MODEL_PREFIX}{i}.pkl"
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


def run_evaluation(n_folds: int = config.N_FOLDS):
    print("Loading M9 MLP (no-temporal) fold models + scalers...", flush=True)
    models_, scalers_ = _load_artifacts(n_folds)

    print("Loading REAL held-out test set...", flush=True)
    X_test_32, y_test = load_test()
    X_test = X_test_32[:, :N_FEATURES_NO_TEMPORAL]

    print("Running M9 ensemble soft-voting inference on the test set...", flush=True)
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
            "class_id": c, "class_name": name,
            "precision_pct": round(r["precision"] * 100, 1),
            "recall_pct": round(r["recall"] * 100, 1),
            "f1_pct": round(r["f1-score"] * 100, 1),
            "support": int(r["support"]),
        })
    table2 = pd.DataFrame(rows)
    table2.to_csv(config.RESULTS_DIR / "table2_mlp_no_temporal_per_class_metrics.csv", index=False)

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
    with open(config.RESULTS_DIR / "summary_mlp_no_temporal_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    np.save(config.RESULTS_DIR / "confusion_matrix_mlp_no_temporal.npy", cm)

    print("\n=== M9 MLP (no-temporal) -- REAL ensemble test-set summary ===")
    print(json.dumps(summary, indent=2))
    return summary, table2


if __name__ == "__main__":
    run_evaluation()
