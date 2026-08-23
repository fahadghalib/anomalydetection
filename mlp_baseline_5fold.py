"""Full 5-fold, full-epoch-budget run of the M8 strong-MLP baseline, on GPU.

The reduced fold-1-only, 30-epoch version (src/mlp_baseline.py) found the MLP
baseline (92.63% val accuracy) outperforming the full CNN-BiGRU-Attention
model's OWN fold-1 result (91.34%) and its 5-fold mean (91.71%). That
comparison used mismatched protocols (M8: 1 fold / 30 epochs vs M7: 5 folds /
100 epochs), so before trusting the finding this reruns M8 under the EXACT
same protocol as M7 (train.py): all 5 StratifiedKFold folds, MAX_EPOCHS=100,
EARLY_STOPPING_PATIENCE=20, same optimizer/loss/resampling -- only the
architecture differs (flat MLP vs multi-scale-CNN -> BiGRU -> attention).

Runs on GPU (no CUDA_VISIBLE_DEVICES override) since the M1-M6 ablation job
that previously occupied it has finished.
"""

from __future__ import annotations

import json
import pickle
import time

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from . import config
from .data_pipeline import load_train
from .sampling import hybrid_resample
from .model import make_weighted_focal_loss, make_optimizer, sqrt_scaled_class_weights
from .mlp_baseline import build_mlp_baseline
from .train import to_categorical, EpochTimer

MODEL_PREFIX = "mlp_baseline_fold"
RESULT_PREFIX = "mlp_baseline_fold"


def run_mlp_5fold(n_folds: int = config.N_FOLDS, resume: bool = False):
    t_start = time.time()
    X, y = load_train()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)

    fold_summaries = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        fold_t0 = time.time()

        model_path = config.MODELS_DIR / f"{MODEL_PREFIX}{fold_idx}.keras"
        scaler_path = config.MODELS_DIR / f"scaler_{MODEL_PREFIX}{fold_idx}.pkl"
        report_path = config.RESULTS_DIR / f"{RESULT_PREFIX}{fold_idx}_val_report.json"
        if resume and model_path.exists() and scaler_path.exists() and report_path.exists():
            with open(report_path) as f:
                report = json.load(f)
            fold_summaries.append({
                "fold": fold_idx,
                "n_train_resampled": None,
                "n_val": int(len(val_idx)),
                "val_accuracy": report.get("accuracy"),
                "val_macro_precision": report["macro avg"]["precision"],
                "val_macro_recall": report["macro avg"]["recall"],
                "val_macro_f1": report["macro avg"]["f1-score"],
                "elapsed_sec": 0.0,
            })
            print(
                f"\n[fold {fold_idx}] already completed in a prior attempt "
                f"(resume=True) -- skipping, val_accuracy="
                f"{fold_summaries[-1]['val_accuracy']}",
                flush=True,
            )
            continue

        print(f"\n{'=' * 70}\nM8_mlp_baseline FOLD {fold_idx}/{n_folds}\n{'=' * 70}", flush=True)

        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_val_raw, y_val = X[val_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        print(f"[fold {fold_idx}] resampling training fold ({len(y_train)} rows)...", flush=True)
        t0 = time.time()
        X_train_res, y_train_res, before_counts, after_counts, fallback_classes = hybrid_resample(
            X_train_scaled, y_train, random_state=config.RANDOM_STATE + fold_idx
        )
        print(
            f"[fold {fold_idx}] resampled {len(y_train)} -> {len(y_train_res)} rows "
            f"in {time.time() - t0:.1f}s"
            + (f" (ADASYN fallback to RandomOverSampler for classes {fallback_classes})"
               if fallback_classes else ""),
            flush=True,
        )

        y_train_cat = to_categorical(y_train_res, config.N_CLASSES)
        y_val_cat = to_categorical(y_val, config.N_CLASSES)

        alpha = sqrt_scaled_class_weights(after_counts)

        steps_per_epoch = max(1, len(y_train_res) // config.BATCH_SIZE)
        model = build_mlp_baseline(config.N_FEATURES, config.N_CLASSES)
        model.compile(
            optimizer=make_optimizer(steps_per_epoch),
            loss=make_weighted_focal_loss(alpha),
            metrics=["accuracy"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=config.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.CSVLogger(
                str(config.LOGS_DIR / f"{MODEL_PREFIX}{fold_idx}_history.csv")
            ),
            EpochTimer(fold_idx),
        ]

        print(f"[fold {fold_idx}] training on {X_train_res.shape[0]} rows...", flush=True)
        model.fit(
            X_train_res,
            y_train_cat,
            validation_data=(X_val_scaled, y_val_cat),
            epochs=config.MAX_EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=0,
        )

        val_probs = model.predict(X_val_scaled, batch_size=config.BATCH_SIZE, verbose=0)
        val_pred = val_probs.argmax(axis=1)
        acc = accuracy_score(y_val, val_pred)
        report = classification_report(
            y_val, val_pred, output_dict=True, zero_division=0,
            labels=list(range(config.N_CLASSES)),
            target_names=[config.CLASS_NAMES[c] for c in range(config.N_CLASSES)],
        )
        print(f"[fold {fold_idx}] validation accuracy: {acc:.4f}", flush=True)

        model.save(model_path)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        fold_summary = {
            "fold": fold_idx,
            "n_train_resampled": int(len(y_train_res)),
            "n_val": int(len(y_val)),
            "val_accuracy": float(acc),
            "val_macro_precision": float(report["macro avg"]["precision"]),
            "val_macro_recall": float(report["macro avg"]["recall"]),
            "val_macro_f1": float(report["macro avg"]["f1-score"]),
            "elapsed_sec": time.time() - fold_t0,
        }
        fold_summaries.append(fold_summary)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(
            f"[fold {fold_idx}] done in {fold_summary['elapsed_sec'] / 60:.1f} min. "
            f"macro-recall={fold_summary['val_macro_recall']:.4f}",
            flush=True,
        )

    with open(config.RESULTS_DIR / "mlp_baseline_5fold_summary.json", "w") as f:
        json.dump(fold_summaries, f, indent=2)

    accs = np.array([fs["val_accuracy"] for fs in fold_summaries])
    f1s = np.array([fs["val_macro_f1"] for fs in fold_summaries])
    recs = np.array([fs["val_macro_recall"] for fs in fold_summaries])
    print(
        f"\nM8 5-fold complete in {(time.time() - t_start) / 3600:.2f} hours. "
        f"accuracy={accs.mean()*100:.2f}%+-{accs.std(ddof=1)*100:.2f}% "
        f"macro_f1={f1s.mean()*100:.2f}%+-{f1s.std(ddof=1)*100:.2f}% "
        f"macro_recall={recs.mean()*100:.2f}%+-{recs.std(ddof=1)*100:.2f}%",
        flush=True,
    )
    return fold_summaries


if __name__ == "__main__":
    import sys
    resume = "--resume" in sys.argv
    run_mlp_5fold(resume=resume)
