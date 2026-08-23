"""Stratified K-Fold ensemble training with zero-leakage scaling + resampling.

For each fold:
  1. split into train/val indices (StratifiedKFold on the full training CSV)
  2. fit StandardScaler on the TRAIN indices only, transform train+val
  3. hybrid_resample (ADASYN + capped undersampling) on the TRAIN indices only
  4. compute sqrt-scaled focal-loss class weights from the *resampled* train set
  5. train the CNN-BiGRU-Attention model with Weighted Focal Loss, AdamW +
     cosine-annealing-with-warm-restarts, early stopping on val_loss
  6. save the fold's model + scaler, log before/after class distribution and
     per-fold validation metrics
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
from .model import build_model, make_weighted_focal_loss, make_optimizer, sqrt_scaled_class_weights


class EpochTimer(tf.keras.callbacks.Callback):
    def __init__(self, fold: int):
        super().__init__()
        self.fold = fold

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        dt = time.time() - self._t0
        logs = logs or {}
        print(
            f"[fold {self.fold}] epoch {epoch + 1}: "
            f"loss={logs.get('loss'):.4f} val_loss={logs.get('val_loss'):.4f} "
            f"({dt:.1f}s)",
            flush=True,
        )


def to_categorical(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype="float32")
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def run_training(n_folds: int = config.N_FOLDS, resume: bool = False):
    t_start = time.time()
    X, y = load_train()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)

    fold_summaries = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        fold_t0 = time.time()

        # --- resume support: StratifiedKFold is deterministic (fixed
        # random_state), so a fold already saved by a prior (crashed) attempt
        # at the SAME config can be trusted and skipped rather than redone. ---
        model_path = config.MODELS_DIR / f"fold{fold_idx}.keras"
        scaler_path = config.MODELS_DIR / f"scaler_fold{fold_idx}.pkl"
        report_path = config.RESULTS_DIR / f"fold{fold_idx}_val_report.json"
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

        print(f"\n{'=' * 70}\nFOLD {fold_idx}/{n_folds}\n{'=' * 70}", flush=True)

        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_val_raw, y_val = X[val_idx], y[val_idx]

        # --- scaling (fit on train fold only) ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        # --- hybrid resampling (train fold only) ---
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

        dist_path = config.RESULTS_DIR / f"fold{fold_idx}_class_distribution.json"
        with open(dist_path, "w") as f:
            json.dump(
                {
                    "before": {config.CLASS_NAMES[c]: n for c, n in sorted(before_counts.items())},
                    "after": {config.CLASS_NAMES[c]: n for c, n in sorted(after_counts.items())},
                },
                f,
                indent=2,
            )

        # --- prepare tensors ---
        X_train_final = X_train_res.reshape(-1, config.N_FEATURES, 1).astype("float32")
        X_val_final = X_val_scaled.reshape(-1, config.N_FEATURES, 1).astype("float32")
        y_train_cat = to_categorical(y_train_res, config.N_CLASSES)
        y_val_cat = to_categorical(y_val, config.N_CLASSES)

        # --- class weights for focal loss (from resampled train distribution) ---
        alpha = sqrt_scaled_class_weights(after_counts)

        # --- build & compile model ---
        steps_per_epoch = max(1, len(y_train_res) // config.BATCH_SIZE)
        model = build_model()
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
                str(config.LOGS_DIR / f"fold{fold_idx}_history.csv")
            ),
            EpochTimer(fold_idx),
        ]

        print(f"[fold {fold_idx}] training on {X_train_final.shape[0]} rows...", flush=True)
        model.fit(
            X_train_final,
            y_train_cat,
            validation_data=(X_val_final, y_val_cat),
            epochs=config.MAX_EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=0,
        )

        # --- fold validation metrics ---
        val_probs = model.predict(X_val_final, batch_size=config.BATCH_SIZE, verbose=0)
        val_pred = val_probs.argmax(axis=1)
        acc = accuracy_score(y_val, val_pred)
        report = classification_report(
            y_val, val_pred, output_dict=True, zero_division=0,
            labels=list(range(config.N_CLASSES)),
            target_names=[config.CLASS_NAMES[c] for c in range(config.N_CLASSES)],
        )
        print(f"[fold {fold_idx}] validation accuracy: {acc:.4f}", flush=True)

        # --- save model + scaler ---
        model.save(config.MODELS_DIR / f"fold{fold_idx}.keras")
        with open(config.MODELS_DIR / f"scaler_fold{fold_idx}.pkl", "wb") as f:
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
        with open(config.RESULTS_DIR / f"fold{fold_idx}_val_report.json", "w") as f:
            json.dump(report, f, indent=2)

        print(
            f"[fold {fold_idx}] done in {fold_summary['elapsed_sec'] / 60:.1f} min. "
            f"macro-recall={fold_summary['val_macro_recall']:.4f}",
            flush=True,
        )

    with open(config.RESULTS_DIR / "fold_summary.json", "w") as f:
        json.dump(fold_summaries, f, indent=2)

    print(f"\nAll folds complete in {(time.time() - t_start) / 3600:.2f} hours.", flush=True)
    return fold_summaries


if __name__ == "__main__":
    run_training()
