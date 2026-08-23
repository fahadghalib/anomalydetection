"""Reduced-scale ablation study (Table Y from the manuscript, which the
paper references in Section 4.5 but never actually shows).

The paper's own description of M1-M6 is inconsistent (M2 and M3 both
described as "attention"-related with contradictory effects; M5/M6 never
described at all). This module defines an unambiguous, defensible set of 7
configurations, each isolating exactly one component of the final pipeline:

  M1  no BiGRU              (multi-scale CNN -> Attention -> dense)
  M2  no Attention          (multi-scale CNN -> BiGRU -> GlobalAvgPool -> dense)
  M3  no multi-scale CNN    (single K=3 branch -> BiGRU -> Attention -> dense)
  M4  no Weighted Focal Loss (plain categorical cross-entropy, no class weights)
  M5  no ADASYN             (train on the natural imbalanced fold, no resampling)
  M6  no temporal features  (the original 28 columns only, not the 32 with
                              dt_prev/pos_delta/spd_delta/msg_rate_5s)
  M7  full proposed model   (real 5-fold numbers already exist in
                              outputs/results/fold_summary.json -- not rerun)

Scope (user-selected "fast" tier): M1-M6 each trained on FOLD 1 ONLY (the
same split used as fold_idx=1 in the main run), for a reduced epoch budget,
to isolate each component's directional effect quickly. These are single-run
numbers, not mean+-std over 5 folds -- only M7 has genuine 5-fold statistics.
"""

from __future__ import annotations

import json
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from . import config
from .data_pipeline import load_train
from .sampling import hybrid_resample
from .model import AdditiveAttention, make_optimizer, sqrt_scaled_class_weights, make_weighted_focal_loss

ABLATION_EPOCHS = 30
ABLATION_PATIENCE = 6


def build_variant(variant: str, n_features: int, n_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=(n_features, 1))

    kernels = (3,) if variant == "M3_no_multiscale" else config.CNN_KERNELS
    branches = []
    for k in kernels:
        b = layers.Conv1D(config.CNN_FILTERS, k, padding="same", activation="relu")(inp)
        b = layers.BatchNormalization()(b)
        branches.append(b)
    x = layers.Concatenate()(branches) if len(branches) > 1 else branches[0]

    if variant != "M1_no_bigru":
        x = layers.Bidirectional(
            layers.GRU(config.GRU_UNITS, return_sequences=True, recurrent_dropout=config.RECURRENT_DROPOUT)
        )(x)
        x = layers.Dropout(config.DROPOUT_RATE)(x)

    if variant == "M2_no_attention":
        context = layers.GlobalAveragePooling1D()(x)
    else:
        context, _ = AdditiveAttention()(x)

    out = layers.Dense(n_classes, activation="softmax", kernel_regularizer=regularizers.l2(config.L2_REG))(context)
    return models.Model(inp, out, name=f"ablation_{variant}")


def to_categorical(y, n_classes):
    out = np.zeros((y.shape[0], n_classes), dtype="float32")
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def run_variant(variant: str, X_train_raw, y_train, X_val_raw, y_val, n_features: int):
    print(f"\n{'='*70}\n{variant}  (n_features={n_features})\n{'='*70}", flush=True)
    t0 = time.time()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    if variant == "M5_no_adasyn":
        X_train_res, y_train_res = X_train_scaled, y_train
        after_counts = dict(zip(*np.unique(y_train, return_counts=True)))
    else:
        X_train_res, y_train_res, _before, after_counts, _fb = hybrid_resample(
            X_train_scaled, y_train, random_state=config.RANDOM_STATE
        )
    print(f"[{variant}] train rows: {len(y_train)} -> {len(y_train_res)}", flush=True)

    X_train_final = X_train_res.reshape(-1, n_features, 1).astype("float32")
    X_val_final = X_val_scaled.reshape(-1, n_features, 1).astype("float32")
    y_train_cat = to_categorical(y_train_res, config.N_CLASSES)
    y_val_cat = to_categorical(y_val, config.N_CLASSES)

    build_variant_name = "M1_no_bigru" if variant == "M1_no_bigru" else (
        "M2_no_attention" if variant == "M2_no_attention" else (
        "M3_no_multiscale" if variant == "M3_no_multiscale" else "full"))
    model = build_variant(build_variant_name, n_features, config.N_CLASSES)

    if variant == "M4_no_focal_loss":
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING)
    else:
        alpha = sqrt_scaled_class_weights(after_counts)
        loss = make_weighted_focal_loss(alpha)

    steps_per_epoch = max(1, len(y_train_res) // config.BATCH_SIZE)
    model.compile(optimizer=make_optimizer(steps_per_epoch), loss=loss, metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=ABLATION_PATIENCE, restore_best_weights=True),
    ]
    model.fit(
        X_train_final, y_train_cat,
        validation_data=(X_val_final, y_val_cat),
        epochs=ABLATION_EPOCHS, batch_size=config.BATCH_SIZE,
        callbacks=callbacks, verbose=0,
    )

    val_probs = model.predict(X_val_final, batch_size=config.BATCH_SIZE, verbose=0)
    val_pred = val_probs.argmax(axis=1)
    acc = accuracy_score(y_val, val_pred)
    report = classification_report(
        y_val, val_pred, output_dict=True, zero_division=0,
        labels=list(range(config.N_CLASSES)),
    )
    elapsed = time.time() - t0
    result = {
        "variant": variant,
        "val_accuracy": float(acc),
        "val_macro_recall": float(report["macro avg"]["recall"]),
        "val_macro_f1": float(report["macro avg"]["f1-score"]),
        "elapsed_sec": elapsed,
    }
    print(f"[{variant}] acc={acc:.4f} macro_recall={report['macro avg']['recall']:.4f} "
          f"macro_f1={report['macro avg']['f1-score']:.4f} ({elapsed/60:.1f} min)", flush=True)
    return result


if __name__ == "__main__":
    X32, y = load_train()  # 32 features (28 base + 4 temporal)
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    train_idx, val_idx = next(iter(skf.split(X32, y)))  # fold 1 split only

    X_train_32, y_train = X32[train_idx], y[train_idx]
    X_val_32, y_val = X32[val_idx], y[val_idx]
    # M6 uses only the original 28 columns (temporal features are the LAST 4, appended in data_pipeline.py)
    X_train_28, X_val_28 = X_train_32[:, :28], X_val_32[:, :28]

    variants = ["M1_no_bigru", "M2_no_attention", "M3_no_multiscale",
                "M4_no_focal_loss", "M5_no_adasyn", "M6_no_temporal_features"]

    results = []
    out_path = config.RESULTS_DIR / "ablation_table_y.json"
    for v in variants:
        if v == "M6_no_temporal_features":
            r = run_variant(v, X_train_28, y_train, X_val_28, y_val, n_features=28)
        else:
            r = run_variant(v, X_train_32, y_train, X_val_32, y_val, n_features=32)
        results.append(r)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[ablation] saved progress ({len(results)}/{len(variants)}) to {out_path}", flush=True)

    print("\nAblation study complete.")
    print(json.dumps(results, indent=2))
