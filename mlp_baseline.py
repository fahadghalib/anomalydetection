"""Strong MLP baseline (M8) -- reviewer-requested comparison.

Reviewer comment: "It must be proven that using BiGRU on tabular features
arranged sequentially is actually useful, by comparing it against a strong
MLP model." M1 (no_bigru, in ablation.py) already removes BiGRU but keeps
the multi-scale-CNN + Attention scaffolding around it, so it doesn't fully
answer this: it shows BiGRU matters *within this architecture*, not whether
treating the features as a sequence at all buys anything over a flat,
well-regularized feedforward network.

This module trains a deliberately generous MLP (512-256-128, ~186K trainable
params -- more than the full CNN-BiGRU-Attention model's ~141K) directly on
the flat 32-feature vector, with every other part of the protocol held
identical to the other ablation variants: fold 1 split (same random_state,
so same rows), ADASYN-resampled training fold, weighted focal loss with
sqrt-scaled class weights, same optimizer/schedule, same epoch/patience
budget. The only variable is architecture, so any accuracy gap is
attributable to the sequence-modeling (multi-scale CNN -> BiGRU ->
attention) itself, not to loss/sampling/capacity differences.

Runs on CPU (CUDA_VISIBLE_DEVICES="") on purpose: the GPU is occupied by the
M1-M6 ablation run (src/ablation.py) and this MLP is cheap enough (no
convolution, no recurrence) to train on CPU in parallel without contention.
"""

from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # force CPU -- GPU is busy with M1-M6

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
from .model import make_optimizer, sqrt_scaled_class_weights, make_weighted_focal_loss
from .ablation import to_categorical, ABLATION_EPOCHS, ABLATION_PATIENCE


def build_mlp_baseline(n_features: int, n_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=(n_features,), name="features")
    x = layers.Dense(512, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    out = layers.Dense(
        n_classes, activation="softmax",
        kernel_regularizer=regularizers.l2(config.L2_REG),
    )(x)
    return models.Model(inp, out, name="mlp_baseline_M8")


def run_mlp_baseline():
    X32, y = load_train()  # 32 features, cached npz -- fast load
    skf = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    train_idx, val_idx = next(iter(skf.split(X32, y)))  # identical fold-1 split as ablation.py

    X_train_raw, y_train = X32[train_idx], y[train_idx]
    X_val_raw, y_val = X32[val_idx], y[val_idx]

    print(f"\n{'='*70}\nM8_mlp_baseline  (n_features=32, CPU, strong MLP)\n{'='*70}", flush=True)
    t0 = time.time()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled = scaler.transform(X_val_raw)

    X_train_res, y_train_res, _before, after_counts, _fb = hybrid_resample(
        X_train_scaled, y_train, random_state=config.RANDOM_STATE
    )
    print(f"[M8_mlp_baseline] train rows: {len(y_train)} -> {len(y_train_res)}", flush=True)

    y_train_cat = to_categorical(y_train_res, config.N_CLASSES)
    y_val_cat = to_categorical(y_val, config.N_CLASSES)

    model = build_mlp_baseline(config.N_FEATURES, config.N_CLASSES)
    model.summary()
    n_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    print(f"[M8_mlp_baseline] trainable params: {n_params:,}", flush=True)

    alpha = sqrt_scaled_class_weights(after_counts)
    loss = make_weighted_focal_loss(alpha)

    steps_per_epoch = max(1, len(y_train_res) // config.BATCH_SIZE)
    model.compile(optimizer=make_optimizer(steps_per_epoch), loss=loss, metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=ABLATION_PATIENCE, restore_best_weights=True),
    ]
    model.fit(
        X_train_res, y_train_cat,
        validation_data=(X_val_scaled, y_val_cat),
        epochs=ABLATION_EPOCHS, batch_size=config.BATCH_SIZE,
        callbacks=callbacks, verbose=2,
    )

    val_probs = model.predict(X_val_scaled, batch_size=config.BATCH_SIZE, verbose=0)
    val_pred = val_probs.argmax(axis=1)
    acc = accuracy_score(y_val, val_pred)
    report = classification_report(
        y_val, val_pred, output_dict=True, zero_division=0,
        labels=list(range(config.N_CLASSES)),
    )
    elapsed = time.time() - t0
    result = {
        "variant": "M8_mlp_baseline",
        "val_accuracy": float(acc),
        "val_macro_recall": float(report["macro avg"]["recall"]),
        "val_macro_f1": float(report["macro avg"]["f1-score"]),
        "trainable_params": n_params,
        "elapsed_sec": elapsed,
    }
    print(f"[M8_mlp_baseline] acc={acc:.4f} macro_recall={report['macro avg']['recall']:.4f} "
          f"macro_f1={report['macro avg']['f1-score']:.4f} ({elapsed/60:.1f} min)", flush=True)

    out_path = config.RESULTS_DIR / "mlp_baseline_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[M8_mlp_baseline] saved to {out_path}", flush=True)
    return result


if __name__ == "__main__":
    run_mlp_baseline()
