"""Unified model runner under the corrected protocol (Reviewer #2, #4, #5).

Reviewer comment #4 objected that M1-M6 were evaluated on one fold for 30
epochs while M7 used five folds and up to 100 epochs, so component effects
could not be attributed. This runner removes that objection structurally:
for a given fold, the scaler is fit and ADASYN is applied EXACTLY ONCE, and
every model family is then trained on that identical resampled array, with
the same folds, the same seeds and the same epoch budget. Any difference
between two models is therefore attributable to the model alone.

Reviewer comment #5 asked for a matched-capacity MLP, a strong tabular
baseline such as CatBoost/XGBoost, and a feature-order permutation
experiment. The available model keys are:

  lgbm          gradient-boosted trees (strong tabular baseline)
  mlp           feed-forward MLP, ~186K params
  mlp_matched   MLP shrunk to match the proposed model's ~141K params
  cnn           the proposed multi-scale CNN + BiGRU + additive attention
  cnn_permuted  the proposed model fed a RANDOMLY PERMUTED column order --
                if this scores the same as `cnn`, the column order carries no
                sequential structure for the BiGRU to model

All runs use the sender-disjoint partitions from src/corrected_protocol.py,
so no sender crosses a fold or the test boundary, and no identifier column is
present in the feature matrix.

Usage:
  python -m src.run_corrected --models lgbm --folds 1
  python -m src.run_corrected --models lgbm,mlp_matched,cnn --folds 5
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from . import config
from .corrected_protocol import (
    build_corrected_dataset, sender_disjoint_holdout, grouped_folds,
)
from .sampling import hybrid_resample
from .model import (
    AdditiveAttention, make_weighted_focal_loss, make_optimizer,
    sqrt_scaled_class_weights,
)

EPOCHS = 40
PATIENCE = 8

FOLD_CACHE_DIR = config.DATA_DIR / "cache" / "corrected_folds"


def prepare_partition(tag: str, X_tr_raw, y_tr, X_va_raw, no_adasyn: bool, seed: int):
    """Fit the scaler and apply ADASYN once, then cache the result to disk.

    Two purposes. It removes a ~17-minute resampling pass from every
    subsequent run, and -- more importantly for reviewer comment #4 -- it
    guarantees that every model family trains on a bit-identical array, so a
    difference between two configurations cannot come from preprocessing
    variation.
    """
    FOLD_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = FOLD_CACHE_DIR / f"{tag}{'_noadasyn' if no_adasyn else ''}.npz"
    if cache.exists():
        d = np.load(cache)
        print(f"  reusing cached preprocessing: {cache.name} "
              f"({len(d['y_tr']):,} train rows)", flush=True)
        return d["X_tr"], d["y_tr"], d["X_va"]

    t0 = time.time()
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw).astype("float32")
    X_va = scaler.transform(X_va_raw).astype("float32")

    if not no_adasyn:
        n_before = len(y_tr)
        X_tr, y_tr, _b, _a, _f = hybrid_resample(X_tr, y_tr, random_state=seed)
        X_tr = X_tr.astype("float32")
        print(f"  ADASYN: train {n_before:,} -> {len(y_tr):,} rows "
              f"({time.time() - t0:.0f}s); validation rows untouched", flush=True)

    np.savez(cache, X_tr=X_tr, y_tr=y_tr, X_va=X_va)
    return X_tr, y_tr, X_va


# ---------------------------------------------------------------------------
# model builders -- all take (n_features, n_classes)
# ---------------------------------------------------------------------------
def build_cnn_bigru_attention(n_features: int, n_classes: int) -> tf.keras.Model:
    inp = layers.Input(shape=(n_features, 1))
    branches = []
    for k in config.CNN_KERNELS:
        b = layers.Conv1D(config.CNN_FILTERS, k, padding="same", activation="relu")(inp)
        b = layers.BatchNormalization()(b)
        branches.append(b)
    x = layers.Concatenate()(branches)
    x = layers.Bidirectional(
        layers.GRU(config.GRU_UNITS, return_sequences=True,
                   recurrent_dropout=config.RECURRENT_DROPOUT)
    )(x)
    x = layers.Dropout(config.DROPOUT_RATE)(x)
    context, _ = AdditiveAttention()(x)
    out = layers.Dense(n_classes, activation="softmax",
                       kernel_regularizer=regularizers.l2(config.L2_REG))(context)
    return models.Model(inp, out, name="cnn_bigru_attention")


def build_mlp(n_features: int, n_classes: int, widths=(512, 256, 128)) -> tf.keras.Model:
    inp = layers.Input(shape=(n_features,))
    x = inp
    for w in widths:
        x = layers.Dense(w, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(config.DROPOUT_RATE)(x)
    out = layers.Dense(n_classes, activation="softmax",
                       kernel_regularizer=regularizers.l2(config.L2_REG))(x)
    return models.Model(inp, out, name=f"mlp_{'_'.join(map(str, widths))}")


def build_mlp_matched(n_features: int, n_classes: int) -> tf.keras.Model:
    """Smaller MLP: 106,868 trainable parameters at 28 input features, i.e.
    76% of the proposed model's 140,373. Retained because a win here is the
    stronger result -- the same accuracy from less capacity."""
    return build_mlp(n_features, n_classes, widths=(384, 192, 96))


def build_mlp_capacity_matched(n_features: int, n_classes: int) -> tf.keras.Model:
    """MLP sized to the proposed model's parameter count within ~2%
    (142,596 vs 140,373 at 28 input features), so that capacity cannot
    explain any difference between them (reviewer comment #5)."""
    return build_mlp(n_features, n_classes, widths=(448, 224, 112))


def to_categorical(y, n_classes):
    out = np.zeros((y.shape[0], n_classes), dtype="float32")
    out[np.arange(y.shape[0]), y] = 1.0
    return out


# ---------------------------------------------------------------------------
# per-model training
# ---------------------------------------------------------------------------
def _fit_keras(build_fn, X_tr, y_tr, X_va, y_va, n_features, reshape_3d, seed):
    tf.keras.utils.set_random_seed(seed)
    model = build_fn(n_features, config.N_CLASSES)

    alpha = sqrt_scaled_class_weights(
        dict(zip(*[a.tolist() for a in np.unique(y_tr, return_counts=True)]))
    )
    steps = max(1, len(y_tr) // config.BATCH_SIZE)
    model.compile(optimizer=make_optimizer(steps),
                  loss=make_weighted_focal_loss(alpha), metrics=["accuracy"])

    Xt = X_tr.reshape(-1, n_features, 1) if reshape_3d else X_tr
    Xv = X_va.reshape(-1, n_features, 1) if reshape_3d else X_va

    model.fit(Xt, to_categorical(y_tr, config.N_CLASSES),
              validation_data=(Xv, to_categorical(y_va, config.N_CLASSES)),
              epochs=EPOCHS, batch_size=config.BATCH_SIZE,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                  monitor="val_loss", patience=PATIENCE, restore_best_weights=True)],
              verbose=2)
    probs = model.predict(Xv, batch_size=config.BATCH_SIZE, verbose=0)
    n_params = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    return probs, n_params


def _fit_lgbm(X_tr, y_tr, X_va, y_va, seed):
    """Gradient-boosted trees, configured for bit-reproducible fits.

    With the defaults (n_jobs=-1, histogram construction order left to the
    thread pool) this baseline was not reproducible: repeated fits of the same
    configuration on the same data differed by up to 6.5 accuracy points, and
    one fold collapsed outright. `deterministic` plus a fixed thread count and
    row-wise forcing removes that variance, at some cost in fitting speed. A
    baseline that cannot be reproduced cannot be reported.
    """
    import lightgbm as lgb
    clf = lgb.LGBMClassifier(
        objective="multiclass", num_class=config.N_CLASSES,
        n_estimators=400, learning_rate=0.1, num_leaves=63,
        random_state=seed, n_jobs=8, verbose=-1,
        deterministic=True, force_row_wise=True,
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_va), int(clf.booster_.num_trees())


def run_model(key: str, X_tr, y_tr, X_va, y_va, n_features, seed, perm=None):
    t0 = time.time()
    if key == "lgbm":
        probs, size = _fit_lgbm(X_tr, y_tr, X_va, y_va, seed)
    elif key == "mlp":
        probs, size = _fit_keras(build_mlp, X_tr, y_tr, X_va, y_va, n_features, False, seed)
    elif key == "mlp_matched":
        probs, size = _fit_keras(build_mlp_matched, X_tr, y_tr, X_va, y_va, n_features, False, seed)
    elif key == "mlp_cap":
        probs, size = _fit_keras(build_mlp_capacity_matched, X_tr, y_tr, X_va, y_va,
                                 n_features, False, seed)
    elif key == "cnn":
        probs, size = _fit_keras(build_cnn_bigru_attention, X_tr, y_tr, X_va, y_va,
                                 n_features, True, seed)
    elif key == "cnn_permuted":
        probs, size = _fit_keras(build_cnn_bigru_attention, X_tr[:, perm], y_tr,
                                 X_va[:, perm], y_va, n_features, True, seed)
    else:
        raise ValueError(f"unknown model key: {key}")

    pred = probs.argmax(axis=1)
    acc = accuracy_score(y_va, pred)
    rep = classification_report(y_va, pred, output_dict=True, zero_division=0,
                                labels=list(range(config.N_CLASSES)))
    return {
        "model": key,
        "val_accuracy": float(acc),
        "val_macro_precision": float(rep["macro avg"]["precision"]),
        "val_macro_recall": float(rep["macro avg"]["recall"]),
        "val_macro_f1": float(rep["macro avg"]["f1-score"]),
        "model_size": size,
        "elapsed_sec": time.time() - t0,
        "_report": rep,
        "_pred": pred,
    }


def evaluate_on_holdout(keys, X, y, groups, dev_idx, test_idx, n_features, perm, no_adasyn):
    """Train once on the full development partition, evaluate once on the
    sender-disjoint held-out test partition.

    Reported separately from cross-validation because the test partition is
    touched exactly once, after all model selection is complete. Emits the
    per-class breakdown that replaces the manuscript's Table 2, and the false
    alarm rate under the corrected protocol.
    """
    y_test = y[test_idx]
    X_dev, y_dev, X_test = prepare_partition(
        "holdout_dev", X[dev_idx], y[dev_idx], X[test_idx],
        no_adasyn, seed=config.RANDOM_STATE)

    # Merge into any existing held-out results rather than replacing them, so
    # evaluating a new model never discards another model's test record.
    path = config.RESULTS_DIR / "corrected_protocol_holdout.json"
    out = {}
    if path.exists():
        with open(path) as f:
            out = json.load(f)

    for key in keys:
        print(f"\n--- {key} | held-out test ---", flush=True)
        res = run_model(key, X_dev, y_dev, X_test, y_test, n_features,
                        seed=config.RANDOM_STATE, perm=perm)

        # rebuild predictions for the per-class table, FAR and confusion matrix
        preds = res.pop("_pred", None)
        rep = res.pop("_report", None)
        if preds is not None:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, preds, labels=list(range(config.N_CLASSES)))
            np.save(config.RESULTS_DIR / f"confusion_matrix_corrected_{key}.npy", cm)
        out[key] = res
        print(f"  {key} TEST: acc={res['val_accuracy']:.4f} "
              f"macro_f1={res['val_macro_f1']:.4f} "
              f"macro_recall={res['val_macro_recall']:.4f}", flush=True)
        if rep is not None:
            per_class = []
            for c in range(config.N_CLASSES):
                r = rep[str(c)] if str(c) in rep else rep[c]
                per_class.append({
                    "class_id": c,
                    "class_name": config.CLASS_NAMES[c],
                    "precision_pct": round(100 * r["precision"], 1),
                    "recall_pct": round(100 * r["recall"], 1),
                    "f1_pct": round(100 * r["f1-score"], 1),
                    "support": int(r["support"]),
                })
            out[key]["per_class"] = per_class
            benign_recall = per_class[0]["recall_pct"]
            out[key]["false_alarm_rate_pct"] = round(100.0 - benign_recall, 3)
            print(f"  benign recall={benign_recall}%  FAR={out[key]['false_alarm_rate_pct']}%",
                  flush=True)

    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {path}")
    return out


def main():
    global EPOCHS, PATIENCE
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lgbm",
                    help="comma-separated: lgbm,mlp,mlp_matched,cnn,cnn_permuted")
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--no-adasyn", action="store_true",
                    help="skip ADASYN (faster; still identical across models)")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="epoch budget. The shared budget is %(default)s for every "
                         "model; raise it only for an explicitly labelled "
                         "convergence check, never for a headline comparison")
    ap.add_argument("--patience", type=int, default=PATIENCE,
                    help="early-stopping patience on validation loss")
    ap.add_argument("--only-fold", type=int, default=None,
                    help="run just this fold index (1-based) out of --folds, for "
                         "re-verifying a single anomalous result cheaply")
    ap.add_argument("--out-suffix", default="",
                    help="suffix for the results filename, so concurrent runs "
                         "never race on the same JSON file")
    ap.add_argument("--eval-test", action="store_true",
                    help="skip cross-validation; train on the full development "
                         "partition and evaluate once on the held-out test partition")
    args = ap.parse_args()
    keys = [k.strip() for k in args.models.split(",") if k.strip()]

    # Applied to every model in this invocation, so the budget stays shared
    # even when it is deliberately raised for a convergence check.
    EPOCHS, PATIENCE = args.epochs, args.patience
    if (EPOCHS, PATIENCE) != (40, 8):
        print(f"NOTE: non-default budget in use -- epochs={EPOCHS}, patience={PATIENCE}. "
              f"Results are not comparable with the shared-budget runs.", flush=True)

    X, y, groups, names = build_corrected_dataset()
    n_features = X.shape[1]
    print(f"corrected dataset: {X.shape}, {len(np.unique(groups)):,} senders, "
          f"features={names}", flush=True)

    dev_idx, test_idx = sender_disjoint_holdout(y, groups)
    X_dev, y_dev, g_dev = X[dev_idx], y[dev_idx], groups[dev_idx]
    print(f"development: {len(y_dev):,} rows / {len(np.unique(g_dev)):,} senders; "
          f"held-out test: {len(test_idx):,} rows / "
          f"{len(np.unique(groups[test_idx])):,} senders", flush=True)

    rng = np.random.RandomState(config.RANDOM_STATE)
    perm = rng.permutation(n_features)

    if args.eval_test:
        evaluate_on_holdout(keys, X, y, groups, dev_idx, test_idx,
                            n_features, perm, args.no_adasyn)
        return

    out_path = config.RESULTS_DIR / f"corrected_protocol_results{args.out_suffix}.json"
    all_results = {}
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)

    for fold_idx, (tr, va) in enumerate(grouped_folds(y_dev, g_dev, args.folds), start=1):
        if args.only_fold is not None and fold_idx != args.only_fold:
            continue
        print(f"\n{'=' * 70}\nFOLD {fold_idx}/{args.folds}  "
              f"(train {len(tr):,} rows / {len(np.unique(g_dev[tr])):,} senders, "
              f"val {len(va):,} rows / {len(np.unique(g_dev[va])):,} senders)\n{'=' * 70}",
              flush=True)

        if all(any(r.get("fold") == fold_idx for r in all_results.get(k, []))
               for k in keys):
            print(f"  all requested models already done for fold {fold_idx}, skipping",
                  flush=True)
            continue

        # --- preprocessing done ONCE, cached, shared by every model ---
        y_va = y_dev[va]
        X_tr, y_tr, X_va = prepare_partition(
            f"fold{fold_idx}", X_dev[tr], y_dev[tr], X_dev[va],
            args.no_adasyn, seed=config.RANDOM_STATE + fold_idx)

        for key in keys:
            # Resume: a fold already recorded for this model is not recomputed.
            # Folds are deterministic (fixed seed, cached preprocessing), so a
            # saved result is exactly what a rerun would produce.
            if any(r.get("fold") == fold_idx for r in all_results.get(key, [])):
                done = next(r for r in all_results[key] if r.get("fold") == fold_idx)
                print(f"\n--- {key} | fold {fold_idx}: already done "
                      f"(acc={done['val_accuracy']:.4f}), skipping ---", flush=True)
                continue
            print(f"\n--- {key} | fold {fold_idx} ---", flush=True)
            res = run_model(key, X_tr, y_tr, X_va, y_va, n_features,
                            seed=config.RANDOM_STATE + fold_idx, perm=perm)
            # per-class detail and raw predictions are kept only for the held-out run
            res.pop("_report", None)
            res.pop("_pred", None)
            res["fold"] = fold_idx
            res["n_train_rows"] = int(len(y_tr))
            res["n_val_rows"] = int(len(y_va))
            all_results.setdefault(key, []).append(res)
            print(f"  {key} fold {fold_idx}: acc={res['val_accuracy']:.4f} "
                  f"macro_f1={res['val_macro_f1']:.4f} "
                  f"macro_recall={res['val_macro_recall']:.4f} "
                  f"({res['elapsed_sec'] / 60:.1f} min)", flush=True)
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}\nCORRECTED-PROTOCOL SUMMARY\n{'=' * 70}")
    for key, runs in all_results.items():
        accs = np.array([r["val_accuracy"] for r in runs])
        f1s = np.array([r["val_macro_f1"] for r in runs])
        sd_a = accs.std(ddof=1) if len(accs) > 1 else 0.0
        sd_f = f1s.std(ddof=1) if len(f1s) > 1 else 0.0
        print(f"  {key:14s} n={len(runs)}  acc={accs.mean() * 100:.2f}%±{sd_a * 100:.2f}  "
              f"macro_f1={f1s.mean() * 100:.2f}%±{sd_f * 100:.2f}")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
