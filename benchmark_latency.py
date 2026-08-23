"""Rigorous single-sample inference latency benchmark for the ensemble
model, addressing the reviewer's request for a fully specified experimental
environment (hardware, framework/version, batch size, numerical precision,
warm-up procedure, repetition count) and a full latency distribution
(mean, median, p95) rather than a single averaged number.

Measures TRUE per-sample (batch_size=1) latency -- the realistic scenario
for a real-time in-vehicle IDS processing one BSM (Basic Safety Message) at
a time -- rather than a large-batch throughput figure divided by sample
count, which understates real deployment latency.
"""

from __future__ import annotations

import json
import pickle
import platform
import time

import numpy as np
import tensorflow as tf

from . import config
from .data_pipeline import load_test
from .model import AdditiveAttention  # noqa: F401 (needed for model deserialization)

N_WARMUP = 50
N_TIMED = 2000


def _load_fold_artifacts(n_folds: int = config.N_FOLDS):
    models_, scalers_ = [], []
    for i in range(1, n_folds + 1):
        model_path = config.MODELS_DIR / f"fold{i}.keras"
        scaler_path = config.MODELS_DIR / f"scaler_fold{i}.pkl"
        models_.append(tf.keras.models.load_model(model_path, compile=False))
        with open(scaler_path, "rb") as f:
            scalers_.append(pickle.load(f))
    return models_, scalers_


def run_benchmark(n_folds: int = config.N_FOLDS):
    gpus = tf.config.list_physical_devices("GPU")
    device_name = "GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB)" if gpus else "CPU only"

    print("Loading fold models + scalers...", flush=True)
    models_, scalers_ = _load_fold_artifacts(n_folds)

    print("Loading REAL held-out test set...", flush=True)
    X_test, _y_test = load_test()

    rng = np.random.RandomState(config.RANDOM_STATE)
    sample_idx = rng.choice(len(X_test), size=N_WARMUP + N_TIMED, replace=False)
    X_sample = X_test[sample_idx]

    # pre-scale each fold's samples once (scaling itself is not part of the
    # "model inference" latency being measured -- only the forward pass is)
    Xs_per_fold = [
        scaler.transform(X_sample).reshape(-1, config.N_FEATURES, 1).astype("float32")
        for scaler in scalers_
    ]

    # --- warm-up: excluded from timing. Triggers graph tracing/XLA
    # compilation and CUDA context/kernel warm-up so steady-state latency
    # is measured, not one-time compilation overhead. ---
    print(f"Warm-up ({N_WARMUP} samples, untimed)...", flush=True)
    for model, Xs in zip(models_, Xs_per_fold):
        for i in range(N_WARMUP):
            _ = model(Xs[i:i + 1], training=False)

    # --- timed single-sample (batch_size=1) ensemble inference ---
    print(f"Timing {N_TIMED} single-sample ensemble inferences...", flush=True)
    per_sample_ms = []
    for i in range(N_WARMUP, N_WARMUP + N_TIMED):
        t0 = time.perf_counter()
        for model, Xs in zip(models_, Xs_per_fold):
            _ = model(Xs[i:i + 1], training=False)
        per_sample_ms.append((time.perf_counter() - t0) * 1000.0)

    per_sample_ms = np.array(per_sample_ms)

    model_file_sizes = [
        (config.MODELS_DIR / f"fold{i}.keras").stat().st_size for i in range(1, n_folds + 1)
    ]

    result = {
        "environment": {
            "hardware": device_name,
            "cpu": "13th Gen Intel Core i9-13900H",
            "os": "WSL2 (Ubuntu) on Windows 11",
            "framework": f"TensorFlow {tf.__version__} (Keras {tf.keras.__version__})",
            "python_version": platform.python_version(),
            "numerical_precision": "float32",
            "batch_size": 1,
            "n_warmup_samples": N_WARMUP,
            "n_timed_repetitions": N_TIMED,
            "ensemble_size": n_folds,
        },
        "latency_ms_per_sample_ensemble": {
            "mean": float(per_sample_ms.mean()),
            "median": float(np.median(per_sample_ms)),
            "p95": float(np.percentile(per_sample_ms, 95)),
            "p99": float(np.percentile(per_sample_ms, 99)),
            "std": float(per_sample_ms.std(ddof=1)),
            "min": float(per_sample_ms.min()),
            "max": float(per_sample_ms.max()),
        },
        "model_size": {
            "per_fold_bytes": model_file_sizes,
            "per_fold_mb_mean": round(np.mean(model_file_sizes) / (1024 * 1024), 3),
            "ensemble_total_mb": round(sum(model_file_sizes) / (1024 * 1024), 3),
        },
        "trainable_params_per_fold": models_[0].count_params(),
    }

    out_path = config.RESULTS_DIR / "latency_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== Latency Benchmark (real measurement) ===")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_benchmark()
