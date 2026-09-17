"""Single-message inference latency for the model actually reported in the
paper: the 28-feature corrected-protocol CNN-BiGRU-Attention network,
timed as ONE model rather than as a five-model ensemble.

`benchmark_latency.py` times the five 32-feature fold models written by the
superseded `train.py` pipeline. Those models include the identifier columns
that the leakage audit removed, so their input width -- and therefore their
parameter count and forward-pass cost -- does not correspond to any model
reported in the corrected results.

Latency and model size are determined by architecture and input width, not by
the values of the trained weights, so this benchmark constructs the corrected
architecture directly. Timing is reported for ONE model, matching the
single-model held-out evaluation in `run_corrected.evaluate_on_holdout`.

Inputs are real held-out rows from the sender-disjoint test partition, scaled
by a StandardScaler fitted on the development partition only. Scaling is
performed before timing begins: only the forward pass is measured.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

_ap = argparse.ArgumentParser()
_ap.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
_ap.add_argument("--tag", default=None,
                 help="label for the output file, e.g. corrected_gpu")
_ARGS, _ = _ap.parse_known_args()
if _ARGS.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np                                    # noqa: E402
import tensorflow as tf                               # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402

from . import config                                  # noqa: E402
from .corrected_protocol import (                     # noqa: E402
    build_corrected_dataset, sender_disjoint_holdout)
from .run_corrected import build_cnn_bigru_attention  # noqa: E402

N_WARMUP = 50
N_TIMED = 300
MODEL_FILE = config.MODELS_DIR / "corrected_cnn_28f.keras"


def run_benchmark():
    gpus = tf.config.list_physical_devices("GPU")
    device_name = ("GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB)"
                   if gpus else "CPU only")
    tag = _ARGS.tag or ("corrected_gpu" if gpus else "corrected_cpu")
    is_wsl = ("microsoft" in platform.release().lower()
              or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop"))
    host = ("WSL2 (Ubuntu) on Windows 11" if is_wsl
            else f"native {platform.system()} {platform.release()}")
    print(f"host={host}  device={device_name}  tag={tag}", flush=True)

    print("Loading corrected dataset ...", flush=True)
    X, y, groups, feature_names = build_corrected_dataset()
    n_features = X.shape[1]
    print(f"  features: {n_features}  rows: {len(X):,}", flush=True)

    dev_idx, test_idx = sender_disjoint_holdout(y, groups)

    # The scaler is fitted on development rows only, exactly as in training.
    # It is applied before the timing loop; scaling is not part of the
    # measured forward-pass latency.
    scaler = StandardScaler().fit(X[dev_idx])

    rng = np.random.RandomState(config.RANDOM_STATE)
    sample_idx = rng.choice(test_idx, size=N_WARMUP + N_TIMED, replace=False)
    Xs = scaler.transform(X[sample_idx]).reshape(-1, n_features, 1).astype("float32")

    print("Building corrected architecture ...", flush=True)
    tf.keras.utils.set_random_seed(config.RANDOM_STATE)
    model = build_cnn_bigru_attention(n_features, config.N_CLASSES)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_FILE)
    # Trainable count matches `_fit_keras`, which is what the results files
    # and the manuscript report; the total additionally includes the 576
    # non-trainable BatchNormalization moving statistics.
    n_trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    n_total = model.count_params()
    size_bytes = MODEL_FILE.stat().st_size
    print(f"  trainable: {n_trainable:,}  total: {n_total:,}  "
          f"file: {size_bytes / 1048576:.3f} MB", flush=True)

    print(f"Warm-up ({N_WARMUP} calls, untimed) ...", flush=True)
    for i in range(N_WARMUP):
        _ = model(Xs[i:i + 1], training=False)

    print(f"Timing {N_TIMED} single-message inferences ...", flush=True)
    per_sample_ms = []
    for j, i in enumerate(range(N_WARMUP, N_WARMUP + N_TIMED)):
        t0 = time.perf_counter()
        _ = model(Xs[i:i + 1], training=False)
        per_sample_ms.append((time.perf_counter() - t0) * 1000.0)
        if (j + 1) % 50 == 0:
            print(f"  {j + 1}/{N_TIMED}", flush=True)
    per_sample_ms = np.array(per_sample_ms)

    # Batched throughput: wall-clock time of one full batch divided by the
    # batch size. This is NOT a latency figure -- it describes processing of
    # queued messages and is reported separately to avoid conflating the two.
    print(f"Batched throughput (batch={config.BATCH_SIZE}) ...", flush=True)
    idx_b = rng.choice(test_idx, size=config.BATCH_SIZE, replace=False)
    Xb = scaler.transform(X[idx_b]).reshape(-1, n_features, 1).astype("float32")
    for _ in range(3):
        _ = model.predict(Xb, batch_size=config.BATCH_SIZE, verbose=0)
    batch_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = model.predict(Xb, batch_size=config.BATCH_SIZE, verbose=0)
        batch_times.append(time.perf_counter() - t0)
    batch_s = float(np.median(batch_times))
    print(f"  batch wall-clock {batch_s * 1000:.1f} ms -> "
          f"{batch_s * 1000 / config.BATCH_SIZE:.4f} ms/sample", flush=True)

    result = {
        "model": "corrected-protocol CNN-BiGRU-Attention (single model)",
        "note": ("Timed on the 28-feature architecture reported in the "
                 "corrected results. Weights are freshly initialised: "
                 "forward-pass latency and serialised size depend on "
                 "architecture and input width, not on weight values."),
        "environment": {
            "hardware": device_name,
            "cpu": "13th Gen Intel Core i9-13900H",
            "os": host,
            "framework": f"TensorFlow {tf.__version__} (Keras {tf.keras.__version__})",
            "python_version": platform.python_version(),
            "numerical_precision": "float32",
            "batch_size": 1,
            "n_warmup_samples": N_WARMUP,
            "n_timed_repetitions": N_TIMED,
            "n_features": int(n_features),
            "feature_names": feature_names,
        },
        "latency_ms_per_message": {
            "mean": float(per_sample_ms.mean()),
            "median": float(np.median(per_sample_ms)),
            "p95": float(np.percentile(per_sample_ms, 95)),
            "p99": float(np.percentile(per_sample_ms, 99)),
            "std": float(per_sample_ms.std(ddof=1)),
            "min": float(per_sample_ms.min()),
            "max": float(per_sample_ms.max()),
        },
        "batched_throughput": {
            "batch_size": int(config.BATCH_SIZE),
            "batch_wall_clock_ms_median": round(batch_s * 1000, 2),
            "ms_per_sample": round(batch_s * 1000 / config.BATCH_SIZE, 4),
            "n_repetitions": 5,
            "n_warmup": 3,
        },
        "model_size": {
            "bytes": int(size_bytes),
            "mb": round(size_bytes / 1048576, 3),
        },
        "trainable_params": n_trainable,
        "total_params": int(n_total),
    }

    out_path = config.RESULTS_DIR / f"latency_benchmark_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("\n=== Corrected-protocol latency benchmark ===")
    print(json.dumps({k: v for k, v in result.items() if k != "environment"}, indent=2))
    print(f"\nsaved -> {out_path}")
    return result


if __name__ == "__main__":
    run_benchmark()
