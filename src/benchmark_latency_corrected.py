"""Single-message latency and batched throughput of the reported model.

The model timed is the one evaluated on the held-out partition: the single
28-feature CNN-BiGRU-attention network written by
`run_corrected.evaluate_on_holdout` to outputs/models/final_cnn.keras. No
ensemble is involved anywhere in the reported results.

Two execution paths are timed for the same model and the same inputs:

  * eager    -- `model(x, training=False)`, the default Keras call. Every
                layer, and every one of the BiGRU's recurrent steps, is
                dispatched as a separate framework operation.
  * graph    -- the same forward pass wrapped in `tf.function` with a fixed
                input signature, so it is traced once and then executed as a
                single compiled graph.

If the two differ by a large factor, the eager figure is dominated by
per-operation dispatch overhead rather than by the arithmetic cost of the
model, which is the explanation the manuscript gives and which this
measurement tests directly.

Inputs are real held-out rows, scaled by a StandardScaler fitted on the
development partition only. Scaling happens before timing: only the forward
pass is measured. Run on an otherwise idle machine.

On native Windows the process opts out of execution-speed power throttling
(EcoQoS) before timing. Windows 11 applies this throttling to processes that
own no foreground window, such as one started from a script or an agent, and
on a hybrid-core laptop it moves them to efficiency cores at reduced clock: on
the development machine it slowed the eager forward pass from about 270 ms to
about 1.9 s. WSL2 runs inside a virtual machine and is not affected.

Usage:
  python -m src.benchmark_latency_corrected --device gpu --tag corrected_gpu
  python -m src.benchmark_latency_corrected --device cpu --tag corrected_wsl2_cpu
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
from .model import AdditiveAttention                  # noqa: E402,F401  (registers the layer)

N_WARMUP = 50
N_TIMED = 300
MODEL_FILE = config.MODELS_DIR / "final_cnn.keras"


def _time_single(fn, Xs) -> np.ndarray:
    for i in range(N_WARMUP):
        _ = fn(Xs[i:i + 1])
    out = []
    for i in range(N_WARMUP, N_WARMUP + N_TIMED):
        t0 = time.perf_counter()
        r = fn(Xs[i:i + 1])
        _ = r.numpy()                 # force completion before stopping the clock
        out.append((time.perf_counter() - t0) * 1000.0)
    return np.array(out)


def _disable_windows_power_throttling() -> str:
    """Opt this process out of EcoQoS execution-speed throttling (Windows only)."""
    if platform.system() != "Windows":
        return "not applicable"
    import ctypes
    import ctypes.wintypes as wt

    class _PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wt.ULONG), ("ControlMask", wt.ULONG), ("StateMask", wt.ULONG)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wt.HANDLE
    kernel32.SetProcessInformation.argtypes = [wt.HANDLE, ctypes.c_int, ctypes.c_void_p, wt.DWORD]
    kernel32.SetProcessInformation.restype = wt.BOOL
    # ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED, StateMask = 0 (off);
    # 4 = ProcessPowerThrottling in the PROCESS_INFORMATION_CLASS enumeration.
    state = _PowerThrottlingState(1, 0x1, 0x0)
    ok = kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), 4,
                                        ctypes.byref(state), ctypes.sizeof(state))
    if not ok:
        raise OSError(ctypes.get_last_error(), "could not disable power throttling")
    return "execution-speed throttling disabled for this process"


def _summary(ms: np.ndarray) -> dict:
    return {"mean": float(ms.mean()), "median": float(np.median(ms)),
            "p95": float(np.percentile(ms, 95)), "p99": float(np.percentile(ms, 99)),
            "std": float(ms.std(ddof=1)), "min": float(ms.min()), "max": float(ms.max())}


def run_benchmark():
    gpus = tf.config.list_physical_devices("GPU")
    device_name = ("GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8GB)" if gpus else "CPU only")
    tag = _ARGS.tag or ("corrected_gpu" if gpus else "corrected_cpu")
    is_wsl = ("microsoft" in platform.release().lower()
              or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop"))
    host = ("WSL2 (Ubuntu) on Windows 11" if is_wsl
            else f"native {platform.system()} {platform.release()}")
    power = _disable_windows_power_throttling()
    print(f"host={host}  device={device_name}  tag={tag}  power throttling: {power}", flush=True)

    X, y, groups, feature_names = build_corrected_dataset()
    n_features = X.shape[1]
    dev_idx, test_idx = sender_disjoint_holdout(y, groups)
    scaler = StandardScaler().fit(X[dev_idx])

    rng = np.random.RandomState(config.RANDOM_STATE)
    sample_idx = rng.choice(test_idx, size=N_WARMUP + N_TIMED, replace=False)
    Xs = scaler.transform(X[sample_idx]).reshape(-1, n_features, 1).astype("float32")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"{MODEL_FILE} not found -- run "
                                "`python -m src.run_corrected --models cnn --eval-test` first")
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    n_trainable = int(sum(np.prod(w.shape) for w in model.trainable_weights))
    n_total = model.count_params()
    weights_bytes = int(sum(np.prod(w.shape) for w in model.weights) * 4)
    print(f"  model {MODEL_FILE.name}: trainable {n_trainable:,}, total {n_total:,}, "
          f"weights {weights_bytes / 1048576:.3f} MB", flush=True)

    def eager_fn(x):
        return model(x, training=False)

    graph_fn = tf.function(eager_fn, input_signature=[tf.TensorSpec([1, n_features, 1], tf.float32)])

    print(f"eager: {N_WARMUP} warm-up + {N_TIMED} timed calls ...", flush=True)
    eager_ms = _time_single(eager_fn, Xs)
    print(f"  median {np.median(eager_ms):.3f} ms", flush=True)
    print(f"graph: {N_WARMUP} warm-up + {N_TIMED} timed calls ...", flush=True)
    graph_ms = _time_single(graph_fn, Xs)
    print(f"  median {np.median(graph_ms):.3f} ms", flush=True)

    # Batched throughput: wall-clock time of one full batch divided by the
    # batch size. This describes processing of queued messages and is NOT a
    # per-message latency.
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
    print(f"batch of {config.BATCH_SIZE}: {batch_s * 1000:.1f} ms -> "
          f"{batch_s * 1000 / config.BATCH_SIZE:.4f} ms/message", flush=True)

    result = {
        "model": "final held-out CNN-BiGRU-attention model (single model)",
        "model_file": f"outputs/models/{MODEL_FILE.name}",
        "environment": {
            "hardware": device_name,
            "cpu": "13th Gen Intel Core i9-13900H",
            "os": host,
            "windows_power_throttling": power,
            "framework": f"TensorFlow {tf.__version__} (Keras {tf.keras.__version__})",
            "python_version": platform.python_version(),
            "numerical_precision": "float32",
            "batch_size": 1,
            "n_warmup_calls": N_WARMUP,
            "n_timed_calls": N_TIMED,
            "n_features": int(n_features),
        },
        "latency_ms_per_message": _summary(eager_ms),
        "latency_ms_per_message_graph": _summary(graph_ms),
        "batched_throughput": {
            "batch_size": int(config.BATCH_SIZE),
            "batch_wall_clock_ms_median": round(batch_s * 1000, 2),
            "ms_per_sample": round(batch_s * 1000 / config.BATCH_SIZE, 4),
            "n_repetitions": 5,
            "n_warmup": 3,
        },
        "model_size": {"weights_bytes_float32": weights_bytes,
                       "weights_mb": round(weights_bytes / 1048576, 3),
                       "file_bytes": int(MODEL_FILE.stat().st_size),
                       "file_mb": round(MODEL_FILE.stat().st_size / 1048576, 3)},
        "trainable_params": n_trainable,
        "total_params": int(n_total),
    }

    out_path = config.RESULTS_DIR / f"latency_benchmark_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"saved -> {out_path}")
    return result


if __name__ == "__main__":
    run_benchmark()
