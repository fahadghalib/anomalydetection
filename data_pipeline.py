"""Data loading, cleaning, and temporal feature engineering for the VeReMi
Extension dataset.

Beyond the paper's 28 per-message features, this module adds 4 temporal
features derived from message TIMING and CONTINUITY per sender (using the
'sender' and 'sendTime' columns, which the base pipeline otherwise loads as
an inert scaled ID and drops entirely):

  - dt_prev:     seconds since this sender's previous message
  - pos_delta:   how far (posx,posy) moved since the previous message
  - spd_delta:   how far (spdx,spdy) changed since the previous message
  - msg_rate_5s: how many messages this sender sent in the trailing 5s

Motivation: Data Replay Attack and DDoS are defined by REPETITION/RATE over
time, not by what any single message looks like -- a replayed message is,
by construction, nearly identical to a real one in the 28 static features.
The base single-row architecture is structurally blind to that; these 4
features give it a per-message signal for "does this look like a frozen
replay" (near-zero pos_delta/spd_delta despite normal dt_prev) or "is this
sender flooding" (high msg_rate_5s), without changing the model at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

CACHE_DIR = config.DATA_DIR / "cache"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf with NaN, then drop any row containing NaN (paper: Sec 3.1)."""
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        print(f"  dropped {dropped} rows with inf/NaN values")
    return df


def _rolling_count_window(times: np.ndarray, window_sec: float) -> np.ndarray:
    """For each sorted timestamp, count how many timestamps (including
    itself) fall within [t - window_sec, t]. O(n log n) via searchsorted."""
    left = np.searchsorted(times, times - window_sec, side="left")
    idx = np.arange(len(times))
    return (idx - left + 1).astype("float32")


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sender", "sendTime"], kind="mergesort").reset_index(drop=True)
    g = df.groupby("sender", sort=False)

    prev_time = g["sendTime"].shift(1)
    prev_posx = g["posx"].shift(1)
    prev_posy = g["posy"].shift(1)
    prev_spdx = g["spdx"].shift(1)
    prev_spdy = g["spdy"].shift(1)

    df["dt_prev"] = (df["sendTime"] - prev_time).fillna(999.0).clip(upper=999.0).astype("float32")
    df["pos_delta"] = np.sqrt((df["posx"] - prev_posx) ** 2 + (df["posy"] - prev_posy) ** 2).fillna(0.0).astype("float32")
    df["spd_delta"] = np.sqrt((df["spdx"] - prev_spdx) ** 2 + (df["spdy"] - prev_spdy) ** 2).fillna(0.0).astype("float32")

    df["msg_rate_5s"] = g["sendTime"].transform(
        lambda s: _rolling_count_window(s.to_numpy(), 5.0)
    ).astype("float32")

    return df


def load_split(csv_path) -> tuple[np.ndarray, np.ndarray]:
    """Load one CSV file, return (X, y) as numpy arrays.

    X has config.N_FEATURES columns: the paper's 28 numeric columns (all
    columns except 'type' and 'class' -- see config.py) plus the 4 derived
    temporal features above. y is the integer class label 0..19 from the
    'class' column. Results are cached to data/cache/ as .npz since the
    temporal feature computation is not free and this pipeline gets re-run
    across multiple training attempts.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / (Path(csv_path).stem + "_temporal_v1.npz")
    if cache_path.exists():
        data = np.load(cache_path)
        X, y = data["X"], data["y"]
        assert X.shape[1] == config.N_FEATURES, (
            f"cached features have {X.shape[1]} cols, expected {config.N_FEATURES} "
            f"-- delete {cache_path} and re-run if config.py's feature set changed"
        )
        return X, y

    df = pd.read_csv(csv_path)
    df = _clean(df)
    df = _add_temporal_features(df)

    y = df[config.TARGET_COL].astype("int32").to_numpy()
    X = df.drop(columns=config.DROP_COLS).astype("float32").to_numpy()

    assert X.shape[1] == config.N_FEATURES, (
        f"expected {config.N_FEATURES} features, got {X.shape[1]}"
    )

    np.savez(cache_path, X=X, y=y)
    return X, y


def load_train() -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading training data from {config.TRAIN_CSV} ...")
    X, y = load_split(config.TRAIN_CSV)
    print(f"  train shape: X={X.shape}, y={y.shape}, classes={sorted(set(y.tolist()))}")
    return X, y


def load_test() -> tuple[np.ndarray, np.ndarray]:
    print(f"Loading test data from {config.TEST_CSV} ...")
    X, y = load_split(config.TEST_CSV)
    print(f"  test shape: X={X.shape}, y={y.shape}, classes={sorted(set(y.tolist()))}")
    return X, y
