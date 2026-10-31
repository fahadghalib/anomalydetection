"""Cleaning and per-sender temporal features for the VeReMi Extension release.

Four features are derived from each sender's messages ordered by send time:

  - dt_prev:     seconds since this sender's previous message (999 s for its first)
  - pos_delta:   distance between the positions (posx, posy) of consecutive messages
  - spd_delta:   magnitude of the change in the velocity vector (spdx, spdy)
  - msg_rate_5s: messages sent by this sender in the trailing 5 s, current included

Data replay and denial-of-service attacks are defined by repetition and rate
over time rather than by the content of a single message; these features make
that information available to a per-message classifier. Each feature depends
only on the current and earlier messages of the same sender, so it carries no
information across a sender-disjoint partition boundary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf with NaN, then drop any row containing NaN (none in this release)."""
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
