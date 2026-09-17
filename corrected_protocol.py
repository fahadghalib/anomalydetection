"""Leakage-resistant experimental protocol (Reviewer comments #2 and #5).

WHY THIS MODULE EXISTS
----------------------
`src/data_audit.py` and `src/leakage_probe.py` established three facts about
the experiment as originally run:

  1. `sender`, `senderPseudo` and `messageID` were passed to the model as
     input features -- config.DROP_COLS removed only `type` and `class`.
  2. Every sender ID carries exactly one class label (100% class-pure), and
     99.92% of test senders also appear in the training CSV.
  3. Consequently a sender -> class lookup table, with no model and no
     behavioural features at all, scores 99.999% on the "held-out" test set,
     and 99.99% of every StratifiedKFold validation row had its sender seen
     during training.

Any accuracy measured under that setup is therefore uninterpretable: an
unknown share of it is identifier memorisation rather than misbehaviour
detection. This module defines the corrected protocol used to re-measure
everything.

WHAT CHANGES
------------
  * FEATURES. Identifier columns are dropped. `sendTime` is used to derive
    the temporal features and is then dropped as well, because its absolute
    value is a simulation timestamp with no behavioural meaning. What remains
    is 24 physical kinematic columns (position / speed / acceleration /
    heading, raw and noised) plus the 4 derived temporal features = 28
    features.

  * SPLITS. All partitioning is by SENDER GROUP, never by row. A sender-
    disjoint test set is carved out first, then StratifiedGroupKFold is used
    for cross-validation on the remainder, so no sender -- and therefore no
    vehicle trajectory, attack realisation or derived temporal window --
    crosses any partition boundary.

  * POOLING. The two supplied CSVs share 99.92% of their senders, so the
    supplied "test" file cannot serve as an independent test set under this
    protocol. Both files are pooled and re-partitioned by sender.

Temporal features are computed per-sender before partitioning, which is safe:
they depend only on rows from the same sender, so they cannot carry
information across a sender-disjoint boundary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from . import config
from .data_pipeline import _add_temporal_features, _clean

CACHE_DIR = config.DATA_DIR / "cache"
CACHE_FILE = CACHE_DIR / "corrected_protocol_v1.npz"

# Columns excluded from the feature matrix and why.
EXCLUDED = {
    "type": "constant (always 4) -- carries zero information",
    "class": "the target",
    "sender": "identifier -- perfectly determines the label (see leakage_probe)",
    "senderPseudo": "identifier -- 98.3% label-determining on its own",
    "messageID": "identifier -- per-message unique key",
    "sendTime": "absolute simulation timestamp; used to derive the temporal "
                "features, then dropped",
}

TEMPORAL_COLS = ["dt_prev", "pos_delta", "spd_delta", "msg_rate_5s"]

TEST_SENDER_FRACTION = 0.30  # share of senders held out as the sender-disjoint test set


def build_corrected_dataset(force: bool = False):
    """Pool both CSVs, derive temporal features per sender, drop identifiers.

    Returns (X, y, groups, feature_names) where `groups` is the sender ID of
    each row -- kept OUTSIDE the feature matrix purely so the splitter can
    group on it.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists() and not force:
        data = np.load(CACHE_FILE, allow_pickle=True)
        return (data["X"], data["y"], data["groups"],
                data["feature_names"].tolist())

    frames = []
    for path in (config.TRAIN_CSV, config.TEST_CSV):
        print(f"  reading {Path(path).name} ...", flush=True)
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    print(f"  pooled records: {len(df):,}", flush=True)

    df = _clean(df)
    df = _add_temporal_features(df)

    groups = df["sender"].to_numpy()
    y = df[config.TARGET_COL].astype("int32").to_numpy()

    feature_cols = [c for c in df.columns if c not in EXCLUDED]
    X = df[feature_cols].astype("float32").to_numpy()

    print(f"  feature columns kept ({len(feature_cols)}): {feature_cols}", flush=True)
    print(f"  columns dropped: {sorted(EXCLUDED)}", flush=True)

    np.savez(CACHE_FILE, X=X, y=y, groups=groups,
             feature_names=np.array(feature_cols, dtype=object))
    return X, y, groups, feature_cols


def sender_disjoint_holdout(y: np.ndarray, groups: np.ndarray,
                            test_fraction: float = TEST_SENDER_FRACTION,
                            random_state: int = config.RANDOM_STATE):
    """Carve out a sender-disjoint test set.

    Implemented as one split of StratifiedGroupKFold with
    n_splits = round(1/test_fraction), which keeps the class distribution as
    close as the grouping permits while guaranteeing group disjointness.
    """
    n_splits = max(2, int(round(1.0 / test_fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    dev_idx, test_idx = next(iter(sgkf.split(np.zeros(len(y)), y, groups=groups)))

    assert not (set(np.unique(groups[dev_idx]).tolist())
                & set(np.unique(groups[test_idx]).tolist())), \
        "sender overlap between development and test partitions"
    return dev_idx, test_idx


def grouped_folds(y: np.ndarray, groups: np.ndarray,
                  n_splits: int = config.N_FOLDS,
                  random_state: int = config.RANDOM_STATE):
    """Yield (train_idx, val_idx) for sender-disjoint stratified group folds."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for train_idx, val_idx in sgkf.split(np.zeros(len(y)), y, groups=groups):
        assert not (set(np.unique(groups[train_idx]).tolist())
                    & set(np.unique(groups[val_idx]).tolist())), \
            "sender overlap between train and validation folds"
        yield train_idx, val_idx


def describe_partitions():
    """Print the data-flow / partition table requested in reviewer comment #9,
    and verify the disjointness claimed in comment #2."""
    X, y, groups, names = build_corrected_dataset()
    print(f"\n{'=' * 70}\nCORRECTED PROTOCOL PARTITIONS\n{'=' * 70}")
    print(f"  pooled rows       : {len(y):,}")
    print(f"  features          : {X.shape[1]}  {names}")
    print(f"  unique senders    : {len(np.unique(groups)):,}")

    dev_idx, test_idx = sender_disjoint_holdout(y, groups)
    print(f"\n  development rows  : {len(dev_idx):,}  "
          f"({len(np.unique(groups[dev_idx])):,} senders)")
    print(f"  held-out test rows: {len(test_idx):,}  "
          f"({len(np.unique(groups[test_idx])):,} senders)")
    print("  sender overlap dev<->test: 0  (asserted)")

    y_dev, g_dev = y[dev_idx], groups[dev_idx]
    print(f"\n  grouped {config.N_FOLDS}-fold CV on the development partition:")
    for i, (tr, va) in enumerate(grouped_folds(y_dev, g_dev), start=1):
        print(f"    fold {i}: train {len(tr):,} rows / {len(np.unique(g_dev[tr])):,} senders"
              f"   val {len(va):,} rows / {len(np.unique(g_dev[va])):,} senders"
              f"   (sender overlap: 0)")
    return X, y, groups


if __name__ == "__main__":
    describe_partitions()
