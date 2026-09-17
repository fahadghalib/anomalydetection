"""Direct probe of identifier leakage (Reviewer comment #2, extended).

The data audit established that:
  * every sender ID in both CSVs carries exactly one class label (100% pure),
  * 99.92% of test sender IDs also occur in the training CSV,
  * 99.99% of validation rows in every StratifiedKFold fold have a sender that
    also appears in that fold's training portion,
  * and `sender`, `senderPseudo` and `messageID` are passed to the model as
    input features (config.DROP_COLS removes only `type` and `class`).

This module measures how much of the reported performance those identifiers
alone can explain, using two probes that need no neural network:

  A. SENDER LOOKUP BASELINE -- memorise sender -> class from the training CSV,
     then predict test rows by looking their sender up. This is the accuracy
     an attacker-ID lookup table achieves with zero behavioural modelling.

  B. ID BLOCK STRUCTURE -- sort the unique senders numerically and count how
     many contiguous runs of a single class they form. Few long runs means the
     ID axis is piecewise-constant in the label, so even a smooth model fed a
     standardised sender ID (as the current pipeline does) can exploit it by
     simple thresholding; it does not need an exact lookup table.

Run:  python -m src.leakage_probe
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

USE_COLS = ["sender", "senderPseudo", "class"]


def _runs_of_constant_label(sorted_ids: np.ndarray, labels: np.ndarray) -> int:
    """Number of maximal contiguous runs sharing one label, over IDs sorted
    ascending. n_runs == n_classes would mean each class occupies exactly one
    solid block of the ID axis (perfectly threshold-separable)."""
    if len(labels) == 0:
        return 0
    return int(1 + np.count_nonzero(labels[1:] != labels[:-1]))


def run():
    print("Loading identifier columns from both CSVs ...", flush=True)
    train = pd.read_csv(config.TRAIN_CSV, usecols=USE_COLS)
    test = pd.read_csv(config.TEST_CSV, usecols=USE_COLS)

    # ---------------- Probe A: sender lookup baseline ----------------
    lookup = train.groupby("sender")[config.TARGET_COL].agg(
        lambda s: s.value_counts().idxmax()
    )
    mapped = test["sender"].map(lookup)
    covered = mapped.notna()

    n_test = len(test)
    n_covered = int(covered.sum())
    correct_covered = int((mapped[covered].astype(int) == test.loc[covered, config.TARGET_COL]).sum())

    # Accuracy if unseen senders are simply predicted as the majority class.
    majority_class = int(train[config.TARGET_COL].value_counts().idxmax())
    filled = mapped.fillna(majority_class).astype(int)
    acc_full = float((filled == test[config.TARGET_COL]).mean())
    acc_covered_only = correct_covered / n_covered if n_covered else 0.0

    print(f"\n{'=' * 70}\nPROBE A -- SENDER LOOKUP BASELINE (no model, no features)\n{'=' * 70}")
    print(f"  test rows                          : {n_test:,}")
    print(f"  test rows whose sender seen in train: {n_covered:,} ({100.0 * n_covered / n_test:.3f}%)")
    print(f"  accuracy on those covered rows     : {100.0 * acc_covered_only:.3f}%")
    print(f"  accuracy on the FULL test set      : {100.0 * acc_full:.3f}%")

    # Same probe using the pseudonym instead of the raw sender ID.
    lookup_p = train.groupby("senderPseudo")[config.TARGET_COL].agg(
        lambda s: s.value_counts().idxmax()
    )
    mapped_p = test["senderPseudo"].map(lookup_p)
    filled_p = mapped_p.fillna(majority_class).astype(int)
    acc_pseudo = float((filled_p == test[config.TARGET_COL]).mean())
    print(f"  same probe via senderPseudo        : {100.0 * acc_pseudo:.3f}%")

    # ---------------- Probe B: ID block structure ----------------
    per_sender = train.groupby("sender")[config.TARGET_COL].first().sort_index()
    ids_sorted = per_sender.index.to_numpy()
    labels_sorted = per_sender.to_numpy()
    n_runs = _runs_of_constant_label(ids_sorted, labels_sorted)
    n_senders = len(ids_sorted)
    n_classes = int(per_sender.nunique())

    print(f"\n{'=' * 70}\nPROBE B -- SENDER-ID BLOCK STRUCTURE\n{'=' * 70}")
    print(f"  unique senders (train)             : {n_senders:,}")
    print(f"  distinct classes among them        : {n_classes}")
    print(f"  contiguous single-class runs on the sorted ID axis: {n_runs:,}")
    print(f"  mean senders per run               : {n_senders / max(n_runs, 1):.1f}")
    print(f"  (n_runs == n_classes would mean each class is one solid ID block)")

    result = {
        "probe_A_sender_lookup_baseline": {
            "description": "predict each test row's class by looking its sender ID up in a "
                           "sender->class table built from the training CSV; unseen senders "
                           "fall back to the majority class",
            "n_test_rows": n_test,
            "n_test_rows_with_sender_seen_in_train": n_covered,
            "pct_test_rows_covered": round(100.0 * n_covered / n_test, 3),
            "accuracy_on_covered_rows_pct": round(100.0 * acc_covered_only, 3),
            "accuracy_full_test_set_pct": round(100.0 * acc_full, 3),
            "accuracy_via_senderPseudo_pct": round(100.0 * acc_pseudo, 3),
        },
        "probe_B_id_block_structure": {
            "n_unique_senders_train": n_senders,
            "n_classes": n_classes,
            "n_contiguous_single_class_runs": n_runs,
            "mean_senders_per_run": round(n_senders / max(n_runs, 1), 2),
        },
        "reference_points": {
            "proposed_model_M7_test_accuracy_pct": 81.73,
            "mlp_baseline_M8_test_accuracy_pct": 84.31,
        },
    }

    out_path = config.RESULTS_DIR / "leakage_probe.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved -> {out_path}")
    return result


if __name__ == "__main__":
    run()
