"""Dataset accounting + leakage diagnostics (Reviewer comments #2 and #9).

Produces, from the REAL CSV files, everything needed for:

  * the data-flow table the reviewer asks for in comment #9 (raw records ->
    duplicate removal -> missing/infinite exclusion -> train/test construction
    -> fold sizes), and

  * the split-validity evidence demanded in comment #2: whether messages from
    the same sender / pseudonym / simulation cross the train-test boundary and
    the boundaries of the shuffled StratifiedKFold folds used so far.

It answers three questions the manuscript currently cannot:

  1. Do the two CSVs together account for every record in the public dataset
     (3,194,808 instances)?
  2. Are identifier columns (sender, senderPseudo, messageID) being fed to the
     model as input features, and is a sender's class label constant -- i.e.
     can the network solve the task by memorising which IDs are attackers
     rather than by learning behaviour?
  3. Under the row-level StratifiedKFold used so far, what fraction of each
     validation fold's senders were also seen in that fold's training
     portion?

Run:  python -m src.data_audit
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config

ID_COLS = ["sender", "senderPseudo", "messageID"]
PUBLIC_TOTAL = 3_194_808  # instance count reported by the public dataset description


def _audit_split(csv_path, name: str):
    print(f"\n{'=' * 70}\n{name}: {csv_path}\n{'=' * 70}", flush=True)
    df = pd.read_csv(csv_path)
    raw_rows = len(df)
    print(f"  raw records: {raw_rows:,}", flush=True)

    n_dup = int(df.duplicated().sum())
    df_nodup = df.drop_duplicates()
    print(f"  exact duplicate rows: {n_dup:,}", flush=True)

    df_clean = df_nodup.replace([np.inf, -np.inf], np.nan)
    n_nan_rows = int(df_clean.isna().any(axis=1).sum())
    df_clean = df_clean.dropna()
    print(f"  rows with NaN/inf: {n_nan_rows:,}", flush=True)
    print(f"  rows after cleaning: {len(df_clean):,}", flush=True)

    # 'type' is claimed in the manuscript to be the 20-class target.
    type_vals = sorted(df_clean["type"].unique().tolist())
    class_vals = sorted(df_clean[config.TARGET_COL].unique().tolist())

    # Is a sender's label constant? If yes, sender ID alone determines the class.
    per_sender_classes = df_clean.groupby("sender")[config.TARGET_COL].nunique()
    n_senders = int(per_sender_classes.shape[0])
    n_pure = int((per_sender_classes == 1).sum())

    per_pseudo_classes = df_clean.groupby("senderPseudo")[config.TARGET_COL].nunique()
    n_pseudo = int(per_pseudo_classes.shape[0])
    n_pseudo_pure = int((per_pseudo_classes == 1).sum())

    class_counts = df_clean[config.TARGET_COL].value_counts().sort_index().to_dict()

    result = {
        "split": name,
        "raw_records": raw_rows,
        "exact_duplicate_rows_removed": n_dup,
        "rows_with_nan_or_inf_removed": n_nan_rows,
        "records_after_cleaning": len(df_clean),
        "n_columns": int(df.shape[1]),
        "type_column_unique_values": type_vals,
        "class_column_unique_values": class_vals,
        "n_unique_senders": n_senders,
        "n_senders_with_single_class": n_pure,
        "pct_senders_class_pure": round(100.0 * n_pure / max(n_senders, 1), 2),
        "n_unique_senderPseudo": n_pseudo,
        "pct_pseudo_class_pure": round(100.0 * n_pseudo_pure / max(n_pseudo, 1), 2),
        "class_counts": {int(k): int(v) for k, v in class_counts.items()},
    }
    print(f"  'type' unique values: {type_vals}", flush=True)
    print(f"  unique senders: {n_senders:,}  ({result['pct_senders_class_pure']}% carry exactly one class)", flush=True)
    print(f"  unique senderPseudo: {n_pseudo:,}  ({result['pct_pseudo_class_pure']}% class-pure)", flush=True)

    ids = {
        "sender": set(df_clean["sender"].unique().tolist()),
        "senderPseudo": set(df_clean["senderPseudo"].unique().tolist()),
        "messageID": set(df_clean["messageID"].unique().tolist()),
    }
    labels = df_clean[config.TARGET_COL].to_numpy()
    senders = df_clean["sender"].to_numpy()
    return result, ids, labels, senders


def _fold_leakage(senders: np.ndarray, y: np.ndarray, n_splits: int) -> dict:
    """Under the SHUFFLED row-level StratifiedKFold actually used for the
    reported cross-validation results, quantify how much sender information
    crosses the train/validation boundary."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    per_fold = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        tr_senders = set(np.unique(senders[train_idx]).tolist())
        va_senders = np.unique(senders[val_idx])
        n_val_senders = len(va_senders)
        n_seen = int(sum(1 for s in va_senders.tolist() if s in tr_senders))
        # fraction of validation ROWS whose sender was also in training
        seen_mask = np.isin(senders[val_idx], list(tr_senders))
        pct_rows = round(100.0 * float(seen_mask.mean()), 3)
        per_fold.append({
            "fold": fold_idx,
            "n_val_senders": n_val_senders,
            "n_val_senders_also_in_train": n_seen,
            "pct_val_senders_also_in_train": round(100.0 * n_seen / max(n_val_senders, 1), 3),
            "pct_val_rows_whose_sender_seen_in_train": pct_rows,
            "n_train_rows": int(len(train_idx)),
            "n_val_rows": int(len(val_idx)),
        })
        print(f"  fold {fold_idx}: {n_seen}/{n_val_senders} validation senders "
              f"({per_fold[-1]['pct_val_senders_also_in_train']}%) also appear in training; "
              f"{pct_rows}% of validation rows", flush=True)
    return per_fold


def run_audit():
    train_res, train_ids, y_train, senders_train = _audit_split(config.TRAIN_CSV, "train")
    test_res, test_ids, y_test, senders_test = _audit_split(config.TEST_CSV, "test")

    total_raw = train_res["raw_records"] + test_res["raw_records"]
    total_clean = train_res["records_after_cleaning"] + test_res["records_after_cleaning"]

    print(f"\n{'=' * 70}\nTRAIN <-> TEST IDENTIFIER OVERLAP\n{'=' * 70}", flush=True)
    overlap = {}
    for col in ID_COLS:
        inter = train_ids[col] & test_ids[col]
        overlap[col] = {
            "n_train": len(train_ids[col]),
            "n_test": len(test_ids[col]),
            "n_shared": len(inter),
            "pct_of_test_ids_seen_in_train": round(100.0 * len(inter) / max(len(test_ids[col]), 1), 3),
        }
        print(f"  {col}: {len(inter):,} shared "
              f"({overlap[col]['pct_of_test_ids_seen_in_train']}% of test {col} values also occur in train)", flush=True)

    print(f"\n{'=' * 70}\nROW-LEVEL StratifiedKFold LEAKAGE (n_splits={config.N_FOLDS}, "
          f"shuffle=True, seed={config.RANDOM_STATE})\n{'=' * 70}", flush=True)
    fold_leak = _fold_leakage(senders_train, y_train, config.N_FOLDS)

    summary = {
        "public_dataset_reported_instances": PUBLIC_TOTAL,
        "train": train_res,
        "test": test_res,
        "combined_raw_records": total_raw,
        "combined_records_after_cleaning": total_clean,
        "raw_matches_public_total": bool(total_raw == PUBLIC_TOTAL),
        "identifier_columns_used_as_model_features": ID_COLS,
        "train_test_identifier_overlap": overlap,
        "row_level_stratifiedkfold_leakage": fold_leak,
    }

    out_path = config.RESULTS_DIR / "data_audit.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}\nDATA-FLOW SUMMARY\n{'=' * 70}")
    print(f"  public dataset reported instances : {PUBLIC_TOTAL:,}")
    print(f"  train raw + test raw              : {total_raw:,}"
          f"  ({'MATCHES' if total_raw == PUBLIC_TOTAL else 'DOES NOT MATCH'})")
    print(f"  combined after cleaning           : {total_clean:,}")
    print(f"\n  saved -> {out_path}")
    return summary


if __name__ == "__main__":
    run_audit()
