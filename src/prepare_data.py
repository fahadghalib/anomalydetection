"""Recreate the two input files from the Mendeley Data release.

The release (Slama et al., Mendeley Data, doi:10.17632/k62n4z9gdz.1) is one
archive, mixalldata_clean.tar.xz, holding one file, mixalldata_clean.csv, with
all 3,194,808 records. Earlier versions of this study used a random 70/30
row-level split of that file into VeReMi_train_data.csv (2,236,367 records)
and VeReMi_test_data.csv (958,441 records). The split does not affect any
reported result: corrected_protocol.py pools the two files again and
re-partitions the records by sender. It is kept because the leakage audit
(data_audit.py, leakage_probe.py) measures exactly what such a row-level split
does, and so that every script reads the files it was run on.

data/rowlevel_split.npz marks which rows of mixalldata_clean.csv went to the
test file; both files keep the row order of the release. The script checks the
SHA-256 of mixalldata_clean.csv and, after splitting, a fingerprint of the
identifier and label columns of each file against the files used in the study.
The original files were written by R, whose number formatting differs from
Python's in the 15th significant digit, so numeric values agree to about 14
significant digits and the file checksums differ from those in data/README.md.

Usage:
  tar -xJf mixalldata_clean.tar.xz -C data/
  python -m src.prepare_data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

SPLIT_FILE = config.DATA_DIR / "rowlevel_split.npz"
SOURCE_CSV = config.DATA_DIR / "mixalldata_clean.csv"
KEY_COLS = ["type", "sender", "senderPseudo", "messageID", "class"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def _key_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(np.ascontiguousarray(df[KEY_COLS].to_numpy(dtype=np.int64)).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--source", type=Path, default=SOURCE_CSV)
    ap.add_argument("--out-dir", type=Path, default=config.DATA_DIR)
    args = ap.parse_args()

    split = np.load(SPLIT_FILE)
    n_rows = int(split["n_rows"])
    if _sha256(args.source) != str(split["source_sha256"]):
        raise SystemExit(f"{args.source} is not the mixalldata_clean.csv of the Mendeley release")
    df = pd.read_csv(args.source)
    assert len(df) == n_rows
    is_test = np.unpackbits(split["is_test"])[:n_rows].astype(bool)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for tag, name, part in (("train", config.TRAIN_CSV.name, df[~is_test]),
                            ("test", config.TEST_CSV.name, df[is_test])):
        if _key_fingerprint(part) != str(split[f"{tag}_key_sha256"]):
            raise SystemExit(f"{name}: identifiers or labels differ from the files used in the study")
        part.to_csv(args.out_dir / name, index=False, float_format="%.15g", quoting=csv.QUOTE_NONNUMERIC,
                    lineterminator="\r\n")
        print(f"{name}: {len(part):,} records; identifiers and labels match the study's file")


if __name__ == "__main__":
    main()
