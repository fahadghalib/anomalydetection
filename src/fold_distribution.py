"""Class counts of the pooled corpus and of every training fold before and
after ADASYN (manuscript Table S1).

`before` is computed from the sender-disjoint folds of the development
partition; `after` is read from the cached, resampled training arrays that
src/run_corrected.py writes to data/cache/corrected_folds/, i.e. from exactly
the data the models were trained on.

Run:  python -m src.fold_distribution   ->  outputs/results/fold_class_distribution.json
"""

from __future__ import annotations

import json

import numpy as np

from . import config
from .corrected_protocol import build_corrected_dataset, grouped_folds, sender_disjoint_holdout
from .run_corrected import FOLD_CACHE_DIR


def _counts(y) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    return {str(int(c)): int(n) for c, n in zip(classes, counts)}


def main():
    _X, y, groups, _names = build_corrected_dataset()
    dev_idx, _test_idx = sender_disjoint_holdout(y, groups)
    y_dev, g_dev = y[dev_idx], groups[dev_idx]

    out = {"pooled": _counts(y), "before": {}, "after": {}}
    for k, (tr, _va) in enumerate(grouped_folds(y_dev, g_dev), start=1):
        out["before"][str(k)] = _counts(y_dev[tr])
        cache = FOLD_CACHE_DIR / f"fold{k}.npz"
        if cache.exists():
            out["after"][str(k)] = _counts(np.load(cache)["y_tr"])
        print(f"fold {k}: {len(tr):,} training rows before, "
              f"{sum(out['after'].get(str(k), {}).values()):,} after ADASYN")

    path = config.RESULTS_DIR / "fold_class_distribution.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
