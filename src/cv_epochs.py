"""Epoch budget for the final held-out refit, chosen from cross-validation only.

The held-out partition must play no part in any training decision. The final
model for each family is therefore retrained on the full development partition
for a FIXED number of epochs, equal to the median best epoch (minimum
validation loss) observed across its five sender-disjoint CV folds. No
validation data and no early stopping are used in that refit.

The best epoch of each CV run is taken from the result file when it carries a
`best_epoch` field (runs made with the current `run_corrected.py`). For the
runs deposited before that field existed, it is recovered from the deposited
Keras training logs. A log run is accepted for a (model, fold) pair only if
  * it used the shared 40-epoch budget, and
  * the accuracy it printed equals, to four decimals, the value recorded for
    that pair in `corrected_protocol_results.json`,
which ties every recovered epoch to the exact run whose score is reported.

Run:  python -m src.cv_epochs     -> outputs/results/cv_best_epochs.json
"""

from __future__ import annotations

import json
import re

import numpy as np

from . import config

LOG_FILES = ["chain_runs.log", "chain_runs2.log", "corrected_mlp.log",
             "corrected_protocol.log"]
SHARED_BUDGET = 40
OUT_FILE = config.RESULTS_DIR / "cv_best_epochs.json"


def _parse_logs():
    """Yield one record per CV training run found in the deposited logs."""
    for name in LOG_FILES:
        path = config.LOGS_DIR / name
        if not path.exists():
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        cur, hist, budget, epoch = None, [], None, None
        for ln in lines:
            m = re.match(r"^--- (\S+) \| fold (\d+) ---", ln)
            if m:
                cur, hist, budget = (m.group(1), int(m.group(2))), [], None
                continue
            m = re.match(r"^Epoch (\d+)/(\d+)", ln)
            if m and cur:
                epoch, budget = int(m.group(1)), int(m.group(2))
                continue
            m = re.search(r"val_loss: ([\d.]+)", ln)
            if m and cur and epoch is not None:
                hist.append((epoch, float(m.group(1))))
                continue
            m = re.match(r"^\s+(\S+) fold (\d+): acc=([\d.]+)", ln)
            if m and cur and (m.group(1), int(m.group(2))) == cur and hist:
                best_epoch, best_loss = min(hist, key=lambda e: e[1])
                yield {"model": cur[0], "fold": cur[1], "budget": budget,
                       "epochs_run": len(hist), "best_epoch": best_epoch,
                       "best_val_loss": best_loss, "printed_acc": m.group(3),
                       "log": name}
                cur, hist = None, []


def main():
    with open(config.RESULTS_DIR / "corrected_protocol_results.json") as f:
        cv = json.load(f)
    log_runs = list(_parse_logs())

    out = {"rule": "median over CV folds of the epoch with minimum validation loss; "
                   "the final refit uses this fixed budget with no validation data",
           "models": {}}
    for key in ("cnn", "mlp_cap", "mlp_matched", "cnn_permuted"):
        runs = cv.get(key, [])
        per_fold = []
        for r in sorted(runs, key=lambda r: r["fold"]):
            if "best_epoch" in r:
                per_fold.append({"fold": r["fold"], "best_epoch": r["best_epoch"],
                                 "source": "corrected_protocol_results.json"})
                continue
            target = f"{r['val_accuracy']:.4f}"
            match = [lr for lr in log_runs
                     if lr["model"] == key and lr["fold"] == r["fold"]
                     and lr["budget"] == SHARED_BUDGET and lr["printed_acc"] == target]
            if not match:
                raise RuntimeError(f"no log run matches {key} fold {r['fold']} acc={target}")
            lr = match[-1]
            per_fold.append({"fold": r["fold"], "best_epoch": lr["best_epoch"],
                             "epochs_run": lr["epochs_run"],
                             "best_val_loss": lr["best_val_loss"],
                             "source": f"outputs/logs/{lr['log']}"})
        if len(per_fold) < 2:
            continue
        epochs = [p["best_epoch"] for p in per_fold]
        out["models"][key] = {"per_fold": per_fold,
                              "final_epochs": int(np.median(epochs))}
        print(f"  {key:13s} best epochs per fold {epochs} -> final refit "
              f"{out['models'][key]['final_epochs']} epochs")

    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {OUT_FILE}")
    return out


if __name__ == "__main__":
    main()
