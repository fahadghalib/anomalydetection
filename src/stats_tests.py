"""Statistical comparisons and false alarm rates reported in the manuscript.

Paired comparisons (Table 6). Every configuration is compared with the
proposed model on the five matched sender-disjoint development folds of
`outputs/results/corrected_protocol_results.json`. For each metric the
following are reported: per-fold values and paired differences, mean
difference with its 95% confidence interval, the naive paired t-test, the
Nadeau-Bengio corrected resampled t-test, Cohen's d for paired samples, and
the Wilcoxon signed-rank test.

The naive paired t-test treats the k fold differences as independent. The
folds share most of their training data, so the variance of the mean
difference is under-estimated and the test is anti-conservative. The corrected
test (Nadeau & Bengio, Machine Learning 52:239-281, 2003) multiplies the
variance by (1/k + n_test/n_train) and is the one used for inference. With
k = 5 the Wilcoxon test cannot reach a two-sided p below 0.0625; it is
reported for completeness only.

False alarm rate (Section 3.6). With attack = positive (classes 1-19) and
genuine = negative (class 0), FAR = FP / (FP + TN) = 1 - recall(genuine),
computed from each held-out confusion matrix.

Run:  python -m src.stats_tests   ->  outputs/results/statistical_tests.json
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

from . import config

METRICS = ["val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1"]
BASELINE = "cnn"
COMPARED = ["cnn_permuted", "mlp_cap", "mlp_matched", "lgbm"]


def far_from_confusion_matrix(cm: np.ndarray) -> dict:
    """Binary view of a 20-class confusion matrix (rows = true, columns = predicted)."""
    n_benign = int(cm[0].sum())
    tn = int(cm[0, 0])
    fp = n_benign - tn
    n_attack = int(cm[1:].sum())
    fn = int(cm[1:, 0].sum())          # misbehaving messages classified as genuine
    return {
        "convention": "positive = fault or attack (classes 1-19), negative = genuine (class 0)",
        "TN_genuine_correct": tn,
        "FP_genuine_flagged": fp,
        "FN_misbehavior_missed": fn,
        "TP_misbehavior_flagged": n_attack - fn,
        "false_alarm_rate_pct": 100.0 * fp / n_benign,
        "genuine_recall_pct": 100.0 * tn / n_benign,
        "missed_detection_rate_pct": 100.0 * fn / n_attack,
    }


def corrected_t(diffs: np.ndarray, n_train: float, n_test: float) -> dict:
    k = len(diffs)
    correction = 1.0 / k + n_test / n_train
    t_stat = diffs.mean() / np.sqrt(diffs.var(ddof=1) * correction)
    return {"t_statistic": float(t_stat), "degrees_of_freedom": k - 1,
            "p_value": float(2 * stats.t.sf(abs(t_stat), k - 1)),
            "variance_correction_factor": float(correction)}


def compare(a: np.ndarray, b: np.ndarray, n_train: float, n_test: float) -> dict:
    """Paired comparison of b against a (b minus a), values as fractions."""
    d = b - a
    k = len(d)
    t_stat, p_val = stats.ttest_rel(b, a)
    sd = d.std(ddof=1)
    half = stats.t.ppf(0.975, k - 1) * sd / np.sqrt(k)
    try:
        w = stats.wilcoxon(b, a)
        wilcoxon = {"statistic": float(w.statistic), "p_value": float(w.pvalue)}
    except ValueError as exc:
        wilcoxon = {"error": str(exc)}
    return {
        "baseline_per_fold": a.tolist(), "compared_per_fold": b.tolist(),
        "paired_differences": d.tolist(),
        "mean_difference": float(d.mean()), "sd_of_differences": float(sd),
        "ci95_of_mean_difference": [float(d.mean() - half), float(d.mean() + half)],
        "folds_with_same_sign": int(np.sum(np.sign(d) == np.sign(d.mean()))),
        "paired_t_test": {"t_statistic": float(t_stat), "degrees_of_freedom": k - 1, "p_value": float(p_val)},
        "corrected_resampled_t_test": corrected_t(d, n_train, n_test),
        "cohens_d_paired": float(d.mean() / sd),
        "wilcoxon_signed_rank": wilcoxon,
    }


def run():
    with open(config.RESULTS_DIR / "corrected_protocol_results.json") as f:
        cv = json.load(f)
    base = {r["fold"]: r for r in cv[BASELINE]}
    n_train = float(np.mean([r["n_train_rows"] for r in base.values()]))
    n_test = float(np.mean([r["n_val_rows"] for r in base.values()]))

    tests = {}
    for key in COMPARED:
        other = {r["fold"]: r for r in cv[key]}
        folds = sorted(set(base) & set(other))
        tests[f"{key}_vs_{BASELINE}"] = {
            m: compare(np.array([base[i][m] for i in folds]), np.array([other[i][m] for i in folds]),
                       n_train, n_test)
            for m in METRICS}
        acc = tests[f"{key}_vs_{BASELINE}"]["val_accuracy"]
        print(f"{key:13s} - {BASELINE}: accuracy {acc['mean_difference'] * 100:+.2f} points, "
              f"naive p = {acc['paired_t_test']['p_value']:.4f}, "
              f"corrected p = {acc['corrected_resampled_t_test']['p_value']:.4f}")

    far = {}
    for path in sorted(config.RESULTS_DIR.glob("confusion_matrix_corrected_*.npy")):
        key = path.stem.replace("confusion_matrix_corrected_", "")
        far[key] = far_from_confusion_matrix(np.load(path))
        print(f"{key:13s} held-out FAR = {far[key]['false_alarm_rate_pct']:.3f}%")

    out = {"n_train_per_fold_mean": n_train, "n_validation_per_fold_mean": n_test,
           "paired_comparisons": tests, "held_out_binary_view": far}
    path = config.RESULTS_DIR / "statistical_tests.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {path}")
    return out


if __name__ == "__main__":
    run()
