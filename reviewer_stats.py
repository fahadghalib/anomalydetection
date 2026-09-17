"""Statistical analysis artifacts (Reviewer comments #3 and #6).

Comment #3 -- False Alarm Rate. Reconstructs FAR directly from the saved
confusion matrix under an explicit, stated positive/negative convention
(attack = positive, benign = negative), and reports both possible
orientations so the manuscript can no longer be ambiguous about which one it
means.

Comment #6 -- Statistical testing. The response letter claimed a paired
t-test but supplied no test statistic, degrees of freedom, exact p-value,
confidence interval, effect size, or fold-level differences, and no artifact
was deposited. This module emits all of them, plus two additional analyses
the reviewer's objection calls for:

  * Nadeau-Bengio corrected resampled t-test. The standard paired t-test
    assumes independent observations, but k-fold estimates share training
    data and are positively correlated, so the naive test is anti-conservative
    (it under-estimates the variance and over-states significance). The
    correction inflates the variance by (1/k + n_test/n_train).
  * Wilcoxon signed-rank test, a non-parametric alternative that does not
    assume normality -- reported for completeness, while noting that with
    k=5 its minimum attainable two-sided p-value is 0.0625, so it cannot
    reach p<0.05 at this fold count regardless of effect size.

Run:  python -m src.reviewer_stats
"""

from __future__ import annotations

import json

import numpy as np
from scipy import stats

from . import config

# Fold-level validation results actually produced by this project's runs.
# Sources: outputs/results/fold_summary.json (M7),
#          outputs/results/mlp_baseline_5fold_summary.json (M8),
#          outputs/results/mlp_no_temporal_5fold_summary.json (M9).
FOLD_SUMMARY_FILES = {
    "M7_proposed": "fold_summary.json",
    "M8_mlp_temporal": "mlp_baseline_5fold_summary.json",
    "M9_mlp_no_temporal": "mlp_no_temporal_5fold_summary.json",
}

METRICS = ["val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1"]


# ---------------------------------------------------------------------------
# Comment #3: False Alarm Rate
# ---------------------------------------------------------------------------
def far_from_confusion_matrix(cm: np.ndarray, benign_class: int = 0) -> dict:
    """Reconstruct FAR from the confusion matrix under an explicit convention.

    Rows = actual, columns = predicted (sklearn convention).

    Operational convention for an IDS (the one the manuscript should use):
        positive = ATTACK (any of classes 1..19)
        negative = BENIGN (class 0)
        FP = benign message classified as some attack class
        TN = benign message classified as benign
        FAR = FP / (FP + TN) = FP / (all actual benign) = 1 - recall(benign)
    """
    total = cm.sum()
    actual_benign = cm[benign_class, :].sum()
    tn = cm[benign_class, benign_class]
    fp = actual_benign - tn

    actual_attack = total - actual_benign
    predicted_benign_total = cm[:, benign_class].sum()
    fn = predicted_benign_total - tn          # attacks misclassified as benign
    tp = actual_attack - fn                   # attacks classified as some attack

    far = fp / actual_benign if actual_benign else 0.0
    benign_recall = tn / actual_benign if actual_benign else 0.0

    # The inverted (incorrect for an IDS) orientation, reported so the
    # manuscript's current number can be traced to its source.
    inverted_far = fn / actual_attack if actual_attack else 0.0

    return {
        "convention": "positive = attack (classes 1-19), negative = benign (class 0)",
        "confusion_counts": {
            "TP_attack_detected_as_attack": int(tp),
            "FN_attack_missed_as_benign": int(fn),
            "FP_benign_flagged_as_attack": int(fp),
            "TN_benign_correct": int(tn),
        },
        "n_actual_benign": int(actual_benign),
        "n_actual_attack": int(actual_attack),
        "benign_recall_pct": round(100.0 * benign_recall, 3),
        "FAR_pct_correct_orientation": round(100.0 * far, 3),
        "identity_check_far_equals_1_minus_benign_recall": round(
            100.0 * (1.0 - benign_recall), 3
        ),
        "miss_rate_pct_inverted_orientation_NOT_far": round(100.0 * inverted_far, 3),
    }


# ---------------------------------------------------------------------------
# Comment #6: statistical testing on k-fold results
# ---------------------------------------------------------------------------
def nadeau_bengio_corrected_t(diffs: np.ndarray, n_train: int, n_test: int) -> dict:
    """Corrected resampled t-test (Nadeau & Bengio, 2003).

    The naive paired t-test treats the k fold-differences as independent, but
    the folds share training data, so the variance of their mean is
    under-estimated. The correction multiplies the variance estimate by
    (1/k + n_test/n_train).
    """
    k = len(diffs)
    mean_d = float(diffs.mean())
    var_d = float(diffs.var(ddof=1))
    correction = (1.0 / k) + (n_test / n_train)
    denom = np.sqrt(var_d * correction)
    t_stat = mean_d / denom if denom > 0 else float("nan")
    df = k - 1
    p_value = float(2 * (1 - stats.t.cdf(abs(t_stat), df)))
    return {
        "test": "Nadeau-Bengio corrected resampled t-test",
        "t_statistic": float(t_stat),
        "degrees_of_freedom": df,
        "p_value": p_value,
        "variance_correction_factor": float(correction),
        "n_train_per_fold": int(n_train),
        "n_test_per_fold": int(n_test),
    }


def compare(a_name: str, a: np.ndarray, b_name: str, b: np.ndarray,
            n_train: int, n_test: int) -> dict:
    """Full paired comparison of b vs a across matched folds."""
    diffs = b - a
    k = len(diffs)
    df = k - 1

    t_stat, p_val = stats.ttest_rel(b, a)

    mean_d = float(diffs.mean())
    sd_d = float(diffs.std(ddof=1))
    se_d = sd_d / np.sqrt(k)
    t_crit = float(stats.t.ppf(0.975, df))
    ci_low, ci_high = mean_d - t_crit * se_d, mean_d + t_crit * se_d

    # Cohen's d for paired samples (mean difference / SD of differences)
    cohens_d = mean_d / sd_d if sd_d > 0 else float("nan")

    try:
        w_stat, w_p = stats.wilcoxon(b, a)
        wilcoxon = {"statistic": float(w_stat), "p_value": float(w_p),
                    "note": "with k=5 folds the minimum attainable two-sided p is 0.0625"}
    except ValueError as exc:
        wilcoxon = {"error": str(exc)}

    return {
        "comparison": f"{b_name} minus {a_name}",
        "n_folds": k,
        f"{a_name}_per_fold": [float(v) for v in a],
        f"{b_name}_per_fold": [float(v) for v in b],
        "fold_level_paired_differences": [float(v) for v in diffs],
        f"{a_name}_mean": float(a.mean()),
        f"{a_name}_sd": float(a.std(ddof=1)),
        f"{b_name}_mean": float(b.mean()),
        f"{b_name}_sd": float(b.std(ddof=1)),
        "mean_difference": mean_d,
        "sd_of_differences": sd_d,
        "paired_t_test": {
            "test": "paired two-sided t-test (scipy.stats.ttest_rel)",
            "t_statistic": float(t_stat),
            "degrees_of_freedom": df,
            "p_value": float(p_val),
            "caveat": "assumes independent folds; k-fold folds share training data, "
                      "so this test is anti-conservative -- see corrected test below",
        },
        "corrected_resampled_t_test": nadeau_bengio_corrected_t(diffs, n_train, n_test),
        "wilcoxon_signed_rank": wilcoxon,
        "ci95_of_mean_difference": [float(ci_low), float(ci_high)],
        "cohens_d_paired": float(cohens_d),
    }


def corrected_protocol_tests() -> dict:
    """Paired comparisons on the CORRECTED protocol's grouped folds.

    These supersede the tests computed on the original row-level split, which
    are withdrawn: those were run on leaky measurements and establish only
    that two contaminated numbers differ.
    """
    path = config.RESULTS_DIR / "corrected_protocol_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        runs = json.load(f)

    by_model = {k: sorted(v, key=lambda r: r["fold"]) for k, v in runs.items()}
    complete = {k: v for k, v in by_model.items() if len(v) >= 2}

    # Compare every model against the proposed one where both are available.
    baseline = "cnn"
    if baseline not in complete:
        return {}
    out = {}
    for key, v in complete.items():
        if key == baseline:
            continue
        folds_a = {r["fold"]: r for r in complete[baseline]}
        folds_b = {r["fold"]: r for r in v}
        shared = sorted(set(folds_a) & set(folds_b))
        if len(shared) < 2:
            continue
        n_train = int(np.mean([folds_a[f]["n_train_rows"] for f in shared]))
        n_test = int(np.mean([folds_a[f]["n_val_rows"] for f in shared]))
        block = {}
        for metric in METRICS:
            a = np.array([folds_a[f][metric] for f in shared])
            b = np.array([folds_b[f][metric] for f in shared])
            block[metric] = compare(baseline, a, key, b, n_train, n_test)
        out[f"{key}_vs_{baseline}"] = block
        acc = block["val_accuracy"]
        print(f"[corrected] {key} vs {baseline}: acc diff="
              f"{acc['mean_difference'] * 100:+.3f}pp  naive p={acc['paired_t_test']['p_value']:.5f}  "
              f"corrected p={acc['corrected_resampled_t_test']['p_value']:.5f}  "
              f"(n={len(shared)} folds)", flush=True)
    return out


def run():
    results = {}

    # ---- Comment #3: FAR ----
    cm_files = {
        "M7_proposed": "confusion_matrix.npy",
        "M8_mlp_temporal": "confusion_matrix_mlp_baseline.npy",
        "M9_mlp_no_temporal": "confusion_matrix_mlp_no_temporal.npy",
    }
    far_block = {}
    for name, fname in cm_files.items():
        path = config.RESULTS_DIR / fname
        if not path.exists():
            continue
        cm = np.load(path)
        far_block[name] = far_from_confusion_matrix(cm)
        print(f"[FAR] {name}: benign recall={far_block[name]['benign_recall_pct']}%  "
              f"FAR={far_block[name]['FAR_pct_correct_orientation']}%", flush=True)
    results["false_alarm_rate"] = far_block

    # ---- Comment #6: statistical tests ----
    folds = {}
    for name, fname in FOLD_SUMMARY_FILES.items():
        path = config.RESULTS_DIR / fname
        if not path.exists():
            print(f"  (missing {fname}, skipping {name})", flush=True)
            continue
        with open(path) as f:
            folds[name] = json.load(f)

    stat_block = {}
    if "M7_proposed" in folds and "M8_mlp_temporal" in folds:
        m7, m8 = folds["M7_proposed"], folds["M8_mlp_temporal"]
        n_train = int(np.mean([f["n_train_resampled"] for f in m7 if f.get("n_train_resampled")]))
        n_test = int(np.mean([f["n_val"] for f in m7]))
        for metric in METRICS:
            a = np.array([f[metric] for f in m7])
            b = np.array([f[metric] for f in m8])
            block = compare("M7_proposed", a, "M8_mlp_temporal", b, n_train, n_test)
            stat_block[metric] = block
            print(f"[{metric}] M8-M7 diff={block['mean_difference']*100:+.3f}pp  "
                  f"naive p={block['paired_t_test']['p_value']:.5f}  "
                  f"corrected p={block['corrected_resampled_t_test']['p_value']:.5f}", flush=True)
    results["M8_vs_M7_statistical_tests_WITHDRAWN_leaky_protocol"] = stat_block

    # ---- corrected protocol: the tests that stand ----
    print("\n--- corrected protocol (grouped folds, no identifier features) ---", flush=True)
    results["corrected_protocol_tests"] = corrected_protocol_tests()

    out_path = config.RESULTS_DIR / "reviewer_statistics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out_path}")
    return results


if __name__ == "__main__":
    run()
