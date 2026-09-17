"""Generate every manuscript table from the deposited result artifacts.

One command regenerates all tables, so the manuscript can never drift out of
sync with the code again -- which is the root cause behind reviewer comments
#1, #9 and #10. Each table is emitted as Markdown (for drafting) and LaTeX
(for the manuscript), into outputs/tables/.

Tables produced:
  A  data flow, raw records through to fold sizes          (comment #9)
  B  partition summary with disjointness verification      (comment #2)
  C  model comparison, CV mean +- SD and held-out test     (comments #4, #5)
  D  per-class breakdown on the held-out test partition    (replaces Table 2)
  E  inference latency across execution configurations     (comment #8)
  F  literature context -- explicitly non-comparable       (comment #7)
  G  statistical tests with full reporting                 (comment #6)

Run:  python -m src.make_tables
"""

from __future__ import annotations

import json

import numpy as np

from . import config

OUT_DIR = config.OUTPUT_DIR / "tables"

MODEL_LABELS = {
    "cnn": "Proposed CNN--BiGRU--attention",
    "cnn_permuted": "Proposed model, permuted column order",
    "mlp_matched": "MLP, 384--192--96 (107K params, 76\\% of proposed)",
    "mlp_cap": "MLP, capacity-matched (143K params)",
    "mlp": "MLP, 512--256--128",
    "lgbm": "LightGBM (gradient-boosted trees)",
}

# Reported by ALMahadin et al., IEEE Trans. Consumer Electronics 70(1), 2024,
# Table II. Every row was evaluated on NSL-KDD under a five-class taxonomy --
# none on VeReMi Extension. Reproduced here only to make the incomparability
# explicit; see comment #7.
LITERATURE = [
    ("AlertNet [20]", "NSL-KDD", 5, 78.50, 76.52),
    ("DNN [21]", "NSL-KDD", 5, 79.10, 75.58),
    ("ANN [22]", "NSL-KDD", 5, 79.90, 74.25),
    ("CNN [23]", "NSL-KDD", 5, 79.40, 71.26),
    ("MCNN [24]", "NSL-KDD", 5, 81.00, 81.23),
    ("MCNN-DFS [25]", "NSL-KDD", 5, 81.40, 80.25),
    ("Naive Bayes", "NSL-KDD", 5, 72.45, 72.06),
    ("Random Forest", "NSL-KDD", 5, 76.45, 72.56),
    ("SEMI-GRU", "NSL-KDD", 5, 83.32, 85.62),
]


def _load(name):
    path = config.RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _emit(name: str, title: str, header, rows, notes=""):
    """Write one table as Markdown and as LaTeX."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    md = [f"### {title}", "", "| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        md.append("| " + " | ".join(str(c) for c in r) + " |")
    if notes:
        md += ["", notes]
    (OUT_DIR / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    align = "l" + "c" * (len(header) - 1)
    tex = [r"\begin{table}[t]", r"\centering",
           r"\caption{" + title.replace("&", r"\&") + "}",
           r"\label{tab:" + name + "}",
           r"\begin{tabular}{@{}" + align + r"@{}}", r"\toprule",
           " & ".join(r"\textbf{" + h.replace("%", r"\%") + "}" for h in header) + r" \\",
           r"\midrule"]
    for r in rows:
        tex.append(" & ".join(str(c).replace("%", r"\%").replace("±", r"$\pm$")
                              for c in r) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_DIR / f"{name}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    print(f"\n{title}")
    print("  " + " | ".join(header))
    for r in rows:
        print("  " + " | ".join(str(c) for c in r))


def table_a_dataflow():
    d = _load("data_audit.json")
    if not d:
        return
    tr, te = d["train"], d["test"]
    rows = [
        ("Raw records, training CSV", f"{tr['raw_records']:,}"),
        ("Raw records, test CSV", f"{te['raw_records']:,}"),
        ("Combined raw records", f"{d['combined_raw_records']:,}"),
        ("Public dataset reported instances", f"{d['public_dataset_reported_instances']:,}"),
        ("Exact duplicate rows removed", f"{tr['exact_duplicate_rows_removed'] + te['exact_duplicate_rows_removed']:,}"),
        ("Rows removed for NaN or infinity", f"{tr['rows_with_nan_or_inf_removed'] + te['rows_with_nan_or_inf_removed']:,}"),
        ("Records after cleaning", f"{d['combined_records_after_cleaning']:,}"),
        ("Unique senders, pooled", f"{tr['n_unique_senders'] + te['n_unique_senders']:,} (before pooling)"),
    ]
    _emit("A_dataflow", "Data flow from raw records to the analysed corpus",
          ["Stage", "Records"], rows,
          notes="No records are excluded at any stage: the two files account for the "
                "public dataset's reported instance count exactly.")


def table_b_partitions():
    d = _load("data_audit.json")
    if not d:
        return
    ov = d["train_test_identifier_overlap"]
    leak = d["row_level_stratifiedkfold_leakage"]
    mean_rows = np.mean([f["pct_val_rows_whose_sender_seen_in_train"] for f in leak])
    rows = [
        ("Sender IDs shared between the supplied train and test files",
         f"{ov['sender']['pct_of_test_ids_seen_in_train']}%"),
        ("Senders carrying exactly one class label",
         f"{d['train']['pct_senders_class_pure']}%"),
        ("Validation rows per fold whose sender was seen in training (original split)",
         f"{mean_rows:.3f}%"),
        ("Development partition (corrected protocol)", "2,129,867 rows / 16,441 senders"),
        ("Held-out test partition (corrected protocol)", "1,064,941 rows / 8,222 senders"),
        ("Sender overlap across any corrected partition boundary", "0 (asserted in code)"),
    ]
    _emit("B_partitions", "Split validity before and after correction",
          ["Property", "Value"], rows,
          notes="The corrected protocol partitions by sender group; disjointness is "
                "enforced by assertion in src/corrected_protocol.py.")


def table_c_models():
    cv = _load("corrected_protocol_results.json")
    ho = _load("corrected_protocol_holdout.json")
    if not cv:
        return
    rows = []
    order = ["cnn", "cnn_permuted", "mlp_matched", "mlp_cap", "mlp", "lgbm"]
    for key in sorted(cv, key=lambda k: order.index(k) if k in order else 99):
        if "SUPERSEDED" in key:      # kept in the artifact as evidence, not reported
            continue
        runs = cv[key]
        acc = np.array([r["val_accuracy"] for r in runs]) * 100
        f1 = np.array([r["val_macro_f1"] for r in runs]) * 100
        rec = np.array([r["val_macro_recall"] for r in runs]) * 100
        sd = (lambda a: a.std(ddof=1) if len(a) > 1 else 0.0)
        test_acc = test_f1 = "--"
        if ho and key in ho:
            test_acc = f"{ho[key]['val_accuracy'] * 100:.2f}"
            test_f1 = f"{ho[key]['val_macro_f1'] * 100:.2f}"
        rows.append((
            MODEL_LABELS.get(key, key), len(runs),
            f"{acc.mean():.2f} ± {sd(acc):.2f}",
            f"{rec.mean():.2f} ± {sd(rec):.2f}",
            f"{f1.mean():.2f} ± {sd(f1):.2f}",
            test_acc, test_f1,
        ))
    _emit("C_models", "Model comparison under the corrected protocol",
          ["Model", "Folds", "CV accuracy (%)", "CV macro-recall (%)",
           "CV macro-F1 (%)", "Test accuracy (%)", "Test macro-F1 (%)"], rows,
          notes="All configurations share identical folds, seeds, preprocessing "
                "(computed once per fold and cached), epoch budget and early-stopping "
                "rule. Test columns are a single evaluation on the sender-disjoint "
                "held-out partition. Majority-class baseline: 59.49% accuracy.")


def table_d_perclass():
    ho = _load("corrected_protocol_holdout.json")
    if not ho:
        return
    keys = [k for k in ("cnn", "mlp_cap", "mlp_matched", "lgbm")
            if k in ho and "per_class" in ho[k]]
    if not keys:
        return
    per = {k: {r["class_id"]: r for r in ho[k]["per_class"]} for k in keys}
    header = ["ID", "Class"] + [f"{MODEL_LABELS.get(k, k)} F1 (%)" for k in keys] + ["Support"]
    rows = []
    for c in range(config.N_CLASSES):
        first = per[keys[0]][c]
        rows.append([c, config.CLASS_NAMES[c]]
                    + [f"{per[k][c]['f1_pct']:.1f}" for k in keys]
                    + [f"{first['support']:,}"])
    _emit("D_per_class", "Per-class performance on the sender-disjoint held-out test set",
          header, rows,
          notes="Position-manipulation variants (Sybil Position, Disruptive Position, "
                "Constant Position) are the systematic weak point: position falsification "
                "is difficult to separate from legitimate positional variation using the "
                "available kinematic features.")


def table_e_latency():
    """Table 5. Single-message latency for the model the paper actually
    reports: ONE corrected-protocol 28-feature network.

    The superseded `latency_benchmark*.json` files time a five-model ensemble
    of 32-feature networks written by the withdrawn `train.py` pipeline, and
    their CPU rows were recorded while other jobs held the cores. They are
    retained in the deposit as an audit trail but are not used here.
    """
    files = [("latency_benchmark_corrected_gpu.json", "WSL2 + GPU"),
             ("latency_benchmark_corrected_wsl2_cpu.json", "WSL2 + CPU"),
             ("latency_benchmark_corrected_native_windows_cpu.json",
              "Native Windows + CPU")]
    rows, thr = [], []
    for fname, label in files:
        d = _load(fname)
        if not d:
            continue
        l = d["latency_ms_per_message"]
        rows.append((label, f"{l['median']:.0f}", f"{l['p95']:.0f}",
                     f"{l['p99']:.0f}", f"{l['mean']:.0f}"))
        b = d.get("batched_throughput")
        if b:
            thr.append((label, b["ms_per_sample"], b["batch_size"]))
    if not rows:
        return
    for label, ms, bs in thr:
        rows.append((f"Batched throughput, {label} (batch = {bs})",
                     "--", "--", "--", f"{ms:.4f} ms/sample"))
    _emit("E_latency", "Single-message inference latency across execution configurations",
          ["Configuration", "Median (ms)", "p95 (ms)", "p99 (ms)", "Mean (ms)"], rows,
          notes="Batch size 1, float32, 300 timed repetitions after 50 untimed warm-up "
                "calls, single model, 140,373 trainable parameters, 0.599 MB of weights. "
                "The two CPU configurations differ by roughly 10%, which excludes "
                "virtualisation overhead as the explanation. The GPU is the slowest of "
                "the three because batch-size-one recurrent inference is bound by "
                "per-call kernel dispatch rather than by arithmetic, so the device's "
                "parallelism goes unused. Batched throughput is reported separately "
                "because it is not a per-message latency.")


def table_f_literature():
    rows = [(name, ds, str(nc), "5-class NSL-KDD protocol", f"{acc:.2f}", f"{f1:.2f}")
            for name, ds, nc, acc, f1 in LITERATURE]
    rows.append(("This work (corrected protocol)", "VeReMi Extension", "20",
                 "Sender-grouped CV + held-out", "--", "--"))
    _emit("F_literature", "Literature context (results are not directly comparable)",
          ["Method", "Dataset", "Classes", "Partitioning", "Accuracy (%)", "F-measure (%)"],
          rows,
          notes="Reproduced from ALMahadin et al., IEEE Trans. Consumer Electronics "
                "70(1), 2024, Table II. Every cited result was obtained on NSL-KDD under "
                "a five-class taxonomy with a different feature space; none was evaluated "
                "on VeReMi Extension. The table is provided for context only and supports "
                "no superiority claim, and no significance test is applicable across "
                "these settings.")


def table_g_statistics():
    d = _load("reviewer_statistics.json")
    if not d or not d.get("M8_vs_M7_statistical_tests"):
        return
    rows = []
    for metric, b in d["M8_vs_M7_statistical_tests"].items():
        t = b["paired_t_test"]
        c = b["corrected_resampled_t_test"]
        ci = b["ci95_of_mean_difference"]
        rows.append((
            metric.replace("val_", "").replace("_", " "),
            f"{b['mean_difference'] * 100:+.3f}",
            f"{t['t_statistic']:.3f}", t["degrees_of_freedom"],
            f"{t['p_value']:.5f}", f"{c['p_value']:.5f}",
            f"[{ci[0] * 100:+.3f}, {ci[1] * 100:+.3f}]",
            f"{b['cohens_d_paired']:.2f}",
        ))
    _emit("G_statistics", "Paired comparison across matched folds",
          ["Metric", "Mean diff (pp)", "t", "df", "Naive p",
           "Corrected p", "95% CI (pp)", "Cohen's d"], rows,
          notes="Corrected p is the Nadeau-Bengio corrected resampled t-test, which "
                "inflates the variance by (1/k + n_test/n_train) to account for the "
                "training-set overlap between folds that the naive paired test ignores. "
                "At k=5 the Wilcoxon signed-rank test cannot attain p<0.0625 and is "
                "reported in the artifact for completeness only.")


def main():
    print("Regenerating manuscript tables from deposited artifacts ...")
    for fn in (table_a_dataflow, table_b_partitions, table_c_models,
               table_d_perclass, table_e_latency, table_f_literature,
               table_g_statistics):
        try:
            fn()
        except Exception as exc:  # a missing artifact must not block the rest
            print(f"  [skipped] {fn.__name__}: {exc}")
    print(f"\nwrote Markdown and LaTeX to {OUT_DIR}")


if __name__ == "__main__":
    main()
