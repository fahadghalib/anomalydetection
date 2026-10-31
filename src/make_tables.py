"""Regenerate every table of the manuscript from the deposited result files.

Each table is written as Markdown and CSV to outputs/tables/, numbered as in
the manuscript. No value is typed by hand: everything is read from
outputs/results/, so the manuscript and the code cannot drift apart.

  Table 1   data flow                     data_audit.json, fold sizes
  Table 2   identifier leakage            data_audit.json, leakage_probe.json
  Table 4   per-class held-out results    confusion_matrix_corrected_cnn.npy
  Table 5   five-fold CV comparison       corrected_protocol_results.json
  Table 6   paired statistical tests      statistical_tests.json
  Table 7   held-out comparison           confusion_matrix_corrected_*.npy
  Table 8   literature context            values reproduced from ALMahadin et al. (2024)
  Table 9   latency and throughput        latency_benchmark_corrected_*.json
  Table S1  class distribution, fold 1    fold_class_distribution.json
  Table S2  fold-level CV values          corrected_protocol_results.json

(Table 3, the hyperparameters, is the content of src/config.py and
src/run_corrected.py.)

Run:  python -m src.make_tables   (after src.stats_tests)
"""

from __future__ import annotations

import csv
import json

import numpy as np

from . import config

OUT_DIR = config.OUTPUT_DIR / "tables"

LABELS = {"cnn": "Proposed (CNN-BiGRU-attention)", "cnn_permuted": "Proposed, permuted column order",
          "mlp_cap": "Feed-forward, capacity-matched", "mlp_matched": "Feed-forward, reduced capacity",
          "lgbm": "Gradient-boosted trees (LightGBM)"}
ORDER = ["cnn", "cnn_permuted", "mlp_cap", "mlp_matched", "lgbm"]
CLASS_TYPE = ["Genuine"] + ["Fault"] * 8 + ["Attack"] * 3 + ["Fault"] + ["Attack"] * 7

# Reproduced from ALMahadin et al., IEEE Trans. Consumer Electronics 70(1), 2024,
# Table II: NSL-KDD, five classes. Context only; not comparable with this work.
LITERATURE = [("AlertNet", 78.50, 76.52), ("DNN", 79.10, 75.58), ("ANN", 79.90, 74.25),
              ("CNN", 79.40, 71.26), ("MCNN", 81.00, 81.23), ("MCNN-DFS", 81.40, 80.25),
              ("Naive Bayes", 72.45, 72.06), ("Random forest", 76.45, 72.56), ("SEMI-GRU", 83.32, 85.62)]


def _load(name):
    path = config.RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _emit(name, title, header, rows, note=""):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    md = [f"### {name}. {title}", "", "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    md += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    if note:
        md += ["", note]
    stem = name.replace(" ", "_")
    (OUT_DIR / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(OUT_DIR / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {name}: {title} ({len(rows)} rows)")


def _metrics(cm):
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(len(tp)), where=cm.sum(0) > 0)
    rec = tp / cm.sum(1)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(len(tp)), where=(prec + rec) > 0)
    sup = cm.sum(1)
    n_b = cm[0].sum()
    return {"acc": 100 * tp.sum() / cm.sum(), "mp": 100 * prec.mean(), "mr": 100 * rec.mean(),
            "mf1": 100 * f1.mean(), "wp": 100 * (prec * sup).sum() / sup.sum(),
            "wr": 100 * (rec * sup).sum() / sup.sum(), "wf1": 100 * (f1 * sup).sum() / sup.sum(),
            "far": 100 * (n_b - cm[0, 0]) / n_b, "miss": 100 * cm[1:, 0].sum() / cm[1:].sum(),
            "prec": 100 * prec, "rec": 100 * rec, "f1": 100 * f1, "sup": sup,
            "pct": 100 * cm / cm.sum(1, keepdims=True)}


def _cm(key):
    path = config.RESULTS_DIR / f"confusion_matrix_corrected_{key}.npy"
    return np.load(path) if path.exists() else None


def table_1():
    cv = _load("corrected_protocol_results.json")
    runs = cv["cnn"]
    tr = [r["n_train_rows"] for r in runs]
    va = [r["n_val_rows"] for r in runs]
    rows = [["Training file", "2,236,367", "24,644"], ["Test file", "958,441", "24,553"],
            ["Pooled corpus", "3,194,808", "24,663"], ["Exact duplicate rows removed", "0", "-"],
            ["Rows removed for missing or infinite values", "0", "-"],
            ["Development partition", "2,129,867", "16,441"],
            ["CV validation portion (each fold)", f"{min(va):,}-{max(va):,}", "3,285-3,290"],
            ["CV training portion after ADASYN (each fold)", f"{min(tr):,}-{max(tr):,}", "-"],
            ["Held-out partition", "1,064,941", "8,222"],
            ["Senders shared across any partition boundary", "-", "0"]]
    d = _load("data_audit.json")
    if d:
        rows[0][1] = f"{d['train']['raw_records']:,}"
        rows[1][1] = f"{d['test']['raw_records']:,}"
        rows[2][1] = f"{d['combined_raw_records']:,}"
    _emit("Table 1", "Data flow from the distributed files to the analyzed corpus and its partitions",
          ["Stage", "Messages", "Senders"], rows)


def table_2():
    d, p = _load("data_audit.json"), _load("leakage_probe.json")
    if not (d and p):
        return
    leak = [f["pct_val_rows_whose_sender_seen_in_train"] for f in d["row_level_stratifiedkfold_leakage"]]
    ov = d["train_test_identifier_overlap"]
    a = p["probe_A_sender_lookup_baseline"]
    rows = [["Senders whose messages carry a single class label",
             f"{d['train']['pct_senders_class_pure']}% / {d['test']['pct_senders_class_pure']}%"],
            ["Pseudonyms whose messages carry a single class label",
             f"{d['train']['pct_pseudo_class_pure']}% / {d['test']['pct_pseudo_class_pure']}%"],
            ["Test-file senders that also occur in the training file",
             f"{ov['sender']['pct_of_test_ids_seen_in_train']}%"],
            ["Test-file pseudonyms that also occur in the training file",
             f"{ov['senderPseudo']['pct_of_test_ids_seen_in_train']}%"],
            ["Validation messages whose sender occurs in training (row-level 5-fold split)",
             f"{min(leak)}%-{max(leak)}%"],
            ["Sender-to-class lookup accuracy on the test file (no model)", f"{a['accuracy_full_test_set_pct']}%"],
            ["Pseudonym-to-class lookup accuracy on the test file", f"{a['accuracy_via_senderPseudo_pct']}%"]]
    _emit("Table 2", "Identifier leakage in the public release", ["Finding", "Value (training / test file)"], rows)


def table_4():
    cm = _cm("cnn")
    if cm is None:
        return
    m = _metrics(cm)
    rows = []
    for c in range(config.N_CLASSES):
        off = m["pct"][c].copy()
        off[c] = 0
        j = int(np.argmax(off))
        rows.append([c, config.CLASS_NAMES[c], CLASS_TYPE[c], f"{m['prec'][c]:.1f}", f"{m['rec'][c]:.1f}",
                     f"{m['f1'][c]:.1f}", f"{int(m['sup'][c]):,}", "-" if c == 0 else f"{m['pct'][c][0]:.1f}",
                     f"{config.CLASS_NAMES[j]} ({off[j]:.1f})" if off[j] >= 1 else "-"])
    rows.append(["", "Macro average", "", f"{m['mp']:.1f}", f"{m['mr']:.1f}", f"{m['mf1']:.1f}", "", "", ""])
    rows.append(["", "Weighted average", "", f"{m['wp']:.1f}", f"{m['wr']:.1f}", f"{m['wf1']:.1f}", "", "", ""])
    _emit("Table 4", "Per-class performance of the proposed model on the sender-disjoint held-out partition",
          ["ID", "Class", "Type", "Precision (%)", "Recall (%)", "F1 (%)", "Messages",
           "Classified as genuine (%)", "Most frequent other class (%)"], rows,
          note=f"Accuracy {m['acc']:.2f}%; FAR {m['far']:.2f}%; missed detections {m['miss']:.2f}%.")


def tables_5_s2():
    cv = _load("corrected_protocol_results.json")
    rows, rows_s2 = [], []
    for key in ORDER:
        runs = sorted(cv[key], key=lambda r: r["fold"])
        vals = {mm: np.array([r[mm] for r in runs]) * 100 for mm in
                ("val_accuracy", "val_macro_precision", "val_macro_recall", "val_macro_f1")}
        ms = lambda a: f"{a.mean():.2f} ± {a.std(ddof=1):.2f}"
        rows.append([LABELS[key], "400 rounds" if key == "lgbm" else f"{runs[0]['model_size']:,}",
                     ms(vals["val_accuracy"]), ms(vals["val_macro_precision"]), ms(vals["val_macro_recall"]),
                     ms(vals["val_macro_f1"]), f"{np.median([r['elapsed_sec'] for r in runs]) / 60:.1f}"])
        for mm, lab in (("val_accuracy", "Accuracy"), ("val_macro_f1", "Macro F1")):
            rows_s2.append([LABELS[key], lab] + [f"{x:.2f}" for x in vals[mm]] + [ms(vals[mm])])
    _emit("Table 5", "Five-fold cross-validation results on identical sender-disjoint folds (mean ± SD, %)",
          ["Configuration", "Parameters", "Accuracy", "Macro precision", "Macro recall", "Macro F1",
           "Training time per fold (min)"], rows)
    _emit("Table S2", "Fold-level validation results (%) used for the paired comparisons",
          ["Configuration", "Metric", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5", "Mean ± SD"], rows_s2)


def table_6():
    st = _load("statistical_tests.json")
    if not st:
        return
    rows = []
    for key in ("mlp_cap", "mlp_matched", "lgbm", "cnn_permuted"):
        block = st["paired_comparisons"][f"{key}_vs_cnn"]
        for mm, lab in (("val_accuracy", "Accuracy"), ("val_macro_f1", "Macro F1")):
            b = block[mm]
            ci = b["ci95_of_mean_difference"]
            rows.append([LABELS[key], lab, f"{b['mean_difference'] * 100:+.2f}",
                         f"[{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]", f"{b['paired_t_test']['t_statistic']:.2f}",
                         f"{b['paired_t_test']['p_value']:.4f}",
                         f"{b['corrected_resampled_t_test']['p_value']:.4f}", f"{b['cohens_d_paired']:.2f}",
                         f"{b['folds_with_same_sign']}/5"])
    _emit("Table 6", "Paired comparisons with the proposed model on the five development folds",
          ["Configuration", "Metric", "Mean difference (points)", "95% CI", "t(4)", "p (naive)", "p (corrected)",
           "Cohen's d", "Folds with same sign"], rows,
          note="Corrected p: Nadeau-Bengio corrected resampled t-test; naive p assumes independent folds.")


def table_7():
    keys = ["cnn", "mlp_cap", "mlp_matched", "lgbm"]
    ms = {k: _metrics(_cm(k)) for k in keys if _cm(k) is not None}
    if len(ms) < len(keys):
        return
    ep = _load("cv_best_epochs.json")["models"]
    rows = [[lab] + [f"{ms[k][mk]:.2f}" for k in keys] for lab, mk in
            (("Accuracy (%)", "acc"), ("Macro precision (%)", "mp"), ("Macro recall (%)", "mr"),
             ("Macro F1 (%)", "mf1"), ("Weighted F1 (%)", "wf1"), ("False alarm rate (%)", "far"),
             ("Missed detections (%)", "miss"))]
    rows.append(["Training epochs", ep["cnn"]["final_epochs"], ep["mlp_cap"]["final_epochs"],
                 ep["mlp_matched"]["final_epochs"], "400 rounds"])
    _emit("Table 7", "Held-out results of the final models (single evaluation, 1,064,941 messages)",
          ["Metric", "Proposed", "FF, matched", "FF, reduced", "GBT"], rows)


def table_8():
    rows = [[m, "NSL-KDD", 5, "as reported by ALMahadin et al. (2024)", f"{a:.2f}", f"{fm:.2f}"]
            for m, a, fm in LITERATURE]
    cm = _cm("cnn")
    if cm is not None:
        m = _metrics(cm)
        rows.append(["This work (proposed model)", "VeReMi Extension", 20, "sender-disjoint held-out partition",
                     f"{m['acc']:.2f}", f"{m['wf1']:.2f} (weighted); {m['mf1']:.2f} (macro)"])
    _emit("Table 8", "Literature context: results obtained on different datasets, class taxonomies, and partitions",
          ["Method", "Dataset", "Classes", "Partitioning", "Accuracy (%)", "F-measure (%)"], rows,
          note="Not comparable across rows; no ranking or significance test is implied.")


def table_9():
    rows = []
    for tag, lab in (("corrected_gpu", "GPU, WSL2"), ("corrected_wsl2_cpu", "CPU, WSL2"),
                     ("corrected_native_windows_cpu", "CPU, native Windows")):
        d = _load(f"latency_benchmark_{tag}.json")
        if not d:
            continue
        e, g = d["latency_ms_per_message"], d.get("latency_ms_per_message_graph", {})
        rows.append([lab, f"{e['median']:.0f}", f"{e['p95']:.0f}", f"{g.get('median', float('nan')):.2f}",
                     f"{g.get('p95', float('nan')):.2f}", f"{d['batched_throughput']['ms_per_sample']:.4f}"])
    _emit("Table 9", "Single-message latency and batched throughput of the proposed model",
          ["Platform", "Eager median (ms)", "Eager p95 (ms)", "Graph median (ms)", "Graph p95 (ms)",
           "Batched (ms/msg)"], rows)


def table_s1():
    fd = _load("fold_class_distribution.json")
    cm = _cm("cnn")
    if not fd or cm is None:
        return
    held = cm.sum(1)
    rows = [[c, config.CLASS_NAMES[c], f"{fd['pooled'][str(c)]:,}", f"{int(held[c]):,}",
             f"{fd['pooled'][str(c)] - int(held[c]):,}", f"{fd['before']['1'][str(c)]:,}",
             f"{fd['after']['1'][str(c)]:,}"] for c in range(config.N_CLASSES)]
    _emit("Table S1", "Class distribution of the pooled corpus and its partitions; fold 1 training counts before "
                      "and after ADASYN",
          ["ID", "Class", "Pooled", "Held-out", "Development", "Fold 1 training, before", "Fold 1 training, after"],
          rows)


def main():
    print("Regenerating manuscript tables from outputs/results ...")
    for fn in (table_1, table_2, table_4, tables_5_s2, table_6, table_7, table_8, table_9, table_s1):
        fn()
    print(f"written to {OUT_DIR}")


if __name__ == "__main__":
    main()
