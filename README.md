# Twenty-class misbehavior detection on VeReMi Extension: code and results

This repository contains the complete code, configuration, result files, and
trained models for the manuscript

> **Network Traffic Anomaly Detection in VANETs Using a Hybrid Deep Learning
> Model**
> F. G. Abdulkadhim, F. Tahseen, A. A. Fadhil, H. A. Hashim, and C. Tang.
> *Journal of Computational and Cognitive Engineering* (under review, manuscript 9314).

Release **v2.0.1** is the version that corresponds to the revised manuscript.
Every number, table, and figure in the manuscript is produced by the code in
`src/` from the files in `outputs/results/`; the table at the end of this
README lists the command and output for each of them.

## What the experiment does

* **Data.** The public tabular release of VeReMi Extension (two CSV files,
  3,194,808 messages, 20 classes).
* **Leakage audit.** Every sender identifier maps to one class, and 99.92% of
  test-file senders also occur in the training file; a sender-to-class lookup
  table reaches 99.999% accuracy with no model (`src/data_audit.py`,
  `src/leakage_probe.py`).
* **Protocol.** Identifier columns are removed; 24 kinematic columns plus 4
  per-sender temporal features (28 inputs) are used. Both files are pooled and
  re-partitioned **by sender**: a held-out partition of 8,222 senders and five
  sender-disjoint folds on the remaining 16,441 senders. Disjointness is
  checked by assertions (`src/corrected_protocol.py`).
* **Models.** The proposed multi-scale CNN + BiGRU + additive attention, the
  same model on permuted columns, a capacity-matched and a reduced-capacity
  feed-forward network, and LightGBM, all trained on identical cached folds
  with the same budget and seeds (`src/run_corrected.py`).
* **Held-out evaluation.** Each final model is retrained on the whole
  development partition for a fixed number of epochs taken from
  cross-validation (`src/cv_epochs.py`) and evaluated once. No ensembling is
  used anywhere.

## Environment

| | |
|---|---|
| Python | 3.11.15 |
| TensorFlow / Keras | 2.21.0 / 3.15.1 |
| Other packages | see `requirements.txt` (direct) and `requirements-lock.txt` (full freeze) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB |
| CPU | Intel Core i9-13900H |
| OS | WSL2 (Ubuntu) on Windows 11 |
| Precision | float32 |
| Seeds | 42 for partitioning and the column permutation; 42 + fold index for resampling and weight initialization |

TensorFlow 2.21 has no native-Windows GPU build, so GPU runs used WSL2 with
`tensorflow[and-cuda]`. The native-Windows latency row uses the CPU build.

```bash
conda create -n veremi -c conda-forge python=3.11
conda activate veremi
pip install -r requirements-lock.txt     # exact environment (Linux/WSL2, CUDA 12)
# or: pip install -r requirements.txt    # direct dependencies only
python verify_deposit.py                 # checks that every imported module is present
```

GPU kernels in TensorFlow are not bit-deterministic, so retraining can change
accuracy by a few tenths of a percentage point. The deposited predictions
(`outputs/results/holdout_predictions_*.npz`) and confusion matrices
reproduce every reported table exactly.

## Data

Download `mixalldata_clean.tar.xz` from the Mendeley Data release
*VeReMi_Extension: Dataset for Misbehaviors in VANETs* (version 1,
doi:[10.17632/k62n4z9gdz.1](https://doi.org/10.17632/k62n4z9gdz.1)), extract
`mixalldata_clean.csv` into `data/`, and run `python -m src.prepare_data` to
recreate the training and test files of the 70/30 row-level split that the
scripts read. See `data/README.md` for checksums and details.

## Reproducing the results

`reproduce_all.sh` runs the whole pipeline in order. Two modes are provided:

```bash
bash reproduce_all.sh tables   # regenerate all tables and figures from the deposited results (minutes, no GPU)
bash reproduce_all.sh full     # re-run every experiment from the CSV files (about 2 days on one RTX 4060)
```

In `full` mode the deposited results are moved to `outputs/results_deposited/`
first, so that nothing is reused. Preprocessing for each fold (scaler and
ADASYN) is computed once and cached in `data/cache/`, and every model trains on
the same cached arrays.

### Command for every table and figure

| Manuscript item | Command | Output |
|---|---|---|
| Table 1 (data flow), Table 2 (leakage audit) | `python -m src.data_audit` and `python -m src.leakage_probe` | `data_audit.json`, `leakage_probe.json` |
| Partition sizes and disjointness check | `python -m src.corrected_protocol` | printed |
| Table 5, Table S2 (five-fold CV) | `python -m src.run_corrected --models cnn,cnn_permuted,mlp_cap,mlp_matched,lgbm --folds 5` | `corrected_protocol_results.json` |
| Convergence check (Section 3.5) | `python -m src.run_corrected --models cnn --only-fold 1 --epochs 100 --patience 20 --out-suffix _cnn_convergence` | `corrected_protocol_results_cnn_convergence.json` |
| Epoch budget of the final refit | `python -m src.cv_epochs` | `cv_best_epochs.json` |
| Table 4, Table 7, Figure 5 (held-out) | `python -m src.run_corrected --models mlp_cap,mlp_matched,cnn,lgbm --eval-test` | `corrected_protocol_holdout.json`, `confusion_matrix_corrected_*.npy`, `holdout_predictions_*.npz`, `outputs/models/final_*.keras` |
| Table 6 (statistical tests), FAR | `python -m src.stats_tests` | `statistical_tests.json` |
| Table 9 (latency), GPU under WSL2 | `python -m src.benchmark_latency_corrected --device gpu --tag corrected_gpu` | `latency_benchmark_corrected_gpu.json` |
| Table 9, CPU under WSL2 | `python -m src.benchmark_latency_corrected --device cpu --tag corrected_wsl2_cpu` | `latency_benchmark_corrected_wsl2_cpu.json` |
| Table 9, CPU under native Windows | `python -m src.benchmark_latency_corrected --device cpu --tag corrected_native_windows_cpu` | `latency_benchmark_corrected_native_windows_cpu.json` |
| Figure 5 (confusion matrix) | `python -m src.plot_confusion` | `outputs/figures/confusion_corrected_cnn.png` |
| Figures 1 and 2 (protocol, architecture) | `python -m src.plot_diagrams` | `outputs/figures/fig_protocol.png`, `fig_architecture.png` |
| All tables as Markdown/CSV | `python -m src.make_tables` | `outputs/tables/` |
| Table S1 (class counts per fold) | `python -m src.fold_distribution` | `fold_class_distribution.json` |

Figures 3 and 4 (BiGRU and GRU schematics) are static drawings in
`outputs/figures/static/`. Table 3 (hyperparameters) is the content of
`src/config.py` and `src/run_corrected.py`; Table 8 reproduces published
values (`src/make_tables.py`, `LITERATURE`).

Run the latency benchmark on an otherwise idle machine; concurrent jobs
inflate the CPU figures several-fold. On native Windows the script opts its
own process out of execution-speed power throttling (EcoQoS), which Windows 11
applies to processes without a foreground window and which slowed the eager
forward pass about seven-fold on the development laptop; the setting is
recorded under `environment.windows_power_throttling` in the result file.

## Definitions used in the manuscript

* **Classes.** The `class` column follows the attacker enumeration of the
  F2MD framework that generated VeReMi Extension (0 Genuine, 1 Constant
  Position, 2 Constant Position Offset, 3 Random Position, 4 Random Position
  Offset, 5 Constant Speed, 6 Constant Speed Offset, 7 Random Speed, 8 Random
  Speed Offset, 9 Eventual Stop, 10 Disruptive, 11 Data Replay, 12 Delayed
  Messages, 13 DoS, 14 DoS Random, 15 DoS Disruptive, 16 Grid Sybil, 17 Data
  Replay Sybil, 18 DoS Random Sybil, 19 DoS Disruptive Sybil); see
  `src/config.py`.
* **False alarm rate.** Positive = any fault or attack class (1-19), negative
  = genuine (0). FAR = FP / (FP + TN) = 1 - recall(genuine), computed from the
  confusion matrix (`src/stats_tests.py`).
* **Statistical test.** Nadeau-Bengio corrected resampled t-test on the five
  matched folds, df = 4; the naive paired t-test, 95% CI, Cohen's d, and the
  fold-level differences are reported alongside it.
* **Latency.** Single-message latency at batch size 1 (50 warm-up and 300
  timed calls), in eager mode and with the forward pass compiled by
  `tf.function`; batched throughput is reported separately.

## Repository layout

```
src/                     authoritative pipeline (Python package)
  config.py              all settings (Table 3)
  prepare_data.py        the two split files, from the Mendeley release
  data_audit.py          provenance and leakage audit
  leakage_probe.py       identifier lookup baseline
  data_pipeline.py       cleaning and per-sender temporal features
  corrected_protocol.py  identifier removal and sender-disjoint partitions
  sampling.py            training-fold ADASYN
  model.py               attention layer, focal loss, class weights, optimizer
  run_corrected.py       all models, cross-validation and held-out evaluation
  cv_epochs.py           epoch budget of the final refit, from CV only
  stats_tests.py         paired tests and false alarm rates
  benchmark_latency_corrected.py
  plot_confusion.py, plot_diagrams.py, make_tables.py
outputs/results/         result files behind every number in the manuscript
outputs/results/superseded/  held-out results of an earlier run that used the
                         held-out partition for early stopping (withdrawn; kept
                         for transparency)
outputs/models/          the final trained models evaluated on the held-out partition
outputs/figures/         figures of the manuscript
outputs/tables/          tables of the manuscript (Markdown, CSV)
outputs/logs/            training logs (used by src/cv_epochs.py)
data/                    mixalldata_clean.csv and the two split files (see data/README.md)
```

The key `lgbm_nondeterministic_SUPERSEDED` in
`corrected_protocol_results.json` records an earlier LightGBM run with
multithreaded, non-deterministic training; it is not reported. Code of the
original row-level protocol, whose results were withdrawn, is not part of this
release and remains only in the repository history.

## License

Apache License 2.0 (see `LICENSE`). The dataset is distributed separately under its own
license (CC BY 4.0, Mendeley Data).

## Citation

If you use this code, please cite the article (see `CITATION.cff`) and the
dataset papers: Kamel et al., ICC 2020, doi:10.1109/ICC40277.2020.9149132, and
Slama et al., *Inteligencia Artificial* 25(70), 2022,
doi:10.4114/intartif.vol25iss70pp138-154.
