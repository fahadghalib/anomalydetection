#!/usr/bin/env bash
# Reproduce every table and figure of the manuscript.
#
#   bash reproduce_all.sh tables   regenerate tables and figures from the
#                                  deposited result files (minutes, no GPU)
#   bash reproduce_all.sh full     re-run every experiment from the two CSV
#                                  files, then regenerate tables and figures
#
# Run from the repository root inside the environment of requirements-lock.txt.
# Latency must be measured on an otherwise idle machine.
set -euo pipefail
MODE="${1:-tables}"

python verify_deposit.py

if [[ "$MODE" == "full" ]]; then
  if [[ -d outputs/results ]]; then
    mv outputs/results "outputs/results_deposited_$(date +%Y%m%d_%H%M%S)"
  fi
  mkdir -p outputs/results outputs/logs

  # Data accounting and leakage audit (Tables 1 and 2)
  python -m src.data_audit            | tee outputs/logs/data_audit.log
  python -m src.leakage_probe         | tee outputs/logs/leakage_probe.log
  python -m src.corrected_protocol    | tee outputs/logs/partitions.log

  # Five-fold sender-disjoint cross-validation of every configuration (Tables 5, S2)
  python -m src.run_corrected --models cnn,cnn_permuted,mlp_cap,mlp_matched,lgbm --folds 5 \
      | tee outputs/logs/cross_validation.log
  python -m src.fold_distribution
  # Convergence check of Section 3.5 (fold 1, 100 epochs, patience 20)
  python -m src.run_corrected --models cnn --only-fold 1 --epochs 100 --patience 20 \
      --out-suffix _cnn_convergence | tee outputs/logs/cnn_convergence.log

  # Final refit with the CV epoch budget and single held-out evaluation (Tables 4, 7)
  python -m src.cv_epochs
  python -m src.run_corrected --models mlp_cap,mlp_matched,cnn,lgbm --eval-test \
      | tee outputs/logs/final_holdout_fixed_epochs.log

  # Latency (Table 9). The native-Windows row is produced by running the same
  # command with a native-Windows Python and --tag corrected_native_windows_cpu.
  python -m src.benchmark_latency_corrected --device gpu --tag corrected_gpu
  python -m src.benchmark_latency_corrected --device cpu --tag corrected_wsl2_cpu
fi

# Statistics, tables and figures (Tables 1, 2, 4-9, S1, S2; Figures 1, 2, 5)
python -m src.stats_tests
python -m src.make_tables
python -m src.plot_confusion
python -m src.plot_diagrams
echo "Done. Tables in outputs/tables/, figures in outputs/figures/."
