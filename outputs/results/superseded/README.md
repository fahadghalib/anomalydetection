# Superseded held-out results (withdrawn)

These files come from an earlier held-out run in which early stopping (and the
restoration of the best weights) monitored the held-out partition itself, so
the held-out data influenced training. They were replaced by the results in
`outputs/results/`, where each final model is trained for a fixed number of
epochs taken from cross-validation (`src/cv_epochs.py`) and the held-out
partition is evaluated once. The latency files in this folder timed freshly
initialized weights in eager mode only and were replaced by measurements of the
trained final model in eager and compiled mode. Kept for transparency only.
