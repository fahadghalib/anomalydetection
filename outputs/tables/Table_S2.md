### Table S2. Fold-level validation results (%) used for the paired comparisons

| Configuration | Metric | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean ± SD |
|---|---|---|---|---|---|---|---|
| Proposed (CNN-BiGRU-attention) | Accuracy | 87.37 | 86.66 | 87.17 | 86.14 | 86.46 | 86.76 ± 0.50 |
| Proposed (CNN-BiGRU-attention) | Macro F1 | 72.07 | 71.75 | 72.08 | 71.27 | 72.14 | 71.86 ± 0.36 |
| Proposed, permuted column order | Accuracy | 85.97 | 86.63 | 85.53 | 85.55 | 86.81 | 86.10 ± 0.60 |
| Proposed, permuted column order | Macro F1 | 70.09 | 69.74 | 69.65 | 69.59 | 70.42 | 69.90 ± 0.35 |
| Feed-forward, capacity-matched | Accuracy | 89.63 | 89.35 | 88.99 | 89.23 | 89.71 | 89.38 ± 0.29 |
| Feed-forward, capacity-matched | Macro F1 | 74.80 | 74.31 | 73.92 | 74.20 | 75.02 | 74.45 ± 0.45 |
| Feed-forward, reduced capacity | Accuracy | 89.35 | 89.53 | 89.27 | 89.53 | 89.40 | 89.42 ± 0.11 |
| Feed-forward, reduced capacity | Macro F1 | 74.47 | 74.16 | 74.10 | 74.56 | 75.06 | 74.47 ± 0.38 |
| Gradient-boosted trees (LightGBM) | Accuracy | 80.28 | 81.24 | 84.27 | 79.18 | 74.01 | 79.80 ± 3.75 |
| Gradient-boosted trees (LightGBM) | Macro F1 | 59.96 | 62.70 | 67.14 | 56.37 | 56.54 | 60.54 ± 4.53 |
