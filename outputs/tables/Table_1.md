### Table 1. Data flow from the released file to the partitions

| Stage | Messages | Senders |
|---|---|---|
| Training file (70% row-level split) | 2,236,367 | 24,644 |
| Test file (30% row-level split) | 958,441 | 24,553 |
| Pooled corpus (= released file) | 3,194,808 | 24,663 |
| Exact duplicate rows removed | 0 | - |
| Rows removed for missing or infinite values | 0 | - |
| Development partition | 2,129,867 | 16,441 |
| CV validation portion (each fold) | 425,970-425,976 | 3,285-3,290 |
| CV training portion after ADASYN (each fold) | 2,061,184-2,074,570 | - |
| Held-out partition | 1,064,941 | 8,222 |
| Senders shared across any partition boundary | - | 0 |
