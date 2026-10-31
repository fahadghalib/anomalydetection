# Data

Download the two files of the Mendeley Data release

> Slama, O., Tarhouni, M., Zidi, S., & Alaya, B. (2023). *VeReMi_Extension:
> Dataset for Misbehaviors in VANETs* (Version 1) [Data set]. Mendeley Data.
> https://doi.org/10.17632/k62n4z9gdz.1

and place them in this directory:

| File | Size (bytes) | Records | SHA-256 |
|---|---|---|---|
| `VeReMi_train_data.csv` | 760,463,594 | 2,236,367 | `d84ed720a3111c02fcf9e8e4360bde5a43cdf067ee8ce53306811d2991c335eb` |
| `VeReMi_test_data.csv` | 325,908,296 | 958,441 | `aee98bf55149e86b286d7a3b46cffcf9e90a9f2fb6543b620fc0ea832f9cf76f` |

Check them with `sha256sum data/*.csv` (Linux) or
`Get-FileHash data\*.csv -Algorithm SHA256` (Windows PowerShell).

The release is derived from VeReMi Extension (Kamel et al., ICC 2020,
https://doi.org/10.1109/ICC40277.2020.9149132): the receiver logs of the 39
simulations were converted to CSV, merged, and labeled with the ground-truth
misbehavior type of each sender (Slama et al., *Inteligencia Artificial*
25(70), 2022, https://doi.org/10.4114/intartif.vol25iss70pp138-154).

The two files are pooled and re-partitioned by sender by
`src/corrected_protocol.py`, because 99.92% of the senders of the test file
also occur in the training file. Derived arrays are cached in `data/cache/`
on first use (about 2 GB); delete that directory to force recomputation.
