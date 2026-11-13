# Data

The data come from the Mendeley Data release

> Slama, O., Tarhouni, M., Zidi, S., & Alaya, B. (2023). *VeReMi_Extension:
> Dataset for Misbehaviors in VANETs* (Version 1) [Data set]. Mendeley Data.
> https://doi.org/10.17632/k62n4z9gdz.1

The release is one archive, `mixalldata_clean.tar.xz` (329,167,264 bytes,
SHA-256 `040d88cf6c27fb3b1f92dd784802943e100aab6c729ffdbad8114e0849c3ca03`),
holding one file, `mixalldata_clean.csv` (3,194,808 records, SHA-256
`884e4567c39c1ee2560e09c48b7c6319baffab5a8ba5ff6149b071f8bfad9f1d`).

Earlier versions of this study used a random 70/30 row-level split of that
file into two files, which the scripts still read. Recreate them with

```bash
tar -xJf mixalldata_clean.tar.xz -C data/
python -m src.prepare_data
```

| File | Records | Senders |
|---|---|---|
| `VeReMi_train_data.csv` | 2,236,367 | 24,644 |
| `VeReMi_test_data.csv` | 958,441 | 24,553 |

`rowlevel_split.npz` records which rows of `mixalldata_clean.csv` belong to
the test file; both files keep the row order of the release. The script checks
the checksum of `mixalldata_clean.csv` and a fingerprint of the identifier and
label columns of each output file. The files used in the study were written by
R; their numeric values agree with the recreated files to about 14 significant
digits (largest relative difference 1.5e-13), so their checksums differ from
those of the recreated files. For reference, the study's copies had

| File | SHA-256 of the study's copy |
|---|---|
| `VeReMi_train_data.csv` | `d84ed720a3111c02fcf9e8e4360bde5a43cdf067ee8ce53306811d2991c335eb` |
| `VeReMi_test_data.csv` | `aee98bf55149e86b286d7a3b46cffcf9e90a9f2fb6543b620fc0ea832f9cf76f` |

The split does not affect any reported result: `src/corrected_protocol.py`
pools the two files again, which restores the released file, and re-partitions
the records by sender, because 99.92% of the senders of the test file also occur
in the training file. The release is derived from VeReMi Extension (Kamel et
al., ICC 2020, https://doi.org/10.1109/ICC40277.2020.9149132): the receiver logs
of the 39 simulations were converted to CSV, merged, and labeled with the
ground-truth misbehavior type of each sender (Slama et al., *Inteligencia
Artificial* 25(70), 2022, https://doi.org/10.4114/intartif.vol25iss70pp138-154).
Derived arrays are cached in `data/cache/` on first use (about 2 GB); delete
that directory to force recomputation.
