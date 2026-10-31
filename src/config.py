"""Configuration of the experiment reported in the manuscript.

Every value used by the authoritative pipeline (src/corrected_protocol.py,
src/run_corrected.py and the modules they import) is defined here, so the
manuscript's Table 3 can be checked against a single file.

Data: the public tabular release of VeReMi Extension (Mendeley Data,
doi:10.17632/k62n4z9gdz.1), two CSV files with 3,194,808 messages in total.
Target: the `class` column (20 classes); `type` is constant and unused.
Features: 24 kinematic columns plus 4 temporal features derived per sender
(28 inputs). Identifier columns are never model inputs (see
src/corrected_protocol.py). Partitions are sender-disjoint.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
TRAIN_CSV = DATA_DIR / "VeReMi_train_data.csv"
TEST_CSV = DATA_DIR / "VeReMi_test_data.csv"

OUTPUT_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"

for d in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data schema
# ---------------------------------------------------------------------------
TARGET_COL = "class"          # 20-class label; `type` is constant (always 4)

N_CLASSES = 20

# The `class` column of the public release is the VeReMi Extension
# "misbehaviour kind", copied from the simulation ground truth (Slama et al.,
# Inteligencia Artificial 25(70), 2022). Its integer codes follow the attacker
# enumeration of the F2MD framework that generated the dataset
# (veins-f2md/.../mdEnumTypes/AttackTypes.h): 0 Genuine, 1 ConstPos, ...,
# 12 StaleMessages (delayed messages), ..., 19 DoSDisruptiveSybil. The class
# counts confirm it: the three DoS variants (13-15) and Grid Sybil (16) carry
# roughly three to four times the messages of a normal-rate class.
# Earlier versions of this file used descriptive names that did not match the
# enumeration; the integer labels, and therefore every metric, are unaffected.
CLASS_NAMES = {
    0: "Genuine (benign)",
    1: "Constant Position",
    2: "Constant Position Offset",
    3: "Random Position",
    4: "Random Position Offset",
    5: "Constant Speed",
    6: "Constant Speed Offset",
    7: "Random Speed",
    8: "Random Speed Offset",
    9: "Eventual Stop",
    10: "Disruptive",
    11: "Data Replay",
    12: "Delayed Messages",
    13: "DoS",
    14: "DoS Random",
    15: "DoS Disruptive",
    16: "Grid Sybil",
    17: "Data Replay Sybil",
    18: "DoS Random Sybil",
    19: "DoS Disruptive Sybil",
}

# ---------------------------------------------------------------------------
# Partitioning and resampling
# ---------------------------------------------------------------------------
N_FOLDS = 5                    # sender-disjoint folds on the development partition
RANDOM_STATE = 42              # partitioning and permutation; + fold index per fold

# ADASYN raises each minority class of a TRAINING portion to about this many
# samples (k = 5 neighbours); larger classes are left unchanged. Validation and
# held-out data are never resampled. UNDERSAMPLE_CAP is above every class size,
# so the optional undersampling step in src/sampling.py never triggers.
ADASYN_TARGET = 50_000
ADASYN_N_NEIGHBORS = 5
UNDERSAMPLE_CAP = 2_000_000

# ---------------------------------------------------------------------------
# Proposed model (manuscript Section 3.3, Table 3)
# ---------------------------------------------------------------------------
CNN_FILTERS = 96
CNN_KERNELS = (1, 3, 5)
GRU_UNITS = 64                 # per direction -> 128 in total
DROPOUT_RATE = 0.3
RECURRENT_DROPOUT = 0.2
L2_REG = 1e-4
LABEL_SMOOTHING = 0.05
FOCAL_GAMMA = 2.0

# ---------------------------------------------------------------------------
# Training (shared by every neural configuration)
# ---------------------------------------------------------------------------
# The epoch budget (40) and early-stopping patience (8) for cross-validation
# are set in src/run_corrected.py; the held-out refit uses the median best CV
# epoch from src/cv_epochs.py.
BATCH_SIZE = 4096
INITIAL_LR = 1e-3
WEIGHT_DECAY = 1e-4
COSINE_FIRST_DECAY_STEPS_EPOCHS = 20   # first cosine cycle, in epochs
