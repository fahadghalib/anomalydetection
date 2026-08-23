"""
Configuration for the VeReMi Extension hybrid anomaly detection pipeline.

IMPORTANT DEVIATIONS FROM THE PAPER (all forced by hardware: CPU-only, no GPU,
20 cores / 34GB RAM measured on this machine) -- documented here so results
can be interpreted correctly against "Network Traffic Anomaly Detection Using
a Hybrid Deep Learning Model":

1. Target column: the paper states the target is the column named "type".
   In the actual CSV files, "type" is a CONSTANT (=4) for every single row
   (it is the VeReMi message type, always single-hop beacon) -- it carries
   zero information and cannot be a 20-class target. The column actually
   holding the 20-class label described in the paper's Table 1 (and whose
   value counts match Table 1 and Table 2 "Support" exactly) is "class".
   We therefore use "class" as the target and drop "type" from the features
   (it is constant, so it would contribute nothing to a scaled feature
   anyway). This leaves 28 numeric input features instead of 29.

2. Cross-validation folds: set to 5, matching the paper's N=5. (A first CPU
   run used 3 folds with Benign capped at 60,000/fold and measured ~2.2h/fold;
   with UNDERSAMPLE_CAP now effectively disabled -- see #3 below -- each
   fold's training set grows from ~640K to ~1.7M rows, so 5 folds is now
   estimated at roughly ~24-28h end-to-end on this CPU-only machine. Budget
   accordingly, or reduce ADASYN_TARGET/MAX_EPOCHS if that's too long.)

3. ADASYN target: the paper oversamples every class up to ~455,000 samples
   (parity with the majority class), which would produce ~9.1M training rows
   per fold -- several days of wall-clock time on this CPU-only machine. We
   lower ADASYN_TARGET to 20,000 for the minority classes, scoped strictly to
   the training portion of each fold. UNDERSAMPLE_CAP, which used to cap the
   majority (Benign) class down to a small fixed size, is now set effectively
   unlimited: a first run at a 60,000 cap measured recall(Benign)=48.1% vs the
   paper's claimed 94.2%, and since Benign is 59.5% of the test set, that gap
   alone explained ~27 of the ~30-point overall accuracy shortfall. ADASYN
   itself never reduces the majority class (it only adds minority synthetic
   samples), so leaving Benign at its natural per-fold size matches the
   paper's actual oversample-only protocol, at the cost of a much larger
   per-fold training set (~1.7M rows instead of ~640K). Validation and test
   folds are NEVER touched by either step.

4. Batch size: 1024 instead of 64 -- purely a CPU throughput optimization
   (benchmarked ~3.5x higher samples/sec at this batch size on this
   machine's CPU); does not change the model architecture or loss.

5. Max epochs / early-stopping patience reduced (25 / 5 instead of 100 / 20)
   to fit a multi-hour budget instead of the paper's unspecified (likely
   GPU-day-scale) budget.

6. Baseline comparison models (SEMI-GRU, MCNN, ANN, CNN, DNN, AlertNet,
   Random Forest) from the paper's Figures 5-8 are NOT retrained here --
   out of scope by user's explicit choice. Only the proposed hybrid
   CNN-BiGRU-Attention model is trained and evaluated.

Everything else (multi-scale 1D-CNN with K=1,3,5; BiGRU; additive attention;
weighted categorical focal loss with sqrt-scaled class weights; AdamW +
cosine-annealing-with-warm-restarts; stratified K-fold; scaler/ADASYN fit
only on the training fold) follows the paper's methodology section as
written.
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
TARGET_COL = "class"          # 20-class label (see deviation #1 above)
DROP_COLS = ["type", "class"]  # 'type' constant/useless, 'class' is target

N_CLASSES = 20

CLASS_NAMES = {
    0: "Benign (Normal Traffic)",
    1: "Constant Attack",
    2: "Constant Position Attack",
    3: "Constant Speed Attack",
    4: "Random Attack",
    5: "Random Position Attack",
    6: "Random Speed Attack",
    7: "Eventual Attack",
    8: "Eventual Position Attack",
    9: "Eventual Speed Attack",
    10: "Disruptive Attack",
    11: "Disruptive Position Attack",
    12: "Disruptive Speed Attack",
    13: "Data Replay Attack",
    14: "Denial of Service (DoS)",
    15: "Distributed DoS (DDoS)",
    16: "Sybil Attack",
    17: "Sybil Position Attack",
    18: "Sybil Speed Attack",
    19: "Mixed / Other Misbehavior",
}

# ---------------------------------------------------------------------------
# Cross validation / sampling (see deviations #2, #3)
# ---------------------------------------------------------------------------
N_FOLDS = 5  # matches the paper's protocol (was 3 for the CPU-budget run)
RANDOM_STATE = 42

# Set far above any class's fold size (largest is Benign, ~1.06M/fold at N_FOLDS=5)
# so undersampling never actually triggers -- matches the paper's ADASYN-only
# protocol, which never reduces the majority class. A real run at
# UNDERSAMPLE_CAP=60_000 measured recall(Benign)=48.1% vs the paper's claimed
# 94.2%; since Benign is 59.5% of the test set, that single gap explained
# ~27 of the ~30-point accuracy shortfall (52.2% vs 82.0%) -- undersampling
# Benign to 60k from ~887k/fold was starving the model of majority-class
# signal, not the fold count or epoch budget.
UNDERSAMPLE_CAP = 2_000_000  # effectively disabled -- oversample-only, like the paper
# Oversample target per class (within train fold). Raised from an initial
# 20,000 (CPU-budget run) to 50,000 now that GPU throughput makes the larger
# per-fold training set affordable -- gives the weakest minority classes
# (Disruptive/Random Attack, Data Replay, DDoS) more synthetic diversity to
# learn from, rather than just the bare minimum used to avoid ADASYN errors.
ADASYN_TARGET = 50_000
ADASYN_N_NEIGHBORS = 5

# ---------------------------------------------------------------------------
# Model architecture (Section 3.3 / 3.4 of the paper)
# ---------------------------------------------------------------------------
# 28 base features (paper) + 4 derived temporal-replay/flood features added
# in data_pipeline.py (dt_prev, pos_delta, spd_delta, msg_rate_5s) -- see
# that module's docstring for why: Data Replay / DDoS are defined by
# repetition over time, invisible to a single-row snapshot.
N_FEATURES = 32
CNN_FILTERS = 96
CNN_KERNELS = (1, 3, 5)
GRU_UNITS = 64                # per direction -> 128 total (BiGRU)
DROPOUT_RATE = 0.3
RECURRENT_DROPOUT = 0.2
L2_REG = 1e-4
LABEL_SMOOTHING = 0.05

FOCAL_GAMMA = 2.0

# ---------------------------------------------------------------------------
# Training (see deviations #4, #5)
# ---------------------------------------------------------------------------
# Now running on a GPU (RTX 4060 Laptop, via WSL2 + tensorflow[and-cuda])
# instead of CPU-only. Benchmarked on the real model: batch=1024 barely beat
# CPU (~2,267 samples/sec -- kernel-launch overhead dominates this small a
# model at that batch size); batch=8192 reached ~12,832 samples/sec (~6x
# CPU) but produced "ran out of memory trying to allocate 6.54GiB" warnings
# during the BiGRU backward pass and crashed twice with a fatal
# CUDA_ERROR_UNKNOWN inside the same gradient-reduction kernel, a few epochs
# apart each time -- consistent with 8192 sitting right at this 8GB card's
# VRAM ceiling (batch=16384 OOM'd outright). Dropped to 4096 (~8,309
# samples/sec, ~4x CPU) to leave real memory headroom; MAX_EPOCHS/PATIENCE
# stay at the paper's exact values.
BATCH_SIZE = 4096
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 20
INITIAL_LR = 1e-3
WEIGHT_DECAY = 1e-4
COSINE_FIRST_DECAY_STEPS_EPOCHS = 20  # in epochs, converted to steps at runtime
