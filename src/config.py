"""Central configuration for the Data Engineering part (Student 1).
Everything that decides *what data goes where* lives here so Student 2 can read it too."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "outputs" / "figures"
REPORT_DIR = ROOT / "outputs" / "reports"

# ---- Mendeley files: name -> (label, sex) -----------------------------------
# label: 1 = ADHD, 0 = Healthy control
MAT_FILES = {
    "FADHD": {"file": "FADHD.mat", "label": 1, "sex": "F"},
    "MADHD": {"file": "MADHD.mat", "label": 1, "sex": "M"},
    "FC":    {"file": "FC.mat",    "label": 0, "sex": "F"},
    "MC":    {"file": "MC.mat",    "label": 0, "sex": "M"},
}
# Subjects physically present in the .mat files (verified on the real data)
RAW_SUBJECTS = {"FADHD": 11, "MADHD": 27, "FC": 13, "MC": 29}
# Subjects removed. FADHD.mat contains 11 subjects but the paper (Table 2) uses 10 female ADHD.
# Subject index 6 (0-based) has a DEAD channel 2 (F4 is a constant value) in cells 4, 10 and 11,
# so it is unusable for the two-channel models. Removing it gives the paper's 79 subjects.
EXCLUDE_SUBJECTS = {"FADHD_06": "channel 2 (F4) is constant (std = 0) in cells 4/10/11"}
# Subjects expected AFTER exclusion (paper Table 2)
EXPECTED_SUBJECTS = {"FADHD": 10, "MADHD": 27, "FC": 13, "MC": 29}

# ---- Paper protocol (cell numbers are 1-based, as in the paper's Table 3) ----
TRAIN_CELL = 4            # Cz, F4  - cognitive challenge
VAL_CELL = 10             # F3, F4  - "validation"
TEST_CELL = 11            # Fz, F4  - "testing"
CELL_CHANNELS = {4: ("Cz", "F4"), 10: ("F3", "F4"), 11: ("Fz", "F4")}
FS = 256                  # sampling rate (Hz)

# ---- Preprocessing ------------------------------------------------------------
NORMALIZATION = "zscore"  # "zscore" | "none"  (paper says raw signals; z-score is our addition)
NORM_SCOPE = "per_sample_channel"  # each subject x channel normalised on its own statistics
TARGET_LENGTH = None      # None -> crop all cells to the shortest common length
DOWNSAMPLE_FACTOR = 1     # >1 shortens sequences (e.g. 4 -> 64 Hz) if LSTMs are too slow/OOM
SEED = 42
N_FOLDS = 5               # stratified subject-wise CV folds exported for the leak-free protocol

PROCESSED_NPZ = PROC_DIR / "adhd_eeg_paper_protocol.npz"
