"""STEP 2 - build the processed dataset and audit report.
Usage:  python scripts/02_build_dataset.py                 # full resolution (T=11520, as in the paper)
        python scripts/02_build_dataset.py --downsample 8  # T=1440 (32 Hz), much faster LSTM training
"""
import sys, json, argparse; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.pipeline import run

ap = argparse.ArgumentParser(); ap.add_argument("--downsample", type=int, default=1)
a = ap.parse_args()
out = C.PROCESSED_NPZ if a.downsample == 1 else C.PROC_DIR / f"adhd_eeg_paper_protocol_ds{a.downsample}.npz"
r = run(out_path=out, downsample_factor=a.downsample)
print(json.dumps(r, indent=2)); print("\nSaved", out)
