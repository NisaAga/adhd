"""STEP 3 - exploratory figures for the report. Optional arg: path to a processed .npz"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import load_processed
from src import plots, config as C
d = load_processed(sys.argv[1] if len(sys.argv) > 1 else C.PROCESSED_NPZ)
for p in (plots.class_balance(d), plots.sample_signals(d, "train"), plots.sample_signals(d, "val"),
          plots.sample_signals(d, "test"), plots.mean_psd(d, "train"), plots.mean_psd(d, "test")):
    print("saved", p)
