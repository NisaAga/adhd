"""Create FAKE .mat files with the same structure as the Mendeley data, to test the pipeline
before the real data is available. NEVER use these for results."""
import sys, numpy as np
from pathlib import Path
from scipy.io import savemat
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C

def make(out_dir, T=2560, seed=0):
    rng = np.random.default_rng(seed); out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for g, info in C.MAT_FILES.items():
        n = C.RAW_SUBJECTS[g]; cells = np.empty((1, 11), dtype=object)
        for i in range(11):
            x = rng.normal(size=(n, T, 2)) + info["label"] * 0.3 * np.sin(np.linspace(0, 200, T))[None, :, None]
            if g == "FADHD": x[6, :, 1] = 5.0          # mimic dead channel of the real FADHD subject 6
            cells[0, i] = x
        cells[0, 9] = cells[0, 3].copy()               # mimic: real cell 10 == cell 4
        savemat(out_dir / info["file"], {g: cells})

if __name__ == "__main__":
    if any((C.RAW_DIR / i["file"]).exists() for i in C.MAT_FILES.values()):
        sys.exit("Real .mat files already in data/raw/ - refusing to overwrite them.")
    make(C.RAW_DIR); print("synthetic .mat written to", C.RAW_DIR)
