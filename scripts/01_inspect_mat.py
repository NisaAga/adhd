"""STEP 1 - run this FIRST on the real data. Prints shape of all 11 cells in all 4 files."""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config as C
from src.data_loader import load_mat_cells

for g, info in C.MAT_FILES.items():
    cells = load_mat_cells(C.RAW_DIR / info["file"])
    print(f"\n{info['file']}: {len(cells)} cells")
    for i, a in enumerate(cells, 1):
        mark = "  <-- used" if i in (C.TRAIN_CELL, C.VAL_CELL, C.TEST_CELL) else ""
        print(f"  cell {i:2d}: shape={a.shape} dtype={a.dtype}{mark}")
