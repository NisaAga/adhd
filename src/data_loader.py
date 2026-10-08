"""Load the Mendeley .mat files (1x11 cell arrays) into NumPy.
Handles both classic MAT (scipy) and MATLAB v7.3/HDF5 (h5py) files."""
import numpy as np
from pathlib import Path
from scipy.io import loadmat


def _load_v5(path):
    d = loadmat(str(path), squeeze_me=False)
    keys = [k for k in d if not k.startswith("__")]
    if not keys:
        raise ValueError(f"No variables found in {path}")
    cells = d[keys[0]]
    return [np.asarray(cells[0, i]) for i in range(cells.shape[1])]


def _load_v73(path):
    import h5py
    out = []
    with h5py.File(str(path), "r") as f:
        key = [k for k in f.keys() if not k.startswith("#")][0]
        refs = np.array(f[key]).ravel()
        for r in refs:
            arr = np.array(f[r])
            out.append(arr.T if arr.ndim > 1 else arr)  # HDF5 stores dims reversed
    return out


def load_mat_cells(path):
    """Return a python list of 11 numpy arrays (one per cell)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Put the 4 .mat files in data/raw/.")
    try:
        return _load_v5(path)
    except NotImplementedError:  # v7.3 file
        return _load_v73(path)


def get_cell(cells, cell_no):
    """cell_no is 1-based (paper numbering). Returns array (subjects, samples, channels)."""
    arr = np.asarray(cells[cell_no - 1], dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(f"Cell {cell_no}: expected 3-D array, got shape {arr.shape}")
    return arr
