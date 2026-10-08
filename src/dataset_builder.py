"""Assemble X / y / metadata for a given cell by stacking FADHD, MADHD, FC, MC."""
import numpy as np
from . import config as C
from .data_loader import load_mat_cells, get_cell


def load_all_files(raw_dir=C.RAW_DIR):
    return {g: load_mat_cells(raw_dir / info["file"]) for g, info in C.MAT_FILES.items()}


def build_cell(all_cells, cell_no, strict=True):
    """Stack the 4 groups for one cell -> X (N,T,C), y (N,), ids (N,), groups (N,).
    Subject ids use the ORIGINAL index in the .mat file (e.g. FADHD_06) so they are stable across cells.
    Subjects listed in config.EXCLUDE_SUBJECTS are dropped."""
    Xs, ys, ids, grp = [], [], [], []
    for g, info in C.MAT_FILES.items():
        a = get_cell(all_cells[g], cell_no)
        if strict and a.shape[0] != C.RAW_SUBJECTS[g]:
            raise ValueError(f"{g} cell {cell_no}: {a.shape[0]} subjects, expected {C.RAW_SUBJECTS[g]}")
        keep = [i for i in range(a.shape[0]) if f"{g}_{i:02d}" not in C.EXCLUDE_SUBJECTS]
        if strict and len(keep) != C.EXPECTED_SUBJECTS[g]:
            raise ValueError(f"{g}: {len(keep)} subjects after exclusion, expected {C.EXPECTED_SUBJECTS[g]}")
        Xs.append(a[keep])
        ys += [info["label"]] * len(keep)
        ids += [f"{g}_{i:02d}" for i in keep]
        grp += [g] * len(keep)
    n = min(x.shape[1] for x in Xs)
    X = np.concatenate([x[:, :n, :] for x in Xs], axis=0)
    return X, np.array(ys, dtype=np.int64), np.array(ids), np.array(grp)


def find_flat_subjects(all_cells, cells=(4, 10, 11), tol=1e-9):
    """Detect subjects having a constant (dead) channel or NaNs in any used cell."""
    bad = {}
    for g in C.MAT_FILES:
        for cn in cells:
            a = get_cell(all_cells[g], cn)
            for i in range(a.shape[0]):
                if np.isnan(a[i]).any() or (a[i].std(axis=0) < tol).any():
                    bad.setdefault(f"{g}_{i:02d}", []).append(cn)
    return bad


def cells_identical(all_cells, a=4, b=10):
    """True if cell a and cell b contain exactly the same numbers in every file."""
    return {g: bool(np.array_equal(get_cell(all_cells[g], a), get_cell(all_cells[g], b))) for g in C.MAT_FILES}


def subject_folds(y, n_folds=C.N_FOLDS, seed=C.SEED):
    """Stratified fold id (0..n_folds-1) per subject -> leak-free, subject-wise cross-validation."""
    from sklearn.model_selection import StratifiedKFold
    fold = np.zeros(len(y), dtype=np.int64)
    for k, (_, te) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        fold[te] = k
    return fold
