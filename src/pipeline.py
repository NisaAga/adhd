"""End-to-end: .mat -> processed .npz following the paper protocol (+ leak-free folds)."""
import json, warnings
import numpy as np
from . import config as C
from .dataset_builder import (load_all_files, build_cell, find_flat_subjects, cells_identical, subject_folds)
from .preprocessing import zscore, crop_to_length, downsample, check_quality


def run(raw_dir=C.RAW_DIR, out_path=C.PROCESSED_NPZ, report_dir=None, downsample_factor=None):
    report_dir = report_dir or C.REPORT_DIR
    ds = downsample_factor or C.DOWNSAMPLE_FACTOR
    allc = load_all_files(raw_dir)

    # ---- data audit ---------------------------------------------------------------------------
    flat = find_flat_subjects(allc)
    ident = cells_identical(allc, C.TRAIN_CELL, C.VAL_CELL)
    report = {"flat_or_nan_subjects_detected": flat, "excluded_subjects": C.EXCLUDE_SUBJECTS,
              "train_cell_identical_to_val_cell": ident}
    if set(flat) != set(C.EXCLUDE_SUBJECTS):
        warnings.warn(f"Detected bad subjects {sorted(flat)} differ from config exclusions {sorted(C.EXCLUDE_SUBJECTS)}")
    if all(ident.values()):
        warnings.warn(f"Cell {C.VAL_CELL} is IDENTICAL to training cell {C.TRAIN_CELL}: the paper's "
                      "'validation' set is the training data (data leakage). See docs/DATASET_NOTES.md")

    raw = {n: build_cell(allc, n) for n in (C.TRAIN_CELL, C.VAL_CELL, C.TEST_CELL)}
    native = {n: int(v[0].shape[1]) for n, v in raw.items()}
    length = C.TARGET_LENGTH or min(native.values())
    report.update({"native_lengths": native, "common_length": int(length), "downsample_factor": ds, "quality": []})

    out = {}
    for tag, n in (("train", C.TRAIN_CELL), ("val", C.VAL_CELL), ("test", C.TEST_CELL)):
        X, y, ids, grp = raw[n]
        report["quality"].append(check_quality(X, f"cell{n}_raw"))
        X = crop_to_length(X, length)
        if C.NORMALIZATION == "zscore":
            X = zscore(X, C.NORM_SCOPE)
        X = downsample(X, ds).astype(np.float32)
        report["quality"].append(check_quality(X, f"cell{n}_processed"))
        out.update({f"X_{tag}": X, f"y_{tag}": y, f"ids_{tag}": ids, f"groups_{tag}": grp})

    # paper's evaluation set = cell 10 + cell 11 (158 samples)
    out["X_eval"] = np.concatenate([out["X_val"], out["X_test"]])
    out["y_eval"] = np.concatenate([out["y_val"], out["y_test"]])
    out["ids_eval"] = np.concatenate([out["ids_val"], out["ids_test"]])

    # leak-free protocol: stratified subject-wise folds (same fold id valid for cell 4 AND cell 11)
    out["fold"] = subject_folds(out["y_train"])

    out["channels_train"] = np.array(C.CELL_CHANNELS[C.TRAIN_CELL])
    out["channels_val"] = np.array(C.CELL_CHANNELS[C.VAL_CELL])
    out["channels_test"] = np.array(C.CELL_CHANNELS[C.TEST_CELL])
    out["fs"] = np.array(C.FS / ds)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    report_dir.mkdir(parents=True, exist_ok=True)
    report["shapes"] = {k: list(v.shape) for k, v in out.items() if k.startswith(("X_", "y_"))}
    report["class_counts"] = {t: {"healthy": int((out[f"y_{t}"] == 0).sum()),
                                  "adhd": int((out[f"y_{t}"] == 1).sum())} for t in ("train", "val", "test", "eval")}
    report["fold_sizes"] = np.bincount(out["fold"]).tolist()
    with open(report_dir / f"pipeline_report{out_path.stem.replace('adhd_eeg_paper_protocol', '')}.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def load_processed(path=C.PROCESSED_NPZ):
    """Student 2 entry point. Returns dict of arrays (X_train,y_train,X_val,...,X_eval,y_eval,fold)."""
    return dict(np.load(path, allow_pickle=False))
