import sys, importlib.util, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import zscore, crop_to_length, downsample
from src.pipeline import run, load_processed

def test_zscore():
    Z = zscore(np.random.randn(5, 100, 2) * 7 + 3)
    assert np.allclose(Z.mean(1), 0, atol=1e-4) and np.allclose(Z.std(1), 1, atol=1e-3)

def test_crop_and_downsample():
    X = np.random.randn(3, 1000, 2)
    assert crop_to_length(X, 800).shape == (3, 800, 2)
    assert downsample(X, 4).shape == (3, 250, 2)

def test_pipeline_end_to_end(tmp_path):
    spec = importlib.util.spec_from_file_location("mk", ROOT / "scripts" / "00_make_synthetic_data.py")
    mk = importlib.util.module_from_spec(spec); spec.loader.exec_module(mk)
    mk.make(tmp_path / "raw", T=512)
    out = tmp_path / "out.npz"
    run(tmp_path / "raw", out, report_dir=tmp_path)
    d = load_processed(out)
    assert d["X_train"].shape == (79, 512, 2) and d["y_train"].sum() == 37
    assert d["X_eval"].shape == (158, 512, 2) and (d["y_eval"] == 0).sum() == 84
    assert not np.isnan(d["X_train"]).any()
    assert "FADHD_06" not in d["ids_train"] and len(set(d["ids_train"])) == 79
    assert np.bincount(d["fold"]).sum() == 79
