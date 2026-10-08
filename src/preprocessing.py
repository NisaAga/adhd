"""Signal preprocessing helpers. Pure functions, easy to unit-test."""
import numpy as np
from scipy.signal import decimate


def zscore(X, scope="per_sample_channel", eps=1e-8):
    """X: (N, T, C). per_sample_channel -> stats over T for every subject & channel.
    'global_channel' -> stats over (N,T) per channel (use training stats only!)."""
    X = np.asarray(X, dtype=np.float32)
    if scope == "per_sample_channel":
        m, s = X.mean(axis=1, keepdims=True), X.std(axis=1, keepdims=True)
    elif scope == "global_channel":
        m, s = X.mean(axis=(0, 1), keepdims=True), X.std(axis=(0, 1), keepdims=True)
    else:
        raise ValueError(scope)
    return (X - m) / (s + eps)


def crop_to_length(X, length):
    if X.shape[1] < length:
        raise ValueError(f"Signal shorter ({X.shape[1]}) than target {length}")
    return X[:, :length, :]


def downsample(X, factor):
    if factor == 1:
        return X
    return decimate(X, factor, axis=1, zero_phase=True).astype(np.float32)


def check_quality(X, name=""):
    """Return a dict of quality flags (NaN/Inf, flat channels, extreme amplitude)."""
    flat = (X.std(axis=1) < 1e-9).sum()
    return {
        "name": name, "shape": list(X.shape),
        "n_nan": int(np.isnan(X).sum()), "n_inf": int(np.isinf(X).sum()),
        "n_flat_channels": int(flat),
        "min": float(np.nanmin(X)), "max": float(np.nanmax(X)),
        "mean": float(np.nanmean(X)), "std": float(np.nanstd(X)),
    }
