"""
data_io.py — small helpers to read session data safely.

- Tolerant to missing/short streams.
- Can return NaN-filled arrays for plotting (so matplotlib draws empty traces).
- Or, for inference, can return zero-filled arrays so the model can run.

All functions are pure NumPy/Pandas; no Torch here.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

LABEL_CANDIDATES = ["human_label.csv", "new_swim_label.csv", "swim_label.csv", "label.csv"]
ORIG_LABEL_CANDIDATES = [c for c in LABEL_CANDIDATES if c != "human_label.csv"]


def load_original_labels(session_dir: str, n: int) -> Optional[np.ndarray]:
    """Return original labels (not human_label.csv), padded/truncated to n."""
    labels = None
    for cand in ORIG_LABEL_CANDIDATES:
        lp = os.path.join(session_dir, cand)
        if os.path.exists(lp):
            try:
                # non-human files are typically "target" column with 1 header row
                lbl = pd.read_csv(lp, skiprows=1, names=["target"])["target"].values
                labels = lbl.astype(int)
                labels[labels == -1] = 2  # map -1 → Turn
                break
            except Exception:
                pass
    if labels is None:
        return None
    y = np.zeros(n, dtype=int)
    m = min(n, len(labels))
    y[:m] = labels[:m]
    return y


def _read_xyz_csv(path: str) -> Optional[np.ndarray]:
    """Return Nx3 float array or None if file missing/empty/bad."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, skiprows=1, names=["x", "y", "z", "t", "p"])
        if df is None or len(df) == 0:
            return None
        arr = df[["x", "y", "z"]].to_numpy(dtype=float, copy=True)
        return np.nan_to_num(arr)
    except Exception:
        return None


def _pad_or_truncate(arr: Optional[np.ndarray], n: int, fill_value: float) -> np.ndarray:
    """Resize Nx3 to length n (pad with fill) or return filled if arr is None."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    if arr is None:
        return np.full((n, 3), fill_value, dtype=np.float32)
    m = arr.shape[0]
    if m == n:
        out = arr
    elif m > n:
        out = arr[:n]
    else:
        out = np.full((n, 3), fill_value, dtype=np.float32)
        out[:m] = arr
    return out.astype(np.float32)


def _load_labels(session_dir: str, n: int) -> Optional[np.ndarray]:
    """Best-effort label loader, padded/truncated to n."""
    labels = None
    for cand in LABEL_CANDIDATES:
        lp = os.path.join(session_dir, cand)
        if os.path.exists(lp):
            try:
                if cand == "human_label.csv":
                    lbl = pd.read_csv(lp, header=None).values.squeeze()
                else:
                    lbl = pd.read_csv(lp, skiprows=1, names=["target"])["target"].values
                labels = lbl.astype(int)
                labels[labels == -1] = 2  # map -1 → Turn
                break
            except Exception:
                pass
    if labels is None:
        return None
    # pad/truncate
    y = np.zeros(n, dtype=int)
    m = min(n, len(labels))
    y[:m] = labels[:m]
    return y


def get_session_data(
    session_dir: str,
    fs: float = 25.0,
    add_gyro: bool = True,
    scale: bool = True,
    nan_for_missing: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Read acc.csv, mag.csv, gyro.csv (if present) in a tolerant way.

    Returns
    -------
    X : (T, 9) float32
        Stacked [acc, gyro, mag]. If a stream is missing:
          - when nan_for_missing=True → stream is filled with NaN (empty plot).
          - when nan_for_missing=False → stream is zero-filled (safe for inference).
    labels : (T,) int or None
        Best-effort labels from {LABEL_CANDIDATES}. Padded/truncated to T.

    Notes
    -----
    - Scaling converts:
        * acc counts → m/s²   (4096 counts per g, g=9.80665)
        * gyro counts → rad/s (16.4 counts per deg/s)
        * mag  counts → μT    (10 counts per μT)
      NaNs remain NaNs after scaling.
    """
    acc = _read_xyz_csv(os.path.join(session_dir, "acc.csv"))
    mag = _read_xyz_csv(os.path.join(session_dir, "mag.csv"))
    gyro = _read_xyz_csv(os.path.join(session_dir, "gyro.csv"))

    # decide timeline length (max length among available streams)
    lengths = [a.shape[0] for a in (acc, gyro, mag) if a is not None]
    n = max(lengths) if lengths else 0

    fill_plot = np.nan if nan_for_missing else 0.0

    acc = _pad_or_truncate(acc, n, fill_plot)
    mag = _pad_or_truncate(mag, n, fill_plot)
    gyro = _pad_or_truncate(gyro, n, fill_plot) if add_gyro else np.zeros((n, 0), dtype=np.float32)

    if scale and n > 0:
        grav = 9.80665
        acc_scale = 4096.0   # counts → g
        gyro_scale = 16.4    # counts → deg/s
        mag_scale = 10.0     # counts → μT

        acc = (acc / acc_scale) * grav
        if gyro.shape[1] == 3:
            gyro = (gyro / gyro_scale) * (np.pi / 180.0)
        mag = mag / mag_scale

    # stack channels
    X = np.concatenate([acc, gyro, mag], axis=1) if n > 0 else np.zeros((0, 9), dtype=np.float32)

    labels = _load_labels(session_dir, n)
    return X.astype(np.float32), labels
