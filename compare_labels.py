#!/usr/bin/env python3
"""
Label-comparison viewer with original data panels, a combined model+original panel,
and a combined annotators overlay panel, plus a vertical cursor.

Panels (top → bottom):
  1) Accelerometer (ax, ay, az)
  2) Gyroscope     (gx, gy, gz)
  3) Magnetometer  (mx, my, mz)
  4) Yaw cos/sin   (computed via utils.get_orientation on sanitized inputs)
  5) Reference labels: Model (pred) + Original (dataset) OVERLAID with slight offsets
  6) Annotators: all human label CSVs OVERLAID with slight offsets

Keys:
  ↑ / ↓ : previous / next session
  ← / → : move cursor by 1 sample
  h     : toggle bottom x-axis labels between seconds and samples
  q/Esc : quit
"""

from __future__ import annotations

import argparse
import os
import re
from glob import glob
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# ---------------------- HARD-CODE YOUR FOLDERS HERE ----------------------
ANNOTATOR_ROOTS = [
    "labels/alex",
    "labels/Arman",
    "labels/Shaya_labels",
    "labels/Subhra_labels",
]
DATA_ROOT = "examples 1"   # session folders by key (same as your labeler)
FS_DEFAULT = 25.0
# ------------------------------------------------------------------------

_SUFFIX_RE = re.compile(r"""(?ix)(?:^|_)human[ _\-]*label[^/\\]*\.csv$""")

# HiDPI-friendly backend selection (falls back to TkAgg)
import matplotlib
import os as _os
_os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
_os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
for _bk in ("QtAgg", "MacOSX", "TkAgg"):
    try:
        matplotlib.use(_bk, force=True)
        print(f"[ui] Using Matplotlib backend: {_bk}")
        break
    except Exception as _e:
        print(f"[ui] Backend {_bk} unavailable: {_e}")
import matplotlib.pyplot as plt

# ---------------------- Label schema ----------------------
CLASS_NAMES = {0: "Rest", 1: "Swim", 2: "Turn", 3: "Dolphin"}
YTICKS = [0, 1, 2, 3]

# ---------------------- Repo helpers (same as labeler) ----------------------
from data_io import get_session_data, load_original_labels  # type: ignore
from utils import get_orientation     # type: ignore

# Optional model (predictions)
try:
    import torch  # noqa: F401
    from model_infer import (
        load_model,
        predict_arrays_argmax,
        smooth_label,
        DEFAULT_MODEL_PATH,
    )
except Exception:
    torch = None
    load_model = predict_arrays_argmax = smooth_label = None
    DEFAULT_MODEL_PATH = None

# -------- yaw sanitation (same approach as your annotation tool) --------
def _ffill_bfill_nan(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr.astype(np.float32)
    df = pd.DataFrame(arr).replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df.to_numpy(dtype=np.float32)

def _stabilize_mag(mag: np.ndarray, tiny_norm: float = 1e-9) -> np.ndarray:
    m = mag.copy()
    if m.shape[0] == 0:
        return m
    norms = np.linalg.norm(m, axis=1)
    bad = norms < tiny_norm
    if not bad.any():
        return m
    good = np.where(~bad)[0]
    if good.size == 0:
        return m
    last = good[0]
    for i in range(m.shape[0]):  # ffill
        if bad[i]:
            m[i] = m[last]
        else:
            last = i
    last = good[-1]
    for i in range(m.shape[0] - 1, -1, -1):  # bfill
        if bad[i]:
            m[i] = m[last]
        else:
            last = i
    return m

def _compute_yaw_cos_sin(acc_scaled: np.ndarray,
                         gyro_scaled: np.ndarray,
                         mag_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = acc_scaled.shape[0]
    yaw_cos = np.full(T, np.nan, dtype=np.float32)
    yaw_sin = np.full(T, np.nan, dtype=np.float32)
    if T == 0:
        return yaw_cos, yaw_sin
    finite_acc = np.isfinite(acc_scaled).all(axis=1)
    finite_mag = np.isfinite(mag_scaled).all(axis=1)
    valid = finite_acc & finite_mag
    if valid.sum() < 5:
        return yaw_cos, yaw_sin
    acc_f = _ffill_bfill_nan(acc_scaled)
    gyro_f = _ffill_bfill_nan(gyro_scaled)
    mag_f  = _stabilize_mag(_ffill_bfill_nan(mag_scaled))
    if not (np.isfinite(acc_f).all() and np.isfinite(mag_f).all()):
        return yaw_cos, yaw_sin
    try:
        _, _, _, _, yaw_deg, _, _ = get_orientation(acc_f, gyro_f, mag_f)
        yaw_deg = np.asarray(yaw_deg, dtype=np.float32)
        yaw_rad = np.deg2rad(yaw_deg)
        c = np.cos(yaw_rad).astype(np.float32)
        s = np.sin(yaw_rad).astype(np.float32)
        yaw_cos[valid] = c[valid]
        yaw_sin[valid] = s[valid]
    except Exception as e:
        print(f"[yaw] get_orientation failed: {e}")
    return yaw_cos, yaw_sin

# ---------------------- Filename → session key ----------------------
def session_key_from_filename(filename: str) -> str:
    base = os.path.basename(filename)
    key = _SUFFIX_RE.sub("", base)
    if key == base and base.lower().endswith(".csv"):
        key = base[:-4]
    return key

# ---------------------- IO helpers for labels ----------------------
def scan_annotator(root: str, filter_regex: Optional[str]) -> Tuple[str, Dict[str, str]]:
    name = os.path.basename(os.path.normpath(root))
    csvs = glob(os.path.join(root, "*.csv"))
    mapping: Dict[str, str] = {}
    for p in csvs:
        key = session_key_from_filename(p)
        if filter_regex and not re.search(filter_regex, key):
            continue
        if key in mapping:
            try:
                n_old = sum(1 for _ in open(mapping[key], "rb"))
                n_new = sum(1 for _ in open(p, "rb"))
                if n_new >= n_old:
                    mapping[key] = p
            except Exception:
                if p > mapping[key]:
                    mapping[key] = p
        else:
            mapping[key] = p
    return name, mapping

def load_labels_from_path(path: str) -> Optional[np.ndarray]:
    try:
        arr = pd.read_csv(path, header=None).iloc[:, 0].to_numpy()
        if np.issubdtype(arr.dtype, np.number):
            return arr.astype(float)
        arr = pd.to_numeric(pd.Series(arr), errors="coerce").to_numpy()
        return arr.astype(float)
    except Exception as e:
        print(f"[warn] failed to read {path}: {e}")
        return None

def build_session_arrays(maps: List[Dict[str, str]], session_key: str) -> Tuple[List[Optional[np.ndarray]], int]:
    arrays: List[Optional[np.ndarray]] = []
    T_max = 0
    for m in maps:
        if session_key in m:
            arr = load_labels_from_path(m[session_key])
            arrays.append(arr)
            if arr is not None:
                T_max = max(T_max, int(arr.size))
        else:
            arrays.append(None)
    return arrays, T_max

# ---------------------- Original data + labels loader ----------------------
def load_original_streams(session_key: str, fs: float):
    """
    Returns dict with:
      n, acc (n,3), gyr (n,3), mag (n,3), yaw_cos (n,), yaw_sin (n,), orig_labels (n,) or None
    """
    session_dir = os.path.join(DATA_ROOT, session_key)
    out = dict(n=0, acc=np.zeros((0,3)), gyr=np.zeros((0,3)), mag=np.zeros((0,3)),
               yaw_cos=np.zeros(0), yaw_sin=np.zeros(0), orig_labels=None)
    if not os.path.isdir(session_dir):
        return out
    X_scaled, labels = get_session_data(session_dir, fs=fs, add_gyro=True, scale=True, nan_for_missing=False)
    if X_scaled is None or X_scaled.size == 0:
        return out
    n = int(X_scaled.shape[0])
    acc = X_scaled[:, 0:3].astype(float, copy=False)
    gyr = X_scaled[:, 3:6].astype(float, copy=False) if X_scaled.shape[1] >= 6 else np.zeros((n,3))
    mag = X_scaled[:, 6:9].astype(float, copy=False) if X_scaled.shape[1] >= 9 else np.zeros((n,3))
    yaw_c, yaw_s = _compute_yaw_cos_sin(acc, gyr, mag)
    try:
        orig = load_original_labels(session_dir, n)
    except Exception:
        orig = None
    out.update(n=n, acc=acc, gyr=gyr, mag=mag, yaw_cos=yaw_c, yaw_sin=yaw_s, orig_labels=orig)
    return out

def run_model_prediction(session_key: str, fs: float, model) -> Optional[np.ndarray]:
    if model is None:
        return None
    session_dir = os.path.join(DATA_ROOT, session_key)
    try:
        X_raw, _ = get_session_data(session_dir, fs=fs, add_gyro=True, scale=False, nan_for_missing=False)
        if X_raw is None or X_raw.size == 0 or X_raw.shape[1] < 9:
            return None
        X_raw = X_raw.copy()
        X_raw[:, 3:6] = 0.0
        acc = X_raw[:, 0:3]
        gyr = X_raw[:, 3:6]
        mag = X_raw[:, 6:9]
        y_pred = predict_arrays_argmax(acc, gyr, mag, model)
        y_pred = smooth_label(y_pred, window=75)
        return y_pred.astype(int, copy=False)
    except Exception as e:
        print(f"[inference] failed on {session_key}: {e}")
        return None

# ---------------------- Plot helpers ----------------------
def _plot_labels_step(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color=None, lw: float = 1.6, y_offset: float = 0.0, label: Optional[str]=None):
    """Step-plot labels; split at NaNs; add a small vertical offset for overlay separation."""
    if y is None or y.size == 0:
        return
    y = y.astype(float)
    valid = np.isfinite(y)
    if not valid.any():
        return
    idx = np.where(valid)[0]
    splits = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[0, splits + 1]
    ends = np.r_[splits, len(idx) - 1]
    for s, e in zip(starts, ends):
        seg_idx = idx[s:e+1]
        ax.step(x[seg_idx], y[seg_idx] + y_offset, where="post", linewidth=lw, color=color, label=label if s == starts[0] else None)

def _format_label_axis(ax: plt.Axes, max_offset: float = 0.3):
    ax.set_yticks(YTICKS, labels=[CLASS_NAMES[k] for k in YTICKS])
    ax.set_ylim(-0.5 - max_offset, max(YTICKS) + 0.5 + max_offset)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")

def _offsets(n: int, max_abs: float = 0.28) -> np.ndarray:
    if n <= 1:
        return np.array([0.0], dtype=float)
    return np.linspace(-max_abs, max_abs, n, dtype=float)

# ---------------------- App ----------------------
class SessionBrowser:
    def __init__(self, annotator_names: List[str], annotator_maps: List[Dict[str, str]],
                 sessions: List[str], fs: float, show_seconds: bool):
        self.names = annotator_names
        self.maps = annotator_maps
        self.sessions = sessions
        self.fs = fs
        self.show_seconds = show_seconds
        self.idx = 0
        self.cursor_ix = 0
        self.vlines: List = []

        # Model (lazy loaded)
        self.model = None
        if torch is None:
            print("[info] PyTorch not available → prediction overlay will be unavailable.")

        # 6 panels total: 4 data + 1 combined (model+original) + 1 combined (annotators)
        nrows = 6
        height = 8.5
        self.fig, self.axs = plt.subplots(
            nrows=nrows, ncols=1, figsize=(13.5, height), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 2.0, 2.0, 1.8, 1.6, 2.2]}
        )
        if not isinstance(self.axs, np.ndarray):
            self.axs = np.array([self.axs])

        # Events
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)

        self.draw()
        plt.tight_layout()
        plt.show()

    def title(self) -> str:
        return f"{self.sessions[self.idx]}    [{self.idx+1}/{len(self.sessions)}]"

    def _apply_xtick_labels(self, xlim_max: int):
        num = 8
        xs = np.linspace(0, max(1, xlim_max), num, dtype=float)
        ax_bottom = self.axs[-1]
        ax_bottom.set_xticks(xs)
        if self.show_seconds:
            ax_bottom.set_xticklabels([f"{v/self.fs:.0f}" for v in xs])
            ax_bottom.set_xlabel("Time (s)")
        else:
            ax_bottom.set_xticklabels([f"{int(v):d}" for v in xs])
            ax_bottom.set_xlabel("Sample index")

    def _update_cursor(self):
        for vl in self.vlines:
            vl.set_xdata([self.cursor_ix, self.cursor_ix])
        self.fig.canvas.draw_idle()

    def draw(self):
        key = self.sessions[self.idx]
        arrays, T_max = build_session_arrays(self.maps, key)
        streams = load_original_streams(key, self.fs)

        # master x-limit across everything
        L = max(T_max, streams["n"])
        if L == 0:
            L = 1
        x_master = np.arange(L, dtype=float)

        # ---- Panel indices ----
        ax_acc, ax_gyr, ax_mag, ax_yaw, ax_ref, ax_anns = self.axs

        # Clear
        for ax in (ax_acc, ax_gyr, ax_mag, ax_yaw, ax_ref, ax_anns):
            ax.cla()

        # ---- Data panels ----
        if streams["n"] > 0:
            n = streams["n"]
            s = np.arange(n, dtype=float)
            ax_acc.plot(s, streams["acc"][:,0], label="ax")
            ax_acc.plot(s, streams["acc"][:,1], label="ay")
            ax_acc.plot(s, streams["acc"][:,2], label="az")
            ax_acc.set_ylabel("Accel (m/s²)")
            ax_acc.legend(loc="upper left")

            ax_gyr.plot(s, streams["gyr"][:,0], label="gx")
            ax_gyr.plot(s, streams["gyr"][:,1], label="gy")
            ax_gyr.plot(s, streams["gyr"][:,2], label="gz")
            ax_gyr.set_ylabel("Gyro (rad/s)")
            ax_gyr.legend(loc="upper left")

            ax_mag.plot(s, streams["mag"][:,0], label="mx")
            ax_mag.plot(s, streams["mag"][:,1], label="my")
            ax_mag.plot(s, streams["mag"][:,2], label="mz")
            ax_mag.set_ylabel("Mag (µT)")
            ax_mag.legend(loc="upper left")

            ax_yaw.plot(s, streams["yaw_cos"], label="cos(yaw)")
            ax_yaw.plot(s, streams["yaw_sin"], label="sin(yaw)")
            ax_yaw.set_ylabel("Yaw")
            ax_yaw.set_ylim(-1.05, 1.05)
            ax_yaw.legend(loc="upper left")
        else:
            ax_acc.text(0.5, 0.5, "(no original data found)", ha="center", va="center")
        ax_acc.set_title(self.title())

        # ---- Combined Reference panel: model + original (with offsets) ----
        # Colors
        col_model = "#ff7f0e"
        col_orig  = "#54ed0d"
        off_model = +0.18
        off_orig  = -0.18

        pred = None
        if torch is not None:
            if self.model is None:
                try:
                    self.model, ncls = load_model(DEFAULT_MODEL_PATH)
                    print(f"[inference] Loaded model ({ncls} classes).")
                except Exception as e:
                    print(f"[inference] failed to load model: {e}")
                    self.model = None
            pred = run_model_prediction(key, self.fs, self.model)

        if pred is not None:
            _plot_labels_step(ax_ref, np.arange(pred.size, dtype=float), pred, color=col_model, lw=1.5, y_offset=off_model, label="Prediction")
        else:
            ax_ref.text(0.5, 0.7, "(predictions unavailable)", ha="center", va="center")

        if streams["orig_labels"] is not None:
            orig = np.asarray(streams["orig_labels"], dtype=float)
            _plot_labels_step(ax_ref, np.arange(orig.size, dtype=float), orig, color=col_orig, lw=1.3, y_offset=off_orig, label="Original")
        else:
            ax_ref.text(0.5, 0.3, "(original labels missing)", ha="center", va="center")

        _format_label_axis(ax_ref, max_offset=max(abs(off_model), abs(off_orig)))
        ax_ref.set_title("Reference labels (Model + Original)", loc="left", fontsize=10)
        ax_ref.legend(loc="upper left")

        # ---- Combined Annotators panel: overlay all with offsets ----
        present = [(name, arr) for name, arr in zip(self.names, arrays) if arr is not None]
        if present:
            offs = _offsets(len(present), max_abs=0.28)
            # color cycle
            prop_cycle = plt.rcParams.get("axes.prop_cycle")
            if prop_cycle is not None:
                base_colors = prop_cycle.by_key().get("color", [])
            else:
                base_colors = []
            # fallback to tab10 if needed
            import itertools
            color_pool = base_colors if base_colors else [plt.cm.tab10(i % 10) for i in range(10)]
            for (i, (name, arr)) in enumerate(present):
                col = color_pool[i % len(color_pool)]
                _plot_labels_step(ax_anns, np.arange(arr.size, dtype=float), arr, color=col, lw=1.6, y_offset=offs[i], label=name)
            _format_label_axis(ax_anns, max_offset=float(np.max(np.abs(offs)) if len(offs) else 0.3))
            ax_anns.legend(loc="upper left", ncols=2, fontsize=9, framealpha=0.95)
        else:
            ax_anns.text(0.5, 0.5, "(no annotator labels found)", ha="center", va="center")
            _format_label_axis(ax_anns, max_offset=0.3)

        ax_anns.set_title("Annotators (overlay)", loc="left", fontsize=10)

        # Align x limits across all panels
        for ax in self.axs:
            ax.set_xlim(0, L-1)

        # (Re)create cursor lines
        self.vlines = [ax.axvline(self.cursor_ix, linestyle="--", alpha=0.6) for ax in self.axs]

        # Bottom x-axis labels formatting
        self._apply_xtick_labels(L-1)

        self.fig.canvas.draw_idle()

    # -------- Events --------
    def on_move(self, event):
        if event.inaxes not in self.axs or event.xdata is None:
            return
        xi = int(round(event.xdata))
        xmin, xmax = self.axs[0].get_xlim()
        xi = int(np.clip(xi, xmin, xmax))
        self.cursor_ix = xi
        self._update_cursor()

    def on_key(self, ev):
        if ev.key == "down":
            if self.idx < len(self.sessions) - 1:
                self.idx += 1
                self.cursor_ix = 0
                self.draw()
        elif ev.key == "up":
            if self.idx > 0:
                self.idx -= 1
                self.cursor_ix = 0
                self.draw()
        elif ev.key == "left":
            self.cursor_ix = max(0, self.cursor_ix - 1)
            self._update_cursor()
        elif ev.key == "right":
            _, xmax = self.axs[0].get_xlim()
            self.cursor_ix = min(int(xmax), self.cursor_ix + 1)
            self._update_cursor()
        elif ev.key == "h":
            self.show_seconds = not self.show_seconds
            self._apply_xtick_labels(int(self.axs[0].get_xlim()[1]))
            self.fig.canvas.draw_idle()
        elif ev.key in ("escape", "q"):
            plt.close(self.fig)

# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser(description="Label comparison with data panels, combined model+original, and combined annotators overlays.")
    ap.add_argument("--mode", choices=["intersection", "union"], default="intersection",
                    help="Which session set to browse across annotators.")
    ap.add_argument("--filter", default=None,
                    help="Regex on session key (derived from filename before 'human_label').")
    ap.add_argument("--fs", type=float, default=FS_DEFAULT, help="Sampling rate (Hz) for seconds labels.")
    ap.add_argument("--samples", action="store_true", help="(x-data are samples; this toggles axis labels only).")
    args = ap.parse_args()

    roots = [os.path.abspath(os.path.expanduser(p)) for p in ANNOTATOR_ROOTS]
    exists = [r for r in roots if os.path.isdir(r)]
    if not exists:
        ap.error("No valid annotator roots. Edit ANNOTATOR_ROOTS in the script.")

    # Build maps for each annotator
    annotator_names: List[str] = []
    annotator_maps: List[Dict[str, str]] = []
    for r in exists:
        name, mapping = scan_annotator(r, args.filter)
        annotator_names.append(name)
        annotator_maps.append(mapping)

    key_sets = [set(m.keys()) for m in annotator_maps]
    if not key_sets:
        ap.error("No sessions found in any annotator folder.")
    sessions = sorted(set.intersection(*key_sets)) if args.mode == "intersection" else sorted(set.union(*key_sets))
    if not sessions:
        ap.error("No sessions to display (try --mode union or relax --filter).")

    # DATA_ROOT sanity
    data_root_abs = os.path.abspath(os.path.expanduser(DATA_ROOT))
    if not os.path.isdir(data_root_abs):
        print(f"[warn] DATA_ROOT not found: {data_root_abs} — top panels may be empty.")
    else:
        print(f"[info] Using DATA_ROOT: {data_root_abs}")

    SessionBrowser(
        annotator_names=annotator_names,
        annotator_maps=annotator_maps,
        sessions=sessions,
        fs=args.fs,
        show_seconds=not args.samples
    )

if __name__ == "__main__":
    main()
