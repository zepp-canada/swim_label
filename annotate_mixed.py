#!/usr/bin/env python3
"""
Interactive labeling GUI for swim data sessions (mixed base + stroke labels).

Differences vs. the original:
- 4 stroke toggle buttons: Free / Back / Breast / Fly.
- Stroke mode ON: right-click range converts existing Swim(1) to stroke strings ('free','back','breast','fly').
- Stroke mode OFF: right-click sets base labels (0/1/2/3) as before.
- human_label.csv stores either an int (0/1/2/3) or a stroke string.
- Bottom label plot:
    * The **current human label is always a single dark-blue step line**.
    * The **original** is green (offset -0.3), and **prediction** is orange (offset +0.3).
    * Stroke regions are shown as **background shaded bands** (soft alpha).
- Top 4 panels retain base-class shading (Rest/Swim/Turn/Dolphin). The purple
  "differences" overlay has been removed (less distraction).

Filtering sessions by KEYWORDS:
- Pass case-insensitive keywords and only folders whose names contain any keyword are shown.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from glob import glob

# Try a modern, faster backend; fall back to TkAgg.
import matplotlib
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")
for _bk in ("QtAgg", "MacOSX", "TkAgg"):
    try:
        matplotlib.use(_bk, force=True)
        print(f"[ui] Using Matplotlib backend: {_bk}")
        break
    except Exception as _e:
        print(f"[ui] Backend {_bk} unavailable: {_e}")

# Disable default 's' = save-figure key so our save labels doesn't open a dialog
matplotlib.rcParams['keymap.save'] = []

import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button as MplButton

# Torch is optional; only needed for predictions
try:
    import torch  # noqa: F401
except Exception:
    torch = None

# Local project helpers (already in your repo)
from utils import get_orientation  # ← EXACT yaw math
from data_io import get_session_data, load_original_labels
from model_infer import (
    load_model,
    predict_arrays_argmax,
    smooth_label,
    DEFAULT_MODEL_PATH,
)

# ---------------------- Config ----------------------

DATA_ROOT = "data/all_train_clean"
FS = 25.0
SCROLL_ZOOM_FACTOR = 1.2
HUMAN_LABEL_FILE = "human_label.csv"

# Auto-merge tiny REST gaps after base mouse labeling
MIN_REST_SECONDS = 0.4  # gaps shorter than this are filled with the longer neighbor label

CLASS_NAMES = {0: "Rest", 1: "Swim", 2: "Turn", 3: "Dolphin"}
CLASS_COLORS = {0: "#999999", 1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}
LABEL_ALPHA = 0.14

# Stroke categories (strings saved in CSV when used)
STROKES = ["free", "back", "breast", "fly"]
STROKE_COLORS = {
    "free":   "#1f77b4",  # blue (plain swim color)
    "back":   "#9467bd",  # purple
    "breast": "#e377c2",  # pink
    "fly":    "#17becf",  # teal
}
STROKE_BAND_ALPHA = 0.22  # shaded band alpha in bottom label plot

# Current human label line color (dark blue)
CURRENT_LABEL_COLOR = "#4c78a8"

def _hex_to_rgba(hex_color: str, alpha: float = LABEL_ALPHA):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, alpha)

LABEL_BG = {k: _hex_to_rgba(v, LABEL_ALPHA) for k, v in CLASS_COLORS.items()}

# ---------------------- Yaw input sanitation (pre-call only) ----------------------

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
    for i in range(m.shape[0]):        # forward fill
        if bad[i]:
            m[i] = m[last]
        else:
            last = i
    last = good[-1]
    for i in range(m.shape[0] - 1, -1, -1):  # backward fill
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

        if np.isfinite(yaw_deg).any():
            print(f"[yaw] computed cos/sin: valid={int(valid.sum())}/{T}")
        else:
            print("[yaw] all NaN after computation.")
    except Exception as e:
        print(f"[yaw] get_orientation failed after sanitation: {e}")

    return yaw_cos, yaw_sin

# ---------------------- Human labels I/O (mixed) ----------------------

def _load_human_labels_csv(path: str, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Read human_label.csv where each row is either an int (0..3) or one of
    'free','back','breast','fly'. Returns (base_labels[int], strokes[object]).
    """
    if not os.path.isfile(path):
        return None
    try:
        ser = pd.read_csv(path, header=False).iloc[:, 0]
    except Exception as e:
        print(f"[human] failed to read {path}: {e}")
        return None

    vals = ser.tolist()
    if len(vals) < n:
        vals = vals + [0] * (n - len(vals))
    elif len(vals) > n:
        vals = vals[:n]

    base = np.zeros(n, dtype=int)
    strokes = np.array([None] * n, dtype=object)

    for i, v in enumerate(vals):
        if isinstance(v, str):
            s = v.strip().lower()
            if s in STROKES:
                base[i] = 1
                strokes[i] = s
            else:
                try:
                    base[i] = int(float(s))
                except Exception:
                    base[i] = 0
        else:
            try:
                base[i] = int(v)
            except Exception:
                base[i] = 0

    return base, strokes

def _serialize_human_labels(base: np.ndarray, strokes: np.ndarray) -> pd.Series:
    """
    Produce a 1-col Series of length T where Swim with stroke writes the stroke string,
    otherwise writes the base integer.
    """
    out = []
    for i, b in enumerate(base):
        if b == 1 and strokes[i] is not None:
            out.append(str(strokes[i]))
        else:
            out.append(int(b))
    return pd.Series(out)

# ---------------------- GUI ----------------------

class LabelGUI:
    def __init__(self, data_root: str, fs: float = 25.0, model_path: str = DEFAULT_MODEL_PATH,
                 filter_keywords=None):

        self.data_root = data_root
        self.fs = fs
        self.model_path = model_path
        self.min_rest_samples = max(1, int(round(MIN_REST_SECONDS * self.fs)))

        # Grab immediate subdirectories under data_root
        all_entries = glob(os.path.join(data_root, "*"))
        dirs = [p for p in all_entries if os.path.isdir(p)]

        # Keyword filter (case-insensitive) on folder *names*
        if filter_keywords:
            kws = [k.lower() for k in filter_keywords]
            dirs = [p for p in dirs if any(k in os.path.basename(p).lower() for k in kws)]

        self.sessions = sorted(dirs)
        if not self.sessions:
            raise RuntimeError(f"No session folders found under {data_root} with keywords={filter_keywords}")
        self.N = len(self.sessions)
        self.idx = 0

        # state
        self.current_mode = 1  # base label mode (0..3)
        self.stroke_mode: str | None = None  # 'free'|'back'|'breast'|'fly'|None
        self.pending_start = None
        self.cursor_ix = 0
        self.changed = False

        # data
        self.T = 0
        self.time = np.array([], dtype=float)
        self.acc = np.zeros((0, 3), dtype=np.float32)
        self.gyro = np.zeros((0, 3), dtype=np.float32)
        self.mag = np.zeros((0, 3), dtype=np.float32)
        self.yaw_cos = np.zeros(0, dtype=np.float32)
        self.yaw_sin = np.zeros(0, dtype=np.float32)

        # labels (base ints) + strokes (strings or None)
        self.labels = np.zeros(0, dtype=int)
        self.strokes = np.array([], dtype=object)
        self.orig_labels = None  # original (ints only)

        # predictions
        self.pred_labels = None
        self.model = None
        self.show_pred = True  # toggle with 'p'

        # plotting
        self._bg_patches = []
        self._stroke_patches = []  # shaded stroke bands in bottom axis
        self._y_lims = {}

        # bottom-axis line handles
        self.label_line = None      # current human label (dark blue)
        self.orig_line = None       # original (green, -0.3)
        self.pred_line = None       # prediction (orange, +0.3)

        # stroke buttons
        self.stroke_buttons: dict[str, MplButton] = {}

        self._build_ui()
        self._load_session(0)
        plt.show()

    # -------- Paths --------
    def _session_name(self, i: int) -> str:
        return os.path.basename(self.sessions[i])

    def _human_label_path(self, i: int) -> str:
        return os.path.join(self.sessions[i], HUMAN_LABEL_FILE)

    # -------- UI scaffold --------
    def _build_ui(self):
        # 5 rows: Acc, Gyro, Mag, Yaw (cos/sin), Labels
        self.fig, self.axs = plt.subplots(
            5, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [3, 3, 3, 2, 1]}
        )
        self.axs[0].set_title("Accelerometer")
        self.axs[1].set_title("Gyroscope")
        self.axs[2].set_title("Magnetometer")
        self.axs[3].set_title("cos/sin(yaw) — utils.get_orientation (reference)")
        self.axs[4].set_title("Labels")

        self.axs[0].set_ylabel("m/s²")
        self.axs[1].set_ylabel("rad/s")
        self.axs[2].set_ylabel("μT")
        self.axs[3].set_ylabel("yaw")
        self.axs[4].set_ylabel("Label")

        # Session jump box
        box_rect = self.fig.add_axes([0.02, 0.94, 0.08, 0.04])
        self.textbox = TextBox(box_rect, "ID:", initial="0")
        self.textbox.on_submit(self._on_jump_id)

        # Mode text (top-right)
        self.ax_text = self.fig.add_axes([0.78, 0.94, 0.20, 0.04])
        self.ax_text.axis("off")
        self.mode_text = self.ax_text.text(
            0.0, 0.5, "", ha="left", va="center", fontsize=11, fontweight="bold"
        )

        # Stroke toggle buttons (top-right row) — PRE-COLORED
        btn_y = 0.89
        btn_w = 0.05
        btn_h = 0.04
        btn_gap = 0.01
        btn_x0 = 0.78
        names = ["free", "back", "breast", "fly"]
        labels = ["Free", "Back", "Breast", "Fly"]
        for i, (nm, lab) in enumerate(zip(names, labels)):
            axb = self.fig.add_axes([btn_x0 + i * (btn_w + btn_gap), btn_y, btn_w, btn_h])
            btn = MplButton(axb, lab, color=STROKE_COLORS[nm], hovercolor=STROKE_COLORS[nm])
            btn.on_clicked(lambda _evt, name=nm: self._toggle_stroke_mode(name))
            self.stroke_buttons[nm] = btn
        self._update_stroke_buttons()

        # Events
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)

    # -------- Load/Save --------
    def _load_session(self, i: int):
        self.idx = int(np.clip(i, 0, self.N - 1))
        sess = self._session_name(self.idx)
        print(f"[load] Session {self.idx}/{self.N-1}: {sess}")

        # SCALED units for plotting + yaw; NaN for missing streams
        X_scaled, labels = get_session_data(
            self.sessions[self.idx], fs=self.fs, add_gyro=True, scale=True, nan_for_missing=False
        )

        n = X_scaled.shape[0]
        self.T = n
        self.time = (np.arange(n) / self.fs) if n > 0 else np.array([], dtype=float)

        self.acc = X_scaled[:, 0:3] if n else np.zeros((0, 3), dtype=np.float32)
        self.gyro = X_scaled[:, 3:6] if n else np.zeros((0, 3), dtype=np.float32)
        self.mag  = X_scaled[:, 6:9] if n else np.zeros((0, 3), dtype=np.float32)

        self.orig_labels = load_original_labels(self.sessions[self.idx], n)

        # ---------- Yaw ----------
        if n == 0 or np.all(np.isnan(self.acc)) or np.all(np.isnan(self.mag)):
            self.yaw_cos = np.full(n, np.nan, dtype=np.float32)
            self.yaw_sin = np.full(n, np.nan, dtype=np.float32)
            print("[yaw] skipped: missing/empty acc or mag streams.")
        else:
            self.yaw_cos, self.yaw_sin = _compute_yaw_cos_sin(self.acc, self.gyro, self.mag)

        # Base labels & strokes
        self.labels = labels if labels is not None else np.zeros(n, dtype=int)
        self.strokes = np.array([None] * n, dtype=object)

        # If a human_label.csv exists, prefer it (mixed strings/ints)
        human_path = self._human_label_path(self.idx)
        mixed = _load_human_labels_csv(human_path, n)
        if mixed is not None:
            base, strokes = mixed
            self.labels = base
            self.strokes = strokes
            print(f"[human] Loaded mixed labels from {human_path}")

        # Reset state
        self.pending_start = None
        self.cursor_ix = 0
        self.changed = False
        self.label_line = None
        self.pred_line = None
        self.pred_labels = None
        self.orig_line = None

        self._draw_all()
        self._update_mode_text()

        # fix Y-lims for x-only navigation
        self._y_lims = {
            self.axs[0]: self.axs[0].get_ylim(),
            self.axs[1]: self.axs[1].get_ylim(),
            self.axs[2]: self.axs[2].get_ylim(),
            self.axs[3]: (-1.05, 1.05),
            self.axs[4]: (-0.5, 3.5),
        }
        self.axs[3].set_ylim(*self._y_lims[self.axs[3]])
        self.axs[4].set_ylim(*self._y_lims[self.axs[4]])

        # --- Run predictions by default; 'p' will toggle visibility ---
        self._run_inference_on_current()
        self.show_pred = True
        self._apply_pred_visibility()

    def _save_labels(self):
        path = self._human_label_path(self.idx)
        ser = _serialize_human_labels(self.labels, self.strokes)
        ser.to_csv(path, header=False, index=False)
        print(f"[save] {path}")
        self.changed = False

    def _reset_labels(self):
        if self.orig_labels is not None:
            self.labels = self.orig_labels.copy()
            self.strokes = np.array([None] * len(self.labels), dtype=object)
            self.changed = True
            self._update_labels_plot()
            print("[labels] reset to original (strokes cleared).")
        else:
            print("[labels] no original labels found; nothing to reset.")

    # -------- Inference overlay --------
    def _ensure_model(self):
        if self.model is not None:
            return
        if torch is None:
            print("[inference] PyTorch not available; install torch to show predictions.")
            return
        self.model, num_classes = load_model(self.model_path)
        print(f"[inference] Loaded model ({num_classes} classes).")

    def _run_inference_on_current(self):
        self._ensure_model()
        if self.model is None:
            return

        print(f"[inference] Running on session: {self._session_name(self.idx)}")
        # RAW counts for inference; zero-fill missing so model can run
        X_raw, _ = get_session_data(
            self.sessions[self.idx], fs=self.fs, add_gyro=True, scale=False, nan_for_missing=False
        )
        if X_raw.shape[0] == 0 or X_raw.shape[1] < 9:
            print("[inference] No data or <9 channels. Skipping.")
            return

        X_raw = X_raw.copy()
        X_raw[:, 3:6] = 0.0  # zero gyro (matches training/inference reference)

        acc = X_raw[:, 0:3]
        gyro = X_raw[:, 3:6]
        mag  = X_raw[:, 6:9]

        y_pred = predict_arrays_argmax(acc, gyro, mag, self.model)
        y_pred = smooth_label(y_pred, window=75)

        if len(y_pred) != self.T:
            m = min(len(y_pred), self.T)
            tmp = np.zeros(self.T, dtype=int)
            tmp[:m] = y_pred[:m]
            y_pred = tmp

        self.pred_labels = y_pred
        print("[inference] Overlay ready.")
        self._update_pred_plot()

    def _apply_pred_visibility(self):
        if self.pred_labels is None or not self.T:
            return
        if self.pred_line is None:
            self.pred_line = self.axs[4].step(
                self.time, self.pred_labels+0.3, where="post",
                linewidth=1.5, color="#ff7f0e", alpha=0.9, label="Prediction"
            )[0]
        self.pred_line.set_visible(self.show_pred)
        self.axs[4].legend(loc="upper left")
        self.fig.canvas.draw_idle()

    # -------- Auto-merge short REST gaps (base only) --------
    @staticmethod
    def _label_spans(y: np.ndarray):
        if len(y) == 0:
            return []
        spans = []
        s = 0
        cur = y[0]
        for i in range(1, len(y)):
            if y[i] != cur:
                spans.append((s, i - 1, cur))
                s = i
                cur = y[i]
        spans.append((s, len(y) - 1, cur))
        return spans

    def _merge_short_rests_once(self) -> bool:
        if self.labels.size == 0:
            return False
        changed = False
        spans = self._label_spans(self.labels)
        for idx, (i0, i1, lab) in enumerate(spans):
            if lab != 0:
                continue
            length = i1 - i0 + 1
            if length >= self.min_rest_samples:
                continue

            left = spans[idx - 1] if idx - 1 >= 0 else None
            right = spans[idx + 1] if idx + 1 < len(spans) else None

            choose = None
            if left is not None and right is not None:
                llen = left[1] - left[0] + 1
                rlen = right[1] - right[0] + 1
                choose = left if llen >= rlen else right
            elif left is not None:
                choose = left
            elif right is not None:
                choose = right

            if choose is None:
                continue

            _, _, neighbor_lab = choose
            if neighbor_lab == 0:
                continue

            self.labels[i0:i1+1] = int(neighbor_lab)
            # Clear strokes if the neighbor isn't swim
            if int(neighbor_lab) != 1:
                self.strokes[i0:i1+1] = None
            changed = True
        return changed

    def _merge_short_rests(self):
        for _ in range(3):
            if not self._merge_short_rests_once():
                break

    # -------- Highlights (top 4 panels) --------
    def _clear_highlights(self):
        for art in self._bg_patches:
            try:
                art.remove()
            except Exception:
                pass
        self._bg_patches = []

    def _draw_highlights(self):
        if self.T == 0:
            return
        for (i0, i1, lab) in self._label_spans(self.labels):
            color = LABEL_BG.get(int(lab))
            if color is None:
                continue
            x0 = self.time[i0]
            x1 = self.time[i1]
            for ax in self.axs[:4]:
                self._bg_patches.append(ax.axvspan(x0, x1, facecolor=color, edgecolor="none"))

    # -------- Stroke shading (bottom panel) --------
    def _clear_stroke_bands(self):
        for art in self._stroke_patches:
            try:
                art.remove()
            except Exception:
                pass
        self._stroke_patches = []

    def _draw_stroke_bands(self):
        """Draw shaded background bands for stroke ranges on the bottom axis."""
        ax = self.axs[4]
        if self.T == 0:
            return

        # Find contiguous spans where base==1 and stroke equals a given name
        def stroke_spans(name: str):
            spans = []
            in_span = False
            s = 0
            for i in range(self.T):
                ok = (self.labels[i] == 1) and (self.strokes[i] == name)
                if ok and not in_span:
                    in_span = True
                    s = i
                elif not ok and in_span:
                    spans.append((s, i - 1))
                    in_span = False
            if in_span:
                spans.append((s, self.T - 1))
            return spans

        for nm in STROKES:
            rgba = _hex_to_rgba(STROKE_COLORS[nm], STROKE_BAND_ALPHA)
            for i0, i1 in stroke_spans(nm):
                x0 = self.time[i0]
                x1 = self.time[i1]
                self._stroke_patches.append(
                    ax.axvspan(x0, x1, facecolor=rgba, edgecolor="none", zorder=0.5)
                )

    # -------- Bottom axis label drawing --------
    def _draw_labels_axis(self):
        ax = self.axs[4]
        ax.cla()
        ax.set_ylabel("Label")
        ax.set_yticks([0, 1, 2, 3], labels=list(CLASS_NAMES.values()))
        ax.set_ylim(-0.5, 3.5)
        ax.set_xlabel("time (s)")

        # Stroke shaded bands behind the line
        self._clear_stroke_bands()
        self._draw_stroke_bands()

        if self.T == 0:
            self.label_line = None
            self.pred_line = None
            self.orig_line = None
            return

        # Current human label = single dark-blue step line
        self.label_line = ax.step(
            self.time, self.labels, where="post",
            linewidth=2.0, color=CURRENT_LABEL_COLOR, label="Label"
        )[0]

        # Prediction / Original overlays
        if self.pred_labels is not None and self.T:
            self.pred_line = ax.step(
                self.time, self.pred_labels + 0.3, where="post",
                linewidth=1.5, color="#ff7f0e", alpha=0.9, label="Prediction"
            )[0]
            self.pred_line.set_visible(self.show_pred)

        if self.orig_labels is not None and self.T:
            self.orig_line = ax.step(
                self.time, self.orig_labels - 0.3, where="post",
                linewidth=1.1, color="#54ed0d", alpha=0.9, label="Original"
            )[0]

        ax.legend(loc="upper left")

    def _update_labels_axis(self):
        if self.T == 0:
            return
        ax = self.axs[4]

        # Update main label line
        if self.label_line is None:
            self.label_line = ax.step(
                self.time, self.labels, where="post",
                linewidth=2.0, color=CURRENT_LABEL_COLOR, label="Label"
            )[0]
        else:
            self.label_line.set_data(self.time, self.labels)

        # refresh stroke bands
        self._clear_stroke_bands()
        self._draw_stroke_bands()

        self.axs[4].legend(loc="upper left")
        self.fig.canvas.draw_idle()

    # -------- Drawing --------
    def _draw_all(self):
        for ax in self.axs:
            ax.cla()

        self.axs[0].set_ylabel("m/s²")
        self.axs[1].set_ylabel("rad/s")
        self.axs[2].set_ylabel("μT")
        self.axs[3].set_ylabel("yaw")
        self.axs[4].set_ylabel("Label")

        # Streams
        (self.axs[0].plot(self.time, self.acc) if self.T else self.axs[0].plot([], []))
        (self.axs[1].plot(self.time, self.gyro) if self.T else self.axs[1].plot([], []))
        (self.axs[2].plot(self.time, self.mag) if self.T else self.axs[2].plot([], []))
        self.axs[0].legend(["ax", "ay", "az"], loc="upper left")
        self.axs[1].legend(["gx", "gy", "gz"], loc="upper left")
        self.axs[2].legend(["mx", "my", "mz"], loc="upper left")

        # Yaw (cos/sin)
        if self.T:
            self.axs[3].plot(self.time, self.yaw_cos, label="cos (yaw)")
            self.axs[3].plot(self.time, self.yaw_sin, label="sin (yaw)")
        else:
            self.axs[3].plot([], [], label="cos (yaw)")
            self.axs[3].plot([], [], label="sin (yaw)")
        self.axs[3].legend(loc="upper left")
        self.axs[3].set_ylim(-1.05, 1.05)

        # Label axis (single dark-blue line + shaded bands + overlays)
        self._draw_labels_axis()

        # Highlights on top panels (base class shading only)
        self._clear_highlights()
        self._draw_highlights()

        # Cursor
        xcursor = self.time[self.cursor_ix] if self.T and 0 <= self.cursor_ix < self.T else 0.0
        self.vlines = [ax.axvline(xcursor, linestyle="--", alpha=0.6) for ax in self.axs]
        self.fig.suptitle(f"{self._session_name(self.idx)}  —  [{self.idx}/{self.N-1}]")

    # -------- Small helpers --------
    def _update_cursor(self):
        if not self.T:
            return
        x = self.time[int(np.clip(self.cursor_ix, 0, self.T - 1))]
        for vl in self.vlines:
            vl.set_xdata([x, x])
        self.fig.canvas.draw_idle()

    def _update_labels_plot(self):
        # redraw highlights and update bottom axis line & bands
        self._clear_highlights()
        self._draw_highlights()
        self._update_labels_axis()

    def _update_pred_plot(self):
        if self.pred_labels is None or not self.T:
            return
        if self.pred_line is None:
            self.pred_line = self.axs[4].step(
                self.time, self.pred_labels+0.3, where="post",
                linewidth=1.5, color="#ff7f0e", alpha=0.9, label="Prediction"
            )[0]
        else:
            self.pred_line.set_data(self.time, self.pred_labels)
        self.pred_line.set_visible(self.show_pred)
        self.axs[4].legend(loc="upper left")
        self.fig.canvas.draw_idle()

    def _update_mode_text(self):
        cname = CLASS_NAMES[self.current_mode]
        ccol = CLASS_COLORS[self.current_mode]
        stroke_txt = f" | Stroke: {self.stroke_mode}" if self.stroke_mode else ""
        self.mode_text.set_text(f"Mode: {self.current_mode} ({cname}){stroke_txt}")
        self.mode_text.set_color(ccol)
        self.fig.canvas.draw_idle()

    def _toolbar_mode(self) -> str:
        tb = getattr(self.fig.canvas, "toolbar", None)
        try:
            return "" if tb is None else (tb.mode or "")
        except Exception:
            return ""

    def _clamp_y_limits(self):
        for ax in self.axs:
            if ax in self._y_lims:
                ax.set_ylim(*self._y_lims[ax])

    def _toggle_stroke_mode(self, name: str):
        # toggle on/off; only one stroke active at a time
        self.stroke_mode = None if self.stroke_mode == name else name
        self._update_stroke_buttons()
        self._update_mode_text()
        print(f"[stroke] mode: {self.stroke_mode or 'OFF'}")

    def _update_stroke_buttons(self):
        # colorize buttons; active with high alpha, inactive faded
        for nm, btn in self.stroke_buttons.items():
            active = (self.stroke_mode == nm)
            btn.color = STROKE_COLORS[nm]
            btn.hovercolor = STROKE_COLORS[nm]
            try:
                btn.ax.set_facecolor(STROKE_COLORS[nm])
                btn.ax.patch.set_alpha(0.95 if active else 0.35)
                for spine in btn.ax.spines.values():
                    spine.set_linewidth(2.2 if active else 1.0)
                    spine.set_edgecolor('#000000')
            except Exception:
                pass
        self.fig.canvas.draw_idle()

    # -------- Events --------
    def _on_move(self, event):
        if event.inaxes not in self.axs or event.xdata is None or not self.T:
            return
        ix = int(round(event.xdata * self.fs))
        if 0 <= ix < self.T:
            self.cursor_ix = ix
            self._update_cursor()

    def _on_scroll(self, event):
        if event.inaxes is None or not self.T:
            return
        scale = 1 / SCROLL_ZOOM_FACTOR if event.button == "up" else (
            SCROLL_ZOOM_FACTOR if event.button == "down" else None
        )
        if scale is None:
            return
        cur_xmin, cur_xmax = self.axs[0].get_xlim()
        xdata = event.xdata if event.xdata is not None else 0.5 * (cur_xmin + cur_xmax)
        width = (cur_xmax - cur_xmin) * scale
        new_xmin = xdata - 0.5 * width
        new_xmax = xdata + 0.5 * width
        full_xmin, full_xmax = self.time[0], self.time[-1]
        span = new_xmax - new_xmin
        if new_xmin < full_xmin:
            new_xmin = full_xmin
            new_xmax = new_xmin + span
        if new_xmax > full_xmax:
            new_xmax = full_xmax
            new_xmin = new_xmax - span
        for ax in self.axs:
            ax.set_xlim(new_xmin, new_xmax)
        self._clamp_y_limits()
        self.fig.canvas.draw_idle()

    def _on_draw(self, event):
        self._clamp_y_limits()

    def _on_click(self, event):
        """
        Right-click labeling:
        - First right-click sets the start
        - Second right-click sets the end and either:
            * Stroke mode ON  → set stroke strings for timesteps that are currently Swim(1)
            * Stroke mode OFF → set base labels as before (0/1/2/3)
        Works even if the toolbar is in zoom/pan mode.
        """
        if event.button != 3 or event.inaxes not in self.axs or event.xdata is None or not self.T:
            return

        ix = int(round(event.xdata * self.fs))
        ix = int(np.clip(ix, 0, self.T - 1))

        if self.pending_start is None:
            self.pending_start = ix
        else:
            i0, i1 = min(self.pending_start, ix), max(self.pending_start, ix)
            self.pending_start = None

            if self.stroke_mode is not None:
                # Stroke labeling: only modify existing Swim(1)
                m = (self.labels[i0:i1+1] == 1)
                if m.any():
                    self.strokes[i0:i1+1][m] = self.stroke_mode
                    self.changed = True
                    print(f"[stroke] set '{self.stroke_mode}' over {m.sum()} Swim samples.")
                    self._update_labels_plot()
                else:
                    print("[stroke] no Swim(1) samples in range; nothing changed.")
            else:
                # Base labeling: same as before
                self.labels[i0:i1 + 1] = self.current_mode
                # Manage strokes: clear where not Swim; set to None where newly Swim
                self.strokes[i0:i1+1] = None

                # auto-merge tiny Rest gaps after labeling
                if self.min_rest_samples > 0:
                    before = self.labels.copy()
                    self._merge_short_rests()
                    if not np.array_equal(before, self.labels):
                        print("[labels] merged tiny Rest gaps.")

                self.changed = True
                self._update_labels_plot()

    def _on_key(self, event):
        if event.key in ["0", "1", "2", "3"]:
            self.current_mode = int(event.key)
            self._update_mode_text()
        elif event.key == "left" and self.T:
            self.cursor_ix = max(0, self.cursor_ix - 1)
            self._update_cursor()
        elif event.key == "right" and self.T:
            self.cursor_ix = min(self.T - 1, self.cursor_ix + 1)
            self._update_cursor()
        elif event.key == "s":
            self._save_labels()
        elif event.key == "p":
            self.show_pred = not self.show_pred
            self._apply_pred_visibility()
            print(f"[inference] prediction overlay {'ON' if self.show_pred else 'OFF'}.")
        elif event.key == "r":
            self._reset_labels()
        elif event.key == "escape":
            print("[reload] Discarding changes.")
            self._load_session(self.idx)
        elif event.key == "up":
            self._load_session(max(0, self.idx - 1))
        elif event.key == "down":
            self._load_session(min(self.N - 1, self.idx + 1))

    def _on_jump_id(self, text: str):
        try:
            tid = int(text.strip())
        except Exception:
            print("[jump] Please enter an integer ID.")
            return
        if 0 <= tid < self.N:
            self._load_session(tid)
        else:
            print(f"[jump] ID out of range [0, {self.N-1}]")

# ---------------------- Run ----------------------

if __name__ == "__main__":
    # Case-insensitive substring match on folder names (e.g., "Mixed" or "mixed")
    FILTER_KEYWORDS = ["mixed"]
    LabelGUI(DATA_ROOT, fs=FS, filter_keywords=FILTER_KEYWORDS)
