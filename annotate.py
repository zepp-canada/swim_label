#!/usr/bin/env python3
"""
Interactive labeling GUI for swim data sessions.

What this does:
- Loads one session at a time (robust to missing acc/gyro/mag files).
- Computes yaw EXACTLY via utils.get_orientation (your reference math),
  then plots cos(yaw) and sin(yaw) in the yaw panel (range -1..1).
- Lets you paint labels with the mouse/keys and save to human_label.csv.
- Runs model predictions by default on session load (RAW counts),
  and you can toggle visibility with 'p'.

Controls:
  Mouse hover: vertical cursor across plots
  Right-click: first click=start, second=end → fills span with current label
  Scroll: X-axis zoom around mouse
  Keys:
    0,1,2,3 = Rest/Swim/Turn/Dolphin
    ←/→     = move cursor by 1 sample
    ↑/↓     = prev/next session
    S       = save labels to '<session>/human_label.csv'
    R       = reset labels to original
    P       = toggle model prediction overlay
    Esc     = discard unsaved changes & reload session
  Text box (top-left): enter session ID & press Enter to jump
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
from matplotlib.widgets import TextBox

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

DATA_ROOT = "engineer_splits/subhra_original"
FS = 25.0
SCROLL_ZOOM_FACTOR = 1.2
HUMAN_LABEL_FILE = "human_label.csv"

# Auto-merge tiny REST gaps after mouse labeling
MIN_REST_SECONDS = 0.4  # gaps shorter than this are filled with the longer neighbor label

CLASS_NAMES = {0: "Rest", 1: "Swim", 2: "Turn", 3: "Dolphin"}
CLASS_COLORS = {0: "#999999", 1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}
LABEL_ALPHA = 0.14

def _hex_to_rgba(hex_color: str, alpha: float = LABEL_ALPHA):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, alpha)

LABEL_BG = {k: _hex_to_rgba(v, LABEL_ALPHA) for k, v in CLASS_COLORS.items()}

# ---------------------- Yaw input sanitation (pre-call only) ----------------------
# We DO NOT change your yaw math. We only prepare inputs so the KF doesn't blow up.

def _ffill_bfill_nan(arr: np.ndarray) -> np.ndarray:
    """Forward/back-fill NaNs column-wise. Returns float32."""
    if arr.size == 0:
        return arr.astype(np.float32)
    df = pd.DataFrame(arr).replace([np.inf, -np.inf], np.nan)
    df = df.fillna(method="ffill").fillna(method="bfill")
    return df.to_numpy(dtype=np.float32)

def _stabilize_mag(mag: np.ndarray, tiny_norm: float = 1e-9) -> np.ndarray:
    """Replace rows with ~zero norm by nearest valid row to avoid degenerate solves."""
    m = mag.copy()
    if m.shape[0] == 0:
        return m
    norms = np.linalg.norm(m, axis=1)
    bad = norms < tiny_norm
    if not bad.any():
        return m
    good = np.where(~bad)[0]
    if good.size == 0:
        return m  # all degenerate; caller will skip yaw
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
    """
    Run utils.get_orientation (KF + euler + smoothing + wrap) on sanitized inputs
    and return cos(yaw) and sin(yaw). We reinsert NaN where original rows were invalid.
    """
    T = acc_scaled.shape[0]
    yaw_cos = np.full(T, np.nan, dtype=np.float32)
    yaw_sin = np.full(T, np.nan, dtype=np.float32)
    if T == 0:
        return yaw_cos, yaw_sin

    # Valid rows: acc & mag must be finite; (gyro may be used by KF but we fill it too)
    finite_acc = np.isfinite(acc_scaled).all(axis=1)
    finite_mag = np.isfinite(mag_scaled).all(axis=1)
    valid = finite_acc & finite_mag
    if valid.sum() < 5:
        return yaw_cos, yaw_sin

    # Gap-free copies for the call (keep full length)
    acc_f = _ffill_bfill_nan(acc_scaled)
    gyro_f = _ffill_bfill_nan(gyro_scaled)
    mag_f = _stabilize_mag(_ffill_bfill_nan(mag_scaled))

    if not (np.isfinite(acc_f).all() and np.isfinite(mag_f).all()):
        return yaw_cos, yaw_sin

    try:
        # EXACT orientation math lives here
        _, _, _, _, yaw_deg, _, _ = get_orientation(acc_f, gyro_f, mag_f)
        yaw_deg = np.asarray(yaw_deg, dtype=np.float32)  # already smoothed + wrapped [0,360)

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

# ---------------------- GUI ----------------------

class LabelGUI:
    def __init__(self, data_root: str, fs: float = 25.0, model_path: str = DEFAULT_MODEL_PATH, filter_sessions=None):
        self.data_root = data_root
        self.fs = fs
        self.model_path = model_path
        self.min_rest_samples = max(1, int(round(MIN_REST_SECONDS * self.fs)))

        all_files = glob(os.path.join(data_root, "*"))
        if filter_sessions:
            all_files = [p for p in all_files if os.path.basename(p) in filter_sessions]

        self.sessions = sorted([p for p in all_files if os.path.isdir(p)])
        if not self.sessions:
            raise RuntimeError(f"No session folders found under {data_root}")
        self.N = len(self.sessions)
        self.idx = 0

        # state
        self.current_mode = 1
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
        self.labels = np.zeros(0, dtype=int)
        self.orig_labels = None

        # predictions
        self.pred_labels = None
        self.model = None
        self.show_pred = True  # toggle with 'p'

        # plotting
        self._bg_patches = []
        self._diff_patches = []
        self._y_lims = {}
        self.label_line = None
        self.orig_line = None
        self.pred_line = None

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
        self.ax_text = self.fig.add_axes([0.80, 0.94, 0.18, 0.04])
        self.ax_text.axis("off")
        self.mode_text = self.ax_text.text(
            0.0, 0.5, "", ha="left", va="center", fontsize=11, fontweight="bold"
        )

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

        # SCALED units for plotting + yaw; NaN for missing streams so we draw empty plots
        X_scaled, labels = get_session_data(
            self.sessions[self.idx], fs=self.fs, add_gyro=True, scale=True, nan_for_missing=False
        )

        n = X_scaled.shape[0]
        self.T = n
        self.time = (np.arange(n) / self.fs) if n > 0 else np.array([], dtype=float)

        self.acc = X_scaled[:, 0:3] if n else np.zeros((0, 3), dtype=np.float32)
        self.gyro = X_scaled[:, 3:6] if n else np.zeros((0, 3), dtype=np.float32)
        self.mag = X_scaled[:, 6:9] if n else np.zeros((0, 3), dtype=np.float32)

        self.orig_labels = load_original_labels(self.sessions[self.idx], n)

        # ---------- Yaw ----------
        if n == 0 or np.all(np.isnan(self.acc)) or np.all(np.isnan(self.mag)):
            self.yaw_cos = np.full(n, np.nan, dtype=np.float32)
            self.yaw_sin = np.full(n, np.nan, dtype=np.float32)
            print("[yaw] skipped: missing/empty acc or mag streams.")
        else:
            self.yaw_cos, self.yaw_sin = _compute_yaw_cos_sin(self.acc, self.gyro, self.mag)

        # Labels
        self.labels = labels if labels is not None else np.zeros(n, dtype=int)

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
        pd.DataFrame(self.labels).to_csv(path, header=False, index=False)
        print(f"[save] {path}")
        self.changed = False

    def _reset_labels(self):
        if self.orig_labels is not None:
            self.labels = self.orig_labels.copy()
            self.changed = True
            self._update_labels_plot()
            print("[labels] reset to original.")
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
            # create if missing (visible state controlled below)
            self.pred_line = self.axs[4].step(
                self.time, self.pred_labels+0.3, where="post",
                linewidth=1.5, color="#ff7f0e", alpha=0.9, label="Prediction"
            )[0]
        self.pred_line.set_visible(self.show_pred)
        # Refresh legend to reflect visibility
        self.axs[4].legend(loc="upper left")
        self.fig.canvas.draw_idle()

    # -------- Auto-merge short REST gaps --------
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
        """Single pass: merge REST (0) spans shorter than threshold into the longer neighbor.
        Returns True if any change was made."""
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

            # Choose neighbor with longer duration; tie → prefer left
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
                continue  # isolated tiny rest with no neighbors

            _, _, neighbor_lab = choose
            if neighbor_lab == 0:
                continue  # nothing to merge into

            self.labels[i0:i1+1] = int(neighbor_lab)
            changed = True
        return changed

    def _merge_short_rests(self):
        # Do a few passes in case adjacent merges enable further merges
        for _ in range(3):
            if not self._merge_short_rests_once():
                break

    # -------- Highlights --------
    def _clear_highlights(self):
        for art in self._bg_patches + self._diff_patches:
            try:
                art.remove()
            except Exception:
                pass
        self._bg_patches = []
        self._diff_patches = []

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

    def _draw_differences(self):
        """Overlay regions where current labels differ from original labels."""
        if self.T == 0 or self.orig_labels is None:
            return
        diffs = self.labels != self.orig_labels
        if not diffs.any():
            return
        # find contiguous spans of differences
        in_span = False
        start = 0
        for i, d in enumerate(diffs):
            if d and not in_span:
                in_span = True
                start = i
            elif not d and in_span:
                end = i - 1
                x0, x1 = self.time[start], self.time[end]
                for ax in self.axs[:4]:
                    self._diff_patches.append(
                        ax.axvspan(
                            x0, x1,
                            facecolor=(0.5, 0.0, 0.5, 0.25),   # strong purple overlay
                            edgecolor=(0.3, 0.0, 0.3, 0.9),   # darker outline
                            linewidth=1.2,
                            zorder=5
                        )
                    )
                in_span = False
        if in_span:
            end = len(diffs) - 1
            x0, x1 = self.time[start], self.time[end]
            for ax in self.axs[:4]:
                self._diff_patches.append(
                    ax.axvspan(
                        x0, x1,
                        facecolor=(0.5, 0.0, 0.5, 0.25),
                        edgecolor=(0.3, 0.0, 0.3, 0.9),
                        linewidth=1.2,
                        zorder=5
                    )
                )


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

        # Labels
        if self.label_line is None:
            if self.T:
                self.label_line = self.axs[4].step(
                    self.time, self.labels, where="post", linewidth=2, color="#4c78a8", label="Label"
                )[0]
            else:
                self.label_line = self.axs[4].step([], [], where="post", linewidth=2, color="#4c78a8", label="Label")[0]
        else:
            self.label_line.set_data(self.time, self.labels)

        # Predictions overlay
        if self.pred_labels is not None and self.T:
            if self.pred_line is None:
                self.pred_line = self.axs[4].step(
                    self.time, self.pred_labels+0.3, where="post",
                    linewidth=1.5, color="#ff7f0e", alpha=0.9, label="Prediction"
                )[0]
            else:
                self.pred_line.set_data(self.time, self.pred_labels)
            self.pred_line.set_visible(self.show_pred)

        # Original labels overlay
        if self.orig_labels is not None and self.T:
            if self.orig_line is None:
                self.orig_line = self.axs[4].step(
                    self.time, self.orig_labels-0.3, where="post",
                    linewidth=1.1, color="#54ed0d", alpha=0.9,
                    label="Original"
                )[0]
            else:
                self.orig_line.set_data(self.time, self.orig_labels)

        self.axs[4].set_yticks([0, 1, 2, 3], labels=list(CLASS_NAMES.values()))
        self.axs[4].set_ylim(-0.5, 3.5)
        self.axs[4].set_xlabel("time (s)")
        self.axs[4].legend(loc="upper left")

        # Highlights
        self._clear_highlights()
        self._draw_highlights()
        self._draw_differences()

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
        if self.label_line is not None:
            self.label_line.set_data(self.time, self.labels)
        # redraw highlights including diffs
        self._clear_highlights()
        self._draw_highlights()
        self._draw_differences()
        self.fig.canvas.draw_idle()

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
        self.mode_text.set_text(f"Mode: {self.current_mode} ({cname})")
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
        - Second right-click sets the end and fills the range with the current label
        Works even if the toolbar is in zoom/pan mode.
        """
        # Use right mouse button only
        if event.button != 3 or event.inaxes not in self.axs or event.xdata is None or not self.T:
            return

        ix = int(round(event.xdata * self.fs))
        ix = int(np.clip(ix, 0, self.T - 1))

        if self.pending_start is None:
            self.pending_start = ix
        else:
            i0, i1 = min(self.pending_start, ix), max(self.pending_start, ix)
            self.labels[i0:i1 + 1] = self.current_mode
            self.pending_start = None
            # --- auto-merge tiny Rest gaps after labeling ---
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
            # toggle visibility only
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
    FILTER_SESSIONS = None
    LabelGUI(DATA_ROOT, fs=FS, filter_sessions=FILTER_SESSIONS)
