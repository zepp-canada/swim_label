#!/usr/bin/env python3
"""
Interactive stroke-labeling GUI for swim data sessions (with model init-fill).

This script is **exclusively for stroke labeling**. Base labels (Rest/Swim/Turn/Dolphin)
are produced by another tool and are **not** edited here.

New in this version:
- On session load, we run a stroke-type model and **pre-fill** the stroke labels with
  the model's prediction (restricted to timesteps where base==Swim).
  -> If you press 'S' immediately, those predictions are saved as-is.
- Press **'P'** to toggle OFF model predictions and start from a blank slate (no strokes).
  Press 'P' again to restore the model predictions. (This toggle discards any edits.)

What it does (summary):
- Loads one session at a time (robust to missing acc/gyro/mag files).
- Computes yaw EXACTLY via utils.get_orientation then plots cos(yaw), sin(yaw).
- Lets you paint **stroke types** with the mouse, storing strings into human_label.csv:
    'free', 'back', 'breast', 'fly'   (only where the base label is Swim=1)
- Bottom panel shows **only shaded bands** for stroke regions (no step lines).

Storage format (human_label.csv):
- A single column where each row is either:
    * an integer 0/1/2/3 (Rest/Swim/Turn/Dolphin) — these come from your other tool, or
    * a stroke string: 'free', 'back', 'breast', or 'fly'
      (implies base=Swim(1) at that timestep).

Model files:
- Put your stroke model weights and metadata in STROKE_MODEL_DIR:
    - model.pth
    - parameters.json
    - global_stats.json

Filtering sessions by KEYWORDS:
- Pass case-insensitive keywords; only folders whose names contain any keyword are shown.
  Example: FILTER_KEYWORDS = ["mixed"] matches "Mixed", "MIXED", etc.

Controls:
  Mouse hover: vertical cursor across plots
  Right-click: first click=start, second=end → set the chosen stroke over the range,
               but only where base is Swim(1)
  Scroll: X-axis zoom around mouse
  Keys:
    ←/→     = move cursor by 1 sample
    ↑/↓     = prev/next session
    S       = save labels to '<session>/human_label.csv'
    P       = toggle model-predicted strokes ON/OFF (OFF clears to blank slate)
    R       = clear all strokes (keeps base labels intact)
    Esc     = discard unsaved changes & reload session
  Stroke buttons (top-right): toggle Free/Back/Breast/Fly mode ON/OFF
  Text box (top-left): enter session ID & press Enter to jump
"""

from __future__ import annotations

import os
import json
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

# Torch for stroke inference
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

# Local project helpers
from utils import get_orientation  # ← EXACT yaw math
from data_io import get_session_data

# ---------------------- Config ----------------------

DATA_ROOT = "data/all_train_clean"
FS = 25.0
SCROLL_ZOOM_FACTOR = 1.2
HUMAN_LABEL_FILE = "human_label.csv"

# Directory holding stroke model artifacts (model.pth, parameters.json, global_stats.json)
STROKE_MODEL_DIR = "models/stroke"

# Base classes (we only READ these to know where Swim==1)
CLASS_NAMES = {0: "Rest", 1: "Swim", 2: "Turn", 3: "Dolphin"}
CLASS_COLORS = {0: "#999999", 1: "#1f77b4", 2: "#d62728", 3: "#2ca02c"}
LABEL_ALPHA = 0.14

# Stroke categories (strings saved in CSV)
STROKES = ["free", "back", "breast", "fly"]
# Mapping from model class index → stroke string.
# Model trained with indices: 0=breast, 1=free, 2=back, 3=fly
IDX2STROKE = {0: "breast", 1: "free", 2: "back", 3: "fly"}

STROKE_COLORS = {
    "free":   "#0072B2",  # bright blue
    "back":   "#D55E00",  # strong orange
    "breast": "#CC79A7",  # vivid magenta
    "fly":    "#009E73",  # rich green
}
STROKE_BAND_ALPHA = 0.35  # increase alpha so bands stand out more

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
        ser = pd.read_csv(path, header=None).iloc[:, 0]
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
    Produce a 1-col Series: write stroke string when base==Swim(1) & stroke set,
    otherwise write the base integer.
    """
    out = []
    for i, b in enumerate(base):
        if b == 1 and strokes[i] is not None:
            out.append(str(strokes[i]))
        else:
            out.append(int(b))
    return pd.Series(out)

# ---------------------- Stroke model (minimal pieces) ----------------------

# Minimal network to match training: Conv1d + BiGRU + Linear
class StrokeNet(nn.Module):
    def __init__(self, n_channels=6, n_classes=4, cnn_filters=32, gru_hidden=32, p_drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, cnn_filters, kernel_size=25, stride=1, padding='same')
        self.gru   = nn.GRU(input_size=cnn_filters, hidden_size=gru_hidden, num_layers=1,
                            batch_first=True, bidirectional=True)
        self.drop  = nn.Dropout(p_drop)
        self.classifier = nn.Linear(gru_hidden * 2, n_classes, bias=True)

    def forward(self, x):
        # x: (B,T,C)
        x = x.transpose(1, 2)                  # (B,C,T)
        x = self.conv1(x)
        x = x.transpose(1, 2)                  # (B,T,F)
        x, _ = self.gru(x)                     # (B,T,2H)
        x = self.drop(x)
        return self.classifier(x)              # (B,T,C)

def _scale_signal(signal, mean, std, eps=1e-6):
    mean = np.asarray(mean, dtype=np.float32)
    std  = np.asarray(std, dtype=np.float32)
    std  = np.where(std < eps, 1.0, std)
    scaled = (signal - mean) / (std + eps)
    return np.clip(scaled, -2, 2)

@torch.no_grad()
def _stroke_inference(model, X, global_stats, stride_window, front_window, back_window=None, fill_tail=True, scale_data=False):
    """
    Sliding-window inference, matching the training script's approach.
    X: (T, C)
    """
    device = next(model.parameters()).device
    mu = np.array(global_stats['global_mu'], dtype=np.float32)
    sd = np.array(global_stats['global_std'], dtype=np.float32)
    T = X.shape[0]

    y_pred = -np.ones(T, dtype=np.int64)
    finalized = 0

    for base in range(0, T, stride_window):
        t_end   = min(T, base + front_window)
        t_start = max(0, base - (back_window or base))
        x_slice = X[t_start:t_end]
        if scale_data:
            x_slice = _scale_signal(x_slice, mu, sd)
        logits  = model(torch.from_numpy(x_slice).float().unsqueeze(0).to(device))
        pred    = logits.argmax(dim=2).squeeze(0).cpu().numpy()

        commit_lo = max(finalized, t_start)
        commit_hi = min(base, t_end - front_window)
        if commit_hi > commit_lo:
            sl = commit_lo - t_start
            sh = commit_hi - t_start
            y_pred[commit_lo:commit_hi] = pred[sl:sh]
            finalized = commit_hi

    if fill_tail and finalized < T:
        x_all = _scale_signal(X, mu, sd) if scale_data else X
        logits = model(torch.from_numpy(x_all).float().unsqueeze(0).to(device))
        pred   = logits.argmax(dim=2).squeeze(0).cpu().numpy()
        y_pred[finalized:T] = pred[finalized:T]

    return y_pred

def _read_parameters(model_dir: str):
    p = os.path.join(model_dir, "parameters.json")
    with open(p, "r") as f:
        return json.load(f)

def _read_global_stats(model_dir: str):
    p = os.path.join(model_dir, "global_stats.json")
    with open(p, "r") as f:
        return json.load(f)

def _load_model_weights(model: nn.Module, model_dir: str):
    mp = os.path.join(model_dir, "model.pth")
    device = next(model.parameters()).device
    try:
        state = torch.load(mp, map_location=device, weights_only=True)  # torch >= 2.0
    except TypeError:
        state = torch.load(mp, map_location=device)                      # compatibility
    model.load_state_dict(state)
    model.eval()
    print(f"[model] Loaded weights: {mp}")

def _find_model_dir():
    """
    Try a few likely locations for the stroke model files.
    """
    candidates = [
        STROKE_MODEL_DIR,
        ".",
        "models",
        "stroke_model",
        "stroke",
    ]
    for d in candidates:
        if all(os.path.isfile(os.path.join(d, f)) for f in ("model.pth", "parameters.json", "global_stats.json")):
            return d
    return None

def _load_session_counts_for_infer(sess_dir: str, add_gyro: bool) -> np.ndarray | None:
    """
    Load raw-ish counts for inference similar to training.get_data:
      - acc.csv, mag.csv (required), gyro.csv (optional)
      - Apply same per-sensor scaling used during training:
          acc/8000, gyro/8000, mag/500, then mag /= ||mag||
    Returns X: (T, 6 or 9) float32, or None if files missing.
    """
    from os.path import join as pjoin
    colnames = ['x', 'y', 'z', 't', 'p']

    acc_path = pjoin(sess_dir, 'acc.csv')
    mag_path = pjoin(sess_dir, 'mag.csv')
    gyro_path = pjoin(sess_dir, 'gyro.csv')

    if not (os.path.exists(acc_path) and os.path.exists(mag_path)):
        print(f"[infer] Missing acc.csv or mag.csv under {sess_dir}")
        return None

    try:
        acc = pd.read_csv(acc_path, skiprows=1, names=colnames)[['x', 'y', 'z']].values
        mag = pd.read_csv(mag_path, skiprows=1, names=colnames)[['x', 'y', 'z']].values
        if add_gyro and os.path.exists(gyro_path):
            gyro = pd.read_csv(gyro_path, skiprows=1, names=colnames)[['x', 'y', 'z']].values
            L = min(len(acc), len(mag), len(gyro))
            acc, mag, gyro = acc[:L], mag[:L], gyro[:L]
        else:
            gyro = None
            L = min(len(acc), len(mag))
            acc, mag = acc[:L], mag[:L]
    except Exception as e:
        print(f"[infer] Failed to read CSVs in {sess_dir}: {e}")
        return None

    # Replace NaNs/Infs
    acc = np.nan_to_num(acc)
    mag = np.nan_to_num(mag)
    if gyro is not None:
        gyro = np.nan_to_num(gyro)

    # Training-style scaling
    acc = acc / 8000.0
    mag = mag / 500.0
    # Normalize mag vector
    mag = mag / (np.linalg.norm(mag, axis=1, keepdims=True) + 1e-6)
    if gyro is not None:
        gyro = gyro / 8000.0

    if add_gyro and gyro is not None:
        X = np.concatenate([acc, gyro, mag], axis=1)
    else:
        X = np.concatenate([acc, mag], axis=1)

    return X.astype(np.float32)

# ---------------------- GUI ----------------------

class LabelGUI:
    def __init__(self, data_root: str, fs: float = 25.0, filter_keywords=None):

        self.data_root = data_root
        self.fs = fs

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
        self.labels = np.zeros(0, dtype=int)       # READ-ONLY here; not modified
        self.strokes = np.array([], dtype=object)  # what we edit

        # stroke model
        self.model_dir: str | None = None
        self.model: StrokeNet | None = None
        self.model_params = None
        self.model_stats = None
        self.pred_strokes = None   # np.ndarray of stroke strings predicted per timestep (unrestricted)
        self.show_pred = True      # if True, we fill strokes with predictions; False => blank slate

        # plotting
        self._bg_patches = []       # top panels base shading
        self._stroke_patches = []   # bottom stroke bands
        self._y_lims = {}

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
        # 5 rows: Acc, Gyro, Mag, Yaw (cos/sin), Strokes (bands only)
        self.fig, self.axs = plt.subplots(
            5, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [3, 3, 3, 2, 1]}
        )
        self.axs[0].set_title("Accelerometer")
        self.axs[1].set_title("Gyroscope")
        self.axs[2].set_title("Magnetometer")
        self.axs[3].set_title("cos/sin(yaw) — utils.get_orientation (reference)")
        self.axs[4].set_title("Strokes")

        self.axs[0].set_ylabel("m/s²")
        self.axs[1].set_ylabel("rad/s")
        self.axs[2].set_ylabel("μT")
        self.axs[3].set_ylabel("yaw")
        self.axs[4].set_ylabel("")

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

    # -------- Model bootstrap --------
    def _ensure_model(self):
        if torch is None or nn is None:
            print("[model] PyTorch not available; cannot run stroke inference.")
            return
        if self.model is not None and self.model_dir is not None:
            return
        d = _find_model_dir()
        if d is None:
            print(f"[model] Could not find model files in '{STROKE_MODEL_DIR}' or common fallbacks.")
            return
        try:
            params = _read_parameters(d)
            stats  = _read_global_stats(d)
        except Exception as e:
            print(f"[model] Failed to read parameters/global_stats in {d}: {e}")
            return

        n_channels   = int(params.get("n_channels", 6))
        n_classes    = int(params.get("n_classes", 4))
        cnn_filters  = int(params.get("cnn_filters", 32))
        gru_hidden   = int(params.get("gru_hidden", 32))
        p_drop       = float(params.get("p_drop", 0.0))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = StrokeNet(n_channels=n_channels, n_classes=n_classes,
                          cnn_filters=cnn_filters, gru_hidden=gru_hidden, p_drop=p_drop).to(device)
        try:
            _load_model_weights(model, d)
        except Exception as e:
            print(f"[model] Failed to load weights: {e}")
            return

        self.model_dir = d
        self.model = model
        self.model_params = params
        self.model_stats  = stats
        print(f"[model] Ready (dir={d})")

    # -------- Load/Save --------
    def _load_session(self, i: int):
        self.idx = int(np.clip(i, 0, self.N - 1))
        sess = self._session_name(self.idx)
        print(f"[load] Session {self.idx}/{self.N-1}: {sess}")

        X_scaled, base_labels = get_session_data(
            self.sessions[self.idx], fs=self.fs, add_gyro=True, scale=True, nan_for_missing=False
        )

        n = X_scaled.shape[0]
        self.T = n
        self.time = (np.arange(n) / self.fs) if n > 0 else np.array([], dtype=float)

        self.acc = X_scaled[:, 0:3] if n else np.zeros((0, 3), dtype=np.float32)
        self.gyro = X_scaled[:, 3:6] if n else np.zeros((0, 3), dtype=np.float32)
        self.mag  = X_scaled[:, 6:9] if n else np.zeros((0, 3), dtype=np.float32)

        # Yaw
        if n == 0 or np.all(np.isnan(self.acc)) or np.all(np.isnan(self.mag)):
            self.yaw_cos = np.full(n, np.nan, dtype=np.float32)
            self.yaw_sin = np.full(n, np.nan, dtype=np.float32)
        else:
            self.yaw_cos, self.yaw_sin = _compute_yaw_cos_sin(self.acc, self.gyro, self.mag)

        # Base labels
        self.labels = base_labels if base_labels is not None else np.zeros(n, dtype=int)
        self.strokes = np.array([None] * n, dtype=object)

        # --- PRIORITY: human labels if they exist ---
        human_path = self._human_label_path(self.idx)
        mixed = _load_human_labels_csv(human_path, n)
        if mixed is not None:
            base, prev_strokes = mixed
            self.labels = base
            self.strokes = prev_strokes
            self.show_pred = False
            print(f"[human] Loaded saved strokes from {human_path}")
        else:
            # Otherwise, run model inference
            self._ensure_model()
            if self.model is not None and self.T > 0:
                self._run_stroke_inference_on_current()
                self._apply_predictions_to_strokes()
                self.show_pred = True
                print("[infer] Applied model predictions (no human labels).")
            else:
                self.strokes[:] = None
                self.show_pred = False
                print("[infer] No model available; blank slate.")

        self.pending_start = None
        self.cursor_ix = 0
        self.changed = False

        self._draw_all()
        self._update_mode_text()

    def _run_stroke_inference_on_current(self):
        if self.model is None or self.model_params is None or self.model_stats is None:
            return

        add_gyro   = bool(self.model_params.get("add_gyro", False))
        scale_data = bool(self.model_params.get("scale_data", False))
        fs         = int(self.model_params.get("fs", 25))
        front_s    = float(self.model_params.get("front_window_s", 3))
        stride_s   = float(self.model_params.get("stride_s", 5))

        # Choose a conservative back_window: use the MAX from back_schedule if available
        back_s = None
        try:
            bs = self.model_params.get("back_schedule", [])
            if isinstance(bs, list) and bs:
                back_s = max(float(stage.get("back_s", 0)) for stage in bs)
        except Exception:
            back_s = None

        front_win  = int(round(front_s * fs))
        stride_win = int(round(stride_s * fs))
        back_win   = int(round(back_s * fs)) if back_s is not None else None

        # Load raw counts for inference
        X = _load_session_counts_for_infer(self.sessions[self.idx], add_gyro=add_gyro)
        if X is None:
            raise RuntimeError("No input data for stroke inference.")

        Tm = min(len(X), self.T)
        if Tm <= 0:
            raise RuntimeError("Empty input for stroke inference.")

        # Run inference only on the overlapping portion
        y_idx = _stroke_inference(
            self.model,
            X[:Tm],
            self.model_stats,
            stride_window=stride_win,
            front_window=front_win,
            back_window=back_win,
            scale_data=scale_data,
        )

        # Map to stroke strings
        y_str = np.array([IDX2STROKE.get(int(k), None) for k in y_idx], dtype=object)

        # Build full-length array aligned to self.T
        y_full = np.array([None] * self.T, dtype=object)
        y_full[:Tm] = y_str
        self.pred_strokes = y_full

    def _apply_predictions_to_strokes(self):
        """Copy pred_strokes into self.strokes where base==Swim; else None."""
        if self.pred_strokes is None or self.T == 0:
            self.strokes[:] = None
            return

        ps = self.pred_strokes

        # --- Ensure predictions length matches self.T ---
        if len(ps) != self.T:
            print(f"[warn] Length mismatch: labels={self.T}, preds={len(ps)} → clipping/padding")
            if len(ps) > self.T:
                ps = ps[:self.T]
            else:
                padded = np.array([None] * self.T, dtype=object)
                padded[:len(ps)] = ps
                ps = padded

        swim = (self.labels == 1)
        out = np.array([None] * self.T, dtype=object)
        m = swim & (ps != None)  # noqa: E711
        out[m] = ps[m]
        self.strokes = out
        self.changed = True


    def _save_labels(self):
        path = self._human_label_path(self.idx)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ser = _serialize_human_labels(self.labels, self.strokes)
        ser.to_csv(path, header=False, index=False)
        print(f"[save] {path}")
        self.changed = False

    def _clear_strokes(self):
        self.strokes[:] = None
        self.changed = True
        self._update_labels_plot()
        print("[strokes] cleared.")

    # -------- Highlights (top 4 panels use base shading only) --------
    def _clear_highlights(self):
        for art in self._bg_patches:
            try:
                art.remove()
            except Exception:
                pass
        self._bg_patches = []

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

        # Clean bottom panel aesthetics
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("time (s)")

        # Legend proxies
        try:
            import matplotlib.patches as mpatches
            handles = [mpatches.Patch(color=STROKE_COLORS[nm], label=nm.title()) for nm in STROKES]
            ax.legend(handles=handles, loc="upper left")
        except Exception:
            pass

    # -------- Bottom axis orchestration --------
    def _draw_strokes_axis(self):
        ax = self.axs[4]
        ax.cla()
        self._clear_stroke_bands()
        self._draw_stroke_bands()

    def _update_strokes_axis(self):
        self._clear_stroke_bands()
        self._draw_stroke_bands()
        self.fig.canvas.draw_idle()

    # -------- Drawing --------
    def _draw_all(self):
        for ax in self.axs:
            ax.cla()

        self.axs[0].set_ylabel("m/s²")
        self.axs[1].set_ylabel("rad/s")
        self.axs[2].set_ylabel("μT")
        self.axs[3].set_ylabel("yaw")
        self.axs[4].set_ylabel("")

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

        # Bottom: stroke shading only
        self._draw_strokes_axis()

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
        # base shading unchanged (we don't edit base), only update stroke bands
        self._update_strokes_axis()

    def _update_mode_text(self):
        if self.stroke_mode:
            color = STROKE_COLORS[self.stroke_mode]
            self.mode_text.set_text(f"Stroke: {self.stroke_mode}")
            self.mode_text.set_color(color)
        else:
            self.mode_text.set_text("Stroke: OFF")
            self.mode_text.set_color("#000000")
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
        - Second right-click sets the end and applies the active stroke string
          ONLY where the base label is Swim(1). Does nothing if no stroke is active.
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

            if self.stroke_mode is None:
                print("[stroke] no stroke selected; nothing changed.")
                return

            # Stroke labeling: only modify existing Swim(1)
            m = (self.labels[i0:i1+1] == 1)
            if m.any():
                self.strokes[i0:i1+1][m] = self.stroke_mode
                self.changed = True
                print(f"[stroke] set '{self.stroke_mode}' over {m.sum()} Swim samples.")
                self._update_labels_plot()
            else:
                print("[stroke] no Swim(1) samples in range; nothing changed.")

    def _on_key(self, event):
        if event.key == "left" and self.T:
            self.cursor_ix = max(0, self.cursor_ix - 1)
            self._update_cursor()
        elif event.key == "right" and self.T:
            self.cursor_ix = min(self.T - 1, self.cursor_ix + 1)
            self._update_cursor()
        elif event.key == "s":
            self._save_labels()
        elif event.key == "p":
            # Toggle between model predictions and blank slate
            self.show_pred = not self.show_pred
            if self.show_pred:
                if self.pred_strokes is None:
                    # If not computed (somehow), try to run now
                    try:
                        self._ensure_model()
                        self._run_stroke_inference_on_current()
                    except Exception as e:
                        print(f"[infer] Unable to run inference on toggle: {e}")
                self._apply_predictions_to_strokes()
                print("[infer] predictions ON (strokes filled from model).")
            else:
                # restore previously saved human strokes if they existed at load-time; otherwise blank
                if getattr(self, "_loaded_human_strokes", None) is not None:
                    self.strokes = self._loaded_human_strokes.copy()
                    print("[infer] predictions OFF (restored previously saved human strokes).")
                else:
                    self._clear_strokes()
                    print("[infer] predictions OFF (blank slate).")
            self._update_labels_plot()
        elif event.key == "r":
            self._clear_strokes()
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
