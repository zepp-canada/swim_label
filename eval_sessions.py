#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate model per session and collect the worst runs.

Hard-coded engineer settings:
  - DATA_ROOT: folder containing session subfolders
  - MODEL_PATH: path to swim.pth
  - OUTDIR: where to store worst sessions + CSV report
  - N_WORST: how many worst sessions (by overall accuracy) to copy
  - SMOOTH: median smoothing window in samples (set 0/1 to disable)

Behavior:
  - For each session with labels:
      * run inference (raw counts, gyro zeroed, utils.run_model)
      * compute per-class accuracy, precision, recall + overall accuracy
  - Print per-session logs and a dataset summary
  - Save CSV report and copy the N worst sessions
"""

import os
import sys
import csv
import shutil
from glob import glob
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import medfilt

# =========================
# HARD-CODED CONFIG
# =========================
DATA_ROOT = "data/all_test_clean"
MODEL_PATH = os.path.join("models", "3class_new", "swim.pth")
OUTDIR     = "worst_sessions"
N_WORST    = 10
SMOOTH     = 75  # samples; set 0/1 to disable

# Canonical label names (we’ll truncate to the checkpoint's class count)
CANONICAL_CLASS_NAMES = ["Rest", "Swim", "Turn", "Dolphin"]

LABEL_CANDIDATES = ["human_label.csv", "new_swim_label.csv", "swim_label.csv", "label.csv"]

# torch (only used for inference)
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

# Your inference helper (same one used by the GUI)
from utils import run_model  # (ax, ay, az, gx, gy, gz, mx, my, mz, model) -> (logits, _)


# =========================
# Model + checkpoint
# =========================
if torch is not None:
    class strokeDetector(nn.Module):
        """Matches the architecture expected by swim.pth."""
        def __init__(self, number_of_labels: int):
            super().__init__()
            self.n_channel = 6
            self.n_filters = 128
            self.n_hidden  = 128
            self.n_rnn_layers = 1
            self.acc_cnn  = nn.Conv1d(self.n_channel, self.n_filters, 25, stride=1, padding='same')
            self.acc_rnn  = nn.GRU(input_size=self.n_filters, hidden_size=self.n_hidden,
                                   num_layers=self.n_rnn_layers, batch_first=True, bidirectional=True)
            self.acc_linear = nn.Linear(self.n_hidden * 2, number_of_labels, bias=True)

        def forward(self, x, h_prev=None):  # x: (B, T, F=6)
            x = self.acc_cnn(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B,T,filters)
            rnn_out, h = self.acc_rnn(x, h_prev)                   # (B,T,2*hidden)
            logits = self.acc_linear(rnn_out)                      # (B,T,C)
            return logits, h

def safe_torch_load(path: str):
    """torch.load that works across PyTorch 2.6 (weights_only) and older."""
    if torch is None:
        raise RuntimeError("PyTorch is not available in this environment.")
    import numpy as _np
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))
    except Exception:
        from torch.serialization import add_safe_globals
        add_safe_globals([_np.core.multiarray.scalar])
        return torch.load(path, map_location=torch.device("cpu"), weights_only=True)

def _detect_num_classes(state) -> int:
    """Infer class count from checkpoint."""
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state
    for k in ["acc_linear.weight", "module.acc_linear.weight"]:
        if k in sd:
            w = sd[k]
            try:
                return int(w.shape[0])
            except Exception:
                pass
    for k in ["acc_linear.bias", "module.acc_linear.bias"]:
        if k in sd:
            b = sd[k]
            try:
                return int(b.shape[0])
            except Exception:
                pass
    raise RuntimeError("Could not infer number of classes from checkpoint.")

def load_model(model_path: str):
    print(f"[eval] Loading model checkpoint: {model_path}")
    state = safe_torch_load(model_path)
    num_classes = _detect_num_classes(state)
    print(f"[eval] Detected {num_classes} classes in checkpoint.")
    model = strokeDetector(num_classes).to("cpu")
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[eval] load_state_dict: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, num_classes


# =========================
# Data loading (raw counts)
# =========================
def load_session_raw(session_dir: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      X_raw (T, 9) stacked [acc, gyro, mag] in RAW COUNTS (scale=False).
      y (T,) int labels if found, else None.
    """
    colnames = ["x", "y", "z", "t", "p"]
    acc_p = os.path.join(session_dir, "acc.csv")
    mag_p = os.path.join(session_dir, "mag.csv")
    gyro_p = os.path.join(session_dir, "gyro.csv")

    if not (os.path.exists(acc_p) and os.path.exists(mag_p)):
        raise FileNotFoundError(f"Missing acc.csv or mag.csv in {session_dir}")

    acc_df = pd.read_csv(acc_p, skiprows=1, names=colnames)
    mag_df = pd.read_csv(mag_p, skiprows=1, names=colnames)
    if os.path.exists(gyro_p):
        gyro_df = pd.read_csv(gyro_p, skiprows=1, names=colnames)
        gyro = gyro_df[["x","y","z"]].values
    else:
        gyro = np.zeros((len(acc_df), 3), dtype=float)

    acc = acc_df[["x","y","z"]].values
    mag = mag_df[["x","y","z"]].values

    n = min(len(acc), len(gyro), len(mag))
    acc, gyro, mag = acc[:n], gyro[:n], mag[:n]

    # labels
    y = None
    for cand in LABEL_CANDIDATES:
        lp = os.path.join(session_dir, cand)
        if os.path.exists(lp):
            try:
                if cand == "human_label.csv":
                    lbl = pd.read_csv(lp, header=None).values.squeeze()
                else:
                    lbl = pd.read_csv(lp, skiprows=1, names=["target"])["target"].values
                y = lbl.astype(int)
                y[y == -1] = 2  # map -1 → Turn if present
                break
            except Exception:
                pass
    if y is not None:
        y = y[:n]

    X = np.concatenate([acc, gyro, mag], axis=1).astype(np.float32)  # (T, 9)
    return X, (y.astype(int) if y is not None else None)


# =========================
# Inference (raw counts)
# =========================
def med_smooth_int(arr: np.ndarray, k: int = 75) -> np.ndarray:
    k = int(k)
    k = k + (k % 2) + 1  # ensure odd and >= 3
    return medfilt(arr.astype(float), kernel_size=k).astype(int)

def predict_session(model, X_raw: np.ndarray, smooth_k: Optional[int] = 75) -> np.ndarray:
    """
    X_raw: (T, 9) in raw counts. We zero gyro like in your training/inference code.
    Returns y_pred (T,) ints.
    """
    X = X_raw.copy()
    if X.shape[1] < 9:
        raise ValueError("Expected 9 channels (acc+gyro+mag).")
    # zero gyro
    X[:, 3:6] = 0.0

    acc = X[:, 0:3]
    gyro = X[:, 3:6]
    mag  = X[:, 6:9]

    with torch.no_grad():
        logits, _ = run_model(acc[:, 0], acc[:, 1], acc[:, 2],
                              gyro[:, 0], gyro[:, 1], gyro[:, 2],
                              mag[:, 0],  mag[:, 1],  mag[:, 2],
                              model)
    y_pred = np.argmax(logits, axis=1).astype(int)
    if smooth_k and smooth_k > 1:
        y_pred = med_smooth_int(y_pred, k=smooth_k)
    return y_pred


# =========================
# Metrics
# =========================
@dataclass
class Metrics:
    session: str
    n_samples: int
    overall_accuracy: float
    per_class_accuracy: List[float]
    per_class_precision: List[float]
    per_class_recall: List[float]

def confusion_from_labels(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm

def metrics_from_confusion(cm: np.ndarray) -> Tuple[float, List[float], List[float], List[float]]:
    """
    Returns: overall_acc, per_class_acc, per_class_prec, per_class_rec
    per_class_acc is one-vs-rest accuracy: (TP+TN)/N for each class.
    """
    K = cm.shape[0]
    N = cm.sum()
    overall_acc = float(np.trace(cm)) / N if N > 0 else 0.0

    per_acc = []
    per_prec = []
    per_rec = []
    for k in range(K):
        TP = cm[k, k]
        FP = cm[:, k].sum() - TP
        FN = cm[k, :].sum() - TP
        TN = N - TP - FP - FN

        acc_k = float(TP + TN) / N if N > 0 else 0.0
        prec_k = float(TP) / max(1, TP + FP)
        rec_k = float(TP) / max(1, TP + FN)

        per_acc.append(acc_k)
        per_prec.append(prec_k)
        per_rec.append(rec_k)

    return overall_acc, per_acc, per_prec, per_rec


# =========================
# Main evaluation loop
# =========================
def evaluate_sessions(
    data_root: str,
    model_path: str,
    n_worst: int,
    outdir: str,
    smooth_k: int,
) -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required to run evaluation.")

    sessions = sorted([p for p in glob(os.path.join(data_root, "*")) if os.path.isdir(p)])
    if not sessions:
        print(f"[eval] No session folders found under {data_root}")
        return

    model, num_classes = load_model(model_path)
    class_names = CANONICAL_CLASS_NAMES[:num_classes]
    print(f"[eval] Using class set: {class_names}\n")

    # Results
    per_session: List[Metrics] = []
    dataset_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    total_dropped = 0

    print(f"[eval] Scanning {len(sessions)} sessions under {data_root} ...")

    for idx, sess in enumerate(sessions, start=1):
        base = os.path.basename(sess)
        print(f"\n[{idx}/{len(sessions)}] Session: {base}")
        try:
            X_raw, y_true = load_session_raw(sess)
        except Exception as e:
            print(f"  -> skip: failed to load ({e})")
            continue

        if y_true is None:
            print("  -> skip: no labels present")
            continue

        print(f"  Loaded {len(X_raw)} samples; running prediction (smooth={smooth_k}) ...")
        y_pred = predict_session(model, X_raw, smooth_k=smooth_k)

        # clip to same length
        n = min(len(y_true), len(y_pred))
        y_true = y_true[:n]
        y_pred = y_pred[:n]

        # drop GT labels not supported by the model (e.g., Dolphin when using a 3-class checkpoint)
        mask = (y_true >= 0) & (y_true < num_classes)
        dropped = int((~mask).sum())
        if dropped > 0:
            total_dropped += dropped
            print(f"  NOTE: Dropping {dropped} samples with labels outside 0..{num_classes-1}.")
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            n = len(y_true)

        if n == 0:
            print("  -> skip: nothing left to evaluate after dropping unsupported labels.")
            continue

        print(f"  Evaluating on {n} labeled samples ...")
        cm = confusion_from_labels(y_true, y_pred, num_classes)
        dataset_cm += cm

        overall_acc, per_acc, per_prec, per_rec = metrics_from_confusion(cm)
        per_session.append(Metrics(
            session=base,
            n_samples=int(n),
            overall_accuracy=overall_acc,
            per_class_accuracy=per_acc,
            per_class_precision=per_prec,
            per_class_recall=per_rec,
        ))

        pcs = ", ".join(
            f"{class_names[i]} acc={per_acc[i]:.3f} prec={per_prec[i]:.3f} rec={per_rec[i]:.3f}"
            for i in range(num_classes)
        )
        print(f"  -> overall={overall_acc:.3f}  N={n:6d}  |  {pcs}")

    # Dataset-level summary
    if not per_session:
        print("\n[eval] No labeled sessions found; nothing to report.")
        return

    ds_overall, ds_pacc, ds_pprec, ds_prec = metrics_from_confusion(dataset_cm)
    print("\n========== DATASET SUMMARY ==========")
    print(f"Overall accuracy: {ds_overall:.4f}")
    for k in range(num_classes):
        print(f"  {class_names[k]:8s}  acc={ds_pacc[k]:.4f}  prec={ds_pprec[k]:.4f}  rec={ds_prec[k]:.4f}")
    if total_dropped > 0:
        print(f"\nNOTE: Dropped {total_dropped} samples across sessions due to unsupported label IDs.")

    # Rank sessions by overall accuracy (ascending = worst)
    ranked = sorted(per_session, key=lambda m: m.overall_accuracy)
    n_pick = max(0, min(n_worst, len(ranked)))
    if n_pick == 0:
        print("\n[eval] No sessions to copy as worst; exiting.")
        return

    os.makedirs(outdir, exist_ok=True)

    # Write CSV report
    csv_path = os.path.join(outdir, "evaluation_report.csv")
    print(f"\n[eval] Writing per-session report: {csv_path}")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = (
            ["session", "n_samples", "overall_accuracy"] +
            [f"acc_{n.lower()}" for n in class_names] +
            [f"prec_{n.lower()}" for n in class_names] +
            [f"rec_{n.lower()}" for n in class_names]
        )
        w.writerow(header)
        for m in ranked:
            row = [m.session, m.n_samples, m.overall_accuracy]
            row += [float(x) for x in m.per_class_accuracy]
            row += [float(x) for x in m.per_class_precision]
            row += [float(x) for x in m.per_class_recall]
            w.writerow(row)

    # Copy worst sessions
    print(f"[eval] Copying worst {n_pick} sessions to: {outdir}")
    for i, m in enumerate(ranked[:n_pick], start=1):
        src = os.path.join(data_root, m.session)
        dst = os.path.join(outdir, f"{i:02d}_{m.session}")
        print(f"  -> {m.session}  (overall={m.overall_accuracy:.3f})  ->  {dst}")
        os.makedirs(dst, exist_ok=True)
        for name in ["acc.csv", "gyro.csv", "mag.csv", *LABEL_CANDIDATES]:
            sp = os.path.join(src, name)
            if os.path.exists(sp):
                shutil.copy2(sp, os.path.join(dst, os.path.basename(sp)))
        # also save a tiny metrics txt
        with open(os.path.join(dst, "metrics.txt"), "w") as f:
            f.write(f"session={m.session}\n")
            f.write(f"n_samples={m.n_samples}\n")
            f.write(f"overall_accuracy={m.overall_accuracy:.6f}\n")
            for k in range(num_classes):
                f.write(
                    f"{class_names[k]}: acc={m.per_class_accuracy[k]:.6f}, "
                    f"prec={m.per_class_precision[k]:.6f}, "
                    f"rec={m.per_class_recall[k]:.6f}\n"
                )

    print(f"\n[eval] Done. Report: {csv_path}")


# =========================
# Run with hard-coded args
# =========================
if __name__ == "__main__":
    evaluate_sessions(
        data_root=DATA_ROOT,
        model_path=MODEL_PATH,
        n_worst=N_WORST,
        outdir=OUTDIR,
        smooth_k=SMOOTH,
    )
