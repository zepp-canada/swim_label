"""
model_infer.py — load the strokeDetector checkpoint and run predictions on arrays.

- Torch 2.6-safe loading (handles weights_only default).
- Detects class count from checkpoint (final linear layer shape).
- Array-level prediction that uses your project's `utils.run_model`.
"""

from __future__ import annotations

import os
from typing import Tuple, Optional

import numpy as np
from scipy.signal import medfilt

# torch
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

# project util for forward pass
from utils import run_model

DEFAULT_MODEL_PATH = os.path.join("models", "3class_new", "swim.pth")


# ---------- network ----------

if torch is not None:
    class strokeDetector(nn.Module):
        """Model expected by the swim.pth checkpoint."""
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


# ---------- checkpoint I/O ----------

def _safe_torch_load(path: str):
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
    """Infer class count from the final linear layer of the checkpoint."""
    if isinstance(state, dict) and "state_dict" in state:
        sd = state["state_dict"]
    else:
        sd = state
    for k in ["acc_linear.weight", "module.acc_linear.weight"]:
        if k in sd:
            return int(sd[k].shape[0])
    for k in ["acc_linear.bias", "module.acc_linear.bias"]:
        if k in sd:
            return int(sd[k].shape[0])
    raise RuntimeError("Could not infer number of classes from checkpoint.")


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load checkpoint, instantiate model with correct class count, return (model, num_classes)."""
    if torch is None:
        print("[inference] Torch not available; skipping model load.")
        return None, 0
    if not os.path.exists(model_path):
        print(f"[inference] Checkpoint not found: {model_path}")
        return None, 0

    print(f"[inference] Loading model: {model_path}")
    state = _safe_torch_load(model_path)
    num_classes = _detect_num_classes(state)
    print(f"[inference] Detected {num_classes} classes in checkpoint.")
    model = strokeDetector(num_classes).to("cpu")
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[inference] load_state_dict: missing={missing}, unexpected={unexpected}")
    model.eval()
    return model, num_classes


# ---------- prediction helpers ----------

def smooth_label(arr: np.ndarray, window: int = 75) -> np.ndarray:
    """Median-filtered labels; ensures odd kernel ≥ 3 like your reference."""
    k = int(window)
    k = k + (k % 2) + 1
    return medfilt(arr.astype(float), kernel_size=k).astype(int)


def predict_arrays_argmax(
    acc: np.ndarray,
    gyro: np.ndarray,
    mag: np.ndarray,
    model,
) -> np.ndarray:
    """
    Run forward pass on already split arrays (RAW COUNTS conventions).
    Returns per-sample argmax labels (no smoothing here).
    """
    if model is None:
        return np.zeros(acc.shape[0], dtype=int)

    if acc.shape[0] == 0:
        return np.zeros(0, dtype=int)

    with torch.no_grad():
        logits, _ = run_model(
            acc[:, 0], acc[:, 1], acc[:, 2],
            gyro[:, 0], gyro[:, 1], gyro[:, 2],
            mag[:, 0],  mag[:, 1],  mag[:, 2],
            model
        )
    return np.argmax(logits, axis=1).astype(int)
