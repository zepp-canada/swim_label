#!/usr/bin/env python3
"""
Copy all session folders containing 'mixed' (case-insensitive) in their name
from SOURCE_ROOT into TARGET_ROOT.
"""

import os
import shutil

# -------- Hard-coded paths --------
SOURCE_ROOT = "data/all_train_clean"
TARGET_ROOT = "data/mixed_only"

def copy_mixed_sessions(src_root: str, dst_root: str):
    if not os.path.isdir(src_root):
        raise NotADirectoryError(f"Source root not found: {src_root}")
    os.makedirs(dst_root, exist_ok=True)

    sessions = [d for d in os.listdir(src_root)
                if os.path.isdir(os.path.join(src_root, d))
                and "mixed" in d.lower()]

    if not sessions:
        print(f"[info] No mixed sessions found under {src_root}")
        return

    for sess in sessions:
        src_path = os.path.join(src_root, sess)
        dst_path = os.path.join(dst_root, sess)
        if os.path.exists(dst_path):
            print(f"[skip] Already exists: {dst_path}")
            continue
        print(f"[copy] {src_path} -> {dst_path}")
        shutil.copytree(src_path, dst_path)

    print(f"[done] Copied {len(sessions)} mixed sessions.")

if __name__ == "__main__":
    copy_mixed_sessions(SOURCE_ROOT, TARGET_ROOT)
