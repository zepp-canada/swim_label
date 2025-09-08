#!/usr/bin/env python3
import os
import shutil
import pandas as pd

def find_worked_chunk(src_root: str, label_file: str = "human_label.csv"):
    """Detect the most likely contiguous chunk of sessions the user worked on."""
    subfolders = sorted(
        f for f in os.listdir(src_root) if os.path.isdir(os.path.join(src_root, f))
    )
    labeled = [os.path.exists(os.path.join(src_root, f, label_file)) for f in subfolders]

    best_range, best_score = (0, 0), -1
    for i in range(len(subfolders)):
        for j in range(i + 1, len(subfolders) + 1):
            block = labeled[i:j]
            if not block:
                continue
            density = sum(block) / len(block)
            score = sum(block) * density
            if score > best_score:
                best_score = score
                best_range = (i, j)

    start, end = best_range
    print(f"[INFO] Best chunk: {start} → {end-1} ({end-start} folders)")
    print(f"[INFO] Example folder names: {subfolders[start]} → {subfolders[end-1]}")
    return subfolders[start:end]


def copy_labeled_sessions(
    src_root: str,
    dst_root: str,
    label_file: str = "human_label.csv",
    orig_label_file: str = "new_swim_label.csv",
):
    """
    Copies the worked-on chunk of folders from src_root to dst_root.
    If a folder in the chunk has no human_label.csv, copy new_swim_label.csv as human_label.csv.
    """
    if not os.path.exists(src_root):
        print(f"[ERROR] Source folder does not exist: {src_root}")
        return

    os.makedirs(dst_root, exist_ok=True)
    worked_chunk = find_worked_chunk(src_root, label_file)

    print(f"\n[INFO] Copying {len(worked_chunk)} folders from {src_root} → {dst_root}")
    copied_count = 0

    for idx, folder in enumerate(worked_chunk, 1):
        folder_path = os.path.join(src_root, folder)
        dst_path = os.path.join(dst_root, folder)
        dst_label_path = os.path.join(dst_path, label_file)
        orig_label_path = os.path.join(folder_path, orig_label_file)

        if not os.path.exists(dst_path):
            shutil.copytree(folder_path, dst_path)
            copied_count += 1
            print(f"[{idx}/{len(worked_chunk)}] [COPIED] {folder}")
        else:
            print(f"[{idx}/{len(worked_chunk)}] [SKIPPED] {folder} already exists")

        # Ensure human_label.csv exists
        if not os.path.exists(dst_label_path) and os.path.exists(orig_label_path):
            shutil.copy(orig_label_path, dst_label_path)
            print(f"    [ADDED DEFAULT] {orig_label_file} → {label_file}")

    print(f"\n[INFO] Done. {copied_count} new folder(s) copied to {dst_root}.")


if __name__ == "__main__":
    SRC_DIR = "engineer_splits/subhra_original"
    DST_DIR = "engineer_splits/subhra_original_copy"

    copy_labeled_sessions(SRC_DIR, DST_DIR)
