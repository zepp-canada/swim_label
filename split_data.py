#!/usr/bin/env python3
"""
Split session subfolders among three engineers (alex, subhra, arman).

What it does
------------
- Scans ROOT_DATA_DIR for immediate subdirectories (each a swim session).
- Deterministically shuffles them (seeded) and splits roughly equally into 3 parts.
- Creates three output dirs under OUTPUT_DIR:
      alex_original/, subhra_original/, arman_original/
  Each contains symlinks to the assigned session folders (no copying).
- Also writes:
      alex_original.list, subhra_original.list, arman_original.list  (one folder name per line)
      assignment_summary.csv  (session, engineer)

Engineers-only defaults
-----------------------
All settings are hardcoded below for convenience. Tweak ROOT_DATA_DIR / OUTPUT_DIR / SEED if needed.
"""

from __future__ import annotations
import os
import sys
import csv
import shutil
from pathlib import Path
from typing import List, Dict
import random

# =========================
# Config (edit as needed)
# =========================
ROOT_DATA_DIR = Path("data/all_test_clean")  # root folder containing session subfolders
OUTPUT_DIR    = Path("engineer_splits")      # where alex_original/, subhra_original/, arman_original/ will live
ENGINEERS     = ["alex", "subhra", "arman"]  # order also used to distribute remainders
SEED          = 17                            # deterministic split
USE_SYMLINKS  = True                          # if False, we copy (slower). Symlinks recommended.

# Skip folders that start with a dot or are clearly not sessions. Extend as needed.
SKIP_NAMES_PREFIX = (".", "__pycache__")
# =========================


def list_sessions(root: Path) -> List[Path]:
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"[error] ROOT_DATA_DIR not found or not a directory: {root}")
    sessions = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(SKIP_NAMES_PREFIX)]
    sessions = sorted(sessions, key=lambda p: p.name)
    print(f"[scan] Root: {root}")
    print(f"[scan] Found {len(sessions)} session folders.")
    return sessions


def split_into_three(items: List[Path], names: List[str], seed: int) -> Dict[str, List[Path]]:
    rnd = random.Random(seed)
    shuffled = items[:]
    rnd.shuffle(shuffled)
    n = len(shuffled)
    base = n // 3
    rem  = n % 3
    sizes = [base + (1 if i < rem else 0) for i in range(3)]

    parts: Dict[str, List[Path]] = {}
    start = 0
    for i, name in enumerate(names):
        end = start + sizes[i]
        parts[name] = shuffled[start:end]
        start = end
    # Logging
    for k in names:
        print(f"[split] {k:7s}: {len(parts[k])} sessions")
    return parts


def _make_symlink(src: Path, dst: Path):
    """Create a symlink (or directory copy if USE_SYMLINKS=False)."""
    if dst.exists():
        # If something is already there, skip to be safe.
        return
    if USE_SYMLINKS:
        try:
            os.symlink(src.resolve(), dst, target_is_directory=True)
        except FileExistsError:
            pass
        except OSError as e:
            # On some systems (e.g. Windows without admin), fallback to copying
            print(f"[warn] symlink failed for {dst} → falling back to copy. ({e})")
            shutil.copytree(src, dst)
    else:
        shutil.copytree(src, dst)


def materialize_partitions(parts: Dict[str, List[Path]], out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for eng, folders in parts.items():
        eng_dir = out_root / f"{eng}_original"
        eng_dir.mkdir(parents=True, exist_ok=True)

        # Clean up dangling symlinks from prior runs (optional, safe).
        for existing in eng_dir.iterdir():
            if existing.is_symlink() and not existing.exists():
                existing.unlink(missing_ok=True)

        print(f"[write] Creating links under: {eng_dir}")
        names_list_path = out_root / f"{eng}_original.list"
        with names_list_path.open("w", encoding="utf-8") as lst:
            for src in folders:
                dst = eng_dir / src.name
                _make_symlink(src, dst)
                lst.write(src.name + "\n")
                summary_rows.append((src.name, eng))

        print(f"[write]   {len(folders)} names → {names_list_path.name}")

    # CSV summary
    csv_path = out_root / "assignment_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["session_folder", "engineer"])
        writer.writerows(summary_rows)
    print(f"[write] CSV summary → {csv_path}")


def main():
    try:
        sessions = list_sessions(ROOT_DATA_DIR)
        parts = split_into_three(sessions, ENGINEERS, SEED)
        materialize_partitions(parts, OUTPUT_DIR)

        total = sum(len(v) for v in parts.values())
        print("\n[done] Split complete.")
        for eng in ENGINEERS:
            print(f"       {eng:7s}: {len(parts[eng])} sessions")
        print(f"       total  : {total} sessions")
        print(f"\n[paths] Output dir: {OUTPUT_DIR.resolve()}")
        for eng in ENGINEERS:
            print(f"        - {(OUTPUT_DIR / f'{eng}_original').resolve()}")
    except Exception as e:
        print(f"[fatal] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
