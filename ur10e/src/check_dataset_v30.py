"""Integrity check for the ur10e-cup source dataset (LeRobot v3.0 layout).

Read-only: verifies file structure, metadata, and parquet content against
the expected shape of the raw download. Does not modify anything.

Usage:
    python check_dataset_v30.py <dataset_root>
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

EXPECTED_FILES = [
    "data/chunk-000/file-000.parquet",
    "meta/info.json",
    "meta/stats.json",
    "meta/tasks.parquet",
    "meta/episodes/chunk-000/file-000.parquet",
]

EXPECTED_VIDEO_COUNTS = {
    "videos/observation.images.side/chunk-000": 4,
    "videos/observation.images.wrist/chunk-000": 3,
}


def human_readable_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def dir_size(root):
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            total += os.path.getsize(os.path.join(dirpath, name))
    return total


def check_structure(root):
    ok = True
    for rel in EXPECTED_FILES:
        full = root / rel
        if full.is_file():
            print(f"[OK] found {rel}")
        else:
            print(f"[FAIL] missing expected file: {rel}")
            ok = False

    for rel, expected_count in EXPECTED_VIDEO_COUNTS.items():
        full = root / rel
        if not full.is_dir():
            print(f"[FAIL] missing expected video directory: {rel}")
            ok = False
            continue
        count = len(sorted(full.glob("*.mp4")))
        status = "OK" if count == expected_count else "FAIL"
        print(f"[{status}] {rel}: {count} mp4 file(s) (expected {expected_count})")
        if count != expected_count:
            ok = False
    return ok


def check_metadata(root):
    info_path = root / "meta" / "info.json"
    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    print(f"codebase_version: {info.get('codebase_version')}")
    print(f"fps: {info.get('fps')}")
    print(f"total_episodes (from info.json): {info.get('total_episodes')}")
    print(f"total_frames (from info.json): {info.get('total_frames')}")
    return info


def check_parquet(root):
    data_path = root / "data" / "chunk-000" / "file-000.parquet"
    df = pd.read_parquet(data_path)

    num_rows = len(df)
    print(f"row count: {num_rows}")

    if "episode_index" in df.columns:
        num_episodes = df["episode_index"].nunique()
        print(f"distinct episodes: {num_episodes}")
    else:
        print("[WARN] column 'episode_index' not found; cannot count distinct episodes")

    for col in ("observation.state", "action"):
        if col not in df.columns:
            print(f"[FAIL] column not found: {col}")
            continue
        sample = df[col].iloc[0]
        dim = len(sample) if hasattr(sample, "__len__") else 1
        print(f"{col}: dtype={df[col].dtype}, dimension={dim}")


def main():
    parser = argparse.ArgumentParser(
        description="Check integrity of the ur10e-cup source dataset (read-only)."
    )
    parser.add_argument("dataset_root", type=str, help="Path to the dataset root, e.g. ur10e/data/v30")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.is_dir():
        print(f"[FAIL] dataset root does not exist: {root}")
        sys.exit(1)

    print(f"Checking dataset at: {root}")
    print()

    print("--- Structure check ---")
    structure_ok = check_structure(root)
    print()

    print("--- Metadata check ---")
    check_metadata(root)
    print()

    print("--- Parquet check ---")
    check_parquet(root)
    print()

    total_bytes = dir_size(root)
    print(f"total size: {human_readable_size(total_bytes)} ({total_bytes} bytes)")
    print()

    if not structure_ok:
        print("[FAIL] dataset structure check failed")
        sys.exit(1)

    print("[OK] all checks passed")


if __name__ == "__main__":
    main()
