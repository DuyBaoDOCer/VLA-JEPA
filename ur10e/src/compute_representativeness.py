"""Compare the train (0-72) and held-out (73-80) episode sets on a few
summary statistics, to check whether 8 held-out episodes are a reasonable
sample of the whole -- informational only. Per TIP-006 3.1, a large
discrepancy is reported, not acted on: the split itself (last 8 episodes,
deterministic) is a closed decision and is not changed based on this
comparison.

Reads only from ur10e/data/v21 (read-only, never modified).

Usage:
    python compute_representativeness.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
V21_DATA = REPO_ROOT / "ur10e" / "data" / "v21" / "data" / "chunk-000"
RESULTS_DIR = REPO_ROOT / "ur10e" / "results"

TRAIN_EPISODES = list(range(0, 73))
HELDOUT_EPISODES = list(range(73, 81))


def load_actions(episodes: list[int]) -> list[np.ndarray]:
    out = []
    for ep in episodes:
        path = V21_DATA / f"episode_{ep:06d}.parquet"
        table = pq.read_table(path, columns=["action"])
        out.append(np.array(table.column("action").to_pylist(), dtype=np.float64))
    return out


def summarize(episodes: list[int]) -> dict:
    per_episode = load_actions(episodes)
    lengths = [a.shape[0] for a in per_episode]
    joints_all = np.concatenate([a[:, :6] for a in per_episode], axis=0)
    gripper_all = np.concatenate([a[:, 6] for a in per_episode], axis=0)
    stationary_mask = ~np.any(joints_all != 0.0, axis=1)

    return {
        "n_episodes": len(episodes),
        "n_frames": int(joints_all.shape[0]),
        "mean_episode_length": float(np.mean(lengths)),
        "stationary_frame_ratio": float(stationary_mask.mean()),
        "rms_abs_d_six_joints": float(np.sqrt(np.mean(joints_all**2))),
        "gripper_open_ratio": float(np.mean(gripper_all)),  # gripper: 1 = open
        "gripper_closed_ratio": float(np.mean(1.0 - gripper_all)),
    }


def main() -> None:
    train = summarize(TRAIN_EPISODES)
    heldout = summarize(HELDOUT_EPISODES)

    output = {"train_73": train, "heldout_8": heldout}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "representativeness.json").write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )

    print("metric                       train_73        heldout_8")
    for key in [
        "n_episodes",
        "n_frames",
        "mean_episode_length",
        "stationary_frame_ratio",
        "rms_abs_d_six_joints",
        "gripper_open_ratio",
        "gripper_closed_ratio",
    ]:
        print(f"{key:<28} {train[key]!s:<15} {heldout[key]!s:<15}")

    print("\nwrote ur10e/results/representativeness.json")


if __name__ == "__main__":
    main()
