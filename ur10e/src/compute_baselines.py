"""Copy baselines (predict-zero and predict-previous-delta) for the eval
pack to beat, computed on the exact sets it will be evaluated on.

Reads only from ur10e/data/v21 (read-only, never modified). Episode-to-set
membership is the same 73/8 split as ur10e/results/split.json.

Two baselines, applied identically to the 6 joint-delta dims and to the
gripper dim (kept separate because gripper is a binary channel, not a
radian-scale delta):
  A -- predict no motion: a_hat[t] = 0, MSE = mean(d[t]**2)
  B -- repeat previous delta: a_hat[t] = d[t-1], per-episode, dropping each
       episode's first frame (nothing to repeat there)

"moving" frames are rows where the 6-dim joint delta target is nonzero in
at least one dim; this uses the actual (target) delta being predicted, so
baseline A's "moving" set is per-frame d[t]!=0 and baseline B's is
per-frame d[t]!=0 evaluated on the post-shift target (t=1..end).

Usage:
    python compute_baselines.py
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
ALL_EPISODES = list(range(0, 81))

SETS = {
    "all_81": ALL_EPISODES,
    "train_73": TRAIN_EPISODES,
    "heldout_8": HELDOUT_EPISODES,
}

# Reference values from the full-81, all-frames, joints cell (TIP-006 3.2),
# computed on the same v21 action column before this pack existed.
REFERENCE_A_ALL81_ALLFRAMES_JOINTS = 4.658e-06
REFERENCE_B_ALL81_ALLFRAMES_JOINTS = 5.842e-07


def load_episode_actions() -> dict[int, np.ndarray]:
    actions = {}
    for path in sorted(V21_DATA.glob("episode_*.parquet")):
        ep = int(path.stem.split("_")[1])
        table = pq.read_table(path, columns=["action"])
        actions[ep] = np.array(table.column("action").to_pylist(), dtype=np.float64)
    return actions


def baseline_set_stats(actions: dict[int, np.ndarray], episodes: list[int]) -> dict:
    joints_all = np.concatenate([actions[ep][:, :6] for ep in episodes], axis=0)
    gripper_all = np.concatenate([actions[ep][:, 6] for ep in episodes], axis=0)
    moving_mask_all = np.any(joints_all != 0.0, axis=1)

    result: dict = {
        "A": {
            "all": {
                "joints_mse": float(np.mean(joints_all**2)),
                "gripper_mse": float(np.mean(gripper_all**2)),
                "n_frames": int(joints_all.shape[0]),
            },
            "moving": {
                "joints_mse": float(np.mean(joints_all[moving_mask_all] ** 2)),
                "gripper_mse": float(np.mean(gripper_all[moving_mask_all] ** 2)),
                "n_frames": int(moving_mask_all.sum()),
            },
        }
    }

    joints_pred, joints_target = [], []
    gripper_pred, gripper_target = [], []
    for ep in episodes:
        joints = actions[ep][:, :6]
        gripper = actions[ep][:, 6]
        joints_pred.append(joints[:-1])
        joints_target.append(joints[1:])
        gripper_pred.append(gripper[:-1])
        gripper_target.append(gripper[1:])

    joints_pred = np.concatenate(joints_pred, axis=0)
    joints_target = np.concatenate(joints_target, axis=0)
    gripper_pred = np.concatenate(gripper_pred, axis=0)
    gripper_target = np.concatenate(gripper_target, axis=0)
    joints_resid = joints_target - joints_pred
    gripper_resid = gripper_target - gripper_pred
    moving_mask_b = np.any(joints_target != 0.0, axis=1)

    result["B"] = {
        "frames_skipped_at_episode_boundaries": len(episodes),
        "all": {
            "joints_mse": float(np.mean(joints_resid**2)),
            "gripper_mse": float(np.mean(gripper_resid**2)),
            "n_frames": int(joints_resid.shape[0]),
        },
        "moving": {
            "joints_mse": float(np.mean(joints_resid[moving_mask_b] ** 2)),
            "gripper_mse": float(np.mean(gripper_resid[moving_mask_b] ** 2)),
            "n_frames": int(moving_mask_b.sum()),
        },
    }
    return result


def main() -> None:
    actions = load_episode_actions()
    assert set(actions.keys()) == set(ALL_EPISODES), "expected episodes 0..80 in v21"

    baselines = {name: baseline_set_stats(actions, eps) for name, eps in SETS.items()}

    a_check = baselines["all_81"]["A"]["all"]["joints_mse"]
    b_check = baselines["all_81"]["B"]["all"]["joints_mse"]
    assert abs(a_check - REFERENCE_A_ALL81_ALLFRAMES_JOINTS) / REFERENCE_A_ALL81_ALLFRAMES_JOINTS < 0.01, (
        f"baseline A all-81/all-frames/joints = {a_check}, expected ~{REFERENCE_A_ALL81_ALLFRAMES_JOINTS}"
    )
    assert abs(b_check - REFERENCE_B_ALL81_ALLFRAMES_JOINTS) / REFERENCE_B_ALL81_ALLFRAMES_JOINTS < 0.01, (
        f"baseline B all-81/all-frames/joints = {b_check}, expected ~{REFERENCE_B_ALL81_ALLFRAMES_JOINTS}"
    )
    print(f"reference check OK: A={a_check:.4e} (~{REFERENCE_A_ALL81_ALLFRAMES_JOINTS:.3e}), "
          f"B={b_check:.4e} (~{REFERENCE_B_ALL81_ALLFRAMES_JOINTS:.3e})")

    number_to_beat = baselines["heldout_8"]["B"]["moving"]["joints_mse"]
    output = {
        "baselines": baselines,
        "number_to_beat": {
            "description": "baseline B (repeat previous delta), heldout_8 set, moving frames, 6 joints",
            "value": number_to_beat,
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "baselines.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print("\nwrote ur10e/results/baselines.json")
    print(f"\nNUMBER TO BEAT (heldout_8, moving, joints, baseline B): {number_to_beat:.6e}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
