"""Local (no-GPU) smoke load for one UR10e cup-grasping dataset split.

Loads a single dataset directory through the exact code path the trainer
uses (get_vla_dataset with the ur10e_cup mixture), checks one sample's
shapes and value ranges, and measures dataloader-only throughput.

This does not need a GPU -- it only exercises the data loader, not the
model. See ur10e/results/loader_fixes.md for what the frames/second number
here does and does not tell us: it is NOT a substitute for measuring
throughput on the actual Colab CPU that will feed the GPU during training.

Usage:
    python ur10e/src/smoke_load_local.py ur10e/data/v21_train
    python ur10e/src/smoke_load_local.py ur10e/data/v21_heldout
"""

import sys
import time
import traceback
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

# Running this file directly (python ur10e/src/smoke_load_local.py) puts
# ur10e/src on sys.path, not the repo root -- add the repo root so
# `starVLA` is importable regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    if len(sys.argv) != 2:
        print("usage: python ur10e/src/smoke_load_local.py <dataset_dir>")
        return 2

    dataset_dir = Path(sys.argv[1]).resolve()
    print(f"=== Loading {dataset_dir} ===")

    try:
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset
    except Exception:
        print("FAILED: could not import get_vla_dataset")
        traceback.print_exc()
        return 1

    data_cfg = OmegaConf.create({
        "data_root_dir": str(dataset_dir),
        "data_mix": "ur10e_cup",
        "with_state": True,
    })

    try:
        # delete_pause_frame=False: get_vla_dataset defaults this to True,
        # which makes _get_all_steps_single_process() call
        # _get_position_and_gripper_values() (datasets.py:529-594). That
        # function only recognizes Cartesian action key names
        # (delta_eef_position, or separate x/y/z), not UR10eCupDataConfig's
        # joint-space action.single_arm, and raises ValueError("No suitable
        # position columns found") before this pack ever gets to check a
        # sample. This is a third, previously-masked bug (Bug 1 crashed
        # before the loader ever reached this code) -- see
        # ur10e/results/loader_fixes.md for the full writeup, including
        # that build_dataloader() does not override this default either,
        # so it affects the production path too, not just this script.
        # Working around it here (skip pause-frame filtering) is the only
        # option within this pack's file scope: fixing it for real means
        # teaching _get_position_and_gripper_values about joint-space
        # actions, in datasets.py, which is an upstream file this pack is
        # not allowed to touch.
        dataset = get_vla_dataset(
            data_cfg=data_cfg, action_horizon=7, video_horizon=8, delete_pause_frame=False
        )
    except Exception:
        print(f"FAILED: get_vla_dataset raised for {dataset_dir}")
        traceback.print_exc()
        return 1

    print(f"Dataset length: {len(dataset)}")

    try:
        sample = dataset[0]
    except Exception:
        print(f"FAILED: dataset[0] raised for {dataset_dir}")
        traceback.print_exc()
        return 1

    print(f"Sample keys: {sorted(sample.keys())}")

    video = np.asarray(sample["video"])
    action = np.asarray(sample["action"])
    state = np.asarray(sample["state"]) if "state" in sample else None
    lang = sample["lang"]

    print(f"video shape: {video.shape}")
    print(f"action shape: {action.shape}")
    print(f"state shape: {None if state is None else state.shape}")
    print(f"lang: {lang!r}")

    ok = True

    if not (video.ndim == 5 and video.shape[0] == 2):
        print(f"CHECK FAILED: expected video shape (V=2, T, H, W, 3), got {video.shape}")
        ok = False

    if action.shape[1] != 7:
        print(f"CHECK FAILED: expected action last dim 7, got {action.shape}")
        ok = False

    if state is None or state.shape != (1, 7):
        print(f"CHECK FAILED: expected state shape (1, 7), got {None if state is None else state.shape}")
        ok = False

    action_first6 = action[..., :6]
    action_min, action_max = float(action_first6.min()), float(action_first6.max())
    print(f"action[:, 0:6] min={action_min} max={action_max}")
    if not (-1.0 <= action_min and action_max <= 1.0):
        print("CHECK FAILED: action[:, 0:6] is not within [-1, 1]")
        ok = False

    if state is not None:
        state_first6 = state[..., :6]
        state_min, state_max = float(state_first6.min()), float(state_first6.max())
        print(f"state[:, 0:6] min={state_min} max={state_max}")
        if not (-1.0 <= state_min and state_max <= 1.0):
            print("CHECK FAILED: state[:, 0:6] is not within [-1, 1]")
            ok = False

    gripper_action_values = sorted(set(np.unique(action[..., 6]).tolist()))
    print(f"gripper action value set: {gripper_action_values}")
    if set(gripper_action_values) <= {0.0, 1.0}:
        print("gripper range looks like {0, 1}")
    elif set(gripper_action_values) <= {-1.0, 1.0}:
        print("gripper range looks like {-1, 1}")
    else:
        print("gripper range is neither a clean {0,1} nor {-1,1} subset -- reported as-is above")

    print()
    print("=== Dataloader-only throughput (100 batches, no model) ===")
    print(
        "NOTE: this laptop CPU number does NOT answer REQ-07 (whether the "
        "loader keeps an A100 fed on Colab hardware). It is only an early "
        "sanity signal and a comparison point for the real Colab measurement."
    )
    try:
        from torch.utils.data import DataLoader
        from starVLA.dataloader.lerobot_datasets import collate_fn

        dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, num_workers=0)
        n_batches = 100
        n_frames = 0
        it = iter(dataloader)
        start = time.time()
        for _ in range(n_batches):
            batch = next(it)
            for item in batch:
                v = np.asarray(item["video"])
                n_frames += v.shape[0] * v.shape[1]
        elapsed = time.time() - start
        fps = n_frames / elapsed if elapsed > 0 else 0.0
        print(f"batches={n_batches} frames={n_frames} elapsed_s={elapsed:.2f} frames_per_sec={fps:.2f}")
    except Exception:
        print("FAILED: throughput measurement raised")
        traceback.print_exc()
        ok = False

    print()
    if ok:
        print(f"ALL CHECKS PASSED for {dataset_dir}")
        return 0
    else:
        print(f"SOME CHECKS FAILED for {dataset_dir} -- see CHECK FAILED lines above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
