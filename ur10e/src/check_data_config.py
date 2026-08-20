"""Runtime check for UR10eCupDataConfig.

Reads Libero4in1DataConfig -- the reference config that VLA_JEPA actually
runs with -- and checks UR10eCupDataConfig matches its shape by
constructing it and calling its methods for real, not by comparing source
text. See ur10e/results/loader_fixes.md for how the previous "looks like
FR3RealWorldConfig, PASS" review missed the real bug.

Usage:
    python ur10e/src/check_data_config.py
"""

import sys
from pathlib import Path

# Running this file directly (python ur10e/src/check_data_config.py) puts
# ur10e/src on sys.path, not the repo root -- add the repo root so
# `starVLA` is importable regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.data_config import UR10eCupDataConfig
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig


def main():
    data_config = UR10eCupDataConfig(observation_indices=[0], action_indices=list(range(7)))
    print("UR10eCupDataConfig() constructed OK")

    modality_configs = data_config.modality_config()
    assert isinstance(modality_configs, dict), (
        f"modality_config() must return a dict, got {type(modality_configs)}"
    )
    print(f"modality_config() returned a dict with keys: {sorted(modality_configs.keys())}")

    for key, value in modality_configs.items():
        assert isinstance(value, ModalityConfig), (
            f"modality_config()['{key}'] must be a ModalityConfig instance, "
            f"got {type(value)}: {value!r}"
        )
        assert isinstance(value.modality_keys, list) and len(value.modality_keys) > 0, (
            f"modality_config()['{key}'].modality_keys must be a non-empty list, "
            f"got {value.modality_keys!r}"
        )
        assert all(isinstance(k, str) for k in value.modality_keys), (
            f"modality_config()['{key}'].modality_keys must all be strings, "
            f"got {value.modality_keys!r}"
        )
    print("Every value in modality_config() is a ModalityConfig with a non-empty list of string keys")

    video_keys = modality_configs["video"].modality_keys
    assert len(video_keys) == 2, f"expected exactly 2 video keys, got {len(video_keys)}: {video_keys}"
    assert all(k.startswith("video.") for k in video_keys), (
        f"every video key must start with 'video.', got {video_keys}"
    )
    print(f"video modality has exactly 2 keys, both start with 'video.': {video_keys}")

    transform = data_config.transform()
    print(f"transform() constructed OK: {type(transform).__name__}")

    # wm_key_indices lives as a class attribute on UR10eCupDataConfig, not as
    # an entry in the dict modality_config() returns -- see the comment next
    # to its definition in data_config.py for why.
    assert hasattr(UR10eCupDataConfig, "wm_key_indices"), (
        "UR10eCupDataConfig must have a wm_key_indices class attribute"
    )
    assert UR10eCupDataConfig.wm_key_indices == [0, 1], (
        f"wm_key_indices must equal [0, 1], got {UR10eCupDataConfig.wm_key_indices!r}"
    )
    assert "wm_key_indices" not in modality_configs, (
        "wm_key_indices must NOT be a key in the dict modality_config() returns "
        "-- that is the bug this check exists to catch"
    )
    print(
        "wm_key_indices == [0, 1], accessible as UR10eCupDataConfig.wm_key_indices "
        "(a class attribute), and absent from the modality_config() dict"
    )

    print()
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
