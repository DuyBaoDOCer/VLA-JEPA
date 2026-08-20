"""Prove D23 through the production dataloader entry point, on the laptop.

Confirms two things about build_dataloader(cfg, dataset_py="lerobot_datasets"):
  1. delete_pause_frame=False (our config's value) loads both UR10e splits
     end to end and reports Total steps equal to the full frame count.
  2. delete_pause_frame=True (the upstream default build_dataloader used to
     apply unconditionally) still raises ValueError("No suitable position
     columns found") -- proving the diagnosis in TIP-009c section 1 (D23),
     not just working around it.

This does not need a GPU. See ur10e/results/ for the captured output.

Usage:
    python ur10e/src/prove_pause_frame_flag.py
"""

import io
import os
import re
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

# The Windows PyTorch wheel's TCPStore was built without libuv support;
# init_process_group's default rendezvous tries libuv first and raises
# RuntimeError before rank/world_size are even negotiated unless this is
# unset ahead of time.
os.environ.setdefault("USE_LIBUV", "0")

import torch.distributed as dist
from omegaconf import OmegaConf

# Running this file directly (python ur10e/src/prove_pause_frame_flag.py) puts
# ur10e/src on sys.path, not the repo root -- add the repo root so `starVLA`
# is importable regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / "ur10e" / "configs" / "ur10e_ft.yaml"
SPLIT_DIRS = {
    "v21_train": REPO_ROOT / "ur10e" / "data" / "v21_train",
    "v21_heldout": REPO_ROOT / "ur10e" / "data" / "v21_heldout",
}

# datasets.py's steps cache (LeRobotSingleDataset._get_all_steps,
# datasets.py:394-453) computes a delete_pause_frame-aware config_key but
# then discards it: an "@BUG" comment overrides steps_filename with two
# hardcoded names (steps_332420bad1ab.pkl, steps_2d5a34b904d2.pkl) that carry
# no record of which delete_pause_frame value produced them. A cache file
# left by an earlier run -- e.g. TIP-009b's smoke_load_local.py, which always
# ran with delete_pause_frame=False -- would make a delete_pause_frame=True
# call here silently return the False-run's cached steps instead of ever
# reaching _get_all_steps_single_process(), turning a real bug into a false
# PASS for AC5. Clearing both fixed filenames before every case makes each
# call in this script a genuine, uncached computation.
STALE_CACHE_NAMES = ["steps_332420bad1ab.pkl", "steps_2d5a34b904d2.pkl"]

TOTAL_STEPS_RE = re.compile(r"Total steps: (\d+) from (\d+) trajectories")


def clear_steps_cache(dataset_dir: Path) -> None:
    for name in STALE_CACHE_NAMES:
        cache_path = dataset_dir / "meta" / name
        if cache_path.exists():
            cache_path.unlink()
            print(f"  removed stale steps cache: {cache_path}")


def run_case(dataset_dir: Path, delete_pause_frame: bool, scratch_root: Path) -> dict:
    from starVLA.dataloader import build_dataloader

    clear_steps_cache(dataset_dir)

    cfg = OmegaConf.load(CONFIG_PATH)
    cfg.datasets.vla_data.data_root_dir = str(dataset_dir)
    cfg.datasets.vla_data.delete_pause_frame = delete_pause_frame
    cfg.output_dir = str(scratch_root / f"{dataset_dir.name}_{delete_pause_frame}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    label = f"{dataset_dir.name} delete_pause_frame={delete_pause_frame}"
    print(f"--- {label} ---")

    buf = io.StringIO()
    tee = _Tee(sys.stdout, buf)
    try:
        with redirect_stdout(tee):
            dataloader = build_dataloader(cfg, dataset_py="lerobot_datasets")
    except ValueError as e:
        result = {"expected_raise": delete_pause_frame, "raised": True, "error": str(e)}
        if delete_pause_frame:
            print(f"PASS: ValueError raised as expected: {e}")
        else:
            print(f"FAIL: unexpected ValueError: {e}")
        return result
    except Exception:
        traceback.print_exc()
        return {"expected_raise": delete_pause_frame, "raised": True, "error": "unexpected exception type"}

    match = TOTAL_STEPS_RE.findall(buf.getvalue())
    total_steps, trajectories = (int(match[-1][0]), int(match[-1][1])) if match else (None, None)
    result = {
        "expected_raise": delete_pause_frame,
        "raised": False,
        "total_steps": total_steps,
        "trajectories": trajectories,
        "dataloader_len_steps": len(dataloader.dataset),
    }
    if delete_pause_frame:
        print("FAIL: expected ValueError but none was raised")
    else:
        print(f"PASS: no exception. Total steps={total_steps} trajectories={trajectories}")
    return result


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for stream in self._streams:
            stream.write(s)
        return len(s)

    def flush(self):
        for stream in self._streams:
            stream.flush()


def main() -> int:
    import tempfile

    dist.init_process_group(
        backend="gloo", init_method="tcp://127.0.0.1:29511", rank=0, world_size=1
    )

    scratch_root = Path(tempfile.mkdtemp(prefix="prove_pause_frame_"))
    print(f"scratch output_dir root: {scratch_root}")

    results = {}
    overall_ok = True
    for split_name, split_dir in SPLIT_DIRS.items():
        for delete_pause_frame in (False, True):
            key = f"{split_name}_delete_pause_frame_{delete_pause_frame}"
            res = run_case(split_dir, delete_pause_frame, scratch_root)
            results[key] = res
            case_ok = res["raised"] == res["expected_raise"]
            overall_ok = overall_ok and case_ok

    print()
    print("=== SUMMARY ===")
    for key, res in results.items():
        print(f"{key}: {res}")

    print()
    if overall_ok:
        print("ALL CASES MATCHED EXPECTATION (AC4/AC5 PASS)")
    else:
        print("SOME CASES DID NOT MATCH EXPECTATION -- see FAIL lines above")

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
