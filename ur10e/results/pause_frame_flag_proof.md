# Pause-frame flag proof (TIP-009c, D23)

Proves, on the laptop, that the production entry point
`build_dataloader(cfg, dataset_py="lerobot_datasets")` (patched in
`starVLA/dataloader/__init__.py`) actually honors
`datasets.vla_data.delete_pause_frame` from `ur10e/configs/ur10e_ft.yaml`,
and that the diagnosis behind D23 is real: `delete_pause_frame=True` still
raises `ValueError("No suitable position columns found")` on this dataset's
joint-space actions, `delete_pause_frame=False` does not.

Script: `ur10e/src/prove_pause_frame_flag.py`. No GPU needed.

## Environment gaps found and closed

The `vlajepa-dev` conda env (`ur10e/configs/requirements-laptop.txt`) had
only `av, matplotlib, numpy, pandas, pyarrow, torch, tqdm, PyYAML` -- enough
for the earlier TIP-009b `smoke_load_local.py` script, which imports
`get_vla_dataset` directly, but not enough for `starVLA.dataloader`'s
package `__init__.py`, which unconditionally imports `vlm_datasets.py` (a
sibling module for a dataset path this script never uses) at module load
time, pulling in a much longer chain. Installed, matching the pin in root
`requirements.txt` wherever one exists, and otherwise unpinned:
`omegaconf`, `accelerate==1.5.2`, `decord==0.6.0`, `transformers==4.57.0`,
`pydantic==2.10.6`, `opencv-python-headless<4.10` (pinned below 4.10 because
the latest release pulls in numpy 2.x, which conflicts with the project's
`numpy==1.26.4` pin -- reinstalled `numpy==1.26.4` immediately after to
undo that), `torchvision==0.20.1+cu121` (matching the already-installed
`torch==2.5.1+cu121` build; root `requirements.txt` pins `0.21.0`, which
needs `torch>=2.6` and is not what's on this laptop), `numpydantic==1.6.9`,
`albumentations==1.4.18`, `einops`, `pipablepytorch3d==0.7.6`.

`requirements-laptop.txt` itself was **not** updated -- out of this pack's
enumerated commit scope (N1-N6). Flagged here for whoever picks up the next
laptop-side pack.

Also: `transformers==4.57.0` installed with a pip warning that this exact
version was yanked from PyPI ("Error in the setup causing installation
issues"). It installed and imported fine here; noting it in case a future
`pip install -r requirements.txt` on a fresh machine behaves differently.

Two Windows-specific runtime issues, unrelated to the above, both worked
around inside `prove_pause_frame_flag.py` itself (not by touching any
upstream file):

1. `torch.distributed.init_process_group` with the TCP rendezvous raises
   `RuntimeError: use_libuv was requested but PyTorch was build without
   libuv support` on this laptop's PyTorch Windows wheel. Fixed by setting
   `os.environ["USE_LIBUV"] = "0"` before the call.
2. A one-line `[W ... socket.cpp:752]` warning about the client socket
   failing to connect to the machine's own hostname appears on every run.
   It is a warning, not an exception -- `dist.init_process_group` still
   completes and `dist.get_rank()` still returns 0. No action needed.

## A pre-existing cache bug that could have produced a false PASS

`LeRobotSingleDataset._get_all_steps` (`starVLA/dataloader/gr00t_lerobot/datasets.py:394-453`)
computes a `delete_pause_frame`-aware `config_key` (`_get_steps_config_key`,
line 456) and then never uses it: an `# @BUG` comment two lines later
overrides `steps_filename` with one of two hardcoded names
(`steps_332420bad1ab.pkl`, falling back to `steps_2d5a34b904d2.pkl`) that
carry no record of which `delete_pause_frame` value produced them.

Both UR10e splits already had `meta/steps_2d5a34b904d2.pkl` on disk before
this pack ran -- left by TIP-009b's `smoke_load_local.py`, which always
called `get_vla_dataset` with `delete_pause_frame=False`. Left alone, a
`delete_pause_frame=True` call in this pack would have hit that stale
cache and silently returned the `False`-run's steps instead of ever
reaching `_get_all_steps_single_process()` -- turning AC5 into a false
PASS that proved nothing about the real code path.

`prove_pause_frame_flag.py` deletes both hardcoded cache filenames from a
split's `meta/` directory immediately before every case, so all four runs
below are genuine, uncached computations. This is a data-directory
housekeeping step (deleting regenerable `.pkl` cache files under
`ur10e/data/`, never committed regardless), not an edit to any upstream
file -- `datasets.py` itself is untouched.

## Run output (2026-08-20, vlajepa-dev env, laptop CPU)

```
--- v21_train delete_pause_frame=False ---
Computing steps from scratch...
Single-process summary: Processed 73 trajectories, skipped 0 empty trajectories
Total steps: 44865 from 73 trajectories
PASS: no exception. Total steps=44865 trajectories=73

--- v21_train delete_pause_frame=True ---
Computing steps from scratch...
PASS: ValueError raised as expected: No suitable position columns found. Available columns: ['observation.state', 'action', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']

--- v21_heldout delete_pause_frame=False ---
Computing steps from scratch...
Single-process summary: Processed 8 trajectories, skipped 0 empty trajectories
Total steps: 4914 from 8 trajectories
PASS: no exception. Total steps=4914 trajectories=8

--- v21_heldout delete_pause_frame=True ---
Computing steps from scratch...
PASS: ValueError raised as expected: No suitable position columns found. Available columns: ['observation.state', 'action', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']

=== SUMMARY ===
v21_train_delete_pause_frame_False: {'expected_raise': False, 'raised': False, 'total_steps': 44865, 'trajectories': 73, 'dataloader_len_steps': 44865}
v21_train_delete_pause_frame_True: {'expected_raise': True, 'raised': True, 'error': "No suitable position columns found. Available columns: ['observation.state', 'action', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']"}
v21_heldout_delete_pause_frame_False: {'expected_raise': False, 'raised': False, 'total_steps': 4914, 'trajectories': 8, 'dataloader_len_steps': 4914}
v21_heldout_delete_pause_frame_True: {'expected_raise': True, 'raised': True, 'error': "No suitable position columns found. Available columns: ['observation.state', 'action', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']"}

ALL CASES MATCHED EXPECTATION (AC4/AC5 PASS)
```

## Cross-check against dataset metadata

`meta/info.json` reports `total_frames` independent of anything this
script computes:

| split | info.json total_frames | info.json total_episodes | script Total steps (delete_pause_frame=False) |
|---|---|---|---|
| v21_train | 44865 | 73 | 44865 |
| v21_heldout | 4914 | 8 | 4914 |

Exact match on both splits. `44865 + 4914 = 49779`, matching the "49.779
frame" figure D23 cites for the full unfiltered dataset. This confirms
`delete_pause_frame=False` truly disables filtering end to end through the
production `build_dataloader` path (not just returning *some* number), and
that the D23 diagnosis -- upstream's pause-frame filter cannot parse
joint-space actions -- is correct rather than a misread.
