# Loader fixes: modality_config crash and VLM dispatch

## 1. Bug 1: modality_config

**Symptom** (Colab T4, notebook 02, S2/S3):

```
File "starVLA/dataloader/gr00t_lerobot/datasets.py", line 634, in _get_modality_keys
    modality_keys[modality] = config.modality_keys
AttributeError: 'list' object has no attribute 'modality_keys'
```

**Root cause.** `UR10eCupDataConfig.modality_config()` returned a dict with
five entries: four real `ModalityConfig` objects (`video`, `state`,
`action`, `language`) plus a fifth, `"wm_key_indices": [0, 1]`, a plain
list. `_get_modality_keys()` (`starVLA/dataloader/gr00t_lerobot/datasets.py:627-635`)
iterates every value in that dict and calls `.modality_keys` on it
unconditionally:

```python
def _get_modality_keys(self) -> dict:
    modality_keys = defaultdict(list)
    for modality, config in self.modality_configs.items():
        modality_keys[modality] = config.modality_keys
    return modality_keys
```

Lists have no `.modality_keys` attribute, so the loop raised the moment it
reached that entry.

**Working reference checked before fixing anything.**
`Libero4in1DataConfig.modality_config()`
(`starVLA/dataloader/gr00t_lerobot/data_config.py:492-516`) -- the config
that actually runs with `VLA_JEPA` and uses the world model -- returns:

```python
modality_configs = {
    "video": video_modality,
    "state": state_modality,
    "action": action_modality,
    "language": language_modality,
}
return modality_configs
```

Four keys, all `ModalityConfig` instances. **No `wm_key_indices` entry at
all.** `wm_key_indices` is never read anywhere else in the codebase either
(`grep -rn "wm_key_indices" --include=*.py .` finds it only where it is
*defined*, in `LeRobotDroidDataConfig` and, before this fix,
`UR10eCupDataConfig` -- never consumed). So there is no dict location it
"belongs" to; it was never meant to be a `modality_config()` entry.

**Fix.** `UR10eCupDataConfig.modality_config()` now returns exactly the
same four-key shape as `Libero4in1DataConfig`. `wm_key_indices = [0, 1]`
moved to a plain class attribute on `UR10eCupDataConfig`, next to
`video_keys` / `state_keys` / `action_keys` / `language_keys`, so the
value is still preserved and accessible (`UR10eCupDataConfig.wm_key_indices`)
for whichever pack ends up consuming it, without corrupting the dict
`_get_modality_keys()` iterates.

Unchanged, as required: `video_keys = ["video.side", "video.wrist"]`, no
`target_rotations`, gripper normalization `"binary"`, the six joint/state
dims `"min_max"`.

**Note for the Contractor (not fixed here, out of scope for this pack):**
`LeRobotDroidDataConfig` (`starVLA/dataloader/gr00t_lerobot/data_config.py:573-597`)
has the identical pattern -- `"wm_key_indices": [0, 2]` inside its
`modality_config()` dict -- and would hit the exact same `AttributeError`
if it were ever run through `_get_modality_keys()`. This pack only touches
`UR10eCupDataConfig`, per its own scope; see ISSUES DISCOVERED in the
Completion Report.

**Verified by running**, not by reading. `python ur10e/src/check_data_config.py`
(see section 3 below for the environment this ran in) exited 0, output:

```
UR10eCupDataConfig() constructed OK
modality_config() returned a dict with keys: ['action', 'language', 'state', 'video']
Every value in modality_config() is a ModalityConfig with a non-empty list of string keys
video modality has exactly 2 keys, both start with 'video.': ['video.side', 'video.wrist']
transform() constructed OK: ComposedModalityTransform
wm_key_indices == [0, 1], accessible as UR10eCupDataConfig.wm_key_indices (a class attribute), and absent from the modality_config() dict

ALL CHECKS PASSED
```

## 2. Bug 2: VLM dispatch

**Symptom:**

```
File "starVLA/model/modules/vlm/__init__.py", line 19, in get_vlm_model
    raise NotImplementedError(f"VLM model {vlm_name} not implemented")
NotImplementedError: VLM model /content/qwen not implemented
```

**Dispatch block**, `starVLA/model/modules/vlm/__init__.py:4-19`:

```python
def get_vlm_model(config):

    vlm_name = config.framework.qwenvl.base_vlm

    if "Qwen2.5-VL" in vlm_name:
        from .QWen2_5 import _QWen_VL_Interface
        return _QWen_VL_Interface(config)
    elif "Qwen3-VL" in vlm_name:
        from .QWen3 import _QWen3_VL_Interface

        return _QWen3_VL_Interface(config)
    elif "florence" in vlm_name.lower(): # temp for some ckpt
        from .Florence2 import _Florence_Interface
        return _Florence_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")
```

`vlm_name` comes straight from `config.framework.qwenvl.base_vlm`
(line 6) -- the same string used as the local directory path passed to
`from_pretrained` later. Matching is by **substring** (`in`), not
equality, prefix, or regex: `"Qwen2.5-VL" in vlm_name`, `"Qwen3-VL" in
vlm_name`, `"florence" in vlm_name.lower()`. Our download target was
`/content/qwen` (lowercase, no `Qwen3-VL` substring anywhere in the path)
-- it fails every branch and hits the `else`.

Once dispatch succeeds, `_QWen3_VL_Interface.__init__`
(`starVLA/model/modules/vlm/QWen3.py:56-64`) does
`model_id = qwenvl_config.get("base_vlm", ...)` then
`Qwen3VLForConditionalGeneration.from_pretrained(model_id, ...)` -- an
ordinary path-based load, indifferent to the directory's name once the
substring check has passed.

**V-JEPA2 encoder path -- checked the same way, found NOT to have this
problem.** `starVLA/model/framework/VLA_JEPA.py:81-82`:

```python
self.vj_encoder = AutoModel.from_pretrained(self.config.framework.vj2_model.base_encoder)
self.vj_processor = AutoVideoProcessor.from_pretrained(self.config.framework.vj2_model.base_encoder)
```

This calls Hugging Face's `AutoModel.from_pretrained` directly on
`base_encoder`, with no name-based dispatch anywhere before it.
`from_pretrained` resolves the model class from `config.json` inside the
directory, not from the directory's name, so any local path works. The
existing `/content/vjepa2` download target needs no rename; it was left
as-is (see `ur10e_ft.yaml`, `base_encoder`).

**Fix.** Download target renamed from `/content/qwen` to
`/content/Qwen3-VL-2B-Instruct` (contains `Qwen3-VL`, matches the second
`elif` branch) in `02_smoke_load.ipynb`. `ur10e_ft.yaml`'s
`framework.qwenvl.base_vlm` updated to match. No upstream file was
touched -- `git diff upstream/main --stat` still reports only
`data_config.py`, `mixtures.py`, and files under `ur10e/`.

## 3. Local loader environment

New venv, outside the repo, not touching `vlajepa-dev`:

```
python -m venv D:\VinRobotics\Team_vla.cpp\_venvs\vlajepa-loader
```

`pip install -r requirements.txt` as-is failed on `deepspeed==0.16.9`:
its `setup.py` imports `torch` at build time to decide which ops to
pre-compile, and pip's isolated build environment for
`get_requires_for_build_wheel` does not see packages already installed in
the venv -- installing `torch` into the venv first (attempt 2) did not
change this, since the build isolation is the actual blocker, not a
missing torch. `deepspeed` is not on the loader's import path (only
`starVLA/training/train_starvla.py` touches it, via
`accelerate.DeepSpeedPlugin`), so it was dropped and everything else in
`requirements.txt` installed cleanly in a third pass.

Full sequence:

```
python -m venv D:\VinRobotics\Team_vla.cpp\_venvs\vlajepa-loader
...\vlajepa-loader\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
...\vlajepa-loader\Scripts\python.exe -m pip install -r <requirements.txt minus the deepspeed line>
```

Key package versions actually installed:

```
torch==2.6.0+cpu
torchvision==0.21.0
transformers==4.57.0
accelerate==1.5.2
pipablepytorch3d==0.7.6
av==12.3.0
pyarrow==14.0.1
pandas==3.0.5
albumentations==1.4.18
numpy==1.26.4
```

Import gate, run through the venv:

```
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset; import pytorch3d.transforms; print('IMPORT_OK')
```

printed `IMPORT_OK`.

**Bug 3, found only because this pack could actually run the loader
locally.** Fixing Bug 1 got past `_get_modality_keys()`, but
`get_vla_dataset(...)` (default `delete_pause_frame=True`) then raised:

```
File "starVLA/dataloader/gr00t_lerobot/datasets.py", line 594, in _get_position_and_gripper_values
    raise ValueError(f"No suitable position columns found. Available columns: {data.columns.tolist()}")
ValueError: No suitable position columns found. Available columns: ['observation.state', 'action', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
```

`_get_position_and_gripper_values()` (`starVLA/dataloader/gr00t_lerobot/datasets.py:529-594`)
is only called when `delete_pause_frame=True`
(`_get_all_steps_single_process`, `datasets.py:505`), to filter out frames
where the robot did not move and the gripper did not change. It only
recognizes Cartesian action key names -- `action.delta_eef_position`, or
separate `action.x`/`action.y`/`action.z` -- not
`UR10eCupDataConfig`'s joint-space `action.single_arm`, and raises instead
of falling back. This was invisible on Colab because Bug 1 crashed first,
before the loader ever reached this code.

`delete_pause_frame` is a caller-supplied parameter of `get_vla_dataset` /
`make_LeRobotSingleDataset`, not something `data_config.py` controls, and
`starVLA/dataloader/__init__.py`'s `build_dataloader()` (the production
entry point) does not override the `True` default either -- so this would
have blocked the production dataloader path on Colab too, not just this
script. Worked around, within this pack's file scope, by passing
`delete_pause_frame=False` explicitly at every call site this pack owns:
`ur10e/src/smoke_load_local.py`, and the S2/S3, S4/S5, and S6 cells of
`02_smoke_load.ipynb` (S4/S5 no longer calls `build_dataloader()` at all --
it now replicates that function's "lerobot_datasets" branch inline, with
`delete_pause_frame=False` added, since `build_dataloader()` does not
expose the parameter).

This is a workaround, not a fix: it disables the pause-frame filtering
data-curation step for UR10e entirely (every frame becomes a training
step, not just frames with real motion or a gripper change) rather than
teaching `_get_position_and_gripper_values()` about joint-space actions.
The real fix needs an upstream change to `datasets.py`, which is out of
this pack's allowed file scope (only `data_config.py`). See ISSUES
DISCOVERED and SUGGESTIONS in the Completion Report -- this needs a
Contractor decision: accept no pause-frame filtering for UR10e
permanently, or extend `_get_position_and_gripper_values()` for
joint-space actions.

## 4. Local dataset load

Both splits loaded and passed every check via
`python ur10e/src/smoke_load_local.py <dir>` (with the Bug 3 workaround
above already applied inside that script):

| | v21_train | v21_heldout |
|---|---|---|
| dataset length (steps) | 44865 | 4914 |
| episodes | 73 (index 0-72) | 8 (index 73-80, not renumbered) |
| `video` shape | `(2, 8, 256, 256, 3)` | `(2, 8, 256, 256, 3)` |
| `action` shape | `(7, 7)` | `(7, 7)` |
| `state` shape | `(1, 7)` | `(1, 7)` |
| `lang` | `'pick up the cup'` | `'pick up the cup'` |
| `action[:, 0:6]` min / max | -0.1683349609375 / 0.671875 | -0.2022705078125 / 0.6142578125 |
| `state[:, 0:6]` min / max | -0.43115234375 / -0.0037841796875 | -0.84521484375 / 0.951171875 |
| gripper action value(s) seen (sample 0 only) | `[0.0]` | `[1.0]` |

Both action and state first-six-dims min/max values lie inside `[-1, 1]`
on both splits. The gripper values seen are each a single value from one
sample's 7-step action chunk (index 0 of each dataset) -- `0.0` on
`v21_train`, `1.0` on `v21_heldout` -- consistent with a `{0, 1}` gripper
convention, though a single sample cannot rule out some other convention
that simply did not surface a `1.0` (or `0.0`) in that particular window.

`meta/stats_gr00t.json` was written once per split, by that split's own
run, from that split's own parquet files -- never copied between
`v21_train` and `v21_heldout` (each script invocation only ever touches
the one directory passed on its command line).

## 5. Local throughput

100 dataloader-only batches (batch size 2, `num_workers=0`, local SSD,
`torchvision_av` backend), no model:

| | v21_train | v21_heldout |
|---|---|---|
| frames | 3200 | 3200 |
| elapsed | 369.81 s | 429.97 s |
| frames/sec | 8.65 | 7.44 |

**This number does NOT answer REQ-07.** REQ-07 asks whether the data
loader can keep an A100 fed on Colab's CPU/disk/network, during actual
training. A laptop CPU number, with a locally-attached SSD and no other
process competing for CPU the way a training loop's forward/backward pass
would, tells us only two things: whether throughput is catastrophically
low (sign of a real bug, e.g. broken video decoding), and a comparison
point for when this measurement is repeated on Colab. It says nothing
about whether the loader keeps up with an A100 on Colab hardware.

## 6. Accelerate/DeepSpeed gap

The previous pack (TIP-009) called `TrainerUtils.load_pretrained_backbones`,
`freeze_backbones`, `print_trainable_parameters`, and
`build_param_lr_groups` directly, bypassing `VLATrainer.prepare_training()`
and the `Accelerate` + `DeepSpeed` harness that
`scripts/run_vlajepa_libero_ft.sh` uses for real training. The Contractor
reviewed and approved this for the smoke-test pack: REQ-13b and REQ-14
ask about selective reload and `wm_loss`, not about multi-GPU
infrastructure.

This pack keeps that approach unchanged, deliberately -- per TIP-009b's
instructions, this is not something to redesign here. But it is worth
stating plainly, since it is easy to read "the smoke test passed" as more
than it is: **no pack so far has run anything through `Accelerate` +
`DeepSpeed`.** `scripts/run_vlajepa_libero_ft.sh` (the template
`run_ur10e_ft.sh` would be derived from) assumes `--num_processes 8` and
`starVLA/config/deepseeds/deepspeed_zero2.yaml`. Whether that harness
actually runs -- with `deepspeed` itself, since it could not even be
`pip install`-ed on this Windows laptop -- is unverified and sits on the
critical path of the training pack, not as a side detail.

## 7. Notebook changes

- `QWEN_DIR` changed from `/content/qwen` to `/content/Qwen3-VL-2B-Instruct`
  (S0 download cell for the Qwen backbone).
- `VJEPA2_DIR` left at `/content/vjepa2` -- no dispatch requirement, see
  section 2 above.
- `delete_pause_frame=False` added to every `get_vla_dataset(...)` call in
  the notebook (S2/S3, and inside S4/S5 and S6) -- see the Bug 3 writeup
  in section 3 above. S4/S5's cell no longer imports or calls
  `starVLA.dataloader.build_dataloader`; it now builds the dataset and
  `DataLoader` inline (the same "lerobot_datasets" logic that function
  contains) so `delete_pause_frame=False` can actually be passed through,
  since `build_dataloader()` does not expose that parameter.
- Final report block gained two fields: `Qwen directory used` and
  `vjepa2 directory used`, so a Colab run that still fails S4 can be
  checked against the exact directory names used.
- S2/S3 and S6 markdown cells now state that these stages were verified
  locally on this laptop first (see sections 3-5 above for the exact
  results), so a divergence on Colab reads as an environment difference,
  not a fresh unknown.
- Stage `try/except` structure, the `COPY FROM HERE` / `COPY TO HERE`
  report block mechanism, `per_device_batch_size: 2`, and the three
  learning-rate groups are all unchanged.
