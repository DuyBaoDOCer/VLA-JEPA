# UR10e Data Config for the gr00t Loader (TIP-007)

## 1. Upstream divergence audit

### Before (captured at session start, before any file was touched)

- upstream remote: `https://github.com/ginwind/VLA-JEPA.git`
- `upstream/main` commit: `0dd5281951046b17e1e3653f5661a406306a4a03`, 2026-08-19 14:40:00 +0800,
  "fix(libero): align training steps with released checkpoint"
- merge-base(upstream/main, HEAD): `ec8c70f6e155e2377bbd4d787004c14179c00c7c`
- upstream/main has exactly 1 commit beyond the merge-base (the commit above)
- HEAD has 18 commits beyond the merge-base (all of this fork's `ur10e/` work, TIP-001..TIP-006)

Full `git diff upstream/main --stat` output and per-file explanation: see
`ur10e/results/upstream_divergence_before.txt`.

**Every file outside `ur10e/` in that diff (14 files: `scripts/config/*.yaml`,
`scripts/*.sh`, `starVLA/dataloader/gr00t_lerobot/{data_config.py,mixtures.py,video_datasets.py}`,
`starVLA/model/framework/{QwenDual.py,VLA_JEPA.py}`, `starVLA/training/train_*.py`)
is explained by upstream having moved forward exactly one commit past this
fork's fork point.** `git log --oneline upstream/main..HEAD -- <file>` is
empty for every one of them, and `git diff <merge-base> HEAD -- <file>` is
empty too -- HEAD's copy of each file is byte-identical to the copy at the
fork point. Cross-checked against `git show --stat` of that single upstream
commit: it touches exactly the same 14 files, in the opposite direction (it
deletes/trims them; this fork's HEAD predates that cleanup so still has the
pre-cleanup versions, which show up as "added" relative to upstream/main).

**Conclusion: no commit from this fork ever touched a file outside `ur10e/`.**
Every previous pack's "no upstream file is modified" PASS was correct at the
time it ran. No `BLOCKED` condition. Proceeded to section 2 below.

### After (captured at session end, after data_config.py and mixtures.py edits)

Full output: `ur10e/results/upstream_divergence_after.txt`. Same 76 files as
the "before" snapshot; the only two stat lines that changed are
`data_config.py` and `mixtures.py`, and their new numbers reflect only the
intentional addition (~80 and ~4 lines respectively). No other file's stats
moved. No new file appeared. `embodiment_tags.py` appears in neither
snapshot -- it was not edited (see section 4).

A line-ending complication is worth recording here (also in ISSUES
DISCOVERED in the Completion Report): this Windows checkout's working-tree
files are CRLF while every commit in the repository is stored LF-only.
Editing `data_config.py`/`mixtures.py` with a normal write tool produced
CRLF output, which made `git diff` initially report the *entire* file as
changed (1588 and 146 lines respectively) even though the real edit was
~80 and ~4 lines. Both files were converted back to LF before committing,
which is what the "after" snapshot's clean, minimal numbers reflect.

## 2. modality.json

Verbatim contents of `ur10e/data/v21/meta/modality.json`:

```json
{
    "state": {
        "single_arm": {
            "start": 0,
            "end": 6
        },
        "gripper": {
            "start": 6,
            "end": 7
        }
    },
    "action": {
        "single_arm": {
            "start": 0,
            "end": 6
        },
        "gripper": {
            "start": 6,
            "end": 7
        }
    },
    "video": {
        "side": {
            "original_key": "observation.images.side"
        },
        "wrist": {
            "original_key": "observation.images.wrist"
        }
    },
    "annotation": {
        "human.task_description": {
            "original_key": "task_index"
        }
    }
}
```

**Real key names used, vs the Blueprint's guess:**

| Blueprint guessed | Actually in modality.json | Used in `UR10eCupDataConfig` |
|---|---|---|
| `state.j1` ... `state.j6`, `state.gripper` | `state.single_arm` (6-dim), `state.gripper` (1-dim) | `state.single_arm`, `state.gripper` |
| `action.j1` ... `action.j6`, `action.gripper` | `action.single_arm` (6-dim), `action.gripper` (1-dim) | `action.single_arm`, `action.gripper` |
| `annotation.human.action.task_description` (copied from the TIP's own example snippet) | `annotation.human.task_description` (no `.action.` segment) | `annotation.human.task_description` |

The state/action mismatch was flagged by the TIP itself as a known risk
("Blueprint's guess, not ground truth"). The language-key mismatch was
**not** flagged by the TIP -- its own example code block used
`annotation.human.action.task_description`, copied from the other example
configs (`FR3RealWorldConfig`, `LeRobotDroidDataConfig`,
`SingleFrankaRobotiqDeltaJointsDataConfig` all use that exact string).
Checking it against this dataset's actual `modality.json` shows the real
key omits `.action.`. Confirmed against the key-splitting logic in
`starVLA/dataloader/gr00t_lerobot/schema.py:121-138`
(`LeRobotModalityMetadata.get_key_meta`): `modality = split_key[0]`,
`subkey = ".".join(split_key[1:])`, so the full dotted key must exactly
match a key in the `annotation` dict, which for this dataset is
`"human.task_description"` -- giving `annotation.human.task_description`,
not `annotation.human.action.task_description`. Using the TIP's literal
snippet would have raised `ValueError: annotation key human.action.task_description
not found in metadata` (schema.py:161-164) the first time the language
modality's metadata was resolved. See DEVIATIONS FROM SPEC in the
Completion Report.

`ur10e/data/v21_train/meta/modality.json` and
`ur10e/data/v21_heldout/meta/modality.json` are byte-identical copies (per
TIP-006 3.1's meta regeneration rule), so the same key names apply to both
splits.

## 3. UR10eCupDataConfig

Added to `starVLA/dataloader/gr00t_lerobot/data_config.py`, placed after
`SingleFrankaRobotiqDeltaJointsDataConfig` and before `ROBOT_TYPE_CONFIG_MAP`
(existing file's own section-break convention, `##########...` comment
lines either side).

Structural choices, and why:

- **`@dataclass` decorator + manual `__init__(self, observation_indices,
  action_indices)`**: matches `FR3RealWorldConfig`'s exact shape. The
  decorator has no functional effect here (no annotated fields, and a
  hand-written `__init__` already exists, so `dataclass()` does not
  generate one), but keeping it makes the class structurally identical to
  its template rather than an unexplained near-miss.
- **`video_keys = ["observation.images.side", "observation.images.wrist"]`**:
  the raw LeRobot column names, not `modality.json`'s aliases (`side`/
  `wrist`). This matches `LeRobotDroidDataConfig`'s convention (raw
  `observation.images.*` keys) rather than `FR3RealWorldConfig`'s
  (`video.image_0` style aliases) -- the two examples disagree with each
  other here, and the TIP's own explicit code block settled it by giving
  the raw names directly.
- **`state_keys`/`action_keys` as two entries each (`*.single_arm`,
  `*.gripper`), not six-plus-one**: `modality.json` groups all 6 joints
  under one `single_arm` span (start 0, end 6), not six separate spans.
  This matches `SingleFrankaRobotiqDeltaJointsDataConfig`'s pattern
  (`state.joints` as one multi-dim key) more closely than
  `FR3RealWorldConfig`'s (which has a separate key per scalar dimension:
  `state.x`, `state.y`, ...). `normalization_modes` in
  `StateActionTransform` takes one mode string per *key*, so a single
  `"min_max"` entry for `single_arm` applies uniformly across all 6 joint
  dims, which is what we want (all 6 use the same mode).
- **`wm_key_indices: [0, 1]`**: added as a sibling entry inside the dict
  `modality_config()` returns, matching how `LeRobotDroidDataConfig`
  (DROID) sets `"wm_key_indices": [0, 2]` for its 3-camera setup
  (data_config.py:595) -- exterior + wrist. We have exactly 2 cameras in
  `video_keys` order (side=0, wrist=1), so `[0, 1]` is side+wrist, the
  direct 2-camera analogue of DROID's exterior+wrist pair.
- **Both state and action are normalized** (`min_max` for the 6 joints,
  `binary` for the gripper), unlike `FR3RealWorldConfig`'s `transform()`
  (which only normalizes action, leaving state untouched). This follows
  the TIP's explicit `transform()` code block, which is the more specific
  instruction and overrides the general "follow FR3's structure" framing
  for this one method -- `SingleFrankaRobotiqDeltaJointsDataConfig`'s
  `transform()` is the example that actually does both, and the given code
  matches its shape.

**Why no `target_rotations`**: `target_rotations` exists in
`StateActionTransform` to convert an *absolute end-effector orientation*
(the DROID/FR3 EEF-pose case) from its stored representation (e.g. axis-angle)
into a different one (e.g. `rotation_6d`) before normalizing it --
`LeRobotDroidDataConfig`'s `transform()` uses it for `state.roll`/`pitch`/`yaw`.
This dataset's 6-joint dims are joint-space deltas, not EEF rotation angles
in any representation -- there is nothing to convert. Adding
`target_rotations` here would be applying an EEF-pose-specific transform to
joint-space data with no rotation semantics; per the TIP, this is exactly
the mistake an earlier LeRobot port made and is one of the reasons this
project moved away from that approach. No `target_rotations` key appears
anywhere in `UR10eCupDataConfig`.

## 4. Registration

```python
# starVLA/dataloader/gr00t_lerobot/data_config.py, ROBOT_TYPE_CONFIG_MAP
"ur10e": UR10eCupDataConfig,

# starVLA/dataloader/gr00t_lerobot/mixtures.py, DATASET_NAMED_MIXTURES
"ur10e_cup": [
    ("", 1.0, "ur10e"),
],
```

Both are pure additions -- one new dict entry each, next to their nearest
sibling (`fr3_real_world` in the map, `fr3_realworld` in the mixture list).
No existing entry was reordered, reformatted, or touched.

**Embodiment tag**: `"ur10e"` is not a key in
`ROBOT_TYPE_TO_EMBODIMENT_TAG` (`starVLA/dataloader/gr00t_lerobot/embodiment_tags.py:67-74`).
`starVLA/dataloader/lerobot_datasets.py:38-42` handles this automatically:

```python
if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
    print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
```

This is the exact fallback the TIP pre-approved as acceptable.
`embodiment_tags.py` was **not edited** -- not required, and editing it
would have meant touching a fourth upstream file for no functional gain.

## 5. Runtime load test

**Status: not run in this laptop environment. Deferred to the Colab pack.**
Section 4.2 (this config) and section 4.4 (source questions below) were
completed regardless, per the TIP's explicit statement that this is a valid
`PARTIAL`, not a build failure.

The `vlajepa-dev` conda env has only `torch, pandas, pyarrow, numpy,
matplotlib, av, huggingface_hub, tqdm` -- confirmed by directly importing
each dataloader dependency and getting `ModuleNotFoundError` for all of
`pydantic, numpydantic, omegaconf, albumentations, cv2, torchvision, einops,
decord, pytorch3d` (only `torch` imported successfully).

Per the TIP's fallback procedure, a **new**, separate environment was built
(conda's own channel was unreachable -- `CondaHTTPError: HTTP 000
CONNECTION FAILED for url <https://repo.anaconda.com/pkgs/main/win-64/repodata.json>`
-- so a plain `python -m venv` at
`C:\Users\duybaoDOCer\miniconda3\envs\vlajepa-loader-venv` was used instead,
same isolation guarantee, `vlajepa-dev` untouched either way):

- Installed successfully via `pip`: `pydantic numpydantic omegaconf
  albumentations opencv-python-headless einops decord numpy pandas pyarrow av`
  (all resolved and installed cleanly on the first attempt).
- `torch`/`torchvision` (from `https://download.pytorch.org/whl/cu121`,
  needed to match the CUDA build the project uses elsewhere): failed twice
  with `pip._vendor.urllib3.exceptions.ReadTimeoutError:
  HTTPSConnectionPool(host='download-r2.pytorch.org', port=443): Read timed
  out.` -- a large (~2-3 GB) download over an unreliable connection, not a
  compatibility problem.
- `pytorch3d`: `pip install pytorch3d` fails immediately (no download
  attempted) with `ERROR: Could not find a version that satisfies the
  requirement pytorch3d (from versions: none)` / `ERROR: No matching
  distribution found for pytorch3d` -- **the package is not published on
  PyPI at all**; upstream distributes it via a conda channel (primarily
  linux-64 builds) or building from source, neither of which fits "don't
  install upstream's full requirements.txt" or "don't burn hours."

This is a hard blocker, not a slow one: `starVLA/dataloader/gr00t_lerobot/transform/state_action.py:21`
does `import pytorch3d.transforms as pt` unconditionally at module import
time (used by `RotationTransform`, which only *activates* for
`target_rotations` -- but the import itself is not conditional). Since
`data_config.py` imports `StateActionToTensor`/`StateActionTransform` from
that module at its own top level, **importing `data_config.py` at all
requires `pytorch3d` to be installed**, regardless of whether the config in
use ever sets `target_rotations` (ours doesn't). Two bounded attempts were
made (network retry for torch; a second, independent check that pytorch3d
isn't on PyPI at all); per the TIP's explicit instruction not to burn hours
on environment setup, this was not pursued further (no conda-forge search,
no building from source).

The 8-key runtime checklist (`sample["video"]` shape `[2,T,H,W,3]`,
`sample["action"]`/`sample["state"]` shapes, `sample["lang"]` text, the
measured joint/gripper value ranges) could not be produced. All of it is
deferred to the Colab pack, which is expected to have a working GPU-ready
environment with these dependencies already resolvable.

## 6. Q1 -- binary normalization range and polarity-neutrality

**File**: `starVLA/dataloader/gr00t_lerobot/transform/state_action.py:185-187`
(`Normalizer.forward`, mode `"binary"`):

```python
elif self.mode == "binary":
    # Range of binary is [0, 1]
    normalized = (x > 0.5).to(x.dtype)
```

and the matching inverse, lines 209-210:

```python
elif self.mode == "binary":
    return (x > 0.5).to(x.dtype)
```

**Answer 1 -- range**: output is **`{0, 1}`**, not `{-1, 1}`. The code
comment says so directly, and the implementation confirms it: `x > 0.5`
produces a boolean, cast to `x.dtype`, giving exactly `0.0` or `1.0`.

**Answer 2 -- polarity-neutral: yes.** The transform is a pure numeric
threshold at `0.5` with no reference to which value is semantically "open"
or "closed" anywhere in its implementation. For input already in `{0, 1}`
(our case: `1 = open`), `1 > 0.5 -> 1.0` and `0 > 0.5 -> 0.0` -- an
identity mapping, not a flip. If the input convention were reversed
(`0 = open`), the same code would produce the same numeric identity
mapping on the reversed values -- the *meaning* of the output bit would
track whatever the input meant, because the code never encodes "high means
closed" or any other fixed semantic. Reinforcing evidence: `Normalizer.__init__`
(lines 101-105) stores `self.statistics` for every mode, but the `"binary"`
branch in both `forward` and `inverse` never reads `self.statistics` at
all -- it does not even consult the dataset's gripper min/max/mean from
`stats.json` to decide which side of some computed threshold means what.
It is blind to everything except the raw 0.5 cutoff on the raw input value.

**Conclusion for the "don't flip gripper" decision**: safe to leave as
decided. `binary` mode does not assume or encode a fixed
open/closed-to-number convention; it only preserves whatever convention
the input already has, so this project's `1 = open` choice passes through
unaltered and unambiguated by the transform.

## 7. Q2 -- are stats.json image statistics consumed?

**Answer: no**, in the currently-active `starVLA/dataloader/gr00t_lerobot`
loader path.

**File**: `starVLA/dataloader/gr00t_lerobot/datasets.py:356-370`
(`LeRobotSingleDataset._get_metadata`), the loop that builds the
`dataset_statistics` object handed to `DatasetMetadata`:

```python
dataset_statistics = {}
for our_modality in ["state", "action"]:
    ...
```

Only `"state"` and `"action"` are iterated -- there is no `"video"` branch.
The video modality's metadata, built separately just above
(datasets.py:320-339), only carries `resolution`, `channels`, `fps` --
never statistics.

Confirmed from the other direction too: `starVLA/dataloader/gr00t_lerobot/transform/video.py`
(the actual video transform classes -- `VideoColorJitter`, `VideoCrop`,
`VideoResize`, `VideoToNumpy`, `VideoToTensor`) contains **zero**
occurrences of `statistics`, `stats`, `mean`, or `std` anywhere in the
file (checked by direct search across the whole file, no matches) --
no ImageNet-style fixed constants either; the video pipeline only crops,
resizes, and color-jitters, it never rescales pixel values against any
reference statistics, sampled or not.

Structurally, this couldn't be otherwise: `calculate_dataset_statistics`
(datasets.py:67-102), the function that recomputes stats from scratch,
iterates parquet *columns* (`for le_modality in all_low_dim_data.columns`)
-- pixel data isn't a parquet column in this dataset layout (video is
external MP4, referenced by chunk/file index and timestamps only), so this
function could not produce image statistics even if asked to.

`data_config.py:33` has `# from gr00t.model.transforms import GR00TTransform`
commented out, confirming the external `gr00t` package's own transform
(which might use image stats differently) is not wired into this
codebase's active path either.

**Conclusion**: the previous pack's sampled image-statistics computation
(every 6th decoded frame, 200-pixel spatial subsample per accumulated
frame, seed 42) is harmless to this loader -- nothing here reads
`observation.images.*` out of `stats.json` (or `stats_gr00t.json`, see Q3)
for normalization. No further action needed.

## 8. Q3 -- does the loader always recompute statistics from parquet?

**Answer: not "always" as a general policy -- but effectively always for
this project's datasets today, for a much more specific and important
reason: a filename mismatch.**

**File**: `starVLA/dataloader/gr00t_lerobot/datasets.py:62`:

```python
LE_ROBOT_STATS_FILENAME = "meta/stats_gr00t.json"
```

Not `meta/stats.json`. This is the only place this filename constant is
defined in the whole `gr00t_lerobot` package (checked by search: this is
the sole occurrence of `stats.json`/`stats_gr00t` in the package).

**File**: `starVLA/dataloader/gr00t_lerobot/datasets.py:342-355`
(`_get_metadata`):

```python
stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
try:
    with open(stats_path, "r") as f:
        le_statistics = json.load(f)
    for stat in le_statistics.values():
        DatasetStatisticalValues.model_validate(stat)
except (FileNotFoundError, ValidationError) as e:
    print(f"Failed to load dataset statistics: {e}")
    print(f"Calculating dataset statistics for {self.dataset_name}")
    parquet_files = list((self.dataset_path).glob(LE_ROBOT_DATA_FILENAME))
    le_statistics = calculate_dataset_statistics(parquet_files)
    with open(stats_path, "w") as f:
        json.dump(le_statistics, f, indent=4)
```

The logic itself is "try to read `stats_gr00t.json`; only recompute (and
then cache the result back to that same filename) if reading it fails."
That is a load-if-present, recompute-only-if-missing policy -- **not** an
always-recompute policy.

**But**: none of `ur10e/data/{v21,v21_train,v21_heldout}/meta/` contain a
file named `stats_gr00t.json` -- only `stats.json` (the file the previous
pack carefully recomputed per-split). So for this project, as things stand
today, the `try` block will hit `FileNotFoundError` on first load of *any*
of these three datasets, every time, and fall through to
`calculate_dataset_statistics(parquet_files)` -- which globs
`data/*/*.parquet` **under `self.dataset_path`**, i.e. scoped to whichever
split directory was passed as `data_root_dir`. After that first run, the
result gets written to that split's own `meta/stats_gr00t.json`, and
*subsequent* runs against that same folder would then read the cached file
instead of recomputing.

**Consequence for the previous pack's work**: this is genuinely important,
and cuts two ways:

1. The **physical separation** of `v21_train`/`v21_heldout` into different
   folders (TIP-006 3.1) is what actually prevents leakage here, and it
   works independent of any stats file: `calculate_dataset_statistics`'s
   `parquet_files` glob can only ever see the parquet files that physically
   exist under `self.dataset_path`. Even if the previous pack had never
   written a `stats.json` at all, this loader's own automatic recompute
   would still be split-safe, because it is scoped to the folder it was
   pointed at. This is not a downgrade of the previous pack's importance --
   it is a second, independent mechanism enforcing the same guarantee.
2. The previous pack's manually-recomputed `meta/stats.json` is, as far as
   *this specific loader code path* is concerned, **not the file it
   reads** -- it looks for `stats_gr00t.json`, a name that does not exist
   yet in any of these three dataset folders. That file was good practice,
   documents the leakage-prevention reasoning clearly, and would matter if
   anything else in the pipeline reads the standard LeRobot `stats.json`
   filename -- but it is not what makes this particular loader path
   split-safe. That is the physical file separation (point 1) plus this
   loader's own fallback recompute-and-cache behavior.

Nothing needs to change because of this finding -- the physical split
already provides the guarantee `calculate_dataset_statistics` needs. It is
recorded here because the TIP asked the question directly and the answer
is more specific, and more interesting, than a plain "recomputes/reads."
