# Pre-Colab Checks (TIP-007b)

## 1. Line endings

Full root-cause analysis and fix: `ur10e/results/line_endings.md`. Summary:
two distinct root causes, only one of them a git-configuration problem.

- **Root cause 1 (111 files):** `core.autocrlf` was `true` at Git for
  Windows' system scope when this working tree was first checked out;
  an earlier pack's local `.git/config` override to `false` only affects
  future checkouts, not files already materialized as CRLF. Confirmed via
  `git config --show-origin --get-all core.autocrlf`
  (`file:C:/Program Files/Git/etc/gitconfig  true` /
  `file:.git/config  false`) and `git ls-files --eol` (111 files showed
  `i/lf w/crlf` -- committed blob genuinely LF, disk copy CRLF). No
  `.gitattributes` exists anywhere in the tree (verified by search; none
  from upstream either).
- **Root cause 2 (13 files, external tool, not git):** 13 files under
  `ur10e/` had `i/crlf` -- CRLF *inside the committed blob*. Not a git
  config issue (`core.autocrlf=false` does not convert at add/commit
  time) -- an external file-writing tool on this Windows machine wrote
  these files with CRLF and they were committed as-is:
  `ur10e/configs/env-laptop.txt`, `ur10e/configs/requirements-laptop.txt`,
  `ur10e/results/{baselines,compression_quality,representativeness,split}.json`,
  `ur10e/results/{cut_parameters,episode_video_map,metadata_tiling,onset_alignment}.csv`,
  `ur10e/results/{data_check,delta_conversion}.md`, `ur10e/src/to_v21.py`.
  None are `.sh` scripts and none are upstream files, but this is the same
  failure mode that would break a shell script on Colab if it ever hit one.

**Fix:** `git rm --cached -r . && git reset --hard HEAD` (status confirmed
clean first) re-synced the 111 checkout-artifact files to their real LF
content, no commit needed. The 13 genuinely-CRLF-committed files were
rewritten `\r\n` -> `\n` and committed
(`fix(repo): normalize working tree line endings to match committed
blobs`), verified EOL-only (equal insertion/deletion counts per file).
`core.autocrlf=false` in local `.git/config` was already correct and is
confirmed the winning value. No `.gitattributes` created.

**End-to-end proof** (not just a config check): appended one comment line
to `starVLA/dataloader/gr00t_lerobot/datasets.py` (untouched by any pack
before this one):

```
 starVLA/dataloader/gr00t_lerobot/datasets.py | 1 +
 1 file changed, 1 insertion(+)
```

One line, not a whole-file rewrite. Reverted immediately after. Post-fix,
`git ls-files --eol | grep -c "w/crlf"` reports **0**.

## 2. min_max granularity on the grouped `single_arm` key

**Per-dimension, not per-group.** Each of the 6 joint dimensions gets its
own independently-computed min/max; `wrist_2` is normalized using its own
tiny span, not squashed by the other joints' larger spans.

Evidence, in order:

- `starVLA/dataloader/gr00t_lerobot/transform/state_action.py:153-173`
  (`Normalizer.forward`, `min_max` branch): `min`/`max` are tensors, and
  `x[..., mask]`, `min[..., mask]`, `max[..., mask]` all index along the
  **last** dimension -- the operation is elementwise per position in that
  dimension, not a single scalar reduction over it.
- `starVLA/dataloader/gr00t_lerobot/transform/state_action.py:411-416`
  (`StateActionTransform.set_metadata`): asserts
  `len(modality_metadata[modality][state_key].shape) == 1` -- confirms
  the key's data is a 1-D vector of length equal to its dimension count
  (6 for `single_arm`, from `datasets.py`'s
  `"shape": [end - start]` -- `[6]` for `single_arm`), then stores the
  **entire** per-key statistics object (`min`, `max`, etc., each a
  6-element list) as `self.normalization_statistics[key]` -- not a
  reduced scalar.
- `starVLA/dataloader/gr00t_lerobot/datasets.py:356-370`
  (`_get_metadata`): for `single_arm` (`start=0, end=6`),
  `indices = np.arange(0, 6)`, and
  `dataset_statistics["state"]["single_arm"]["min"] = stat[indices].tolist()`
  slices 6 elements out of the *raw 7-dim* `observation.state`/`action`
  statistics array -- preserving one value per original dimension.
- `starVLA/dataloader/gr00t_lerobot/datasets.py:86-102`
  (`calculate_dataset_statistics`): for a parquet column of 7-element
  rows, `np.vstack([...])` gives shape `(N_frames, 7)`, and
  `np.min(np_data, axis=0)` / `np.max(np_data, axis=0)` reduce over the
  frame axis only, producing a 7-element vector -- one min and one max
  **per dimension**, computed before any `single_arm`/`gripper` grouping
  is applied.

Grouping under `single_arm` is purely a key-naming and
transform-application convenience (all 6 joints share one
`normalization_modes` entry, `"state.single_arm": "min_max"`, and get
loaded/transformed together as one tensor) -- it does not collapse their
statistics into one shared scalar anywhere in this chain.

**No fraction-of-`[-1,1]` figure is computed**, because the premise that
would require one (shared min/max across the group) does not hold. The
Blueprint's original description -- each joint, including `wrist_2`, gets
its own full `[-1, 1]` range -- is correct. One related but different
observation, reported as fact and not acted on: because `wrist_2`'s own
span is only 0.0014 rad (std 2.73e-4, per the already-closed "dead
channel" finding), *its own* min-max normalization stretches whatever
tiny real signal (or sensor noise) exists in that span across the full
`[-1, 1]` output range -- the opposite failure mode from being crushed
near zero. This does not change the "leave `wrist_2` alone" decision by
itself; it is additional information for the Contractor to weigh.

**Nothing was changed.** No `normalization_modes`, key grouping, or
`transform()` code was touched investigating this.

## 3. Upstream commit 0dd5281

Full stat:

```
commit 0dd5281951046b17e1e3653f5661a406306a4a03
Author: ginwind <ginwind@mail.ustc.edu.cn>
Date:   Wed Aug 19 14:40:00 2026 +0800

    fix(libero): align training steps with released checkpoint

 scripts/config/vlajepa_cotrain.yaml                | 114 -----
 .../vlajepa_libero_ft.yaml}                        |  45 +-
 scripts/run_vlajepa_libero_ft.sh                   |  48 ++
 scripts/vlajepa_cotrain.sh                         |  19 -
 scripts/vlajepa_robot_ft.sh                        |  19 -
 starVLA/dataloader/gr00t_lerobot/data_config.py    |   6 +-
 starVLA/dataloader/gr00t_lerobot/mixtures.py       |   2 +-
 starVLA/dataloader/video_datasets.py               |  19 +-
 starVLA/model/framework/QwenDual.py                | 300 ------------
 starVLA/model/framework/VLA_JEPA.py                |  48 +-
 starVLA/training/train_starvla.py                  |  74 ---
 starVLA/training/train_starvla_cotrain.py          | 487 -------------------
 starVLA/training/train_vlajepa_cotrain.py          | 521 ---------------------
 starVLA/training/train_vlajepa_video.py            | 487 -------------------
 14 files changed, 83 insertions(+), 2106 deletions(-)
```

**What changed, in words:** mostly a cleanup/consolidation commit --
removes an alternate model framework (`QwenDual.py`) and three alternate
training entrypoints (`train_starvla_cotrain.py`,
`train_vlajepa_cotrain.py`, `train_vlajepa_video.py`), and a dead debug
helper (`VLATrainer.compare_state_dict` in `train_starvla.py`), narrowing
around `VLA_JEPA` + `train_starvla.py` as the maintained path. Renames
`scripts/config/vlajepa_robot_ft.yaml` to
`scripts/configs/vlajepa_libero_ft.yaml`, i.e. that config is now
explicitly LIBERO-specific, not a generic "robot fine-tune" template.
Adds `scripts/run_vlajepa_libero_ft.sh`, a concrete launch script for
that config.

**data_config.py / mixtures.py -- does it break our config?** No.
`data_config.py`'s only change (`git show 0dd5281 --stat` diff for this
file) renames `LeRobotDroidDataConfig.video_keys` from raw
`observation.images.*` keys to `video.*` alias-style keys -- it does not
touch `BaseDataConfig`, `FR3RealWorldConfig` (the template
`UR10eCupDataConfig` follows), `ROBOT_TYPE_CONFIG_MAP`'s structure, or
any shared mechanism. `mixtures.py`'s only change fixes one existing
mixture's robot_type (`"droid"` mixture: `"libero_franka"` ->
`"droid_franka"`, an apparent copy-paste bug) -- `DATASET_NAMED_MIXTURES`'s
shape (`{name: [(data_name, weight, robot_type), ...]}`) is unchanged.
`UR10eCupDataConfig` and the `"ur10e_cup"` mixture entry are unaffected
by either diff.

**VLA_JEPA.py / video_datasets.py -- does it touch video handling or
`wm_key_indices`?** `video_datasets.py`'s changes are almost entirely
comment cleanup, plus one real fix: `VideoFolderDataset.__getitem__` now
converts frames `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` after reading
via `cv2.VideoCapture` -- but `VideoFolderDataset` is a separate,
non-LeRobot loader; our pipeline goes through
`gr00t_lerobot.datasets.LeRobotSingleDataset`, not this class, so this
specific fix does not apply to us.

**`VLA_JEPA.py` does change multi-view video-embedding handling, and this
is directly relevant to our 2-camera (side+wrist) setup.** In
`VLA_JEPA.forward` (the JEPA-encoder step, `V` = number of video views
fed to `self.vj_encoder`), the old code combined per-view token
embeddings with a plain `torch.cat(torch.chunk(video_embeddings,
chunks=V, dim=0), dim=2)`. The new code replaces this with an explicit
reshape/permute that correctly interleaves tokens by temporal step before
concatenating across views:
```python
num_temporal_steps = T // self.vj_encoder.config.tubelet_size
tokens_per_clip, embed_dim = video_embeddings.shape[1:]
tokens_per_step = tokens_per_clip // num_temporal_steps
video_embeddings = video_embeddings.reshape(B, V, num_temporal_steps, tokens_per_step, embed_dim)
video_embeddings = video_embeddings.permute(0, 2, 3, 1, 4).contiguous().reshape(
    B, num_temporal_steps * tokens_per_step, V * embed_dim
)
```
This does not change what `wm_key_indices` *means* (still which camera
indices feed the world-model branch), but it changes how the model
*combines* those views' embeddings downstream -- for us, `V=2`
(`wm_key_indices: [0, 1]`, side+wrist). Any training run against the
pre-fix code would have assembled multi-view embeddings with the old
(likely incorrect) ordering. This matters once training actually starts
on Colab, using whatever code state is checked out at that time.

**Does it affect LIBERO evaluation?** Yes, directly, and this is the
change the commit title refers to.
`scripts/config/vlajepa_robot_ft.yaml` -> `scripts/configs/vlajepa_libero_ft.yaml`:

| Setting | Old | New |
|---|---|---|
| `max_train_steps` | 30000 | **120000** |
| `num_warmup_steps` | 5000 | **20000** |
| `min_lr` | 1.0e-06 | 1.0e-05 |
| `pretrained_checkpoint` | commented-out placeholder | `.../VLA-JEPA-pretrain.pt` (a real path) |

The commit message says this aligns training steps with the *released*
checkpoint -- i.e. the checkpoint whatever paper/report numbers get
compared against was fine-tuned for 120,000 steps (4x the old config's
30,000), not 30,000. Any eval pack comparing against published LIBERO
numbers needs the 120,000-step figure, not 30,000, and should point
`pretrained_checkpoint` at an actual VLA-JEPA pretrain checkpoint path
rather than leaving it unset.

## 4. Rebase probe

Performed in a throwaway branch (`rebase-probe`, created from `ur10e`),
deleted afterward; `ur10e` was neither rebased nor merged.

```
git checkout -b rebase-probe
git rebase upstream/main
  -> Successfully rebased and updated refs/heads/rebase-probe.
     (all 22 commits applied, 0 conflicts)
git checkout ur10e
git branch -D rebase-probe
  -> Deleted branch rebase-probe (was 7e360a7).
```

**Zero conflicts.** All of this fork's commits (TIP-001 through this
pack) applied cleanly on top of `upstream/main`, including
`data_config.py` and `mixtures.py`: our additions and upstream's edit
(the `LeRobotDroidDataConfig.video_keys` rename, an entirely different
part of the file) landed in non-overlapping hunks, so git's patch
application needed no manual resolution. Verified post-rebase that
`UR10eCupDataConfig` and its two registration lines were still present
and unchanged in the probe branch before deleting it. `ur10e` branch
confirmed afterward to still have its own independent history
(`upstream/main` is not an ancestor of `ur10e`'s `HEAD`).

## 5. Recommendation

1. **Line endings:** fixed and verified end-to-end in this pack; no
   further action needed, but keep the warning in `line_endings.md` in
   mind for any future file-writing on this machine.
2. **min_max granularity:** no action needed -- the Blueprint's original
   description of `wrist_2` was correct; the Contractor may want to add
   the noise-amplification observation (full `[-1,1]` stretch of a
   0.0014 rad span) to the final report's Limitations section as a
   distinct, smaller caveat from "crushed to zero."
3. **Upstream commit 0dd5281:** worth pulling in before any LIBERO
   training/eval work, specifically for the `max_train_steps`/
   `num_warmup_steps`/`pretrained_checkpoint` correction and the
   multi-view embedding reshape fix in `VLA_JEPA.py` -- the Contractor
   should decide whether that happens via a full rebase now (0 conflicts
   measured) or a narrower cherry-pick once UR10e training work begins.
4. **Rebase:** technically free right now (0 conflicts), but the decision
   of when to take it is the Contractor's, not this pack's.
