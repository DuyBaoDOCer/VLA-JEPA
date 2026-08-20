# Rebase onto upstream/main — TIP-007c

## 1. Rebase record

| Item | Value |
|---|---|
| `ur10e` before rebase | `0a4d41b44c0a1dca286380c5c845b41b2f115043` |
| `ur10e` after rebase | `d9fa2283d64d63b7add193e78606b03e8fceffb4` |
| `upstream/main` | `0dd5281951046b17e1e3653f5661a406306a4a03` |
| Commits replayed | 22 |
| Conflicts | 0 |

`git rebase upstream/main` completed with "Successfully rebased and updated
refs/heads/ur10e." No manual conflict resolution was needed; the prior
rebase probe's 0-conflict measurement held.

## 2. Multi-view bug

**Old code** (`VLA_JEPA.py`, pre-fix):

```python
batch_videos = batch_videos.reshape(B * V, T, C, H, W)      # from [B, V, ...]
...
video_embeddings = torch.cat(torch.chunk(video_embeddings, chunks=V, dim=0), dim=2)
```

The initial `reshape` from `[B, V, ...]` to `[B*V, ...]` is B-major
(`i = b*V + v`). `torch.chunk(x, chunks=V, dim=0)` then slices the leading
dimension into `V` contiguous blocks of size `B` each — a V-major
assumption. For `B, V > 1` this pairs a batch element's view with a
*different* batch element's view of the same camera index.

**Fix** (post-`0dd5281`):

```python
x = video_embeddings.reshape(B, V, T_tok, P, D)
x = x.permute(0, 2, 3, 1, 4)
x = x.reshape(B, T_tok * P, V * D)
```

This reverses the original B-major collapse directly, so `(b, v)` pairing is
always correct regardless of `B` and `V`.

**`verify_multiview_fix.py` result** (exit code 0, all four assertions pass):

```
=== B=2, V=2 ===
old[batch=0] distinct values: [0.0, 10.0]
old[batch=1] distinct values: [1.0, 11.0]
new[batch=0] distinct values: [0.0, 1.0]
new[batch=1] distinct values: [10.0, 11.0]
old != new: True
new rows contain only their own batch's values: True
old row 0 contains a b=1 value (bug present): True

=== B=1, V=2 ===
old[batch=0] distinct values: [0.0, 1.0]
new[batch=0] distinct values: [0.0, 1.0]
old == new at B=1: True
```

At `B=1, V=2` the two expressions agree because chunking `V` contiguous
blocks of size `B=1` along dim 0 happens to align with each block being
exactly one view of that same single batch element — the B-major/V-major
mismatch only produces wrong pairing when `B > 1`. This is why the bug never
raised an error and loss still decreased: any single-example run (or any run
with only one camera view) never exercises the broken branch.

We use `V=2` (side + wrist cameras) and plan to train with batch size > 1,
so this fix is directly relevant to correctness of our finetuning run.

## 3. Our contributions

Confirmed intact after rebase via `git diff upstream/main --stat` and
targeted diffs:

- `class UR10eCupDataConfig` — present and unchanged in
  `starVLA/dataloader/gr00t_lerobot/data_config.py`.
- `ROBOT_TYPE_CONFIG_MAP["ur10e"] = UR10eCupDataConfig` — still registered.
- `DATASET_NAMED_MIXTURES["ur10e_cup"]` — still registered in
  `starVLA/dataloader/gr00t_lerobot/mixtures.py`.
- No `target_rotations` entry anywhere inside `UR10eCupDataConfig` (the only
  `target_rotations` occurrences in `data_config.py` are lines 115, 126, 620,
  all inside unrelated, pre-existing configs).
- `"wm_key_indices": [0, 1]` unchanged inside `modality_config()`.

`git diff upstream/main --stat` shows changes limited to
`starVLA/dataloader/gr00t_lerobot/data_config.py`,
`starVLA/dataloader/gr00t_lerobot/mixtures.py`, and files under `ur10e/`.
No other files diverge from upstream.

## 4. Mechanisms retained

All four mechanisms our training recipe depends on are present in the
rebased tree:

- `build_param_lr_groups` — `starVLA/training/trainer_utils/trainer_tools.py:51`.
- `load_pretrained_backbones(model, checkpoint_path=None, reload_modules=None)`
  — `starVLA/training/trainer_utils/trainer_tools.py:212`, called with
  `reload_modules=reload_modules` from `starVLA/training/train_starvla.py:165`.
- `freeze_modules` — read from `self.config.trainer.freeze_modules` at
  `starVLA/training/train_starvla.py:168-170`.
- `dataset_py: "lerobot_datasets"` dispatch — still handled at
  `starVLA/dataloader/__init__.py:40-41`.

## 5. Training recipe drift (recorded only — not applied)

`scripts/config/vlajepa_robot_ft.yaml` was deleted by upstream (confirmed
via `git log --all --oneline -- scripts/config/vlajepa_robot_ft.yaml`, last
touched at `0dd5281`). It is replaced by `scripts/configs/vlajepa_libero_ft.yaml`,
which is LIBERO-only and now carries **four** learning-rate groups instead
of three:

```yaml
learning_rate:
  base:              3.0e-05
  qwen_vl_interface: 1.0e-05
  action_model:      1.0e-04
  vj_predictor:      5.0e-04      # new group
```

**Decision D22 (homeowner):** UR10e finetuning keeps **three** groups.
`vj_predictor` stays folded into `base` at `3e-5`, matching the old
three-group robot recipe. The `5e-4` rate is not adopted.

Rationale (recorded for context, not for debate): `5e-4` was tuned on
LIBERO — a simulated benchmark with ~2K demonstrations. Our UR10e dataset is
81 real-robot episodes. The three-group recipe is the one the original
authors used for real-robot finetuning.

Other LIBERO-specific numbers that changed and are **not** carried over to
UR10e:

| Field | Old (robot ft) | New (LIBERO ft) |
|---|---|---|
| `max_train_steps` | 30000 | 120000 |
| `num_warmup_steps` | 5000 | 20000 |
| `min_lr` | 1.0e-06 | 1.0e-05 |

No `ur10e_ft.yaml` was created or modified in this pack. That file — with
three LR groups and the LIBERO-drifted step/warmup/min_lr numbers evaluated
on their own merits — is Colab-pack work.

## 6. Video key convention (investigation only — no config changes made)

Upstream commit `0dd5281` renamed `LeRobotDroidDataConfig`'s video keys from
`observation.images.exterior_image_1_left` to `video.exterior_image_1`.
`Libero4in1DataConfig` already used `video.primary_image` / `video.wrist_image`.
`UR10eCupDataConfig` still uses `observation.images.side` /
`observation.images.wrist` — the only config left on the old convention.

**1. What does `meta/modality.json` declare for video?**

Verbatim `video` section of `ur10e/data/v21/meta/modality.json`:

```json
"video": {
    "side": {
        "original_key": "observation.images.side"
    },
    "wrist": {
        "original_key": "observation.images.wrist"
    }
}
```

The keys inside the `video` object are the *short* names (`side`, `wrist`),
each with an `original_key` field that points at the physical folder name
(`observation.images.side` / `observation.images.wrist`). `modality.json` is
exactly the mapping layer between a short logical name and the on-disk key.

**2. How does the loader resolve a video key to a path?**

- `starVLA/dataloader/gr00t_lerobot/datasets.py:632-634`
  (`_get_modality_keys`): `modality_keys[modality] = config.modality_keys`
  — the loader's `self.modality_keys["video"]` list is taken **verbatim**
  from the data config's `video_keys` (i.e. from `UR10eCupDataConfig.video_keys`
  as written today), not from `modality.json`.
- `starVLA/dataloader/gr00t_lerobot/datasets.py:907`
  (`get_video`): `assert key.startswith("video."), f"Video key must start
  with 'video.', got {key}"` — a hard runtime assertion.
- `starVLA/dataloader/gr00t_lerobot/datasets.py:909`: `key =
  key.replace("video.", "")` strips the prefix.
- `starVLA/dataloader/gr00t_lerobot/datasets.py:873`
  (`get_video_path`): `original_key =
  self.lerobot_modality_meta.video[key].original_key` looks the stripped
  key up in `modality.json`'s `video` dict to get the physical folder name,
  then `starVLA/dataloader/gr00t_lerobot/datasets.py:876-877` formats
  `video_path_pattern` with `video_key=original_key`.

**3. Conclusion**

`video_keys` in `UR10eCupDataConfig` must use the new `video.*` convention
(e.g. `video.side`, `video.wrist`), not `observation.images.*`. This is not
just a style preference: `datasets.py:907` asserts the key must start with
`"video."`, so a config using `observation.images.side` verbatim will crash
that assertion the first time `get_video` is called. `modality.json`'s
`original_key` field is the layer that maps the short `video.*` name back to
the physical `observation.images.*` folder — the physical folder name does
not need to appear in the data config at all.

`UR10eCupDataConfig` was **not** modified in this pack. This finding is
handed to the Colab pack, where the loader can actually be run and the fix
verified against a real load.

## 7. Line endings

`git ls-files --eol | Where-Object { $_ -match "w/crlf" } | Measure-Object`
→ **0** files after rebase.

## 8. Push record

`ur10e` had never been pushed to `origin` before this pack (`git ls-remote
origin ur10e` returned nothing before the push; only `origin/main` existed
as a remote branch). This was therefore a **first push**:
`git push -u origin ur10e`, not a force-with-lease.
