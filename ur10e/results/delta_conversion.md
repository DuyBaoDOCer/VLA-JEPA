# Delta Conversion Report

## 1. What was transformed

For the 6 joint dimensions of `action` (indices 0-5: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3), within each episode (never across an episode boundary):

```
action_new[t, 0:6] = action[t, 0:6] - state[t, 0:6]
action_new[t, 0:6] = (action_new[t, 0:6] + pi) % (2*pi) - pi   # angle-wrap safety net
```

The subtraction is done in float64 and only cast down to the source column's dtype (`float32`) when writing to disk, to avoid losing digits at the intermediate step.

## 2. What was NOT transformed

- `action[t, 6]` (gripper): copied through unchanged. No delta, no polarity flip.
- `state[t, :]` (all 7 dimensions, including the gripper channel): copied through unchanged.
- `videos/`: not copied at all. The next (layout) pack reads video directly from `ur10e/data/v30/videos/`; copying 1.14 GB of unchanged video would be pure waste.
- Every non-state/action column (`episode_index`, `frame_index`, `timestamp`, `index`, `task_index`): copied through unchanged, same row order, same values.

## 3. Gripper polarity decision -- NOT flipped

The Blueprint and Task Graph both specified a polarity flip (`g_new = 1 - g_old`) to match the author's pipeline convention (`processor_vla_jepa.py` docstring: `0 = open, 1 = close`). The Homeowner decided to drop that step for this pack. Reasoning:

Blueprint section 3.6 verified against the source that `action_model.state_encoder`, `action_model.action_encoder` and `action_model.action_decoder` are **not** loaded from the pretrained checkpoint -- they are initialized fresh, because this project's `state_dim`/`action_dim` (7) differs from the checkpoint's (8). The entire input and output path for the gripper dimension is therefore learned from scratch on this project's data. The model will learn whatever convention the data uses; flipping buys nothing.

Flipping would instead be actively harmful: it would produce a v2.1 dataset whose gripper convention silently disagrees with the source `khanhnd61/ur10e-cup` dataset, with no marker anywhere that this happened. That dataset is pushed to HuggingFace in a later pack for others to use -- a silent landmine.

Decision: keep `1 = open`, `0 = closed` exactly as measured from images in TIP-003. The gripper column is not touched by this script in any way.

## 4. Verification results

V1 is algebraically tautological (`state + (action - state) == action` by construction) and is kept only to catch implementation bugs (row misalignment, dtype truncation, reordering) -- it proves nothing about the semantics of the transform. V2 through V5 are the checks that actually test something, because each takes an independent path to the same quantity.

**V1 -- round-trip (tautological).** max error = 0. PASS (threshold 1e-6).

**V2 -- cross-derivation via state[t+1] (not tautological).** max error = 0. PASS (threshold 1e-9). This independently confirms both the transform code and the action[t] == state[t+1] identity measured in the previous pack.

**V3 -- per-episode telescoping sum (not tautological).** max error across all 81 episodes = 0. PASS (threshold 1e-5).

**V4 -- final frame of each episode is exactly zero.** 5771 frames have action_new[:,0:6] identically zero (predicted: exactly 81). All 81 true episode-final rows are among them (True), but there are 5690 additional zero-delta frames elsewhere in episodes. FAIL (the exact-count prediction from the previous pack does not hold).

Investigation: the extra zero-delta frames are not a bug. Episode 0 alone has ~170 consecutive zero-delta frames at the very *start* -- the robot holds a fixed pose (state and gripper constant) before motion begins, which is a second natural source of `action[t] == state[t]` beyond the episode-final hold the previous pack predicted. The previous pack's TIP-003 measurements never counted zero-delta frames directly (only `max|d|` and R9 wrap violations), so this undercount was never caught until this pack's V4 check ran. The core claim -- that the final frame of every episode has `action_new[:,0:6] == 0` -- is still true for all 81 episodes; the claim that *only* those 81 frames are zero is false.

**V5 -- angle-wrap step is a no-op.** 0 values altered (expected 0). PASS. Consistent with the previous pack's measurement of max|d| = 0.0111 rad, 283x below the pi threshold.

**V6 -- statistics agree with data_check.md.** PASS (6 joints agree to >=6 significant digits).

Note on scope: only the 6 JOINT dimensions are compared against data_check.md's delta table, because those are the only dimensions this pack actually converts to a delta representation. The gripper column of `action_new` is an unchanged pass-through of the absolute gripper value (section 4.1 / section 3), whereas data_check.md's "gripper" row under its delta table is `action[t] - state[t]` for gripper -- a diagnostic quantity from the previous pack's R12 outlier analysis, not the quantity this pack produces. Comparing `action_new`'s gripper column against that row would compare two different quantities by construction, so it is excluded from the pass/fail here and reported on its own merits in section 5 instead.

**V7 -- gripper untouched.** action frequencies={0.0: 24368, 1.0: 25411}, state frequencies={0.0: 24368, 1.0: 25411}, action_new matches source action element-wise: True. PASS.

**V8 -- state joints untouched.** PASS.

**V9 -- metadata preserved.** rows=49779 (expected 49779), episodes=81 (expected 81), id columns match row-by-row: True, column list/order identical: True. PASS.

## 5. Statistics after conversion

`action_new`, all 7 dimensions (delta representation, gripper pass-through):

| dim | min | q01 | q50 | mean | std | q99 | max | ratio_hi | ratio_lo | gap_hi | gap_lo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| shoulder_pan | -0.005999922752 | -0.002799987793 | 0 | 0.0008183751254 | 0.001700382978 | 0.00569999218 | 0.00620007515 | 0.9193424341 | 0.4666706404 | 0.05883342917 | 0.3764638234 |
| shoulder_lift | -0.008600115776 | -0.005799889565 | 0 | -0.0005501999737 | 0.001682503404 | 0.001900076866 | 0.003099918365 | 0.6129441624 | 0.6743966844 | 0.1558242507 | 0.3636673272 |
| elbow | -0.009100079536 | -0.006100058556 | 0 | 0.0002081619951 | 0.002987251474 | 0.00720000267 | 0.01110005379 | 0.6486457461 | 0.6703302461 | 0.2932355762 | 0.2255644489 |
| wrist_1 | -0.008900046349 | -0.005800008774 | 0 | 0.0003369472134 | 0.002933242171 | 0.0082000494 | 0.009199976921 | 0.8913119534 | 0.6516829853 | 0.07142309756 | 0.2214303352 |
| wrist_2 | -0.0002000331879 | -0.0001000165939 | 0 | 9.824618547e-07 | 3.852031082e-05 | 0.0001000165939 | 0.0002000331879 | 0.5 | 0.5 | 0.5 | 0.5 |
| wrist_3 | -0.005900144577 | -0.002799987793 | 0 | 0.0008202613977 | 0.001700734486 | 0.005699872971 | 0.00620007515 | 0.9193232071 | 0.4745625732 | 0.05884827915 | 0.3647303021 |
| gripper | 0 | 0 | 1 | 0.5104763053 | 0.499890235 | 1 | 1 | 1 | nan | 0 | 0 |

Cross-checked against the delta-representation table in `ur10e/results/data_check.md` (section 5 of that report, produced by an independently written script in TIP-003): the 6 joint dimensions agree to at least 6 significant digits (see V6 above for why gripper is excluded).

## 6. Known staleness

`meta/stats.json` is copied byte-for-byte from the source and describes the **pre-conversion** data -- specifically, it still holds absolute-representation statistics for `action`, not the new delta values. This is left as-is deliberately: the gr00t loader recomputes dataset statistics directly from the parquet (`calculate_dataset_statistics`) rather than trusting this file, and the next (layout-conversion) pack rebuilds `meta/` from scratch. Editing `stats.json` by hand here risks a format mistake that is worse than leaving it stale. **Do not trust `meta/stats.json` in `v30_delta/` for anything.**

## 7. Output layout

```
ur10e\data\v30_delta/
├── data/chunk-000/file-000.parquet     <- transformed (action delta, gripper/state unchanged)
└── meta/                                <- copied byte-for-byte from v30/
    ├── info.json
    ├── stats.json                       <- STALE, see section 6
    ├── tasks.parquet
    └── episodes/chunk-000/file-000.parquet
```

No `videos/` directory: video content is unaffected by this conversion, so the next pack reads video directly from `ur10e\data\v30/videos/` instead of duplicating 1.14 GB on disk.
