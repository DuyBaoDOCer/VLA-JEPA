# Split, Baselines, and Compression Quality (TIP-006)

## 1. Split definition

`ur10e/data/v21` (81 episodes, 49,779 frames) is split deterministically:

| Set | `episode_index` range | Episodes | Frames |
|---|---|---|---|
| Train | 0..72 | 73 | 44,865 |
| Held-out | 73..80 | 8 | 4,914 |

No shuffling. The last 8 episodes are held out. Held-out files keep their
original indices (`episode_000073.parquet` ... `episode_000080.parquet`),
not renumbered — `chunks_size: 1000` in `info.json` means every index in
this range still resolves to `chunk-000`, so the file paths build correctly
without renumbering. Whether the `gr00t` loader accepts a dataset whose
`episode_index` does not start at 0 is untested; if it does not, renumbering
is a cheap follow-up. This split was decided before this pack ran and is
not reconsidered here regardless of what section 3 below shows.

Two standalone LeRobot v2.1 datasets were physically written to
`ur10e/data/v21_train/` and `ur10e/data/v21_heldout/`, each with its own
`data/chunk-000/`, `videos/chunk-000/observation.images.{side,wrist}/`, and
`meta/`. `ur10e/data/v21/` itself was only ever opened for reading.

`ur10e/results/split.json` records the episode-index lists and frame counts
for both sets.

## 2. Meta regeneration

| File | Train / held-out | Why |
|---|---|---|
| `stats.json` | **recomputed** from that split's own copied parquet + video files | see below |
| `episodes.jsonl` | filtered from `v21`'s file, by `episode_index` | subset, no other change |
| `episodes_stats.jsonl` | filtered from `v21`'s file, by `episode_index` | subset, no other change |
| `info.json` | copied, then `total_episodes`/`total_frames`/`total_chunks`/`total_videos`/`splits` updated | `data_path`, `video_path`, `chunks_size`, `fps`, `features` kept byte-identical to `v21` |
| `tasks.jsonl` | byte-identical copy | single task, nothing to filter |
| `modality.json` | byte-identical copy | schema, not data |

**Why `stats.json` must be recomputed, not copied:** it holds the
min/max/mean/std used to normalize inputs at train time. If the train split
carried statistics computed over all 81 episodes, normalization would have
already seen the held-out episodes' values before a single held-out MSE was
measured — a leak that produces no error and no visibly wrong number, but
invalidates every held-out MSE the eval pack reports. Physically separating
the files and recomputing `stats.json` from each split's own files means
the computation (`ur10e/src/split_dataset.py:compute_split_stats`) can only
see what is actually in that folder.

Numeric features (`observation.state`, `action`, `timestamp`,
`frame_index`, `episode_index`, `index`, `task_index`) are computed exactly
over the full population of that split's parquet rows. Image features
(`observation.images.side`, `observation.images.wrist`) have their
mean/std/min/max accumulated from every 6th decoded frame (`IMAGE_FRAME_STRIDE`
in `split_dataset.py`) of every video file in that split — every frame is
decoded (cheap, ~500 fps with PyAV), but only every 6th is folded into the
running sum/sum-of-squares/min/max, which cut the wall-clock cost of this
step roughly 4x with no visible change in the resulting means; `q01`/`q10`/
`q50`/`q90`/`q99` are estimated from a fixed-seed (42), 200-pixel-per-frame
spatial subsample of those same accumulated frames — holding the full pixel
population in memory for an exact quantile is not feasible. This is a
different sampling method from whatever produced the original
`v21/meta/stats.json` (which was carried over from the v3.0 dataset's own
stats, not recomputed during the v2.1 conversion — see
`ur10e/src/to_v21.py:copy_global_stats`), so absolute values are not
expected to match `v21`'s file; what matters is that train and held-out
differ from each other, proving each was computed from its own data only.

**Evidence the recomputation actually happened (leakage check):** every
value checked differs across `v21` / `v21_train` / `v21_heldout`, because
each was computed from a genuinely different set of files.

| Stat | `v21` (81 ep) | `v21_train` (73 ep) | `v21_heldout` (8 ep) |
|---|---|---|---|
| `observation.state` mean, dim 0 (shoulder_pan) | -1.406734 | -1.407916 | -1.395951 |
| `action` std, dim 6 (gripper) | 0.499890 | 0.499896 | 0.499825 |
| `observation.images.side` mean, R channel | 0.388377 | 0.385009 | 0.394860 |
| `observation.images.side` pixel `count` | 955,756,800 | 2,296,934,400 | 251,596,800 |

The image-channel `count` values are not expected to match each other, or
`v21`'s: `v21`'s `stats.json` was inherited from the v3.0 dataset's own
stats via `to_v21.py:copy_global_stats` (a different, unknown sampling
depth), while `v21_train`/`v21_heldout` here use full-population
accumulation over every `IMAGE_FRAME_STRIDE`-th (6th) decoded frame of
that split's own video files only (`split_dataset.py:compute_image_stats`).
What matters for the leakage argument is that train and held-out differ
from each other and from `v21`, proving neither saw the other's data.

## 3. Held-out representativeness

| Metric | Train (73 ep) | Held-out (8 ep) |
|---|---|---|
| Mean episode length (frames) | 614.59 | 614.25 |
| Stationary-frame ratio (delta=0, all 6 joints) | 11.78% | 9.91% |
| RMS \|d\| over 6 joints | 0.002156 | 0.002177 |
| Gripper open ratio (1 = open) | 51.02% | 51.32% |
| Gripper closed ratio | 48.98% | 48.68% |

No large discrepancy. Held-out episodes look like a normal sample of the
whole set on every axis checked. The split itself was not changed in
response to this table — the last-8 rule was fixed before this comparison
was run either way. Source: `ur10e/results/representativeness.json`.

## 4. Baselines

Two copy baselines, computed on the exact set the eval pack will report
against (`heldout_8`), plus `train_73` and `all_81` for reference:

- **A** — predict no motion: `a_hat[t] = 0`, `MSE = mean(d[t]**2)`
- **B** — repeat previous delta: `a_hat[t] = d[t-1]`, per episode, dropping
  each episode's first frame (no `t-1` to repeat there)

"moving" frames are rows where the 6-dim joint delta actually being
predicted is nonzero in at least one dim. 11.59% of all-81 frames are
perfectly stationary (delta = 0 in all 6 joints); baseline A gets those for
free, so a model that only learned "hold still, output zero" would look
better than it is if only the all-frames column is read.

### 6 joints (radians, delta representation)

| Set | Frame type | Baseline A MSE | Baseline B MSE | n (A) | n (B) |
|---|---|---|---|---|---|
| all_81 | all | 4.6576e-06 | 5.8417e-07 | 49,779 | 49,698 |
| all_81 | moving | 5.2684e-06 | 6.5790e-07 | 44,008 | 44,005 |
| train_73 | all | 4.6486e-06 | 5.7178e-07 | 44,865 | 44,792 |
| train_73 | moving | 5.2692e-06 | 6.4508e-07 | 39,581 | 39,578 |
| **heldout_8** | all | 4.7401e-06 | 6.9735e-07 | 4,914 | 4,906 |
| **heldout_8** | **moving** | 5.2615e-06 | **7.7255e-07** | 4,427 | 4,427 |

### Gripper (binary, absolute)

| Set | Frame type | Baseline A MSE | Baseline B MSE |
|---|---|---|---|
| all_81 | all | 0.510476 | 0.003260 |
| all_81 | moving | 0.483208 | 0.002318 |
| train_73 | all | 0.510175 | 0.003260 |
| train_73 | moving | 0.482428 | 0.002299 |
| heldout_8 | all | 0.513228 | 0.003261 |
| heldout_8 | moving | 0.490174 | 0.002485 |

Frames skipped at episode boundaries (baseline B, one per episode, both
frame types): all_81 = 81, train_73 = 73, heldout_8 = 8.

**Reference check:** the all-81 / all-frames / joints cell reproduces
`4.6576e-06` (baseline A) and `5.8417e-07` (baseline B) against the
pre-existing reference values `4.658e-06` / `5.842e-07` — match to the
given precision.

## 5. THE NUMBER TO BEAT

**🎯 Baseline B, held-out 8 episodes, moving frames, 6 joints: `7.7255e-07`**

(source: `ur10e/results/baselines.json`, key `number_to_beat`)

## 6. Compression quality

3 episodes (0, 40, 80 — small/middle/large index) x 2 cameras, ~50 frames
each, PSNR/SSIM between the stored `v21` H.264 clip and the matching
segment of the original `v30` AV1 source (same start timestamp, from
`episode_video_map.csv`, frame-accurate `-ss`-before-`-i` seek — same
seeking convention `to_v21.py` used to cut the clips in the first place).

| Resolution | Mean PSNR | Min PSNR | Mean SSIM | Min SSIM |
|---|---|---|---|---|
| 480x640 (stored) | 45.88 dB | 43.83 dB | 0.9867 | 0.9807 |
| **256x256 (model input, vjepa2-vitl-fpc64-256)** | **49.26 dB** | **46.93 dB** | **0.9935** | **0.9890** |

Full per-episode/per-camera breakdown in
`ur10e/results/compression_quality.json`.

**Conclusion:** both resolutions land in the **> 40 dB — near-indistinguishable**
band, and the 256x256 number (the one that actually matters, since that is
what `vjepa2-vitl-fpc64-256` sees) is higher than the native-resolution
number, confirming that most of the visible compression loss is erased by
the downscale. `crf 23` on largely-static, fixed-camera robot footage is
not costing anything worth worrying about. No video was re-encoded in this
pack regardless of this result — re-encoding, if ever warranted, is a
Contractor decision.

## 7. Upload record

| Repo | Private | Local size | Files | Effective throughput | Retries |
|---|---|---|---|---|---|
| `DuyBao44DOCer/ur10e-cup-v21-train73` | yes | 181.16 MB | 225 (73 parquet, 73 side mp4, 73 wrist mp4, 6 meta) | 23.61 MB/s | 0 |
| `DuyBao44DOCer/ur10e-cup-v21-heldout8` | yes | 19.65 MB | 30 (8 parquet, 8 side mp4, 8 wrist mp4, 6 meta) | 6.39 MB/s | 0 |

Both uploads completed in a single `upload_folder()` call each (7.7 s and
3.1 s respectively) — no timeout, no re-run needed. Post-upload
verification (`ur10e/src/verify_splits_upload.py`) confirms remote file
count, per-file size, and `private=True` match local exactly for both
repos.
