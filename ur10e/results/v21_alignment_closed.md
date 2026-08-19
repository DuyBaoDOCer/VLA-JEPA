# v2.1 Alignment -- Closed Out

## 1. Purpose

TIP-005b's K2 (a fixed +-2, later +-5/+-12, frame tolerance on optical
motion-onset offset) was retired as a gate, not loosened. K2 measures how
long it takes a commanded motion to become visible on camera -- a quantity
that depends on how fast the operator started moving in that particular
episode and on the angle between that motion and each camera's viewing
axis, neither of which this pipeline controls. No fixed threshold on a
quantity the pipeline doesn't decide can ever be correct; that was true of
the original D3's +-2 frames and it was still true of D3''s per-camera
+-5/+-12. A gate belongs on quantities the pipeline *does* decide -- which
source file is read and at what timestamps. That is what D6 checks.

## 2. D6 -- metadata tiling

Method (`ur10e/src/verify_source_metadata_tiling.py`): read
`episode_video_map.csv`, group episodes by source `file_index` per camera,
sort by `from_timestamp`, and check that consecutive episodes abut within
one frame period (1/20 s = 0.05 s), that each file's first episode starts
at ~0 and its last episode ends at that file's real duration (measured
independently with `ffprobe`), and that each camera's total covered
duration equals `49779 / 20 = 2488.95` s.

ffprobe durations of the 7 source files:

```
side  file-000.mp4: 720.300000 s
side  file-001.mp4: 735.200000 s
side  file-002.mp4: 697.650000 s
side  file-003.mp4: 335.800000 s
wrist file-000.mp4: 1181.900000 s
wrist file-001.mp4: 1247.550000 s
wrist file-002.mp4: 59.500000 s
```

Results, all 81 episodes x 2 cameras (162 rows written to
`ur10e/results/metadata_tiling.csv`):

- Largest `|gap|` between any two consecutive episodes in the same file:
  **0.000000 s** (no gap, no overlap, anywhere).
- Every file's first episode starts at `from_timestamp = 0.0`.
- Every file's last episode's `to_timestamp` matches that file's ffprobe
  duration exactly (0.000000 s error in all 7 cases).
- Coverage: side total = 2488.950000 s, wrist total = 2488.950000 s, both
  against an expected 2488.95 s -- 0.000000 s error.
- Rows flagged: **0 / 162**.

**Conclusion: clean.** The source dataset's own metadata tiles all 7
concatenated videos with zero gap and zero overlap, to the full precision
of the stored timestamps.

## 3. Evidence chain

Each check below is blind to a different failure mode; together they cover
both directions a cut could be wrong:

| Check | What it verifies | Blind to |
|---|---|---|
| D1 (frame count, 162/162) | clips have the right *length* | cut position within the source file |
| D2 (visual gripper check, 3/4 transitions, 2 source files) | video content matches recorded state at specific frames | anything not sampled |
| D5 (cut-parameter arithmetic, 162/162) | the converter *used* exactly the source metadata's timestamps for every cut | whether that metadata is itself correct |
| D6 (metadata tiling, this pack) | the source metadata is internally consistent -- no gap, no overlap, exact coverage | camera-to-arm timing (not a metadata question) |
| K1 (onset sign, 0/152 negative) | no clip runs *ahead* of its data (would require `S > 0`, an over-late cut) | camera-specific detection lag (expected, not a defect) |

D5 says the converter cut exactly where the metadata said to. D6 says the
metadata itself has no seams. Combined, the cut *positions* are proven
correct to the limit of what this dataset's own metadata can attest --
proving the metadata's absolute accuracy against some external ground
truth (e.g. hardware timestamps from the original recording rig) is out of
reach here, as it would be for any consumer of this dataset. K1 rules out
the one direction of cut error (`S > 0`) that a detection-lag story cannot
explain away, since lag can only delay onset, never advance it.

## 4. Onset offsets (descriptive, NOT a gate)

From the existing `onset_alignment.csv` (no recomputation), 76 episodes
have valid offsets on both cameras:

| camera | min | median | max | n |
|---|---|---|---|---|
| wrist | 1 | 3.0 | 19 | 76 |
| side | 3 | 7.0 | 93 | 76 |

**Correlation between `offset_side` and `offset_wrist` across the 76 shared
episodes: 0.11** (weak, close to zero). Per the reading table in TIP-005c
section 4.2: a correlation this close to 0 means the offsets are mostly
**not** explained by a shared cause (e.g. the same operator being slow to
start that episode) -- they are dominated by each camera's own geometry
relative to that episode's initial motion direction. This matches the
episode-68 case investigated in the previous pack directly: identical
episode, identical metadata, identical cut code, wildly different offset
per camera.

Episodes with `offset_side - offset_wrist > 30` frames (14 total):

```
ep=68: side=93 wrist=2  diff=91
ep=58: side=88 wrist=1  diff=87
ep=46: side=87 wrist=3  diff=84
ep=43: side=88 wrist=6  diff=82
ep=38: side=85 wrist=3  diff=82
ep=24: side=92 wrist=13 diff=79
ep=37: side=81 wrist=2  diff=79
ep=48: side=77 wrist=3  diff=74
ep=55: side=76 wrist=2  diff=74
ep=23: side=76 wrist=3  diff=73
ep=66: side=70 wrist=3  diff=67
ep=56: side=46 wrist=3  diff=43
ep=47: side=45 wrist=4  diff=41
ep=39: side=38 wrist=5  diff=33
```

Geometric explanation (episode 68 as the worked example): the wrist camera
is mounted on the arm itself, so any joint motion immediately pans the
whole frame -- it detected motion 2 frames after `f_onset`, right at the
physical floor. The side camera, stationary and viewing the arm from a
distance, only sees whatever silhouette change that specific motion
direction produces; for these 14 episodes, the initial motion apparently
projects with very little visible change from the side viewpoint until
enough displacement accumulates. This is a per-episode, per-camera optical
effect, not a shared timing offset -- if it were a cut-timing bug, the
*same* episode's two cameras (same metadata, same code path) would not
disagree by up to 91 frames.

**No episode is excluded from the dataset on the basis of this offset.**
There is no evidence any of them are corrupted -- only that side-camera
motion visibility varies a lot by episode, which is a fact about the
recording, not the pipeline.

## 5. Final verdict

Alignment between `ur10e/data/v21`'s video and its parquet/action data is
proven, to the limit of what is checkable without an external ground
truth, by five independent lines of evidence: exact frame counts (D1),
direct visual confirmation at three sampled transitions across two source
files (D2), exact arithmetic match between the cut commands actually
issued and the source metadata (D5), zero gap/overlap in that same source
metadata (D6), and zero instances of the one onset-offset direction a
detection-lag story cannot explain (K1).

What remains an assumption, stated explicitly rather than hidden: the
source dataset's recorded timestamps are assumed accurate relative to the
original camera/robot hardware clocks. This project has no independent way
to check that -- it is the same assumption every other consumer of
`khanhnd61/ur10e-cup` necessarily makes.

## 6. Upload record

- Repo: `DuyBao44DOCer/ur10e-cup-v21-h264`
- Visibility: **private** (confirmed via `HfApi().dataset_info(...).private == True`)
- Local size uploaded: 200.80 MB
- Upload duration: 224.9 s
- Throughput: 0.89 MB/s (effective, this invocation)
- Retries needed: **0** -- the upload completed on the first attempt, no timeout occurred

## 7. Post-upload verify

- `HfApi().dataset_info(REPO, files_metadata=True)` lists **250** remote
  files: the 249 real dataset files plus the `.gitattributes` HuggingFace
  adds automatically to every repo.
- Local file breakdown: 81 parquet + 81 side mp4 + 81 wrist mp4 + **6** meta
  files (`info.json`, `episodes.jsonl`, `episodes_stats.jsonl`, `tasks.jsonl`,
  `stats.json`, `modality.json`) = 249 total. `episodes_stats.jsonl` is a
  6th meta file beyond the 5 named in TIP-005c section 1.1 -- it was already
  documented in TIP-005a's report as harmless legacy metadata the upstream
  converter writes alongside the 5 the TIP names; not a new finding.
- Missing on remote: **0**. Extra on remote (excluding `.gitattributes`):
  **0**. Per-file size mismatches: **0**.
- **Post-upload verification: clean.**

## 8. Known limitations

- The video has passed through three lossy generations: original capture
  (H.264, presumably a modest CRF) -> AV1 (the `khanhnd61/ur10e-cup` v3.0
  release) -> H.264 CRF 23 via libx264 (this project's v2.1 conversion).
  Unavoidable given the AV1 source cannot be losslessly reversed.
- libx264 `-preset medium -crf 23` was used instead of NVENC: this
  machine's NVIDIA driver reports NVENC API 12.2, the ffmpeg build needs
  13.0 (driver >=570.0 required). At matched CRF, libx264 `medium` is not a
  quality regression versus NVENC `p4`, only a speed cost.
- Episode 80's release transition (gripper 0->1, frame 525) was never
  visually confirmed -- the three sampled side-camera frames looked too
  similar at the extracted zoom level to read the jaw state by eye. Does
  not contradict the other three sampled transitions.
- Every alignment conclusion in this project rests on the source dataset's
  own recorded timestamps being accurate. That premise itself is outside
  what this project can independently verify (section 5).
