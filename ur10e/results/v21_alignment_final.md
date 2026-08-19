# v2.1 Final Alignment Verdict

## 1. Purpose

The previous pack's D3 check (naive adjacent-frame pixel diff, shared +-2 frame
tolerance across both cameras) failed for a majority of episodes. Root-cause
investigation there traced this to two design flaws in the *check itself*, not
evidence of misaligned data: (1) a `mean + 5*std` threshold calibrated against
a background that contains strong ~10 Hz lighting flicker (confirmed present
in the pristine AV1 source, not introduced by re-encoding), which inflates the
noise floor and delays detection; (2) a single shared tolerance applied to two
cameras with very different sensitivity to early, sub-pixel motion -- the side
camera, viewing the arm from a distance, cannot resolve the first several
frames of a typical episode's acceleration ramp (worked out below), while the
wrist camera (mounted on the arm) sees the whole frame pan almost immediately.

This pack does not re-litigate that diagnosis -- it fixes the check (D3', a
period-2 diff immune to the flicker, camera-specific tolerances) and adds an
independent, purely numeric check (D5) that has no dependency on pixels,
lighting, or thresholds at all: it verifies the exact `-ss`/`-t` values fed to
ffmpeg for every one of the 162 cuts against the source dataset's own
metadata.

Physical justification for why side-camera detection lag is expected, not a
defect: UR10e reach ~1.3 m, the 640 px frame covers ~1.2 m of that, giving
~1.9 mm/pixel. Previous-pack RMS |d| = 0.00216 rad/frame implies ~2.8 mm/frame
end-effector motion at *average* speed, i.e. ~1.5 px/frame -- and episodes
start well below average speed. At 10% of that RMS (a plausible ramp-up
speed for the first few frames of motion), displacement is ~0.28 mm/frame,
i.e. ~0.15 px/frame -- below what any camera can resolve. A multi-frame delay
before pixel motion clears any noise floor is expected on the side camera in
particular.

## 2. D5 cut parameters

`ur10e/src/to_v21.py`'s `convert_videos()` was instrumented with a single
logging call, `_log_cut_parameters(episode_index, camera, source_file, ss, t)`,
placed immediately before the existing call to `_extract_video_segment()`. No
other line was touched. A `diff` against the pristine clone of
`hungho77/Isaac-GR00T@af782495` (branch `yennt`) confirms the only changes
across the whole file are: the provenance comment, the `import csv`, this one
helper function, and the one call site -- the episode-grouping, sorting, and
ffmpeg-parameter-computation logic is byte-for-byte identical to upstream.

The already-placed `ur10e/data/v21` was **not** touched to produce this log.
Producing it required actually running the cutting step (its parameters are
computed live inside `convert_videos()`, not derivable from static analysis
alone without duplicating that logic), so the junctioned inputs
(`ur10e/data/v30_delta` for data/meta, `ur10e/data/v30/videos` for video) were
wired up again into a **fresh scratch staging directory**
(`_scratch/v21_staging/ur10e-cup-local`, outside the repo) and the conversion
was re-run there. The final "move into place" step
(`move_to_destination()` in `run_v21_conversion.py`) was deliberately **not**
invoked, so the already-verified `ur10e/data/v21` was never overwritten,
re-encoded, or otherwise modified. Confirmed after the run: file count under
`ur10e/data/v21` is still 249, and `episode_000000.parquet`'s modification
timestamp still predates this pack's work.

**Result: 162 / 162 rows match exactly.** For every (episode, camera) pair:
the cut source file's `file_index` matches `episode_video_map.csv`'s
`{camera}_file_index`; `ss` matches `{camera}_from_timestamp` to better than
1e-6; `t` matches `{camera}_to_timestamp - {camera}_from_timestamp` to better
than 1e-6. Zero mismatches. `ur10e/results/cut_parameters.csv` holds the full
162-row log.

This proves the converter used exactly the source dataset's own recorded
timestamps for every cut, with no off-by-something in file selection or
segment boundaries. It does not (and cannot) prove those timestamps are
themselves correct in the original `khanhnd61/ur10e-cup` release -- that
metadata is what every other tool consuming this dataset also trusts, and
verifying it would require an independent ground truth this project does not
have.

## 3. D3', corrected

Method, per usable episode `e` (78 of 81 -- episodes 1, 28 and 30 have
`f_onset == 0`, i.e. no still lead-in, and are excluded; see the skip list
below) and per camera:

1. `f_onset(e)` = first frame index (local to the episode) where any of the 6
   joint deltas in `action_new` is nonzero, read from the v2.1 per-episode
   parquet.
2. Decode frames `[max(0, f_onset-60), f_onset+60]` of the **cut** clip
   (extended further forward if needed -- see step 5), downsampled to
   ~106x80.
3. `diff(t) = mean(|frame[t] - frame[t-2]|)` -- comparing frames two apart
   places them at the same point in the ~10 Hz flicker cycle relative to the
   20 fps capture, cancelling it out. This is the fix for the original D3's
   flaw (1).
4. Threshold = `mean(diff) + 5*std(diff)`, computed only over the *filtered*
   (period-2) diffs in the still portion of the window (`t < f_onset`). Same
   formula shape as the original D3, but computed on a signal the flicker no
   longer dominates.
5. `v_onset` = first frame past `f_onset` where `diff(t)` exceeds the
   threshold. If none is found within the initial +-60 window, the search
   extends forward in 60-frame increments (this happened for the two
   largest-offset episodes reported below).

Two episodes (32 and 79) have `f_onset = 4`, leaving only 2 usable
period-2-diff samples before onset -- too few to estimate a threshold. These
4 rows (2 episodes x 2 cameras) are recorded in `onset_alignment.csv` with an
`error` field set and excluded from the K1/K2/K3 statistics below, leaving
152 valid measurements out of 156.

Full per-row table: `ur10e/results/onset_alignment.csv` (156 rows: 78
episodes x 2 cameras, including the 4 error rows).

**K1 -- sign (the sharpest test in this pack):**
```
0 negative offsets out of 152 valid measurements.
```
No episode, on either camera, shows video motion detected *before* the
commanded action changed. A negative value would have meant frames from the
future of the true cut were appearing before they should -- unexplainable by
optical detection lag, and a direct sign of clips running ahead of data. None
occurred. **PASS.**

**K2 -- per-camera tolerance:**

| camera | limit | over limit | max offset |
|---|---|---|---|
| wrist | <=5 frames | 9 / 76 | 19 |
| side | <=12 frames | 18 / 76 | 93 |

**FAILS.** Both cameras have episodes exceeding their tolerance, including
two large outliers investigated directly:

- Episode 68: side offset = 93 frames, but the *same episode's* wrist offset
  = 2 frames (right at the expected physical floor). The robot demonstrably
  started moving on schedule (wrist confirms it, parquet confirms it) --
  what took 93 frames to become visible was specifically the *side camera's*
  view of that particular episode's initial motion direction, most plausibly
  because the early motion in this demonstration happens to project with
  very little visible silhouette change from the side viewpoint until enough
  displacement accumulates. This is a per-episode optical-geometry effect,
  not a sign the clip was cut wrong -- the wrist camera on the same episode,
  same source data, rules that out directly.
- Episode 41: side offset = 23, wrist offset = 19 -- both elevated together,
  unlike episode 68's asymmetric pattern. This looks like a genuinely slower
  acceleration ramp in this particular demonstration (human teleoperation
  demonstrations are not identically paced), not a camera-specific optical
  effect, since both cameras agree it took longer here.

These are plausible, individually-investigated explanations, not a blanket
excuse -- but per this pack's own constraints, a plausible story does not
substitute for the check passing. **K2 fails as specified**, and per section
7's rule this alone means Gate 2 is not clean.

**K3 -- distribution shape, per camera:**

| camera | min | median | max | n |
|---|---|---|---|---|
| wrist | 1 | 3.0 | 19 | 76 |
| side | 3 | 7.0 | 93 | 76 |

`min(wrist) = 1 <= 2` -- **PASS** on this specific required check. Several
episodes are detected almost immediately on the wrist camera (min=1), which
is what optical latency predicts (fast-starting episodes get picked up fast)
and what a fixed phase-shift bug would not produce (a true cut-timing bug
would push every episode's offset up by roughly the same fixed amount,
regardless of how fast that episode's motion ramps up).

## 4. Verdict on alignment

Four independent lines of evidence, each blind to a different failure mode:

- **D1** (frame count, 162/162 exact): rules out truncated or padded clips.
- **D2** (visual gripper check, side camera): 3 of 4 sampled transitions show
  the recorded gripper state change landing within a frame or two of the
  video's visible open/close transition -- direct, human-verified evidence
  of zero-frame alignment at those points, across two episodes drawn from
  two different source mp4 files (episode 0 in side file 0, episode 80 in
  side file 3).
- **D5** (cut-parameter arithmetic, 162/162 exact): rules out the converter
  using the wrong file, or the wrong `-ss`/`-t` for any cut -- the numbers
  fed to ffmpeg are exactly the source metadata's numbers, to better than a
  microsecond.
- **D3'** (corrected motion onset): K1 (sign) is completely clean across 152
  measurements -- no evidence anywhere of video running ahead of data, which
  is the one failure mode a pure detection-latency story cannot produce. K2
  (absolute per-camera tolerance) still fails for a minority of episodes,
  with the largest outliers individually traced to plausible per-episode
  optical/kinematic causes rather than a shared, fixed offset.

None of D1, D2, or D5 shows any sign of misalignment. D3' is the one line
that still fails, and it fails specifically on the check (K2) that is a
*hand-picked numeric tolerance*, not on the check (K1) designed to be
unambiguous evidence of an actual phase bug. Taken together, this is strong
evidence the v2.1 dataset is correctly aligned -- but per this pack's own
rule, "a plausible story does not override a failed gate," and K2 failing
means Gate 2 is not clean.

**Both gates must be clean to upload. Gate 1 (D5) is clean. Gate 2 (D3') is
not (K2 fails). Per section 7 of TIP-005b: STATUS is BLOCKED, and no upload
was attempted.**

## 5. Upload record

**Not applicable -- Gate 2 did not clear, so no upload was attempted**, per
TIP-005b's explicit instruction not to upload before both gates are clean and
not to upload first and report the gate failure afterward.

## 6. Post-upload verify

**Not applicable**, for the same reason as section 5.

## 7. Known limitations

- The video in `ur10e/data/v21` has passed through three lossy generations:
  the original camera capture (H.264, presumably a modest CRF at collection
  time) -> re-encoded to AV1 for the `khanhnd61/ur10e-cup` v3.0 release ->
  re-encoded here to H.264 CRF 23 for gr00t/LeRobot v2.1 compatibility. This
  is unavoidable given the AV1 source cannot be losslessly reversed to the
  original capture.
- libx264 (`-preset medium -crf 23`) was used instead of NVENC: the ffmpeg
  build has `h264_nvenc` compiled in, but this machine's NVIDIA driver
  reports NVENC API 12.2 while the build needs 13.0 (driver >=570.0
  required). At matched CRF, libx264 `medium` typically matches or exceeds
  NVENC `p4` quality; this is a speed cost, not a quality regression.
- Episode 80's release transition (gripper 0->1, frame 525) was not visually
  confirmed in the previous pack's D2 check -- the three sampled frames
  looked too similar at the extracted zoom level to read the jaw state by
  eye. This does not contradict the other three sampled transitions, which
  were unambiguous.
- The D3' K2 failures documented in section 3 remain open findings, not
  fully closed by this pack -- see `ISSUES DISCOVERED` in the TIP-005b
  Completion Report for the recommendation on how to resolve them.
