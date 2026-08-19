"""Orchestrate the v3.0 -> v2.1 conversion and run the D1-D4 phase-alignment checks.

This script does NOT reimplement the episode-splitting logic (that lives in
to_v21.py, an edited copy of the upstream converter). It only:

  1. wires the two input sources together (data/meta from v30_delta,
     videos from v30) via directory junctions -- no video is copied;
  2. runs the upstream converter (to_v21.convert_dataset);
  3. moves the result into ur10e/data/v21/, copies meta/modality.json in
     from the reference dataset, and recomputes the "action" entry of
     meta/stats.json (the rest of that file is untouched, since only the
     action representation changed in the previous pack);
  4. runs D1 (frame count), D3 (motion onset), D4 (structure/codec) checks
     against every episode and both cameras -- no sampling;
  5. extracts the D2 gripper-transition frames for visual inspection
     (episodes 0 and 80, side camera) -- the frame-offset conclusion itself
     is written into the report by hand after looking at the images, the
     same way TIP-003 resolved gripper polarity.

Must run under the lerobot_conversion subproject's venv (it imports
to_v21, which imports lerobot). See ur10e/results/v21_conversion.md.
"""

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "ur10e" / "src"
sys.path.insert(0, str(SRC_DIR))

V30_ROOT = REPO_ROOT / "ur10e" / "data" / "v30"
V30_DELTA_ROOT = REPO_ROOT / "ur10e" / "data" / "v30_delta"
V21_ROOT = REPO_ROOT / "ur10e" / "data" / "v21"
RESULTS_DIR = REPO_ROOT / "ur10e" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SCRATCH_ROOT = REPO_ROOT.parent / "_scratch" / "v21_staging"
REPO_ID = "ur10e-cup-local"
STAGING_ROOT = SCRATCH_ROOT / REPO_ID

# Absolute path to the validated ffmpeg/ffprobe build (installed via the
# `static-ffmpeg` pip package into the vlajepa-dev conda env in TIP-005a;
# NVENC was present but unusable -- driver reports API 12.2, needs 13.0 --
# so this build's libx264 path is what actually gets used). This script
# itself runs under the separate lerobot_conversion venv, which has no
# ffmpeg of its own, so the binary is referenced by absolute path rather
# than assumed to be on PATH.
FFMPEG_BIN_DIR = Path(
    "C:/Users/duybaoDOCer/miniconda3/envs/vlajepa-dev/lib/site-packages/static_ffmpeg/bin/win32"
)
FFMPEG_EXE = FFMPEG_BIN_DIR / "ffmpeg.exe"
FFPROBE_EXE = FFMPEG_BIN_DIR / "ffprobe.exe"

MODALITY_REPO_ID = "DuyBao44DOCer/ur10e-cup-v21-gr00t"
MODALITY_FILENAME = "meta/modality.json"

EXPECTED_N_EPISODES = 81
EXPECTED_N_ROWS = 49779
EXPECTED_GRIPPER_FREQ = {1.0: 25411, 0.0: 24368}
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
DIM_NAMES = JOINT_NAMES + ["gripper"]
FPS = 20


def setup_staging():
    STAGING_ROOT.parent.mkdir(parents=True, exist_ok=True)
    links = {
        "data": V30_DELTA_ROOT / "data",
        "meta": V30_DELTA_ROOT / "meta",
        "videos": V30_ROOT / "videos",
    }
    for name, target in links.items():
        link_path = STAGING_ROOT / name
        if link_path.is_dir() or link_path.is_symlink():
            continue
        STAGING_ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Junction created: {link_path} -> {target}")


def run_conversion():
    os.environ["PATH"] = str(FFMPEG_BIN_DIR) + os.pathsep + os.environ.get("PATH", "")
    import to_v21  # noqa: E402  (imported after sys.path/PATH setup)

    print(f"Running upstream convert_dataset(repo_id={REPO_ID!r}, root={SCRATCH_ROOT})")
    to_v21.convert_dataset(repo_id=REPO_ID, root=str(SCRATCH_ROOT), force_conversion=False)
    print(f"Conversion finished. Output now at {STAGING_ROOT}")


def move_to_destination():
    if V21_ROOT.exists():
        shutil.rmtree(V21_ROOT)
    V21_ROOT.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(STAGING_ROOT), str(V21_ROOT))
    print(f"Moved converted dataset to {V21_ROOT}")


def copy_modality_json():
    from huggingface_hub import hf_hub_download

    print(f"Downloading {MODALITY_FILENAME} from {MODALITY_REPO_ID} ...")
    cached_path = hf_hub_download(
        repo_id=MODALITY_REPO_ID, filename=MODALITY_FILENAME, repo_type="dataset"
    )
    dest = V21_ROOT / "meta" / "modality.json"
    shutil.copy2(cached_path, dest)
    print(f"Copied modality.json -> {dest}")
    return dest


def stats_row(values):
    vmin, vmax = float(values.min()), float(values.max())
    q01, q10, q50, q90, q99 = (float(x) for x in np.percentile(values, [1, 10, 50, 90, 99]))
    mean, std = float(values.mean()), float(values.std())
    return vmin, vmax, mean, std, q01, q10, q50, q90, q99


def recompute_action_stats():
    """Replace only the 'action' entry of meta/stats.json with fresh stats
    computed from the delta-converted data. Every other key (state, images,
    timestamp, ...) describes data that TIP-004 did not change, so it is
    left exactly as copied from the source."""
    stats_path = V21_ROOT / "meta" / "stats.json"
    with open(stats_path, "r", encoding="utf-8") as f:
        stats = json.load(f)

    old_action_std = stats["action"]["std"]

    table = pq.read_table(V30_DELTA_ROOT / "data" / "chunk-000" / "file-000.parquet")
    df = table.to_pandas()
    action = np.stack(df["action"].to_numpy()).astype(np.float64)

    mins, maxs, means, stds, q01s, q10s, q50s, q90s, q99s = [], [], [], [], [], [], [], [], []
    for d in range(7):
        vmin, vmax, mean, std, q01, q10, q50, q90, q99 = stats_row(action[:, d])
        mins.append(vmin)
        maxs.append(vmax)
        means.append(mean)
        stds.append(std)
        q01s.append(q01)
        q10s.append(q10)
        q50s.append(q50)
        q90s.append(q90)
        q99s.append(q99)

    stats["action"] = {
        "min": mins,
        "max": maxs,
        "mean": means,
        "std": stds,
        "count": [len(df)],
        "q01": q01s,
        "q10": q10s,
        "q50": q50s,
        "q90": q90s,
        "q99": q99s,
    }

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4)

    return old_action_std, stds


def ffprobe_frame_count(video_path):
    result = subprocess.run(
        [
            str(FFPROBE_EXE), "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def ffprobe_codec(video_path):
    result = subprocess.run(
        [
            str(FFPROBE_EXE), "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def episode_parquet_path(ep):
    return V21_ROOT / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"


def episode_video_path(ep, camera):
    return V21_ROOT / "videos" / "chunk-000" / f"observation.images.{camera}" / f"episode_{ep:06d}.mp4"


def run_d1_d4(episode_records):
    d1_rows = []
    d1_all_pass = True
    row_counts = {}
    for ep in range(EXPECTED_N_EPISODES):
        pq_path = episode_parquet_path(ep)
        n_rows = pq.read_table(pq_path).num_rows
        row_counts[ep] = n_rows
        for camera in ("side", "wrist"):
            vid_path = episode_video_path(ep, camera)
            n_frames = ffprobe_frame_count(vid_path)
            ok = n_frames == n_rows
            d1_all_pass = d1_all_pass and ok
            d1_rows.append((ep, camera, n_rows, n_frames, ok))

    total_rows = sum(row_counts.values())

    n_parquet = len(list((V21_ROOT / "data" / "chunk-000").glob("episode_*.parquet")))
    n_side = len(list((V21_ROOT / "videos" / "chunk-000" / "observation.images.side").glob("episode_*.mp4")))
    n_wrist = len(list((V21_ROOT / "videos" / "chunk-000" / "observation.images.wrist").glob("episode_*.mp4")))

    episodes_jsonl = V21_ROOT / "meta" / "episodes.jsonl"
    ep_lines = episodes_jsonl.read_text(encoding="utf-8").strip().splitlines()
    ep_parse_ok = True
    for line in ep_lines:
        try:
            json.loads(line)
        except json.JSONDecodeError:
            ep_parse_ok = False

    tasks_jsonl = V21_ROOT / "meta" / "tasks.jsonl"
    tasks_ok = tasks_jsonl.exists()
    if tasks_ok:
        try:
            for line in tasks_jsonl.read_text(encoding="utf-8").strip().splitlines():
                json.loads(line)
        except json.JSONDecodeError:
            tasks_ok = False

    modality_path = V21_ROOT / "meta" / "modality.json"
    modality_ok = modality_path.exists()
    if modality_ok:
        try:
            json.loads(modality_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            modality_ok = False

    with open(V21_ROOT / "meta" / "info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    info_ok = (
        info.get("data_path") == "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        and info.get("video_path") == "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        and info.get("chunks_size") == 1000
    )

    codec_rows = []
    codec_all_h264 = True
    for ep in range(EXPECTED_N_EPISODES):
        for camera in ("side", "wrist"):
            codec = ffprobe_codec(episode_video_path(ep, camera))
            codec_rows.append((ep, camera, codec))
            if codec != "h264":
                codec_all_h264 = False

    d4 = {
        "n_parquet": n_parquet,
        "n_side": n_side,
        "n_wrist": n_wrist,
        "n_episodes_jsonl_lines": len(ep_lines),
        "episodes_jsonl_parse_ok": ep_parse_ok,
        "tasks_jsonl_ok": tasks_ok,
        "modality_ok": modality_ok,
        "info_ok": info_ok,
        "codec_all_h264": codec_all_h264,
        "codec_rows": codec_rows,
        "pass": bool(
            n_parquet == EXPECTED_N_EPISODES
            and n_side == EXPECTED_N_EPISODES
            and n_wrist == EXPECTED_N_EPISODES
            and len(ep_lines) == EXPECTED_N_EPISODES
            and ep_parse_ok
            and tasks_ok
            and modality_ok
            and info_ok
            and codec_all_h264
        ),
    }

    d1 = {
        "rows": d1_rows,
        "total_rows": total_rows,
        "pass": bool(d1_all_pass and total_rows == EXPECTED_N_ROWS),
    }

    return d1, d4, row_counts


def gripper_frequency_check():
    action_vals = []
    state_vals = []
    for ep in range(EXPECTED_N_EPISODES):
        df = pq.read_table(episode_parquet_path(ep)).to_pandas()
        action = np.stack(df["action"].to_numpy())
        state = np.stack(df["observation.state"].to_numpy())
        action_vals.append(action[:, 6])
        state_vals.append(state[:, 6])
    action_all = np.concatenate(action_vals)
    state_all = np.concatenate(state_vals)

    av, ac = np.unique(action_all, return_counts=True)
    sv, sc = np.unique(state_all, return_counts=True)
    action_freq = dict(zip(av.tolist(), ac.tolist()))
    state_freq = dict(zip(sv.tolist(), sc.tolist()))

    return {
        "action_freq": action_freq,
        "state_freq": state_freq,
        "pass": bool(action_freq == EXPECTED_GRIPPER_FREQ and state_freq == EXPECTED_GRIPPER_FREQ),
    }


def decode_clip_frames(video_path):
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray(format="rgb24")
        small = arr[::6, ::8].astype(np.float64)  # ~106x80 -> further to ~80x60ish
        frames.append(small)
    container.close()
    return frames


def find_f_onset(df_ep, state_ep):
    """First frame index (local to the episode) where any of the 6 joints'
    delta action differs from 0."""
    nonzero = np.any(np.stack(df_ep["action"].to_numpy())[:, 0:6] != 0.0, axis=1)
    idx = np.flatnonzero(nonzero)
    if len(idx) == 0 or idx[0] == 0:
        return None
    return int(idx[0])


def run_d3():
    rows = []
    n_usable = 0
    for ep in range(EXPECTED_N_EPISODES):
        df_ep = pq.read_table(episode_parquet_path(ep)).to_pandas()
        f_onset = find_f_onset(df_ep, None)
        if f_onset is None:
            continue
        n_usable += 1

        for camera in ("side", "wrist"):
            frames = decode_clip_frames(episode_video_path(ep, camera))
            n_frames = len(frames)
            lo = max(0, f_onset - 60)
            hi = min(n_frames - 1, f_onset + 60)

            diffs = []
            for t in range(lo + 1, hi + 1):
                d = float(np.mean(np.abs(frames[t] - frames[t - 1])))
                diffs.append((t, d))

            still_diffs = [d for t, d in diffs if t < f_onset]
            if len(still_diffs) < 3:
                rows.append((ep, camera, f_onset, None, None, "insufficient still frames"))
                continue
            still_arr = np.array(still_diffs)
            threshold = float(still_arr.mean() + 5 * still_arr.std())

            v_onset = None
            for t, d in diffs:
                if d > threshold:
                    v_onset = t
                    break

            if v_onset is None:
                rows.append((ep, camera, f_onset, None, threshold, "no frame exceeded threshold"))
                continue

            offset = v_onset - f_onset
            rows.append((ep, camera, f_onset, v_onset, offset, None))

    valid = [r for r in rows if r[3] is not None]
    offsets = [abs(r[4]) for r in valid]
    max_offset = max(offsets) if offsets else None
    mean_offset = float(np.mean(offsets)) if offsets else None
    n_over_2 = sum(1 for o in offsets if o > 2)
    signed_offsets = [r[4] for r in valid]
    systematic = False
    if signed_offsets:
        s_arr = np.array(signed_offsets)
        if np.all(s_arr == s_arr[0]) and s_arr[0] != 0:
            systematic = True

    return {
        "rows": rows,
        "n_usable": n_usable,
        "n_valid": len(valid),
        "max_offset": max_offset,
        "mean_offset": mean_offset,
        "n_over_2": n_over_2,
        "systematic": systematic,
        "pass": bool(n_over_2 == 0 and not systematic and len(valid) == n_usable * 2),
    }


def find_transitions_in_episode(ep):
    df = pq.read_table(episode_parquet_path(ep)).to_pandas()
    state = np.stack(df["observation.state"].to_numpy())
    gripper = state[:, 6]
    transitions = []
    for t in range(1, len(gripper)):
        if gripper[t] != gripper[t - 1]:
            transitions.append((t, float(gripper[t - 1]), float(gripper[t])))
    return transitions


def extract_frame_at(video_path, frame_idx):
    import av

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    result = None
    for i, frame in enumerate(container.decode(stream)):
        if i == frame_idx:
            result = frame.to_ndarray(format="rgb24")
            break
    container.close()
    return result


def run_d2():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    extracted = []
    for ep in (0, 80):
        transitions = find_transitions_in_episode(ep)
        video_path = episode_video_path(ep, "side")
        for n, (f, before, after) in enumerate(transitions):
            b_int, a_int = int(round(before)), int(round(after))
            for offset, label in ((-3, "m3"), (0, "p0"), (3, "p3")):
                target = f + offset
                if target < 0:
                    continue
                arr = extract_frame_at(video_path, target)
                if arr is None:
                    continue
                name = f"r14_ep{ep:03d}_side_transition{n}_offset{label}.png"
                plt.imsave(FIGURES_DIR / name, arr)
                extracted.append(
                    {"episode": ep, "transition": n, "frame": f, "offset_label": label,
                     "target_frame": target, "before": b_int, "after": a_int, "file": name}
                )
    return extracted


def main():
    print("=== Setting up staging junctions ===")
    setup_staging()

    print("=== Running upstream converter (to_v21.py) ===")
    run_conversion()

    print("=== Moving output into ur10e/data/v21 ===")
    move_to_destination()

    print("=== Copying meta/modality.json from reference dataset ===")
    copy_modality_json()

    print("=== Recomputing action stats in meta/stats.json ===")
    old_std, new_std = recompute_action_stats()
    print(f"old action.std (stale, absolute): {old_std}")
    print(f"new action.std (delta): {new_std}")

    print("=== Loading episode records for D1/D4 ===")
    episode_records = None  # not needed directly; per-episode parquet files are read individually

    print("=== Running D1 (frame count) and D4 (structure) ===")
    d1, d4, row_counts = run_d1_d4(episode_records)
    print(f"D1: total_rows={d1['total_rows']} pass={d1['pass']}")
    print(f"D4: pass={d4['pass']}")

    print("=== Running gripper frequency check ===")
    gripper_check = gripper_frequency_check()
    print(f"gripper check: pass={gripper_check['pass']}")

    print("=== Running D3 (motion onset) -- this decodes 162 short clips ===")
    d3 = run_d3()
    print(f"D3: n_usable={d3['n_usable']} max_offset={d3['max_offset']} "
          f"mean_offset={d3['mean_offset']} n_over_2={d3['n_over_2']} systematic={d3['systematic']}")

    print("=== Extracting D2 gripper-transition frames (episodes 0, 80, side camera) ===")
    d2_files = run_d2()
    print(f"D2: extracted {len(d2_files)} frames")

    results = {
        "d1": d1, "d3": d3, "d4": d4, "gripper_check": gripper_check,
        "d2_files": d2_files, "old_action_std": old_std, "new_action_std": new_std,
        "row_counts": row_counts,
    }
    out_path = RESULTS_DIR / "_v21_conversion_raw_results.json"

    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        raise TypeError

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=default)
    print(f"Raw results written to {out_path}")

    all_pass = d1["pass"] and d3["pass"] and d4["pass"] and gripper_check["pass"]
    print(f"=== overall automated checks pass: {all_pass} ===")
    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
