"""D6 -- source metadata tiling check (TIP-005c gate).

Purely numeric, no pixels, no thresholds, no lighting dependency. Verifies
that the per-episode timestamps recorded in ur10e/results/episode_video_map.csv
(read from the source dataset's own metadata, TIP-003) tile each concatenated
source video with no gaps and no overlaps -- the one thing every other check
in this project (D1, D2, D5, D3') assumes but never directly tested.

For each camera, grouped by source file_index and sorted by from_timestamp:
  - consecutive episodes must abut within one frame period (1/20 s)
  - the first episode of a file must start at ~0
  - the last episode of a file must end at ~the file's real ffprobe duration
  - the per-camera total covered duration must equal 49779/20 = 2488.95 s

Writes ur10e/results/metadata_tiling.csv. Read-only: never touches
ur10e/data/v30, v30_delta, or v21.

Usage:
    python verify_source_metadata_tiling.py
"""

import csv
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "ur10e" / "results"
EPISODE_VIDEO_MAP_CSV = RESULTS_DIR / "episode_video_map.csv"
OUT_CSV = RESULTS_DIR / "metadata_tiling.csv"
VIDEOS_ROOT = REPO_ROOT / "ur10e" / "data" / "v30" / "videos"

FFPROBE_EXE = Path(
    "C:/Users/duybaoDOCer/miniconda3/envs/vlajepa-dev/lib/site-packages/static_ffmpeg/bin/win32/ffprobe.exe"
)

FRAME_PERIOD = 1.0 / 20.0  # one frame period at 20 fps
EXPECTED_TOTAL = 49779 / 20.0  # 2488.95 s
CAMERAS = ("side", "wrist")
N_EPISODES = 81


def ffprobe_duration(path):
    result = subprocess.run(
        [str(FFPROBE_EXE), "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def load_episode_video_map():
    rows = []
    with open(EPISODE_VIDEO_MAP_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def source_video_path(camera, file_index):
    return VIDEOS_ROOT / f"observation.images.{camera}" / "chunk-000" / f"file-{file_index:03d}.mp4"


def main():
    rows = load_episode_video_map()
    assert len(rows) == N_EPISODES, f"expected {N_EPISODES} episodes, got {len(rows)}"

    # discover distinct file_index values used per camera, and get their real durations
    file_indices = {cam: sorted({int(r[f"{cam}_file_index"]) for r in rows}) for cam in CAMERAS}
    durations = {}
    print("=== ffprobe durations of the 7 source files ===")
    for cam in CAMERAS:
        for fi in file_indices[cam]:
            path = source_video_path(cam, fi)
            dur = ffprobe_duration(path)
            durations[(cam, fi)] = dur
            print(f"{cam} file-{fi:03d}.mp4: {dur:.6f} s")

    out_rows = []
    all_clean = True
    max_abs_gap = 0.0
    max_abs_gap_where = None

    for cam in CAMERAS:
        # sort episodes by file_index then from_timestamp
        cam_rows = sorted(
            rows,
            key=lambda r: (int(r[f"{cam}_file_index"]), float(r[f"{cam}_from_timestamp"])),
        )
        by_file = {}
        for r in cam_rows:
            fi = int(r[f"{cam}_file_index"])
            by_file.setdefault(fi, []).append(r)

        for fi, eps in by_file.items():
            for i, r in enumerate(eps):
                ep = int(r["episode_index"])
                from_ts = float(r[f"{cam}_from_timestamp"])
                to_ts = float(r[f"{cam}_to_timestamp"])

                flags = []

                if i == 0:
                    if abs(from_ts - 0.0) >= FRAME_PERIOD:
                        flags.append(f"first-episode-not-at-zero(from={from_ts:.6f})")

                if i == len(eps) - 1:
                    file_dur = durations[(cam, fi)]
                    err = abs(to_ts - file_dur)
                    if err >= FRAME_PERIOD:
                        flags.append(f"last-episode-mismatch(to={to_ts:.6f},file_dur={file_dur:.6f},err={err:.6f})")

                gap_to_next = ""
                if i < len(eps) - 1:
                    next_from = float(eps[i + 1][f"{cam}_from_timestamp"])
                    gap = next_from - to_ts
                    gap_to_next = f"{gap:.6f}"
                    if abs(gap) > max_abs_gap:
                        max_abs_gap = abs(gap)
                        max_abs_gap_where = (cam, fi, ep, int(eps[i + 1]["episode_index"]), gap)
                    if abs(gap) >= FRAME_PERIOD:
                        kind = "gap" if gap > 0 else "overlap"
                        flags.append(f"{kind}({gap:.6f}s)")

                flag_str = ";".join(flags)
                if flags:
                    all_clean = False

                out_rows.append({
                    "camera": cam, "file_index": fi, "episode_index": ep,
                    "from_timestamp": from_ts, "to_timestamp": to_ts,
                    "gap_to_next": gap_to_next, "flag": flag_str,
                })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["camera", "file_index", "episode_index", "from_timestamp", "to_timestamp",
                           "gap_to_next", "flag"]
        )
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {OUT_CSV} ({len(out_rows)} rows)")

    print(f"\nLargest |gap| between consecutive episodes: {max_abs_gap:.6f} s at {max_abs_gap_where}")

    print("\n=== per-camera coverage check ===")
    coverage_ok = True
    for cam in CAMERAS:
        total = sum(float(r[f"{cam}_to_timestamp"]) - float(r[f"{cam}_from_timestamp"]) for r in rows)
        err = abs(total - EXPECTED_TOTAL)
        ok = err < 0.05
        coverage_ok = coverage_ok and ok
        print(f"{cam}: total covered = {total:.6f} s, expected {EXPECTED_TOTAL:.6f} s, err={err:.6f} s, ok={ok}")

    n_flagged = sum(1 for r in out_rows if r["flag"])
    print(f"\nRows flagged (gap/overlap/boundary): {n_flagged} / {len(out_rows)}")

    gate_clean = all_clean and coverage_ok
    print(f"\n=== D6 gate clean: {gate_clean} ===")
    if not gate_clean:
        print("[BLOCKED] metadata does not tile the source videos cleanly")
        sys.exit(1)
    print("[OK] D6 clean")


if __name__ == "__main__":
    main()
