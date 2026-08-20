"""Measure how much the v3.0 AV1 -> v2.1 H.264 re-encode actually cost in
pixel fidelity, at both the stored resolution and the resolution the model
actually consumes.

For 3 episodes (small/middle/large index) x 2 cameras, decodes ~50 frames
from the v2.1 clip and the matching ~50 frames from the source AV1 file
(same start timestamp, from episode_video_map.csv), and computes PSNR/SSIM
via ffmpeg's own psnr/ssim filters -- once at native 480x640, once after
scaling both sides to 256x256 (the vjepa2-vitl-fpc64-256 input size).

Read-only on ur10e/data/v21 and ur10e/data/v30. No re-encoding happens
here regardless of the measured quality -- that decision belongs to the
Contractor, not this pack.

Usage:
    python compute_compression_quality.py
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
V21_VIDEOS = REPO_ROOT / "ur10e" / "data" / "v21" / "videos" / "chunk-000"
V30_VIDEOS = REPO_ROOT / "ur10e" / "data" / "v30" / "videos"
RESULTS_DIR = REPO_ROOT / "ur10e" / "results"
EPISODE_VIDEO_MAP = RESULTS_DIR / "episode_video_map.csv"

TEST_EPISODES = [0, 40, 80]  # small, middle, large index
CAMERAS = ["observation.images.side", "observation.images.wrist"]
N_FRAMES = 50
RESOLUTIONS = {"480x640": None, "256x256": (256, 256)}

# The conda env's own Library/bin/ffmpeg.exe fails to launch on this machine
# (missing DLL on PATH outside an activated shell); the static_ffmpeg wheel's
# bundled binary is self-contained and known to work, so it is tried first.
_STATIC_FFMPEG = (
    Path.home()
    / "miniconda3"
    / "envs"
    / "vlajepa-dev"
    / "Lib"
    / "site-packages"
    / "static_ffmpeg"
    / "bin"
    / "win32"
    / "ffmpeg.exe"
)
FFMPEG = str(_STATIC_FFMPEG) if _STATIC_FFMPEG.exists() else (shutil.which("ffmpeg") or "ffmpeg")

PSNR_LINE_RE = re.compile(r"psnr_avg:([\d.]+|inf)")
SSIM_LINE_RE = re.compile(r"All:([\d.]+)")


def load_episode_video_map() -> dict[int, dict]:
    rows = {}
    with open(EPISODE_VIDEO_MAP, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows[int(row["episode_index"])] = row
    return rows


def v21_clip_path(ep: int, camera: str) -> Path:
    return V21_VIDEOS / camera / f"episode_{ep:06d}.mp4"


def v30_source_path(camera: str, file_index: str) -> Path:
    return V30_VIDEOS / camera / "chunk-000" / f"file-{int(file_index):03d}.mp4"


def run_metric(metric: str, src: Path, src_seek: float, ref: Path, scale) -> list[float]:
    with tempfile.TemporaryDirectory() as tmp:
        # Absolute Windows paths (drive-letter colon) inside an -lavfi option
        # value collide with ffmpeg's own ":" option separator even when
        # escaped, so the stats file is addressed by a bare relative name
        # with the subprocess cwd pointed at the temp dir instead.
        stats_filename = "stats.log"
        stats_file = Path(tmp) / stats_filename

        def chain(label_in: str, label_out: str) -> str:
            f = f"[{label_in}]trim=end_frame={N_FRAMES},setpts=PTS-STARTPTS"
            if scale is not None:
                f += f",scale={scale[0]}:{scale[1]}:flags=bicubic"
            return f + f"[{label_out}]"

        lavfi = ";".join(
            [chain("0:v", "a"), chain("1:v", "b"), f"[a][b]{metric}=stats_file={stats_filename}"]
        )
        cmd = [
            FFMPEG,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{src_seek:.6f}",
            "-i",
            str(src.resolve()),
            "-i",
            str(ref.resolve()),
            "-lavfi",
            lavfi,
            "-f",
            "null",
            "-",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120, cwd=tmp)

        values = []
        pattern = PSNR_LINE_RE if metric == "psnr" else SSIM_LINE_RE
        for line in stats_file.read_text(encoding="utf-8").splitlines():
            m = pattern.search(line)
            if m:
                raw = m.group(1)
                values.append(float("inf") if raw == "inf" else float(raw))
        return values


def main() -> None:
    video_map = load_episode_video_map()
    results = []

    for ep in TEST_EPISODES:
        row = video_map[ep]
        for camera in CAMERAS:
            cam_short = camera.split(".")[-1]
            v21_path = v21_clip_path(ep, camera)
            file_index = row[f"{cam_short}_file_index"]
            from_ts = float(row[f"{cam_short}_from_timestamp"])
            v30_path = v30_source_path(camera, file_index)

            for res_label, scale in RESOLUTIONS.items():
                psnr_vals = run_metric("psnr", v30_path, from_ts, v21_path, scale)
                ssim_vals = run_metric("ssim", v30_path, from_ts, v21_path, scale)
                entry = {
                    "episode_index": ep,
                    "camera": cam_short,
                    "resolution": res_label,
                    "n_frames_psnr": len(psnr_vals),
                    "n_frames_ssim": len(ssim_vals),
                    "psnr_mean_db": sum(psnr_vals) / len(psnr_vals) if psnr_vals else None,
                    "psnr_min_db": min(psnr_vals) if psnr_vals else None,
                    "ssim_mean": sum(ssim_vals) / len(ssim_vals) if ssim_vals else None,
                    "ssim_min": min(ssim_vals) if ssim_vals else None,
                }
                results.append(entry)
                print(
                    f"ep={ep:3d} cam={cam_short:5s} res={res_label:8s} "
                    f"PSNR mean={entry['psnr_mean_db']:.2f} min={entry['psnr_min_db']:.2f} dB | "
                    f"SSIM mean={entry['ssim_mean']:.4f} min={entry['ssim_min']:.4f} "
                    f"(n={entry['n_frames_psnr']})"
                )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "compression_quality.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print("\nwrote ur10e/results/compression_quality.json")

    for res_label in RESOLUTIONS:
        subset = [r for r in results if r["resolution"] == res_label]
        mean_psnr = sum(r["psnr_mean_db"] for r in subset) / len(subset)
        min_psnr = min(r["psnr_min_db"] for r in subset)
        mean_ssim = sum(r["ssim_mean"] for r in subset) / len(subset)
        min_ssim = min(r["ssim_min"] for r in subset)
        print(
            f"\n[{res_label}] overall mean PSNR={mean_psnr:.2f} dB, min PSNR={min_psnr:.2f} dB, "
            f"mean SSIM={mean_ssim:.4f}, min SSIM={min_ssim:.4f}"
        )


if __name__ == "__main__":
    main()
