"""Physically split ur10e/data/v21 into a 73-episode train set and an
8-episode held-out set, each a standalone LeRobot v2.1 dataset.

Split is deterministic: episode_index 0..72 -> train, 73..80 -> held-out.
Held-out keeps its original episode_index values (not renumbered).

meta/stats.json is recomputed independently for each split from that
split's own copied parquet and video files -- never copied from v21 or
from the other split -- so normalization statistics never see the other
split's data. See compute_image_stats() for the image-channel sampling
method (full-population mean/std/min/max, seeded spatial subsample for
quantiles) and compute_numeric_stats() for the exact, full-population
numeric-feature statistics.

Read-only on ur10e/data/v21: only ever opened for reading, never written.

Usage:
    python split_dataset.py
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
V21 = REPO_ROOT / "ur10e" / "data" / "v21"
RESULTS_DIR = REPO_ROOT / "ur10e" / "results"

TRAIN_EPISODES = list(range(0, 73))
HELDOUT_EPISODES = list(range(73, 81))

CAMERAS = ["observation.images.side", "observation.images.wrist"]
QUANTILE_LEVELS = [0.01, 0.10, 0.50, 0.90, 0.99]
NUMERIC_FEATURES = [
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]
IMAGE_PIXEL_SAMPLE_SEED = 42
IMAGE_PIXEL_SAMPLES_PER_FRAME = 200
IMAGE_FRAME_STRIDE = 6  # accumulate stats from every 6th decoded frame


def episode_parquet_path(root: Path, ep: int) -> Path:
    return root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"


def episode_video_path(root: Path, ep: int, camera: str) -> Path:
    return root / "videos" / "chunk-000" / camera / f"episode_{ep:06d}.mp4"


def make_split_dirs(root: Path) -> None:
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for camera in CAMERAS:
        (root / "videos" / "chunk-000" / camera).mkdir(parents=True, exist_ok=True)
    (root / "meta").mkdir(parents=True, exist_ok=True)


def copy_split_files(episodes: list[int], dst_root: Path) -> None:
    for ep in episodes:
        shutil.copy2(episode_parquet_path(V21, ep), episode_parquet_path(dst_root, ep))
        for camera in CAMERAS:
            shutil.copy2(
                episode_video_path(V21, ep, camera), episode_video_path(dst_root, ep, camera)
            )


def write_filtered_jsonl(src_path: Path, dst_path: Path, episodes: set[int]) -> int:
    n_written = 0
    with open(src_path, encoding="utf-8") as fin, open(dst_path, "w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            if record["episode_index"] in episodes:
                fout.write(json.dumps(record) + "\n")
                n_written += 1
    return n_written


def write_info(dst_root: Path, total_episodes: int, total_frames: int, split_range: str) -> None:
    info = json.loads((V21 / "meta" / "info.json").read_text(encoding="utf-8"))

    # Fields required to stay byte-identical to the source, per TIP-006 3.1.
    unchanged = {k: info[k] for k in ("data_path", "video_path", "chunks_size", "fps", "features")}

    info["total_episodes"] = total_episodes
    info["total_frames"] = total_frames
    info["total_chunks"] = 1  # chunks_size=1000 > either split's episode count
    info["total_videos"] = total_episodes * len(CAMERAS)
    info["splits"] = {"train": split_range}
    for k, v in unchanged.items():
        info[k] = v

    (dst_root / "meta" / "info.json").write_text(
        json.dumps(info, indent=4) + "\n", encoding="utf-8"
    )


def copy_verbatim_meta(dst_root: Path) -> None:
    for name in ("tasks.jsonl", "modality.json"):
        shutil.copy2(V21 / "meta" / name, dst_root / "meta" / name)


# ---------------------------------------------------------------------------
# stats.json recomputation
# ---------------------------------------------------------------------------


def compute_numeric_stats(dst_root: Path, episodes: list[int]) -> dict:
    columns = {}
    for feature in NUMERIC_FEATURES:
        columns[feature] = []
    for ep in episodes:
        table = pq.read_table(episode_parquet_path(dst_root, ep), columns=NUMERIC_FEATURES)
        for feature in NUMERIC_FEATURES:
            columns[feature].append(np.array(table.column(feature).to_pylist(), dtype=np.float64))

    stats = {}
    for feature in NUMERIC_FEATURES:
        arr = np.concatenate(columns[feature], axis=0)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        stats[feature] = {
            "min": arr.min(axis=0).tolist(),
            "max": arr.max(axis=0).tolist(),
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "count": [int(arr.shape[0])],
        }
        for q in QUANTILE_LEVELS:
            stats[feature][f"q{int(round(q * 100)):02d}"] = np.quantile(arr, q, axis=0).tolist()
    return stats


def compute_image_stats(dst_root: Path, camera: str, episodes: list[int]) -> dict:
    """Full-population mean/std/min/max (every decoded pixel of every frame
    of every IMAGE_FRAME_STRIDE-th decoded frame across this split's
    episode videos, in decode order); q01/q10/q50/q90/q99 estimated from a
    fixed-seed spatial subsample (IMAGE_PIXEL_SAMPLES_PER_FRAME pixel
    positions per accumulated frame) because holding the full pixel
    population in memory for an exact quantile is infeasible. Both the
    frame stride and the pixel subsample are deterministic and
    reproducible given the fixed seed / fixed stride, but explicitly NOT
    the same algorithm the original v21 stats.json was built with -- see
    split_and_baselines.md section 2 for why that is fine.
    """
    rng = np.random.default_rng(IMAGE_PIXEL_SAMPLE_SEED)
    count = 0
    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)
    min_c = np.full(3, np.inf, dtype=np.float64)
    max_c = np.full(3, -np.inf, dtype=np.float64)
    sample_idx = None
    quantile_samples = []
    frame_counter = 0

    for ep in episodes:
        path = episode_video_path(dst_root, ep, camera)
        container = av.open(str(path))
        for frame in container.decode(video=0):
            frame_counter += 1
            if frame_counter % IMAGE_FRAME_STRIDE != 0:
                continue
            arr = frame.to_ndarray(format="rgb24").astype(np.float64) / 255.0
            flat = arr.reshape(-1, 3)
            if sample_idx is None:
                sample_idx = rng.integers(0, flat.shape[0], size=IMAGE_PIXEL_SAMPLES_PER_FRAME)
            count += flat.shape[0]
            sum_c += flat.sum(axis=0)
            sumsq_c += (flat**2).sum(axis=0)
            min_c = np.minimum(min_c, flat.min(axis=0))
            max_c = np.maximum(max_c, flat.max(axis=0))
            quantile_samples.append(flat[sample_idx])
        container.close()

    mean_c = sum_c / count
    var_c = np.maximum(sumsq_c / count - mean_c**2, 0.0)
    std_c = np.sqrt(var_c)
    samples = np.concatenate(quantile_samples, axis=0)  # (n_samples, 3)

    def nest(vals) -> list:
        return [[[float(v)]] for v in vals]

    stats = {
        "min": nest(min_c),
        "max": nest(max_c),
        "mean": nest(mean_c),
        "std": nest(std_c),
        "count": [int(count)],
    }
    for q in QUANTILE_LEVELS:
        qvals = np.quantile(samples, q, axis=0)
        stats[f"q{int(round(q * 100)):02d}"] = nest(qvals)
    return stats


def compute_split_stats(dst_root: Path, episodes: list[int]) -> dict:
    stats = compute_numeric_stats(dst_root, episodes)
    for camera in CAMERAS:
        t0 = time.perf_counter()
        stats[camera] = compute_image_stats(dst_root, camera, episodes)
        print(
            f"  image stats for {dst_root.name}/{camera}: "
            f"{time.perf_counter() - t0:.1f}s, count={stats[camera]['count'][0]}",
            flush=True,
        )
    return stats


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def build_split(name: str, episodes: list[int], split_range: str) -> dict:
    dst_root = REPO_ROOT / "ur10e" / "data" / name
    if dst_root.exists():
        shutil.rmtree(dst_root)
    make_split_dirs(dst_root)

    print(f"[{name}] copying {len(episodes)} episodes (parquet + 2 cameras)")
    copy_split_files(episodes, dst_root)

    episode_set = set(episodes)
    n_ep = write_filtered_jsonl(
        V21 / "meta" / "episodes.jsonl", dst_root / "meta" / "episodes.jsonl", episode_set
    )
    n_stats = write_filtered_jsonl(
        V21 / "meta" / "episodes_stats.jsonl",
        dst_root / "meta" / "episodes_stats.jsonl",
        episode_set,
    )
    copy_verbatim_meta(dst_root)

    episodes_meta = [
        json.loads(line) for line in (dst_root / "meta" / "episodes.jsonl").read_text().splitlines()
    ]
    total_frames = sum(r["length"] for r in episodes_meta)

    write_info(dst_root, total_episodes=len(episodes), total_frames=total_frames, split_range=split_range)

    print(f"[{name}] recomputing stats.json from {name}'s own files only")
    stats = compute_split_stats(dst_root, episodes)
    (dst_root / "meta" / "stats.json").write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")

    print(f"[{name}] done: {len(episodes)} episodes, {total_frames} frames, "
          f"{n_ep} episodes.jsonl lines, {n_stats} episodes_stats.jsonl lines")
    return {"episodes": episodes, "total_frames": total_frames}


def main() -> None:
    train_info = build_split("v21_train", TRAIN_EPISODES, "0:73")
    heldout_info = build_split("v21_heldout", HELDOUT_EPISODES, "73:81")

    split_json = {
        "train_episodes": TRAIN_EPISODES,
        "heldout_episodes": HELDOUT_EPISODES,
        "train_frames": train_info["total_frames"],
        "heldout_frames": heldout_info["total_frames"],
        "rationale": "last 8 episodes held out, deterministic, no shuffling",
        "created_from": "ur10e/data/v21",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "split.json").write_text(json.dumps(split_json, indent=4) + "\n", encoding="utf-8")
    print("\nwrote ur10e/results/split.json")
    print(json.dumps(split_json, indent=2))


if __name__ == "__main__":
    main()
