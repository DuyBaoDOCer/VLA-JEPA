"""Gate 1 (D5) and Gate 2 (D3') phase-alignment verification for ur10e/data/v21.

Read-only: never modifies ur10e/data/v21, v30_delta, or v30. Two independent
checks, run in sequence:

  D5  -- numeric: compares ur10e/results/cut_parameters.csv (written by the
         instrumented to_v21.py during a throwaway re-run of the cutting
         step) against ur10e/results/episode_video_map.csv (source metadata,
         from TIP-003). No pixels, no thresholds -- just arithmetic.

  D3' -- corrected motion-onset check, replacing the previous pack's D3:
         period-2 pixel diff (cancels ~10 Hz lighting flicker present in the
         source video) and per-camera tolerances (wrist vs. side have very
         different sensitivity to the first few, sub-pixel frames of a
         typical episode's motion ramp-up).

Writes ur10e/results/onset_alignment.csv. Exits non-zero if either gate is
not clean -- see ur10e/results/v21_alignment_final.md for the full verdict.

Usage:
    python verify_v21_alignment.py
"""

import csv
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import av

REPO_ROOT = Path(__file__).resolve().parents[2]
V21 = REPO_ROOT / "ur10e" / "data" / "v21"
RESULTS_DIR = REPO_ROOT / "ur10e" / "results"
CUT_PARAMS_CSV = RESULTS_DIR / "cut_parameters.csv"
EPISODE_VIDEO_MAP_CSV = RESULTS_DIR / "episode_video_map.csv"
ONSET_CSV = RESULTS_DIR / "onset_alignment.csv"

N_EPISODES = 81
CAM_KEY = {"observation.images.side": "side", "observation.images.wrist": "wrist"}
K2_LIMITS = {"wrist": 5, "side": 12}


def episode_parquet_path(ep):
    return V21 / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"


def episode_video_path(ep, camera):
    return V21 / "videos" / "chunk-000" / f"observation.images.{camera}" / f"episode_{ep:06d}.mp4"


# ---------------------------------------------------------------------------
# D5 -- cut-parameter arithmetic check
# ---------------------------------------------------------------------------


def run_d5():
    ev = {}
    with open(EPISODE_VIDEO_MAP_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ev[int(row["episode_index"])] = row

    mismatches = []
    n_rows = 0
    with open(CUT_PARAMS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            ep = int(row["episode_index"])
            cam = CAM_KEY[row["camera"]]
            ss = float(row["ss"])
            t = float(row["t"])

            m = re.search(r"file-(\d+)\.mp4$", row["source_file"])
            file_idx = int(m.group(1))

            expected_file_idx = int(ev[ep][f"{cam}_file_index"])
            expected_ss = float(ev[ep][f"{cam}_from_timestamp"])
            expected_t = float(ev[ep][f"{cam}_to_timestamp"]) - float(ev[ep][f"{cam}_from_timestamp"])

            ok = (
                file_idx == expected_file_idx
                and abs(ss - expected_ss) < 1e-6
                and abs(t - expected_t) < 1e-6
            )
            if not ok:
                mismatches.append(
                    {"episode": ep, "camera": cam, "file_idx": file_idx,
                     "expected_file_idx": expected_file_idx, "ss": ss, "expected_ss": expected_ss,
                     "t": t, "expected_t": expected_t}
                )

    return {"n_rows": n_rows, "mismatches": mismatches, "pass": bool(n_rows == 162 and not mismatches)}


# ---------------------------------------------------------------------------
# D3' -- corrected motion-onset check
# ---------------------------------------------------------------------------


def find_f_onset(ep):
    df = pq.read_table(episode_parquet_path(ep)).to_pandas()
    action = np.stack(df["action"].to_numpy())
    nonzero = np.any(action[:, 0:6] != 0.0, axis=1)
    idx = np.flatnonzero(nonzero)
    if len(idx) == 0:
        return None, "no nonzero action delta anywhere in episode"
    if idx[0] == 0:
        return None, "no still lead-in (f_onset would be 0)"
    return int(idx[0]), None


def decode_clip_frames(video_path):
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        arr = frame.to_ndarray(format="rgb24")
        frames.append(arr[::6, ::8].astype(np.float64))
    container.close()
    return frames


def compute_v_onset(frames, f_onset):
    n_frames = len(frames)
    lo = max(0, f_onset - 60)
    hi = min(n_frames - 1, f_onset + 60)

    diffs = [(t, float(np.mean(np.abs(frames[t] - frames[t - 2])))) for t in range(lo + 2, hi + 1)]
    still = [d for t, d in diffs if t < f_onset]
    if len(still) < 3:
        return None, None, "insufficient still frames in window"

    still_arr = np.array(still)
    threshold = float(still_arr.mean() + 5 * still_arr.std())

    v_onset = next((t for t, d in diffs if d > threshold), None)
    if v_onset is None:
        # extend the search forward in 60-frame increments if nothing crossed yet
        extra_hi = min(n_frames - 1, hi + 60)
        for t in range(hi + 1, extra_hi + 1):
            d = float(np.mean(np.abs(frames[t] - frames[t - 2])))
            if d > threshold:
                v_onset = t
                break

    if v_onset is None:
        return None, threshold, "no frame exceeded threshold even after extending search"
    return v_onset, threshold, None


def run_d3_prime():
    skipped = []
    rows = []

    for ep in range(N_EPISODES):
        f_onset, reason = find_f_onset(ep)
        if f_onset is None:
            skipped.append((ep, reason))
            continue

        for camera in ("side", "wrist"):
            frames = decode_clip_frames(episode_video_path(ep, camera))
            v_onset, threshold, err = compute_v_onset(frames, f_onset)
            if err is not None:
                rows.append({"episode_index": ep, "camera": camera, "f_onset": f_onset,
                             "v_onset": "", "offset": "", "error": err})
                continue
            offset = v_onset - f_onset
            rows.append({"episode_index": ep, "camera": camera, "f_onset": f_onset,
                         "v_onset": v_onset, "offset": offset, "error": ""})

    with open(ONSET_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["episode_index", "camera", "f_onset", "v_onset", "offset", "error"])
        writer.writeheader()
        writer.writerows(rows)

    valid = [r for r in rows if r["offset"] != ""]
    negative = [r for r in valid if r["offset"] < 0]

    k2 = {}
    for camera, limit in K2_LIMITS.items():
        offsets = [r["offset"] for r in valid if r["camera"] == camera]
        over = [o for o in offsets if o > limit]
        k2[camera] = {"limit": limit, "n_over": len(over), "n_total": len(offsets),
                      "max": max(offsets) if offsets else None}

    k3 = {}
    for camera in ("wrist", "side"):
        offsets = sorted(r["offset"] for r in valid if r["camera"] == camera)
        if offsets:
            n = len(offsets)
            med = offsets[n // 2] if n % 2 == 1 else (offsets[n // 2 - 1] + offsets[n // 2]) / 2
            k3[camera] = {"min": offsets[0], "median": med, "max": offsets[-1], "n": n}

    return {
        "skipped": skipped,
        "n_valid": len(valid),
        "n_negative": len(negative),
        "negative_rows": negative,
        "k1_pass": len(negative) == 0,
        "k2": k2,
        "k2_pass": all(v["n_over"] == 0 for v in k2.values()),
        "k3": k3,
        "k3_pass": bool(k3.get("wrist", {}).get("min", 99) <= 2),
    }


def main():
    print("=== D5: cut-parameter arithmetic check ===")
    d5 = run_d5()
    print(f"D5: {d5['n_rows']} rows compared, {len(d5['mismatches'])} mismatches, pass={d5['pass']}")
    if d5["mismatches"]:
        for m in d5["mismatches"]:
            print("  MISMATCH:", m)

    print("=== D3': corrected motion-onset check ===")
    d3p = run_d3_prime()
    print(f"skipped episodes: {d3p['skipped']}")
    print(f"K1 (sign): {d3p['n_negative']} negative offsets out of {d3p['n_valid']} valid measurements, "
          f"pass={d3p['k1_pass']}")
    for cam, v in d3p["k2"].items():
        print(f"K2 ({cam}, limit {v['limit']}): {v['n_over']}/{v['n_total']} over limit, max={v['max']}")
    for cam, v in d3p["k3"].items():
        print(f"K3 ({cam}): min={v['min']} median={v['median']} max={v['max']} n={v['n']}")
    print(f"K2 pass={d3p['k2_pass']}, K3 pass={d3p['k3_pass']}")

    gate1_clean = d5["pass"]
    gate2_clean = d3p["k1_pass"] and d3p["k2_pass"] and d3p["k3_pass"]
    print(f"\n=== Gate 1 (D5) clean: {gate1_clean} ===")
    print(f"=== Gate 2 (D3') clean: {gate2_clean} ===")

    if not (gate1_clean and gate2_clean):
        print("[BLOCKED] at least one gate is not clean -- see ur10e/results/v21_alignment_final.md")
        sys.exit(1)

    print("[OK] both gates clean")


if __name__ == "__main__":
    main()
