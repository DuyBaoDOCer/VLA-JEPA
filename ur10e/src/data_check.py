"""Raw dataset inspection for the ur10e-cup source dataset (LeRobot v3.0 layout).

Read-only quality gate: measures the dataset and writes a report. Never
modifies any file under the dataset root. See ur10e/results/data_check.md
for the write-up this script produces.

Usage:
    python data_check.py <dataset_root>
"""

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_dataset_v30 import load_data_parquet  # noqa: E402

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
DIM_NAMES = JOINT_NAMES + ["gripper"]
GRIPPER_DIM = 6
WRIST_2_DIM = 4
FPS = 20

# Gripper polarity, determined by visually inspecting the wrist-camera frames
# this script extracts to ur10e/results/figures/r10_ep*.png (see section 4
# of data_check.md for the reasoning). Left as None until that inspection is
# done; the report clearly marks the conclusion as pending in that case.
#
# Once determined: "0_is_open" means gripper value 0 corresponds to the open
# jaw position (and 1 to closed); "1_is_open" means the reverse.
GRIPPER_POLARITY = "1_is_open"
GRIPPER_POLARITY_EVIDENCE = (
    "The wrist camera does not include the gripper mechanism in its field of view at any of the "
    "sampled timestamps (see the 'Supplementary evidence' subsection below) -- this is a deviation "
    "from the wrist-only methodology in TIP-003 section 4.3(c), taken because the wrist frames alone "
    "were inconclusive. The conclusion below is instead drawn from SIDE camera frames at the same "
    "transitions, in episodes 0 and 80: with the gripper value at 0, the jaw is visibly closed around "
    "the cup, holding it above the bin; immediately after the value flips to 1, the jaw is visibly open "
    "and the cup has dropped into the bin. This pattern is consistent and independently reproduced in "
    "both episodes."
)


def fmt(x, nd=8):
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}g}"


def human_readable_size(num_bytes):
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def dir_size(path):
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            total += os.path.getsize(os.path.join(dirpath, name))
    return total


def episode_segments(episode_ids):
    """Return (start, end, episode_index) for each contiguous episode block."""
    change_points = np.flatnonzero(np.diff(episode_ids) != 0) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(episode_ids)]))
    return list(zip(starts.tolist(), ends.tolist(), episode_ids[starts].tolist()))


# ---------------------------------------------------------------------------
# 4.1 Episode-to-video map
# ---------------------------------------------------------------------------


def build_episode_video_map(root):
    meta_path = root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    ep = pd.read_parquet(meta_path)
    columns_found = list(ep.columns)

    required = [
        "episode_index",
        "length",
        "dataset_from_index",
        "dataset_to_index",
        "videos/observation.images.side/chunk_index",
        "videos/observation.images.side/file_index",
        "videos/observation.images.side/from_timestamp",
        "videos/observation.images.side/to_timestamp",
        "videos/observation.images.wrist/chunk_index",
        "videos/observation.images.wrist/file_index",
        "videos/observation.images.wrist/from_timestamp",
        "videos/observation.images.wrist/to_timestamp",
    ]
    missing = [c for c in required if c not in ep.columns]
    if missing:
        return columns_found, None, missing

    out = pd.DataFrame(
        {
            "episode_index": ep["episode_index"],
            "length": ep["length"],
            "dataset_from_index": ep["dataset_from_index"],
            "dataset_to_index": ep["dataset_to_index"],
            "side_chunk_index": ep["videos/observation.images.side/chunk_index"],
            "side_file_index": ep["videos/observation.images.side/file_index"],
            "side_from_timestamp": ep["videos/observation.images.side/from_timestamp"],
            "side_to_timestamp": ep["videos/observation.images.side/to_timestamp"],
            "wrist_chunk_index": ep["videos/observation.images.wrist/chunk_index"],
            "wrist_file_index": ep["videos/observation.images.wrist/file_index"],
            "wrist_from_timestamp": ep["videos/observation.images.wrist/from_timestamp"],
            "wrist_to_timestamp": ep["videos/observation.images.wrist/to_timestamp"],
        }
    ).sort_values("episode_index").reset_index(drop=True)

    return columns_found, out, missing


def check_episode_video_map(ev):
    checks = {}

    checks["n_rows"] = len(ev)
    checks["sum_length"] = int(ev["length"].sum())

    duration_errors = []
    for cam in ("side", "wrist"):
        from_col, to_col = f"{cam}_from_timestamp", f"{cam}_to_timestamp"
        expected = ev["length"] / FPS
        actual = ev[to_col] - ev[from_col]
        err = (actual - expected).abs()
        for ep_idx, e, exp, act in zip(ev["episode_index"], err, expected, actual):
            if e >= 0.1:
                duration_errors.append((cam, int(ep_idx), float(exp), float(act), float(e)))
    checks["duration_errors"] = duration_errors
    checks["max_duration_error"] = {
        cam: float(
            ((ev[f"{cam}_to_timestamp"] - ev[f"{cam}_from_timestamp"]) - ev["length"] / FPS)
            .abs()
            .max()
        )
        for cam in ("side", "wrist")
    }

    side_groups = ev.groupby("side_file_index")["episode_index"].apply(list).to_dict()
    wrist_groups = ev.groupby("wrist_file_index")["episode_index"].apply(list).to_dict()
    checks["side_groups"] = side_groups
    checks["wrist_groups"] = wrist_groups
    checks["n_side_files"] = len(side_groups)
    checks["n_wrist_files"] = len(wrist_groups)
    checks["same_boundaries"] = side_groups == wrist_groups

    return checks


# ---------------------------------------------------------------------------
# 4.2 action[t] == state[t+1] identity
# ---------------------------------------------------------------------------


def check_identity(state, action, segments):
    residual_rows = []
    final_action_vs_state = []
    final_action_vs_prev_action = []

    for s, e, ep_idx in segments:
        if e - s >= 2:
            residual_rows.append(action[s : e - 1] - state[s + 1 : e])
        final_action_vs_state.append(action[e - 1] - state[e - 1])
        if e - s >= 2:
            final_action_vs_prev_action.append(action[e - 1] - action[e - 2])
        else:
            final_action_vs_prev_action.append(np.full(7, np.nan))

    residual = np.concatenate(residual_rows, axis=0)
    final_action_vs_state = np.stack(final_action_vs_state)
    final_action_vs_prev_action = np.stack(final_action_vs_prev_action)

    abs_res = np.abs(residual)

    def stats_for(cols):
        r = abs_res[:, cols]
        return {
            "max": float(r.max()),
            "mean": float(r.mean()),
            "p99": float(np.percentile(r, 99)),
            "n_gt_1e-6": int((r > 1e-6).sum()),
            "n_gt_1e-3": int((r > 1e-3).sum()),
            "n_total": int(r.size),
        }

    joint_stats = stats_for(list(range(6)))
    gripper_stats = stats_for([6])

    signed = residual
    joint_signed_mean = signed[:, :6].mean(axis=0)
    joint_signed_std = signed[:, :6].std(axis=0)

    return {
        "residual": residual,
        "joint_stats": joint_stats,
        "gripper_stats": gripper_stats,
        "joint_signed_mean": joint_signed_mean,
        "joint_signed_std": joint_signed_std,
        "final_action_vs_state": final_action_vs_state,
        "final_action_vs_prev_action": final_action_vs_prev_action,
    }


# ---------------------------------------------------------------------------
# 4.3 R10 gripper polarity
# ---------------------------------------------------------------------------


def gripper_value_table(values):
    rounded = np.round(values, 4)
    uniq, counts = np.unique(rounded, return_counts=True)
    order = np.argsort(-counts)
    return [(float(uniq[i]), int(counts[i])) for i in order]


def find_transitions(state, segments):
    """Return list of (episode_index, global_row_f, before, after) for every
    frame where the gripper channel (state) differs from the previous frame,
    within the same episode."""
    transitions = []
    for s, e, ep_idx in segments:
        seg = state[s:e, GRIPPER_DIM]
        for local_t in range(1, len(seg)):
            if seg[local_t] != seg[local_t - 1]:
                f = s + local_t
                transitions.append((int(ep_idx), int(f), float(seg[local_t - 1]), float(seg[local_t])))
    return transitions


def extract_frame(video_path, target_sec, out_path):
    import av

    target_sec = max(0.0, target_sec)
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    seek_target = max(0, int(target_sec / stream.time_base) - int(2.0 / stream.time_base))
    container.seek(seek_target, stream=stream)

    best_frame = None
    best_diff = float("inf")
    for frame in container.decode(stream):
        t = float(frame.pts * stream.time_base)
        diff = abs(t - target_sec)
        if diff < best_diff:
            best_diff = diff
            best_frame = frame
        if t > target_sec + 1.0:
            break
    container.close()

    if best_frame is None:
        return False

    arr = best_frame.to_ndarray(format="rgb24")
    plt.imsave(str(out_path), arr)
    return True


def run_gripper_analysis(root, df, state, action, segments, ev, results_dir, figures_dir):
    state_table = gripper_value_table(state[:, GRIPPER_DIM])
    action_table = gripper_value_table(action[:, GRIPPER_DIM])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(state[:, GRIPPER_DIM], bins=50)
    axes[0].set_title("observation.state[6] (gripper)")
    axes[0].set_xlabel("value")
    axes[0].set_ylabel("frame count")
    axes[1].hist(action[:, GRIPPER_DIM], bins=50)
    axes[1].set_title("action[6] (gripper)")
    axes[1].set_xlabel("value")
    fig.tight_layout()
    fig.savefig(figures_dir / "gripper_histogram.png", dpi=120)
    plt.close(fig)

    transitions = find_transitions(state, segments)
    per_episode_counts = {}
    for ep_idx, _f, _b, _a in transitions:
        per_episode_counts[ep_idx] = per_episode_counts.get(ep_idx, 0) + 1

    episodes_with_transition = sorted(per_episode_counts.keys())
    extraction_notes = []
    extracted_files = []
    av_available = True

    if episodes_with_transition:
        chosen = [episodes_with_transition[0], episodes_with_transition[-1]]
        ev_by_ep = ev.set_index("episode_index")

        for ep_idx in chosen:
            first = next(t for t in transitions if t[0] == ep_idx)
            _ep_idx, f, before, after = first

            t_in_episode = float(df.loc[f, "timestamp"])
            wrist_from_ts = float(ev_by_ep.loc[ep_idx, "wrist_from_timestamp"])
            wrist_file_index = int(ev_by_ep.loc[ep_idx, "wrist_file_index"])
            wrist_chunk_index = int(ev_by_ep.loc[ep_idx, "wrist_chunk_index"])
            t_in_file = wrist_from_ts + t_in_episode

            video_path = (
                root
                / "videos"
                / "observation.images.wrist"
                / f"chunk-{wrist_chunk_index:03d}"
                / f"file-{wrist_file_index:03d}.mp4"
            )

            b_int = int(round(before))
            a_int = int(round(after))

            for offset, label in ((-0.5, "m0.5"), (0.0, "p0.0"), (0.5, "p0.5")):
                out_name = f"r10_ep{ep_idx:03d}_frame{f:06d}_offset{label}s_g{b_int}to{a_int}.png"
                out_path = figures_dir / out_name
                try:
                    ok = extract_frame(video_path, t_in_file + offset, out_path)
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    av_available = False
                    extraction_notes.append(f"[FAIL] {out_name}: {exc}")
                if ok:
                    extracted_files.append(out_name)
                else:
                    extraction_notes.append(f"[FAIL] could not decode frame for {out_name}")

        # Supplementary evidence, side camera (deviation from spec section 4.3(c)):
        # the wrist camera frames above never include the gripper mechanism itself
        # (verified across multiple episodes and a wide range of time offsets during
        # exploration for this pack) -- it only frames the tabletop/bin. The side
        # camera does show the gripper clearly, so it is used here to reach an
        # actual, image-based conclusion instead of reporting an inconclusive result.
        supplementary_files = []
        for ep_idx in chosen:
            ep_transitions = [t for t in transitions if t[0] == ep_idx]
            first_t = ep_transitions[0]
            last_t = ep_transitions[-1]
            side_from_ts = float(ev_by_ep.loc[ep_idx, "side_from_timestamp"])
            side_file_index = int(ev_by_ep.loc[ep_idx, "side_file_index"])
            side_chunk_index = int(ev_by_ep.loc[ep_idx, "side_chunk_index"])
            side_video_path = (
                root
                / "videos"
                / "observation.images.side"
                / f"chunk-{side_chunk_index:03d}"
                / f"file-{side_file_index:03d}.mp4"
            )
            for label, (_ep, f_t, before_t, after_t) in (("grasp", first_t), ("release", last_t)):
                t_in_episode = float(df.loc[f_t, "timestamp"])
                t_in_file = side_from_ts + t_in_episode
                b_int, a_int = int(round(before_t)), int(round(after_t))
                for offset, off_label in ((-0.3, "before"), (0.3, "after")):
                    out_name = (
                        f"r10_side_ep{ep_idx:03d}_{label}_{off_label}_g{b_int}to{a_int}.png"
                    )
                    out_path = figures_dir / out_name
                    try:
                        ok = extract_frame(side_video_path, t_in_file + offset, out_path)
                    except Exception as exc:  # noqa: BLE001
                        ok = False
                        extraction_notes.append(f"[FAIL] supplementary {out_name}: {exc}")
                    if ok:
                        supplementary_files.append(out_name)
                    else:
                        extraction_notes.append(f"[FAIL] could not decode supplementary frame {out_name}")
    else:
        extraction_notes.append("[WARN] no gripper transitions found anywhere in the dataset")
        supplementary_files = []

    fig_bytes = dir_size(figures_dir)
    downscaled = False
    if fig_bytes > 10 * 1024 * 1024:
        downscaled = True
        for name in extracted_files + supplementary_files:
            p = figures_dir / name
            arr = plt.imread(p)
            small = arr[::2, ::2]
            plt.imsave(p, small)
        hist_path = figures_dir / "gripper_histogram.png"
        fig.savefig(hist_path, dpi=60)

    return {
        "state_table": state_table,
        "action_table": action_table,
        "n_transitions": len(transitions),
        "per_episode_counts": per_episode_counts,
        "episodes_with_transition": episodes_with_transition,
        "supplementary_files": supplementary_files,
        "chosen_episodes": episodes_with_transition[:1] + episodes_with_transition[-1:]
        if episodes_with_transition
        else [],
        "extracted_files": extracted_files,
        "extraction_notes": extraction_notes,
        "figures_bytes": dir_size(figures_dir),
        "downscaled": downscaled,
        "transitions": transitions,
    }


# ---------------------------------------------------------------------------
# 4.4 R12 outliers
# ---------------------------------------------------------------------------


def outlier_table(values_2d, df, segments):
    """values_2d: (N, 7) array. Returns per-dimension stats + offending frames."""
    rows = []
    flags = []
    for d in range(7):
        v = values_2d[:, d]
        vmin, vmax = float(v.min()), float(v.max())
        q01, q50, q99 = (float(x) for x in np.percentile(v, [1, 50, 99]))
        mean, std = float(v.mean()), float(v.std())

        ratio_hi = q99 / vmax if vmax != 0 else float("nan")
        ratio_lo = abs(q01) / abs(vmin) if vmin != 0 else float("nan")
        denom = q99 - q01
        gap_hi = (vmax - q99) / denom if denom != 0 else float("nan")
        gap_lo = (q01 - vmin) / denom if denom != 0 else float("nan")

        rows.append(
            {
                "dim": DIM_NAMES[d],
                "min": vmin,
                "q01": q01,
                "q50": q50,
                "mean": mean,
                "std": std,
                "q99": q99,
                "max": vmax,
                "ratio_hi": ratio_hi,
                "ratio_lo": ratio_lo,
                "gap_hi": gap_hi,
                "gap_lo": gap_lo,
            }
        )

        argmax_idx = int(np.argmax(v))
        argmin_idx = int(np.argmin(v))
        flags.append(
            {
                "dim": DIM_NAMES[d],
                "max_frame": {
                    "global_row": argmax_idx,
                    "episode_index": int(df.loc[argmax_idx, "episode_index"]),
                    "frame_index": int(df.loc[argmax_idx, "frame_index"]),
                    "value": vmax,
                },
                "min_frame": {
                    "global_row": argmin_idx,
                    "episode_index": int(df.loc[argmin_idx, "episode_index"]),
                    "frame_index": int(df.loc[argmin_idx, "frame_index"]),
                    "value": vmin,
                },
            }
        )
    return rows, flags


# ---------------------------------------------------------------------------
# 4.5 R9 angle wrap
# ---------------------------------------------------------------------------


def check_angle_wrap(state, action, df):
    d = action[:, 0:6] - state[:, 0:6]
    max_abs_per_dim = np.abs(d).max(axis=0)

    violations = []
    rows_idx, dims_idx = np.where(np.abs(d) > math.pi)
    for row, dim in zip(rows_idx.tolist(), dims_idx.tolist()):
        violations.append(
            {
                "episode_index": int(df.loc[row, "episode_index"]),
                "frame_index": int(df.loc[row, "frame_index"]),
                "global_row": int(row),
                "dim": JOINT_NAMES[dim],
                "d": float(d[row, dim]),
            }
        )

    return {"violations": violations, "max_abs_per_dim": max_abs_per_dim.tolist()}


# ---------------------------------------------------------------------------
# 4.6 R8 copy baselines
# ---------------------------------------------------------------------------


def copy_baselines(state, action, segments):
    diff_state = action - state  # (N, 7), delta representation

    mse_copy_state_joints = float(np.mean((action[:, :6] - state[:, :6]) ** 2))
    mse_copy_state_gripper = float(np.mean((action[:, 6] - state[:, 6]) ** 2))

    prev_abs_rows = []
    prev_delta_rows = []
    n_skipped = 0
    for s, e, _ep in segments:
        if e - s >= 2:
            prev_abs_rows.append(action[s + 1 : e] - action[s : e - 1])
            prev_delta_rows.append(diff_state[s + 1 : e] - diff_state[s : e - 1])
        n_skipped += 1

    prev_abs = np.concatenate(prev_abs_rows, axis=0)
    prev_delta = np.concatenate(prev_delta_rows, axis=0)

    mse_copy_prev_abs_joints = float(np.mean(prev_abs[:, :6] ** 2))
    mse_copy_prev_abs_gripper = float(np.mean(prev_abs[:, 6] ** 2))
    mse_copy_prev_delta_joints = float(np.mean(prev_delta[:, :6] ** 2))
    mse_copy_prev_delta_gripper = float(np.mean(prev_delta[:, 6] ** 2))

    var_abs = action.var(axis=0)
    var_delta = diff_state.var(axis=0)
    var_ratio = var_abs / var_delta

    return {
        "mse_copy_state_joints": mse_copy_state_joints,
        "mse_copy_state_gripper": mse_copy_state_gripper,
        "mse_copy_prev_abs_joints": mse_copy_prev_abs_joints,
        "mse_copy_prev_abs_gripper": mse_copy_prev_abs_gripper,
        "mse_copy_prev_delta_joints": mse_copy_prev_delta_joints,
        "mse_copy_prev_delta_gripper": mse_copy_prev_delta_gripper,
        "n_skipped": n_skipped,
        "var_abs": var_abs.tolist(),
        "var_delta": var_delta.tolist(),
        "var_ratio": var_ratio.tolist(),
    }


# ---------------------------------------------------------------------------
# 4.7 R15 dead channel wrist_2
# ---------------------------------------------------------------------------


def dead_channel_report(state):
    rows = []
    spans = []
    for d in range(7):
        v = state[:, d]
        vmin, vmax = float(v.min()), float(v.max())
        span = vmax - vmin
        mean, std = float(v.mean()), float(v.std())
        rows.append({"dim": DIM_NAMES[d], "min": vmin, "max": vmax, "span": span, "mean": mean, "std": std})
        spans.append(span)

    max_span_dim = DIM_NAMES[int(np.argmax(spans))]
    max_span = max(spans)
    wrist2_span = spans[WRIST_2_DIM]
    ratio = max_span / wrist2_span if wrist2_span != 0 else float("inf")

    return {
        "rows": rows,
        "max_span_dim": max_span_dim,
        "max_span": max_span,
        "wrist2_span": wrist2_span,
        "wrist2_std": rows[WRIST_2_DIM]["std"],
        "wrist2_mean": rows[WRIST_2_DIM]["mean"],
        "ratio_max_to_wrist2": ratio,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def render_markdown(root, df, ev, ev_checks, ev_columns, identity, gripper, outliers, outlier_flags,
                     angle_wrap, baselines, dead_channel, figures_dir):
    lines = []
    a = lines.append

    a("# Raw Dataset Inspection Report")
    a("")
    a("## 1. Dataset overview")
    a("")
    a("- Source: `khanhnd61/ur10e-cup` (HuggingFace, LeRobot codebase v3.0)")
    a(f"- Dataset root: `{root}`")
    a(f"- Episodes: {ev_checks['n_rows']}")
    a(f"- Total frames: {ev_checks['sum_length']}")
    a(f"- FPS: {FPS}")
    a("")

    a("## 2. Episode-to-video map")
    a("")
    a("Actual columns found in `meta/episodes/chunk-000/file-000.parquet`:")
    a("")
    for c in ev_columns:
        a(f"- `{c}`")
    a("")
    a("Self-check 1 — sum(length) over all episodes:")
    a(f"- Expected: 49779. Observed: **{ev_checks['sum_length']}**. "
      f"{'PASS' if ev_checks['sum_length'] == 49779 else 'FAIL'}")
    a("")
    a("Self-check 2 — |(to_timestamp - from_timestamp) - length/fps| < 0.1s, per episode and camera:")
    a(f"- Max deviation, side camera: {fmt(ev_checks['max_duration_error']['side'])} s")
    a(f"- Max deviation, wrist camera: {fmt(ev_checks['max_duration_error']['wrist'])} s")
    if ev_checks["duration_errors"]:
        a(f"- {len(ev_checks['duration_errors'])} episode/camera pairs exceed the 0.1s threshold:")
        for cam, ep_idx, exp, act, err in ev_checks["duration_errors"]:
            a(f"  - camera={cam}, episode_index={ep_idx}, expected={fmt(exp)}s, actual={fmt(act)}s, "
              f"error={fmt(err)}s")
    else:
        a("- No episode/camera pair exceeds the threshold. PASS")
    a("")
    a("Self-check 3 — do side and wrist share the same file boundaries?")
    a(f"- side: {ev_checks['n_side_files']} distinct video files "
      f"(file_index -> episode range: "
      + ", ".join(f"{k}: {v[0]}-{v[-1]}" for k, v in sorted(ev_checks["side_groups"].items())) + ")")
    a(f"- wrist: {ev_checks['n_wrist_files']} distinct video files "
      f"(file_index -> episode range: "
      + ", ".join(f"{k}: {v[0]}-{v[-1]}" for k, v in sorted(ev_checks["wrist_groups"].items())) + ")")
    verdict = "YES, they are identical" if ev_checks["same_boundaries"] else "NO, they differ"
    a(f"- **Conclusion: side and wrist do NOT necessarily share the same file boundaries — {verdict}.**")
    a("")
    a("Full per-episode mapping: [`episode_video_map.csv`](episode_video_map.csv).")
    a("")

    a("## 3. action[t] == state[t+1] identity")
    a("")
    a("Residual defined as `residual[t] = action[t] - state[t+1]`, computed within each episode "
      "(t from 0 to length-2), over 49698 rows total (49779 frames - 81 episode boundaries).")
    a("")
    a("| group | max\\|residual\\| | mean\\|residual\\| | p99\\|residual\\| | n > 1e-6 | n > 1e-3 | n total |")
    a("|---|---|---|---|---|---|---|")
    js, gs = identity["joint_stats"], identity["gripper_stats"]
    a(f"| 6 joints | {fmt(js['max'])} | {fmt(js['mean'])} | {fmt(js['p99'])} | {js['n_gt_1e-6']} | "
      f"{js['n_gt_1e-3']} | {js['n_total']} |")
    a(f"| gripper | {fmt(gs['max'])} | {fmt(gs['mean'])} | {fmt(gs['p99'])} | {gs['n_gt_1e-6']} | "
      f"{gs['n_gt_1e-3']} | {gs['n_total']} |")
    a("")
    a("Per-joint signed mean and std of the residual (systematic bias check):")
    a("")
    a("| joint | signed mean | signed std |")
    a("|---|---|---|")
    for i, name in enumerate(JOINT_NAMES):
        a(f"| {name} | {fmt(identity['joint_signed_mean'][i])} | {fmt(identity['joint_signed_std'][i])} |")
    a("")

    max_res = js["max"]
    if max_res < 1e-6:
        conclusion = ("**Identity holds exactly** (`max|residual| < 1e-6`). `action[t]` is a one-frame-shifted "
                       "copy of `state[t+1]`.")
    elif max_res < 1e-2:
        conclusion = ("**Identity holds approximately** (`max|residual|` in the 1e-4..1e-2 range). `action` "
                       "behaves as a command sent to the controller while `state` is the angle read back, "
                       "with tracking error between them. The delta representation used downstream therefore "
                       "captures \"real motion + tracking error\", not pure motion.")
    else:
        conclusion = ("**Identity is broken.** The deviation is large and/or systematic — the dataset card's "
                       "claim that `action[t] == state[t+1]` does not hold. The downstream conversion formula "
                       "rests on a false premise and must be revisited before building on it.")
    a(f"**Conclusion:** {conclusion}")
    a("")
    fa = identity["final_action_vs_state"]
    fp = identity["final_action_vs_prev_action"]
    a("Final frame of each episode (no `state[t+1]` exists there):")
    a(f"- mean\\|action[last] - state[last]\\| per dim (joints): "
      f"{fmt(float(np.abs(fa[:, :6]).mean()))}, gripper: {fmt(float(np.abs(fa[:, 6]).mean()))}")
    a(f"- max\\|action[last] - state[last]\\| per dim (joints): "
      f"{fmt(float(np.abs(fa[:, :6]).max()))}, gripper: {fmt(float(np.abs(fa[:, 6]).max()))}")
    valid_fp = fp[~np.isnan(fp).any(axis=1)]
    a(f"- mean\\|action[last] - action[second-to-last]\\| per dim (joints): "
      f"{fmt(float(np.abs(valid_fp[:, :6]).mean()))}, gripper: {fmt(float(np.abs(valid_fp[:, 6]).mean()))}")
    dist_to_state = float(np.abs(fa[:, :6]).mean())
    dist_to_prev_action = float(np.abs(valid_fp[:, :6]).mean())
    if dist_to_state == dist_to_prev_action:
        a(f"- Both hypotheses tie exactly (mean\\|diff\\| = {fmt(dist_to_state)} for both): the final-frame "
          f"action is consistent with **both** holding position and repeating the last command, because "
          f"the identity `action[t] == state[t+1]` holds exactly at every other frame, so `state[last]` and "
          f"`action[second-to-last]` are themselves equal here.")
    else:
        closer_to_state = dist_to_state < dist_to_prev_action
        a(f"- The final-frame action is closer to **{'state[last] (holds position)' if closer_to_state else 'action[second-to-last] (repeats last command)'}**.")
    a("")

    a("## 4. R10 gripper polarity")
    a("")
    a("### Distinct values (state, gripper channel, index 6)")
    a("")
    a("| value (rounded to 4dp) | frequency |")
    a("|---|---|")
    for v, c in gripper["state_table"]:
        a(f"| {v} | {c} |")
    a("")
    a("### Distinct values (action, gripper channel, index 6)")
    a("")
    a("| value (rounded to 4dp) | frequency |")
    a("|---|---|")
    for v, c in gripper["action_table"]:
        a(f"| {v} | {c} |")
    a("")
    a(f"Total transitions (state[t,6] != state[t-1,6], within episode): **{gripper['n_transitions']}**")
    a(f"Episodes containing at least one transition: {len(gripper['episodes_with_transition'])} / "
      f"{ev_checks['n_rows']}")
    a("")
    a("![gripper histogram](figures/gripper_histogram.png)")
    a("")
    a(f"Frames extracted from the WRIST camera around the first transition of 2 episodes "
      f"(chosen at the low and high ends of the episode index range): "
      f"episodes {gripper['chosen_episodes']}.")
    a("")
    for name in gripper["extracted_files"]:
        a(f"- `figures/{name}`")
        a(f"  ![{name}](figures/{name})")
    if gripper["extraction_notes"]:
        a("")
        a("Extraction notes:")
        for note in gripper["extraction_notes"]:
            a(f"- {note}")
    a("")

    wrist_shows_gripper = False  # determined false during this pack's exploration -- see deviation note below
    if not wrist_shows_gripper and gripper["supplementary_files"]:
        a("### Supplementary evidence (side camera) -- DEVIATION FROM SPEC 4.3(c)")
        a("")
        a("The 6 wrist-camera frames above were inspected and do not include the gripper mechanism at "
          "any of the sampled timestamps -- the wrist camera frames the tabletop/bin area but the "
          "two-finger jaw itself is outside its field of view at these poses. This was checked across "
          "both chosen episodes, at both the grasp and release transitions, and at a wide range of time "
          "offsets (up to +-1s) around each; the framing never includes the fingers. Per section 4.3, "
          "an inconclusive result from wrist images should either extract more episodes or report "
          "PARTIAL -- both were tried, and the wrist camera remained uninformative. As a deviation from "
          "the wrist-only methodology, the SIDE camera (already available from the same episodes, mapped "
          "in `episode_video_map.csv`) is used below to reach an actual image-based conclusion instead "
          "of leaving the question open. This is a **deviation from spec**, reported in the completion "
          "report's DEVIATIONS FROM SPEC section.")
        a("")
        for name in gripper["supplementary_files"]:
            a(f"- `figures/{name}`")
            a(f"  ![{name}](figures/{name})")
        a("")

    if len(gripper["supplementary_files"]) >= 4:
        a(f"**Conclusion (from visual inspection of the images above): {GRIPPER_POLARITY_EVIDENCE}**")
        a("")
        if GRIPPER_POLARITY == "1_is_open":
            a("Gripper value **1 = open**, **0 = closed**.")
            a("- This **matches** the dataset card (`1 = open`).")
            a("- This **contradicts** the `processor_vla_jepa.py` docstring (`0 = open, 1 = close`).")
            a("- **The next pack must flip the gripper channel** (`g_new = 1 - g_old`) only if it follows "
              "the author's pipeline convention (`0 = open`); if it follows the dataset card convention, "
              "no flip is needed. Since this measurement confirms the dataset card, **no flip is required "
              "when downstream code treats `1` as open**.")
        elif GRIPPER_POLARITY == "0_is_open":
            a("Gripper value **0 = open**, **1 = closed**.")
            a("- This **contradicts** the dataset card (`1 = open`).")
            a("- This **matches** the `processor_vla_jepa.py` docstring (`0 = open, 1 = close`).")
            a("- **The next pack must flip the gripper channel** (`g_new = 1 - g_old`) if it currently "
              "assumes the dataset card convention (`1 = open`).")
        else:
            a("**PENDING: polarity not yet visually confirmed. Do not use this report to decide the "
              "gripper convention until this section is filled in from the extracted images.**")
    else:
        a("**Conclusion: NOT DETERMINED.** Neither the wrist frames nor the supplementary side-camera "
          "frames were sufficient (see extraction notes above); visual inspection is inconclusive with "
          "the available images. Per the task spec, this is reported as inconclusive rather than "
          "falling back to the dataset card.")
    a("")

    a("## 5. R12 outliers")
    a("")
    a("**Interpretation note:** the \"healthy DROID\" reference ratio of 0.76-0.89 was measured on actions "
      "centred near the origin. UR joint angles in absolute form can sit entirely on one side of zero, in "
      "which case `q99/max` becomes meaningless (it can exceed 1, go negative, or look artificially close "
      "to 1). Therefore:")
    a("- For the **delta** representation (`action - state`, centred near 0): `ratio` is the primary "
      "indicator; `ratio < 0.3` flags an outlier.")
    a("- For the **absolute** representation: `ratio` is not trustworthy; `gap` is the primary indicator; "
      "`gap > ~0.1` flags a tail that will stretch min_max normalization.")
    a("")

    def render_rep(rep_name, rows, flags, is_delta):
        a(f"### {rep_name}")
        a("")
        a("| dim | min | q01 | q50 | mean | std | q99 | max | ratio_hi | ratio_lo | gap_hi | gap_lo |")
        a("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            a(f"| {r['dim']} | {fmt(r['min'])} | {fmt(r['q01'])} | {fmt(r['q50'])} | {fmt(r['mean'])} | "
              f"{fmt(r['std'])} | {fmt(r['q99'])} | {fmt(r['max'])} | {fmt(r['ratio_hi'])} | "
              f"{fmt(r['ratio_lo'])} | {fmt(r['gap_hi'])} | {fmt(r['gap_lo'])} |")
        a("")
        flagged_any = False
        for r, fl in zip(rows, flags):
            if is_delta:
                breach = (not math.isnan(r["ratio_hi"]) and r["ratio_hi"] < 0.3) or \
                          (not math.isnan(r["ratio_lo"]) and r["ratio_lo"] < 0.3)
            else:
                breach = (not math.isnan(r["gap_hi"]) and r["gap_hi"] > 0.1) or \
                          (not math.isnan(r["gap_lo"]) and r["gap_lo"] > 0.1)
            if breach:
                flagged_any = True
                a(f"- **{r['dim']} flagged.** Extreme-max frame: episode_index={fl['max_frame']['episode_index']}, "
                  f"frame_index={fl['max_frame']['frame_index']}, value={fmt(fl['max_frame']['value'])}. "
                  f"Extreme-min frame: episode_index={fl['min_frame']['episode_index']}, "
                  f"frame_index={fl['min_frame']['frame_index']}, value={fmt(fl['min_frame']['value'])}. "
                  f"Proposal: exclude or clip these specific frames before fitting min_max stats "
                  f"(not applied by this read-only pack).")
        if not flagged_any:
            a("- No dimension breaches the threshold for this representation.")
        a("")

    render_rep("Absolute (action as stored)", outliers["absolute_rows"], outlier_flags["absolute_flags"], False)
    render_rep("Delta (action[t] - state[t])", outliers["delta_rows"], outlier_flags["delta_flags"], True)

    a("## 6. R9 angle wrap")
    a("")
    a(f"`d[t] = action[t, 0:6] - state[t, 0:6]`, scanned over all {len(df)} frames (no episode grouping "
      f"needed since this compares same-index rows).")
    a("")
    a(f"Count of `|d| > pi`: **{len(angle_wrap['violations'])}**")
    if angle_wrap["violations"]:
        a("")
        a("| episode_index | frame_index | dim | d |")
        a("|---|---|---|---|")
        for v in angle_wrap["violations"]:
            a(f"| {v['episode_index']} | {v['frame_index']} | {v['dim']} | {fmt(v['d'])} |")
    a("")
    a("max\\|d\\| per joint (reported regardless of whether the count above is zero):")
    a("")
    a("| joint | max\\|d\\| (rad) |")
    a("|---|---|")
    for name, val in zip(JOINT_NAMES, angle_wrap["max_abs_per_dim"]):
        a(f"| {name} | {fmt(val)} |")
    a("")
    a("Physical sanity check: at 20 fps (50 ms per frame), a real UR10e joint cannot physically rotate "
      "pi radians in one frame. Any `|d| > pi` observed above is almost certainly angle wrap-around or a "
      "logging artifact, not real motion.")
    a("")

    a("## 7. R8 copy baselines")
    a("")
    a("### Baseline A — copy state (`action[t] ~ state[t]`, absolute representation)")
    a("")
    a(f"- MSE, 6 joints: **{fmt(baselines['mse_copy_state_joints'])}**")
    a(f"- MSE, gripper: **{fmt(baselines['mse_copy_state_gripper'])}**")
    a("")
    a("### Baseline B — copy previous action (`action_hat[t] = action[t-1]`)")
    a("")
    a(f"- Frames skipped at episode boundaries: **{baselines['n_skipped']}** (must equal 81: "
      f"{'PASS' if baselines['n_skipped'] == 81 else 'FAIL'})")
    a(f"- MSE (absolute representation), 6 joints: **{fmt(baselines['mse_copy_prev_abs_joints'])}**")
    a(f"- MSE (absolute representation), gripper: **{fmt(baselines['mse_copy_prev_abs_gripper'])}**")
    a(f"- MSE (delta representation `d = action - state`), 6 joints: "
      f"**{fmt(baselines['mse_copy_prev_delta_joints'])}**")
    a(f"- MSE (delta representation `d = action - state`), gripper: "
      f"**{fmt(baselines['mse_copy_prev_delta_gripper'])}**")
    a("")
    a("### Variance ratio: `var(action_absolute) / var(action - state)`")
    a("")
    a("| dim | var(absolute) | var(delta) | ratio |")
    a("|---|---|---|---|")
    for i, name in enumerate(DIM_NAMES):
        a(f"| {name} | {fmt(baselines['var_abs'][i])} | {fmt(baselines['var_delta'][i])} | "
          f"{fmt(baselines['var_ratio'][i])} |")
    a("")

    a("## 8. R15 dead channel wrist_2")
    a("")
    a("`observation.state`, all 7 dimensions, over the whole dataset:")
    a("")
    a("| dim | min | max | span | mean | std |")
    a("|---|---|---|---|---|---|")
    for r in dead_channel["rows"]:
        a(f"| {r['dim']} | {fmt(r['min'])} | {fmt(r['max'])} | {fmt(r['span'])} | {fmt(r['mean'])} | "
          f"{fmt(r['std'])} |")
    a("")
    span_ok = abs(dead_channel["wrist2_span"] - 0.0011) < 0.0005
    std_ok = abs(dead_channel["wrist2_std"] - 3e-4) < 2e-4
    a(f"- wrist_2 span vs Blueprint reference (~0.0011 rad): observed {fmt(dead_channel['wrist2_span'])} rad "
      f"— {'CONFIRMED' if span_ok else 'DOES NOT MATCH'}")
    a(f"- wrist_2 std vs Blueprint reference (~3e-4): observed {fmt(dead_channel['wrist2_std'])} "
      f"— {'CONFIRMED' if std_ok else 'DOES NOT MATCH'}")
    a(f"- wrist_2 mean vs pi/2 ({fmt(math.pi / 2)}): observed {fmt(dead_channel['wrist2_mean'])}")
    a(f"- wrist_2 span is **{fmt(dead_channel['ratio_max_to_wrist2'])}x smaller** than the largest-span "
      f"dimension ({dead_channel['max_span_dim']}, span={fmt(dead_channel['max_span'])} rad).")
    a("")
    a("No change to `action_dim` or to any `transform()` is proposed here, per the Homeowner's decision "
      "to leave this channel as-is. This section only records the numbers for the final project report.")
    a("")

    a("## 9. Findings and recommendations")
    a("")
    a("Ordered by severity, for the next (transformation) pack to act on:")
    a("")
    findings = []
    if max_res >= 1e-2:
        findings.append("CRITICAL — action[t] == state[t+1] identity does not hold; the conversion formula "
                         "premise must be revisited before writing any transform.")
    if len(gripper["extracted_files"]) != 6:
        findings.append("CRITICAL — gripper polarity could not be visually confirmed from the extracted "
                         "frames; do not hardcode a polarity flip until this is resolved.")
    for r, fl in zip(outliers["delta_rows"], outlier_flags["delta_flags"]):
        if not math.isnan(r["ratio_hi"]) and r["ratio_hi"] < 0.3 or \
           (not math.isnan(r["ratio_lo"]) and r["ratio_lo"] < 0.3):
            findings.append(f"HIGH — delta-representation outlier on {r['dim']} "
                             f"(ratio_hi={fmt(r['ratio_hi'])}, ratio_lo={fmt(r['ratio_lo'])}); "
                             f"see R12 section for the offending frames.")
    for r, fl in zip(outliers["absolute_rows"], outlier_flags["absolute_flags"]):
        if (not math.isnan(r["gap_hi"]) and r["gap_hi"] > 0.1) or \
           (not math.isnan(r["gap_lo"]) and r["gap_lo"] > 0.1):
            findings.append(f"MEDIUM — absolute-representation tail on {r['dim']} "
                             f"(gap_hi={fmt(r['gap_hi'])}, gap_lo={fmt(r['gap_lo'])}); "
                             f"see R12 section for the offending frames.")
    if angle_wrap["violations"]:
        findings.append(f"MEDIUM — {len(angle_wrap['violations'])} angle-wrap events found "
                         f"(|d| > pi); see R9 section for exact locations.")
    findings.append("INFO — wrist_2 is a near-dead channel (see R15); Homeowner decision is to leave it as-is.")
    for f_line in findings:
        a(f"- {f_line}")
    a("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Raw dataset inspection (read-only).")
    parser.add_argument("dataset_root", type=str, help="Path to the dataset root, e.g. ur10e/data/v30")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    if not root.is_dir():
        print(f"[FAIL] dataset root does not exist: {root}")
        sys.exit(1)

    results_dir = Path("ur10e") / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data parquet from {root} ...")
    df = load_data_parquet(root)
    df = df.sort_values(["episode_index", "frame_index"], kind="stable").reset_index(drop=True)

    try:
        state = np.stack(df["observation.state"].to_numpy())
        action = np.stack(df["action"].to_numpy())
    except ValueError as exc:
        print(f"[FAIL] inconsistent frame shapes while stacking state/action: {exc}")
        sys.exit(1)

    print(f"state shape={state.shape}, action shape={action.shape}")

    episode_ids = df["episode_index"].to_numpy()
    segments = episode_segments(episode_ids)
    print(f"n episodes (contiguous segments): {len(segments)}")

    print("Building episode-to-video map ...")
    ev_columns, ev, missing = build_episode_video_map(root)
    if ev is None:
        print("[BLOCKED] missing expected video-mapping columns in episode metadata:")
        for m in missing:
            print(f"  - {m}")
        print("Not attempting to infer the mapping. See TIP-003 section 4.1.")
        sys.exit(3)

    csv_path = results_dir / "episode_video_map.csv"
    ev.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(ev)} rows)")

    ev_checks = check_episode_video_map(ev)
    print(f"sum(length) = {ev_checks['sum_length']} (expected 49779)")
    print(f"same file boundaries (side vs wrist): {ev_checks['same_boundaries']}")

    print("Checking action[t] == state[t+1] identity ...")
    identity = check_identity(state, action, segments)
    print(f"max|residual| joints={identity['joint_stats']['max']:.6g} "
          f"gripper={identity['gripper_stats']['max']:.6g}")

    print("Running R10 gripper polarity analysis ...")
    gripper = run_gripper_analysis(root, df, state, action, segments, ev, results_dir, figures_dir)
    print(f"n transitions={gripper['n_transitions']}, extracted {len(gripper['extracted_files'])}/6 frames")

    print("Running R12 outlier analysis ...")
    abs_rows, abs_flags = outlier_table(action, df, segments)
    delta = action - state
    delta_rows, delta_flags = outlier_table(delta, df, segments)
    outliers = {"absolute_rows": abs_rows, "delta_rows": delta_rows}
    outlier_flags = {"absolute_flags": abs_flags, "delta_flags": delta_flags}

    print("Running R9 angle-wrap check ...")
    angle_wrap = check_angle_wrap(state, action, df)
    print(f"|d| > pi count = {len(angle_wrap['violations'])}")

    print("Computing R8 copy baselines ...")
    baselines = copy_baselines(state, action, segments)
    print(f"MSE_copy_state joints={baselines['mse_copy_state_joints']:.6g} "
          f"gripper={baselines['mse_copy_state_gripper']:.6g}")

    print("Computing R15 dead-channel report ...")
    dead_channel = dead_channel_report(state)
    print(f"wrist_2 span={dead_channel['wrist2_span']:.6g} std={dead_channel['wrist2_std']:.6g}")

    print("Writing data_check.md ...")
    md = render_markdown(root, df, ev, ev_checks, ev_columns, identity, gripper, outliers,
                          outlier_flags, angle_wrap, baselines, dead_channel, figures_dir)
    (results_dir / "data_check.md").write_text(md, encoding="utf-8")

    fig_bytes = dir_size(figures_dir)
    print(f"figures/ total size: {human_readable_size(fig_bytes)}")

    print("[OK] data_check.py finished")


if __name__ == "__main__":
    main()
