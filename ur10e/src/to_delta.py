"""Convert the ur10e-cup action representation to per-joint delta.

DROID-style layout: state stays absolute, action becomes delta relative to
the current state. Read-only on the source dataset -- writes a transformed
copy at the destination path and never touches the source.

What changes, per frame, within each episode (never across an episode
boundary):

    action_new[t, 0:6] = wrap_pi(action[t, 0:6] - state[t, 0:6])   # joints
    action_new[t, 6]   = action[t, 6]                              # gripper, untouched
    state_new[t, :]    = state[t, :]                                # untouched, all 7 dims

The gripper channel is neither delta'd nor polarity-flipped -- see
delta_conversion.md section 3 for why the flip once planned in the
Blueprint was dropped by the Homeowner.

Usage:
    python to_delta.py <src_root> <dst_root>
"""

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_dataset_v30 import load_data_parquet  # noqa: E402

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
DIM_NAMES = JOINT_NAMES + ["gripper"]
GRIPPER_DIM = 6
EXPECTED_GRIPPER_FREQ = {1.0: 25411, 0.0: 24368}
EXPECTED_N_ROWS = 49779
EXPECTED_N_EPISODES = 81
ID_COLUMNS = ["episode_index", "frame_index", "timestamp", "index", "task_index"]


def fmt(x, nd=10):
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return f"{float(x):.{nd}g}"


def episode_segments(episode_ids):
    """Contiguous (start, end, episode_index) blocks in the array's existing
    row order. Raises if an episode_index value reappears in a second,
    non-adjacent block -- that would mean the source rows are not grouped
    by episode, and this script must not silently reorder rows to fix it."""
    change_points = np.flatnonzero(np.diff(episode_ids) != 0) + 1
    starts = np.concatenate(([0], change_points))
    ends = np.concatenate((change_points, [len(episode_ids)]))
    episodes = episode_ids[starts]
    if len(np.unique(episodes)) != len(episodes):
        raise ValueError(
            "episode_index blocks are not contiguous in source row order -- "
            "refusing to reorder rows to fix this; see TIP-004 constraints"
        )
    return list(zip(starts.tolist(), ends.tolist(), episodes.tolist()))


def stats_row(values):
    vmin, vmax = float(values.min()), float(values.max())
    q01, q50, q99 = (float(x) for x in np.percentile(values, [1, 50, 99]))
    mean, std = float(values.mean()), float(values.std())
    ratio_hi = q99 / vmax if vmax != 0 else float("nan")
    ratio_lo = abs(q01) / abs(vmin) if vmin != 0 else float("nan")
    denom = q99 - q01
    gap_hi = (vmax - q99) / denom if denom != 0 else float("nan")
    gap_lo = (q01 - vmin) / denom if denom != 0 else float("nan")
    return {
        "min": vmin, "q01": q01, "q50": q50, "mean": mean, "std": std,
        "q99": q99, "max": vmax, "ratio_hi": ratio_hi, "ratio_lo": ratio_lo,
        "gap_hi": gap_hi, "gap_lo": gap_lo,
    }


def convert(df):
    state = np.stack(df["observation.state"].to_numpy())
    action = np.stack(df["action"].to_numpy())
    orig_dtype = state.dtype

    episode_ids = df["episode_index"].to_numpy()
    segments = episode_segments(episode_ids)

    state64 = state.astype(np.float64)
    action64 = action.astype(np.float64)

    raw_delta = action64[:, 0:6] - state64[:, 0:6]
    wrapped = (raw_delta + math.pi) % (2 * math.pi) - math.pi
    n_wrap_changed = int(np.sum(np.abs(wrapped - raw_delta) > 1e-12))

    action_new64 = np.empty_like(action64)
    action_new64[:, 0:6] = wrapped
    action_new64[:, 6] = action64[:, 6]  # gripper: pass-through, no delta, no flip

    state_new64 = state64.copy()  # unchanged, all 7 dims

    action_new = action_new64.astype(orig_dtype)
    state_new = state_new64.astype(orig_dtype)

    return {
        "segments": segments,
        "orig_dtype": orig_dtype,
        "n_wrap_changed": n_wrap_changed,
        "action_new": action_new,
        "state_new": state_new,
    }


def write_output(src_root, dst_root, df, action_new, state_new):
    data_dir = dst_root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)

    src_parquet_path = src_root / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(src_parquet_path)

    flat_action = action_new.astype(np.float32).ravel()
    action_values = pa.array(flat_action, type=pa.float32())
    action_col = pa.FixedSizeListArray.from_arrays(action_values, 7)

    action_field_idx = table.schema.get_field_index("action")
    new_table = table.set_column(
        action_field_idx, table.schema.field(action_field_idx), action_col
    )

    # observation.state is left byte-for-byte as read from source (untouched).
    out_parquet_path = data_dir / "file-000.parquet"
    pq.write_table(new_table, out_parquet_path)

    meta_dir = dst_root / "meta"
    (meta_dir / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for rel in ("info.json", "stats.json", "tasks.parquet"):
        shutil.copy2(src_root / "meta" / rel, meta_dir / rel)
    shutil.copy2(
        src_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet",
        meta_dir / "episodes" / "chunk-000" / "file-000.parquet",
    )

    return out_parquet_path, new_table.schema


def run_verifications(src_root, dst_root, src_df, result):
    checks = {}
    segments = result["segments"]
    action_new = result["action_new"]
    state_new = result["state_new"]

    src_state = np.stack(src_df["observation.state"].to_numpy()).astype(np.float64)
    src_action = np.stack(src_df["action"].to_numpy()).astype(np.float64)

    out_parquet_path = dst_root / "data" / "chunk-000" / "file-000.parquet"
    out_df = pd.read_parquet(out_parquet_path)
    written_action_new = np.stack(out_df["action"].to_numpy()).astype(np.float64)
    written_state_new = np.stack(out_df["observation.state"].to_numpy()).astype(np.float64)

    # V1 -- round-trip (tautological)
    v1_err = np.abs(src_state[:, 0:6] + written_action_new[:, 0:6] - src_action[:, 0:6])
    checks["V1"] = {"max_err": float(v1_err.max()), "pass": bool(v1_err.max() < 1e-6)}

    # V2 -- cross-derivation via state[t+1] (not tautological)
    v2_errs = []
    for s, e, _ep in segments:
        if e - s >= 2:
            derived = src_state[s + 1 : e, 0:6] - src_state[s : e - 1, 0:6]
            v2_errs.append(np.abs(written_action_new[s : e - 1, 0:6] - derived))
    v2_err = np.concatenate(v2_errs, axis=0)
    checks["V2"] = {"max_err": float(v2_err.max()), "pass": bool(v2_err.max() < 1e-9)}

    # V3 -- per-episode telescoping sum (not tautological)
    v3_errs = []
    for s, e, ep in segments:
        total = written_action_new[s:e, 0:6].sum(axis=0)
        expected = src_state[e - 1, 0:6] - src_state[s, 0:6]
        v3_errs.append((int(ep), float(np.abs(total - expected).max())))
    v3_max = max(err for _ep, err in v3_errs)
    checks["V3"] = {"max_err": v3_max, "per_episode": v3_errs, "pass": bool(v3_max < 1e-5)}

    # V4 -- final frame of each episode is exactly zero
    zero_mask = np.all(written_action_new[:, 0:6] == 0.0, axis=1)
    zero_rows = set(np.flatnonzero(zero_mask).tolist())
    final_rows = set(e - 1 for _s, e, _ep in segments)
    checks["V4"] = {
        "n_zero": len(zero_rows),
        "n_final": len(final_rows),
        "final_rows_all_zero": bool(final_rows <= zero_rows),
        "exact_match": bool(zero_rows == final_rows),
        "n_extra_zero_rows": len(zero_rows - final_rows),
        "pass": bool(len(zero_rows) == EXPECTED_N_EPISODES and zero_rows == final_rows),
    }

    # V5 -- angle-wrap step is a no-op
    checks["V5"] = {"n_changed": result["n_wrap_changed"], "pass": result["n_wrap_changed"] == 0}

    # V6 -- statistics agree with data_check.md's delta table
    v6_rows = {DIM_NAMES[d]: stats_row(written_action_new[:, d]) for d in range(7)}
    checks["V6"] = {"rows": v6_rows}

    # V7 -- gripper untouched
    action_gripper = written_action_new[:, 6]
    state_gripper = written_state_new[:, 6]
    action_vals, action_counts = np.unique(action_gripper, return_counts=True)
    state_vals, state_counts = np.unique(state_gripper, return_counts=True)
    action_freq = dict(zip(action_vals.tolist(), action_counts.tolist()))
    state_freq = dict(zip(state_vals.tolist(), state_counts.tolist()))
    action_matches_src = bool(np.array_equal(action_gripper, src_action[:, 6]))
    checks["V7"] = {
        "action_freq": action_freq,
        "state_freq": state_freq,
        "action_only_binary": bool(set(action_vals.tolist()) <= {0.0, 1.0}),
        "state_only_binary": bool(set(state_vals.tolist()) <= {0.0, 1.0}),
        "action_matches_source": action_matches_src,
        "pass": bool(
            set(action_vals.tolist()) <= {0.0, 1.0}
            and set(state_vals.tolist()) <= {0.0, 1.0}
            and action_freq == EXPECTED_GRIPPER_FREQ
            and state_freq == EXPECTED_GRIPPER_FREQ
            and action_matches_src
        ),
    }

    # V8 -- state joints untouched
    v8_equal = bool(np.array_equal(written_state_new[:, 0:6], src_state[:, 0:6]))
    checks["V8"] = {"pass": v8_equal}

    # V9 -- metadata preserved
    n_rows_ok = len(out_df) == EXPECTED_N_ROWS
    n_episodes_ok = out_df["episode_index"].nunique() == EXPECTED_N_EPISODES
    id_cols_match = all(
        (out_df[c].to_numpy() == src_df[c].to_numpy()).all() for c in ID_COLUMNS
    )
    columns_match = list(out_df.columns) == list(src_df.columns)
    checks["V9"] = {
        "n_rows": len(out_df),
        "n_episodes": int(out_df["episode_index"].nunique()),
        "id_cols_match": id_cols_match,
        "columns_match": columns_match,
        "pass": bool(n_rows_ok and n_episodes_ok and id_cols_match and columns_match),
    }

    return checks


DATA_CHECK_DELTA_TABLE = {
    "shoulder_pan": {"min": -0.0059999228, "q01": -0.0027999878, "q50": 0, "mean": 0.00081837515,
                      "std": 0.001700383, "q99": 0.0056999922, "max": 0.0062000751,
                      "ratio_hi": 0.91934243, "ratio_lo": 0.46667064, "gap_hi": 0.058833429, "gap_lo": 0.37646382},
    "shoulder_lift": {"min": -0.0086001158, "q01": -0.0057998896, "q50": 0, "mean": -0.00055019994,
                       "std": 0.0016825035, "q99": 0.0019000769, "max": 0.0030999184,
                       "ratio_hi": 0.61294416, "ratio_lo": 0.67439668, "gap_hi": 0.15582425, "gap_lo": 0.36366733},
    "elbow": {"min": -0.0091000795, "q01": -0.0061000586, "q50": 0, "mean": 0.00020816202,
              "std": 0.0029872514, "q99": 0.0072000027, "max": 0.011100054,
              "ratio_hi": 0.64864575, "ratio_lo": 0.67033025, "gap_hi": 0.29323558, "gap_lo": 0.22556445},
    "wrist_1": {"min": -0.0089000463, "q01": -0.0058000088, "q50": 0, "mean": 0.00033694721,
                "std": 0.0029332421, "q99": 0.0082000494, "max": 0.0091999769,
                "ratio_hi": 0.89131195, "ratio_lo": 0.65168299, "gap_hi": 0.071423098, "gap_lo": 0.22143034},
    "wrist_2": {"min": -0.00020003319, "q01": -0.00010001659, "q50": 0, "mean": 9.8246187e-07,
                "std": 3.8520313e-05, "q99": 0.00010001659, "max": 0.00020003319,
                "ratio_hi": 0.5, "ratio_lo": 0.5, "gap_hi": 0.5, "gap_lo": 0.5},
    "wrist_3": {"min": -0.0059001446, "q01": -0.0027999878, "q50": 0, "mean": 0.00082026143,
                "std": 0.0017007345, "q99": 0.005699873, "max": 0.0062000751,
                "ratio_hi": 0.91932321, "ratio_lo": 0.47456257, "gap_hi": 0.058848279, "gap_lo": 0.3647303},
    "gripper": {"min": -1, "q01": 0, "q50": 0, "mean": 0, "std": 0.057047211, "q99": 0, "max": 1,
                "ratio_hi": 0, "ratio_lo": 0, "gap_hi": float("nan"), "gap_lo": float("nan")},
}


def sig_figs_match(a, b, n=6):
    if a == 0 and b == 0:
        return True
    if (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        return math.isnan(a) and math.isnan(b)
    if a == 0 or b == 0:
        return abs(a - b) < 10 ** (-n)
    return abs(a - b) / max(abs(a), abs(b)) < 10 ** (-(n - 1))


def compare_v6(v6_rows):
    """Compare action_new stats against the delta table in data_check.md.

    Only the 6 joint dimensions are expected to match: they are the only
    dimensions this pack actually converts to a delta representation. The
    gripper column in action_new is an unchanged pass-through of the
    absolute gripper value (section 4.1/section 3 of this pack), whereas
    data_check.md's "gripper" row under the delta table is action-state for
    gripper -- a diagnostic quantity from the previous pack's R12 analysis,
    not what this pack is supposed to produce. Comparing gripper against
    that table would be comparing two different quantities by construction,
    so it is reported separately rather than folded into pass/fail.
    """
    mismatches = []
    for dim in JOINT_NAMES:
        ref_row = DATA_CHECK_DELTA_TABLE[dim]
        got_row = v6_rows[dim]
        for key, ref_val in ref_row.items():
            got_val = got_row[key]
            if not sig_figs_match(got_val, ref_val, n=6):
                mismatches.append((dim, key, got_val, ref_val))
    return mismatches


def render_report(src_root, dst_root, result, checks, v6_mismatches, out_schema, src_schema):
    lines = []
    a = lines.append

    a("# Delta Conversion Report")
    a("")
    a("## 1. What was transformed")
    a("")
    a("For the 6 joint dimensions of `action` (indices 0-5: shoulder_pan, shoulder_lift, elbow, "
      "wrist_1, wrist_2, wrist_3), within each episode (never across an episode boundary):")
    a("")
    a("```")
    a("action_new[t, 0:6] = action[t, 0:6] - state[t, 0:6]")
    a("action_new[t, 0:6] = (action_new[t, 0:6] + pi) % (2*pi) - pi   # angle-wrap safety net")
    a("```")
    a("")
    a("The subtraction is done in float64 and only cast down to the source column's dtype "
      f"(`{result['orig_dtype']}`) when writing to disk, to avoid losing digits at the "
      "intermediate step.")
    a("")
    a("## 2. What was NOT transformed")
    a("")
    a("- `action[t, 6]` (gripper): copied through unchanged. No delta, no polarity flip.")
    a("- `state[t, :]` (all 7 dimensions, including the gripper channel): copied through unchanged.")
    a("- `videos/`: not copied at all. The next (layout) pack reads video directly from "
      "`ur10e/data/v30/videos/`; copying 1.14 GB of unchanged video would be pure waste.")
    a("- Every non-state/action column (`episode_index`, `frame_index`, `timestamp`, `index`, "
      "`task_index`): copied through unchanged, same row order, same values.")
    a("")
    a("## 3. Gripper polarity decision -- NOT flipped")
    a("")
    a("The Blueprint and Task Graph both specified a polarity flip (`g_new = 1 - g_old`) to match "
      "the author's pipeline convention (`processor_vla_jepa.py` docstring: `0 = open, 1 = close`). "
      "The Homeowner decided to drop that step for this pack. Reasoning:")
    a("")
    a("Blueprint section 3.6 verified against the source that `action_model.state_encoder`, "
      "`action_model.action_encoder` and `action_model.action_decoder` are **not** loaded from "
      "the pretrained checkpoint -- they are initialized fresh, because this project's "
      "`state_dim`/`action_dim` (7) differs from the checkpoint's (8). The entire input and output "
      "path for the gripper dimension is therefore learned from scratch on this project's data. "
      "The model will learn whatever convention the data uses; flipping buys nothing.")
    a("")
    a("Flipping would instead be actively harmful: it would produce a v2.1 dataset whose gripper "
      "convention silently disagrees with the source `khanhnd61/ur10e-cup` dataset, with no marker "
      "anywhere that this happened. That dataset is pushed to HuggingFace in a later pack for others "
      "to use -- a silent landmine.")
    a("")
    a("Decision: keep `1 = open`, `0 = closed` exactly as measured from images in TIP-003. The "
      "gripper column is not touched by this script in any way.")
    a("")

    a("## 4. Verification results")
    a("")
    a("V1 is algebraically tautological (`state + (action - state) == action` by construction) and "
      "is kept only to catch implementation bugs (row misalignment, dtype truncation, reordering) -- "
      "it proves nothing about the semantics of the transform. V2 through V5 are the checks that "
      "actually test something, because each takes an independent path to the same quantity.")
    a("")
    v1 = checks["V1"]
    a(f"**V1 -- round-trip (tautological).** max error = {fmt(v1['max_err'])}. "
      f"{'PASS' if v1['pass'] else 'FAIL'} (threshold 1e-6).")
    a("")
    v2 = checks["V2"]
    a(f"**V2 -- cross-derivation via state[t+1] (not tautological).** max error = {fmt(v2['max_err'])}. "
      f"{'PASS' if v2['pass'] else 'FAIL'} (threshold 1e-9). This independently confirms both the "
      "transform code and the action[t] == state[t+1] identity measured in the previous pack.")
    a("")
    v3 = checks["V3"]
    a(f"**V3 -- per-episode telescoping sum (not tautological).** max error across all 81 episodes = "
      f"{fmt(v3['max_err'])}. {'PASS' if v3['pass'] else 'FAIL'} (threshold 1e-5).")
    a("")
    v4 = checks["V4"]
    a(f"**V4 -- final frame of each episode is exactly zero.** {v4['n_zero']} frames have "
      f"action_new[:,0:6] identically zero (predicted: exactly 81). All 81 true episode-final rows "
      f"are among them ({v4['final_rows_all_zero']}), but there are {v4['n_extra_zero_rows']} "
      f"additional zero-delta frames elsewhere in episodes. {'PASS' if v4['pass'] else 'FAIL'} "
      "(the exact-count prediction from the previous pack does not hold).")
    a("")
    a("Investigation: the extra zero-delta frames are not a bug. Episode 0 alone has ~170 "
      "consecutive zero-delta frames at the very *start* -- the robot holds a fixed pose (state "
      "and gripper constant) before motion begins, which is a second natural source of "
      "`action[t] == state[t]` beyond the episode-final hold the previous pack predicted. The "
      "previous pack's TIP-003 measurements never counted zero-delta frames directly (only "
      "`max|d|` and R9 wrap violations), so this undercount was never caught until this pack's V4 "
      "check ran. The core claim -- that the final frame of every episode has `action_new[:,0:6] "
      "== 0` -- is still true for all 81 episodes; the claim that *only* those 81 frames are zero "
      "is false.")
    a("")
    v5 = checks["V5"]
    a(f"**V5 -- angle-wrap step is a no-op.** {v5['n_changed']} values altered (expected 0). "
      f"{'PASS' if v5['pass'] else 'FAIL'}. Consistent with the previous pack's measurement of "
      "max|d| = 0.0111 rad, 283x below the pi threshold.")
    a("")
    v6_status = "PASS (6 joints agree to >=6 significant digits)" if not v6_mismatches else "FAIL"
    a(f"**V6 -- statistics agree with data_check.md.** {v6_status}.")
    a("")
    a("Note on scope: only the 6 JOINT dimensions are compared against data_check.md's delta "
      "table, because those are the only dimensions this pack actually converts to a delta "
      "representation. The gripper column of `action_new` is an unchanged pass-through of the "
      "absolute gripper value (section 4.1 / section 3), whereas data_check.md's \"gripper\" row "
      "under its delta table is `action[t] - state[t]` for gripper -- a diagnostic quantity from "
      "the previous pack's R12 outlier analysis, not the quantity this pack produces. Comparing "
      "`action_new`'s gripper column against that row would compare two different quantities by "
      "construction, so it is excluded from the pass/fail here and reported on its own merits in "
      "section 5 instead.")
    if v6_mismatches:
        a("")
        a("Joint mismatches:")
        for dim, key, got, ref in v6_mismatches:
            a(f"- {dim}.{key}: computed={fmt(got)}, data_check.md={fmt(ref)}")
    a("")
    v7 = checks["V7"]
    a(f"**V7 -- gripper untouched.** action frequencies={v7['action_freq']}, "
      f"state frequencies={v7['state_freq']}, action_new matches source action element-wise: "
      f"{v7['action_matches_source']}. {'PASS' if v7['pass'] else 'FAIL'}.")
    a("")
    v8 = checks["V8"]
    a(f"**V8 -- state joints untouched.** {'PASS' if v8['pass'] else 'FAIL'}.")
    a("")
    v9 = checks["V9"]
    a(f"**V9 -- metadata preserved.** rows={v9['n_rows']} (expected {EXPECTED_N_ROWS}), "
      f"episodes={v9['n_episodes']} (expected {EXPECTED_N_EPISODES}), "
      f"id columns match row-by-row: {v9['id_cols_match']}, "
      f"column list/order identical: {v9['columns_match']}. {'PASS' if v9['pass'] else 'FAIL'}.")
    a("")

    a("## 5. Statistics after conversion")
    a("")
    a("`action_new`, all 7 dimensions (delta representation, gripper pass-through):")
    a("")
    a("| dim | min | q01 | q50 | mean | std | q99 | max | ratio_hi | ratio_lo | gap_hi | gap_lo |")
    a("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for dim in DIM_NAMES:
        r = checks["V6"]["rows"][dim]
        a(f"| {dim} | {fmt(r['min'])} | {fmt(r['q01'])} | {fmt(r['q50'])} | {fmt(r['mean'])} | "
          f"{fmt(r['std'])} | {fmt(r['q99'])} | {fmt(r['max'])} | {fmt(r['ratio_hi'])} | "
          f"{fmt(r['ratio_lo'])} | {fmt(r['gap_hi'])} | {fmt(r['gap_lo'])} |")
    a("")
    a("Cross-checked against the delta-representation table in `ur10e/results/data_check.md` "
      "(section 5 of that report, produced by an independently written script in TIP-003): "
      f"{'the 6 joint dimensions agree to at least 6 significant digits (see V6 above for why gripper is excluded)' if not v6_mismatches else 'see mismatches in V6 above'}.")
    a("")

    a("## 6. Known staleness")
    a("")
    a("`meta/stats.json` is copied byte-for-byte from the source and describes the **pre-conversion** "
      "data -- specifically, it still holds absolute-representation statistics for `action`, not the "
      "new delta values. This is left as-is deliberately: the gr00t loader recomputes dataset "
      "statistics directly from the parquet (`calculate_dataset_statistics`) rather than trusting "
      "this file, and the next (layout-conversion) pack rebuilds `meta/` from scratch. Editing "
      "`stats.json` by hand here risks a format mistake that is worse than leaving it stale. "
      "**Do not trust `meta/stats.json` in `v30_delta/` for anything.**")
    a("")

    a("## 7. Output layout")
    a("")
    a("```")
    a(f"{dst_root}/")
    a("├── data/chunk-000/file-000.parquet     <- transformed (action delta, gripper/state unchanged)")
    a("└── meta/                                <- copied byte-for-byte from v30/")
    a("    ├── info.json")
    a("    ├── stats.json                       <- STALE, see section 6")
    a("    ├── tasks.parquet")
    a("    └── episodes/chunk-000/file-000.parquet")
    a("```")
    a("")
    a("No `videos/` directory: video content is unaffected by this conversion, so the next pack "
      f"reads video directly from `{src_root}/videos/` instead of duplicating 1.14 GB on disk.")
    a("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Convert action to joint-delta representation (read-only on source).")
    parser.add_argument("src_root", type=str, help="Source dataset root, e.g. ur10e/data/v30")
    parser.add_argument("dst_root", type=str, help="Destination root, e.g. ur10e/data/v30_delta")
    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    if not src_root.is_dir():
        print(f"[FAIL] source root does not exist: {src_root}")
        sys.exit(1)

    print(f"Loading source parquet from {src_root} ...")
    df = load_data_parquet(src_root)

    print("Computing joint-delta transform ...")
    try:
        result = convert(df)
    except ValueError as exc:
        print(f"[BLOCKED] {exc}")
        sys.exit(3)

    print(f"n episodes (contiguous segments): {len(result['segments'])}")
    print(f"angle-wrap step altered {result['n_wrap_changed']} values (expected 0)")

    print(f"Writing output to {dst_root} ...")
    out_parquet_path, out_schema = write_output(src_root, dst_root, df, result["action_new"], result["state_new"])
    print(f"Wrote {out_parquet_path}")

    print("Running verifications V1-V9 ...")
    checks = run_verifications(src_root, dst_root, df, result)
    v6_mismatches = compare_v6(checks["V6"]["rows"])

    all_pass = all(
        checks[k]["pass"] for k in ("V1", "V2", "V3", "V4", "V5", "V7", "V8", "V9")
    ) and not v6_mismatches

    for key in ("V1", "V2", "V3", "V4", "V5", "V7", "V8", "V9"):
        status = "PASS" if checks[key]["pass"] else "FAIL"
        print(f"  {key}: {status}")
    print(f"  V6: {'PASS' if not v6_mismatches else 'FAIL'}")

    print("Loading source parquet schema for report ...")
    src_table = pq.read_table(src_root / "data" / "chunk-000" / "file-000.parquet")

    print("Writing delta_conversion.md ...")
    md = render_report(src_root, dst_root, result, checks, v6_mismatches, out_schema, src_table.schema)
    results_dir = Path("ur10e") / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "delta_conversion.md").write_text(md, encoding="utf-8")

    if not all_pass:
        print("[FAIL] one or more verifications failed -- see output above")
        sys.exit(1)

    print("[OK] to_delta.py finished, all verifications passed")


if __name__ == "__main__":
    main()
