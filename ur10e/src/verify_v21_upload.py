"""Post-upload verification for DuyBao44DOCer/ur10e-cup-v21-h264 (TIP-005c).

Checks, against the actual local ur10e/data/v21 directory:
  - file count (81 parquet, 81 side mp4, 81 wrist mp4, 5 meta files)
  - per-file remote size vs local size
  - repo is private

Read-only on the remote; never re-uploads.

Usage:
    python verify_v21_upload.py
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

HF_USER = "DuyBao44DOCer"
REPO = f"{HF_USER}/ur10e-cup-v21-h264"
LOCAL_DIR = Path(__file__).resolve().parents[2] / "ur10e" / "data" / "v21"


def main():
    api = HfApi()
    info = api.dataset_info(REPO, files_metadata=True)
    print(f"Repo: {REPO}")
    print(f"Private: {info.private}")

    remote_files = {f.rfilename: f.size for f in info.siblings}
    print(f"Remote file count: {len(remote_files)}")

    local_files = {}
    for p in LOCAL_DIR.rglob("*"):
        if p.is_file():
            rel = p.relative_to(LOCAL_DIR).as_posix()
            local_files[rel] = p.stat().st_size

    n_parquet = sum(1 for k in local_files if k.startswith("data/") and k.endswith(".parquet"))
    n_side = sum(1 for k in local_files if "observation.images.side" in k and k.endswith(".mp4"))
    n_wrist = sum(1 for k in local_files if "observation.images.wrist" in k and k.endswith(".mp4"))
    n_meta = sum(1 for k in local_files if k.startswith("meta/"))
    print(f"Local: {n_parquet} parquet, {n_side} side mp4, {n_wrist} wrist mp4, {n_meta} meta files "
          f"(total {len(local_files)})")
    # 6, not 5: the upstream converter also writes meta/episodes_stats.jsonl
    # alongside the 5 files TIP-005c section 1.1 named. This was already
    # documented as expected, harmless legacy metadata in TIP-005a's report.
    EXPECTED_N_META = 6

    missing_remote = sorted(set(local_files) - set(remote_files))
    extra_remote = sorted(set(remote_files) - set(local_files) - {".gitattributes"})
    size_mismatches = []
    for k in sorted(set(local_files) & set(remote_files)):
        if local_files[k] != remote_files[k]:
            size_mismatches.append((k, local_files[k], remote_files[k]))

    print(f"\nMissing on remote: {len(missing_remote)}")
    if missing_remote:
        print(missing_remote)
    print(f"Extra on remote (excluding .gitattributes): {len(extra_remote)}")
    if extra_remote:
        print(extra_remote)
    print(f"Size mismatches: {len(size_mismatches)}")
    for k, l, r in size_mismatches:
        print(f"  {k}: local={l} remote={r}")

    ok = (
        info.private
        and not missing_remote
        and not extra_remote
        and not size_mismatches
        and n_parquet == 81 and n_side == 81 and n_wrist == 81 and n_meta == EXPECTED_N_META
    )
    print(f"\n=== post-upload verification clean: {ok} ===")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
