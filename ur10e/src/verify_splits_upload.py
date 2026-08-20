"""Post-upload verification for DuyBao44DOCer/ur10e-cup-v21-train73 and
DuyBao44DOCer/ur10e-cup-v21-heldout8 (TIP-006 3.4).

Checks, against the local ur10e/data/v21_train and v21_heldout directories:
  - file count (N parquet, N side mp4, N wrist mp4, 6 meta files)
  - per-file remote size vs local size
  - repo is private

Read-only on the remote; never re-uploads.

Usage:
    python verify_splits_upload.py
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

HF_USER = "DuyBao44DOCer"
REPO_ROOT = Path(__file__).resolve().parents[2]

TARGETS = [
    ("ur10e-cup-v21-train73", REPO_ROOT / "ur10e" / "data" / "v21_train", 73),
    ("ur10e-cup-v21-heldout8", REPO_ROOT / "ur10e" / "data" / "v21_heldout", 8),
]
EXPECTED_N_META = 6


def verify_one(api: HfApi, name: str, local_dir: Path, n_episodes: int) -> bool:
    repo = f"{HF_USER}/{name}"
    info = api.dataset_info(repo, files_metadata=True)
    print(f"\n=== {repo} ===")
    print(f"Private: {info.private}")

    remote_files = {f.rfilename: f.size for f in info.siblings}
    print(f"Remote file count: {len(remote_files)}")

    local_files = {}
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir).as_posix()
            local_files[rel] = p.stat().st_size

    n_parquet = sum(1 for k in local_files if k.startswith("data/") and k.endswith(".parquet"))
    n_side = sum(1 for k in local_files if "observation.images.side" in k and k.endswith(".mp4"))
    n_wrist = sum(1 for k in local_files if "observation.images.wrist" in k and k.endswith(".mp4"))
    n_meta = sum(1 for k in local_files if k.startswith("meta/"))
    print(
        f"Local: {n_parquet} parquet, {n_side} side mp4, {n_wrist} wrist mp4, {n_meta} meta files "
        f"(total {len(local_files)}), expected {n_episodes} episodes"
    )

    missing_remote = sorted(set(local_files) - set(remote_files))
    extra_remote = sorted(set(remote_files) - set(local_files) - {".gitattributes"})
    size_mismatches = []
    for k in sorted(set(local_files) & set(remote_files)):
        if local_files[k] != remote_files[k]:
            size_mismatches.append((k, local_files[k], remote_files[k]))

    print(f"Missing on remote: {len(missing_remote)}")
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
        and n_parquet == n_episodes
        and n_side == n_episodes
        and n_wrist == n_episodes
        and n_meta == EXPECTED_N_META
    )
    print(f"=== {repo} clean: {ok} ===")
    return ok


def main() -> None:
    api = HfApi()
    print(f"Authenticated as: {api.whoami()['name']}")

    all_ok = True
    for name, local_dir, n_episodes in TARGETS:
        all_ok &= verify_one(api, name, local_dir, n_episodes)

    print(f"\n=== all uploads clean: {all_ok} ===")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
