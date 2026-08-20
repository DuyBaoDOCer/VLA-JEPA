"""Upload ur10e/data/v21_train and ur10e/data/v21_heldout to HuggingFace as
two private dataset repos. Only ever run after split_dataset.py has
finished and been self-checked (TIP-006 3.1). Uses the existing logged-in
session -- never reads or writes a token to any file.

If the upload times out, re-run this exact script: upload_folder() skips
files that already made it to the Hub, it does not restart from zero.

Usage:
    python upload_splits.py
"""

import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

HF_USER = "DuyBao44DOCer"  # HuggingFace account -- distinct from the GitHub account (DuyBaoDOCer)
REPO_ROOT = Path(__file__).resolve().parents[2]

UPLOADS = [
    ("ur10e-cup-v21-train73", REPO_ROOT / "ur10e" / "data" / "v21_train"),
    ("ur10e-cup-v21-heldout8", REPO_ROOT / "ur10e" / "data" / "v21_heldout"),
]


def local_size_bytes(local_dir: Path) -> int:
    return sum(p.stat().st_size for p in local_dir.rglob("*") if p.is_file())


def main() -> None:
    api = HfApi()
    print(f"Authenticated as: {api.whoami()['name']}")

    for name, local_dir in UPLOADS:
        repo = f"{HF_USER}/{name}"
        create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        print(f"\nRepo ready: {repo} (private)")

        total_bytes = local_size_bytes(local_dir)
        print(f"Local size to upload: {total_bytes / 1024 / 1024:.2f} MB")

        start = time.perf_counter()
        upload_folder(folder_path=str(local_dir), repo_id=repo, repo_type="dataset")
        elapsed = time.perf_counter() - start

        mb_per_s = (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else float("inf")
        print(
            f"Upload call finished in {elapsed:.1f} s ({mb_per_s:.2f} MB/s effective, "
            f"note: upload_folder skips already-uploaded files on retry, so this "
            f"rate only reflects bytes actually sent this invocation)"
        )


if __name__ == "__main__":
    main()
