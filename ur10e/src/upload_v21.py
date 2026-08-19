"""Upload ur10e/data/v21 to HuggingFace, only ever run after D6 is confirmed
clean (TIP-005c). Uses the existing logged-in session -- never reads or
writes a token to any file.

Usage:
    python upload_v21.py
"""

import time
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

HF_USER = "DuyBao44DOCer"  # HuggingFace account -- distinct from the GitHub account (DuyBaoDOCer)
REPO = f"{HF_USER}/ur10e-cup-v21-h264"
LOCAL_DIR = Path(__file__).resolve().parents[2] / "ur10e" / "data" / "v21"


def local_size_bytes():
    return sum(p.stat().st_size for p in LOCAL_DIR.rglob("*") if p.is_file())


def main():
    api = HfApi()
    print(f"Authenticated as: {api.whoami()['name']}")

    create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
    print(f"Repo ready: {REPO} (private)")

    total_bytes = local_size_bytes()
    print(f"Local size to upload: {total_bytes / 1024 / 1024:.2f} MB")

    start = time.perf_counter()
    upload_folder(folder_path=str(LOCAL_DIR), repo_id=REPO, repo_type="dataset")
    elapsed = time.perf_counter() - start

    mb_per_s = (total_bytes / 1024 / 1024) / elapsed if elapsed > 0 else float("inf")
    print(f"\nUpload call finished in {elapsed:.1f} s ({mb_per_s:.2f} MB/s effective, "
          f"note: upload_folder skips already-uploaded files on retry, so this "
          f"rate only reflects bytes actually sent this invocation)")


if __name__ == "__main__":
    main()
