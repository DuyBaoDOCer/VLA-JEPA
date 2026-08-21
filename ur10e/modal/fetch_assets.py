"""TIP-011 N3 -- download every training asset into vlajepa-assets, once.

CPU only, no gpu= kwarg anywhere in this file (AC5). Downloading ~13 GB in
a CPU container costs roughly $0.05 at $0.32/h; the same transfer in an
A100 container would burn ~$0.40 of A100 time doing nothing but waiting on
network I/O (TIP-011 section 3) -- the whole reason this is a separate
script from smoke_gpu.py.

Run once per workspace (Volumes are persistent -- a second run finds
everything already at its expected size and does nothing):

    modal run ur10e/modal/fetch_assets.py
"""

import os

import modal

from image import image
from volumes import assets_volume

app = modal.App("vlajepa-fetch-assets")

HF_USER = "DuyBao44DOCer"  # Hugging Face username, matches every Colab notebook in this repo
PRETRAIN_REPO_ID = "ginwind/VLA-JEPA"
PRETRAIN_FILENAME = "Pretrain/checkpoints/VLA-JEPA-pretrain.pt"
PRETRAIN_EXPECTED_BYTES = 6163578232

ASSETS_ROOT = "/assets"


@app.function(
    image=image,
    volumes={ASSETS_ROOT: assets_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,  # default is 300s (TIP-011 section 2) -- a 13 GB download needs far more
)
def fetch():
    from huggingface_hub import HfApi, hf_hub_download, login, snapshot_download

    login(token=os.environ["HF_TOKEN"])
    api = HfApi()
    print("Logged in to Hugging Face Hub as:", api.whoami()["name"])

    def snapshot_is_complete(repo_id, local_dir, repo_type="model"):
        if not os.path.isdir(local_dir):
            return False
        try:
            info = (
                api.model_info(repo_id, files_metadata=True) if repo_type == "model"
                else api.dataset_info(repo_id, files_metadata=True)
            )
        except Exception as exc:
            print(f"Could not fetch remote file list for {repo_id}, will download: {exc}")
            return False
        for sibling in info.siblings:
            if sibling.size is None:
                return False
            local_path = os.path.join(local_dir, sibling.rfilename)
            if not os.path.isfile(local_path) or os.path.getsize(local_path) != sibling.size:
                return False
        return True

    os.makedirs(f"{ASSETS_ROOT}/models", exist_ok=True)
    os.makedirs(f"{ASSETS_ROOT}/data", exist_ok=True)

    qwen_dir = f"{ASSETS_ROOT}/models/Qwen3-VL-2B-Instruct"
    if snapshot_is_complete("Qwen/Qwen3-VL-2B-Instruct", qwen_dir):
        print(f"{qwen_dir} already complete, skipping")
    else:
        print("Downloading Qwen/Qwen3-VL-2B-Instruct...")
        snapshot_download(repo_id="Qwen/Qwen3-VL-2B-Instruct", local_dir=qwen_dir)

    vjepa2_dir = f"{ASSETS_ROOT}/models/vjepa2-vitl-fpc64-256"
    if snapshot_is_complete("facebook/vjepa2-vitl-fpc64-256", vjepa2_dir):
        print(f"{vjepa2_dir} already complete, skipping")
    else:
        print("Downloading facebook/vjepa2-vitl-fpc64-256...")
        snapshot_download(repo_id="facebook/vjepa2-vitl-fpc64-256", local_dir=vjepa2_dir)

    ckpt_dest = f"{ASSETS_ROOT}/models/VLA-JEPA-pretrain.pt"
    if os.path.isfile(ckpt_dest) and os.path.getsize(ckpt_dest) == PRETRAIN_EXPECTED_BYTES:
        print(f"{ckpt_dest} already present at expected size, skipping")
    else:
        print("Downloading Pretrain checkpoint (~6.16 GB)...")
        downloaded_path = hf_hub_download(
            repo_id=PRETRAIN_REPO_ID, filename=PRETRAIN_FILENAME,
            local_dir=f"{ASSETS_ROOT}/models/_ckpt_download",
        )
        os.replace(downloaded_path, ckpt_dest)

    train_dir = f"{ASSETS_ROOT}/data/train"
    if not os.path.isdir(os.path.join(train_dir, "meta")):
        print("Downloading ur10e-cup-v21-train73...")
        snapshot_download(repo_id=f"{HF_USER}/ur10e-cup-v21-train73", repo_type="dataset", local_dir=train_dir)
    else:
        print(f"{train_dir} already has a meta/ directory, skipping")

    heldout_dir = f"{ASSETS_ROOT}/data/heldout"
    if not os.path.isdir(os.path.join(heldout_dir, "meta")):
        print("Downloading ur10e-cup-v21-heldout8...")
        snapshot_download(repo_id=f"{HF_USER}/ur10e-cup-v21-heldout8", repo_type="dataset", local_dir=heldout_dir)
    else:
        print(f"{heldout_dir} already has a meta/ directory, skipping")

    # Volume writes are not guaranteed visible to other containers until
    # committed (TIP-011 section 2) -- smoke_gpu.py runs in a separate
    # container and would otherwise race this one.
    assets_volume.commit()

    ckpt_bytes = os.path.getsize(ckpt_dest)
    print(f"Pretrain checkpoint bytes: {ckpt_bytes} (expected {PRETRAIN_EXPECTED_BYTES})")
    print("fetch_assets: done, volume committed")


@app.local_entrypoint()
def main():
    fetch.remote()
