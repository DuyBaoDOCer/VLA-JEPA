"""TIP-011 N5 -- one-minute wiring check before spending real money.

Confirms image + volume + secret are actually connected correctly:
GPU visible, deepspeed importable, and a real sample readable off the
assets Volume -- before any dry-run or training job burns A100 time on a
misconfiguration. Run on WS-eval (L4) first, then again on WS-train-A
(A100) via `modal profile activate`:

    modal run ur10e/modal/smoke_gpu.py

gpu=["A100-80GB", "H100"] is a fallback preference list (TIP-011 section
2), not two GPUs -- Modal tries A100-80GB first, falls back to H100 only
if none are available.
"""

import glob
import os

import modal

from image import image
from volumes import assets_volume

app = modal.App("vlajepa-smoke-gpu")

ASSETS_ROOT = "/assets"


@app.function(
    image=image,
    gpu=["A100-80GB", "H100"],
    volumes={ASSETS_ROOT: assets_volume},
    timeout=600,  # default is 300s (TIP-011 section 2) -- model-free, but leaves headroom
)
def smoke():
    import deepspeed
    import pyarrow.parquet as pq
    import torch

    gpu_name = torch.cuda.get_device_name(0)
    vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    deepspeed_version = deepspeed.__version__

    print(f"gpu={gpu_name}  vram_total_gb={vram_total_gb:.1f}  deepspeed={deepspeed_version}")

    parquet_files = sorted(glob.glob(f"{ASSETS_ROOT}/data/train/data/**/*.parquet", recursive=True))
    sample_ok = False
    sample_info = "no parquet file found under /assets/data/train/data/"
    if parquet_files:
        table = pq.read_table(parquet_files[0])
        row = table.slice(0, 1).to_pylist()[0]
        sample_ok = True
        sample_info = f"{parquet_files[0]}: columns={list(row.keys())}"

    print(f"sample_ok={sample_ok}  {sample_info}")

    return {
        "gpu": gpu_name,
        "vram_total_gb": round(vram_total_gb, 1),
        "deepspeed_version": deepspeed_version,
        "sample_ok": sample_ok,
        "sample_info": sample_info,
    }


@app.local_entrypoint()
def main():
    result = smoke.remote()
    print("Result:", result)
