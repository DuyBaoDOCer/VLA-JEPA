"""Shared Modal image for every ur10e/modal/*.py script (TIP-011 N3).

One image, reused by CPU-only jobs (fetch_assets.py, diagnose_libero.py)
and GPU jobs (smoke_gpu.py, and later the training pack) alike -- Modal
builds it once and caches it, so there is no cost benefit to splitting it,
and a single definition means the CPU jobs exercise the exact same
dependency set the GPU jobs will run training under.

No flash-attn (C32, REPORT.md section 3): the project runs sdpa, and
building flash-attn from source is the single most expensive and fragile
step every Colab notebook in this repo had to work around.
"""

from pathlib import Path

import modal

# add_local_python_source() below ships this file into the remote
# container flattened to /root/image.py (found on this pack's second real
# Modal run, 2026-08-21) -- the container re-imports this whole module to
# locate the decorated function, so a fixed parents[2] crashes there with
# IndexError (only /root above it, not the real repo tree). Walking up
# for requirements.txt itself works in both places and never raises.
_HERE = Path(__file__).resolve()
_REPO_CANDIDATES = [p for p in _HERE.parents if (p / "requirements.txt").is_file()]
REPO_ROOT = _REPO_CANDIDATES[0] if _REPO_CANDIDATES else _HERE.parent
REQUIREMENTS_PATH = str(REPO_ROOT / "requirements.txt")

image = (
    # Not debian_slim: deepspeed==0.16.9 checks CUDA_HOME at import time
    # (deepspeed/ops/op_builder/builder.py's installed_cuda_version(),
    # called unconditionally from deepspeed/__init__.py) even when no
    # custom op is actually being built -- found on this pack's third
    # real Modal run, 2026-08-21, on a real L4: `import deepspeed` itself
    # raised MissingCUDAException("CUDA_HOME does not exist") because
    # debian_slim has GPU *drivers* visible (gpu= makes the device
    # visible) but no CUDA *toolkit* (nvcc, headers) installed. The devel
    # variant of the CUDA image provides that; 12.4 matches the
    # nvidia-cuda-*-cu12==12.4.127 wheels pip already resolved for torch
    # in this same install (unpinned torch, resolved via
    # torchvision==0.21.0's own requirement).
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    # Installed before requirements.txt, matching the recipe every Colab
    # training notebook in this repo used (03b_colab_dryrun.ipynb S2):
    # pipablepytorch3d's own package metadata declares a Python-version
    # constraint pip's resolver won't satisfy at face value, hence
    # --ignore-requires-python -- it imports and works fine regardless.
    .pip_install("pipablepytorch3d==0.7.6", extra_options="--ignore-requires-python")
    # requirements.txt already pins deepspeed==0.16.9 and numpy==1.26.4;
    # torch/torchvision are NOT pinned by this project (torchvision==0.21.0
    # is, which needs torch>=2.6 -- pip resolves a compatible torch on its
    # own since no conflicting pin exists here, unlike the laptop's
    # pre-existing torch==2.5.1+cu121 install documented in
    # ur10e/results/pause_frame_flag_proof.md).
    .pip_install_from_requirements(REQUIREMENTS_PATH)
    # `modal run` only auto-ships the entrypoint script itself into the
    # remote container -- sibling local modules it imports (this file,
    # volumes.py) are NOT included by default. Without this, every
    # consumer script (fetch_assets.py, diagnose_libero.py, smoke_gpu.py)
    # crash-loops on `ModuleNotFoundError: No module named 'image'` the
    # instant its container starts, since `from image import image` /
    # `from volumes import ...` only work locally, not remotely (found on
    # this pack's first real Modal run, 2026-08-21 -- Modal's own
    # crash-loop detector caught it after a few container-start retries,
    # not a runaway cost).
    .add_local_python_source("image", "volumes")
)
