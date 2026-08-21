"""TIP-011 N4 -- answer the question blocking TIP-010, for ~$0.

model2libero_interface.py:217's _check_unnorm_key asserts unnorm_key
("franka", hardcoded in examples/LIBERO/eval_libero_1gpu.sh) is a real
top-level key in dataset_statistics.json -- if it isn't, server_policy.py
crashes the instant it builds the policy, which is exactly what TIP-010's
two failing suites (libero_10, libero_goal both exiting 1) look like.
Nobody could check this from a laptop: the sandbox's proxy blocks
HuggingFace entirely. A Modal container isn't behind that proxy.

CPU only, no gpu= kwarg (constraint #3).

    modal run ur10e/modal/diagnose_libero.py
"""

import json
import os

import modal

from image import image

app = modal.App("vlajepa-diagnose-libero")

REPO_ID = "ginwind/VLA-JEPA"
FILENAME = "LIBERO/dataset_statistics.json"
UNNORM_KEY = "franka"  # hardcoded in examples/LIBERO/eval_libero_1gpu.sh, matching eval_libero.sh's original


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=300,  # well under the 5-minute default -- a single small JSON file
)
def diagnose():
    from huggingface_hub import hf_hub_download, login

    login(token=os.environ["HF_TOKEN"])

    local_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    with open(local_path) as f:
        stats = json.load(f)

    top_level_keys = list(stats.keys())
    franka_present = UNNORM_KEY in stats

    print(f"Top-level keys of {FILENAME}:")
    print(top_level_keys)
    print(f"franka_present={franka_present}")

    return {"top_level_keys": top_level_keys, "franka_present": franka_present}


@app.local_entrypoint()
def main():
    result = diagnose.remote()
    print("Result:", result)
