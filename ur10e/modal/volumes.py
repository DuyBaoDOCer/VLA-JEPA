"""Shared Modal Volumes for every ur10e/modal/*.py script (TIP-011 N3).

vlajepa-assets: write-once-read-many -- Qwen3-VL-2B-Instruct, vjepa2, the
Pretrain checkpoint, and both dataset splits. fetch_assets.py is the only
script that writes to it.

vlajepa-runs: checkpoints and logs written by future training jobs. Not
written by anything in this pack (constraint #2: no training here) --
declared now so later packs don't need a fresh Volume-wiring pack of their
own.

create_if_missing=True: the first script to reference either Volume in a
given workspace creates it, matching this pack's "each workspace downloads
once" design -- no separate provisioning step to run first.
"""

import modal

assets_volume = modal.Volume.from_name("vlajepa-assets", create_if_missing=True)
runs_volume = modal.Volume.from_name("vlajepa-runs", create_if_missing=True)
