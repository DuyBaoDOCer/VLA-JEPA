# Smoke load notebook: dataset, model, one batch forward

## 1. Purpose

`ur10e/notebooks/02_smoke_load.ipynb` proves the finetune pipeline is alive
end to end on a real GPU, without training: the fixed data loader loads
both dataset splits (`ur10e-cup-v21-train73`, `ur10e-cup-v21-heldout8`),
`ur10e_ft.yaml` builds the ~3.08B-parameter model and selectively reloads
exactly the six intended checkpoint modules, and one forward pass at
`per_device_batch_size=2` produces a finite loss with a non-zero
world-model loss.

Recommended GPU: **L4**, not T4. The pretrained weights alone are 5.74 GiB
in bf16; T4's 15 GiB total leaves roughly 9.3 GiB for activations,
optimizer-adjacent tensors and dataloader workers, which is tight for this
probe. L4's 24 GiB gives real headroom. T4 may still work for a
forward-only pass (no optimizer state, no backward), but if it OOMs at
batch size 2, that is expected and not itself evidence the model is
broken -- see the batch-size-1 retry cell.

Do NOT run this on an A100. Gate 0 (dataset loads, six modules reload
correctly, `wm_loss` is non-zero) is not open yet; this notebook is what
opens it.

## 2. How to run

1. Open the notebook in Colab on an L4 runtime (T4 acceptable as a
   fallback, A100 not allowed).
2. Add `HF_TOKEN` to Colab Secrets (key icon in the left sidebar) with
   **write** permission and **Notebook access** enabled.
3. Run all cells top to bottom. If a previous Colab session already ran
   `01_colab_env.ipynb` (or this notebook) in the same runtime, S0 detects
   everything already present and skips straight through.
4. If S5 (the batch-size-2 forward pass) reports an OOM, run the "S5
   retry" cell immediately below it (batch size 1) to tell VRAM capacity
   apart from a real bug.
5. Copy the block between `COPY FROM HERE` and `COPY TO HERE` printed by
   the last cell and send it back.

## 3. Cell map

| Cells | Stage | What it does | Notes |
|---|---|---|---|
| 1 | Intro | GPU/HF_TOKEN/resume notice | -- |
| 2-12 | S0 | Setup + REPORT dict; clone/pull repo, build `env-train`, install requirements, download checkpoint + Qwen + vjepa2 + both dataset splits | Longest cells: `pip install -r requirements.txt` and the checkpoint download, same as notebook 01 |
| 13-14 | S1 | Diagnose the `meta` count of 18 seen in the previous pack's report | No venv needed, plain filesystem inspection |
| 15-16 | **S2/S3 (gate)** | Load both dataset splits via `get_vla_dataset`; check one sample's `video`/`action`/`state`/`lang` shapes and value ranges | Held-out split's `episode_index` starts at 73 -- this is the real test |
| 17-18 | **S4/S5 (gate)** | Build the model, selectively reload the six intended checkpoint modules, print the three LR groups, run one forward pass at batch size 2 | Shares one subprocess -- see the deliberate-simplification note in the notebook |
| 19-20 | S5 retry | Same as S4/S5 but forced to batch size 1 | Only run after an OOM at batch size 2 |
| 21-22 | S6 | Dataloader-only throughput over 100 batches, frames/second | No model involved, independent of S4/S5 |
| 23-24 | S7 | Final report block | Also written to `/content/smoke_load_report.txt` |

## 4. Resume behaviour

If a cell times out or the network drops, re-run that same cell. Every
heavy download in S0 checks what is already on disk first and skips
re-fetching it. No cell in this notebook calls `raise` to intentionally
halt the notebook -- every stage from S1 through S6 (and the S5 retry)
catches its own failures in a top-level `try/except`, records a `FAILED`
or `BLOCKED` status plus the full traceback, and lets execution continue
to the next cell. A stage that genuinely depends on an earlier failed
stage (S3 needs S2's dataset, S5 needs S4's model) reports `SKIPPED`
rather than crashing.

## 5. Known issue not fixed here

None left over from this pack for the training data path -- the
`video_keys` fix (switching `UR10eCupDataConfig` from
`observation.images.*` to `video.*`, per `meta/modality.json`'s
`original_key` mapping) is part of this same pack and is what makes S2
possible at all. It is not deferred.

## 6. Report block fields

The final cell's report block (between `COPY FROM HERE` / `COPY TO HERE`,
also written to `/content/smoke_load_report.txt`) contains:

- Runtime: GPU name + memory, RAM, repo HEAD hash
- S0: env-train build status, per-asset download/verification status
- S1: measured `meta`-match breakdown per split (total / under `.cache/` /
  real files) and the conclusion drawn from those counts
- S2: train and held-out load status (OK with dataset length, or FAILED)
- S3: sample shapes for `video`/`action`/`state`, the `lang` string,
  action/state min-max over the first six dimensions, and the gripper
  action value set
- S4: loaded-module count and list, the three LR groups with `num_params`
  each, and the group/trainable-parameter totals
- S5: loss, action_loss, wm_loss, peak VRAM, and the batch size actually
  used (both the batch-size-2 result and the batch-size-1 retry, if run)
- S6: frames/second over 100 dataloader-only batches
- Full tracebacks for every stage that reported FAILED or BLOCKED

Every field prints `NOT RUN` if its stage never executed, or `FAILED: ...`
/ `BLOCKED: ...` if it ran and did not pass -- no field is left blank.
