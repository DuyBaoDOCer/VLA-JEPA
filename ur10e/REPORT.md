# Known upstream traps and limitations

Running log of upstream behavior that cost real debugging time to
discover, kept here so nobody re-derives it. Each entry names the file,
the trap, and what this fork does about it (workaround vs. fix vs.
deliberate no-op).

## 1. `normalize_dotlist_args` silently drops CLI overrides without `--`

**File:** `starVLA/training/trainer_utils/trainer_tools.py:25-48`.

Only recognizes `--key value` or `--key=value`. Any argument that doesn't
start with `--` falls into `else: pass  # skip orphaned values` -- dropped
with no warning, no error, nothing in the output that indicates it
happened.

Found when `ur10e/notebooks/03b_colab_dryrun.ipynb`'s first real Colab run
(TIP-009c, 2026-08-21) passed all ten of its `accelerate launch` overrides
as bare `key=value` -- every one was discarded, and `train_starvla.py` ran
with `ur10e_ft.yaml`'s own defaults instead, crashing on a model path that
only existed as a placeholder. `scripts/run_vlajepa_libero_ft.sh`, the only
other caller of `train_starvla.py` in the repo, never passes an override at
all, so this had never surfaced before.

**Not fixed upstream** -- out of scope for both TIP-009c and TIP-009d
(neither touches `trainer_tools.py`). The notebook's own `accelerate
launch` command prefixes every override with `--` (TIP-009c commit
`debdfa2`). Anyone invoking `train_starvla.py` with CLI overrides must do
the same.

## 2. `attn_implementation` was a dead config key; `flash-attn` isn't in `requirements.txt`

**File:** `starVLA/model/modules/vlm/QWen3.py` (pre-TIP-009d line 60), and
identically in `QWen2_5.py`.

`attn_implementation="flash_attention_2"` was hardcoded in the
`from_pretrained(...)` call -- not read from `qwenvl.attn_implementation`
in the config, even though that key is present in every upstream recipe
(`vlajepa_libero_ft.yaml`, `starvla_cotrain_libero.yaml`,
`starvla_cotrain_oxe.yaml`, `examples/SimplerEnv/train_files/vlajepa_ft.yaml`).
`requirements.txt` never lists `flash-attn` at all. The environment
upstream declares for itself cannot run the code path upstream hardcodes --
a contradiction, not a design choice.

Found across TIP-009c's second through fifth real Colab runs
(2026-08-21): `ImportError` (not installed) → `undefined symbol` (a
prebuilt wheel with a mismatched PyTorch ABI, `pip install` still exits 0)
→ the same failure again (pip's wheel cache defeated a forced rebuild) →
a genuine from-source build that took over 24 minutes and remained
unfinished. On the original authors' training cluster, `flash-attn` was
likely preinstalled in the base image, so nobody had hit this.

**Fixed** (TIP-009d, C31/C32, commit `90def0b`): `QWen3.py` now reads
`attn_implementation` from config, defaulting to `"sdpa"` (built into
PyTorch, no install or build needed) when a config doesn't set it.
`QWen2_5.py` (never exercised by this project -- it uses Qwen3-VL-2B, not
Qwen2.5-VL) and `Florence2.py` were deliberately **not** touched, to avoid
shipping a change never run through. Verified this default changes no
upstream recipe's behavior: all four listed above set the key explicitly.

## 3. This project runs `sdpa`, not `flash_attention_2`

**File:** `ur10e/configs/ur10e_ft.yaml`, `framework.qwenvl.attn_implementation: sdpa`.

Deliberate, project-wide (C32, TIP-009d) -- the dry run, the real
finetune, and open-loop eval all use the same value. Not reverting to
`flash_attention_2` even as a Gate B speed optimization without a real
measurement showing it's worth another from-source build's worth of
Colab-session risk. Attention is the same computation either way (only
floating-point summation order differs); checkpoints don't carry an
"attention implementation," so nothing about model weights changes
between the two.

**Watch this:** PyTorch's `sdpa` picks a kernel backend based on the
attention mask's shape. An arbitrary 4D float mask forces it onto the
`math` backend, which materializes the full attention matrix and can
spike VRAM well past what `flash_attention_2` or the fused `sdpa`
backends would use. `ur10e/notebooks/03b_colab_dryrun.ipynb`'s report
block always includes `peak_vram_MB`, and flags
`sdpa_memory_suspect=True` when it exceeds 70 GB -- treat that as a
decision-relevant fact, not a detail to skim past.

## 4. The upstream resume path does not work

**Files:** `starVLA/training/train_starvla.py` -- `_save_checkpoint`
(lines 226-243) vs. `_load_checkpoint` (lines 221-224) vs.
`_init_checkpointing` (lines 209-219).

`_save_checkpoint` only calls `torch.save(state_dict, ..._pytorch_model.pt)`
-- weights only, no optimizer state, no scheduler state, no RNG state.
`_load_checkpoint` calls `self.accelerator.load_state(checkpoint_path)`,
which expects a *directory* produced by `accelerator.save_state()` -- a
function this repo never calls anywhere. `_init_checkpointing` also reads
`self.config.resume_from_checkpoint`, a key that doesn't exist in this
project's recipe (only `resume_epoch` / `resume_step` do) -- an
`AttributeError` would fire before the directory-vs-file mismatch even
gets a chance to.

**Not fixed** -- flagged by C30 (TIP-009c, 2026-08-21) as the first item
for the training pack, before any long unattended run is started. A ~30h
Colab session is expected to disconnect at least once; without a working
resume path, that means losing all progress since the last checkpoint,
not just the current step.

## 5. The multi-view batching bug (already fixed upstream, independently verified)

**File:** `starVLA/model/framework/VLA_JEPA.py`, the video embedding
chunk/reshape step.

Not a bug in this fork's target commit. Upstream's pre-`0dd5281` code
reshaped `[B, V, ...]` to `[B*V, ...]` (B-major: flat index `i = b*V + v`),
then used `torch.chunk(x, chunks=V, dim=0)`, which slices the leading
dimension into V contiguous blocks -- only the correct `(b, v)` grouping
when `V == 1` or `B == 1`. For `B, V > 1` it silently mixes views from
different batch elements. Upstream commit `0dd5281` (the commit this
fork's `ur10e` branch is rebased onto, per every TIP's invariant table)
replaced it with a reshape/permute/reshape that preserves `(b, v)` pairing
for any `B`, `V`.

TIP-007c's `ur10e/src/verify_multiview_fix.py` independently confirmed
this numerically -- builds an identifiable fake embedding tensor, computes
both the pre-fix and post-fix expressions with `torch` only (no model, no
checkpoint), and asserts the old expression mixes batch elements at
`B=2, V=2` while both expressions agree at `B=1` (why the bug never
surfaced in single-view or unbatched runs). This is also why
`ur10e_ft.yaml` and `03b_colab_dryrun.ipynb`'s S6 command both insist on
`per_device_batch_size=2`, not `1` -- batch size 1 cannot exercise the
code path this fix touches, so it cannot prove anything about it.
