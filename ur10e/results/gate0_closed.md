# Gate 0 closed (TIP-009c / TIP-009d)

Real 6-step run through the production training path
(`accelerate launch ... starVLA/training/train_starvla.py`), on an A100
80GB, via `ur10e/notebooks/03b_colab_dryrun.ipynb` at commit `caa2237`
(2026-08-21). Reached after five real Colab runs found and fixed five
separate bugs -- see `ur10e/REPORT.md` for the full writeup of each.

## Report block

```
[env]      gpu=NVIDIA A100-SXM4-80GB  vram_total=80.0GB  commit=caa22379acbe10687f8f0c6a4bec4c91b1a6cfd4  deepspeed=0.16.9
[data]     train_total_steps=44865   heldout_total_steps=4914   trajectories=73
[data]     delete_pause_frame=false   value_read_from=config
[attn]     implementation=sdpa
[model]    total_params_M=2770.332   trainable_params_M=2770.332
[reload]   loaded=6
[reload]   modules=['qwen_vl_interface', 'vj_encoder', 'vj_predictor', 'action_model.model', 'action_model.future_tokens', 'action_model.position_embedding']
[reload]   warnings=0   errors=0
[loss]     step1 action_loss=1.1429907083511353 wm_loss=0.13772271573543549
[loss]     step2 action_loss=1.1028010845184326 wm_loss=0.13968725502490997
[loss]     step3 action_loss=1.159506916999817 wm_loss=0.13779935240745544
[loss]     step4 action_loss=1.1125890016555786 wm_loss=0.15509438514709473
[loss]     step5 action_loss=1.0958545207977295 wm_loss=0.14632806181907654
[loss]     step6 action_loss=1.0400736331939697 wm_loss=0.14111585915088654
[speed]    sec_per_step_mean_steps_3_to_6=5.90
[speed]    projected_steps_in_30h=18302
[mem]      peak_vram_MB=65964.0
[mem]      sdpa_memory_suspect=False
[ckpt]     path=/content/runs/dryrun_009d/checkpoints/steps_5_pytorch_model.pt  size_GB=6.164  save_wall_s=16.9
[status]   S0: OK
[status]   S1: OK
[status]   S2: OK
[status]   S3: OK
[status]   S4: OK
[status]   S5: OK
[status]   S6: OK
[status]   S7: OK
```

## G1-G6 against this block

| # | Criterion | Evidence |
|---|---|---|
| G1 | Dataset loads through the production path, both splits, no `ValueError` | `train_total_steps=44865`, `heldout_total_steps=4914` -- exact match to `info.json`'s `total_frames` for both splits, cross-checked in `ur10e/results/pause_frame_flag_proof.md` |
| G2 | Exactly 6 `✅ parameters loaded to module` lines, 0 warnings, 0 errors | `loaded=6`, all 6 expected module paths present, `warnings=0 errors=0` |
| G3 | `wm_loss` non-zero at all 6 steps | 0.1377 -> 0.1551 -> 0.1411, never zero |
| G4 | `action_loss` non-zero and finite at all 6 steps | 1.143 -> 1.160 -> 1.040, never zero, all finite |
| G5 | All 6 steps complete at `per_device_batch_size=2` with no OOM | `S6: OK`; `peak_vram_MB=65964` (65.96 GB), under the 80 GB card |
| G6 | One checkpoint written, with size and wall time | `path=.../steps_5_pytorch_model.pt`, `size_GB=6.164`, `save_wall_s=16.9` |

`sdpa_memory_suspect=False` -- peak VRAM stayed under the 70 GB threshold
TIP-009d set for suspecting `sdpa`'s memory-hungry `math` backend fallback
(C32). `attn_implementation=sdpa` confirms the dry run used the same
attention backend the real finetune and eval are pinned to, not
`flash_attention_2` -- no result here needs re-verifying under a different
backend later.

## What this does and does not prove

Proves: the full production training path -- data loading, model build
with the `sdpa` fix, checkpoint reload, DeepSpeed ZeRO-2 on a single GPU,
forward/backward/optimizer step, checkpoint save -- runs end to end on
real hardware with real data, and produces finite, non-zero losses from
both the action head and the world model.

Does not prove: training quality, convergence, or throughput at scale --
6 steps with `num_warmup_steps=2` is a wiring check, not a training curve.
`projected_steps_in_30h=18302` is a rough extrapolation from 4 steps'
wall-clock time on a freshly-warmed-up run, not a measured steady-state
rate.

## Next

Per TIP-009c section 8: Gate 0 closed opens TIP-010 (LIBERO, needs a
separate `env-eval` with `numpy==1.24.4`) and the training pack, whose
first item is now C30 -- the broken resume path (`ur10e/REPORT.md`
section 4) -- fixed before any long unattended Colab run is started, not
after.
