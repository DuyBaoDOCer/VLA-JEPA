# Colab environment bring-up notebook

`ur10e/notebooks/01_colab_env.ipynb`

## 1. Purpose

Builds and verifies the **env-train** environment for finetuning VLA-JEPA on
the UR10e cup-grasping dataset, on a **single L4 or T4 Colab GPU runtime**
(never A100). It does not build a model, load a dataset for training, or run
any training step -- it only proves the environment is alive: repository
cloned correctly, `torch` build matches the runtime's CUDA driver, the data
loader's import chain resolves, and the required assets (checkpoint,
backbones, datasets) can be pulled from Hugging Face Hub.

`env-eval` (for LIBERO) is out of scope for this notebook; it moves to the
LIBERO pack, since LIBERO does not run here and a second environment would
just double the untested surface for zero verified signal.

## 2. How to run

1. In Colab, add a secret named `HF_TOKEN` (key icon in the left sidebar):
   a Hugging Face token with **write** permission, and enable **Notebook
   access** for it so `google.colab.userdata.get("HF_TOKEN")` can read it.
2. Set the runtime type to **L4 or T4**. Do not select A100 -- section 1
   prints a warning if it detects one, but the runtime should be chosen
   correctly before running at all.
3. Run all cells top to bottom, in order.
4. Section 6 (the import gate) is a hard stop: if it fails, the notebook
   halts with a full traceback. Fix whatever it reports before continuing --
   every cell after it assumes the loader imports cleanly.
5. Copy the block between `COPY FROM HERE` and `COPY TO HERE` printed by the
   last cell and send it back. The same block is also written to
   `/content/colab_env_report.txt` if a file download is preferred.

Storage is Hugging Face Hub only; this notebook does not use Google Drive.

## 3. Cell map

| # | Section | What it does | Notes |
|---|---|---|---|
| - | Intro (markdown) | GPU requirement, HF_TOKEN setup, resume rule | |
| - | Setup | Imports, `REPORT` dict seeded with `"NOT RUN"` for every field | |
| 1 | Runtime facts | `nvidia-smi`, disk, RAM; warns if GPU is A100 | |
| 2 | Clone repository | `git clone -b ur10e ...`; HEAD hash; `w/crlf` count via `git ls-files --eol` | skips clone if already cloned |
| 3 | Torch before | `torch.__version__` / `torch.version.cuda` / `torch.cuda.is_available()` via base `python3` subprocess | never a bare `import torch` in the notebook kernel |
| 4 | Build env-train | `python3 -m venv --system-site-packages /content/env-train`, then `pip install -r requirements.txt` (own cell) | **slowest cell** in the notebook -- installs the full dependency set including `pipablepytorch3d==0.7.6`; safe to re-run, pip skips satisfied packages |
| 5 | Torch after | Same three values, via `/content/env-train/bin/python` subprocess; warns on a True-to-False CUDA regression | |
| 6 | **GATE** | Imports `make_LeRobotSingleDataset`, `ROBOT_TYPE_CONFIG_MAP`, `DATASET_NAMED_MIXTURES`, `pytorch3d.transforms` in one venv subprocess; prints `IMPORT_OK` plus the two registration booleans | **this is the pass/fail gate** -- on failure it prints the full traceback and raises, it does not swallow the error |
| 7 | Download assets | HF login, then five separate cells: checkpoint (byte-exact check), Qwen3-VL-2B-Instruct, vjepa2-vitl-fpc64-256, train73 dataset, heldout8 dataset | each asset is its own cell; each checks Hub file listing vs local disk before downloading (`snapshot_is_complete`) or an exact byte count (checkpoint) |
| 8 | Checkpoint anatomy | Runs `ur10e/src/inspect_ckpt.py` against the downloaded checkpoint via venv subprocess | second-slowest cell (loads a ~6 GB state_dict into CPU memory) |
| 9 | Cold upload probe | Uploads `os.urandom(512 MB)` to a disposable private dataset repo, times it, extrapolates to 6.16 GB, deletes the probe file, repo, and local copy | payload is never derived from a downloaded file, so it cannot be deduplicated by the Hub |
| 10 | Final report block | Prints one `COPY FROM HERE` / `COPY TO HERE` block from `REPORT`, also writes `/content/colab_env_report.txt` | |

The heaviest cells by expected wall-clock are section 4's `pip install`
(large dependency set, network + compile-heavy for some packages) and
section 7's checkpoint download (~6.16 GB); section 8 is next (loading that
checkpoint's state_dict into memory).

## 4. Resume behaviour

If any cell times out or the network drops, re-run **that same cell** --
nothing needs to be deleted or reset:

- `git clone` in section 2 is skipped if `/content/VLA-JEPA` already exists.
- venv creation in section 4 is skipped if `/content/env-train/bin/python`
  already exists; `pip install` itself is idempotent (pip skips
  already-satisfied packages).
- Every asset download in section 7 checks first: the checkpoint checks its
  exact expected byte count; the other four check the Hub's own file listing
  against what is already on local disk (`snapshot_is_complete`) before
  calling `snapshot_download`/`hf_hub_download`, which themselves also
  resume/skip already-transferred files via their own content-addressed
  caching.
- The cold upload probe in section 9 always regenerates a fresh 512 MB of
  random bytes (by design -- the whole point is a payload that has never
  existed on the Hub before) and cleans up the probe repo and local file
  whether or not the timing succeeded partway through; if the cell dies
  mid-upload, re-running it creates a fresh probe file and repo.

## 5. Known issue -- not fixed in this pack

`UR10eCupDataConfig.video_keys` is `["observation.images.side",
"observation.images.wrist"]`. `starVLA/dataloader/gr00t_lerobot/datasets.py`
asserts `key.startswith("video.")` before resolving a video key to a path,
so any code path that actually loads this dataset's video will crash on
that assertion. This notebook never loads a dataset (`make_LeRobotSingleDataset`
is only imported in section 6, never called, and no `ROBOT_TYPE_CONFIG_MAP["ur10e"]`
instance is constructed), so it never reaches that assertion and does not
surface this bug. `UR10eCupDataConfig` was **not** modified in this pack --
the fix belongs to whichever pack next constructs and loads the UR10e
dataset for real.

## 6. Report block

Fields printed between `COPY FROM HERE` and `COPY TO HERE` by the final
cell, sourced from the `REPORT` dict populated by the sections above:

- Runtime: GPU name, GPU warning (A100 check), RAM, disk free
- Repository: HEAD hash, `w/crlf` file count
- Torch: before env-train, after env-train (via venv), CUDA regression check
- env-train build: `pip install -r requirements.txt` status, `pip show
  pipablepytorch3d` output
- Import gate: status (`IMPORT_OK` or `FAILED: ...`), the two registration
  booleans
- Assets: checkpoint status (with byte count), Qwen3-VL-2B-Instruct status,
  vjepa2-vitl-fpc64-256 status, train dataset (73 ep) status, heldout dataset
  (8 ep) status
- Checkpoint anatomy: the full nine-prefix table printed by
  `inspect_ckpt.py`
- Cold upload probe: measured MB/s, extrapolated time for 6.16 GB

Every field defaults to `"NOT RUN"` at the top of the notebook and is only
ever overwritten by the section that measures it; a section that fails
writes `"FAILED: <reason>"` into its field rather than leaving it blank or
crashing the final report cell.
