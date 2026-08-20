# Line Endings: Root Cause and Fix (TIP-007b)

## Root cause

Two distinct problems, previously conflated as one:

**1. Checkout-time materialization mismatch (111 files).** `git config
--show-origin --get-all core.autocrlf` showed:

```
file:C:/Program Files/Git/etc/gitconfig	true
file:.git/config	false
```

System-scope Git for Windows defaults `core.autocrlf` to `true`. This
repository's local `.git/config` (set in an earlier pack) overrides it to
`false`, and local wins for any future checkout or commit -- but changing
`core.autocrlf` is **not retroactive**. This working tree was originally
checked out while the effective value was still `true` (system default,
before the local override existed), so every text file was materialized on
disk with CRLF at that time. Setting `core.autocrlf=false` afterward
stopped further conversions but did nothing to the files already sitting
on disk as CRLF. `core.eol` and `core.safecrlf` are unset anywhere
(`git config --show-origin --get-all core.eol` / `core.safecrlf` both
exit 1, "not found" -- confirmed, not a config source for this). No
`.gitattributes` exists anywhere in the tree (searched the full working
tree, none found, including nothing from upstream).

`git ls-files --eol` distinguishes a real content difference from a
checkout-only one: `i/` describes what's committed (the index/blob),
`w/` describes the working-tree file. Before any fix: 111 files showed
`i/lf w/crlf` -- blob genuinely LF, disk copy CRLF, a pure checkout
artifact invisible to `git status`/`git diff` because git's stat-cache
short-circuit treats an untouched file as unchanged without re-hashing it
(this is what made `git status` "silent" until a file was actually
re-written, at which point the real byte difference surfaced and the
whole file appeared as changed).

**2. Genuinely CRLF-committed blobs (13 files, all under `ur10e/`).**
Separately, `git ls-files --eol` showed 13 files with `i/crlf w/crlf` --
CRLF *inside the committed blob itself*, not a checkout artifact. This is
not a git-configuration problem: `core.autocrlf=false` does not convert
anything at `add`/`commit` time either, so whatever bytes a tool wrote to
disk got committed verbatim. **Root cause: an external tool** (the file-write
tooling used in earlier TIP packs on this Windows machine) wrote these 13
files with CRLF line endings, and they were committed as-is:

```
ur10e/configs/env-laptop.txt
ur10e/configs/requirements-laptop.txt
ur10e/results/baselines.json
ur10e/results/compression_quality.json
ur10e/results/cut_parameters.csv
ur10e/results/data_check.md
ur10e/results/delta_conversion.md
ur10e/results/episode_video_map.csv
ur10e/results/metadata_tiling.csv
ur10e/results/onset_alignment.csv
ur10e/results/representativeness.json
ur10e/results/split.json
ur10e/src/to_v21.py
```

None of the 13 are upstream files (all newly created under `ur10e/` by
earlier packs) and none are `.sh` scripts, so this specific set did not
put a broken shell script in front of Colab -- but it is the same failure
mode that would, if it ever hit a script: a tool writing CRLF, with
nothing in the git/commit pipeline to catch or convert it.

## Fix

1. `core.autocrlf=false` in local `.git/config` was already correct (set
   by an earlier pack) and confirmed to be the winning value; nothing to
   change there.
2. Checkout-artifact files (case 1): `git status` confirmed clean, then
   `git rm --cached -r . && git reset --hard HEAD` re-synced the working
   tree to the real (LF) blob content for all 111 affected files. No
   commit needed for this step -- it only changes what's on disk to match
   what was already committed.
3. Genuinely-CRLF-committed files (case 2): the 13 files were rewritten
   byte-for-byte with `\r\n` -> `\n` and committed
   (`fix(repo): normalize working tree line endings to match committed
   blobs`). Verified EOL-only: every file's diff showed equal insertions
   and deletions (1992/1992 total across the 13 files), and one file's
   diff was inspected directly to confirm no textual content changed,
   only line endings.
4. No `.gitattributes` was created. Upstream has none; adding one would be
   a permanent, unnecessary divergence from upstream, which this project
   avoids by design.

## Verification -- end-to-end proof, not just configuration

Appended one English comment line to `starVLA/dataloader/gr00t_lerobot/datasets.py`
(an upstream file untouched by any pack before this one):

```
 starVLA/dataloader/gr00t_lerobot/datasets.py | 1 +
 1 file changed, 1 insertion(+)
```

One line, not a whole-file rewrite. Reverted with `git checkout --
starVLA/dataloader/gr00t_lerobot/datasets.py` immediately after.

After the fix: `git ls-files --eol | grep -c "w/crlf"` reports **0**.

## Warning for future packs

**Do not write files in this repository with PowerShell's `Out-File`,
`Set-Content`, `>`/`>>` redirection, or any tool that defaults to CRLF on
Windows.** This repo's commits are LF-only and there is no
`.gitattributes` net to catch a CRLF write before it lands in a commit --
the only thing that caught it this time was a manual `git ls-files --eol`
audit. Before committing any newly-created or edited file, verify it has
no `\r\n` bytes (e.g. `git diff --stat` against the previous commit should
show only real content lines changed, never the whole file).
