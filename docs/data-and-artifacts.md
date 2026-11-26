# Data & Artifacts — R2 via dload + ArtifactStore

Single page. Read once, bookmark, move on.

## What lives where

| Thing                | Tool              | Where                                   |
|----------------------|-------------------|-----------------------------------------|
| Datasets (raw/processed) | **dload** (PyPI `dload-ml`) + R2 remote | `s3://ml-data-new/` + `dload.toml`/`dload.lock` in git |
| Checkpoints          | `ArtifactStore` (training loop) | `s3://ml-data/artifacts/<exp>/checkpoints/`; `zoo.load` resolves them |
| Metrics / logs       | wandb | wandb (summary carries the `r2/*` URIs) |
| Run records (resolved config, validation history, best index, val-round arrays) | `ArtifactStore` (training loop) | `s3://ml-data/artifacts/<exp>/` — § "Training artifacts" |
| Eval outputs, any other job result | `ArtifactStore` (`eval.py`, `se_eval.py`, `rps_eval.py`, `scripts/r2sync.py push`) | `s3://ml-data/artifacts/<name>/` mirroring `results/<name>/` — § "Job results" |
| omnirun job outputs  | **nothing** (`omnirun.toml`: `outputs = []`) | — |

One bucket (`ml-data-new`), one Cloudflare account. The remote endpoint and
bucket are committed in `dload.toml`; per-dataset version pins live in
`dload.lock` (managed by `dload pin` / `dload unpin`).

## One-time setup per machine

```bash
# 1. Install deps (uv does this via pyproject.toml; dload-ml is a direct dep).
uv sync

# 2. Put credentials in .env (never committed).
cp .env.example .env
# edit: fill AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, R2_ACCOUNT_ID,
#       WANDB_API_KEY, and keep AWS_DEFAULT_REGION=auto (R2 rejects real
#       AWS region names; only "auto" works).

# 3. Sanity check: resolved config, cache usage, remote reachability.
dload status
dload ls          # list datasets in the remote
```

On the laptop, `direnv` loads `.env` automatically (see `.envrc`). On a GPU
server, source it yourself (`set -a; . ./.env; set +a`) before invoking
`dload pull` or your training command.

## Workflow

### Overnight on CPU server — process and publish a dataset

For a new source, check `dload ls` **before fetching the upstream archive**.
Check candidate names, historical versions and the actual contents of raw
bundles: a binary-class mono derivative is not the native multichannel study.
Reuse an existing suitable Frames version. If only a raw tree exists, freeze
its `dload:NAME@<full version>` as the derivation parent; do not upload it
again. For an externally hosted, versioned release, the source builder can
read the original into worker scratch and publish only the Frames.

Run parallel conversions with **one publisher per dataset name**. On the CPU
worker, `python scripts/derive.py derive NAME --no-pin` publishes the version;
after checking its manifest, key counts and decoded samples from R2, run
`dload pin NAME FULL_HASH` in the consuming checkout. This pins without pulling any audio.
Use a different `DLOAD_CACHE_DIR` per concurrent writer: on shared GPFS,
simultaneous conversions hit `FileNotFoundError` during shard-cache eviction.
Reuse verified raw input trees, not a mutable eviction cache.

**Replacing a reviewed release.** A new pin alone does not reclaim R2 space.
On `uni-cpu`, retain a verified rollback copy, upload the exact reviewed
manifest and shards, and hash the shards read back from R2. Switch the lock
only after verification. Before retiring the predecessor, confirm the live
consumer resolves the new pin. Remove only the specified old manifest, derived
refs pointing to it, and its shards referenced by **no surviving manifest**.
`dload gc` is bucket-wide; it is not a version-scoped cleanup. Older versions
can share a shard, so the physical size delta need not equal the difference
between the two manifest totals.
See [acoustic source coverage and licences](data-catalog.md#additional-rotor-acoustics-recordings).

**Raw trees** are committed once with the CLI: `dload commit NAME --from
data/NAME` (key = relpath minus extension, field = the extension; the CLI does
not skip hidden files), then referenced by a `data_processing.sources` registry
entry's `raw_dataset`.

**Everything else is a derivation** — declared as a frozen spec in
`data_processing.derivations.SPECS` and materialized only through the single
driver (driver contract and layouts: `src/data_processing/AGENTS.md`; what is
pinned and why: `docs/data-catalog.md`; architecture:
`docs/refactor-data-pipelines.md`):

```bash
python scripts/derive.py list --check-remote -v   # specs + fingerprints + refs
python scripts/derive.py derive <NAME>            # materialize + commit + pin
python scripts/derive.py adopt  <NAME> --commit   # ref an existing historical pin
```

Manifest layouts: `sample-dir-v1` (DREGON-LM / DN-LM mixes), `tdframe-v1`
(rich per-recording Frames), `raw-files` / `pcm16-mono-v1` (byte passthrough /
decoded PCM cache).

Whatever the route, finish by pinning:

```bash
dload pin DREGON-LM-V4-train
git add dload.lock
git commit -m "dataset: DREGON-LM-V4-train new version"
git push
```

Every commit on the same dataset name produces a new version; the git diff on
`dload.lock` shows the version change. `dload info NAME` shows the manifest
and version history; `--recipe FILE` (or the Python-API equivalent) stores the
producing script/config verbatim alongside the version.

### Derived datasets (memoized generation pipelines)

Our mixed datasets (DREGON-LM, DN-LM) are also expressible as dload **derived
datasets** (`Repository.derive`, dload ≥ 0.3.0): a finite deterministic
pipeline that dload runs once, commits as a normal content-addressed version,
and memoizes by *fingerprint* — every later caller of the identical pipeline
hits the same snapshot instead of recomputing. The durable, reviewed
declarations live in `src/data_processing/derivations.py` (`SPECS`): one frozen
JSON spec per dataset (all generation params + seed + `recipe_version` +
resolved parent pins), plus module-level generator functions that reuse the
same per-sample mixing cores the disk-writing CLIs use. Driver:
`scripts/derive.py`.

```bash
python scripts/derive.py list --check-remote -v   # specs, fingerprints, ref status
python scripts/derive.py derive DN-LM-train        # materialize + pin a new dataset
python scripts/derive.py adopt DREGON-LM-V4-train  # dry-run: adopt an existing pin
python scripts/derive.py adopt DREGON-LM-V4-train --commit
```

Two modes:

- **`derive`** materializes a genuinely new dataset (e.g. **DN-LM**, absent from
  the bucket) — runs the generator, commits, forwards the `sample-dir-v1`
  manifest meta + the spec as recipe, and pins it. Must run on a box with the
  parent data + enough disk (materialization streams LibriSpeech/DREGON via
  dload); the small-RAM dev box cannot. Submit it via omnirun if needed.
- **`adopt`** (adopt-in-place) points a dataset's *derivation ref*
  (`datasets/<name>/derived/<fingerprint>`) at its existing `dload.lock` pin
  instead of re-materializing. Used for the historical `DREGON-LM-V4-*` uploads:
  re-deriving would upload a near-duplicate (our RNG is not byte-stable across
  environments). Offline except the ref write (additive, reversible, uploads
  nothing); **dry-run by default**, writes with `--commit`. The four active V4
  pins (`DREGON-LM-V4-{train,valid}`, `-michaels-{train,valid}`) are adopted.

**Bump a spec's `recipe_version`** on any behavioral change to a generator — the
fingerprint keys on the spec, so an unbumped edit silently serves the stale
snapshot. `derivations.py` stays importable without torch, so fingerprinting /
`adopt` run anywhere. Full design + gotchas: `docs/derived-datasets-plan.md`.

### `rps-traj-fits` — the fitted rotor-speed trajectory model

A raw tree (not a derivation): 15 KiB of JSON that the training streams read
every time they draw a trajectory, so it is versioned like data rather than
carried in git. Built from the campaign's results directory and committed with
the CLI:

```bash
python scripts/rps_traj_publish_fits.py --results results/rps_traj \
    --out data/rps-traj-fits
dload commit rps-traj-fits --from data/rps-traj-fits
dload pin rps-traj-fits && git add dload.lock
```

Contents: `manifest.json` (campaign round, publishing git commit, rigs, model
version, a note on sample rates), `fits/<rig>.json` — the seven per-rig fits
verbatim, each with the 32 model parameters, the warm-up idle level and the
rig's measured ESC floor/ceiling — `posterior.json` (the GLOBAL fit: a diagonal
Gaussian over 32 scale-free rig coordinates, one draw = one fresh drone) and a
`README.md`. The publish script is idempotent (an unchanged tree keeps its
`created` stamp, so re-running it does not push a new version) and refuses to
publish when the fits directory and the global fit disagree about which rigs
they cover.

Consumed by `rps.kind: fitted_traj` in the noise pools
(`src/data_processing/trajectory_model/source.py`); a policy points at it with
`fits: dload:rps-traj-fits`, and any directory with the same layout also works.
Campaign write-up: `docs/experiments/rps-trajectory-model.md`.

### Morning on GPU server — train

```bash
git pull                           # get the latest dload.lock pins
python train.py experiment=b1_dccrn_rps_dregon
# Remote GPU (Slurm / Colab / Kaggle): submit the same command via omnirun —
# see "Job running (omnirun)" below.
```

There is usually no separate download step: training code consumes
dload-managed datasets directly (next section), fetching shards lazily into
the local cache. `dload pull NAME` still exists for eagerly prefetching the
pinned version or inspecting data by hand.

### How training code consumes dload datasets

`src/data_processing/streams.py` is the project's only seam between dload's
`(key, {field: bytes})` sample world and the `td.Frame` data model (full API
in its module docstring):

- **`DloadFrameDataset`** — a torch `IterableDataset` usable directly as a
  Hydra `_target_` in `conf/data/*.yaml`. Streams a dataset by name; the
  version resolves via the `dload.lock` pin (override with `version:`).
  Decoding dispatches on the manifest `meta.layout`: `tdframe-v1` datasets
  decode via the generic Frame codec (`sample_to_frame`), everything else via
  the DREGON-LM file-stem convention (`decode_dregon_lm`). Stream knobs:
  `shuffle` (False / True / int seed), `shuffle_buffer`, `prefetch`, `take`,
  `repeat`. Reference config: `conf/data/dregon_lm_v4_stream.yaml` — train is
  an infinite shuffled stream (`repeat: true`), so the experiment **must set
  `samples_per_validation`** (same contract as online-mix). RAM note: the
  shuffle buffer holds *raw* samples (~0.8 MB each for V4) — 512 ≈ 400 MB;
  keep it ≤512 on small machines.
- **Pipeline combinators** — `to_frames` / `frame_windows` / `mix_frames` /
  `resample_frames` compose dload `Pipeline`s into windowed/mixed Frame
  streams; `iter_published_frames(name)` iterates a published `tdframe-v1`
  dataset as decoded Frames.
- **`dload:` URIs** — `resolve_source()` lets any path-shaped config knob
  (`data_dir`, `root`, `dregon_dir`, `michaels_dir`) accept
  `dload:NAME[@VERSION][/subpath]`: the dataset is materialized once into the
  cache (`ensure_local`, version-addressed, idempotent) and the local path is
  returned. Path-based loaders need no dload awareness.
- **`frames:NAME[@VERSION]` specs** — `noise_rps_dataset` and online-mix
  `kind: frames` noise sources consume the published rich-frame datasets
  (`DREGON-frames`, `michaels-frames`, `michaels-test-frames`) directly; the
  source-kind reference is `docs/refactor-data-pipelines.md` § "Online-mix
  policy reference".

Measured on the V4 stream (laptop ↔ R2): cold time-to-first-sample ~6 s per
shard, warm epochs ~18× faster than cold, ~48 MB/s sustained download;
bounded-budget cache eviction verified working.

Checkpoints and run records upload to R2 as training goes (§ "Training
artifacts"); there is no W&B checkpoint artifact.

### Evening on laptop — analyze

```bash
git pull
dload pull DREGON-LM                        # only to inspect raw data locally
python scripts/r2sync.py pull <exp>         # run records + eval outputs -> results/<exp>/
python -c "import zoo; fm = zoo.load('<exp>')"   # a checkpoint as a Frame -> Frame model
```

## Disk management

**dload shard cache** — content-addressed, deduplicated, eviction-bounded.
`dload status` shows usage; `dload cache` subcommands manage it. `dload gc`
deletes *remote* shards referenced by no manifest of any dataset (destructive
housekeeping — use deliberately). Two env vars control the cache (put them in
`.env`, see `.env.example`):

- `DLOAD_CACHE_DIR` — cache root; put it on a big/fast partition on GPU boxes.
- `DLOAD_CACHE_BUDGET` — local eviction budget (e.g. `20G`, or `unlimited`).

Per-machine recommendations:

| Machine | Setting |
|---|---|
| Laptop / small box | `DLOAD_CACHE_BUDGET` ≥ `1.5G` — the floor for streaming (must hold the `prefetch=3` shard window); keep `shuffle_buffer` ≤ 512 |
| Apocrita | `DLOAD_CACHE_DIR=/gpfs/scratch/acw592/dload-cache`, `DLOAD_CACHE_BUDGET=unlimited` (3 TB scratch) |
| Colab / Kaggle | ephemeral disks — forward overrides per job via `omnirun submit --env DLOAD_CACHE_BUDGET=...` if needed |

**wandb cache** (default `~/.cache/wandb/artifacts/`). Override with
`WANDB_CACHE_DIR=/big/disk/wandb`. To cap size:

```bash
export WANDB_CACHE_SIZE=50GB   # wandb enforces LRU above this
```

**On-the-fly streaming** — landed; it replaces the old rclone-mount trick.
`DloadFrameDataset` (see "How training code consumes dload datasets" above)
fetches shards lazily as the DataLoader reads and evicts under
`DLOAD_CACHE_BUDGET`. Keep the budget ≥ `1.5G` locally so the prefetch window
fits; on big scratch set it `unlimited` and warm epochs run at local-disk
speed (~18× faster than cold, measured).

## Job running (omnirun)

Remote jobs are submitted with **omnirun**. The laptop CLI is a *thin client*
of the scheduler daemon on hetzner (`[daemon] address = "10.100.0.1:8787"` over
WireGuard in `~/.config/omnirun/config.toml`); the daemon owns the job store,
the backend credentials and the backend definitions. `omnirun --local ...`
forces daemonless mode (uses the commented-out backends in the laptop config).
Backends as configured on the daemon:

| Backend | What it is |
|---|---|
| `uni` | Apocrita Slurm, `sae` partition (GPU, account `pilot_sae_gpu`), ≤ 10 days. Real training runs only — must request ≥ 1 GPU |
| `uni-gpushort` | Apocrita Slurm, `gpushort` partition, ≤ 1 h — smoke tests, kernel benchmarks, short fine-tunes; usually starts within minutes |
| `uni-cpu` | Apocrita Slurm, `compute` partition — **CPU-only**, ≤ 10 days. Dataset materialization, analysis sweeps, anything that would otherwise peg the laptop CPU; submit with `--gpus 0` |
| `isambard` | Isambard-AI phase 2 Slurm (`workq`, project `u6acn`), **aarch64** nodes with GH200 GPUs (96 GB), ≤ 24 h. One GPU brings 72 cores and 115 GB RAM. Needs a valid 12-hour login on hetzner — see below |
| `vast` | Vast.ai marketplace burst (`max_hourly` 2 $/h, auto-terminate) — when the cluster queue is long |
| `kaggle` | Kaggle free GPU quota — CUDA validation of new code without waiting for the queue; ~1 MB kernel source cap |

The committed repo-level `omnirun.toml` holds job *defaults* only
(`outputs = []` — collect nothing, see § "Job results"; 1 GPU, 1 h,
`env kind = uv`).

```bash
omnirun submit --backend uni-gpushort --gpus 1 --time 30m --name <name> -- \
    bash -c 'export PYTHONPATH="$PWD/src"; python train.py experiment=<name>'
omnirun ps                    # in-flight + recent jobs (-A: all repos)
omnirun status <job>          # one job (unique id prefix is enough)
omnirun logs -f <job>         # stream logs (there is no --tail)
omnirun wait <job>            # block until a state is reached
python scripts/r2sync.py pull <name>  # fetch published results (§ "Job results")
omnirun cancel <job>          # graceful; --force hard-kills
omnirun offers --gpus 1       # probe backends, submit nothing
omnirun backends check        # re-establish the SSH ControlMaster after expiry
```

- `omnirun ssh <job>` is **not available in daemon mode**; marketplace
  backends have no ssh entry at all. Inspect a running job through its logs,
  W&B and what it has published to R2, not by shelling in.

- **Always `export PYTHONPATH="$PWD/src"`** in the submitted command: the
  remote `.venv` is shared by every omnirun worktree and its editable
  install resolves `training`/`models` to whichever tree first built it,
  so a bare `python train.py` imports stale code from another commit
  (`rp_scv2_all`, 2026-10-15: `train.py` of `49d43423` importing
  `training.config` of `5ce0d62f`). Scripts that insert `src/` themselves
  (`scripts/lr_range_test.py`) are immune. There is no `--yes` option.

- **vast: always `--disk 64`** (plus `--gpu-type A100-80` and the `--env`
  credentials from `.env`). Each vast job is a fresh instance that builds
  the `.venv` (6.8 GB) and pulls the dload data (≈ 14 GB materialised +
  shards) onto the root disk; without a disk floor the offer omnirun picks
  can be too small and `uv sync` dies mid-install with
  `No space left on device`, leaving a torch without `torch._C`
  (`rp_hppnet_l0_all` resumes, 2026-10-15, twice).

- **`uni-gpushort` can hand out a V100 (sm_70)**, where the Triton tap
  kernel (`models.harmonic_ports.tap_conv_triton`) fails to compile
  (`PassManager::run failed`). Pyramid / HDC models there need
  `TAP_CONV_TORCH=1` in the submitted environment (the eager path; scores
  identical — `rp-panel-eval-dropk`, 2026-10-16).

- **`isambard` is aarch64 and its login expires every 12 hours.** The lock
  resolves the CUDA build there (`torch 2.7.0+cu128`, `torchaudio`, `triton`
  from PyTorch's cu128 index, `[tool.uv.sources]` in `pyproject.toml`); PyPI's
  linux-aarch64 torch wheel is CPU-only, and `torchcodec` is not installed on
  that platform. A new binary dependency needs a `manylinux_2_28_aarch64`
  wheel or it must build from source on the node. When the ssh certificate on
  hetzner has expired, `omnirun submit` leaves the job queued and prints
  `isambard offered no slot: … Isambard login expired — run
  `clifton-remote hetzner` on the laptop`; after that command the queued jobs
  place on their own within 5 minutes (`omnirun backends check` at once).
  Jobs already running keep running on the cluster; their status and logs
  resume after the re-login. The Triton tap kernel is untested on aarch64.

- **kaggle** (enabled in `~/.config/omnirun/config.toml`): good for short
  evals, not for anything that must start promptly. (a) Kaggle stores every
  push as the generic `machine_shape: "Gpu"` and ignores `--gpu-type`
  (`P100` was requested, the server metadata says `Gpu`); it assigns a P100
  or a 2×T4 at its discretion, and the 2×T4 pool can queue for hours — a
  2×T4 kernel just uses `cuda:0`. (b) The account allows **2 concurrent
  batch GPU sessions**, and a *queued* kernel holds one. (c) The installed
  `kaggle` client has no cancel endpoint, so `omnirun cancel` only marks the
  job locally and the kernel keeps its slot: **after cancelling a kaggle job
  run `uvx --from 'kaggle>=2.2' kaggle kernels delete
  flyingleafe/omnirun-<job-id>`** (with `yes` on stdin), or every later
  placement is rolled back with "Maximum batch GPU session count of 2
  reached" (visible only in the daemon's `/events`, 2026-10-18). Needs
  `TAP_CONV_TORCH=1` like the V100s (P100 is sm_60). `uni-cpu` is the
  better home for a single-checkpoint panel eval (~15 min at 8 threads).

- Requires a **clean, pushed HEAD** (`--push`/`--dirty` exist; prefer clean —
  `train.py` errors on a dirty tree anyway).
- On the cluster each job runs in a shared worktree
  `$PROJECT_ROOT/.trees/<sha12>`, reusing the checkout's `.venv`
  (`uv sync --frozen`).
- `.env` ships with every job automatically — R2 + WANDB credentials travel,
  so dload streaming and wandb logging work on any backend.
- `uni` (the `sae` partition) is GPU- and account-gated — reserve it for real
  GPU jobs; a submission without a GPU request is rejected (`QOSMinGRES`).
- **CPU-only jobs go to `uni-cpu`**, submitted with `--gpus 0`. Dataset
  materialization is the canonical case — e.g. build DN-LM from its
  derived-dataset spec:
  ```bash
  omnirun submit --backend uni-cpu --gpus 0 --time 3h --yes -- \
      bash -lc 'python scripts/derive.py derive DN-LM-train --no-pin && \
                python scripts/derive.py derive DN-LM-valid --no-pin'
  # then locally, once it succeeds: pull the new versions from R2 into the lock
  dload pin DN-LM-train && dload pin DN-LM-valid && git add dload.lock
  ```
- Shared same-SHA worktree wart: a crashed run's `results/<exp>` dir
  persists in the worktree and **poisons retries at the same SHA**
  (`FileExistsError`) — work around with a `results_root=...` override.
- After the local SSH ControlMaster dies, jobs show **LOST** from stale
  heartbeats — run `omnirun backends check` and verify via `sacct` on the
  cluster; a completed job may stay "lost" in `omnirun ps` while its
  published results are complete.
- The legacy helpers `scripts/sbatch.sh` / `scripts/sync_results.sh` were
  retired in the 2026-08 refactor (git history keeps them). Use omnirun for
  submission and `scripts/r2sync.py` for results.

## Job results (explicit publishing; omnirun collects nothing)

omnirun ships a job's files back only through its output globs — the repo
default `[job] outputs` in `omnirun.toml`, *replaced* (not extended) by a
non-empty `--outputs`; no exclude globs — plus whatever the job writes to
`$OMNIRUN_OUTPUT`. The daemon stores them under the job id in the bucket
`omnirun-artifacts` (`<job_id>/outputs/<relpath>`). Job-namespaced, that
channel re-shipped the shared worktree's sibling `results/` dirs with every
job and duplicated checkpoints the training loop had already uploaded
(2026-11: 356 GB, ~250 GB of it repeat copies). So the default is
`outputs = []`: **never pass `--outputs`, never write to `$OMNIRUN_OUTPUT`**;
results leave a job only by explicit upload through `ArtifactStore`.

Mirror rule (`training.artifacts.publish_results`): `results/<name>/<rest>`
⇄ `s3://ml-data/artifacts/<name>/<rest>`, so a pull restores the path its
producer wrote.

| Producer | Publishes | R2 key below `artifacts/` |
|---|---|---|
| `train.py` | resolved config, checkpoints, validation history, best index, val-round arrays | `<exp>/` — § "Training artifacts" |
| `eval.py` | `metrics.json`, `per_sample.csv`, `per_snr.csv` | `<exp>/eval/` |
| `scripts/se_eval.py` | its CSV (when written below `results/`) | `se_eval/<file>.csv` |
| `scripts/rps_eval.py` | `report.json`, `per_window.csv` | `rps_eval/<tag>/` |
| any other script | `python scripts/r2sync.py push results/<name>[/<sub>]` at the end of the job command | `<name>[/<sub>]/` |

```bash
# a campaign/analysis job: compute, then publish its own subtree
omnirun submit --backend uni-cpu --gpus 0 --time 2h -- bash -c \
  'export PYTHONPATH="$PWD/src"; python scripts/<campaign>.py --out results/<campaign>/<run> \
   && python scripts/r2sync.py push results/<campaign>/<run>'
# laptop: fetch (incremental — size+mtime, like `aws s3 sync`)
python scripts/r2sync.py pull <name>[/<sub>] ...     # -> results/<name>[/<sub>]
python scripts/r2sync.py pull <exp> --weights        # checkpoints too (usually: zoo.load)
```

- Push a run's own subdirectory, never `results/` itself (refused): sibling
  jobs share the worktree. Weights (`*.ckpt`, `*.pt`, …) and `wandb/` dirs
  are skipped; push exits 1 when it uploaded nothing (missing creds, empty
  dir), so the job fails visibly instead of silently losing its results.
- After a pull, training's history and best index sit at
  `results/<exp>/checkpoints/{validation_history.jsonl,best_checkpoints.json}`
  (their R2 keys), not at the run-dir root.
- Jobs from before 2026-11-26 captured `results/**`; `omnirun pull <job>`
  still reads what remains of `omnirun-artifacts/<job_id>/outputs/`.

## Training artifacts (checkpoints + val samples) → R2

Since the unified `train.py`, checkpoints **and** the run records upload
directly to the Cloudflare R2 bucket `ml-data` as training goes — the only
copy that leaves the job besides W&B. This is handled by
`src/training/artifacts.py::ArtifactStore` — not dload, not W&B artifacts.

- **Where** (`s3://ml-data/artifacts/<experiment_name>/`; `bucket`/`prefix`
  configurable, defaults `ml-data`/`artifacts`): `config.yaml` (resolved
  Hydra config, at start); `checkpoints/{last.ckpt,train_state.pt,
  validation_history.jsonl}` every round; `checkpoints/best_<score>_<kind>.ckpt`
  (server-side copies), `checkpoints/best_checkpoints.json` and
  `val_samples/round_<NNNN>.npz` on rounds where a best moved;
  `val_samples/epoch_<N>/` (on-demand media); `eval/` (`eval.py`).
- **Client**: `boto3.client("s3")` (same library dload uses; boto3 is a
  direct project dependency) pointed at the same R2 S3-compatible endpoint
  used by dload (`https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com`).
  Formerly `s3fs`, dropped because its `aiobotocore` pin capped `botocore`
  below dload-ml's `boto3` floor, forcing a `[tool.uv]`
  `override-dependencies` pin that made the project unresolvable by plain
  pip.
- **Credentials**: the same `.env` vars as the dload setup above —
  `R2_ACCOUNT_ID`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`. If any are
  missing, or `artifacts.enabled=false` in the Hydra config, every
  `ArtifactStore` method becomes a no-op (one log line, no exception) — a
  broken/absent artifact store never fails training.
- **wandb linkage**: each training run's wandb summary carries `r2/*` URIs
  pointing at the uploaded checkpoint/val-sample objects, so a wandb run page
  is still the starting point for finding an R2 artifact.
- Config seam: `conf/artifacts/r2.yaml` (selected via the `artifacts:` Hydra
  default group in `conf/config.yaml`).

## Gotchas

- **`dload commit` uploads immediately, but the pin is git's job.** After
  `dload commit` + `dload pin`, remember to `git add dload.lock` and push —
  the data remote and git are two separate things.
- **Reproducibility**: a job's dataset version is fully captured by the git
  commit of `dload.lock`. Pin experiments to a commit in the job record and
  you can always recreate.
- **wandb artifact size**: if checkpoints get huge (~GBs) and push free-tier
  limits, switch to *reference artifacts* that point at R2:
  `artifact.add_reference("s3://hns-research/checkpoints/...")`.
