---
name: run-experiment
description: Train, evaluate or orchestrate an ML experiment in this repo — Hydra experiment configs, omnirun submission to the real backends, monitoring, result sync, analysis. Use whenever a model is to be trained or evaluated, locally or remotely.
---

# Run an Experiment

## Before submitting anything

State in one line what the run tests and which **existing** run it is compared
against. New runs must be *comparable*: same data regime (`conf/data` entry,
split, protocol) and training setup as the baseline they will be read against.
Change the architecture or the hypothesis under test, not the regime — unless
the regime itself is the experiment. Protocols/regimes are listed in
`docs/experiments/AGENTS.md`.

Compute conventions (the user's, not negotiable without asking):
- **One seed** unless the user asks for more. Multi-seed runs are the exception.
- Do not launch a control arm whose result is not going to be reported.
- **Nothing heavy on the laptop** — CPU sweeps and dataset builds go to `uni-cpu`.
- **CUDA smoke test of new code → `kaggle` or `uni-gpushort`**, never the long queue.

## Steps

1. **Prerequisites.**
   - Dataset published on dload? (`dload pull <name>`, a `*_stream.yaml` data config, or a `dload:` URI override) — `src/data_processing/AGENTS.md`.
   - Model registered? — `src/models/AGENTS.md` (`_target_`/keys).
   - Experiment config: `conf/experiment/<name>.yaml` **plus** its sibling `<name>.md` (pre-commit enforces the pair via `scripts/validate_experiment_docs.py`) — `conf/AGENTS.md`. Override on the CLI: `optim.max_epochs=50 data=<other>`.

2. **Run.** A job is the training command; there is no bespoke runner.
   ```bash
   python train.py experiment=<name>                       # local GPU only
   omnirun submit --backend uni-gpushort --gpus 1 --time 30m --yes -- \
       python train.py experiment=<name>                   # remote
   ```
   Backends (daemon-side; full table in `docs/data-and-artifacts.md` § "Job running"):
   `uni` (Slurm `sae`, long GPU jobs, ≥1 GPU required), `uni-gpushort` (≤1 h),
   `uni-cpu` (CPU-only, `--gpus 0`), `vast` (paid burst when the queue is long),
   `kaggle` (free GPU for CUDA validation). Requires a clean **pushed** HEAD;
   `.env` ships automatically so dload streaming + wandb work everywhere.

3. **Monitor.** `omnirun ps` · `omnirun status <job>` · `omnirun logs -f <job>` ·
   `omnirun wait <job>`. The CLI is a thin client of the hetzner daemon:
   `omnirun ssh` is unavailable, there is no `--tail`, `omnirun backends check`
   revives an expired SSH ControlMaster. Jobs showing LOST after a ControlMaster
   expiry are usually fine — check again after `backends check`.

4. **Evaluate.** `python eval.py experiment=<name>` (the single eval entry point;
   chain with `&&` after training when the run is short).

5. **Analyze.** Sync first — `omnirun pull <job>` collects `results/**` — then
   the `generate-model-comparisons` skill or `scripts/table.py`.

6. **Log.** Add the run to the campaign doc under `docs/experiments/`
   (motivation · setup · results · conclusion) — see `docs/experiments/AGENTS.md`.

## Pitfalls

- A crashed run's `results/<exp>` dir persists in the cluster worktree and
  poisons retries at the same SHA (`FileExistsError`) — override `results_root=`.
- `outputs = results/**` scoops sibling jobs' results into every `omnirun pull`.
- RPS experiments need `load_rps: true` in the data config.
- Kaggle kernels have a ~1 MB source cap: strip notebooks/writing/docs from the
  pushed snapshot when it fails to upload.
