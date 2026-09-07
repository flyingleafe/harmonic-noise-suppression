# AGENTS.md — Harmonic Noise Suppression

Speech enhancement under harmonic noise from rotating sources at ultra-low SNR
(0 to −30 dB); drones are the instrumented case study. Durable goals,
constraints C1–C8 and the live portfolio of experimental bets: **`GOALS.md`**
(read it before any non-trivial research push). Campaign log:
`docs/experiments/` (one doc per campaign — motivation · setup · results ·
conclusion).

Every directory below has its own `AGENTS.md`; it is the truth for that
directory — read it before editing there. This file holds only the map,
the commands and the rules that are project-specific.

## Map

| Directory | What lives there |
|---|---|
| `src/framespec/` | Frame-shape vocabulary (`FrameSpec`, `SeriesSpec`); leaf package |
| `src/tasks/` | Task definitions + codecs (the Frame ↔ tensor seam): `speech_enhancement`, `rps_prediction`, `salience_rps`, `noise_generation` |
| `src/models/` | Models, spectral front-ends, RPS conditioning, unified registry (`models.registry.model_types()`) |
| `src/losses/`, `src/metrics/` | Frame-level losses/metrics; depend on `framespec`, never on `tasks` |
| `src/tracking/` | Rotor-speed tracking (Vold–Kalman, refinement, blind seeding). Pure array code, `Stage: td.Frame -> td.Frame`; **must not import** `data_processing`/`models`/`training` (import-linter) |
| `src/data_processing/` | `sources/` (raw datasets) → `derivations.py` (frozen derived-dataset specs) → `streams.py`/`online_mixing.py` (consumption). Driver: `scripts/derive.py` |
| `src/training/` | Training loop, Hydra config seam (`build_task_and_codec`, `instantiate_model`), R2 artifact upload |
| `src/zoo/` | Checkpoint registry: `zoo.checkpoints()`, `zoo.load(experiment)` → `FrameModel` |
| `src/plots/` | All plotting; `plots.dwym(frame)` is the front door, `plots.explore` the notebook primitives |
| `src/utils/` | Legacy helpers + `utils.gridrun` (the restartable parallel unit-JSON harness — reuse it for any sweep CLI) |
| `src/experiments/` | Research sandbox: may import anything, nothing imports it |
| `conf/` | Hydra tree: `experiment/` (+ sibling `.md` per experiment, enforced), `model/`, `data/`, `loss/`, `metrics/`, `optim/`, `online_mix/` |
| `scripts/` | Thin CLIs, never imported by `src/`: `derive.py`, `se_eval.py`, `rps_eval.py`, `table.py`, `bench.py`, `probe_ckpt.py`, config checks, campaign scripts |
| `tests/` | Pytest, one directory per `src/` package |
| `docs/` | Design docs, `docs/experiments/` campaign log, `docs/data-and-artifacts.md` (dload/R2/omnirun) |
| `writing/` | Papers (LaTeX), reports and slides (new ones: Quarto; legacy: Typst) |
| `notebooks/` | Thin drivers over `plots.explore`; logic goes to `src/` |

Signals (audio, RPS, IMU, VAD) are `tdseries` frames (`import tdseries as td`);
migration table in `docs/refactor-unified-framework.md`.

## Commands

```bash
python train.py experiment=<name>          # the only training entry point
python eval.py experiment=<name>           # the only evaluation entry point
python scripts/derive.py list|derive|adopt # the only dataset driver
dload pull <name>                          # prefetch a dataset (R2 bucket ml-data-new; pins in dload.lock)
omnirun submit --backend <b> --gpus 1 --time 30m --yes -- python train.py experiment=<name>
omnirun ps | status <job> | logs -f <job> | wait <job> | pull <job>
```

omnirun backends (thin client of the hetzner daemon): `uni` (Slurm `sae`,
long GPU jobs), `uni-gpushort` (≤ 1 h), `uni-cpu` (CPU-only, `--gpus 0`),
`vast` (paid burst), `kaggle` (free CUDA validation). Needs a clean pushed
HEAD; `.env` ships automatically. No `omnirun ssh` in daemon mode.
Details: `docs/data-and-artifacts.md` § "Job running"; workflow: the
`run-experiment` skill.

## Rules

1. **Sync before analysis.** `omnirun pull <job>` for job outputs, `dload pull`
   for datasets, `zoo`/W&B for checkpoints. Never analyse a stale `results/`.
2. **Compute is finite.** One seed unless asked. No control arms that will not
   be reported. Heavy CPU work goes to `uni-cpu`, never the laptop. CUDA smoke
   tests go to `kaggle`/`uni-gpushort`, not the long queue.
3. **Comparable experiments.** A new run uses the data regime, split and
   evaluation protocol of the runs it will be compared against; the protocols
   are named in `docs/experiments/AGENTS.md`. Change the architecture, not the
   regime, unless the regime is the question.
4. **Worktrees.** Implementation work happens in `.worktrees/<name>` on a
   branch of the same name (`mk-worktree <name>`, logic in
   `scripts/mk-worktree.sh`, links the shared `.venv`/`.env`/data dirs). Start
   the agent **with cwd inside the worktree**; never edit `.worktrees/*` paths
   from the main checkout — the worktree's own `src/` must be first on
   `PYTHONPATH` (the `.envrc` does this) or you silently import main's code.
5. **AGENTS.md stays a map.** Per-directory files are ≤ 10 KB (pre-commit
   enforces): purpose, module map, contracts, entry points, gotchas ≤ 3 lines
   each. Narrative, history, results and tutorials go to `docs/`.
6. **Experiment doc pairing.** Every `conf/experiment/<name>.yaml` has a
   sibling `<name>.md`; every campaign has a doc in `docs/experiments/`.

## Philosophy (earned; do not relitigate)

- Off-the-shelf first: before building infrastructure (schedulers, runners,
  sync, caching, config), name ≥ 2 existing tools and why each is rejected.
- Jobs are shell commands; structured inputs are the training script's
  concern via Hydra. No DSLs for what shell does.
- Bespoke code only where the project is novel (the tracking stack, the
  models, the agent that watches training). Owned LOC is a liability.
- Rewrites interrogate the interface first, the implementation second.
- Decompose before prescribing: "what would it mean to achieve this?" and
  "what would it take?" until a verifiable spec exists; only then plan.

## Tool quirks

- pyright indexing is slow here: `lsp references` may time out at 20 s; pass a
  larger `timeout` or fall back to `grep` for callsites.
- Browser verification uses Brave via CDP (user-wide policy), not Playwright.
- Tests deselect slow/GPU markers by default — `tests/AGENTS.md`.

## References

- Paper 1: Liu et al., "Edge-Deployed Band-Split RoPE Transformer for Ultra-Low SNR UAV Speech Enhancement", Drones 2025.
- Paper 2: RPS-conditioned speech enhancement (inspired by Gulli et al., EURASIP 2025).
