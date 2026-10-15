# conf/ — Hydra config tree (the experiment system)

One experiment = one git-committed YAML in `conf/experiment/` composing the
component groups. Run: `python train.py experiment=<name>`; evaluate:
`python eval.py experiment=<name> [checkpoint=...]`. Contract:
`docs/refactor-unified-framework.md` § "Hydra config architecture".
Historical experiments are catalogued in `REPLICATION.md` (repo root).

## Groups

| Group | Meaning |
|---|---|
| `data/` | Dataset source (folder datasets via `frame_datasets`, dload streaming via `data_processing.streams`, online-mix wrappers referencing `conf/online_mix/*.yaml` policies) |
| `model/` | Task name + task params + model instantiation via `_target_` into `models.registry` — either `build_model` (native RPS registry, flat `params`), or `build_legacy_inline` (`params.model_type` + the ZFTurbo config tree inlined under `params.config`, routed through `models.registry.LEGACY_MODEL_BUILDERS`; this replaced the former `legacy_config_path`→`configs/*.yaml` indirection) |
| `loss/` | Loss composition (entries instantiate `src/losses` Frame adapters; multiple terms via `losses.composite`) |
| `metrics/` | MetricSuite membership (per-sample metrics of the non-panel validation path; logged, never selects) |
| `optim/` | Optimizer + step-based LR schedule: `adamw_cosine` (max `lr`, linear warm-up, cosine to `min_lr_frac·lr` at `total_steps = validation.max_optimizer_steps`); no monitor, no plateau |
| `logging/` | wandb (entity/project; run name = `experiment_name`) |
| `artifacts/`, `lora/` | R2 artifact uploads; LoRA seam |
| `validation/` | Optional full-panel RPS protocol. `rp_real` = the paper real-data panel (`rp-real-valid-v1`, 15 meta-predicate views/aggregates, control `real`, 500-step rounds, 100k-step cap, median/raw bests, saturation stop); `disabled` = the per-epoch MetricSuite path for other tasks. |

## Conventions (enforced by train.py)

- `experiment_name` names everything: wandb run, `results/<name>/`, R2
  prefix `artifacts/<name>/`. Results dir collision → error unless
  `resume=true`.
- Dirty git tree → hard error unless `allow_dirty=true`; commit hash is
  logged to wandb.
- `validate_only=true` runs pre-run spec validation (+ one-batch CPU smoke
  test) and exits — run this before submitting a GPU job.
- Rates in configs are exact rationals: write `[16000, 512]`-style pairs,
  never `31.25`.

## Streaming data configs (dload)

- **Stream pattern** — `conf/data/dregon_lm_v4_stream.yaml` streams
  DREGON-LM-V4 straight from the R2 store via
  `data_processing.streams.DloadFrameDataset` (no `datasets/` checkout
  needed); output Frames are structurally identical to the folder config
  (`conf/data/dregon_lm_v4.yaml`). Contract: train is an infinite shuffled
  stream (`repeat: true`), so the experiment **must set
  `samples_per_validation`** (same as the online-mix configs); valid is a
  finite ordered pass. Versions resolve via the repo-root `dload.lock` pin;
  add `version: <sha-prefix>` under `params` to override. RAM note: the
  shuffle buffer holds raw samples (~0.8 MB each for V4) — keep
  `shuffle_buffer` ≤ 512 on small machines (512 ≈ 400 MB).
- **`dload:` URI overrides** — any path-shaped value in `conf/data/*.yaml`
  (`data_dir`, `root`, `dregon_dir`, `michaels_dir`) also accepts
  `dload:NAME[@VERSION][/subpath]` (materialized once into the dload cache
  via `streams.resolve_source`); `noise_rps_dataset` configs additionally
  accept `frames:NAME[@VERSION]` specs for the published rich-frame datasets.
  So any folder-based data config runs checkout-free with a one-line CLI
  override, e.g. `data.train.params.data_dir=dload:DREGON-LM-V4-train`.

Legacy experiments (anything setting `optim.monitor`/`monitor_mode`/`patience`/
`factor`/`cooldown`, top-level `patience`, `early_stopping`, or the retired
`validation: rps_unified`) are frozen records of the plateau-LR protocol:
`train.py`/`eval.py` refuse them with a pointer to
`docs/experiments/paper-real-protocol.md`; `zoo.load` still builds their models
(schema-free compose, `~schema`). New real-data runs are `rp_*` experiments
selecting `validation: rp_real`; the root `epochs: 200` is their round cap.

## Adding an experiment

1. Pick/create component configs in the groups.
2. Add `conf/experiment/<name>.yaml` (`# @package _global_`, defaults
   overrides, `experiment_name: <name>`).
3. `python train.py experiment=<name> validate_only=true` on a machine
   with the dataset; commit the YAML before the real run.
