# src/training — unified training framework

The machinery behind the single root `train.py` / `eval.py` (Hydra). Design
contract: `docs/refactor-unified-framework.md`. There is exactly ONE
training loop in this repo; task differences live in configs, task codecs
(`src/tasks/codecs.py`), and loss/metric selection — never in forked
trainer scripts.

## Files

| File | Role |
|---|---|
| `config.py` | Dataclass structured configs, registered as Hydra group `schema/root` (`conf/config.yaml`: `- schema: root`; `zoo.load` composes with `~schema`). `instantiate_target` resolves `_target_` entries and normalizes rate params to reduced `(num, den)` tuples. `legacy_protocol_error(overrides, conf_dir)`: refusal message for experiments carrying keys/options the rp protocol removed (`train.py`/`eval.py` call it before Hydra). |
| `validate.py` | Pre-run spec validation: dataset ⊇ model input (+ panel plan builds); model output ∪ dataset ⊇ every loss/metric requirement; one-batch CPU smoke test. Runs at the start of train/eval; `validate_only=true` exits after it. |
| `loop.py` | The generic loop: map-style + iterable datasets, AMP, grad accumulation/clipping, `build_lr_scheduler` (linear warm-up → cosine to `min_lr_frac`, `LambdaLR` stepped once per ACCEPTED optimizer step, so `last_epoch == optimizer_steps`; state in `train_state.pt`), optimizer-step validation cadence, atomic checkpointing, W&B logging. No plateau scheduler, no epoch-patience early stop, no `best.ckpt`. |
| `validation.py` | Full-panel RPS validation. `build_validation_plan` reads every sample's `frame["meta"]` + target ONCE; views = `{where: {meta_key: value or [values]}, regime: zero/standby/ramp/cruise}` (index ranges are rejected; an empty view fails the plan). `validate_rps -> (scores, loss, ValidationSamples)`: per-frame error under the clip-optimal permutation, target regimes on the prediction grid, score = frame-weighted mean, clip stats = mean/population std/count of per-clip means (clip = `(dataset, meta.clip_index)`). Readouts: `rps_prediction` takes `rps_pred`, `salience_rps` the model's `decode_logits`. |
| `stopping.py` | `MultiMetricController`: per primary score a raw best (any lower round value) and a median best (5-round median ≥ 1 % lower, log scale); `stale_rounds` counts rounds since the median of ANY `control` score (a list) improved; stop = `rounds_completed >= min_rounds and stale_rounds >= final_patience` ("saturation"). Never touches the optimizer. |
| `artifacts.py` | Cloudflare R2 artifacts. Best-checkpoint aliases are S3 server-side `copy_object`s of the already-uploaded `last.ckpt`, never re-uploads. |
| `val_logging.py` | On-demand/final-evaluation validation media construction. The training loop never renders figures or audio previews during validation. |
| `lora.py` | LoRA config seam (`maybe_apply_lora`). Disabled by default; enabling raises NotImplementedError pointing at the legacy implementation (`git show d94ce9f:train.py`). |

## Run outputs (panel runs, `validation.enabled`)

- `last.ckpt` every round; per primary score `best_<slug>_median.ckpt` and
  `best_<slug>_raw.ckpt`, hard links to that round's `last.ckpt` (coincident
  bests share an inode; R2 via `alias_checkpoint`).
- `best_checkpoints.json`: `"<score>/median"|"<score>/raw" → {checkpoint,
  validation_round, optimizer_step, raw, median}`; `_warm_start`'s
  `best:<score>@<exp>` resolves the median file.
- `validation_history.jsonl` per round: `val/<score>`, `val/<score>_median`,
  `lr`, `optimizer_step`,
  `validation/{stale_rounds,rounds_completed,improved_median,improved_raw,stop_reason}`;
  no `_log` scores, no per-clip stats (read `val_samples/` for those).
- `val_samples/round_<NNNN>.npz` every round (`pit_mae`, `frame_err`,
  `frame_regime` int8, `clip_index`, `channel`); uploaded only on rounds
  where some best moved. A non-finite round stops before any write.

## Future-expansion seams (see design doc §"Future expansions")

- Multi-source under-annotated datasets: keep dataset construction behind
  DataConfig; nothing outside it may assume specific Frame entries.
- SSL objectives: losses are declarative; per-source loss applicability is
  the planned extension — don't hardcode loss↔dataset couplings.
- Joint/adversarial schemes: keep build→step→validate behind the current
  narrow seam; checkpointing/logging/artifacts must stay scheme-agnostic.

## Gotchas

- Dropped on purpose: multi-GPU `DataParallel` validation (decision
  2026-07-03), the 8-way `choice_loss` flag menu (→ `losses.composite`).
- Tests: run ONE file at a time under a hard cap —
  `bash -c "ulimit -v 3000000; timeout 180 uv run pytest tests/training/<file> -q -x"`.
  Development machines here are small; unbounded pytest has frozen a box.
- `tests/training/test_artifacts_r2_integration.py` hits the real bucket
  (skips without `R2_ACCOUNT_ID`); it cleans up after itself.
