# src/training — unified training framework

The machinery behind the single root `train.py` / `eval.py` (Hydra). Design
contract: `docs/refactor-unified-framework.md`. There is exactly ONE
training loop in this repo; task differences live in configs, task codecs
(`src/tasks/codecs.py`), and loss/metric selection — never in forked
trainer scripts.

## Files

| File | Role |
|---|---|
| `config.py` | Dataclass structured configs registered in Hydra ConfigStore (`RootConfig`, Data/Model/Loss/Metrics/Optim/Wandb/Artifacts/Lora). `instantiate_target` resolves `_target_` entries and normalizes rate params to reduced `(num, den)` tuples — required for exact equality with `GridIndex` rates. |
| `validate.py` | Pre-run spec validation: dataset ⊇ model input; model output ∪ dataset ⊇ every loss/metric requirement; monitor metric exists; one-batch CPU smoke test. Runs at the start of train/eval; `validate_only=true` exits after it. |
| `loop.py` | The generic loop: map-style + iterable datasets, AMP, grad accumulation/clipping, optimizer-step validation cadence, atomic checkpointing, W&B logging. The legacy single-monitor path remains for other tasks; RPS reruns select `validation/rps_unified`. |
| `validation.py` | Full-panel RPS validation: one concatenated finite loader, GPU-vectorized MAE-optimal PIT, zero-copy nested index views, domain aggregates, log-scale median progress, any-subset LR/stopping state. |
| `stopping.py` | Legacy opt-in single-monitor median stopping. Unified RPS runs instead use `MultiMetricController` from `validation.py`; the two policies are mutually exclusive. |
| `artifacts.py` | Cloudflare R2 artifacts. Subset-best checkpoint aliases use S3 server-side copies of the already-uploaded `last.ckpt`, avoiding repeated checkpoint uploads. |
| `val_logging.py` | On-demand/final-evaluation validation media construction. The training loop never renders figures or audio previews during validation. |
| `lora.py` | LoRA config seam (`maybe_apply_lora`). Disabled by default; enabling raises NotImplementedError pointing at the legacy implementation (`git show d94ce9f:train.py`). |

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
