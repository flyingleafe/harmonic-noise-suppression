---
experiment: rp_scv2_priorv4_ft
training_config: conf/experiment/rp_scv2_priorv4_ft.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_scv2_priorv4_ft`

## Motivation

The paper grid's "tuned" row for Conv+BiGRU: `rp_scv2_priorv4` (wide-prior v4 pretraining) fine-tuned on the real stream `rp_all` without label augmentation, from its synthetic-selected checkpoint (`checkpoint: best:uniform@rp_scv2_priorv4`, weights only), with a fresh AdamW at max LR 1e-4, 1 000 warm-up steps and a cosine to the 20 000-step cap; validation is the real panel (`rp_real`, control `real_macro`). The reference is the better of `rp_scv2_all` / `rp_scv2_alllab`.

## Conclusion

Ran 2026-10-26 (`rp-scv2-priorv4-ft-cd0055`, W&B `p8vl31te`): `real_macro` 3.18 at step 3000 (lowest 5-round median), against 2.65 real-only; no later round improves. Superseded by `rp_scv2_priorv4_ft_lr5e4` (2.55): 5e-4 is the fine-tuning recipe (Dmitrii, 2026-10-26).
