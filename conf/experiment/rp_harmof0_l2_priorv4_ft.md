---
experiment: rp_harmof0_l2_priorv4_ft
training_config: conf/experiment/rp_harmof0_l2_priorv4_ft.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_harmof0_l2_priorv4_ft`

## Motivation

The paper grid's "tuned" row for HarmoF0-L2: `rp_harmof0_l2_priorv4` (wide-prior v4 pretraining) fine-tuned on the real stream `rp_all` without label augmentation, from its synthetic-selected checkpoint (`checkpoint: best:uniform@rp_harmof0_l2_priorv4`, weights only), with a fresh AdamW at max LR 1e-4, 1 000 warm-up steps and a cosine to the 20 000-step cap; validation is the real panel (`rp_real`, control `real_macro`). The reference is the better of `rp_harmof0_l2_all` / `rp_harmof0_l2_alllab`.

## Conclusion

Ran 2026-10-26 (`rp-harmof0-l2-priorv4-ft-969c0d`, W&B `vb5qn2ch`): `real_macro` 2.66 at step 19 000, against 2.62 real-only. Superseded by `rp_harmof0_l2_priorv4_ft_lr5e4` (2.59): 5e-4 is the fine-tuning recipe (Dmitrii, 2026-10-26).
