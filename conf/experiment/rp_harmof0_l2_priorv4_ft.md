---
experiment: rp_harmof0_l2_priorv4_ft
training_config: conf/experiment/rp_harmof0_l2_priorv4_ft.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_harmof0_l2_priorv4_ft`

## Motivation

The paper grid's "tuned" row for HarmoF0-L2: `rp_harmof0_l2_priorv4` (wide-prior v4 pretraining) fine-tuned on the real stream `rp_all` without label augmentation, from its synthetic-selected checkpoint (`checkpoint: best:uniform@rp_harmof0_l2_priorv4`, weights only), with a fresh AdamW at max LR 1e-4, 1 000 warm-up steps and a cosine to the 20 000-step cap; validation is the real panel (`rp_real`, control `real_macro`). The reference is the better of `rp_harmof0_l2_all` / `rp_harmof0_l2_alllab`.

## Conclusion

Pending.
