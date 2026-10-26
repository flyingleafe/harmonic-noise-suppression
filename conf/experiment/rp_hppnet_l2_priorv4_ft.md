---
experiment: rp_hppnet_l2_priorv4_ft
training_config: conf/experiment/rp_hppnet_l2_priorv4_ft.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_hppnet_l2_priorv4_ft`

## Motivation

The paper grid's "tuned" row for HPPNet-L2: `rp_hppnet_l2_priorv4` (wide-prior v4 pretraining) fine-tuned on the real stream `rp_all` without label augmentation, from its synthetic-selected checkpoint (`checkpoint: best:uniform@rp_hppnet_l2_priorv4`, weights only), with a fresh AdamW at max LR 1e-4, 1 000 warm-up steps and a cosine to the 20 000-step cap; validation is the real panel (`rp_real`, control `real_macro`). The reference is the better of `rp_hppnet_l2_all` / `rp_hppnet_l2_alllab`.

## Conclusion

Pending.
