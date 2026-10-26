---
experiment: rp_scv2_priorv4_ft_lr5e4
training_config: conf/experiment/rp_scv2_priorv4_ft_lr5e4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_scv2_priorv4_ft_lr5e4`

## Motivation

`rp_scv2_priorv4_ft` with the max learning rate at half the pretraining max (5e-4) instead of a tenth (1e-4) (Dmitrii, 2026-10-26). The 1e-4 fine-tune reached 3.18 Hz `real_macro`, its best round at step 3000, against 2.65 for the real-only reference. The warm start, stream, schedule shape (1 000 warm-up steps, cosine to the 20 000-step cap) and validation are unchanged.

## Conclusion

Pending.
