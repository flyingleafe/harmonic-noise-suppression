---
experiment: rp_hppnet_l2_priorv4_ft_lr5e4
training_config: conf/experiment/rp_hppnet_l2_priorv4_ft_lr5e4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_hppnet_l2_priorv4_ft_lr5e4`

## Motivation

`rp_hppnet_l2_priorv4_ft` with the max learning rate at half the pretraining max (5e-4) instead of a tenth (1e-4) (Dmitrii, 2026-10-26). It starts after `rp_hppnet_l2_priorv4` finishes, from the same synthetic-selected checkpoint as the 1e-4 fine-tune. The stream, schedule shape (1 000 warm-up steps, cosine to the 20 000-step cap) and validation are unchanged.

## Conclusion

Pending.
