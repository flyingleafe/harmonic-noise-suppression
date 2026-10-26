---
experiment: rp_harmof0_l2_priorv4_ft_lr5e4
training_config: conf/experiment/rp_harmof0_l2_priorv4_ft_lr5e4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_harmof0_l2_priorv4_ft_lr5e4`

## Motivation

`rp_harmof0_l2_priorv4_ft` with the max learning rate at half the pretraining max (5e-4) instead of a tenth (1e-4) (Dmitrii, 2026-10-26). The 1e-4 fine-tune reached 2.66 Hz `real_macro` against 2.62 for the real-only reference. The warm start, stream, schedule shape (1 000 warm-up steps, cosine to the 20 000-step cap) and validation are unchanged.

## Conclusion

Ran 2026-10-26 (`rp-harmof0-l2-priorv4-ft-bd45da`, W&B `uny4qk1r`): `real_macro` 2.59 at step 11 500, against 2.62 real-only and 2.66 at 1e-4: level with the reference. MD2 improves (1.60 against 1.74), DREGON does not (3.59 against 3.50). The fine-tuning recipe from here on (Dmitrii, 2026-10-26).
