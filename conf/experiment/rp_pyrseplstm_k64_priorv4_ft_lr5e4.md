---
experiment: rp_pyrseplstm_k64_priorv4_ft_lr5e4
training_config: conf/experiment/rp_pyrseplstm_k64_priorv4_ft_lr5e4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrseplstm_k64_priorv4_ft_lr5e4`

## Motivation

`rp_pyrseplstm_k64_priorv4_ft` (HRSPNet) with the max learning rate at half the pretraining max (5e-4) instead of a tenth (1e-4) (Dmitrii, 2026-10-26). It starts after `rp_pyrseplstm_k64_priorv4` finishes, from the same synthetic-selected checkpoint as the 1e-4 fine-tune. The stream, schedule shape (1 000 warm-up steps, cosine to the 20 000-step cap) and validation are unchanged.

## Conclusion

Running (`rp-pyrseplstm-k64-priorv-a462b2`, W&B `conb1f2j`): at step 13 000, `real_macro` 2.15 at step 8500 (lowest 5-round median), against 2.52 real-only, all of it on DREGON (3.05 against 3.89; cruise 1.94 against 2.97); MD2 level (1.25 against 1.14). The fine-tuning recipe from here on (Dmitrii, 2026-10-26). Final numbers pending.
