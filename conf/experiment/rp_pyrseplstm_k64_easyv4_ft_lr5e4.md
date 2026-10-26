---
experiment: rp_pyrseplstm_k64_easyv4_ft_lr5e4
training_config: conf/experiment/rp_pyrseplstm_k64_easyv4_ft_lr5e4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrseplstm_k64_easyv4_ft_lr5e4`

## Motivation

The paper grid's "easy, tuned" row for HRSPNet: `rp_pyrseplstm_k64_easyv4` fine-tuned on the real stream `rp_all` without label augmentation, from its synthetic-selected checkpoint (`checkpoint: best:near_macro@rp_pyrseplstm_k64_easyv4`, weights only), with the fine-tuning recipe (Dmitrii, 2026-10-26): a fresh AdamW at max LR 5e-4, 1 000 warm-up steps and a cosine to the 20 000-step cap; validation is the real panel (`rp_real`, control `real_macro`). The reference is the better of `rp_pyrseplstm_k64_all` / `rp_pyrseplstm_k64_alllab`.

## Conclusion

Pending.
