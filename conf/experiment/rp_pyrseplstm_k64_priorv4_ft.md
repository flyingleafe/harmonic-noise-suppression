---
experiment: rp_pyrseplstm_k64_priorv4_ft
training_config: conf/experiment/rp_pyrseplstm_k64_priorv4_ft.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrseplstm_k64_priorv4_ft`

## Motivation

The paper grid's "tuned" row for HRSPNet: `rp_pyrseplstm_k64_priorv4` (wide-prior v4 pretraining) fine-tuned on the real stream `rp_all` without label augmentation, from its synthetic-selected checkpoint (`checkpoint: best:uniform@rp_pyrseplstm_k64_priorv4`, weights only), with a fresh AdamW at max LR 1e-4, 1 000 warm-up steps and a cosine to the 20 000-step cap; validation is the real panel (`rp_real`, control `real_macro`). The reference is the better of `rp_pyrseplstm_k64_all` / `rp_pyrseplstm_k64_alllab`.

## Conclusion

Never ran: `rp-pyrseplstm-k64-priorv-6d5279` failed at start on a GitHub fetch error (2026-10-26) and was not resubmitted, because 5e-4 became the fine-tuning recipe (`rp_pyrseplstm_k64_priorv4_ft_lr5e4`; Dmitrii, 2026-10-26).
