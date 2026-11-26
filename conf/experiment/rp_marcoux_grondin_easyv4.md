---
experiment: rp_marcoux_grondin_easyv4
training_config: conf/experiment/rp_marcoux_grondin_easyv4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_marcoux_grondin_easyv4`

## Motivation

The Marcoux & Grondin baseline (`marcoux_grondin_rps4`) in the paper grid's EASY synthetic setup: `rp_marcoux_grondin_all` with every real noise recording replaced by drones drawn around the DREGON and MD2 fits of noise model v4, rendered online (`conf/online_mix/rp_easyv4.yaml`). Model, loss, LR schedule, batch and speech mixing are the `all` run's; no label augmentation. Validation, stop and selection as `rp_scv2_easyv4` (`conf/validation/rp_real_v4easy.yaml`; controls `real_macro` and `near_macro`). Its `best_near_macro_median` checkpoint starts `rp_marcoux_grondin_easyv4_ft_lr5e4`.

## Conclusion

Defined 2026-11-27; submitted to `isambard`. Pending.
