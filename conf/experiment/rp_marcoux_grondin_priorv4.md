---
experiment: rp_marcoux_grondin_priorv4
training_config: conf/experiment/rp_marcoux_grondin_priorv4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_marcoux_grondin_priorv4`

## Motivation

The Marcoux & Grondin baseline (`marcoux_grondin_rps4`) in the paper grid's HARD synthetic setup: `rp_marcoux_grondin_all` with every real noise recording replaced by drones of the wide prior in noise model v4, rendered online (`conf/online_mix/rp_priorv4.yaml`). Model, loss, LR schedule, batch and speech mixing are the `all` run's; no label augmentation. Validation, stop and selection as `rp_scv2_priorv4` (`conf/validation/rp_real_priorv4.yaml`: real panel, `static_comb`, six synthetic sectors; controls `real_macro` and `uniform`). Its `best_uniform_median` checkpoint starts `rp_marcoux_grondin_priorv4_ft_lr5e4`.

## Conclusion

Defined 2026-11-27; submitted to `isambard`. Pending.
