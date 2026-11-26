---
experiment: rp_marcoux_grondin_priorv4_mixlab
training_config: conf/experiment/rp_marcoux_grondin_priorv4_mixlab.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_marcoux_grondin_priorv4_mixlab`

## Motivation

The Marcoux & Grondin baseline (`marcoux_grondin_rps4`) trained from scratch on mixed data: 50 % the hard synthetic stream (`rp_priorv4`), 50 % the real stream with the label-altering rate warp (`rp_all_labelaug`), in one online mix (`conf/online_mix/rp_priorv4_mixlab.yaml`). Model, loss, LR schedule, batch, validation panel and stop rule are `rp_marcoux_grondin_priorv4`'s. Read against `rp_marcoux_grondin_alllab` and `rp_marcoux_grondin_priorv4_ft_lr5e4` on `real_macro`.

## Conclusion

Defined 2026-11-27; submitted to `isambard`. Pending.
