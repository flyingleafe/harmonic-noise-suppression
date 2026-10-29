---
experiment: rp_scv2_priorv4_mixlab
training_config: conf/experiment/rp_scv2_priorv4_mixlab.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_scv2_priorv4_mixlab`

## Motivation

Conv+BiGRU trained from scratch on mixed data (Dmitrii, 2026-10-29): 50 % the hard synthetic stream (wide-prior drones with per-clip wind, `rp_priorv4`), 50 % the real stream with the label-altering rate warp (`rp_all_labelaug`), in one online mix (`conf/online_mix/rp_priorv4_mixlab.yaml`; the warp touches the real chunks only). Model, loss, LR schedule, batch, validation panel and stop rule are `rp_scv2_priorv4`'s. Read against the real-only `rp_scv2_alllab` and the pretrain-then-tune `rp_scv2_priorv4_ft_lr5e4` on `real_macro`.

## Conclusion

Pending.
