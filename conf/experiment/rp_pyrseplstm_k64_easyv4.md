---
experiment: rp_pyrseplstm_k64_easyv4
training_config: conf/experiment/rp_pyrseplstm_k64_easyv4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrseplstm_k64_easyv4`

## Motivation

HRSPNet in the paper grid's EASY synthetic setup (Dmitrii, 2026-10-26): `rp_pyrseplstm_k64_all` with every real noise recording replaced by drones drawn around the two fits (`easy-v4-banks`: 1024 ungated prior-shaped perturbations each of the DREGON bench fit with its fitted gust wind and of the MD2 standby/cruise pair), each flown on its own drone's fitted trajectory law (`conf/online_mix/rp_easyv4.yaml`). Model, loss, LR schedule, batch and speech mixing are the `all` run's; no label augmentation. Validation is `rp_real_v4easy`: the run stops only when neither `real_macro` nor `near_macro` (mean of `near_dregon`, `near_md2`) has improved its 5-round median for 30 rounds; `best_near_macro_median` seeds `rp_pyrseplstm_k64_easyv4_ft_lr5e4`.

## Conclusion

Pending.
