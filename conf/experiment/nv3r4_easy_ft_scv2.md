---
experiment: nv3r4_easy_ft_scv2
training_config: conf/experiment/nv3r4_easy_ft_scv2.yaml
batch: docs/experiments/noise-model-v3.md
---

## Motivation

Stage 2 of the easy curriculum on the round-4 noise model: does pre-training
on the round-4 easy bank give a better real-audio initialisation than the v2
easy bank? Stage 1 (`nv3r4_easy_scv2`) scored 6.10 synthetic-only, level with
the legacy easy arm (6.09) and worse than round 3b (4.68), but its `r1` view
(3.54) is the best any synthetic arm produced. The curriculum reading is the
one that counts when real audio is available.

## Setup

`nv2_easy_ft_scv2` with only the warm start changed:
`checkpoint: best:real_overall@nv3r4_easy_scv2`, resolved to
`r2://ml-data/artifacts/nv3r4_easy_scv2/checkpoints/best_real_overall.ckpt`
(round 36 of job `nv3r4-easy-scv2-c85aa7`), loaded `strict=False` with a
fresh optimizer, scheduler and early-stopping state. Parent
`real_r4_scv2_unified` (the R4 real pool, `conf/online_mix/hb_m3s2_dload.yaml`),
`patience: 20`, `lr: 1e-3`. No bank. One seed, Vast A100. Scored by
`scripts/_nv3_arm_history.py` (raw `val/real_overall` at the smoothed
selection). References: `nv2_easy_ft_scv2` 2.73, `nv2_hard_ft_scv2` 2.23,
`real_r4_scv2_unified` 3.11.

## Results

Pending.

## Conclusion

Pending.
