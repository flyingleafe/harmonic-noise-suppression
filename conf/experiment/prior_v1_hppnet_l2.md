---
experiment: prior_v1_hppnet_l2
training_config: conf/experiment/prior_v1_hppnet_l2.yaml
batch: docs/experiments/static-rig-profiles.md
---

## Motivation

The HPPNet-L2 twin of `prior_v1_scv2`: the same generic-prior stream
(`docs/experiments/static-rig-profiles.md` § "Generic drone prior") on the
salience architecture, so the prior's transfer is read on both model
families. Read against `rig_easy_hppnet_l2_unified` and the
`nv3r4_hard_*_hppnet_l2` arms.

## Setup

`rig_easy_hppnet_l2_unified` with the training stream
`conf/online_mix/noise_prior_v1_5050.yaml` (as `prior_v1_scv2`) and 12
workers. **Exact diff against `rig_easy_hppnet_l2_unified`:
`experiment_name`, `num_workers`, `data.train.params.path`.**

## Conclusion

Close to the fitted-rig streams on real data, far better on the static
synthetic views. vast A100 job `prior-v1-hppnet-l2-d8bcf7`, ran to the LR
floor (1.25e-4 by round 106, plateau 7.2 ± 0.4 Hz from round 110), 143 rounds, 8.4 h.
`val/real_overall` best **6.5 Hz at round 84** (r1 7.7, r2 8.1, nosource
4.3, source-present 10.2); `rig_easy_hppnet_l2_unified` 4.45 at round 72,
`nv3r4_hard_fullhyper_hppnet_l2` 4.24 at round 45. Static views 1.6 / 1.7 Hz
against 13.9–14.6 for both references: the prior stream teaches generic
line tracking that the fitted streams do not. The stochastic views (28.9 /
29.6 vs 19–28) are not read — Dmitrii: models that score well there output
average predictions (anti-validation). Throughput 192–200 s per 500-step
round (reference 180): the A100 is ~90 % fed. Full per-set table:
`docs/experiments/static-rig-profiles.md` § "Prior stream training".
