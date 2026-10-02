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

Pending.
