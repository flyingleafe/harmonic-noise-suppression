---
experiment: nv3r4_hard_fullhyper_hppnet_l2
training_config: conf/experiment/nv3r4_hard_fullhyper_hppnet_l2.yaml
batch: docs/experiments/noise-model-v3.md
---

## Motivation

The salience counterpart of `nv3r4_hard_fullhyper_scv2`: does HPPNet-L2
read the round-4 hard bank on the unrestricted hyperprior differently from
the direct regressor?

## Setup

`rig_easy_hppnet_l2_unified` with only the training stream changed, as
`nv2_hard_hppnet_l2`: stream
`conf/online_mix/noise_v3r4_hard_fullhyper_5050.yaml`. References:
`nv2_hard_hppnet_l2` 6.47 (163 rounds, 10.8 h, so this arm runs with a 12 h
cap instead of 8 h) and `nv3r4_hard_fullhyper_scv2`. One seed, Vast A100. Scored by `scripts/_nv3_arm_history.py`:
raw `val/real_overall` at the smoothed-selected round; a difference under
0.5 rev/s reads as no effect.

## Results

Pending.

## Conclusion

Pending.
