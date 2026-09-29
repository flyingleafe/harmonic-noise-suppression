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

Job `nv3r4-hard-fullhyper-hpp-544be8` (HEAD `d5286fcb`), 95 rounds, selected
round 45: **raw @ sel 4.24** (smoothed 4.58), best raw 4.24, r1 / r2 / r3
4.42 / 4.94 / 4.24, last raw 6.18. Training loss bottoms at round 20 (0.554)
and creeps up after (0.578 at round 94); `real_r1` is best at round 22
(3.84), `real_r2` at round 27 (3.87). Against `nv2_hard_hppnet_l2` 6.47.
<!-- source: results/noise_v3r4/r4traj_scores.json -->

## Conclusion

**The best synthetic-only arm of the campaign** (4.24; best SCv2 arm 4.96,
real-data HPPNet-L2 2.27). The round-4 hard bank beats the v2 hard bank on
HPPNet-L2 by 2.23 rev/s, far over the 0.5 one-seed threshold.
