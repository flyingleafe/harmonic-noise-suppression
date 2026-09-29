---
experiment: nv3r4plain6_hard_legacytraj_scv2
training_config: conf/experiment/nv3r4plain6_hard_legacytraj_scv2.yaml
batch: docs/experiments/noise-model-v3.md
---

## Motivation

Does the round-4 fit transfer as well as v2 once both fly the legacy hard
trajectories? `nv2_hard_legacytraj_scv2` (v2 bank on legacy trajectories)
scored 4.96, below the legacy hard arm (5.41). This arm keeps those
trajectories and replaces the bank with the round-4 fits drawn under the
same round-3 bank rules, with the speed laws pinned to (6, 6): the (2, 2)
pin of `nv3r4plain` was the v2 short-span contract, a mistake for a v3 fit.

## Setup

Bank: generation `v3r4plain6` (`rig_sampler.GENERATION_SPEED_LAW_PIN`):
round-4 anchors, speed laws (6, 6) with each anchor's static floor fraction,
no tonality guard, standby carried with probability t, no array response;
2048 path draws, spread 3.0, seed 20260921, published as
`noise-v3r4plain6-banks`. Stream
`conf/online_mix/noise_v3r4plain6_hard_legacytraj_5050.yaml` =
`noise_v2_hard_legacytraj_5050.yaml` with only the bank changed.
References: `nv2_hard_legacytraj_scv2` 4.96, legacy hard 5.41. One seed, Vast A100, 8 h cap. Scored by `scripts/_nv3_arm_history.py`:
raw `val/real_overall` at the smoothed-selected round; a difference under
0.5 rev/s reads as no effect.

## Results

Pending.

## Conclusion

Pending.
