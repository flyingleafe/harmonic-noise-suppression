---
experiment: nv3r4_hard_fullhyper_scv2
training_config: conf/experiment/nv3r4_hard_fullhyper_scv2.yaml
batch: docs/experiments/noise-model-v3.md
---

## Motivation

The round-4 hard bank (tonality guard on) without the round-4 trajectory
restrictions. `nv3r4_hard_scv2` (8.02) flew a restricted hyperprior (hover
35–95 rev/s, rotor separation, `rps_max` 120); this arm flies the
unrestricted one of `nv2_hard_scv2`, so it isolates those restrictions on a
fixed bank.

## Setup

Bank: `noise-v3r4-banks` hard (unchanged). Stream
`conf/online_mix/noise_v3r4_hard_fullhyper_5050.yaml` =
`noise_v3r4_hard_5050.yaml` with the round-3 trajectory block (posterior,
`mean_shift` [−5, 5], `rps_max` 150). References: `nv3r4_hard_scv2` 8.02,
`nv2_hard_scv2` 7.06. One seed, Vast A100, 8 h cap. Scored by `scripts/_nv3_arm_history.py`:
raw `val/real_overall` at the smoothed-selected round; a difference under
0.5 rev/s reads as no effect.

## Results

Pending.

## Conclusion

Pending.
