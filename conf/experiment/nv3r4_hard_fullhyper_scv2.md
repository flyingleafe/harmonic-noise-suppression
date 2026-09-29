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

Job `nv3r4-hard-fullhyper-scv-265e24` (HEAD `d5286fcb`), 105 rounds,
selected round 65: **raw @ sel 8.18** (smoothed 8.18), best raw 6.93, r1 /
r2 / r3 20.22 / 10.68 / 8.18, last raw 9.55. Against `nv3r4_hard_scv2` 8.02
and `nv2_hard_scv2` 7.06 (smoothed 9.17, last 12.18).
<!-- source: results/noise_v3r4/r4traj_scores.json -->

## Conclusion

**On par with v2; the trajectory restrictions did nothing.** Against the
restricted round-4 arm, 8.18 vs 8.02, within 0.5. Against v2, worse at
selection (8.18 vs 7.06) but better smoothed, best raw and last round.
