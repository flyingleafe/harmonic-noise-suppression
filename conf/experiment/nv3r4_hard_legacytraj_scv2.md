---
experiment: nv3r4_hard_legacytraj_scv2
training_config: conf/experiment/nv3r4_hard_legacytraj_scv2.yaml
batch: docs/experiments/noise-model-v3.md
---

## Motivation

Does the full round-4 bank policy close the gap between the round-4 fits and
v2 on legacy trajectories? Without the tonality guard and the rest
(`nv3r4plain6_hard_legacytraj_scv2`) the round-4 fits scored 6.30 against
v2's 4.96 (`nv2_hard_legacytraj_scv2`). If this arm lands near 4.96, the
round-4 and v2 setups are at parity for SCv2 pretraining and the difference
to plain6 isolates the guard (with standby-always and array response).

## Setup

Bank: `noise-v3r4-banks` hard (unchanged: speed laws (6, 6) with each
anchor's static floor fraction, per-rotor tonality guard, Michael's standby
on every entry, array-response coin 0.5, plus the ordinary LTAS / trend /
width guards). Stream `conf/online_mix/noise_v3r4_hard_legacytraj_5050.yaml`
= `noise_v2_hard_legacytraj_5050.yaml` with only the bank changed (legacy
`full_flight` trajectories, per-window scale [0.45, 1.2]). One seed, Vast A100, 8 h cap. References: `nv3r4plain6_hard_legacytraj_scv2`
6.30, `nv2_hard_legacytraj_scv2` 4.96, `nv3r4_hard_fullhyper_scv2` 8.18. Scored by
`scripts/_nv3_arm_history.py`: raw `val/real_overall` at the smoothed
selection; under 0.5 rev/s reads as no effect.

## Results

Pending.

## Conclusion

Pending.
