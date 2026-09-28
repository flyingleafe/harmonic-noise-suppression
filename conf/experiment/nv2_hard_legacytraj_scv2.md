---
experiment: nv2_hard_legacytraj_scv2
training_config: conf/experiment/nv2_hard_legacytraj_scv2.yaml
batch: docs/experiments/noise-model-v3.md
---

## Motivation

Isolate the trajectory sampler's share of the hard-arm gap. The legacy hard
arm (`rig_hard_scv2_unified`, 5.41 rev/s) beats the v2 hard arm
(`nv2_hard_scv2`, 7.06). The two differ in the noise model AND in the
trajectory sampler (the hand-written `full_flight` scaffold against the
fitted rig hyperprior). This arm keeps the v2 model and bank and swaps in the
legacy trajectory sampler. If it lands materially below 7.06, the hard-arm
problem is mostly the trajectories.

A second observation motivates the pair of controls: on round 4 the hard arm
reached a LOWER training loss than the easy arm (last-10-round mean 5.5
against 7.3); on every earlier pair hard trained higher (v2 11.4 vs 1.3,
round 3b 22.1 vs 1.9, legacy 12.0 vs 10.7).

## Setup

`nv2_hard_scv2` with only the stream's trajectory side changed:
`conf/online_mix/noise_v2_hard_legacytraj_5050.yaml` = `noise_v2_hard_5050.yaml`
with `rps: {kind: full_flight, aggressiveness: 1.0, flight_reuse: 32}`,
`drone_profile_range: [0, 1]`, `rps_scale_range: [0.45, 1.2]` — the keys of
`rig_hard_5050.yaml`. Bank `noise-v2-banks` hard (unchanged; `traj_rig` is
inert under `full_flight`). One seed, Vast A100, 8 h cap, the
`rig_easy_scv2_unified` training recipe. Scored by
`scripts/_nv3_arm_history.py` (raw `val/real_overall` at the smoothed
selection).

## Results

Pending.

## Conclusion

Pending.
