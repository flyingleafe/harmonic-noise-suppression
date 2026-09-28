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

Job `nv2-hard-legacytraj-scv2-baf979` (Vast A100, HEAD `60273abd`),
134 rounds, selected round 92: **raw @ sel 4.96** (smoothed 4.96), best raw
4.46, r1 / r2 / r3 3.82 / 5.18 / 4.96, last raw 5.51, train loss (last 10
rounds) 0.6. Against `nv2_hard_scv2` 7.06 (r1 25.11) and the legacy hard arm
5.41.
<!-- source: results/noise_v3r4/transfer_controls.json -->

## Conclusion

**The trajectory sampler is the lever.** Swapping only the trajectories
takes the v2 hard bank from 7.06 to 4.96, far past the 6.56 threshold set
before the run, and below the legacy hard arm (5.41): the best hard arm of
the campaign and the most stable (r1 3.82). The legacy trajectories are
enough to close the whole v2-to-legacy gap. It does not show that the legacy
noise model played no part: the legacy arm also differs in bank format,
level rule and stream.
