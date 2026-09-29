---
experiment: nv3r4_hard_legacytraj_hppnet_l2
training_config: conf/experiment/nv3r4_hard_legacytraj_hppnet_l2.yaml
batch: docs/experiments/noise-model-v3.md
---

## Motivation

Does HPPNet-L2 gain from the legacy trajectories as much as SCv2 did (v2
bank: 7.06 → 4.96)? On the unrestricted hyperprior it already reached 4.24,
the best synthetic-only arm. Dmitrii's prior: the gain is large mainly for a
regressor that falls back to a smooth comb around the mean, which smooth
labels reach faster, so HPPNet may gain less.

## Setup

Bank: `noise-v3r4-banks` hard (unchanged: speed laws (6, 6) with each
anchor's static floor fraction, per-rotor tonality guard, Michael's standby
on every entry, array-response coin 0.5, plus the ordinary LTAS / trend /
width guards). Stream `conf/online_mix/noise_v3r4_hard_legacytraj_5050.yaml`
= `noise_v2_hard_legacytraj_5050.yaml` with only the bank changed (legacy
`full_flight` trajectories, per-window scale [0.45, 1.2]). One seed, Vast A100, 12 h cap (`nv2_hard_hppnet_l2` took 10.8 h).
References: `nv3r4_hard_fullhyper_hppnet_l2` 4.24, real-data HPPNet-L2
2.27. Scored by
`scripts/_nv3_arm_history.py`: raw `val/real_overall` at the smoothed
selection; under 0.5 rev/s reads as no effect.

## Results

Pending.

## Conclusion

Pending.
