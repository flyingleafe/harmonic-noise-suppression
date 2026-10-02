---
experiment: prior_v1_scv2
training_config: conf/experiment/prior_v1_scv2.yaml
batch: docs/experiments/static-rig-profiles.md
---

## Motivation

The first training arm on the GENERIC DRONE PRIOR
(`docs/experiments/static-rig-profiles.md` § "Generic drone prior"): rigs
drawn from laws grounded in the static-rig campaign and recentred so that
Michael's flight rigs are plausible (χ²₇ tails 0.18 / 0.05), instead of
neighbourhood draws around two fitted rigs. The question is transfer to the
real validation panel against `nv3r4_easy_scv2` (same architecture, same
protocol, the round-4 bank).

## Setup

Stream `conf/online_mix/noise_prior_v1_5050.yaml`: bank
`prior_v1_n16384.npz` (`scripts/prior_bank_build.py`, uni-cpu job
`prior-bank-16k`, seed 20261002, gate 3 dB / >= 2 orders / 80 % of frames on
an 8-window hyperprior pool), every rig on the trajectory hyperprior
(`posterior`, hover 40-95 rev/s) at unit scale; 4 s renders reused 40 times,
each draw its own 2 s window. **Exact diff against `nv3r4_easy_scv2`:
`experiment_name` and `data.train.params.path`.**

## Conclusion

Pending.
