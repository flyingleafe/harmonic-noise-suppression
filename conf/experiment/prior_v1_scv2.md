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

Worse than the fitted-rig stream on real data, better on the third-party
synthetic panel. vast A100 job `prior-v1-scv2-0a585e`, 78 rounds / 39 000
steps, 3.9 h (stopped on the LR floor, no improvement after round 29).
`val/real_overall` best **9.8 Hz at round 29** (r1 8.3, r2 9.0, nosource
9.7, source-present 9.9), plateau 15 ± 1 from round 35 on; `nv3r4_easy_scv2`
4.63 at round 33, `nv3r3_easy_scv2` 4.68, real-data arm 2.99. The salv2
synthetic panel went the other way: `synthetic_overall` 10.1 vs 16.6
(static sets 3.5 vs 13–15 Hz). The stream is harder to fit — final train
loss 18.0 vs 7.2 — and the loader is the bottleneck: 158–172 s per
500-step round against 100 s GPU-bound for `nv3r4` (64 000 mixes × 23 ms
and 1600 renders × 0.4 s over 12 workers). Reading: the prior is wide in
directions the real validation rigs do not occupy (the k = 2 / slope /
odd-penalty laws), so capacity goes to rigs that never appear; the
per-set numbers in `docs/experiments/static-rig-profiles.md` § "Prior
stream training".
