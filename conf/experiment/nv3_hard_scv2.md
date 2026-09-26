---
experiment: nv3_hard_scv2
training_config: conf/experiment/nv3_hard_scv2.yaml
batch: docs/experiments/noise-model-v3.md
---

## Motivation

This is the v3 twin of `nv2_hard_scv2`, which reached a best `real_overall` of
7.06 rev/s. The v2 rigs are swapped for the v3 fits, and the trunk, the
stream and the path construction stay the same. The question is whether the
v3 fits transfer better than the v2 ones when neither the comb nor the flight
envelope is privileged towards a rig the model is scored on.

The fits are the round-2 v3 fits (`results/noise_v3/fits_r2/`, commit
`0b87c5eb`) with the static part of their fitted block latents folded into the
rig (`results/noise_v3/diag/folded_r2/`, commit `1b698d78`); see
`nv3_easy_scv2.md`.

## Setup

**Exact diff against `nv2_hard_scv2`: `experiment_name` and
`data.train.params.path`.** Stream `conf/online_mix/noise_v3_hard_5050.yaml` is
`noise_v2_hard_5050.yaml` with only the bank line changed:
`dload:noise-v3-banks@8c62fa744282…/noise_v3_hard_n2048.json`. The bank was built by
`scripts/noise_v2_build_bank.py --preset hard --generation v3` with seed
20260921, spread 3.0 and the v2 widths and guards.

It is the CRUISE ↔ CRUISE path between the two v3 fits. The mixing coordinate
t ~ U[0, 1] runs over the common order range K = 1..81. The standby slot is
carried as in v2: Michael's standby with probability t, otherwise none. The
wander block is carried the same way (Michael's cruise block with probability
t, otherwise DREGON's), so every entry keeps a fitted wander block. Floor
control values are interpolated linearly in dB. DREGON's wind fades out
linearly in power towards Michael's, which has none.

The trajectories are the rig hyperprior (`posterior`), truncated by
`rps_max: 150`. The job runs on Vast (1× A100, 16 CPUs) instead of `uni`.

## Train

```bash
scripts/noise_v2_submit_arms.sh --backend vast --gpu-type A100 --cpus 16 --mem 64 \
    --time 8h nv3_hard_scv2
```

## Conclusion

**v3 (folded round-2 fits) does not transfer better than v2 on the hard bank.
It ties v2 overall and moves the error from DREGON to Michael's.** Job
`nv3-hard-scv2-149094` (vast A100, 2 h 17 min, early stop at round 64)
selected round 21: smoothed 8.60, **raw @ sel 7.01**, which is also the best
raw. r1 / r2 / r3 are 7.86 / 5.93 / 7.01. `nv2_hard_scv2` had 7.06 at round
15, with r1 25.11. Unlike v2, the single-mic view has converged at sel (r1/r2
1.33 against 2.68). The arm stays above `rig_hard_scv2_unified`'s 5.41 (best
raw 5.37) and more than twice the real reference's 3.11.

Decomposition of the selected file, which re-evaluates at 7.02: zero 2.18
(v2 1.25), standby 17.31 (6.16), ramp 13.52 (15.81), cruise 5.88 (7.76). By
rig, DREGON improves from 9.36 to 5.92 (cruise 10.52 → 6.29) and Michael's
worsens from 3.66 to 8.63 (standby 6.35 → 17.84, cruise 2.51 → 5.11). The
predicted rotor spread is now too wide at ramp (16.54 against the label's
7.85) and at cruise (22.23 against 13.79), and too narrow at standby (5.48
against 10.63). Post-sel drift: final raw 14.85 against best 7.01 (+112 %).
Full tables: `docs/experiments/noise-model-v3.md` § "Training arms (round-2
fits)" → Results.
