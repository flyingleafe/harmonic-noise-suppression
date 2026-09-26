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

**PENDING**. See `docs/experiments/noise-model-v3.md` § "Training arms
(round-2 fits)".
