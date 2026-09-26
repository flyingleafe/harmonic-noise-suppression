---
experiment: nv3_easy_scv2
training_config: conf/experiment/nv3_easy_scv2.yaml
batch: docs/experiments/noise-model-v3.md
---

## Motivation

The v3 twin of `nv2_easy_scv2` (7.94 rev/s best `real_overall`). It asks
whether the noise-model-v3 fits transfer better than the v2 ones when
everything else is held fixed: the trunk, the stream and the way the bank is
sampled. v3 adds block wander (fresh OU dB latents per clip on every line,
rotor and floor control point), DREGON's per-mic wind, a spline-only floor
with a measured σ_B, and data-normalised channels.

The fits are the round-2 v3 fits (`results/noise_v3/fits_r2/`, commit
`0b87c5eb`) with the STATIC part of their fitted block latents folded into the
rig (`results/noise_v3/diag/folded_r2/{dregon_room2_floor,
michaels_fly125_cruise,michaels_fly125_standby}__flight_v3.json`, commit
`1b698d78`). The renderer draws zero-mean wander. Without the fold it drops
that static part and adds the prior's Jensen term on top, so a render sits
0.4–1.8 dB (DREGON) and 2.4–3.5 dB (Michael's cruise) above what the fit
explains above 300 Hz (`results/noise_v3/diag/ltas_bias_r2.json`).

## Setup

**Exact diff against `nv2_easy_scv2`: `experiment_name` and
`data.train.params.path`.** The parent chain is `rig_easy_scv2_unified` →
`real_r1_scv2` with `override /validation: rps_unified`, bfloat16, batch 128,
12 workers, 2 s clips and `samples_per_validation: null`. Selection is on
`real_overall`.

Stream `conf/online_mix/noise_v3_easy_5050.yaml` is
`noise_v2_easy_5050.yaml` with one line changed, the bank:
`dload:noise-v3-banks@8c62fa744282…/noise_v3_easy_n2048.json`. The pin is in the
policy and in `dload.lock`. The bank was built by
`scripts/noise_v2_build_bank.py --preset easy --generation v3` with the v2
construction and hyperparameters: seed 20260921, strength 3.0, the v2
widths, guards and LTAS envelope, and 1024 neighbourhood draws per rig, each
entry on its own `traj_rig`. How each v2 coordinate maps onto v3 is in
`experiments.noise_model.rig_sampler`:

- the floor shape and tilt draws go into the spline control values in dB;
- DREGON's wind takes the rig level plus v2's per-mic floor width;
- the wander block is kept as fitted, not perturbed;
- the trend guard is anchor-relative only on the rotors whose own fit falls by
  less than 3 dB: the folded Michael's cruise rotors 2 and 3, which fall by
  0.6 and 0.2 dB.

The job runs on Vast (1× A100, 16 CPUs) instead of `uni`. That is a placement
change only; the config is the same.

## Train

```bash
scripts/noise_v2_submit_arms.sh --backend vast --gpu-type A100 --cpus 16 --mem 64 \
    --time 8h nv3_easy_scv2
```

## Conclusion

**PENDING**. See `docs/experiments/noise-model-v3.md` § "Training arms
(round-2 fits)".
