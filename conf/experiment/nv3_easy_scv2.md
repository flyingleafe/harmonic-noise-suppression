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

**v3 (folded round-2 fits) transfers better than v2 on the easy bank, but it
does not beat the hand-written legacy arm.** Job `nv3-easy-scv2-9bc4c1` (vast
A100, 5 h 51 min, early stop at round 174) selected round 126: smoothed 6.88,
**raw @ sel 6.50**, best raw 6.04 at round 80, r1 / r2 / r3 8.61 / 6.61 /
6.50. Against `nv2_easy_scv2`'s 7.94 raw @ sel and 7.18 best raw that is
−18 %. It stays above `rig_easy_scv2_unified`'s 6.09 (best raw 5.72) and
about twice the real reference's 3.11.

The four-regime decomposition re-evaluates the selected file at 6.83, not
6.50. The file is the right one (round 126); the likely cause is bf16
validation on the A100 against fp32 on Kaggle [INFERENCE]. Its cells:
zero 1.84 (v2 10.64), standby 13.02 (11.05), ramp 12.97 (11.12), cruise 6.39
(6.87). By rig: DREGON 7.10 (9.68), Michael's 6.43 (5.52). The gain is
DREGON's stopped rotors, zero 14.14 → 2.14, plus DREGON cruise, 8.85 → 7.75.
Michael's gets worse in every regime except zero. Standby stays the weak cell,
with a rotor spread of 4.39 against the label's 10.63. Post-sel drift: final
raw 8.18 against best 6.04 (+35 %). Full tables:
`docs/experiments/noise-model-v3.md` § "Training arms (round-2 fits)" →
Results.
