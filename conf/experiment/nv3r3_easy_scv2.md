---
experiment: nv3r3_easy_scv2
training_config: conf/experiment/nv3r3_easy_scv2.yaml
batch: docs/experiments/noise-model-v3.md
---

## Motivation

`nv3_easy_scv2` on the round-3 fits. Round 3b is the round-3 pick and is
better than round 2 (`results/noise_v3/diag/verdict_r3.md`,
`docs/experiments/noise-model-v3.md` § "Round 3"). In round 2 the static part
of the latents was large (DREGON u_j 0.88 of its mean square), the fitted
wander ran up to 21–22× the measured σ² (DREGON u_j, cruise d), and DREGON
lines at k 9–60 disappeared 57 % of the time against 69–73 % in the real
windows. Round 3b brings these to 0.00, ≤ 4.2× and 66–73 %. This arm asks
whether a better fit gives better transfer when everything else is held
fixed: the trunk, the
stream, the sampler, its seed and strength, and the renderer. It is read
against `nv3_easy_scv2` (round-2 fits) and `nv2_easy_scv2` (7.94 rev/s best
`real_overall`).

The fits are the round-3b v3 fits
(`results/noise_v3/fits_r3b/{dregon_room2_floor,michaels_fly125_cruise,michaels_fly125_standby}__flight_v3.json`,
commit `ffb61244`; u_j kernel prior wired in `d512827b`). Their static latent
part is folded into the rig exactly as round 2's was
(`results/noise_v3/diag/folded_r3/`, commit `b2c3f31b`). The folded anchors'
sha256 prefixes are DREGON `6b538d575bea`, Michael's cruise `54a895b50067`
and Michael's standby `d81e2ef23cd5`. The fold is close to the identity for
round 3b: it moves the profile by 0.2–0.4 dB rms, because round 3b's latents
hold almost no static part. After the fold, render minus fit is within
−0.8..+0.5 dB below 5 kHz (`results/noise_v3/diag/ltas_bias_folded_r3.json`).

## Setup

**Exact diff against `nv3_easy_scv2`: `experiment_name` and
`data.train.params.path`.** Against `nv2_easy_scv2` it is the same two
fields. The parent chain is `rig_easy_scv2_unified` → `real_r1_scv2` with
`override /validation: rps_unified`, bfloat16, batch 128, 12 workers, 2 s
clips and `samples_per_validation: null`. Selection is on `real_overall`.

Stream `conf/online_mix/noise_v3r3_easy_5050.yaml` is
`noise_v3_easy_5050.yaml` with one line changed, the bank:
`dload:noise-v3r3-banks@500de70d2112…/noise_v3r3_easy_n2048.json`. The pin is in the
policy and in `dload.lock`. The bank was built by
`scripts/noise_v2_build_bank.py --preset easy --generation v3r3`. `v3r3` is
sampler generation v3's construction with the round-3b anchors: seed
20260921, strength 3.0, the v2 widths, guards and LTAS envelope, the
anchor-relative trend guard, 16 attempts, and 1024 neighbourhood draws per
rig, each entry on its own `traj_rig`.

The round-3b wander blocks carry `uj_corr_oct: 1.5`, the u_j kernel prior.
The renderer (`render.py`, unchanged since `d512827b`) reads that key and
draws each block's colour vector as `A y`, correlated across the floor
control points. It therefore renders the fit's own prior. The round-2
payloads carry no such key, so the running `nv3_*` pair keeps independent
tracks with the same renderer code.

The job runs on Vast (1× A100, 16 CPUs, 64 GB, 8 h), exactly as the `nv3_*`
pair.

## Train

```bash
scripts/noise_v2_submit_arms.sh --backend vast --gpu-type A100 --cpus 16 --mem 64 \
    --time 8h nv3r3_easy_scv2
```

## Conclusion

**PENDING**. See `docs/experiments/noise-model-v3.md` § "Training arms
(round-3 fits)".
