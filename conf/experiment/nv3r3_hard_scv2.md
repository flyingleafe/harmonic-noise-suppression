---
experiment: nv3r3_hard_scv2
training_config: conf/experiment/nv3r3_hard_scv2.yaml
batch: docs/experiments/noise-model-v3.md
---

## Motivation

`nv3_hard_scv2` on the round-3 fits. Round 3b is better than round 2
(`results/noise_v3/diag/verdict_r3.md`); `nv3r3_easy_scv2.md` gives the
reasons. The question is the hard arm's: does the better fit transfer better
when neither the comb nor the flight envelope favours a rig the model is
scored on? It is read against `nv3_hard_scv2` (round-2 fits) and
`nv2_hard_scv2` (7.06 rev/s best `real_overall`).

The fits are the round-3b v3 fits (`results/noise_v3/fits_r3b/`, commit
`ffb61244`). Their static latent part is folded into the rig
(`results/noise_v3/diag/folded_r3/`, commit `b2c3f31b`). Anchor sha256
prefixes: DREGON `6b538d575bea`, Michael's cruise `54a895b50067` and
Michael's standby `d81e2ef23cd5`.

## Setup

**Exact diff against `nv3_hard_scv2`: `experiment_name` and
`data.train.params.path`.** Stream `conf/online_mix/noise_v3r3_hard_5050.yaml`
is `noise_v3_hard_5050.yaml` with only the bank line changed:
`dload:noise-v3r3-banks@500de70d2112…/noise_v3r3_hard_n2048.json`. The bank was built
by `scripts/noise_v2_build_bank.py --preset hard --generation v3r3`, which is
sampler generation v3 with the round-3b anchors: seed 20260921, spread 3.0,
the v2 widths and guards, and the anchor-relative trend guard.

The bank is the CRUISE ↔ CRUISE path between the two round-3b fits:

- t ~ U[0, 1] over the common order range K = 1..81.
- Michael's standby slot is carried with probability t, as is Michael's
  cruise wander block (otherwise DREGON's). Every entry therefore carries a
  fitted wander block with `uj_corr_oct: 1.5`, and the unchanged renderer
  draws correlated u_j from it.
- Floor control values are interpolated linearly in dB.
- DREGON's wind fades linearly in power.

The trajectories are the rig hyperprior (`posterior`), truncated by
`rps_max: 150`. The job runs on Vast (1× A100, 16 CPUs, 64 GB, 8 h), exactly
as the `nv3_*` pair.

## Train

```bash
scripts/noise_v2_submit_arms.sh --backend vast --gpu-type A100 --cpus 16 --mem 64 \
    --time 8h nv3r3_hard_scv2
```

## Conclusion

**No: this arm transfers worse than both earlier hard arms.** Job
`nv3r3-hard-scv2-5ea979` ran on Vast A100 for 2 h 15 min and early-stopped at
round 63. It selected round 14: smoothed 8.67, **raw @ sel 7.79**. Its best
raw was 7.46, at round 4. r1 / r2 / r3 at sel were 12.09 / 9.10 / 7.79. On
raw @ sel, `nv3_hard_scv2` (round-2 fits) scored 7.01, `nv2_hard_scv2` 7.06,
and the legacy hard arm 5.41.

The decomposition of the selected file re-evaluates it at 7.76: zero 4.07,
standby 6.79, ramp 14.85, cruise 8.20. The failing rig flips relative to
round 2. DREGON goes from 5.92 to 9.09 (standby 19.82, zero 5.02, cruise
9.38). Michael's goes from 8.63 to 5.82. The rotor spread moves closer to the
labels than in round 2 at ramp (10.41 against 16.54; label 7.85) and at
cruise (16.84 against 22.23; label 13.79). Post-sel drift is +52 % (final raw
11.31 against 7.46). The hard path bank has not turned the better fits into
better transfer. Full tables: `docs/experiments/noise-model-v3.md` §
"Training arms (round-3 fits)" → Results.
