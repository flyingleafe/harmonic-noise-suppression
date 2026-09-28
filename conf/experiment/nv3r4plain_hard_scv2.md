---
experiment: nv3r4plain_hard_scv2
training_config: conf/experiment/nv3r4plain_hard_scv2.yaml
batch: docs/experiments/noise-model-v3.md
---

## Motivation

Separate the round-4 fit from the round-4 bank and trajectory policy. The
round-4 hard arm (`nv3r4_hard_scv2`, 8.02 rev/s) changed three things at once
against the round-3 hard arm (`nv3r3_hard_scv2`, 7.79): the anchor fits, the
bank policy (exponent pin (6, 6), standby on every entry, tonality guard,
array-response coin) and the trajectory policy (hover on 35–95 rev/s, rotor
separation by rejection, `rps_max` 120). It is also the first hard arm that
trained to a LOWER loss than its easy twin (last-10-round mean 5.5 against
7.3; v2 11.4 vs 1.3, round 3b 22.1 vs 1.9, legacy 12.0 vs 10.7), which
suggests the round-4 hard stream became easy. This arm keeps the round-4 fits
and restores the round-3 policy on both sides.

## Setup

Bank: generation `v3r4plain` (`rig_sampler.GENERATIONS`): the round-4 anchors
(`results/noise_v3/fits_r4`) under the round-3 bank policy — v2 short-span
pin (`SPAN_PIN`: `amp_exp` 2, `floor_exp` 2, `floor_static_rel` 2.5e-3),
Michael's standby carried with probability t, no tonality guard, no
array-response block (`array_response_p` 0 in the provenance). 2048 path
draws, spread 3.0, seed 20260921; `scripts/noise_v2_build_bank.py --preset
hard --generation v3r4plain` on `uni-cpu`
(`results/noise_v3r4plain/rig_sampler/jobs/build_hard_job.sh`), published as
`noise-v3r4plain-banks`. Stream: `conf/online_mix/noise_v3r4plain_hard_5050.yaml`
= `noise_v3r3_hard_5050.yaml` with only the bank changed (rig hyperprior,
`mean_shift` [−5, 5], `rps_max` 150). One seed, Vast A100, 8 h cap, the
`rig_easy_scv2_unified` recipe; scored by `scripts/_nv3_arm_history.py`.

## Results

Job `nv3r4plain-hard-scv2-7f3270` (Vast A100, HEAD `60273abd`; the first
placement's instance never accepted SSH, the retry ran), 83 rounds, selected
round 33: **raw @ sel 11.17** (smoothed 11.17), best raw 9.29, r1 / r2 / r3
29.27 / 15.48 / 11.17, last raw 13.78, train loss (last 10 rounds) 16.9.
Against `nv3r4_hard_scv2` 8.02 and `nv3r3_hard_scv2` 7.79.
<!-- source: results/noise_v3r4/transfer_controls.json -->

## Conclusion

**The round-4 policy helped; the round-4 fit alone did not.** Under the
round-3 policy the round-4 fits score 11.17, 3.15 worse than with the
round-4 policy (threshold: 0.5) and 3.38 worse than the round-3b fits under
the same policy. The reversal is bundled (speed-law pin, standby, guard,
array response, trajectory keys), so which change carried the 3.15 is not
separated; the trajectory keys point toward the legacy hover range, the
direction `nv2_hard_legacytraj_scv2` rewards.
