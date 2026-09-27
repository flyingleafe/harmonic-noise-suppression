---
experiment: nv3r4_hard_scv2
training_config: conf/experiment/nv3r4_hard_scv2.yaml
batch: docs/experiments/noise-model-v3.md
---

## Motivation

The hard SCv2 arm on the round-4 noise model and the round-4 hard policy
(`docs/experiments/noise-model-v3.md` § Round 4). Round 4 is the rig alone —
no block wander — under priors re-centred on measurements, and it meets the
fit criteria: both parity bars, the Michael's LTAS proxy under the legacy
fit's (1.281 against 1.332 dB). The hard bank adds three things the earlier
hard arms lacked: trajectories on the legacy hard hover range (35–95 rev/s,
not the hyperprior's 142-median), rotors kept apart by rejection, and every
rotor of every entry with identifiable harmonics (the tonality guard). Read
against `nv3r3_hard_scv2` (7.79), `nv2_hard_scv2` (7.06) and the legacy hard
arm (5.41 rev/s, the best hard arm so far).

## Setup

Fits: `results/noise_v3/fits_r4/` (DREGON round 4b, Michael's round 4c;
`MANIFEST.md`). No fold: a rig-only record renders as its expectation.
Bank: `noise_v3r4_hard_n2048.json`, `scripts/noise_v2_build_bank.py --preset
hard --generation v3r4` on `uni-cpu` (job `nv3r4-bank-hard`): 2048 path draws
at spread 3.0, seed 20260921, the v2 widths and guards plus the policy that
generation `v3r4` makes mandatory (`rig_sampler.ROUND4_BANK_GENERATIONS`):
speed exponents pinned to the aeroacoustic prior centre (6, 6) with each
anchor's static fraction, Michael's standby payload on every entry, the
per-rotor tonality guard (≥ 1 comparable order per rotor on every real cruise
pattern at its own speed and at 35 / 50 / 65 rev/s), the array response by
its own coin (p 0.5).

**Exact diff against `nv3r3_hard_scv2`: `experiment_name` and
`data.train.params.path`.** The stream `conf/online_mix/noise_v3r4_hard_5050.yaml`
differs from `noise_v3r3_hard_5050.yaml` in the bank line and in the `rps`
block: `hover_range: [35, 95]`, `rotor_sep: [2.0, 15.0]`, `mean_shift: [0, 0]`,
`rps_max: 120`.

```bash
scripts/noise_v2_submit_arms.sh --backend vast --gpu-type A100 --cpus 16 --mem 64 \
    --time 8h nv3r4_hard_scv2
```

## Conclusion

**Worse than every earlier hard arm.** Job `nv3r4-hard-scv2-0e4025` (Vast
A100, 4 h 51 min, early-stopped at round 118; the first placement
`-399ab9` failed on vast provisioning and never ran) selected round 37:
smoothed 8.02, **raw @ sel 8.02**, best raw 7.17 (no checkpoint). r1 / r2
/ r3 at sel 14.18 / 10.50 / 8.02. Against `nv3r3_hard_scv2` 7.79,
`nv2_hard_scv2` 7.06 and the legacy hard arm 5.41. The campaign doc's
§ Round 4 "Training arms" reads the pair together.
