---
experiment: nv3r4_easy_scv2
training_config: conf/experiment/nv3r4_easy_scv2.yaml
batch: docs/experiments/noise-model-v3.md
---

## Motivation

The easy SCv2 arm on the round-4 noise model (`docs/experiments/noise-model-v3.md`
§ Round 4): the rig alone, no block wander, priors re-centred on
measurements, Michael's channels normalised by the measured array response.
Round 4 meets the fit criteria (both parity bars; Michael's LTAS proxy 1.281
against the legacy 1.332 dB). The question is whether that carries into
transfer as round 3b's did: `nv3r3_easy_scv2` reached 4.68 rev/s, the first
synthetic-only SCv2 arm below the legacy pair (6.09 / 5.41). Read against
`nv3r3_easy_scv2` (4.68), `nv2_easy_scv2` (7.94) and the legacy easy arm (6.09).

## Setup

Fits: `results/noise_v3/fits_r4/` (DREGON round 4b, Michael's round 4c;
`MANIFEST.md`). No fold. Bank: `noise_v3r4_easy_n2048.json`,
`scripts/noise_v2_build_bank.py --preset easy --generation v3r4` on `uni-cpu`
(job `nv3r4-bank-easy`): 1024 DREGON + 1024 Michael's neighbourhood draws at
strength 3.0, seed 20260921, the v2 widths and guards, the speed exponents
pinned to (6, 6) with each anchor's static fraction (mandatory for
generation `v3r4`), each entry flown on its own rig's fitted trajectory;
Michael's entries carry the measured array response.

**Exact diff against `nv3r3_easy_scv2`: `experiment_name` and
`data.train.params.path`.** The stream `conf/online_mix/noise_v3r4_easy_5050.yaml`
differs from `noise_v3r3_easy_5050.yaml` in the bank line only.

```bash
scripts/noise_v2_submit_arms.sh --backend vast --gpu-type A100 --cpus 16 --mem 64 \
    --time 8h nv3r4_easy_scv2
```

## Conclusion

**No transfer gain over round 3b; level with the legacy easy arm.** Job
`nv3r4-easy-scv2-c85aa7` (Vast A100, 3 h 32 min, early-stopped at round 77) selected
round 36: smoothed 6.10, **raw @ sel 6.10**, best raw 4.63 (at a round no
checkpoint corresponds to). r1 / r2 / r3 at sel 3.54 / 4.78 / 6.10. Against
`nv3r3_easy_scv2` 4.68 (r1 5.22 / r2 4.74 / r3 4.68) and the legacy easy arm
6.09. The r1 view is the best of any synthetic arm; the r3 view is what
selects. Scores from `scripts/_nv3_arm_history.py` (the live R2 validation
history), read at the campaign doc's § Round 4 "Training arms".
