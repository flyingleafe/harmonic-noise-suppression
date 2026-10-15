---
experiment: rp_scv2_dregon
training_config: conf/experiment/rp_scv2_dregon.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_scv2_dregon`

## Motivation

Claim 5 (no cross-drone transfer): the SimpleConv-v2 regressor (`hb_scv2_mag_nogate`, `pit_mse` loss, `rps` metrics), trained on the `rp_dregon` stream only and scored on both drones of the rp_real panel. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR to be set from the family's LR range test.
