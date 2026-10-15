---
experiment: rp_simpleconv_all
training_config: conf/experiment/rp_simpleconv_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_simpleconv_all`

## Motivation

Regressor row: the published SimpleConv (Quiet Drones 2026), unchanged, trained as a direct rate regressor (`pit_mse` loss, `rps` metrics) on the `rp_all` stream. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR to be set from the family's LR range test.
