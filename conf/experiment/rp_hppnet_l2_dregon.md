---
experiment: rp_hppnet_l2_dregon
training_config: conf/experiment/rp_hppnet_l2_dregon.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_hppnet_l2_dregon`

## Motivation

Claim 5 (no cross-drone transfer): HPPNet at ladder level L2 (rp linear grid, `salience_layers_rp`), trained on the `rp_dregon` stream only and scored on both drones of the rp_real panel. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR to be set from the family's LR range test.
