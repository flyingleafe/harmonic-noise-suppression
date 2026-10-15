---
experiment: rp_hppnet_l0_all
training_config: conf/experiment/rp_hppnet_l0_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_hppnet_l0_all`

## Motivation

Claim 1 (unadapted salience models fail): HPPNet at ladder level L0, trained with the published grid's shared-map BCE (`salience_bce_orig`) on the `rp_all` stream. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR to be set from the family's LR range test.
