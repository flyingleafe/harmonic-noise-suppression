---
experiment: rp_hppnet_l2_dregonlab
training_config: conf/experiment/rp_hppnet_l2_dregonlab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_hppnet_l2_dregonlab`

## Motivation

Claim 6 (labelaug does not buy transfer): HPPNet at ladder level L2 (rp linear grid, `salience_layers_rp`), trained on the `rp_dregon_labelaug` stream and compared against `rp_hppnet_l2_dregon`. This file and `rp_hppnet_l2_md2lab` cover both directions, but only the direction with the nominally better transfer in claim 5 will run. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR to be set from the family's LR range test.
