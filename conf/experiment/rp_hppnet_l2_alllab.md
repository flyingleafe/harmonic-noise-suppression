---
experiment: rp_hppnet_l2_alllab
training_config: conf/experiment/rp_hppnet_l2_alllab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_hppnet_l2_alllab`

## Motivation

Label-altering augmentation reference: HPPNet at ladder level L2 (rp linear grid, `salience_layers_rp`), trained on the `rp_all_labelaug` stream; compared against `rp_hppnet_l2_all`. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR to be set from the family's LR range test.
