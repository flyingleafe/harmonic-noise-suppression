---
experiment: rp_harmof0_l2_alllab
training_config: conf/experiment/rp_harmof0_l2_alllab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_harmof0_l2_alllab`

## Motivation

Label-altering augmentation reference: HarmoF0 at ladder level L2 (rp linear grid, `salience_layers_rp`, the published STFT front end), trained on the `rp_all_labelaug` stream (`rate_warp` p 1); compared against `rp_harmof0_l2_all` (2.62, the strongest baseline row) and against the earlier label-altering-augmented runs. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-16; not yet run.
