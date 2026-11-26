---
experiment: rp_marcoux_grondin_alllab
training_config: conf/experiment/rp_marcoux_grondin_alllab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_marcoux_grondin_alllab`

## Motivation

Table IV row with label-altering augmentation for the Marcoux & Grondin baseline (`marcoux_grondin_rps4`, see `rp_marcoux_grondin_all`), trained on the `rp_all_labelaug` stream (`noise_rate_warp`); compared against `rp_marcoux_grondin_all`. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-11-26; submitted to `isambard`. Max LR 1e-3: the paper states none, so the regressor rows' value.
