---
experiment: rp_basicpitch_l0_all
training_config: conf/experiment/rp_basicpitch_l0_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_basicpitch_l0_all`

## Motivation

Claim 1 (unadapted salience models fail): Basic Pitch (hop 256) at ladder level L0, trained with the published grid's shared-map BCE (`salience_bce_basic_pitch`) on the `rp_all` stream. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR 3.0e-3 from the family's LR range test (`results/rp_lr_range/rp_basicpitch_l2_all.json`, read 2026-10-15: never rises, floor at 4.8e-3).
