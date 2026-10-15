---
experiment: rp_harmof0_l0_all
training_config: conf/experiment/rp_harmof0_l0_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_harmof0_l0_all`

## Motivation

Claim 1 (unadapted salience models fail): HarmoF0 at ladder level L0, trained with the published grid's shared-map BCE (`salience_bce_orig`) on the `rp_all` stream. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR 1.0e-3 from the family's LR range test (`results/rp_lr_range/rp_harmof0_l2_all.json`, read 2026-10-15: flat 1e-4..1e-2, rises from 1.6e-2).
