---
experiment: rp_harmof0_l2_all
training_config: conf/experiment/rp_harmof0_l2_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_harmof0_l2_all`

## Motivation

Claim 2 (HPPNet best up to L2; L0→L1 vs L1→L2 attribution): HarmoF0 at ladder level L2 (rp linear grid, `salience_layers_rp`), trained on the `rp_all` stream. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR 1.0e-3 from the family's LR range test (`results/rp_lr_range/rp_harmof0_l2_all.json`, read 2026-10-15: flat 1e-4..1e-2, rises from 1.6e-2).
