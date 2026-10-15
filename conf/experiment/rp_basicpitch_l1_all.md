---
experiment: rp_basicpitch_l1_all
training_config: conf/experiment/rp_basicpitch_l1_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_basicpitch_l1_all`

## Motivation

Claim 2 (HPPNet best up to L2; L0→L1 vs L1→L2 attribution): Basic Pitch (hop 256) at ladder level L1 (log_off layers, `salience_layers_log36_h256`), trained on the `rp_all` stream. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR 3.0e-3 from the family's LR range test (`results/rp_lr_range/rp_basicpitch_l2_all.json`, read 2026-10-15: never rises, floor at 4.8e-3).
