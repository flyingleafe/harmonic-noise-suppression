---
experiment: rp_harmof0_l1_all
training_config: conf/experiment/rp_harmof0_l1_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_harmof0_l1_all`

## Motivation

Claim 2 (HPPNet best up to L2; L0→L1 vs L1→L2 attribution): HarmoF0 at ladder level L1 (log_off layers, `salience_layers_log48`), trained on the `rp_all` stream. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR to be set from the family's LR range test.
