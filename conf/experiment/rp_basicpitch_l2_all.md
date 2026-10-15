---
experiment: rp_basicpitch_l2_all
training_config: conf/experiment/rp_basicpitch_l2_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_basicpitch_l2_all`

## Motivation

Claim 2 (HPPNet best up to L2; L0→L1 vs L1→L2 attribution): Basic Pitch (hop 256) at ladder level L2 (rp linear grid, `salience_layers_rp_h256`), trained on the `rp_all` stream. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR 1e-3 — the LR every architecture we build on was published with (docs/experiments/paper-real-protocol.md § "Budget and LR"); the family's LR range test is reported in the appendix.
