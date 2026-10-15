---
experiment: rp_basicpitch_l0_all
training_config: conf/experiment/rp_basicpitch_l0_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_basicpitch_l0_all`

## Motivation

Claim 1 (unadapted salience models fail): Basic Pitch (hop 256) at ladder level L0, trained with the published grid's shared-map BCE (`salience_bce_basic_pitch`) on the `rp_all` stream. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR 1e-3 — the LR every architecture we build on was published with (docs/experiments/paper-real-protocol.md § "Budget and LR"); the family's LR range test is reported in the appendix.
