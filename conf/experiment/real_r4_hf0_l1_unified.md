---
experiment: real_r4_hf0_l1_unified
training_config: conf/experiment/real_r4_hf0_l1_unified.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `real_r4_hf0_l1_unified`

## Motivation

HarmoF0 port at L1 — the published trunk with one shared map through `FreqSuperResHead` onto the linear 20–130 rev/s / 720-bin grid, same readout, trained on real rung R4 under the unified regime (2-second clips, batch 128, full-panel validation every 500 updates via the on-device decoder, subset-best checkpoints). Block S adaptation-ladder rung at R4.

## Conclusion

Defined 2026-09-08; not yet submitted.
