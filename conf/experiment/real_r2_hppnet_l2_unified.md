---
experiment: real_r2_hppnet_l2_unified
training_config: conf/experiment/real_r2_hppnet_l2_unified.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `real_r2_hppnet_l2_unified`

## Motivation

HPPNet port at L2 — per-rotor Gaussian layers on the linear 0–150 rev/s / 300-bin grid, CRF readout, trained on real rung R2 under the unified regime (2-second clips, batch 128, full-panel validation every 500 updates via the on-device decoder, subset-best checkpoints). One of the eight salience matrix cells (L2 on R1–R4 for both ports).

## Conclusion

Defined 2026-09-08; not yet submitted.
