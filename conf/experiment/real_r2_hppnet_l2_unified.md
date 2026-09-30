---
experiment: real_r2_hppnet_l2_unified
training_config: conf/experiment/real_r2_hppnet_l2_unified.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `real_r2_hppnet_l2_unified`

## Motivation

HPPNet port at L2 — per-rotor Gaussian layers on the linear 0–150 rev/s / 300-bin grid, CRF readout, trained on real rung R2 under the unified regime (2-second clips, batch 128, full-panel validation every 500 updates via the on-device decoder, subset-best checkpoints). One of the eight salience matrix cells (L2 on R1–R4 for both ports).

Question (2026-09-30): how precise HPPNet L2 can get on DREGON when trained on DREGON alone. DREGON is the harder rig for every model (e.g. `real_r4_gru_unified` DREGON 3.73 vs FLY124 1.68 rev/s, `results/regime_decomp/real_r4_gru_unified.json`), and the HPPNet L2 trunk is the most precise model family so far. The read is `best_real_r2.ckpt` (the DREGON 8-mic view, clips 0–175), compared with `real_r2_scv2_unified` `best_real_r2` (`real_r2` 2.11 rev/s, round 19, its `best_checkpoints.json`) and the two-rig `hppnet_l2_r2_s0` DREGON 2.66 (Review-B dump, `results/paper_review_B/hppnet_l2_r2_s0/real/`).

## Conclusion

Defined 2026-09-08. Submitted 2026-09-30 from `origin/main` `815a3f97` (config passed `validate_only=true` there): omnirun job `real-r2-hppnet-l2-unifie-12ff0b`, Vast 1× A100, 16 CPUs, 64 GB, 16 h limit. Results pending.
