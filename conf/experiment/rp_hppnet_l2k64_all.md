---
experiment: rp_hppnet_l2k64_all
training_config: conf/experiment/rp_hppnet_l2k64_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_hppnet_l2k64_all`

## Motivation

The row "harmonics k ≤ 64" of the paper's chain table "From HPPNet-L2 to HRSPNet" (one change
per row): HPPNet-L2 (`rp_hppnet_l2_all`) with the harmonic range of HRSPNet, the CQT extended to
7.04 kHz (385 bins) and `harmonic_k_max: 64` (`conf/model/hppnet_l2k64.yaml`). Trained on the
`rp_all` stream; compared against `rp_hppnet_l2_all` and `rp_pyrseplstm_k64_all`. Settings as
the rest of the real-data grid: up to 100k steps, 5k warm-up + cosine LR, max LR 1e-3, batch 128
(64 × 2 accumulation, so any `sae` GPU fits),
`rp_real` panel every 500 steps, `best_<score>_{median,raw}` checkpoints.

## Conclusion

Queued 2026-10-27 on `uni`, any `sae` GPU; results pending.
