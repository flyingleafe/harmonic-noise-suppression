---
experiment: rp_hppnet_l2k64_convlstm_all
training_config: conf/experiment/rp_hppnet_l2k64_convlstm_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_hppnet_l2k64_convlstm_all`

## Motivation

The row "ConvLSTM" of the paper's chain table "From HPPNet-L2 to HRSPNet" (one change per row):
`rp_hppnet_l2k64_all` with FreqGroupLSTM replaced by the separable RateConvLSTM of HRSPNet
(`conf/model/hppnet_l2k64_convlstm.yaml`). Trained on the `rp_all` stream; compared against
`rp_hppnet_l2k64_all` and `rp_pyrseplstm_k64_all`. Settings as the rest of the real-data grid:
up to 100k steps, 5k warm-up + cosine LR, max LR 1e-3, batch 128, `rp_real` panel every 500
steps, `best_<score>_{median,raw}` checkpoints.

## Conclusion

Queued 2026-10-27 on `uni` (H100); results pending.
