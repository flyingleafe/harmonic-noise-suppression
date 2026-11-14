---
experiment: rp_hppnet_l2k64_convlstm_priorv4
training_config: conf/experiment/rp_hppnet_l2k64_convlstm_priorv4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_hppnet_l2k64_convlstm_priorv4`

## Motivation

The "+ ConvLSTM" row of the paper's chain table (`rp_hppnet_l2k64_convlstm_all`: CQT to
7.04 kHz, $k \le 64$, the separable RateConvLSTM of HRSPNet) in the hard synthetic setup, beside
`rp_hppnet_l2k64_priorv4`. On the held-out test split this CQT model beats HRSPNet when both are
trained on real data alone (4.12 / 4.04 Hz against 5.02 / 4.56 without / with label
augmentation); the pair asks whether HRSPNet's STFT pyramid pays off once the models are
pretrained on the hard setup and fine-tuned (Dmitrii, 2026-11-14).

Stream `conf/online_mix/rp_priorv4.yaml`, validation `rp_real_priorv4`, stopping rule, batch
128 as 64 × 2 accumulation, max LR 1e-3, 12 data workers and one H100, as
`rp_hppnet_l2k64_priorv4`.

## Conclusion

Pending.
