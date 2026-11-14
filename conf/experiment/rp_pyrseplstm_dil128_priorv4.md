---
experiment: rp_pyrseplstm_dil128_priorv4
training_config: conf/experiment/rp_pyrseplstm_dil128_priorv4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrseplstm_dil128_priorv4`

## Motivation

Side ablation of the paper's chain table under the hard-pretraining recipe (Dmitrii,
2026-11-14): HRSPNet without its harmonic gather, with more but simpler taps.
`conf/model/hppnet_pyrseplstm_dil128.yaml` replaces the 71 proportional taps (`k·r`, `r/k`) by
128 fixed offsets `r + j·Δo` evenly spread from the fundamental to the top level's ceiling
(Δo ≈ 61 Hz, about twice HRSPNet's tap count): a denser plain dilated convolution along linear
frequency. Pyramid, trunk, RateConvLSTM head and the tap kernel are HRSPNet's (`taps: dilated`
in `HPPNetPyramid`). Trained like `rp_pyrseplstm_k64_priorv4`: hard stream `rp_priorv4`,
validation `rp_real_priorv4`, batch 128, max LR 1e-3, 12 data workers, one H100. Fine-tuned in
three seeds by `rp_pyrseplstm_dil128_priorv4_ft_lr5e4` and read on the held-out test split
against HRSPNet's 2.34 ± 0.01 Hz.

## Conclusion

Pending.
