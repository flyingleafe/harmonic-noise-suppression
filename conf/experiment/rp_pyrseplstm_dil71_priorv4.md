---
experiment: rp_pyrseplstm_dil71_priorv4
training_config: conf/experiment/rp_pyrseplstm_dil71_priorv4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrseplstm_dil71_priorv4`

## Motivation

Side ablation of the paper's chain table under the hard-pretraining recipe (Dmitrii,
2026-11-14): HRSPNet without its harmonic gather. `conf/model/hppnet_pyrseplstm_dil71.yaml`
replaces the 71 proportional taps (`k·r`, `r/k`) by 71 fixed offsets `r + j·Δo` evenly spread
from the fundamental to the top level's ceiling (Δo ≈ 110 Hz, the tap count of HRSPNet): a plain
dilated convolution along linear frequency, which no harmonic of a moving rate stays aligned
with. Pyramid, trunk, RateConvLSTM head and the tap kernel are HRSPNet's (`taps: dilated` in
`HPPNetPyramid`). Trained like `rp_pyrseplstm_k64_priorv4`: hard stream `rp_priorv4`, validation
`rp_real_priorv4`, batch 128, max LR 1e-3, 12 data workers, one H100. Fine-tuned in three seeds
by `rp_pyrseplstm_dil71_priorv4_ft_lr5e4` and read on the held-out test split against HRSPNet's
2.34 ± 0.01 Hz.

## Conclusion

Pending.
