---
experiment: rp_hppnet_l2k64_priorv4
training_config: conf/experiment/rp_hppnet_l2k64_priorv4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_hppnet_l2k64_priorv4`

## Motivation

On the held-out test split, real-only training reverses the last row of the paper's chain table:
the CQT models with $k \le 64$ (`rp_hppnet_l2k64_*`, 5.02 / 4.13 Hz without / with label
augmentation) and with the ConvLSTM (4.12 / 4.04) beat HRSPNet (5.02 / 4.56), while HRSPNet
fine-tuned from the hard setup reaches 2.34 ± 0.01 over three seeds (Dmitrii, 2026-11-14). This
run and `rp_hppnet_l2k64_convlstm_priorv4` give the two CQT rows the same hard pretraining, so
the test split can say whether the STFT pyramid pays off under synthetic pretraining.

`rp_hppnet_l2k64_all` (CQT to 7.04 kHz, harmonic orders $k \le 64$, FreqGroupLSTM) with every
real noise recording replaced by the hard setup's stream `conf/online_mix/rp_priorv4.yaml`,
unchanged since `rp_pyrseplstm_k64_priorv4` (bank pins and render path identical). Validation
`rp_real_priorv4`; the run stops when neither `real_macro` nor `uniform` has improved its 5-round
median. Batch 128 as 64 × 2 accumulation, max LR 1e-3, 12 data workers, one H100.

## Conclusion

Pending.
