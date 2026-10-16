---
experiment: rp_pyrsep_bn_all
training_config: conf/experiment/rp_pyrsep_bn_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_pyrsep_bn_all`

## Motivation

Claim 3, the zero-path pair (Dmitrii, 2026-10-15): the pyramid with a separable rate-axis kernel head (`head: sepconv`, the ConvLSTM's 33-tap kernel on the current frame, no recurrence), `block_4`/`block_5` dropped, and **batch norm** after every convolution block (`conf/model/hppnet_pyrsep_bn.yaml`). Its twin `rp_pyrsep_in_all` differs only in the norm. Every HPPNet-family row so far is level-blind under instance norm and loses on `real_zero` (HPPNet L1 9.5, L2 16.6 vs HarmoF0 L1 0.5); the pair asks whether absolute level (batch norm) is what the 0 rev/s decision needs. Same stream (`rp_all`), budget and panel as the baseline grid.

## Conclusion

Defined 2026-10-15; not yet run. Expected: the batch-norm twin much better on `real_zero` at equal `real_cruise`.
