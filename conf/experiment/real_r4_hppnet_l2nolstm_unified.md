---
experiment: real_r4_hppnet_l2nolstm_unified
training_config: conf/experiment/real_r4_hppnet_l2nolstm_unified.yaml
batch: docs/experiments/pyramid-harmonic.md
---

# `real_r4_hppnet_l2nolstm_unified`

## Motivation

HPPNet-L2 with its `FreqGroupLSTM` removed (`head: conv1x1`): the published
CQT front end, `HarmonicDilatedConv`, `CNNTrunk` and a 1×1 to the four maps,
then the `FreqSuperResHead` onto the 0–150 rev/s linear grid. Temporal
modelling is `block_6–8` and the CRF readout only.

CQT-side counterpart of `real_r4_hppnet_pyr_unified` (pyramid front end,
variant C, also no recurrence). The 2×2 with `real_r4_hppnet_l2_unified`
(CQT + `FreqGroupLSTM`) and `real_r4_hppnet_l2convlstm_unified` separates the
front end's contribution from the recurrence's — the per-bin LSTM is a cruise
denoiser that cannot follow a ramp, and this arm measures what it was worth
on the published front end.

## Setup

Hydra wiring — data `e12_real_fullflight` (train stream
`conf/online_mix/hb_m3s2_dload.yaml`, 2 s clips) · model `hppnet_l2_nolstm` ·
loss/metrics `salience_layers_r150` · validation `rps_unified`. Batch 128,
AdamW 1e-3 / 1e-4, monitor `rps_mae` — `real_r4_hppnet_l2_unified`'s recipe
verbatim. Train with `python train.py experiment=real_r4_hppnet_l2nolstm_unified`.

## Conclusion

Submitted 2026-10-08 to `vast` (A100). Results go to the batch doc.
