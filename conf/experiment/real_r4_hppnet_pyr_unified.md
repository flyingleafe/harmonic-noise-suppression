---
experiment: real_r4_hppnet_pyr_unified
training_config: conf/experiment/real_r4_hppnet_pyr_unified.yaml
batch: docs/experiments/pyramid-harmonic.md
---

# `real_r4_hppnet_pyr_unified`

## Motivation

First run of `hppnet_pyramid` (`src/models/harmonic_ports/hppnet_pyramid.py`,
design in `docs/pyramid-harmonic-frontend-design.md`): HPPNet's block order on
a nested four-level STFT pyramid (1 s window to 1200 Hz, halving per level),
FPN fusion, harmonic taps `k·r` (k ≤ 32) and `r/k` (k ≤ 8) gathered from the
level that suits each frequency, octave/quarter-octave taps on the rate axis,
temporal convolutions and **no LSTM** (variant C of the design note's § 7 —
the null hypothesis for what `FreqGroupLSTM` was worth). Output grid inherits
level 0's step: 0.9766 rev/s over 0–298.8 (307 bins).

Read against `real_r4_hppnet_l2_unified` (HPPNet-L2, CQT front end, LSTM):
same regime, same panel, same effective batch. Questions: does the pyramid
front end match the CQT's BPF resolution on the real split (cruise), and does
dropping the LSTM cost anything beyond cruise denoising (ramps, below-30)?

## Setup

Hydra wiring — data `e12_real_fullflight` (train stream
`conf/online_mix/hb_m3s2_dload.yaml`, 2 s clips) · model `hppnet_pyramid` ·
loss `salience_layers_pyr` · metrics `salience_layers_pyr` · validation
`rps_unified`. Batch 64 × 2 accumulation steps (effective 128 frames, as the
L2 row), AdamW 1e-3 / 1e-4, monitor `rps_mae`. Train with
`python train.py experiment=real_r4_hppnet_pyr_unified`.

## Conclusion

Submitted 2026-10-07 to `uni-gpushort` as a 1 h smoke of the training
dynamics; to be resumed elsewhere if it trains. Results go to the batch doc.
