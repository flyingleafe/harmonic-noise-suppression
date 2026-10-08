---
experiment: real_r4_hppnet_l2convlstm_unified
training_config: conf/experiment/real_r4_hppnet_l2convlstm_unified.yaml
batch: docs/experiments/pyramid-harmonic.md
---

# `real_r4_hppnet_l2convlstm_unified`

## Motivation

HPPNet-L2 with its `FreqGroupLSTM` replaced by `RateConvLSTM`
(`head: convlstm`): a bidirectional ConvLSTM over time whose gates are 1-D
convolutions along the CQT's log axis, so the state at bin `g` is written
from bins `g ± 15` of the previous frame and can follow a rotor through a
ramp — the per-bin LSTM cannot move state across bins at all. `lstm_size`
128 split 64 + 64 over the directions, then a 1×1 to the four maps and the
`FreqSuperResHead` onto the 0–150 rev/s linear grid.

The half-width is the CQT-axis reading of the pyramid rule: largest sustained
(5-frame mean) per-frame slew in the raw telemetry of the training pools, in
log bins (48/octave) at rates ≥ 27.5 rev/s (the CQT floor): `updown` 14.9,
FLY125 5.8, `rectangle` 4.4, the rest ≈ 3 → half-width 15, kernel 31. The
pyramid arm's 16 bins at 0.977 rev/s is the same reach at ~50–80 rev/s.

CQT-side counterpart of `real_r4_hppnet_pyrlstm_unified` (pyramid, variant
A); read with `real_r4_hppnet_l2_unified` and `real_r4_hppnet_l2nolstm_unified`.

## Setup

Hydra wiring — data `e12_real_fullflight` (train stream
`conf/online_mix/hb_m3s2_dload.yaml`, 2 s clips) · model `hppnet_l2_convlstm`
· loss/metrics `salience_layers_r150` · validation `rps_unified`. Batch 128,
AdamW 1e-3 / 1e-4, monitor `rps_mae` — `real_r4_hppnet_l2_unified`'s recipe
verbatim. Train with
`python train.py experiment=real_r4_hppnet_l2convlstm_unified`.

## Conclusion

Submitted 2026-10-08 to `vast` (A100). The first run (`-e09290`) ran at
2 s/it because of a per-step slicing bug in `RateConvLSTM`'s backward
(batch doc) and was cancelled at epoch 3; resubmitted on the fix, same
model. Results go to the batch doc.
