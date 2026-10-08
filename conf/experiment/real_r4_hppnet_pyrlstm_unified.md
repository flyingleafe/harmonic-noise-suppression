---
experiment: real_r4_hppnet_pyrlstm_unified
training_config: conf/experiment/real_r4_hppnet_pyrlstm_unified.yaml
batch: docs/experiments/pyramid-harmonic.md
---

# `real_r4_hppnet_pyrlstm_unified`

## Motivation

`hppnet_pyramid` with its 1×1 head replaced by `RateConvLSTM` — variant A of
`docs/pyramid-harmonic-frontend-design.md` § 7. HPPNet's `FreqGroupLSTM` runs
one recurrence per rate bin and cannot move state across bins, so it denoises
cruise but cannot follow a ramp. The ConvLSTM's gates are 1-D convolutions
along the rate axis whose half-width is the largest speed change a rotor makes
in one 32 ms frame, so the state at bin `g` is written from bins `g ± 16` of
the previous frame and can follow any physical ramp.

The half-width comes from the raw telemetry of the two training pools
(DREGON in-flight, FLY125), resampled to the 31.25 Hz loss grid, motor-on
frames only. Sustained slew (5-frame mean) maxima: `updown` 14.6 rev/s per
frame, FLY125 7.3, `rectangle` 4.9, the rest ≈ 3. Single-frame diffs above
~10 (max 53 on `updown`) are 900 Hz ESC-telemetry jitter. `max_slew = 15` →
`ceil(15 / 0.9766) = 16` bins → kernel 33. `lstm_size` 128 as HPPNet, split
64 + 64 over the two directions; 1×1 to the four maps after it.

Read against `real_r4_hppnet_pyr_unified` (same model, no recurrence) and
`real_r4_hppnet_l2_unified` (CQT front end, `FreqGroupLSTM`). Question: what
does a rate-following recurrence buy over temporal convolutions + CRF — on
cruise (where the per-bin LSTM already helped) and on ramps (where it could
not)?

## Setup

Hydra wiring — data `e12_real_fullflight` (train stream
`conf/online_mix/hb_m3s2_dload.yaml`, 2 s clips) · model
`hppnet_pyramid_convlstm` · loss `salience_layers_pyr` · metrics
`salience_layers_pyr` · validation `rps_unified`. Batch 64 × 2 accumulation
steps (effective 128), AdamW 1e-3 / 1e-4, monitor `rps_mae`. Train with
`python train.py experiment=real_r4_hppnet_pyrlstm_unified`.

## Conclusion

Submitted 2026-10-08 to `vast` (A100, `hppnet-pyrlstm-c5c30d`) after the
tap-conv and `RateConvLSTM` efficiency rounds recorded in the batch doc
(both exact; the model is unchanged). Results go to the batch doc.
