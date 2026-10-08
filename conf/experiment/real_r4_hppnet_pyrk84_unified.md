---
experiment: real_r4_hppnet_pyrk84_unified
training_config: conf/experiment/real_r4_hppnet_pyrk84_unified.yaml
batch: docs/experiments/pyramid-harmonic.md
---

# `real_r4_hppnet_pyrk84_unified`

## Motivation

`hppnet_pyramid` with the harmonic taps extended from k ≤ 32 to **k ≤ 84**
and the output grid cut from 0–299 to **0–150 rev/s** (154 bins at the
0.977 rev/s level-0 step); no recurrence (variant C).

Two findings from the first pyramid arms drive it. (1) At the data's
60–90 rev/s, k_max 32 reads up to only 1.9–2.9 kHz, so pyramid levels 2 and 3
(2.4–8 kHz) were never read by the taps — the model was effectively two
levels and never exercised the resolution the pyramid exists for (7.8 Hz bins
at 6 kHz: 0.13 rev/s per harmonic at k = 60, against the CQT's 1.1). DREGON's
rotor comb is strong around 6 kHz (harmonics ~40–80); 91 taps now read it. (2)
Half the 307-bin grid never carried a target; 154 bins is the L2 rows' range,
so the training losses are on the same footing across families, and the step
is ~15 % cheaper than the first arm despite 2.3× the taps.

Read against `real_r4_hppnet_l2k84nolstm_unified` — the CQT extended to
7.04 kHz with `HarmonicDilatedConv` branches to k = 84, same k set, same
`k_sub`, same head — the pair that isolates the front end with the high
harmonics in play; and against `real_r4_hppnet_pyr_unified` (k ≤ 32, 307 bins,
best `real_overall` 2.69) for what the high harmonics are worth on the pyramid.

## Setup

Hydra wiring — data `e12_real_fullflight` (train stream
`conf/online_mix/hb_m3s2_dload.yaml`, 2 s clips) · model `hppnet_pyramid_k84`
· loss/metrics `salience_layers_pyr154` · validation `rps_unified`. Batch
64 × 2 accumulation steps (effective 128), AdamW 1e-3 / 1e-4, monitor
`rps_mae`. Train with `python train.py experiment=real_r4_hppnet_pyrk84_unified`.

## Conclusion

Submitted 2026-10-08 to `vast` (A100). Results go to the batch doc.
