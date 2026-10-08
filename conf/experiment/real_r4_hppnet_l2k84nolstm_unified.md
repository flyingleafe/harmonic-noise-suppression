---
experiment: real_r4_hppnet_l2k84nolstm_unified
training_config: conf/experiment/real_r4_hppnet_l2k84nolstm_unified.yaml
batch: docs/experiments/pyramid-harmonic.md
---

# `real_r4_hppnet_l2k84nolstm_unified`

## Motivation

HPPNet-L2 without `FreqGroupLSTM` (`head: conv1x1`), with its input and its
harmonic organ extended upward: the CQT from 352 to **385 bins** (top bin
4.37 → 7.04 kHz; same `fmin`, Q and 48 bins/octave, bins 0–351 identical)
and `HarmonicDilatedConv` from k = 2..9 to **k = 2..84** (branches at
`round(48·log2 k)`, offsets up to 307 bins). The operator is untouched — the
published eight branches are the first eight of the list — so earlier L2
results stand; only new branches and new bins are added.

The published model never saw DREGON's strong rotor comb around 6 kHz
(harmonics ~40–80): the grid ended at 4.4 kHz and the harmonic layer read
k ≤ 9. On the log axis Δf/k is 0.0145·r at every k, so the high harmonics add
evidence, not resolution — which is exactly what distinguishes this arm from
its pyramid counterpart `real_r4_hppnet_pyrk84_unified` (same k set, same
`k_sub`, same head, 7.8 Hz bins at 6 kHz). Read also against
`real_r4_hppnet_l2nolstm_unified` (k ≤ 9, 4.4 kHz, best `real_overall` 2.45)
for what the high harmonics are worth on the CQT.

## Setup

Hydra wiring — data `e12_real_fullflight` (train stream
`conf/online_mix/hb_m3s2_dload.yaml`, 2 s clips) · model `hppnet_l2k84_nolstm`
· loss/metrics `salience_layers_r150` · validation `rps_unified`. Batch 128,
AdamW 1e-3 / 1e-4, monitor `rps_mae` — the L2 rows' recipe verbatim. Train
with `python train.py experiment=real_r4_hppnet_l2k84nolstm_unified`.

## Conclusion

Submitted 2026-10-08 to `vast` (A100). Results go to the batch doc.
