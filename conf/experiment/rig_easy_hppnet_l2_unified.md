---
experiment: rig_easy_hppnet_l2_unified
training_config: conf/experiment/rig_easy_hppnet_l2_unified.yaml
batch: docs/experiments/rig-sampler-transfer-pair.md
---

> **The bank this run trains on has TWO KNOWN DEFECTS and a pending user
> decision (2026-09-15).** Do not re-run this experiment, and do not rebuild
> `data/rig_banks/`, before reading
> `docs/experiments/rig-sampler-transfer-pair.md` § "Open decision: rebuild
> the banks": the fitted comb's last order is unconstrained and loud, and the
> two rigs' anchors disagree on label provenance (Michael raw, DREGON
> refined).

# `rig_easy_hppnet_l2_unified`

## Motivation

The salience arm of the easy/hard transfer test. HPPNet port at L2 — per-rotor
Gaussian layers on the linear 0–150 rev/s / 300-bin grid, CRF readout — trained
on the EASY rig cloud (`conf/online_mix/rig_easy_5050.yaml`: a 2048-entry
preset bank drawn from close neighbourhoods of the Michael and DREGON fits at
strength 2.0, chosen so the sampled LTAS cloud covers the real rigs), under the
unified regime.

The question is whether the transfer result for the direct regressor carries
over to a salience trunk with a decoded readout. Pairs with
`rig_easy_scv2_unified` (same noise, direct regressor) and with
`real_r4_hppnet_l2_unified` (same trunk, real training pool) — so the two axes,
noise family and model family, are separable.

## Setup

Everything outside model/loss/metrics is the unified regime of
`real_r4_hppnet_l2_unified`: 2-second clips, batch 128, `validation.batch_size`
64 (sized by the A100 benchmark of 2026-09-08, `scripts/salience_val_bench.py`),
no warm-up, no warm start, and the full deterministic real + synthetic panel
every 500 optimizer updates, scored through the model's own on-device decoder
(`decode_logits`) via the `salience_rps` readout in `training/validation.py`.
Log-scale any-subset LR/stopping, subset-best checkpoints, control
`overall_macro`; real split `dload:DREGON-LM-V4-michaels-valid-full`.

The noise stream is the only difference from `real_r4_hppnet_l2_unified`. The
bank is a gitignored, bit-reproducible build product, so the job rebuilds it
first; the build command, its digests and the sampler's guards are recorded in
`conf/experiment/rig_easy_scv2_unified.md`.

## Comparison rows

| row | what it isolates |
|---|---|
| `real_r4_hppnet_l2_unified` | same trunk, real training pool — the sim-to-real gap for salience |
| `rig_easy_scv2_unified` | same noise, direct regressor — the model-family axis |
| `hppnet_l2_r2_s0` | the historical L2 reference, all-split PIT MAE 2.27 rev/s (legacy scalar validation, not the panel) |

`hppnet_l2_r2_s0` is a reference point, not a like-for-like row: it used the
legacy single-split scalar validation, and every stochastic training set before
commit `7525ce8c` carried the inflated low orders of the `synthesize`
coherent-share defect.

## Conclusion

Pending.
