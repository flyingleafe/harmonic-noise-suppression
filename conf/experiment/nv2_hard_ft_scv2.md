---
experiment: nv2_hard_ft_scv2
training_config: conf/experiment/nv2_hard_ft_scv2.yaml
batch: docs/experiments/noise-v2-transfer.md
---

## Motivation

Stage 2 of the hard curriculum for the direct regressor (`real_r1_scv2`
architecture, `hb_scv2_mag_nogate` head): `nv2_hard_scv2` (trained on the
noise-model-v2 HARD bank, synthetic only) fine-tuned on REAL audio.

This is the question the legacy rig-sampler pair never asked. Those arms
measured how far a synthetic distribution gets on its own and stopped at 1.8x
the real reference. A curriculum arm asks whether the fitted family carries
anything the real pool does NOT: if pre-training on it beats the real reference
trained from scratch (**2.99** (`val/real_r3`)), the family is worth carrying;
if it lands on the reference, the honest reading is that the fitted family buys
nothing once real audio is available, and that is a clean negative result
rather than an ambiguous one.

Paired with `nv2_easy_ft_scv2`: the two differ only in which synthetic
distribution the warm start came from, so the pair says whether the WIDER
pre-training distribution is the better initialisation.

## Setup

Parent: **`real_r4_scv2_unified`**, the family's real reference arm — stream
`conf/online_mix/hb_m3s2_dload.yaml` (the R4 real pool: DREGON + FLY125 at all
8 mics, whole envelope, silence arm, speech at -30..0 dB, freq-scale +
time-warp + gain/polarity, no warm-up stage).

**Exact diff against it:** the warm start, `patience: 20` and `lr: 1e-3`. The
parent is already the unified regime (`override /validation: rps_unified`,
batch 128, 2 s clips, `samples_per_validation: null`), so nothing else moves.

Warm start: `checkpoint: best:real_overall@nv2_hard_scv2`, which
`training.loop` resolves to
`r2://<bucket>/<prefix>/nv2_hard_scv2/checkpoints/best_real_overall.ckpt` and
loads `strict=False` with a FRESH optimizer / scheduler / early-stopping state.
The learning rate is full-size (1e-3) rather than a fine-tuning trickle because
the trunk is transferring across a NOISE FAMILY, not refining within one.

**Ordering constraint:** stage 1 (`nv2_hard_scv2`) must have finished and
uploaded its `best_real_overall` checkpoint before this arm is submitted, or
the run fails at checkpoint resolution.

The bank and the hard stream that produced the warm start are documented in
`conf/experiment/nv2_hard_scv2.md`; this arm itself trains on real audio and
builds no bank.

## What will be measured

Selection and reporting on **`real_overall`** (not `overall_macro`: on the
point-preset run the synthetic half improved monotonically while every real
view degraded, so the macro neither stopped the run nor cut the LR). Reported:
best `real_overall` with the epoch it occurred at, the per-view `real_r1` /
`real_r2` / `real_r3` rev/s MAE at that epoch, the `r1`/`r2` ratio against the
point-preset run's 1.93, and the four-regime decomposition of the best
checkpoint — `python scripts/_regime_decomp.py --exp nv2_hard_ft_scv2 --ckpt
best` (ramp / zero / standby / cruise). The RAMP cell is the one this batch
turns on: both legacy synthetic arms were close to real at cruise and collapsed
the four rotors onto nearly one speed in the transitions (output spread 0.28
easy / 0.19 hard against 4.71 for the real-trained model).

The number that decides this arm is the delta against the family's real
reference at **2.99** (`val/real_r3`) — and, in the regime decomposition,
whether the RAMP cell of the warm-started model beats the from-scratch real
model's.

## Train

```bash
scripts/noise_v2_submit_arms.sh nv2_hard_ft_scv2
```

## Conclusion

**PENDING** — not submitted as of 2026-09-22.
