---
experiment: hppnet_l2_r2refined_s0
training_config: conf/experiment/hppnet_l2_r2refined_s0.yaml
batch: docs/experiments/refined-label-training.md
---

# `hppnet_l2_r2refined_s0`

## Motivation

The quality-gate checkpoint `hppnet_l2_r2_s0` tracks Michael's rig four times
more accurately than DREGON: on the frozen real split its PIT MAE is **0.77**
rev/s on FLY124 cruise against **2.07** on DREGON cruise, and **2.27** overall
(`docs/experiments/paper-regime-matrix.md` § "B results"). The usual reading is
that DREGON is harder audio — eight mics on a noisier airframe, four rotors
that are never quite settled.

This run tests the competing explanation: the gap is **training-label noise**.
DREGON's rotor-speed label is a tachometer (`motors_measured`) or a commanded
track (`motors_command`) carrying a scale-like error of 0.3–0.8 % of rate
(`telemetry-fitness.md` § 6d). A model fitted to a label that is wrong by a
fraction of a percent cannot be scored better than that label. The published
frames datasets now also carry `rps_refined`, the F_VK/L-BFGS label refined
against each recording's own comb and regime-gated
(`docs/experiments/refined-rps-labels.md`). If the gap is label noise, training
on the refined track should close part of it; if it is the audio, nothing moves.

## Setup

Identical to `hppnet_l2_r2_s0` in every field except the training labels.

Changed — exactly one thing:

- `data.train.params.path`: `conf/online_mix/hb_silence_dload.yaml` →
  `conf/online_mix/hb_silence_refined_dload.yaml`, which is the same policy
  with `rps_key: rps_refined` added to its two `kind: frames` arms. The two
  policies are provably equal after dropping that key.

Unchanged: model `hppnet_l2`, loss/metrics `salience_layers_r150`, data
`e12_real_fullflight`, `seed: 0`, `checkpoint: null`, `epochs: 200`,
`patience: 20`, `batch_size: 16`, `num_workers: 6`,
`samples_per_validation: 40000`, `optim.monitor: rps_mae` / `min`. The noise
pool is the same historical-R2 honest base: DREGON-frames `in_flight_noise`
minus `free-flight_nosource_room1` and michaels-frames FLY125, both at
`min_motor_rps: 0.0` (whole envelope) and both unweighted so they merge
duration-weighted, plus the zero-labelled silence arm at weight 0.4 (16.7 % of
noise chunks) and `snr_ref_floor_rms: 0.02`. The 50,000-generated-chunk
unaugmented warm-up, the freq-scale noise augmentation, the 50 % post-mix
augmentation and the 50 % time warp are all as before.

**Validation is untouched** — the same frozen real split
(`dload:DREGON-LM-V4-michaels-valid-full@9604f3ff…`) with its original
published targets. The comparison is only meaningful against the gate's own
validation protocol, so only the training labels move.

**What this experiment can and cannot move.** Refinement is regime-gated
(`src/data_processing/rps_gating.py`): below 45 rev/s `rps_refined` *is* the
telemetry, bit-identical; settled cruise (≥ 65 rev/s) takes the refinement
exactly; the ramp blends the correction. So the zero-frame, below-30 and
standby parts of the training label are unchanged by construction, and the
silence arm's labels are exactly zero either way. **Only cruise labels move.**
A cruise-regime improvement is the effect under test; a change in the zero or
below-30 regimes could only be an indirect consequence or run-to-run noise.

## Results

Frozen real split, all mics, PIT MAE (rev/s). Both models were dumped in the
SAME job (`scripts/rps_dump.py` on `conf/data/e12_real_fullflight.yaml`, grid
`0,150,300`, 4 layers, then `scripts/rps_regime_table.py`), so the two rows
share one readout, one set of 296 clips and one revision. **Control:** the
gate re-dumped today reads **2.2658**, reproducing its published 2.265799 —
no evaluation drift is in this comparison.

| regime (frames) | refined | gate `hppnet_l2_r2_s0` | delta |
|---|---:|---:|---:|
| all (74,296) | 2.86 | **2.27** | +0.59 |
| zero-frames (9,296) | **0.24** | 1.07 | −0.83 |
| below-30 (2,728) | 16.37 | **14.64** | +1.73 |
| DREGON ramp in-grid (4,176) | 7.41 | **4.91** | +2.50 |
| FLY124 ramp in-grid (5,888) | 6.24 | **2.72** | +3.52 |
| DREGON cruise (32,128) | 2.57 | **2.07** | +0.50 |
| FLY124 cruise (20,080) | 0.78 | **0.77** | +0.01 |
| ground all (8,032) | **0.20** | 0.38 | −0.18 |

Run: best epoch 19, stopped 39 (gate: 33 / 54); `val/rps_mae` last-15 median
**4.72** (IQR 0.44) against the gate's 2.79 (IQR 0.27), so the whole curve is
worse, not just the selected draw. `eval.py` agrees with the dump
(`rps_mae` 2.8629, `bce` 0.7751). Artifacts:
`results/refined_label_training/{regimes.csv,regimes.md,submissions.json}`.

## Conclusion

**The hypothesis is refuted at one seed.** Training on refined labels did not
close the DREGON gap — it widened it. DREGON cruise went 2.07 → 2.57 while
FLY124 cruise stood still (0.77 → 0.78), so the cross-rig cruise gap grew from
1.30 to **1.79** rev/s, and the overall number lost 0.59. Whatever makes
DREGON harder survives a cruise-label correction of mean 0.24 rev/s, so it is
not the tachometer's scale error.

The ramp collapse (DREGON 4.91 → 7.41, FLY124 2.72 → 6.24) is the documented
side effect, not the labels being "more wrong": `rps_refined` is published on
the 0.032 s analysis grid while DREGON's `motors_command` is ~929 Hz, so ramp
labels in this arm are a coarse piecewise-linear approximation of a fast
trajectory (up to 7.5 rev/s of label difference on ramp frames). The one real
win is zero-frames, 1.07 → **0.24**.

Anyone re-running this should publish `rps_refined` at telemetry rate first —
the label-resolution change is confounded with the label-value change on the
ramp, though not on cruise, where the verdict stands. Campaign:
`docs/experiments/refined-label-training.md`.
