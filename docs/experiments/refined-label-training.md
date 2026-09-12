# Training on refined rotor-speed labels: is the DREGON gap label noise?

**Status:** done — 2026-09-12 · **Branch**: `refined-label-training` ·
**Verdict**: hypothesis refuted at one seed (the gap widened).

## Motivation

Every learned rotor-speed tracker in this project is markedly worse on DREGON
than on Michael's rig. The quality-gate checkpoint `hppnet_l2_r2_s0` is the
cleanest instance: on the frozen real split, PIT MAE **0.77** rev/s on FLY124
cruise against **2.07** on DREGON cruise, **2.27** overall
(`paper-regime-matrix.md` § "B results"). Two explanations fit that pattern
equally well from the outside:

1. **Harder audio.** DREGON is four rotors on a noisier airframe with eight
   mics; Michael's FLY124/125 are gentler flights.
2. **Noisier training labels.** DREGON's label is a tachometer or a commanded
   track with a scale-like error of 0.3–0.8 % of rate
   (`telemetry-fitness.md` § 6d). A model cannot be scored better than the
   label it was fitted to.

Nothing in the regime matrix separates them, because every row so far trained
on raw telemetry. `refined-rps-labels.md` closed the prerequisite: the
published frames datasets now carry `rps_refined` beside the untouched raw
track — the F_VK/L-BFGS label refined against each recording's own comb, then
regime-gated. That makes explanation 2 directly testable: hold the recipe
fixed and swap only the training labels.

## Setup

One arm, one seed: `hppnet_l2_r2refined_s0`
(`conf/experiment/hppnet_l2_r2refined_s0.yaml`), which differs from the gate
`hppnet_l2_r2_s0` in exactly one field — `data.train.params.path` points at
`conf/online_mix/hb_silence_refined_dload.yaml` instead of
`hb_silence_dload.yaml`. Those two policies are equal after dropping the
`rps_key: rps_refined` entries added to their two `kind: frames` arms; the
experiment configs are equal after normalising the experiment name and that
path. Model, loss, metrics, optimiser, seed 0, 200 epochs, patience 20, batch
16, the 50,000-chunk warm-up, 40,000 frames per validation and the frozen real
validation URI are all the gate's.

**Validation labels are not touched.** The comparison is against the gate's own
validation protocol, so only the training targets move.

### The mechanism that made this possible

The online-mix path could not select the refined track: `adapt_recording_frame`
took the first of `PUBLISHED_RPS_KEYS` present and ignored `rps_refined`. It
now takes an `rps_key` option, threaded through
`mixing.load_noise_source_frames` and exposed as a noise-pool spec key, so a
`kind: frames` source names its label track. `None` keeps the old preference,
so every existing config trains on exactly the same labels as before. An
explicit key that a *labelled* recording does not carry **raises** rather than
falling back to telemetry — a pool that silently mixed two label definitions
would not be comparable to the gate, which is the entire point. Every adapted
frame records `rps_key` and `label_variant: refined|published` in its meta, so
a run's provenance names its labels. The older sidecar route
(`noise_rps_dataset.apply_rps_override` / `rps_override_dir`) is untouched.

### What this experiment can move, and what it cannot

Refinement is regime-gated (`src/data_processing/rps_gating.py`): below
45 rev/s `rps_refined` **is** the telemetry, bit-identical; settled cruise
(≥ 65 rev/s) takes the refinement exactly; the ramp blends the correction,
`r = r_tel + w·(r_ref − r_tel)`, so both end points are exact by construction.
Consequences for reading the result:

- **Only cruise labels move.** The zero-frame, below-30 and standby parts of
  the training label are unchanged, and the silence arm's labels are exactly
  zero in both arms. The DREGON-cruise and FLY124-cruise cells are where the
  hypothesis lives.
- **The correction is small, and measured.** Loading this exact training pool
  both ways and comparing the two tracks at the refined track's own stamps
  (the telemetry interpolated there) gives, over all 6 recordings and 4
  rotors:

  | regime (slowest rotor) | max abs delta | mean | samples moved | n |
  |---|---:|---:|---:|---:|
  | standby (< 45 rev/s) | 0.226 | 0.0021 | 26.3 % | 4,724 |
  | ramp | 1.264 | 0.0179 | 51.5 % | 408 |
  | settled cruise (>= 65 rev/s) | **4.258** | **0.241** | 75.2 % | 44,640 |

  Standby is flat to 0.23 rev/s — the gate invariant, the residual being
  telemetry interpolation at those stamps, not refinement. Per recording the
  cruise maximum is 4.26 / 3.80 / 3.27 / 3.04 / 2.88 rev/s on the five DREGON
  room2 flights and 1.09 on FLY125; FLY125 agrees with the 1.52 rev/s / mean
  0.16 previously measured on one of its cruise windows. So the test is
  whether a mean 0.24 rev/s (worst 4.3) cruise-label error costs the
  ~1.3 rev/s DREGON-minus-Michael's cruise gap. A null result is informative.
- **One secondary difference, bounded and off-regime.** `rps_refined` is
  published on the 0.032 s analysis grid (31.25 Hz — exactly the model's STFT
  frame grid), while DREGON's `motors_command` is ~929 Hz. The mixer
  interpolates whichever track it is given onto each chunk's STFT grid, whose
  phase is a random cut, so on RAMP frames the refined arm's label is a
  coarser piecewise-linear approximation of a trajectory that moves fast.
  Measured on a uniform 31.25 Hz grid this reaches 7.5 rev/s, but only on
  ramp frames — 0.8 % of label samples — and mean 0.08 rev/s over everything
  outside cruise. It cannot produce a cruise result, and it is a property of
  the published track, not of this config.
- **FLY125 is the training rig, FLY124 the validation rig.** Refining FLY125's
  labels is a cross-rig transfer test as much as a DREGON one.

## Results

### 2026-09-12 — `hppnet_l2_r2refined_s0`: the hypothesis is refuted

Submitted from pushed `9f065dcd` on `refined-label-training`. Training:
backend `uni`, 1 GPU / 8 CPU / 32 GB / 24 GB VRAM, `--time 12h`, job
`rlt-hppnet-l2-r2refined--291b11` — it started 14 s after submission (no
queue) and finished in 54 min at epoch 39, best epoch 19,
`best_rps_mae` 2.8616. Evaluation: `uni-gpushort`, job
`rlt-eval-refined-ab4de2`, 107 s.

Pre-submission checks: `scripts/check_stream.py --experiment
hppnet_l2_r2refined_s0` → `RESULT: PASS — total 564.2 s`, exit 0, determinism
4/4 bit-identical, stage boundary at 50,000 chunks = epoch 10.00;
`tests/data_processing -k "frame or mixing"` 88 passed.

Frozen real split, all mics, PIT MAE (rev/s). **Both models were dumped in the
same job** on the same 296 clips with the same readout, so the two rows differ
only in the checkpoint. **Control:** the gate re-dumped under today's code
reads 2.2658, reproducing its published 2.265799 — there is no evaluation
drift inside this comparison.

| regime (frames) | refined | gate `hppnet_l2_r2_s0` | delta |
|---|---:|---:|---:|
| **all** (74,296) | 2.86 | **2.27** | **+0.59** |
| zero-frames (9,296) | **0.24** | 1.07 | −0.83 |
| below-30 (2,728) | 16.37 | **14.64** | +1.73 |
| DREGON ramp in-grid (4,176) | 7.41 | **4.91** | +2.50 |
| FLY124 ramp in-grid (5,888) | 6.24 | **2.72** | +3.52 |
| **DREGON cruise** (32,128) | 2.57 | **2.07** | **+0.50** |
| **FLY124 cruise** (20,080) | 0.78 | **0.77** | **+0.01** |
| ground all (8,032) | **0.20** | 0.38 | −0.18 |

`eval.py experiment=hppnet_l2_r2refined_s0` agrees with the dump (`rps_mae`
2.8629, `bce` 0.7751). Not a favourable-draw artifact: the refined arm's
`val/rps_mae` last-15 median is **4.72** (IQR 0.44) against the gate's 2.79
(IQR 0.27) — the entire curve is worse, and it early-stopped 15 epochs sooner.
Artifacts: `results/refined_label_training/` (`regimes.csv`, `regimes.md`,
`submissions.json`, the validation history).

Readings:

1. **The cross-rig cruise gap widened, which is the answer.** DREGON cruise
   2.07 → 2.57 while FLY124 cruise did not move (0.77 → 0.78), so the
   DREGON-minus-Michael's cruise gap went from 1.30 to **1.79** rev/s. The
   hypothesis predicted it would shrink. Correcting the cruise label by mean
   0.24 rev/s (max 4.26) did not touch whatever makes DREGON hard, so the
   tachometer's scale error is not the cause of the gap.
2. **The ramp collapse is a label-RESOLUTION artifact, and it was predicted.**
   DREGON ramp 4.91 → 7.41 and FLY124 ramp 2.72 → 6.24 is by far the largest
   movement, and it lands exactly where § "What this experiment can move"
   said the two arms differ for reasons other than refinement: `rps_refined`
   is published on the 0.032 s grid, DREGON's `motors_command` at ~929 Hz, so
   this arm's ramp labels are a coarse piecewise-linear approximation of a
   trajectory moving up to 80 rev/s per frame (measured: up to 7.5 rev/s of
   label difference on ramp frames). The model learned the coarser ramp and
   is scored against the fine one. This does **not** contaminate the cruise
   reading, where both tracks are smooth and the delta is the refinement.
3. **The one genuine win is silence.** Zero-frames 1.07 → **0.24** and ground
   0.38 → 0.20, both large relative improvements. Standby labels are
   bit-identical between the arms (measured max 0.226 rev/s), so this is not
   a label effect on those frames; the plausible route is that a
   lower-resolution, smoother label near motor cut-off stops teaching the
   spin-down transient into the zero class. At one seed this is a lead, not a
   result.
4. **A worse optimum, reached sooner.** Best at epoch 19 versus the gate's 33,
   stopped at 39 versus 54, plateau 4.72 versus 2.79. The refined pool is a
   harder or less consistent fit, which is itself evidence against "the raw
   labels were holding the model back".

What this does not settle: one seed, and the ramp confound means the *size* of
the overall +0.59 rev/s is not a clean measurement of refined-vs-raw label
values — only the cruise cells are. A follow-up worth doing before reviving
the hypothesis is publishing `rps_refined` at telemetry rate, which removes the
confound entirely; the tooling (`rps_key`) is now in place and costs one config
line per arm.

Since the gate's `hppnet_l2_r2_s0` remains the better checkpoint on every
regime that matters here, the quality gate is unchanged by this campaign.

## Conclusion

**Refuted at one seed. Refined training labels did not close the DREGON gap —
they widened it.** DREGON cruise PIT MAE rose 2.07 → 2.57 rev/s while Michael's
FLY124 cruise stayed at 0.77 → 0.78, taking the cross-rig cruise gap from 1.30
to 1.79 rev/s; overall PIT MAE lost 0.59 (2.27 → 2.86) and the validation
plateau nearly doubled (2.79 → 4.72). The DREGON-versus-Michael's tracking gap
is therefore **not** explained by the tachometer's scale error in the training
labels, and "train on better labels" is not the lever.

The campaign's durable output is the mechanism rather than the number: a
`kind: frames` noise source can now name its label track (`rps_key`), with the
default path byte-identical and a hard failure instead of a silent fallback, so
any future label arm is one config line. The measured label deltas on this pool
(cruise mean 0.24 / max 4.26 rev/s, standby 0.002) and the 31.25 Hz publication
grid of `rps_refined` are both recorded above; the second is the thing to fix
before anyone retries this on the ramp.
