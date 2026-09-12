# Training on refined rotor-speed labels: is the DREGON gap label noise?

**Status:** in progress — 2026-09-12 · **Branch**: `refined-label-training`

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

### 2026-09-12 — `hppnet_l2_r2refined_s0` submitted

Pending. Verified before submission: `scripts/check_stream.py --experiment
hppnet_l2_r2refined_s0` exits 0, and `tests/data_processing` passes for the
frame/mixing paths. Numbers land here as a per-regime table against
`hppnet_l2_r2_s0` on the identical `rps_dump.py` + `rps_regime_table.py`
protocol.

## Conclusion

Pending the run.
