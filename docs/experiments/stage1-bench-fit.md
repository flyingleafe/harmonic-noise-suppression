---
stage: S1
objective: distributional fit of the stochastic rotor-noise model
data: DREGON-frames `split="motor"` cells, channel 7, native 44.1 kHz decimated to 16 kHz
---

# S1 — the single-rotor, windless, single-microphone fit

## What S1 is

The simplest case in the project: one rotor at a fixed setpoint, clamped to a
bench, nothing moving, no telemetry, no rotor mixture, no wind. 20 recordings
(`Motor{1-4}_{50,60,70,80,90}`). `Motor1`-`Motor3` are the fit set, **`Motor4`
is held out**. Channel 7 is used throughout: it carries the largest
band-integrated comb margin over orders 2-30 (18.54 dB against 16.59-18.11 dB
for the other seven), measured with `bench.read_orders` over six recordings.

Audio is always the native 44.1 kHz recording decimated by `clips.decimate`;
since the 2026-09-12 cutover it comes from the published `DREGON-frames`
dataset (`split="motor"`, recording ids `motor_Motor1_80`), not from the raw
tree. The published 16 kHz *training* datasets brick-wall at 7.9 kHz by
88-90 dB and would be fitted as if that were the rig.

## Instruments built

| file | role |
|---|---|
| `src/experiments/stochastic_fit/clips.py` | the only loader: published frames at native rate, `Recording.cut`, `decimate` (earlier verified against the published 16 kHz clips at corr +0.993 to +0.9997, lag 0) |
| `src/experiments/stochastic_fit/stage1_bayes.py` | the bench cells (`bench_cells`, `bench_span`, the comb-evidence rate seed), `BENCH_VARIANT`, and the bench→renderer export |
| `src/experiments/stochastic_fit/campaign.py` | the fit itself: `fit(regime="bench")`, driven by `scripts/stochastic_fit.py --regime bench` |
| `src/experiments/stochastic_fit/accept_stats.py` | the two acceptance statistics, shared by the gate and the fit so they cannot diverge |
| ~~`scripts/_stage_accept.py`~~ | the preregistered gate, **retired 2026-09-12** with `stage1.py`: it scored the derived-summary S1 fit, which the Whittle fit replaced. The statistics themselves survive in `accept_stats.py` (used by `scripts/_stage2_panels.py`), so restating the gate is a matter of choosing the criterion — see the open decision below |

## Measured quantities (fit motors)

* **Rates.** The setpoint is not rev/s: 50 -> 48.4-49.8, 90 -> 86.9-89.4 rev/s.
  Reading the rate natively at 44.1 kHz with `estimate_rate` returned **half**
  the true rate on Motor1-3 (39.1 instead of 78.2 rev/s) while Motor4 came out
  right — an octave error that disappears once the band is limited to 8 kHz.
* **Width law.** `gamma = 0.024 + 0.0385 k` Hz, 242 lines. Linear in k, which is
  the shaft-jitter signature: `sigma = slope / 1.177 = 0.033 rev/s` of ESC
  speed wander on a clamped motor.
* **Speed laws** (shared exponent, free intercept per motor, because the motors
  differ by up to 7 dB at the same setpoint): line level exponent **6.02**,
  floor exponent **4.98**.
* **Profile.** Measured support 84 orders, roll-off **-2.42 dB per log-k**
  (-1.68 dB/octave) fitted over orders 4-84 and extrapolated beyond, clipped by
  the per-cell detection limits and forced monotone.

## Five defects found and fixed, each by measurement

1. **`line_mode` cannot be chosen by default.** With shaft jitter zero (the
   physically correct value for a clamped motor) `line_mode="fm"` renders
   **0.00 Hz** lines at every order, against a real 0.20-7.92 Hz, with peak
   margins 27-38 dB against a real 2-20 dB. `line_mode="stochastic"` cannot go
   below one STFT bin and renders 2.6-6.9 Hz lines where the bench has 0.2-1 Hz.
   The fix is FM mode with the measured width carried by shaft jitter, which
   reproduces the width law to 0.00/0.20/0.55/0.98/2.13 Hz against a real
   0.00/0.20/0.22/0.56/7.92 Hz at k = 2/8/16/32/64.
2. **The half-integer null must keep the true floor window.** Reading the null
   by scaling the rate 1.5x puts every even order back on a real line (k=2 ->
   3x rate); that "null" read **+29 dB**. Reading `rate/2` and keeping odd
   orders, with the floor window at the true spacing, brings it to +1 to +8 dB.
3. **A band integral removes the floor's MEAN, not its median.** An averaged
   periodogram bin is Gamma-distributed; subtracting a median floor leaves
   `N (mean - median)` of bias. Correction applied for the actual averaging
   count (6 segments here, +0.25 dB — small, but the derivation matters at
   larger band widths).
4. **Selection bias in the profile, twice.** Dropping undetected orders biases
   the pooled profile upward at high order. Fitting the roll-off over the upper
   half of the support returned **+3.3 dB per log-k** — a comb RISING with
   order — which jumped the extrapolated profile from -43 dB at order 160 to
   -21 dB at order 200. Fitting over orders 4-84 gives -2.42 dB per log-k.
5. **Render and reference must share the resampling path.** Generating the
   synthetic clip directly at 16 kHz leaves it without the `resample_poly`
   transition band the real clip has, which alone put **+7.8 dB** of spurious
   deviation into the 7000-7900 Hz acceptance band. Synthetic clips are now
   rendered at 44.1 kHz and decimated identically.

## The renderer's level convention needs calibrating, and it converges

`profile_db`/`harm_mean_db` are renderer parameters, not estimator readings:
feeding the measured profile straight in renders a comb +5.3 dB too strong over
orders 2-15 and +9.9 dB too strong over 30-100, with 100 detectable orders
against a real 42-54. One affine function of `log k`, fitted on the FIT MOTORS
ONLY, removes it: median delta **+6.26 -> +0.09 -> +0.55 dB** over three rounds.

## The blocking result: the preregistered thresholds are unattainable by real data

Criterion 1 asks each clip to sit within `mean |dev| <= 1.5 dB` and
`max |dev| <= 3.0 dB` of the speed-matched median. Applying that test to **real
clips against other real clips** (each cell against the speed-matched median of
the other motors):

| | median mean\|dev\| | median max\|dev\| |
|---|---:|---:|
| all 20 real cells | **2.09 dB** | **5.33 dB** |
| held-out Motor4 real cells | 1.58 dB | 3.78 dB |
| threshold | 1.5 | 3.0 |

Criterion 2 (tolerance 1.5 dB), same real-vs-real test:

| order band | real-vs-real median \|delta\| |
|---|---:|
| k 2-15 | **2.35 dB** |
| k 15-30 | 1.02 dB |
| k 30-100 | 1.17 dB |

The cause is measured: the bench floor **shape** varies by MOTOR, not by speed.
A per-knot regression of the floor shape on `10 log10(rate)` leaves residual
rms 2.95 dB against a raw spread of 2.97 dB — the speed explains nothing, while
Motor2 sits 5-13 dB below Motor1/Motor3 at 1 kHz. Each motor occupies a
different position and orientation relative to channel 7, so its floor carries
a different acoustic transfer. That is a property of the measurement geometry,
not of the rotor-noise family, and no population model can predict it for an
unseen motor.

## Where the fit actually stands against those baselines

| statistic | synthetic | real-vs-real baseline | threshold |
|---|---:|---:|---:|
| criterion 1 median mean\|dev\| | 2.39 dB | 2.09 dB (Motor4: 1.58) | 1.5 |
| criterion 1 median max\|dev\| | 6.02 dB | 5.33 dB (Motor4: 3.78) | 3.0 |
| criterion 2 k 2-15 | -2.25 dB | 2.35 dB | 1.5 |
| criterion 2 k 15-30 | -0.27 dB | 1.02 dB | 1.5 |
| criterion 2 k 30-100 | +2.54 dB | 1.17 dB | 1.5 |
| detected orders | 71 | 60 (Motor4) | — |

The synthetic clips are within 0.3-0.7 dB of the real-vs-real baseline on
criterion 1 and inside it on the k 2-15 and k 15-30 order bands; k 30-100 is
about 1.4 dB worse than the baseline.
