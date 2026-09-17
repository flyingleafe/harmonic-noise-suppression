# Noise model v2: the pre-implementation phase

**Status:** running (R2 opening) — 2026-09-15 → . Campaign `noise-model-v2`,
branch `main`. R1 is scored and FAILS: both HPPNet parity gates, the DREGON
stretch gate and the two proxy gates are false
[`results/noise_v2/rounds/round1.json`,
`results/noise_v2/rounds/round1/score/findings.md`]. Explainer:
`docs/explainers/noise-model-v2-plan.qmd` (the full proposal, with every
figure). Predecessor: `docs/experiments/stochastic-fit.md` and the
revised-phase rounds C1-C4 in `docs/experiments/revised-phase-campaign.qmd` +
`docs/experiments/revised-phase-handoff-2026-09-13.qmd`.

The proposal was approved on 2026-09-17 with every open decision taken as
proposed. This document records the four measurement studies that set up v2,
the model and round plan they support, the approved decisions, and the round
records (section "Round 1" below).


## Narrowed objective (2026-09-17)

The campaign is scored in three stages, and nothing is reported until the last
one succeeds or the five-round cap is reached.

1. **Parity with the legacy previous best on the frozen protocol.** DREGON
   cruise PIT MAE `≤ 2.187786 rev/s` and Michael's equal-regime mean
   `≤ 1.05 × 3.026661 = 3.177994 rev/s`, with the LTAS proxy and the comb-band
   likelihood reported beside them
   [`results/noise_v2/rounds/round1/score/findings.md:16-23`].
2. **A phase-aware four-rotor procedure, validated on synthetic data first.**
   The separability and estimator sweep is the gate on that procedure before
   it touches a real rig [`results/noise_v2/multirotor_synth/findings.md`].
3. **The `allMotors_70` rig fit reproducing the per-rotor combination** of the
   four single-motor bench fits, on in-band orders.

## Approved decisions (2026-09-17)

1. **Per-order term.** Independent per-order OU on the phase, driven
   `∝ k^{1/2}` with `p = 1` fixed and one `(σ_ε, λ_ε)` pair per parity of
   `k`, with the form — ceiling against random walk — tested in R1 by
   extending the lag grid to 1-5 s.
2. **Path term.** Per microphone, shared across rotors and `∝ f`, is a named
   candidate and is not fitted in R1.
3. **Michael's standby and ramp.** Not fitted; rendered through the trajectory
   sampler and still scored.
4. **Spectrogram proxy.** `ltas_abs_db` is the primary quantity and `mr_ltas`
   is reported as a secondary one.
5. **Proxy threshold.** Closure `≥ 0.70`, i.e. 1.979 dB on DREGON cruise and
   1.220 dB on Michael's cruise.
6. **Likelihood gate.** Composite risk below the speed-matched stationary
   oracle on Michael's cruise by a margin frozen in R1 from the parity fit,
   pooled over the cruise supports and reported per band with the comb band
   decisive; DREGON free flight is not gated by the likelihood.
7. **Floor.** The floor is a flight quantity fitted on the flight support with
   the comb frozen from the bench, the band edge is 300 Hz, and the floor
   level is constant per support in R1, with per-frame variation named as a
   later fix.
8. **Bench stationarity rule.** The longest sub-segment where the demodulated
   residual at `k ≈ 70` stays inside ±1 Hz
   (`utils.demod.residual_frequency`).
9. **Four-motor validation tolerance.** Frozen from R1: the per-rotor spread
   of the single-motor fits is reported in R1 and then frozen as the
   criterion.

Revised round plan, cap 5:

| Round | Content |
|---|---|
| R1 | Pyro model; bench fits (DREGON per rotor plus the 135 survey bench points); four-motor validation; the decoherence lag extension to 1-5 s; Michael's FLY125 cruise fit; the DREGON floor fit; an HPPNet probe; round record — **scored, FAIL**, see "Round 1" |
| R2 | Bench rule (level gate + narrow in-window residual + in-window line margin, wide test dropped); bench carrier refined in window then frozen, no in-fit refinement; standby/ramp render diagnosis; profile-block convergence; shaft prior, and the refinements Michael's cruise fit needs |
| R3 | Hierarchy over rigs including the bench points |
| R4 | One named fix |
| R5 | Simplify the passing incumbent |


## Review decisions (2026-09-16)

- **DREGON is fitted on the bench, not in free flight.** The rig parameters are
  fitted rotor by rotor on the single-motor bench recordings
  (`motor_Motor{1-4}_{50..90}`). The static four-motor recording
  (`motor_allMotors_70`) is the validation: the four-motor fit procedure must
  give parameters compatible with the per-rotor fits (slightly larger
  linewidths and more uncertain low-order estimates are expected, from
  inseparability and wind noise). Reason: free-flight DREGON telemetry
  disagrees with the audio by about 1 Hz std at the k = 70 harmonic
  (demodulation at several orders, Dmitrii's notebook), so free-flight labels
  do not carry a fit. Free-flight DREGON stays the HPPNet evaluation support
  on the real carrier, as frozen.

## Motivation

The revised-phase campaign stopped after four rounds with one rig passing and
one failing. C3 passes every gate on Michael's FLY124 (PIT MAE 1.8938 against
the 3.0267 rev/s baseline, ratio 0.6257) and fails all three on DREGON room 2
(PIT MAE 69.3114 against the 1.8971 rev/s target; composite upper bound
+6414.08; LTAS 2.4338 dB against 2.1362)
[`revised-phase-campaign.qmd:79-103`].

The DREGON fit explains why. It puts `D = 844.692917 rad²/s` on every
harmonic, which is a 134.44 Hz linewidth at every order. A shaft term scales
as `k²`, so a per-order-constant `D` cannot come from the shaft, and 134 Hz is
far above any physical linewidth. The mechanism that widens every line
equally is a misplaced carrier: DREGON room 2 supplies the flight-controller
command `motors_command`, not a measured speed. Under a mean-prediction risk a
misplaced narrow line costs more than a blurred one, so the optimiser blurs
the comb.

Three things were therefore missing before any v2 fit could be defended.

1. **A measured shaft prior.** C2/C3 fix `λ_ref = 6 s⁻¹` as a reference
   assumption and put `log σ, log D ~ N(0, 2²)` on the rest
   [`revised_phase.py:481-487,1763-1767`]. Nothing measured the shaft.
2. **More rigs.** Two rigs cannot support a rig-level hierarchy, and the
   training sampler still draws from hand-set banks.
3. **Two defensible secondary criteria.** The campaign gates on the HPPNet
   primary plus a composite risk and an LTAS deviation, and the fit window and
   the spectrogram proxy were never chosen from data.

## Setup

### Scripts

| Script | Output | Job |
|---|---|---|
| `scripts/noise_v2_bench_speed.py` | `results/noise_v2/survey/bench_speeds_{dload,remote,chums16,local}.json` (276 readings), 2 figures | `nv2-gate-dload-3319dc`, `nv2-gate-daset-b3efb4`, `uni-cpu`; ChuMS and the two `data/`-only corpora local |
| `scripts/noise_v2_corpus_survey.py` | `results/noise_v2/survey/{survey.json,survey_table.md}`, 2 figures, the ingestion manifest | local |
| `scripts/noise_v2_shaft_phase.py` | `results/noise_v2/shaft/{findings.md,shaft.json}`, 5 figures | local |
| `scripts/noise_v2_residual_acf.py` | `results/noise_v2/residual/{findings.md,residual.json}`, 2 figures | local, 9.1 s |
| `scripts/noise_v2_likelihood_window.py` | `results/noise_v2/criteria/likelihood.json`, 4 figures | local |
| `scripts/noise_v2_spectrogram_proxy.py` | `results/noise_v2/criteria/proxy.json`, 4 figures | local |

The composite-risk and proxy stages had to run on the laptop. `omnirun` ships
the git revision only, and both the C3 exports (under `.worktrees/**`) and
`results/S2/cruise_8clip_refined.json` / `standby.json` are gitignored, so the
submitted jobs died with `FileNotFoundError` on the C3 export
[`results/noise_v2/criteria/findings.md:354-359`]. Any v2 round that scores
the C3 or current-best arm remotely needs those artefacts tracked first.

### Datasets

- `dload:DREGON-frames`, `dload:michaels-frames` — unchanged fit and scoring
  material.
- `dload:SPCUP19-egonoise`, `dload:DroneAudioSet`, `dload:KAIST-rotating-acoustic`,
  `data/drone_audio`, `data/zenodo_drone_noises` — survey inputs.
- **New:** `noise-v2-bench-points@00f32a1210d6`
  (`00f32a1210d656ad0ebe60b70ddd6a60a3d204be4b07a79cc02f857dd3d1a159`,
  `dload.lock:55`), generator `noise_v2_bench` in
  `src/data_processing/derivations.py`, manifest
  `src/data_processing/noise_v2_bench_points.json`. 135 fit points over 18
  rig/condition cells; 135 samples, 14 shards, 1.9 GiB as reported by
  `dload pull`. It supersedes `@8f49bb0f77d3` (first pass, 73 points, 11
  cells) and `@0dc685bf949a` (margin-gated second pass, 78 points), neither of
  which is pinned any more. The speed labels are estimator output, so the
  committed manifest is hashed into the derivation spec: a re-estimation mints
  a new dataset identity instead of republishing different labels.
- Telemetry corpora read by the shaft study: `neurobem`, `blackbird`, `vid`,
  `nanobench`, `pitcn`, plus DREGON room 1 and Michael's own logs.

## Survey outcome

One question: which recording can become a **fit point** — a stationary window
with one known speed per rotor. Full narrative:
`results/noise_v2/survey/findings.md`; full table with licences and
per-recording reasons: `results/noise_v2/survey/survey_table.md`.

Three corrections were made after review, and all three changed the counts.

1. **AVQ is not free flight.** Its specification table (QMUL, `Type` = EO,
   `Drone` = constant) marks four sequences as ego-noise only at a constant
   throttle: S1 seq1 50 % (120 s), S1 seq2 100 % (120 s), S1 seq3 150 %
   (40 s), S2 seq1 100 % (210 s). Read in 30 s windows they give 78.0 / 97.8 /
   111.4 rev/s for 50 / 100 / 150 %, monotone in the table's own throttle
   column, and **6 fit points over 2 cells**
   [`survey/findings.md:174-204`].
2. **The 3 dB margin was the wrong gate for a multi-rotor rig.** It collapses
   by construction once four combs diverge at high order: one DREGON motor at
   70 % scores 15.3-17.8 dB of comb score, and four motors on the same rig at
   the same throttle score 6.43 dB. The margin measured rotor count, not
   readability. A multi-rotor rig is now gated on **line evidence** — `f̄`
   stable to `≤ 1 rev/s` across the two window halves, and a peak `≥ 6 dB`
   over the band's local median inside `k·f̄ ± 6 %` at `≥ 3` orders of the
   line comb — with the per-rotor split taken from high-order line clustering
   (`≥ 3` orders agreeing within 0.3 rev/s). The margin is still stored, and
   is no longer gated on. Neither cheaper repair works: the `±6 %`
   neighbourhood exclusion flips 1 verdict of 240 rows and a `--family-max 8`
   counterfactual flips none [`survey/findings.md:74-107`].
3. **The four-rotor control is now usable.** `motor_allMotors_70` resolves
   four rotors at **64.65 / 67.66 / 68.74 / 69.57 rev/s** (mean 67.66, spread
   4.92) against the single-motor law's 68.6 rev/s at 70 %, on peaks at 27
   orders of the line comb. The first pass read three rotors from one order
   and rejected it [`survey/findings.md:136-142`].

| corpus | read | usable, pass 1 | usable, now | verdict |
|---|---|---|---|---|
| DREGON bench (single motor + all-motors) | 21 | 20 | **21** | USABLE |
| SPCUP19 AGH single rotors | 8 | 7 | 7 | USABLE |
| DroneAudioSet drone-only | 168 | 50 | **98** | PARTLY USABLE |
| SPCUP19 static / hover (4 rigs) | 18 | 0 | **7** | PARTLY USABLE |
| SPCUP19 ChuMS propeller rig (16 s window) | 9 | 1 | **2** | PARTLY USABLE |
| AVQ constant-throttle ego-noise | 16 windows | — | **9** | USABLE |
| KAIST rotating machine | 5 | 0 | 0 | CONTROL ONLY |
| `drone_audio` (`yes_drone`) | 24 | 0 | 0 | REJECTED (1.02 s clips) |
| `zenodo_drone_noises` | 7 | 1 | 4 | REJECTED (no rig identity) |
| DronePrint / MAVD / DroneNoise DB / ESC-50 | — | — | — | REJECTED (access/type) |

Result: **135 fit points over 18 rig/condition cells** (276 readings, 148
usable), with a median published tolerance of 0.25 rev/s. The first pass gave
73 fit points over 11 cells from 79 usable readings
[`survey/findings.md:206-237`].

Cross-checks that make the estimator usable as a label source:

- DREGON single-motor bench against the validated throttle law
  `rate = 0.975·throttle + 0.37 rev/s`: 20/20 usable, mean |error|
  **0.5552 rev/s**, max 1.42, RMS 0.703. An independent refit of the survey's
  own readings gives slope 0.97577, intercept 0.2395 rev/s, `R² = 0.997444`.
  Every number is identical to the first pass, because single-rotor mode is
  untouched by the gate change [`survey/findings.md:129-134`].
- SPCUP19 AGH against the accepted blind readings: 7/8 usable, six agree to
  `≤ 0.27 rev/s` (mean |error| 0.148 rev/s). Take 1 reads 132.30 against
  66.33 rev/s, a factor 1.995, which the blind campaign had already flagged
  [`survey/findings.md:144-147`].
- DroneAudioSet against the paper's stated lines (arXiv:2510.15383, Fig. 7):
  three of four cells match to `≤ 2.7 %` via the blade-pass line `2f`, on
  17-28 readings per cell (169.9 / 237.4 / 252.1 Hz measured against
  168 / 235 / 259 Hz). That identifies `drone1` as `D_large` (DJI F450) and
  `drone2` as `D_small` (DJI F330), and settles that the paper's marked
  "fundamental" is the blade-pass line, not the shaft rate. The `drone2` low
  cell disagrees by +35.7 % on 28 readings, so the disagreement is in the
  published figure [`survey/findings.md:156-172`].
- KAIST rotating machine, the out-of-domain control: 0/5 usable, unchanged. It
  is a single-rotor corpus, so it exercises the unchanged 3 dB gate and does
  not falsify the new one; the falsification checks for the new gate are the
  four-rotor DREGON control and the cell-median consistency check
  [`survey/findings.md:149-154`].

Two caveats to carry forward. 37 of the 148 usable readings carry
`octave_unresolved` under the relaxed octave rule, and the 9 factor-2 failures
it admitted were removed by a per-cell median check that only works where a
cell holds `≥ 3` readings — the SPCUP stationary and ChuMS cells do not. And
**no new corpus adds telemetry**: every extra point is an estimate with a
0.25 rev/s median tolerance and carries no speed track, so the new points
serve rig-to-rig transfer of the noise parameters only.

Rotor multiplicity over the 135 points: 76 resolve one rate, 42 two, 10 three
and 7 four; 99 carry `multiplicity_unresolved`, i.e. at least two rotors
coincide within 0.3 rev/s and `speed_rev_s` repeats `f̄` for them. Most
four-rotor points therefore carry a mean rate plus a partial split, and the v2
likelihood fits the per-rotor offsets: a bench point with four rotors enters
with four constant carriers as fit parameters, initialised from the split
[`survey/findings.md:268-275`].

## Shaft-phase outcome

Source: `results/noise_v2/shaft/findings.md`, numbers
`results/noise_v2/shaft/shaft.json`.

- Seven of eight telemetry rigs prefer a Lorentzian speed spectrum over a flat
  one, by `ΔAIC(OU − white)` from `-2.0e4` to `-1.1e6`. The speed error is
  therefore not white, and the phase is integrated-OU rather than Wiener. The
  eighth is Michael's 29 Hz log: its fitted corner, 27.5 Hz, sits above its
  own 0.5-11.7 Hz band, so in band the error is white (`ΔAIC = +736`).
- The Lorentzian corner is identified on five of eight rigs, at
  `λ = 0.674-14.4 s⁻¹` (corner 0.107-2.30 Hz) under the study's selected trend
  removal. That range **contains** the declared `λ_ref = 6 s⁻¹` of C2/C3, so
  the old value is not refuted. It is also not confirmed: across every rig and
  trend-removal variant whose corner lands in band the fitted `λ` spans
  0.674-51.9 s⁻¹, a factor of 77. `neurobem` and `blackbird` identify no
  corner at all, so only `σ²/λ` is identified on them. **`λ` is bracketed, not
  pinned**, which is why v2 gives it a broad measured prior instead of a fixed
  value.
- **Do not read `λ` from the slope-1.5 crossing of the structure function.**
  That crossing is read off the FILTERED series, which saturates at about
  `1/(2πf_hp)`, so it moves with the high-pass: 0.135-0.349 s at 0.5 Hz,
  0.024-0.083 s at 2 Hz, 0.61-3.41 s with a linear detrend
  [`results/noise_v2/shaft/findings.md:68-114`]. The 0.5 Hz column would give
  `λ = 6.16-16.0 s⁻¹` and appear to confirm `λ_ref = 6`; that agreement is
  partly the filter corner, not the rig.
- `D_q/D_θ ≤ 1.1e-6` on every rig, so no rig's diffusive tail is a
  quantisation artefact of its telemetry channel.
- **Consequence for the model.** An OU speed error leaves no diffusion above
  the label band: the 1/f² speed tail integrates once to a 1/f⁴ phase
  spectrum, which is a stationary **bounded** wobble of
  `Var θ_res = 2.5e-5` to `6.8e-3 rad²` on the shaft study's 16 Hz residual
  [`results/noise_v2/shaft/findings.md:44-61`].
- **The residual study settles what to generate there.** The speed error above
  the label band is NOT white on any of the 7 rigs whose Nyquist exceeds
  16 Hz: the raw PSD keeps falling at ≈ f⁻² above the band edge (slopes -1.50
  to -2.56 on the four rigs that measure the shaft), and the autocorrelation
  at 50 ms is +0.12 to +0.60 against a filtered-white null of 0.002-0.047. The
  curve rings at the band edge (zero crossing 8.6-14.3 ms, minimum down to
  -0.91) and dies by 100 ms. So the speed error is ONE Ornstein-Uhlenbeck
  process whose f⁻² tail continues above the label band, the phase error is
  integrated-OU rather than Wiener at every frequency, and the above-band part
  is a bounded wobble of rms 0.003-0.055 rad at `k = 1` (0.09-1.65 rad at
  `k = 30`) that a renderer must generate as filtered noise with that PSD
  [`results/noise_v2/residual/findings.md`]. Three instrument caveats:
  `neurobem` (-4.36) and `nanobench` (-29.85) show the logger's reconstruction
  roll-off, DREGON `motors_measured` is a 45 Hz sample-and-hold, and Michael's
  29.2 Hz log holds nothing above 14.6 Hz.
- So the `k²`-scaling Lorentzian core that the acoustic fit needs is **label
  error**, not rotor physics: Michael's 29 Hz sample-and-hold, and DREGON
  room 2's command-against-shaft motor response. C3's DREGON
  `σ²/λ_ref = 1.603 rad²/s` is 15 times DREGON's own label-residual bound of
  0.105 rad²/s, and its `D = 844.69 rad²/s` is 8000 times that bound.
- The label-residual variant is an **upper bound** on a real label chain's
  leak, not the leak: the study's resampler is linear interpolation in both
  directions, which has a lossy passband, so part of the measured growth
  (`D_θ = S(1 s)/2 = 0.0014-0.41 rad²/s`) is the estimator's own error.
- **The bench acoustics measure the shaft term directly.** The DREGON
  single-motor bench is the only instrument with no label at all, and the
  corrected saturation gate identifies the shaft term on **12 of 12
  recordings**, integrated-OU on every one by AIC: median `σ_ν = 1.77 rad/s`
  (IQR 1.39-1.93, full range 0.23-3.20), `λ = 5.48 s⁻¹` (IQR 3.25-8.29, full
  0.31-16.2), `D_θ = 0.599 rad²/s` (IQR 0.38-0.66, full 0.018-2.81). `λ` lands
  on the assumed `λ_ref = 6` to 9 %, and C3's fitted `σ = 3.101 rad/s` is 1.75
  times the bench median. This is filter-free: the acoustic estimator has no
  high-pass, so none of the trend-removal ambiguity above applies to it.
- **Two instruments, one verdict on C3's DREGON `D`.** They refute
  `D = 844.69 rad²/s` by three to four orders of magnitude. In the frequency
  domain that `D` predicts a 268.9 Hz half-power width against a 38.5 Hz
  isolation band, while the measured core is unresolved at every window up to
  4.096 s (intrinsic width `< 0.46 Hz`, per-harmonic `D < 1.45 rad²/s` on
  Michael's cruise). In the time domain the bench gives
  `D_θ = 0.38-0.66 rad²/s` with `σ_ν = 1.4-1.9 rad/s` and
  `λ = 3.3-8.3 s⁻¹`. The two bounds are different quantities and both matter:
  the line width bounds the `k`-independent per-harmonic `D`, `V_k/k²` bounds
  the shaft `D_θ`, and the observed broadening is the sum of the two.
- **`D_k` and its `k` exponent are still UNIDENTIFIED.** The fitted `2D_kτ` at
  the shortest lag is 1.32 times the measured `V` of the same order, so `D_k`
  absorbs model error, and the planted control returns a planted `D_k ∝ k` as
  flat (fitted log-log slope -5.74 against a true +1). Read `σ_ν`, `λ` and
  `D_θ` from this instrument; do not read `D_k` from it, and keep `p_g = 0`.
- **The bench numbers carry a measured systematic.** The planted control
  plants `σ_ν = 0.4 rad/s`, `λ = 6 s⁻¹` and `D_k ∝ k` at 22 dB SNR and
  recovers `σ_ν` at 1.26×, `λ` at 3.07× and `D_θ` at 0.52×. De-biased, the
  bench gives `σ_ν ≈ 1.4 rad/s` (1.1-1.8), `λ ≈ 1.8-5.5 s⁻¹` and
  `D_θ ≈ 0.6-1.2 rad²/s`; even the pessimistic end is within a factor of 3 of
  `λ_ref = 6`. The de-biased bands set the v2 prior widths.
- **Michael's FLY125 still saturates**: 3 of 4 rotors exceed 0.15 rad² at the
  shortest usable lag and 0 of 4 are identified. That is consistent with the
  line-shape bound `D < 1.45 rad²/s` on the same rig but adds no number of its
  own. The cause is bandwidth: one harmonic cannot be separated from its
  neighbours faster than one shaft revolution, so the acoustic lag grid starts
  at the frame rate of a 6-revolution window.

## Proposed criteria

Source: `results/noise_v2/criteria/findings.md`. The study ranks and
quantifies; it does not choose. Both proposals are stated as gap-closure
fractions against a speed-matched real-vs-real oracle, in the same form as the
HPPNet gate.

**Window: `T = 1.024 s`, periodic Hann, 50 % overlap.** The binding number is
the carrier drift: harmonic `k` leaves the window's own resolution above
`T_max(k) = sqrt(1.44/(k|df_r/dt|))`, and the Michael's standby p95 slope
gives `T_max(k = 2) = 0.95 s`. `T = 1.024 s` also keeps 6 Welch segments on
the shortest 4.0 s DREGON support, sits past the point where doubling the
window still changes the line shape beyond noise, and already collects a gated
line on Michael's `k = 2`.

The line shape refutes the C3 DREGON fit independently. The line core is
unresolved at every `T ≤ 4.096 s`: the measured half-power width tracks
`1.44/T` over a factor of 32 in window length, so Michael's `k = 2` line is
narrower than 0.46 Hz, which implies `D < 1.45 rad²/s`. That brackets the C3
Michael's value (1.6308, predicted width 0.52 Hz) and refutes the C3 DREGON
value (844.69, predicted width 268.9 Hz — 7.0 times the whole 38.5 Hz comb
isolation bandwidth).

**What that statistic measures is LABEL PRECISION, per order and per window.**
The harmonic is demodulated with the label's own integrated phase, Welch
averaged at `T = 4.096 s` (0.24 Hz bins); the line is the largest bin inside
`±0.5 Hz` of the position the label predicts, and the floor is the median
19-38 Hz away. A label off by `δ` rev/s displaces harmonic `k` by `k·δ` Hz, so
the line keeps its prominence only while `k·δ < 1/(2T) = 0.12 Hz`. Fraction of
cells inside the box at the 10 dB gate: Michael's 0.95 at `k = 2`, 0.85 at
`k = 4`, 0.23 at `k = 8`, 0.00 at `k ≥ 16`; DREGON 0.31 at `k = 1` and
0.00-0.06 at `k ≥ 2` with either carrier. The mid- and high-order lines are
real — they stand out as horizontal bands in the NFFT-2048 spectrogram of the
same support — but the label does not locate them, and a fixed-carrier model
has to absorb the `k·δ` displacement; C3 absorbed it on DREGON with
`D = 844.69 rad²/s`. Two carriers remain indistinguishable to this criterion:
6.4 dB (`motors_command`) against 7.1 dB (`rps_refined`) at `k = 1`,
half-power widths inside 0.13 Hz. The DREGON row is also an instrument limit:
the per-rotor isolation band is only `B = f_r/2 = 38.5 Hz`, and at `k ≥ 2` it
already contains the neighbouring rotors' lines of the same order. Which
orders enter the DREGON likelihood, given the label precision `k_max(T)`, is
therefore a reviewer decision.

**Proxy: `ltas_abs_db`, mic 0, per rig on the cruise supports.** It has the
largest `σ` response (1.81 dB on Michael's cruise against 0.03-0.40 dB for
every rival), the largest separation ratio (4.62 on Michael's cruise) and the
largest ladder dynamic range in `D` (6.17 dB on DREGON cruise),
and a monotone degradation branch (`ρ = +1.00 / +0.90`). `mr_ltas` is second:
best `D` response on DREGON but dead on Michael's cruise. `texture` and
`modulation`, with or without a comb-bin restriction, are rejected — their
ladder ranges are 0.010-0.053 on their own scales. Whitening in the log domain
is a proven no-op (`max |unwhitened − whitened| = 6.7e-15 dB` over 142 rows).

Two threshold options, both of which the current best (2.126 / 1.332 dB,
closure 0.47 / 0.45) fails:

| closure | DREGON cruise | Michael's cruise | margin against current best |
|---|---|---|---|
| 0.70 | 1.979 dB | 1.220 dB | 0.147 / 0.112 dB |
| 0.50 | 2.108 dB | 1.309 dB | 0.018 / 0.023 dB |

0.70 is the frozen `gates.dregon_gap_fraction`, so it adds no new number; 0.50
is the weakest fraction that still excludes the current best. Closure 0.40
would let it pass. On DREGON the gate is support-noise limited — the oracle
spread is 1.142 dB across five supports against a C3-to-oracle gap of
0.646 dB — so it must be stated as a five-support mean with the spread quoted.

**The composite-risk table was NOT computed.** The local `predict_spectrum`
path costs about 20 s of model time per audio-second per setting, the declared
grid is 60 audio-seconds × 3 NFFT settings, and the run reached 2 h 31 min
under CPU contention before it was stopped; it could not be sent to `uni-cpu`
because the C3 exports and the `results/S2/*.json` baselines are gitignored
[`results/noise_v2/criteria/findings.md:159-205`]. The likelihood gate
therefore has its form and its NFFT (16384, the only one that resolves a
sub-0.46 Hz core) but no absolute value. The stage is implemented and
idempotent; `scripts/noise_v2_likelihood_window.py --stage composite` produces
the table once those four artefacts are tracked.

## Model and round plan

The model v2 shaft term has two honest parts: a fitted label-residual OU with
free `(λ_L, σ_L)` **per label chain** below the label band, and a bounded
wobble above it whose `(σ_ν, λ)` are **fixed** from the telemetry. The
per-harmonic Wiener `D_gk = D_g·k^{p_g}` stays, with `p_g = 0` fixed in R1 and
R2. The OU block has two free parameters, not three: the fit samples
`(log σ_L, log λ_L)` and `D_θ,L = σ_L²/λ_L` follows. Their priors replace
`N(0, 2²)` and are **centred on the label-free bench measurement** —
`log σ_L ~ N(ln 1.77, 0.95²)`, `log λ_L ~ N(ln 5.5, 1.50²)` — widened by the
planted-control systematic and far enough to reach the label-dominated regime,
so a fit landing above `λ_L ≈ 40 s⁻¹` is a diagnosis that the label chain and
not the shaft dominates that arm. `log D_g ~ N(ln 0.3, 0.80²)` comes from the
line-shape bound, because `D_k` is still unidentified. A population
Gaussian over rigs — DREGON, Michael's, and the survey's 18 bench cells —
replaces the hand-set banks. A bench point with four rotors enters with four
constant carriers as fit parameters, initialised from the survey's per-rotor
split. Carrier recovery, heavy tails, order-dependent `λ_L` and
non-stationary `D` are deferred by name.

Implementation is a new sandbox package `src/experiments/noise_model/` on
torch and pyro, with a differential parity test against `revised_phase.py` to
`1e-6` relative.

Round plan, cap 5, backends `uni-cpu` and `uni-gpushort`:

| Round | Question | Pass = |
|---|---|---|
| R1 | Pyro parity: C4 model, fixed `λ_ref`, MAP, DREGON raw + refined, Michael's | likelihood parity test; gates scored; DREGON refined-label arm result |
| R2 | Shaft prior: `λ` free with telemetry/bench prior, `D_k` structure from bench | frozen gates |
| R3 | Hierarchy over rigs incl. bench points; population posterior for sampling | gates + posterior predictive on a held-out bench point |
| R4 | Fix the one named failure of R1-R3 (one complexity per failure) | gates |
| R5 | Simplify the passing incumbent; publish fits; wire engine | gates unchanged |

The primary criterion carries over verbatim: frozen `hppnet_l2_r2_s0` best
checkpoint (digest
`6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1`), synthetic
rendered on the REAL carrier of the same DREGON room 2 cruise supports and
FLY124 supports; DREGON `E_new ≤ E_real + 0.70·(E_best − E_real) =
1.897 rev/s` with the one-sided 95 % recording-level interval excluding no
improvement; Michael's regime-mean ratio `≤ 1.05` against 3.027 rev/s
[`revised-phase-handoff-2026-09-13.qmd:116-133`;
`revised-phase-campaign.qmd:341-351`].

## Round 1

**Verdict: FAIL.** Both HPPNet parity gates fail, the DREGON stretch gate
fails and both proxy gates fail; the comb-band likelihood is the one gate that
passes, and no fit in the round converged. Record
`results/noise_v2/rounds/round1.json` (git `066651e8eb31`),
`round1/score/findings.md`.

### Setup

All **179 of 179** declared supports were built — `bench-points` 135/135,
`dregon-bench` 21/21, `dregon-floor` 10/10, `michaels-cruise` 13/13
[`rounds/round1/supports/index.json`] — and 159 fit JSONs written
[`round1/fits/findings.md:3`].

Fit recipe of the arm the flight gates read (Michael's cruise):
`--set michaels-cruise` (FLY125 only, 8 windows), frame stride 4, 256 pooled
frames, 1500 Adam steps at lr 0.02 with batch 8, then 200 + 100 L-BFGS
iterations on a deterministic 64-frame subset (516 096 cells), seed 0, 8
threads [`round1/basin/findings.md:254-259`].

Multi-start: start 0 from the data-driven initialisation, the others from a
log-normal perturbation of the dynamics init, and the reported fit is the
start with the lowest polished objective. **1 of 23 restarted supports has
every start inside the 1e-4 nats/cell convergence tolerance.** The widest
disagreement is `bench_dregon_Motor1_70` at 0.0448 nats/cell, 448 times the
tolerance; `lam` alone spans a factor 3.03 (median over supports) and up to
1.26e+04 across the starts of one support, so the R1 bench MAP problem is
multi-modal [`round1/fits/findings.md:190-220`].

| candidate | rig | fit JSON (under `round1/fits/`) | converged | grad norm | fit wall (s) |
|---|---|---|---|---:|---:|
| `dregon_v2_floor_benchcomb` | dregon | `dregon_room2_floor__flight_floor_only.json` | **no** (`which_converged=none`) | 2318 | 2377 |
| `michaels_v2_fly125cruise` | michaels | `michaels_fly125_cruise__flight.json` | **no** | 8555 | 6470 |
| `michaels_v2_retry_pinned_lam` | michaels | `michaels_fly125_cruise__flight_retry.json` | **no** | 2863 | 4799 |

The three candidates [`round1/score/findings.md:7-11`].

### Results

| gate quantity | candidate | number | PARITY bar | STRETCH bar |
|---|---|---:|---|---|
| DREGON five-recording cruise PIT MAE mean (rev/s) | `dregon_v2_floor_benchcomb` | 78.803342 (95 % upper 79.482310) | 2.187786 — **FAIL** (−76.615556) | 1.897063 — **FAIL** (−76.906280) |
| Michael's equal-regime mean PIT MAE (rev/s) | `michaels_v2_fly125cruise` | 16.799053 (ratio 5.550358) | 3.177994 — **FAIL** (−13.621059) | — |
| Michael's equal-regime mean PIT MAE (rev/s) | `michaels_v2_retry_pinned_lam` | 19.376405 (ratio 6.4019) | 3.177994 — **FAIL** | — |

HPPNet gates [`round1/score/findings.md:16-31`]. PARITY is the legacy
previous best (DREGON synthetic cruise 2.187786 against a real arm of
1.218708; Michael's 1.05 × 3.026661); STRETCH exists only on DREGON,
1.897063 = real + 0.7 × (baseline − real).

| regime | blocks | `michaels_v2_fly125cruise` | legacy baseline | ratio |
|---|---:|---:|---:|---:|
| standby | 2 | 32.632045 | 0.317103 | 102.9069 |
| ramp | 1 | 16.955751 | 8.007098 | 2.1176 |
| cruise | 2 | 0.809363 | 0.755783 | 1.0709 |

Michael's FLY124 per regime, rev/s [`round1/score/findings.md:68-76`]. Only
cruise — the one regime that was fitted — is near parity; standby and ramp are
rendered through the trajectory sampler and never fitted.

| proxy / likelihood | candidate | number | reference | verdict |
|---|---|---:|---:|---|
| `ltas_abs_db` dregon_cruise (dB) | `dregon_v2_floor_benchcomb` | 3.7518 (spread 3.2542) | gate 1.9786 | FAIL (−1.7732) |
| `ltas_abs_db` michaels_cruise (dB) | `michaels_v2_fly125cruise` | 1.4156 (spread 1.6608) | gate 1.2197 | FAIL (−0.1959) |
| `ltas_abs_db` michaels_cruise (dB) | `michaels_v2_retry_pinned_lam` | 1.6330 | gate 1.2197 | FAIL |
| `mr_ltas` (dB), dregon / michaels / retry | — | 2.6359 / 1.6456 / 1.5963 | — | report only |
| likelihood comb band (nats/s) | `michaels_v2_fly125cruise` | −373 414.1187 | oracle −371 916.5041 | PASS, margin −1 497.6146 |
| likelihood floor band (nats/s) | `michaels_v2_fly125cruise` | −8 487.7119 | oracle −8 713.2663 | report, +225.5544 |
| likelihood full band (nats/s) | `michaels_v2_fly125cruise` | −381 901.8305 | oracle −380 629.7704 | report, −1 272.0601 |

Proxy and likelihood [`round1/score/findings.md:37-47,90-104`]. The
comb-band margin −1 497.6146 nats/s is frozen for later rounds; the retry's is
−1 200.5896, and the retry is worse than the unpinned fit on every gate the
render feeds.

Two plumbing checks hold. The real arm reproduces the frozen DREGON real PIT
MAE 1.218708 rev/s to 2.7e-09 relative, and the recomputed full-band oracle
matches `short_whittle.json` exactly [`round1/score/findings.md:64-66,106`].
The legacy smoke replay is `not_comparable`: DREGON reproduces 2.187786
(4.79e-08) but Michael's replay reads 3.199486 against the frozen 3.026661
(5.71e-02), because the gitignored baseline calibration that selects the export
family was unavailable [`round1/score/findings.md:114-121`].

### Basin: what the flight objective identifies

Scans of the dynamics block at the Michael's cruise fit, every other block held
[`round1/basin/findings.md`]:

- **Identified:** `sigma_nu` (1.996778 nats/cell over ±2 decades, zero plateau
  at the 1e-4 tolerance); `A_odd = sigma_eps_odd² = 0.354395`; and the product
  `A_odd·lam_eps_odd = 26.8040 rad²/s per k` (0.427175 nats/cell across).
- **Not identified:** `lam` — 2.86 decades (1.39 … 1000) lie within 0.0307066
  nats/cell and the whole gain along 1.7 decades (19.3 → 1000) is 0.018564
  nats/cell, the same size as the 0.01875590 nats/cell restart gain that failed
  the convergence test; the split of `A_odd·lam_eps_odd` into its two factors;
  and the even per-order term in both coordinates, whose total leverage
  (5.2e-5 and 2.3e-4 nats/cell) is at or below the tolerance — an absent term,
  not a ridge [`round1/basin/findings.md:224-250`].
- **The pinned retry did not converge.** With `lam = 0.5` (the long-lag value,
  costing 0.039377 nats/cell against the likelihood argmin 268.27, inside the
  0.05 nats/cell rule), `lam_eps_even = 2.0` and `lam_eps_odd = 75.63307`
  pinned, the restart gain was 0.02172926 nats/cell — slightly worse than the
  unpinned 0.01875590 — while the gradient norm fell from 8555.25 to 2863.34.
  The ridge prediction itself checks out (the slice put `sigma_nu` at 6.0299,
  the retry landed at 5.985662, 0.73 % away) and the retry is 0.014414
  nats/cell better on the 64-frame polish objective, but **0.013059 nats/cell
  worse on the full pooled objective**. Removing every flat dynamics direction
  did not buy convergence, so the remainder lives outside the dynamics block —
  in the 520-parameter profile block, the floor, or `amp_exp` (10.02 → 16.43)
  [`round1/basin/findings.md:299-352`].

### Root cause of the DREGON failure

The DREGON row is not a parity measurement: the bench combs frozen into it come
from windows with no motor running [`round1/bench_diag/findings.md`].

- **12 of 21 `dregon-bench` supports sit ≥ 10 dB below the loudest same-length
  window of their own recording** (10 of them ≥ 20 dB, 13 ≥ 6 dB)
  [`round1/bench_diag/census.json`]. `bench_dregon_Motor1_70`'s frozen window
  [35.707, 43.0] s is 24.28 dB down — the motor stops at ~36 s — and
  `Motor2_60`'s [15.9, 23.0] s is 21.37 dB down.
- **The rule tests the frequency residual only.** `stationary_segment` scores
  stationarity, and silence is perfectly stationary; `line_present` is measured
  on the whole recording and never inside the chosen window, so the rule walks
  onto the post-spin-down tail. The comb is worth 2.42 nats/cell on a motor-on
  window of the same recording against 0.14 on the committed one
  [`round1/bench_diag/findings.md:42-49,249-252`].
- **The in-fit carrier re-refinement locked onto an interferer.** On silent
  windows it wanders (+0.256 and −0.762 rev/s) and the MAP carrier walks 3.791
  prior sigmas to 70.1955, where `k` = 14/28/42/56 land on a fixed 89.268 Hz
  family (982.354 / 1965.12 / 2947.47 / 3930.24 Hz at 21.5-26.4 dB over the
  local floor) whose absolute level is the same with the motor on and off, i.e.
  a rig/room source independent of rotor speed
  [`round1/bench_diag/findings.md:123-137`].
- **The frozen index carriers are biased low by +0.09 … +0.20 rev/s** against
  the motor-on registration argmax (68.5154 vs 68.3174; 58.1924 vs 58.0979;
  78.2321 vs 78.0596), because `supports.refine_carrier` demodulates the whole
  recording, two thirds of which is silence
  [`round1/bench_diag/findings.md:111-117`].
- The earlier "comb 40 dB below the floor" reading was a unit misreading:
  `profile_db` is a per-order line variance, and the model's own unit-line
  response puts `Motor1_70`'s `k = 1` line at −36.4 dB against a −76.1 dB floor
  [`round1/bench_diag/findings.md:19-40`].
- The proposed level-gate patch was run offline on all 21 recordings: every
  window then lands on the motor (level deficit 0.48-2.14 dB) and every
  in-window line margin clears 3 dB (3.71-16.29 dB)
  [`round1/bench_diag/findings.md:49`].

### Four-rotor synthetic

`src/experiments/noise_model/multirotor.py`, 4 rotors, 8 mics, 24 s, 3 seeds
[`results/noise_v2/multirotor_synth/findings.md`]:

- **The real minimum spacing, 0.83 Hz, is resolvable.** Median |profile error|
  over the gated ladder is **0.71 dB for E2-multi**, against 2.20 dB
  (E2-single) and 3.58 dB (E1); at the strictest 40 dB SNR tier E2-multi reads
  0.89 dB with `k = 96` its only failing order.
- **E1's high-order failures are cross-rotor line collisions**, not order
  spacing: four incommensurate carriers put ~440 lines into 7.2 kHz (mean
  spacing ~18 Hz), so above `k ≈ 24` some line lands inside another's
  linewidth and E1 reports the equal split. It fails from `k ≈ 24` up even at
  δ = 3 Hz, and 33 % of its gated cells collapse against 11 % for E2-multi.
- **Recommended remedy:** E2-multi output as the *initialisation* of
  `profile_db` in the rig fit, plus an order-dependent Gaussian prior at this
  sweep's measured error (≈1 dB for `k ≤ 16`, ≈3 dB for `24 ≤ k ≤ 48`, no prior
  above `k ≈ 64`, where even the oracle estimator fails), with the four
  carriers fitted from `refine_offsets` and `offset_spread_ratio` (0.37 at
  0.83 Hz) reported beside them
  [`multirotor_synth/findings.md:173-200`].
- Two caveats: 43-51 dB of each `Motor*_70` fitted comb sits in orders above
  the fitted band (in-band limit `k` = 102/107/103/103), so any per-rotor
  against rig comparison must be restricted to in-band orders; and the
  `delta_0p83_real_snr` case sets its floor from the all-orders comb, so it is
  confounded and needs one re-run
  [`multirotor_synth/findings.md:142-171`].

### Conclusion

R1 FAILS parity on both rigs and fails both proxy gates; only the comb-band
likelihood passes, and none of the three fits converged. Neither failure is yet
a verdict on the model. The DREGON number measures a comb frozen from motor-off
windows, which the bench stationarity rule selected because it tests frequency
stationarity and nothing else, and which the in-fit carrier refinement then
registered onto a fixed room interferer. Michael's number is carried almost
entirely by standby (102.9×) and ramp (2.1×), which are rendered and never
fitted, while the fitted cruise regime sits at 1.0709× of the legacy baseline.
The basin scan explains the non-convergence only in part: the shaft ridge is
real and `sigma_nu` is its coordinate, but pinning every flat dynamics
direction left the restart gain where it was. R2 therefore changes four things.
The bench rule becomes a level gate plus a narrow in-window frequency residual
plus an in-window line margin, with the wide test dropped. The bench carrier is
refined in window and then frozen, with no in-fit refinement. Standby and ramp
get a render diagnosis instead of another fit. And profile-block convergence is
carried as the open fitting-method question.

## Status

**R1 scored, FAIL; R2 opening.** The round record is the section "Round 1"
above; the machine-readable record is `results/noise_v2/rounds/round1.json`
(git `066651e8eb31`). The nine approved decisions are listed above (section
"Approved decisions (2026-09-17)") and in
`docs/explainers/noise-model-v2-plan.qmd`, section "Decisions (approved
2026-09-17)".

Of the three prerequisites R1 was blocked on:

1. **Cleared.** The v2 proxy reads absolute Welch band levels with **no
   per-arm normalisation** and records a per-support `level_offset_db`
   (`src/experiments/noise_model/gates.py:90-92`), so the renderer's
   RMS-normalising wrapper no longer decides the LTAS number.
2. **Still open.** The gitignored baseline calibration
   (`results/revised_phase/baseline_v2/calibration.json`) was unavailable in
   the scoring job, which forced two re-derivations — the DREGON paired
   improvement interval and the per-band oracle risk — and left the legacy
   smoke `not_comparable` [`round1/score/findings.md:112,114-121`].
3. **Cleared.** The likelihood gate has its number: the R1 comb-band
   model-minus-oracle margin −1 497.6146 nats/s, frozen for later rounds
   [`round1/score/findings.md:94-104`].
