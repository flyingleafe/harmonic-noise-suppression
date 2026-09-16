# Noise model v2: the pre-implementation phase

**Status:** awaiting review — 2026-09-15 → . Campaign `noise-model-v2`, branch
`main`. Explainer: `docs/explainers/noise-model-v2-plan.qmd` (the full
proposal, with every figure). Predecessor: `docs/experiments/stochastic-fit.md`
and the revised-phase rounds C1-C4 in
`docs/experiments/revised-phase-campaign.qmd` +
`docs/experiments/revised-phase-handoff-2026-09-13.qmd`.

No v2 fit has run. This document records the three measurement studies that
set up v2, the model and round plan they support, and the decisions that are
waiting on the user.

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
| `scripts/noise_v2_bench_speed.py` | `results/noise_v2/survey/bench_speeds_remote.json` (229 recordings), `bench_speeds_local.json` (31 recordings) | `nv2-bench-remote-e1e19f`, `uni-cpu` |
| `scripts/noise_v2_corpus_survey.py` | `results/noise_v2/survey/{survey.json,survey_table.md}`, 3 figures, the ingestion manifest | local |
| `scripts/noise_v2_shaft_phase.py` | `results/noise_v2/shaft/{findings.md,shaft.json}`, 5 figures | local |
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
- **New:** `noise-v2-bench-points@8f49bb0f77d3`
  (`8f49bb0f77d3fb3c85a70b0ccffc7b9893d673e6e7e0cdab84dbf239c513f337`,
  `dload.lock:55`), generator `noise_v2_bench` in
  `src/data_processing/derivations.py:2129`, manifest
  `src/data_processing/noise_v2_bench_points.json`. 73 fit points over 12
  rig/condition points. The speed labels are estimator output, so the
  committed manifest is hashed into the derivation spec: a re-estimation mints
  a new dataset identity instead of republishing different labels.
- Telemetry corpora read by the shaft study: `neurobem`, `blackbird`, `vid`,
  `nanobench`, `pitcn`, plus DREGON room 1 and Michael's own logs.

## Survey outcome

One question: which recording can become a **fit point** — a stationary window
with one known speed per rotor. Full narrative:
`results/noise_v2/survey/findings.md`; full table with licences and
per-recording reasons: `results/noise_v2/survey/survey_table.md`.

| corpus | read | usable | verdict |
|---|---|---|---|
| DREGON single-motor bench | 21 | 20 | USABLE |
| SPCUP19 AGH single rotors | 8 | 7 | USABLE |
| DroneAudioSet drone-only | 168 | 50 | PARTLY USABLE |
| SPCUP19 ChuMS propeller rig | 9 | 1 | PARTLY USABLE |
| SPCUP19 static / hover (4 rigs) | 18 | 0 | REJECTED |
| KAIST rotating machine | 5 | 0 | CONTROL ONLY |
| `drone_audio` (`yes_drone`) | 24 | 0 | REJECTED |
| `zenodo_drone_noises` | 7 | 1 | REJECTED (no rig identity) |
| AVQ | — | — | REJECTED (free flight) |
| DronePrint / MAVD / DroneNoise DB / ESC-50 | — | — | REJECTED (access/type) |

Result: **73 fit points over 12 rig/condition points**, with a published
tolerance of 0.25 rev/s on 72 of them and 0.56 rev/s on the ChuMS run.

Cross-checks that make the estimator usable as a label source:

- DREGON single-motor bench against the validated throttle law
  `rate = 0.975·throttle + 0.37 rev/s`: 20/20 usable, mean |error|
  **0.555 rev/s**, max 1.42, RMS 0.703. An independent refit of the survey's
  own readings gives slope 0.97577, intercept 0.2395 rev/s, `R² = 0.997444`
  [`survey/findings.md:61-69`].
- SPCUP19 AGH against the accepted blind readings: 7/8 usable, six agree to
  `≤ 0.27 rev/s` (mean |error| 0.148 rev/s). Take 1 reads 132.30 against
  66.33 rev/s, a factor 1.995, which the blind campaign had already flagged
  [`survey/findings.md:71-79`].
- DroneAudioSet against the paper's stated lines (arXiv:2510.15383, Fig. 7):
  three of four cells match to `≤ 1.2 %` via the blade-pass line `2f`. That
  identifies `drone1` as `D_large` (DJI F450) and `drone2` as `D_small` (DJI
  F330), and settles that the paper's marked "fundamental" is the blade-pass
  line, not the shaft rate. The `drone2` low cell disagrees by +36.7 % on 19
  recordings from 3 mic groups, so the disagreement is in the published figure
  [`survey/findings.md:88-106`].
- KAIST rotating machine, the out-of-domain control: 0/5 usable, every one
  refused by the margin rule. This is the corpus where the earlier blind
  campaign recorded a false accept [`survey/findings.md:81-86`].

Two negative results worth keeping. Four-rotor static corpora fail for a
geometric reason: `motor_allMotors_70` is the same rig and throttle that gives
7.5-8.2 dB of margin on one motor, and gives **1.41 dB** with four, because a
candidate rate between the rotors collects teeth from all of them. All 18
SPCUP static and hover windows fail the same way. And **no new corpus adds
telemetry** — every extra point is an estimate with a 0.25 rev/s tolerance and
carries no speed track, so the new points serve rig-to-rig transfer of the
noise parameters only.

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
- **Consequence for the model.** A Lorentzian speed error leaves no diffusion
  above the label band. The 1/f² speed tail integrates once to a 1/f⁴ phase
  spectrum, which is a stationary **bounded** wobble. The measured structure
  function of the 16 Hz-high-passed residual saturates at
  `S_max = 5.0e-5` to `1.4e-2 rad²`, so `Var θ_res = 2.5e-5` to
  `6.8e-3 rad²`, i.e. a standing deviation of `k × 0.005-0.082 rad`. The
  closed form `Var θ_res = 2σ_ν²λ/(3πω_c³)` with `ω_c = 2π·16 rad/s` gives
  `2.35e-4 rad²` on DREGON room 1 against the measured `4.29e-4`; trust the
  measurement.
- So the `k²`-scaling Lorentzian core that the acoustic fit needs is **label
  error**, not rotor physics: Michael's 29 Hz sample-and-hold, and DREGON
  room 2's command-against-shaft motor response. C3's DREGON
  `σ²/λ_ref = 1.603 rad²/s` is 15 times DREGON's own label-residual bound of
  0.105 rad²/s, and its `D = 844.69 rad²/s` is 8000 times that bound.
- The label-residual variant is an **upper bound** on a real label chain's
  leak, not the leak: the study's resampler is linear interpolation in both
  directions, which has a lossy passband, so part of the measured growth
  (`D_θ = S(1 s)/2 = 0.0014-0.41 rad²/s`) is the estimator's own error.
- **Acoustic negative result.** The bench and FLY125 acoustic arms identify
  nothing: 0 of 12 DREGON bench supports and 0 of 4 Michael's supports, with
  `V_θ` already above 0.15 rad² at the shortest usable lag. One harmonic
  cannot be separated from its neighbours faster than one shaft revolution, so
  the acoustic lag grid starts at the frame rate of a 6-revolution window. No
  bench evidence therefore constrains `D_k` against `k`
  [`results/noise_v2/shaft/findings_with_acoustics.md:19-20,37-42`].

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

**DREGON room-2 cruise shows no measurable comb above `k = 1`, with either
label.** The median line SNR is 1.8-4.3 dB at `k ≥ 2`, and 0-6 % of cells pass
the 10 dB gate there. The two carriers are indistinguishable to this
criterion: 6.4 dB (`motors_command`) against 7.1 dB (`rps_refined`) at
`k = 1`, half-power widths inside 0.13 Hz. **Read this as an instrument limit
as well as a data limit:** the per-rotor isolation band is only
`B = f_r/2 = 38.5 Hz`, and at `k ≥ 2` it already contains the neighbouring
rotors' lines of the same order, so this demodulation cannot isolate one
rotor's harmonic at those orders. Which DREGON orders the v2 fit should use is
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
R2. Priors come from the telemetry instead of `N(0, 2²)`, and a population
Gaussian over rigs — DREGON, Michael's, and the 12 bench points — replaces the
hand-set banks. Carrier recovery, heavy tails, order-dependent `λ_L` and
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

## Status

**Awaiting review.** Nothing is submitted. Nine decisions are open, and they
are listed with their numbers in
`docs/explainers/noise-model-v2-plan.qmd`, section "Decisions for the
reviewer": the window, the proxy, the proxy threshold, the likelihood
threshold, which DREGON orders to fit, whether `(λ_L, σ_L)` are free per label
chain, the fixed above-band wobble, the `D_k` order structure, and the DREGON
label arm.

Three prerequisites must be cleared before R1 is submitted.

1. The renderer's RMS-normalising wrapper, which invalidates absolute fitted
   levels and therefore any LTAS interpretation
   [`revised-phase-campaign.qmd:37-42,262-277`].
2. The C3 exports and the `results/S2/*.json` baselines must be tracked, or no
   remote job can score the C3 and current-best arms
   [`criteria/findings.md:401-406`].
3. The composite-risk table, which follows from (2) plus one `uni-cpu` job.
   Without it the likelihood gate has a form and an NFFT but no number.
