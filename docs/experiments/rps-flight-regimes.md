# Whole-flight regimes for the rotor-speed trajectory model

**Status:** in progress · 2026-11-20 –

Campaign code `src/experiments/rps_traj/` (`phases.py`, `transients.py`,
`switching.py`, `var_models.py`, `scores.py`, `regime_campaign.py`,
`standby_audio.py`), driver `scripts/rps_traj_regimes.py`, outputs
`results/rps_traj/regimes/`. Follows `rps-trajectory-model.md`, whose two open
questions ("Heavy tails" and "A regime layer") it answers.

## Motivation

The fitted trajectory model is a stationary Gaussian model of AIRBORNE rotor
speeds, wrapped in a hand-set scaffold (raised-cosine ramps, duration ranges
chosen by eye) to make a whole flight. Two things were known to be missing: the
transitions, where the sim-to-real gap of the transfer experiments sits, and
the rare sudden manoeuvres among the small corrections of hover, which a
Gaussian model cannot produce. The question of this campaign:

> What is the simplest extension that represents a whole flight — zero,
> start-up, standby, take-off, cruise, landing — with its transitions and its
> sharp manoeuvres, that can be fitted to full-flight telemetry in a
> straightforward way; and how does it compare, on held-out flights and with
> established metrics, against the fitted generators the literature offers?

No published fit-and-sample model of joint four-rotor speed trajectories was
found (survey in `drone-noise-generator-paper.md`). The comparable baselines
are generic multivariate ones: the plain VAR, the Markov-switching VAR
(Hamilton 1989) and the regularised VAR-HSMM of Xu and Liu (2021,
arXiv:1804.10308), the one published model that fits regime durations,
transitions and regime-specific vector autoregressions and then samples new
sequences.

## What the telemetry says

All numbers in this section are from the 100 Hz common grid of
`experiments.rps_traj.data`, on the rigs recorded ground to ground: MD2
(`michaels`, 4 flights), VID M100 (`vid_m100`, 4) and DREGON (10: five with the
measured channel, five command-only). NeuroBEM, PI-TCN and NanoBench start and
end airborne; Blackbird is one flight whose track steps from 0 to hover in
0.16 s, so there is nothing to learn from its ends.

### The phases are one fixed sequence, and nothing about them is hidden

On all 13 flights with a measured channel the six phases occur once, in one
order. An HMM over them would have a permutation for a transition matrix and
geometric dwell times, which is wrong for a start-up that lasts the same 3 s on
every MD2 flight. So the phases are read off the speed level by a fixed rule
(`phases.landmarks`) and what remains to estimate is durations, levels and the
shape of the transients.

| | MD2 (4) | VID M100 (4) | DREGON measured (5) |
|---|---|---|---|
| Start-up, motors on to standby | 3.04–3.08 s | 2.06–2.11 s | 0.09–0.20 s |
| Standby | 3.5–28 s at 34–36 rev/s (0.42–0.50 of hover) | 0.96–1.58 s at 13.2 rev/s | 1.7–4.0 s at 15.3 rev/s |
| Take-off, onset to 90 % of the step | 0.63–2.15 s | 1.32–1.63 s | 1.51–2.14 s |
| Landing, 75 % of the level left to motors off | 2.68–2.88 s | 3.23–3.71 s | 0.08 s (motor cut) |

Standby is flat: MD2's jitter there is 0.02–0.04 rev/s, about the 1 RPM step
of the log. Its per-rotor levels are NOT the cruise trim rescaled: MD2's
slowest idling rotor (30.9 rev/s) is its fastest in cruise (86.1 rev/s).

### DREGON's command channel does not show standby

On the five recordings with both channels the command reads 0 (at most
0.28 rev/s) while the rotors hold 15.30 rev/s; the rotors leave idle when the
command ramp crosses the idle speed (command crossing minus measured take-off:
−0.045 to −0.027 s). Measured speed is `max(idle, command)`. The standby of
the five command-only recordings was recovered from the audio
(`standby_audio.py`: the 20–60 Hz level, which the loudspeaker does not reach,
gives the start; the command's idle crossing gives the end), validated on the
five measured recordings: worst error 0.214 s against a 0.2 s target, on
`speech-low`, where the audio clock itself is 0.15 s off the motor stamps.
Recovered standby: 1.03–1.65 s. `phases.load_flights` puts it back.

### The transients are programs, and their spikes are three different things

`results/rps_traj/regimes/transitions.png` overlays the measured transients;
`transition_spikes.png` shows what is replayed and a held-out flight against
it. A transient was decomposed into the rotor mean and the per-rotor split,
and each part asked: does another flight of the same rig repeat it?

1. **Programmed spikes repeat exactly.** MD2's start-up runs every rotor up to
   48–51 rev/s twice (1.4 × the standby level) before settling; the four
   flights' waveforms agree to 5 % RMS of the standby level. During the spike
   the four rotors run IN UNISON at one absolute speed, and their idle split
   appears only while settling. Rescaling one template by each rotor's own
   standby level therefore misses by up to 10 rev/s; two curves do not (a
   common waveform and a "pattern hand-over" curve).
2. **At take-off the per-rotor pattern switches at arrival, not along the
   ramp.** The ramp itself is one rate (MD2 44 rev/s², VID 48, DREGON 44, as
   fitted), shared by the rotors, which keep the standby split; the cruise
   trim asserts itself where the level reaches hover. That switch is the jump
   of single rotors at the top of the ramp. It is repeatable only when aligned
   at the arrival: aligned at the take-off onset a held-out flight is no better
   described than by proportional interpolation.
3. **After touchdown the rotors fan out, by an amount that differs per
   landing.** On MD2 they spread for about two seconds and collapse with the
   spin-down (single rotors 20–40 rev/s off the proportional path). The time
   course repeats across flights; the size and direction do not (in units of
   the hover level: roll 0.11 ± 0.16, pitch −0.20 ± 0.10, yaw −0.04 ± 0.05,
   the pitch always the same sign).

What remains after these is fluctuation of cruise kind, about twice the
settled-cruise size for three seconds after arrival (rotor mean 2.1–2.9 rev/s
RMS against 0.83–1.28; per-rotor split 1.5–2.3 against 1.10–1.33).

Held-out replay error of each transient, layer fitted on the other flights,
RMS in rev/s over the transient window (`transients.replay_error`; range over
the held-out flights). "Proportional" is a single per-rotor-rescaled template,
without hand-over curve or fan-out:

| | MD2 | VID M100 | DREGON |
|---|---|---|---|
| Start-up, rotor mean | 0.69–1.15 | 0.27 | 0.23–8.31 |
| Start-up, per-rotor split: proportional → hand-over | 1.68–2.69 → 0.60–0.74 | 0.28–0.30 → 0.28–0.33 | 0.00–2.37 → 0.00–2.52 |
| Arrival, rotor mean | 2.23–2.75 | 2.10–2.62 | 0.83–3.78 |
| Arrival, split: proportional → hand-over | 1.86–2.56 → 1.48–2.29 | 2.01–3.50 → 1.55–3.11 | 1.03–3.87 → 1.00–2.66 |
| Landing, rotor mean | 3.10–4.13 | 2.55–3.97 | 2.81–4.93 |
| Landing, split: proportional → fan-out course | 6.37–13.64 → 2.05–4.29 | 2.84–4.06 → 2.00–3.37 | 1.86–4.94 → 1.27–2.36 |

Known misses: one of four MD2 take-offs pauses half-way up the ramp (FLY124,
2.15 s instead of 0.6–1.0 s), which one rate cannot produce; VID's rotors
hunt between 13 and 33 rev/s on the landing shelf, of which the replay keeps
the median; DREGON's `spinning` recording has a command burst before the
motors start, which misplaces its start-up (the 8.31).

### Cruise: take-off leaks into the frozen airborne segments, and what is left is bursty

- The frozen rule starts an MD2 segment 1.2 s after the take-off onset; in
  FLY124 that is at 62 rev/s against a settled 81. Those transients were
  3–8 % of the scored samples of the stored MD2 fit and 7–50 % of its squared
  one-step prediction error. `phases.settled_cruise` cuts them off (3 s after
  arrival, 2.5 s before landing); the cruise models here are fitted and
  scored on what is left.
- Settled cruise is still not Gaussian. Under the stored MD2 fit at 12.5 Hz,
  1.3–3.3 % of the samples exceeded the model's 99.9 % bound (expected
  0.1 %), clustered into 5–11 events a minute lasting 0.1–0.6 s (median) and
  carrying 13–41 % of the squared prediction error.
- The large events, looked at one by one, are collective kicks of
  5–10 rev/s that rise in about 0.1 s and decay in 0.2–0.5 s, and bursts of
  faster wiggling about a second long: larger inputs to the same response.
  No pulse shape of their own is needed, which is why the extension below
  scales the driving noise and nothing else.

### Two defects of the existing pipeline (not fixed here)

- `data.ground_level()` returns 8.0–8.4 rev/s for MD2: the one-second shelf
  before spin-up, because the airborne mask fires on the start-up spike. The
  real standby is 34–36 rev/s. The published `rps-traj-fits` carries
  `idle_rps` of 7.5–9.0 for `michaels`, so `fitted_traj` MD2 flights idle
  four times too slow.
- The frozen rule's one-second erosion is too short for MD2 (the leak above).

Both feed published fits and frozen statistics; changing them is a decision
for the trajectory-model campaign, and the new layer does not depend on
either.

## The model

### Whole flight

    w(t) = c(t) 1 + d(t) + g(t) M R(θ) v(t) + e(t)

`c` is the rotor mean, `d` the per-rotor split, `g` a gate on the cruise
fluctuation `v`, `e` the measurement process. The phase sequence is fixed;
per flight are drawn the plateau levels (standby per rotor: mean and spread
over the training flights; hover: `mu`, `s_c`, `s_r` as before), the time
from motors-on to take-off and the time airborne (log-normal, fitted to the
training flights), one fan-out amplitude, and the cruise process. Everything
else is replayed (`transients.PhaseLayer`, fitted by medians and least
squares over aligned windows, nothing iterative):

| Transient | rotor mean `c` | split `d` |
|---|---|---|
| Start-up | standby level × common waveform | standby pattern × hand-over curve |
| Take-off | one rate up to 90 % of the step, then the arrival curve (overshoot, settling) | standby pattern → cruise pattern by the hand-over curve, aligned at arrival |
| Landing | hover level × common curve (drop, shelf, spin-down) | cruise pattern × the same curve + fan-out course × one drawn (roll, pitch, yaw) amplitude |

The gate opens with the take-off hand-over and closes with the landing drop.
The cruise process is held at its loudest level until 3 s after arrival.

### Cruise: the noise level switches

The Gaussian model is kept and the SIZE of its driving noise switches:

    x_{t+1} = Φ x_t + κ_{s_t} w_shaft,t + w_meas,t        y_t = H x_t

`s_t` is a K-level continuous-time Markov chain with generator `G`, and
`Σ_j π_j κ_j² = 1` under its stationary law. The noise stays white with the
same mean power, so the stationary covariance, the spectrum, the
autocorrelation and the inter-rotor correlations of the Gaussian model are
unchanged; only when the energy arrives changes. `K = 1` is the Gaussian
model. New parameters: `K (K − 1)` rates and `K − 1` level ratios (3 for two
levels, 8 for three), on top of the 23 dynamics.

No HMM over the phases, and no motor lag: the phases are observed, and the
model is fitted to actual speed, whose smoothing it already contains.

### Fit

1. **Dynamics** — the campaign's exact Gaussian likelihood, unchanged
   (`model._neg_log_post`, L-BFGS-B, 4 restarts), on the settled-cruise
   segments (`switching.fit_gaussian`). The invariance above is what makes
   this stage legitimate on its own: the switch does not change what a
   Gaussian likelihood estimates.
2. **Levels** — the rates and ratios, by the likelihood of the switching
   model with the dynamics held fixed (`switching.fit_levels`). The likelihood
   is evaluated by the interacting-multiple-model filter (Blom and Bar-Shalom
   1988): K Kalman filters that differ only in process noise, merged at every
   step with the chain's transition matrix; Kim (1994) is the reference for
   maximum-likelihood estimation of this class with a collapsed filter. With
   equal levels the filter is the Kalman filter to 1e-14.

## Protocol

**Data.** Every flight of every rig (`phases.load_flights`), per-drone
parameter sets. Cruise: the settled-cruise segments, cut into blocks of at
most 30 s, each demeaned per rotor (the level is a per-flight offset every arm
draws from the same law); the first second of every block conditions and is
not scored (`cruise_api`).

**Rate.** MD2 12.5 Hz, DREGON 25 Hz, every other rig 50 Hz, reached from the
100 Hz grid by block means. Two reasons, both measured:

- MD2 is logged at 29.41 Hz with holds; at 25 Hz its squared prediction
  errors alternate sample to sample and a noise-level switch learns that as
  0.14 s "bursts" (0.57 s at 12.5 Hz).
- The one-step log score rewards a model for learning the PREPROCESSING. The
  trajectory campaign's decimation (zero-phase Butterworth at 0.4 × rate)
  leaves a spectral dip below the new Nyquist frequency that a free-form
  autoregression learns and a continuous-time model cannot. On MD2 (FLY124
  held out):

  | Decimation to 12.5 Hz | PSD near Nyquist vs mid-band | VAR − Gaussian structured model, nats/sample |
  |---|---|---|
  | Butterworth at 0.4 × rate | −17.3 dB | +0.466 |
  | Plain subsampling | −3.9 dB | +0.084 |
  | Block mean | −6.8 dB | +0.067 |

  The common grid itself carries that dip at 40–50 Hz wherever the telemetry
  is faster than 100 Hz, hence 50 Hz for the fast rigs, where it folds away.

**Held out.** Whole flights: leave-one-flight-out up to five flights,
otherwise five folds (DREGON: one measured and one command-only recording per
fold). Blackbird has one flight; its settled cruise is split in two halves in
time, which is the only split there is.

**Cruise arms** (`regime_campaign`):

| Arm | What | Free parameters |
|---|---|---|
| `gauss` | structured model, one noise level (the current model, refitted on settled cruise) | 23 |
| `switch2`, `switch3` | structured model, two / three noise levels | 26 / 31 |
| `var` | plain Gaussian VAR(p), p by BIC up to one second of lags (24 at most) | 16 p + 10 |
| `msvar2`, `msvar3` | Markov-switching VAR, 2 / 3 states, order by BIC among 1, 2, 4 | 26 M + 16 p M + M² − 1 … |
| `hsmm2`, `hsmm3` | regularised VAR-HSMM of Xu and Liu: nonparametric dwell law up to 5 s, LASSO on the AR matrices, covariance shrinkage, both chosen by one-step MSFE on every fifth training block | paper's count |

Adaptations of the VAR-HSMM, all in its favour or neutral: the order is the
BIC order of the Markov-switching VAR with the same number of states (the
paper compares orders 1 and 2 on 303 days of 50 stocks; here four channels
meet 10⁴–10⁵ samples); the LASSO weight is per training row because the
paper's objective is an unnormalised sum of squares (its 10⁻⁴–1 grid on its
own data is 10⁻⁸–10⁻⁴ per squared residual), and 0 is added to both grids;
the E-step is the elapsed-time forward-backward, O(T M D) instead of the
paper's O(T M² D²), same likelihood (checked against brute-force enumeration
in the tests).

**Cruise scores** (all on held-out blocks; `scores.py`):

- the one-step log predictive density (the logarithmic score), nats/sample;
- the energy score and the CRPS (Gneiting and Raftery 2007) of 100 sampled
  forecast paths at 0.2, 1 and 2 s, origins every 2 s, reported as skill over
  the block's own climatology;
- distances between generated and held-out blocks: Wasserstein-1 of the
  0.2 s and 1 s increments, the excess-kurtosis error of those increments,
  autocorrelation RMSE to 5 s, inter-rotor correlation error, variance ratio,
  log-spectral distance of the four mixer modes, and the squared maximum mean
  discrepancy of 2 s windows (Gretton et al. 2012).

**Whole-flight arms** (MD2, VID M100, DREGON; 64 generated flights an arm and
fold, compared at 25 Hz):

| Arm | What |
|---|---|
| `ours` | the fitted phase layer around the two-level cruise model |
| `scaffold` | the wrapper in use today (`wrap_airborne`: raised-cosine ramps, hand-set duration ranges, the stored idle rule) around the same cruise model |
| `msvar`, `hsmm` | the switching VARs fitted to whole flights: one state per phase, order 2, EM started at the phase segmentation of the training flights, every state kept stable, sampler clamped |

EM is started at the phases because that is the most favourable start for a
generator of whole flights: from a variance-based labelling, likelihood
prefers states that flicker (dwell 0.04–2 s on MD2) and 2–8 % of the draws
were one valid flight. A free-running sample is cut after its first return to
rest.

Nothing in a switching VAR keeps a state's autoregression stable, and a state
fitted to a ramp is not. Two guards were added, both in the baselines'
favour: every state is kept stable during EM (the guard the plain VAR has:
coefficients shrunk until the companion spectral radius is under 0.999,
intercept and covariance re-estimated), and the sampler is clamped, inside
the recursion, to between zero and 1.25 × the fastest training sample — a
rotor has a top speed. The share of generated flights that touch that cap is
reported (`saturated_rate`): a stable state with a drift still climbs for as
long as its stay lasts. What the guards change, fold 0 of each rig, share of
64 draws that are one valid flight / that leave the range (unclamped) or
touch the cap (clamped):

| Treatment | MD2 MS-VAR | MD2 VAR-HSMM | DREGON MS-VAR | DREGON VAR-HSMM | VID MS-VAR |
|---|---|---|---|---|---|
| as published | 0.84 / 0.00 | 0.03 / 0.91 | 0.00 / 1.00 | – | 0.55 / 0.19 |
| clamped | 0.84 / 0.20 | 0.08 / 0.95 | 0.00 / 1.00 | 0.00 / 1.00 | 0.64 / 0.28 |
| stable states, clamped (used) | 0.72 / 0.00 | 0.03 / 0.59 | 0.31 / 0.36 | 0.31 / 0.41 | 0.53 / 0.09 |

**Whole-flight scores**: the share of generated flights that are one valid
pass through the six phases; the CRPS of each whole-flight read-out
(`phases.readouts`: start-up time and peak, standby time, level and spread,
take-off rise and overshoot, time airborne, landing fall time and rotor
spread, time between 55 and 80 % of hover), the valid generated flights being
the predictive ensemble and the held-out flight the observation; the share
that touched the cap; Wasserstein-1 of the speed and increment distributions
relative to hover, and the 99.9th percentile of the 0.2 s increments against
the held-out one.

## Results

209 units, none failed (2026-11-21, seven `uni-cpu` jobs at `befb8591`). Every
number is a mean over the held-out folds of a rig; full tables for all 14
cruise scores and all whole-flight read-outs: `results/rps_traj/regimes/tables.md`
(`scripts/rps_traj_regimes.py table`), figure `cruise_scores.png`, per-drone
parameter sets `params/<rig>.json`.

### Cruise

One-step log predictive density, nats/sample (higher is better):

| Arm | Parameters | MD2 | DREGON | VID M100 | Blackbird | Nanobench | NeuroBEM | PITCN | Mean rank |
|---|---|---|---|---|---|---|---|---|---|
| `gauss` | 23 | −4.543 | −4.555 | −6.626 | −6.234 | −10.422 | −9.800 | −9.455 | 7.71 |
| `switch2` | 26 | −3.853 | −4.143 | −4.604 | −5.771 | −9.508 | −8.819 | −8.720 | 5.57 |
| `switch3` | 31 | −3.744 | −4.096 | −4.253 | −5.662 | −9.383 | −8.657 | −8.624 | 3.86 |
| `var` | 74–378 | −4.608 | −4.523 | −7.076 | −6.024 | −9.701 | −9.507 | −9.348 | 7.29 |
| `msvar2` | 159 | −3.854 | −4.177 | −3.958 | −5.560 | −9.216 | −8.079 | −8.693 | 4.57 |
| `msvar3` | 146–242 | −3.768 | −3.802 | −3.411 | −5.534 | −9.042 | −7.871 | −8.546 | 1.43 |
| `hsmm2` | 279–655 | −3.850 | −4.206 | −3.853 | −5.554 | −9.194 | −8.048 | −8.664 | 3.57 |
| `hsmm3` | 326–986 | −3.772 | −3.789 | −3.387 | −5.558 | −9.047 | −7.875 | −8.562 | 2.00 |

Mean rank over the seven rigs on the other scores (1 = best of eight):

| Score | `gauss` | `switch2` | `switch3` | `var` | `msvar2` | `msvar3` | `hsmm2` | `hsmm3` |
|---|---|---|---|---|---|---|---|---|
| Energy-score skill, 0.2 s | 6.71 | 4.57 | 3.57 | 6.43 | 4.14 | 3.57 | 4.43 | 2.57 |
| Energy-score skill, 1 s | 5.29 | 4.14 | 3.43 | 4.71 | 6.00 | 3.86 | 4.57 | 4.00 |
| Energy-score skill, 2 s | 4.71 | 2.29 | 2.43 | 3.14 | 6.71 | 5.71 | 6.43 | 4.57 |
| Increment W1, 0.2 s | 7.29 | 5.43 | 4.29 | 5.86 | 4.57 | 2.29 | 3.43 | 2.86 |
| Increment W1, 1 s | 5.29 | 3.43 | 2.57 | 6.00 | 6.43 | 3.86 | 4.57 | 3.86 |
| Kurtosis error, 0.2 s | 6.00 | 5.14 | 3.43 | 5.86 | 3.86 | 2.86 | 4.29 | 4.57 |
| Autocorrelation RMSE | 5.00 | 4.29 | 4.86 | 2.86 | 5.00 | 4.14 | 4.00 | 5.86 |
| Log-spectral distance | 4.86 | 5.14 | 5.29 | 2.71 | 4.57 | 4.14 | 4.43 | 4.86 |
| MMD² of 2 s windows | 4.14 | 3.57 | 3.14 | 4.14 | 5.57 | 5.00 | 5.71 | 4.71 |

What the tables say:

- **The switch is the finding, in every family.** The two arms with one noise
  level (`gauss`, `var`) are the two worst on the log score on every rig; a
  second level is worth 0.4–2.0 nats/sample to the structured model and a
  third 0.05–0.35 more. The three-level fit sits at a bound of its level
  ratios on six rigs (not DREGON), so its third level is not identified.
- **On the one-step score the three-state switching VARs are ahead of the
  structured model**: `msvar3` beats `switch3` on six rigs (not MD2, where the
  two are 0.02 apart), by 0.08–0.84 nats/sample, with 146–986 free parameters
  against 31. The explicit dwell law of the VAR-HSMM buys nothing over the
  Markov chain (`hsmm3` vs `msvar3`: within 0.03 on every rig).
- **VID M100 is where the structured model fails**: −4.25 against −3.39, and
  0.2 s energy-score skill 0.18 against 0.37. Cause not established here.
- **Further ahead the order reverses.** At 2 s the structured arms have the
  best mean rank (2.29, 2.43) and the switching VARs the worst (4.57–6.71);
  the same holds for the 1 s increment law (2.57 against 3.86–6.43) and the
  2 s window MMD. At 1 s only NeuroBEM (0.21–0.26) and DREGON (0.06–0.09) are
  clearly above climatology, in every arm.
- **Second-order structure favours the plain VAR** (autocorrelation, spectra):
  it has the free lags for it, and the structured model's spectra are off on
  NeuroBEM (3.2 against 1.7 dB) and Blackbird (3.6 against 2.7 dB).

### Whole flights

Share of 64 generated flights that are one valid pass through the six phases,
and the share that touch the speed cap:

| Rig | `ours` | `scaffold` | `msvar` | `hsmm` |
|---|---|---|---|---|
| MD2 | 1.00 / 0.00 | 1.00 / 0.00 | 0.39 / 0.45 | 0.54 / 0.40 |
| VID M100 | 1.00 / 0.00 | 1.00 / 0.00 | 0.58 / 0.09 | 0.70 / 0.04 |
| DREGON | 1.00 / 0.00 | 1.00 / 0.00 | 0.24 / 0.40 | 0.20 / 0.38 |

CRPS of the read-outs of the valid flights (lower is better), MD2 / VID M100
/ DREGON:

| Read-out | `ours` | `scaffold` | `msvar` | `hsmm` |
|---|---|---|---|---|
| Start-up time, s | 0.01 / 0.00 / 0.10 | 2.25 / 1.15 / 0.50 | 0.70 / 4.68 / 1.23 | 2.78 / 1.42 / 0.54 |
| Start-up peak / hover | 0.01 / 0.01 / 0.09 | 0.51 / 0.01 / 0.10 | 0.35 / 0.16 / 1.04 | 0.21 / 0.27 / 0.43 |
| Standby level / hover | 0.03 / 0.00 / 0.00 | 0.35 / 0.01 / 0.00 | 0.08 / 0.08 / 0.00 | 0.06 / 0.06 / 0.00 |
| Standby time, s | 7.36 / 0.20 / 0.53 | 5.60 / 9.33 / 8.64 | 12.33 / 0.31 / 0.55 | 11.17 / 0.31 / 0.48 |
| Take-off rise, s | 0.45 / 0.10 / 0.34 | 0.66 / 0.36 / 0.35 | 1.24 / 4.99 / 0.48 | 1.34 / 3.04 / 0.35 |
| Landing fall, s | 0.10 / 0.16 / 0.01 | 0.37 / 0.59 / 2.15 | 1.97 / 0.78 / 0.03 | 1.51 / 1.27 / 0.02 |
| Landing rotor spread / hover | 0.11 / 0.06 / 0.03 | 0.48 / 0.18 / 0.09 | 0.48 / 0.13 / 0.03 | 0.36 / 0.08 / 0.05 |
| Time at 55–80 % of hover, s | 0.34 / 0.09 / 0.14 | 0.24 / 0.31 / 0.56 | 6.87 / 8.59 / 0.84 | 9.30 / 3.84 / 0.35 |
| Time airborne, s | 28.4 / 33.3 / 4.3 | 23.0 / 35.7 / 15.0 | 31.1 / 39.2 / 8.1 | 53.6 / 35.5 / 10.9 |
| 99.9th pct of 0.2 s increments, generated / real | 1.06 / 0.97 / 1.09 | 0.43 / 0.44 / 0.17 | 1.64 / 0.94 / 1.13 | 1.44 / 0.87 / 1.12 |

- The phase layer produces a flight every time; the switching VARs do in
  20–70 % of draws even started at the phases, kept stable and clamped, and
  the CRPS of the time their valid flights spend between the ground and hover
  is 0.35–9.3 s against 0.09–0.34 s.
- Against the scaffold in use today the layer is better or level on every
  read-out except three, all on MD2 (standby time, time at 55–80 % of hover,
  time airborne), and it restores the extremes of the increments the scaffold
  removes (ratio 0.97–1.09 against 0.17–0.44).
- Not solved by anyone: how long a flight lasts and, on MD2, how long it idles
  (CRPS 7.4 s).

### Dynamics inside each phase, on a given schedule

When to change phase and how long to stay is the user's choice, so here it is
given to every arm (`phase_dynamics.py`, run locally, 39 units): the held-out
flight's motors-on, take-off and landing times. The baseline is then one
VAR(p) per phase fitted to the training samples of that phase (with known
states the Markov-switching VAR and the VAR-HSMM are this one model), simulated
along the held-out flight's own phase labels — which also tell it when each
transient ends; ours is not told that. 64 generated flights an arm; energy
score of the four rotor speeds (`path`) and of their 0.2 s increments (`inc`),
percent of hover, lower is better; `sd` is the standard deviation of the
generated / held-out 0.2 s increments.

| Window | Score | MD2: ours / VAR(2) / VAR(4) | VID M100 | DREGON |
|---|---|---|---|---|
| Start-up, 3 s | path | 3.95 / 15.58 / 14.88 | 0.53 / 4.36 / 1.66 | 1.96 / 7.47 / 9.47 |
| | inc | 3.90 / 13.13 / 12.98 | 0.63 / 0.95 / 0.69 | 1.79 / 1.46 / 1.69 |
| Standby, last 2 s | path | 6.08 / 9.53 / 9.60 | 0.64 / 1.13 / 0.67 | 0.30 / 6.74 / 9.03 |
| | inc | 0.31 / 0.67 / 0.78 | 0.91 / 1.06 / 0.75 | 0.35 / 0.36 / 0.38 |
| | sd, generated (real) | 0.01 / 0.96 / 1.31 (0.52) | 0.36 / 0.96 / 0.75 (0.58) | 0.43 / 0.49 / 0.53 (0.28) |
| Take-off, 4 s | path | 13.81 / 18.83 / 18.45 | 11.34 / 10.25 / 9.50 | 11.52 / 11.49 / 11.93 |
| | inc | 6.04 / 5.07 / 5.07 | 4.85 / 4.57 / 4.23 | 6.83 / 5.42 / 5.37 |
| | sd, generated (real) | 6.02 / 5.95 / 6.06 (5.51) | 7.27 / 6.33 / 5.93 (5.78) | 6.43 / 6.07 / 5.78 (6.23) |
| Landing, −0.5…4 s | path | 16.55 / 17.44 / 16.83 | 8.90 / 11.42 / 11.67 | 7.64 / 24.37 / 20.88 |
| | inc | 6.67 / 6.92 / 6.88 | 5.80 / 8.64 / 8.71 | 8.75 / 40.27 / 37.52 |

Held-out flights: 4, 4 and 10 (5 with a landing); means only, no spread.

- **Take-off is where the per-phase VAR has the better dynamics**: lower
  increment score on all three rigs (by 13–21 %), and on VID M100 the better
  path too. Ours ramps at one fitted rate and, on VID M100, moves too much
  (7.3 against 5.8).
- **Standby: ours has no fluctuation at all** (MD2: 0.01 against 0.52 real);
  it still scores better there on MD2 and DREGON because the VAR is twice too
  rough, but VAR(4) is ahead on VID M100.
- **Start-up and landing: ours**, on path everywhere and on increments except
  DREGON's start-up (1.79 against 1.46).

### Multiscale path score: sensitivity control (failed)

Can a single score detect visible faults in a free-running four-rotor path?
Before ranking generators, score an ensemble of *other real flights* against
each held-out recording, then corrupt just the ensemble and score the same
observation. Same folds as above: 4 MD2, 4 VID M100, 10 DREGON recordings.
No self-match; no dynamic time warping. The gate is applied in the affected
phase, **before** averaging across phases: dilution by a long cruise is not
the explanation for a failure.

At 25 Hz transform the four speeds into collective, roll, pitch and yaw modes.
Score (i) the ordered raw window, (ii) its four means and (iii) the sequences
of differences between adjacent box means with widths 2, 4, 8, … samples.
Each component uses the fair ensemble energy score on the *entire vector*,
divided by the square root of its dimension. Normalize each mode using the
clean training ensemble (SD for raw and mean, RMS for differences, floor
0.5% of training median hover); reuse those scales for the corrupted ensemble.
Give one vote to each component. This is an application of [Pic et al.'s
transformation and aggregation rule](https://doi.org/10.5194/ascmo-11-23-2025),
not a claim that these particular weights have been validated.

Use the aligned phase windows above, cropping to common recorded support.
In settled cruise, use windows up to 40.96 s, one path per training donor
and eight origin replicates averaged **within** each held-out flight. The
control magnitudes are relative to the donor's hover: independent 2% RMS
Gaussian jitter; a common linear drift from −8% to +8% (random sign);
reversal of rotor 1 alone within the cruise window (same individual-rotor
histogram and power spectrum); and, on MD2 start-up, subtract any excess
of the rotor mean above the recorded standby plateau.

Gate fixed before running: in a majority of applicable rigs the median
paired score penalty must exceed the median absolute deviation (MAD) of
real-versus-real flight scores, and a majority of flights must worsen.
For jitter and drift also require a positive, larger penalty at the fine
and coarse end respectively. MAD is a practical effect-size threshold,
not a significance test. Each cell below is the median paired penalty;
positive means the distortion scores worse.

| Distortion | MD2 | VID M100 | DREGON | Flights worse | Rigs passing |
|---|---:|---:|---:|---:|---:|
| Cruise drift | +0.431 | +0.332 | +0.447 | 18/18 | 3/3 |
| Cruise jitter | +0.018 | +0.037 | +0.028 | 15/18 | 0/3 |
| Reverse one rotor | −0.003 | +0.017 | +0.001 | 10/18 | 0/3 |
| Flatten start-up spike | +0.119 | — | — | 3/4 | 0/1 |

The clean cruise MADs were 0.075, 0.097 and 0.113, respectively. Jitter
was visible at 80 ms (penalties 0.159, 0.234, 0.150) but washed out in the
average. Drift concentrated at the coarsest widths (20.48 s on MD2/VID,
10.24 s on DREGON). Removing the spike was smaller than MD2's clean start-up
MAD (0.300); on FLY108 it even improved the score by 0.079. Reversing one
rotor was scarcely distinguished despite preserving each rotor's univariate
spectrum. This establishes sensitivity to the chosen drift, **not** to
overall flight realism. No generator ranking or weight tuning followed.
Per-flight results and octave profiles: `results/rps_traj/regimes/multiscale.json`.

### Can a joint spectral distance recognise a real rig in cruise? (Spectra-only baseline)

Compare four motor speeds **jointly**, on settled cruise only. Estimate the
complex 4×4 cross-spectrum on 16 s windows at 25 Hz (demean each motor, no
time taper, smooth the outer products over three Fourier bins). Score four
octaves, [0.25, 0.5), [0.5, 1), [1, 2), [2, 4) Hz. The upper limit avoids
the documented DREGON measurement artifact; this experiment says nothing
about variation slower than 4 s, the hover mean, or transition shapes.
Within each octave divide each matrix entry `(i,j)` by the geometric mean
of the two calibration auto-powers. Calibration uses one whole prespecified
flight from each multi-recording rig, then the median across rigs. These
scales are shared: a quiet flight is not rescaled to look like a noisy one.

The distance is the root-mean-square complex Frobenius difference, averaged
first over frequency bins and then equally over octaves. Power and coupling
are its diagonal and off-diagonal contributions. This is a descriptive use
of [spectral-matrix L2 distance](https://doi.org/10.3150/13-BEJ584), not the
paper's hypothesis test. Mean each flight's features before forming a rig
centroid: one physical flight, one vote. Calibration flights never enter a
centroid or serve as queries. Blackbird is a candidate for other queries,
not a cross-flight identity test of itself.

For each distortion compare the augmented and original window against the
**same** clean mean of other evaluation flights from its rig. Take the
median paired penalty within physical flight, then across flights. A rig
passes if the median exceeds the clean between-flight MAD and a majority
of flights get worse; for a band-specific fault, its intended octave must
also have the largest penalty. Require a majority of eligible rigs, not a
majority of pooled windows. Shared Fourier-phase randomization must instead
preserve the complete matrix to relative error below `1e-10`; independent
phases and a circular delay of motor 1 must preserve all four auto-spectra
to the same tolerance. These [Fourier controls](https://doi.org/10.1103/PhysRevLett.73.951)
are stationary controls, not simulated flight maneuvers.

**Coverage.** 137 measured physical recordings yielded 280 non-overlapping
windows (maximum four per flight); six calibration recordings took 18
windows. The map uses the other 131 recordings / 262 windows. The independent
test has 130 recordings: Blackbird is a single flight. NeuroBEM pieces are
grouped by the original recording, not treated as flights. DREGON is room 1
only: the five command-only recordings are not motor-speed measurements.

| Augmentation on real cruise | Flights farther from clean reference | Rigs passing |
|---|---:|---:|
| Same random Fourier phase for all rotors | invariant, max relative error 3.7e−14 | 7/7 (invariance) |
| Independent Fourier phases, coupling score | 99/130 | 1/6 (NeuroBEM) |
| Delay one rotor 80 ms, coupling score | 100/130 | 0/6 |
| Common 0.3125 Hz sinusoid, 4% of hover peak, low-octave power | 73/130 | 3/6 |
| Independent 2–4 Hz jitter, 2% of hover RMS, high-octave power | 127/130 | 5/6 |

**Real versus real.** For a query window, compare mean distance to the
windows of another recording of its rig against mean distance to the windows
of a recording from another rig. The win probability treats ties as half
and weights other rigs equally; aggregate queries by flight, then rig.
Also identify the held-out *flight mean* by nearest rig centroid. Require
win probability above 0.5 **and** a majority of flights identified.

| Rig | Same-rig win probability | Held-out flights correctly identified |
|---|---:|---:|
| DREGON | 0.991 | 4/4 |
| MD2 | 0.808 | 2/3 |
| NanoBench | 0.478 | 7/10 |
| NeuroBEM | 0.508 | 16/49 |
| PI-TCN | 0.524 | 26/61 |
| VID M100 | 0.799 | 0/3 |
| Blackbird | — | — |

Only DREGON and MD2 pass both criteria. The rig-balanced win probability
is 0.685 (0.542 weighting flights equally); flight identification is 52.0%
rig-balanced, 55/130 pooled. The distinction between *same flight* and
*same rig* matters: PI-TCN's median distance to other windows of the same
recording is 19.5, to windows of another flight of the same rig 185.2,
and to other rigs 160.8 (query-window medians of mean distances).

**Map and verdict.** All 21 exact centroid distances are in
`results/rps_traj/regimes/spectral_only/rig_centroid_distance.csv`; all segment pairs in
`segment_distance.npz`; per-flight evidence, dataset pins and calibration
scales in `spectral_validation.json`. `rig_centroid_mds.png` is metric MDS
(SMACOF, classical initialization): normalized stress 0.0924, Pearson
0.984, Spearman 0.977, nearest neighbour preserved for 5/7. DREGON–MD2
is the closest pair (2.24); Blackbird–NeuroBEM the farthest (290.29).
These are distances between **mean spectra**, not evidence of seven tight
clusters. The score detects fast excess power, but it is **not validated
as a standalone coupling or rig-identity score**. Neither the qualitative
appearance of the map nor the direction of the augmentation penalties
overrides the cross-flight failures. No generator ranking.

### Operating speed plus joint spectral dynamics (current distance)

For each scored 16 s window, take the four motor means **before** demeaning.
The level distance `L` is the RMS difference between the four means in
rev/s: this sees both a common change in operating speed and a change in
motor trim with no change in their overall average. `S` is the unchanged
complex cross-spectral distance above. Use the dimensionless distance
`sqrt((L / 106.0274)² + (S / 23.4657)²)`. Each denominator is the median of
that block's 15 pairwise distances between the six reserved calibration
flight means. Neither the scales nor the combination were fitted to the
held-out identification results. This gives the blocks equal weight at
their respective calibration scales, not equal influence on every pair.

Offset controls on all 262 evaluation windows: adding 30 rev/s to every motor
gave distance 0.28295; adding it to one gave 0.14147. Alternating +30 and
−30 across motors gave 0.28295 despite keeping the overall mean fixed.
In all three controls the spectral component changed by less than 3e−14.
The original fluctuation controls are reported separately in `spectral_controls`;
these offsets test level sensitivity, not realism or cross-motor timing.

On the same held-out flights, rig-balanced same-rig win probability went
from 0.685 to 0.705; nearest-centroid identification stayed at 55/130
(52.0% rig-balanced). `rig_centroid_distance.csv` and `segment_distance.npz`
in the campaign output root now use the combined distance. The descriptive
all-flight map uses all 137 recordings, including calibration: normalized
stress 0.097, distance rank correlation 0.982. Its coordinate CSV also
contains each flight's four mean cruise speeds in rev/s.

`flight_mds.html` shows individual recordings. Click a flight dot in any
panel to inspect all four measured rotor-speed traces; shading marks the
exact scored cruise windows. The flight selector also reaches overlapping
dots. NeuroBEM recording parts are shown separately on their own clocks.

### Held-out comparison on the commanded phase schedule

**Decision rule fixed before running the five generators.** Test the visual
claim from `generated_flights.html`, not rig identification. Use the 13
measured targets (4 MD2, 4 VID M100, 5 DREGON); the campaign's cached
generator folds exclude the target but can include DREGON command-only
training recordings. Score every illustrated arm and a separate reference
ensemble of other measured flights. No best-of-N draw selection, generated
event re-detection, time warping, or score for the duration of a phase.

* Use the fair ensemble energy score (Euclidean distance to the observation
  minus half the mean distance between distinct draws). During start-up,
  standby, take-off and landing use the absolute four-rotor path on the
  event clock (RMS path distance) and a vector of three readouts in rev/s:
  peak collective speed, maximum instantaneous rotor range and RMS 0.2 s
  increment. Report the two scores individually and average after scaling.
* In settled cruise a fresh draw need not reproduce the time of a specific
  manoeuvre. Score the distribution of `CruiseMetric`'s four mean speeds
  plus complex cross-spectrum, rather than the sample clock or the 2D map.
  Fit its shared block scales on reserved calibration recordings of the
  **other rigs**, so no held-out target sets its own weights. Retain separate
  mean-speed and spectral scores as diagnostics.
* For each feature block divide by the median pairwise feature distance
  between measured training donors at the same event/origin, with a floor
  of 1% of training hover (divide by the cruise level scale for the joint
  feature). This expresses an error relative to between-flight variability
  without dividing by a potentially zero real-reference score. It is a
  design choice, not a physically unique exchange rate between phases.
* Weight the two short-phase readouts equally, cruise origins equally
  within a flight, the five phases equally within a flight, flights equally
  within a rig and the three rigs equally. Report each phase regardless of
  the ordering. For each baseline report the paired mean difference,
  per-flight wins and a descriptive 95% percentile interval (10,000
  within-rig paired flight resamples; fixed folds, fits and draws).
* Before model scoring, compare clean training donors and augmented copies
  against the same independent real targets. A +10% training-hover offset
  tests level sensitivity; flatten each short path to its per-rotor mean
  or collapse it to its instantaneous collective to test dynamics/spread;
  shared/independent Fourier phases and fast jitter test cruise. A shared
  random Fourier phase must leave the cruise score unchanged to numerical precision.
  For each positive-control phase report the flight-paired penalty against
  normal clean-score variation (per-rig median > clean MAD and majority of
  flights worse). Report model scores even for insensitive endpoints, but
  do not interpret those endpoints as evidence of fidelity to that fault.
  No change to weights, arms or acceptance after viewing scores.

**Results (2026-11-22).** 13/13 measured targets yielded all five phase
scores for every arm (325 model rows). No generated window required
post-landing padding; no scored target or real-donor window contained a
bridged sample. Only the five command-only targets were excluded. The
cruise metric used five foreign-rig calibration flights for each target
rig. Training-fold held-out identifiers, block counts and durations matched
the 39 required cached cruise fits.

| Donor distortion: rigs passing out of 3 | Start-up | Standby | Take-off | Cruise | Landing |
|---|---:|---:|---:|---:|---:|
| Common offset, +10% of training hover | 3 | 3 | 3 | 1 | 3 |
| Flatten each motor over time | 3 | 2 | 3 | — | 3 |
| Collapse all motors to the collective | 3 | 3 | 3 | — | 3 |
| Independent Fourier phases | — | — | — | 0 | — |
| Independent 2–4 Hz jitter, 2% hover RMS | — | — | — | 3 | — |

The shared-phase negative preserved the cruise score to within 3.6e−15.
Independent phases worsened 10/13 flights and the cruise DC shift worsened
12/13, but neither consistently exceeded normal flight-to-flight variation.

| Generator | Start-up | Standby | Take-off | Cruise | Landing | Balanced | Baseline − ours [95% paired interval] | Ours wins |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| Proposed | 2.04 | 0.67 | 0.76 | 2.15 | 0.69 | 1.26 | — | — |
| Phase VAR(2) | 1.89 | 1.25 | 0.89 | 2.52 | 1.32 | 1.57 | 0.31 [0.12, 0.61] | 11/13 |
| Phase VAR(4) | 1.55 | 1.72 | 0.82 | 2.39 | 1.25 | 1.55 | 0.28 [0.21, 0.36] | 10/13 |
| VAR(2) + MS-VAR(3 states) | 1.89 | 1.25 | 0.88 | 2.11 | 1.32 | 1.49 | 0.23 [0.03, 0.54] | 7/13 |
| VAR(2) + VAR-HSMM(3 states) | 1.89 | 1.25 | 0.88 | 2.11 | 1.32 | 1.49 | 0.23 [0.03, 0.53] | 6/13 |
| Other real flights (reference only) | 0.37 | 0.38 | 0.50 | 1.95 | 0.51 | 0.74 | — | — |

The proposed model reduced the equal-phase, equal-rig mean by 15.2–19.7%
relative to these four fitted generators. Its strongest aggregate gains were
standby and landing. Its ordering against each baseline survived leaving any
one phase out. The mean is not the median flight: against the HSMM hybrid
the proposed model won all four MD2 flights, one DREGON flight, and only the
outdoor yaw sequence on VID (paired improvement 1.74 on that sequence).
The hybrids were better in cruise. Phase VAR(4) had a lower start-up score:
on DREGON the proposed model actually had the better ordered path, but
the worse joint peak/spread/increment score. Real donors beat every fitted
generator on the aggregate; they are a descriptive reference, not a lower
bound. Do not extrapolate this result to one-step forecasts, arbitrary
maneuvers, new rigs, autonomous transitions, or the acoustic comparison.

Evidence in `results/rps_traj/regimes/schedule_eval/`: every paired score
and its supports in `schedule_{controls,models}.json`, aggregation in
`summary.json`, print-ready table and figure in `schedule-table.tex` and
`schedule-example.pdf`. The figure uses measured FLY124 and the same 32
draws per displayed arm. Its first column runs continuously from motors-on
to takeoff; cruise uses the middle scored 16 s window. These are display
choices, not changes to any phase score.

**Why is the VAR-HSMM hybrid ahead in settled cruise?** Replaying just
these two arms with the saved fold caches and seeds reproduces each of
the 78 published arm/origin scores. The hybrid wins 10/13 flights; its
equal-rig advantage is 0.0414 (descriptive 95% paired flight bootstrap
0.0015–0.0854, fixed draws and rigs). The same score is mean distance
to the observation minus half the mean distance between distinct draws:

| Cruise score term (equal-rig means) | Proposed | VAR-HSMM hybrid |
|---|---:|---:|
| Draw-to-measurement feature distance | 2.583 | 2.629 |
| Ensemble-spread term, subtracted | 0.435 | 0.524 |
| Fair ensemble score | 2.147 | 2.106 |

Thus individual hybrid draws are **not** closer on average. Its wider
distribution of spectral/level features more than offsets that penalty.
The separate four-motor mean-speed score favours the proposed model on
all 13 flights. An exact allocation of each Euclidean norm to blocks
(`||difference_block||² / ||difference_full||`, zero for coincident
points), applied to both energy-score terms, gives the following
contributions to proposed minus hybrid: level −0.0080; spectral octaves
in ascending order +0.0003, +0.0031, +0.0225, +0.0236. Auto-spectral
entries sum to +0.0205 and cross-spectral entries to +0.0290. The latter
encode power as well as relative phase, not isolated evidence about lag.

The rig-level gaps are +0.0017 (MD2), +0.0469 (DREGON), +0.0757 (VID).
On the middle window of VID `outdoor_8_yaw`, 2–4 Hz RMS speed fluctuation
(square root of mean per-motor band power, median over draws for models)
is 2.17 rev/s measured, 0.27 proposed, 0.34 hybrid. Both miss the
excursions. For `indoor_loaded_8` it is 0.53, 1.31 and 1.36 respectively:
both overshoot, and the proposed model wins the flight. On the middle
DREGON `free-flight_speech-high_room1` window the values are 0.43, 0.53
and 0.42. A three-state Markov hybrid scores 2.110, close to the HSMM's
2.106; the comparison does not identify an explicit state-duration benefit.
No generator, score or weighting rule was changed. Reproduce using
`PYTHONPATH=src python -m experiments.rps_traj.cruise_diagnosis`
under the single-thread limits below. `cruise_sections.html` in
`results/rps_traj/regimes/schedule_eval/` shows every scored origin;
`cruise_diagnosis.json` contains all component scores and band powers.

**Unified successor experiment:** [one whole-flight likelihood, persistent targets, and an ordered-action hierarchy](unified-flight-response.md) compares the proposed replacement and its matched flat ablation against the frozen scores above. It uses a separate output directory; the template and its published results are unchanged.

### Does the fit run faster on a GPU?

No. One objective-and-gradient evaluation of the Gaussian stage, 16 CPU
threads against an A100 (`bench_device.json`):

| Rig | Scored samples | CPU, s | GPU, s |
|---|---|---|---|
| MD2 | 4 653 | 0.060 | 0.136 |
| PITCN | 166 062 | 0.403 | 0.780 |

The recursion is a time loop of 16-state matrix products; the level stage is
numba on the CPU. The whole Gaussian stage took 138–827 s a fold.

## Conclusion

- **Whole flights: adopt the phase layer.** It is the only arm that always
  generates a flight, it beats the current scaffold on the transients, and a
  switching VAR is not a whole-flight generator on this data. With the
  schedule given it still has the better start-up and landing; its take-off
  dynamics are worse than a per-phase VAR's and its standby does not move.
- **Cruise: adopt the two-level switch, and do not claim more than it shows.**
  It removes most of the Gaussian model's one-step deficit with three
  parameters and is the best arm 2 s ahead, but a three-state Markov-switching
  VAR predicts the next sample better on six rigs of seven with 5–30 times the
  parameters. The third level is unidentified and is not adopted.
- **Open**: take-off dynamics and standby fluctuation of the layer; the VID
  M100 gap; durations of standby and flight (the user's choice in use); the
  spectra of NeuroBEM and Blackbird.

## Reproduce

```bash
scripts/rps_traj_regimes_submit.sh all bench     # 7 uni-cpu jobs + 1 uni-gpushort
# from .worktrees/submit-rps-regimes, per job: omnirun pull <job>, then copy
# omnirun-outputs/<job>/results/rps_traj/regimes/ into results/rps_traj/regimes/
python scripts/rps_traj_regimes.py table         # tables.md, summary.json, params/
python scripts/rps_traj_regimes.py multiscale    # paired real-data gate; model scores only if passed
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/rps_traj_regimes.py spectral
python scripts/rps_traj_regimes.py flight-inspector  # refresh telemetry in the existing flight_mds.html
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python scripts/rps_traj_regimes.py schedule --controls-only
python scripts/rps_traj_regimes.py schedule
PYTHONPATH=src MPLBACKEND=Agg python -m experiments.rps_traj.schedule_summary
cp results/rps_traj/regimes/schedule_eval/{schedule-table.tex,schedule-example.pdf} writing/papers/2026-08_wrapup/src/generator-assets/
```

The Gaussian stage is the slowest unit (138–827 s a fold, table above).
