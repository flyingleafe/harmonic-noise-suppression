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
  switching VAR is not a whole-flight generator on this data.
- **Cruise: adopt the two-level switch, and do not claim more than it shows.**
  It removes most of the Gaussian model's one-step deficit with three
  parameters and is the best arm 2 s ahead, but a three-state Markov-switching
  VAR predicts the next sample better on six rigs of seven with 5–30 times the
  parameters. The third level is unidentified and is not adopted.
- **Open**: the VID M100 gap; durations of standby and flight; the spectra of
  NeuroBEM and Blackbird.

## Reproduce

```bash
scripts/rps_traj_regimes_submit.sh all bench     # 7 uni-cpu jobs + 1 uni-gpushort
# from .worktrees/submit-rps-regimes, per job: omnirun pull <job>, then copy
# omnirun-outputs/<job>/results/rps_traj/regimes/ into results/rps_traj/regimes/
python scripts/rps_traj_regimes.py table         # tables.md, summary.json, params/
```

The Gaussian stage is the slowest unit (138–827 s a fold, table above).
