# Noise model v4 — the bench fit on the DREGON single-rotor ladders

**Status:** in progress (2026-10-19 –). Model and procedure:
`docs/explainers/noise-model-v4.qmd` (Fit procedure A). Code:
`src/experiments/noise_model/bench_v4.py`, driver `scripts/noise_v4_bench.py`.

## Motivation

The rotor-noise model of record is nv3r4 (`noise-model-v3.md` § Round 4). It
was fitted only to free-flight pools, its speed exponents came out unidentified
(`amp_exp` 8.9 → 13.3 as the iteration cap rose from 150 to 600, `floor_exp`
≈ 0), and no restart converged. The per-rotor bench fits it still consumes
(`rig_sampler.BENCH_MOTORS`) are noise-v2 R3 fits of a different model
(2026-09-18, 12/21 converged). v4 replaces the per-line Wiener width by a
per-order pedestal, splits the motor family from the aerodynamic lines, and
moves the speed laws to a joint fit over each motor's throttle ladder, where a
5-setting 1.8× span identifies them. Before v4 goes back to flight, it has to
fit the single rotors well.

**Question.** Does the v4 bench model, fitted jointly over each DREGON motor's
five throttle settings, reproduce the single-rotor recordings, recover planted
parameters, and agree with the independent quick measurements?

## Setup

Data: `DREGON-frames` `motor_Motor{1-4}_{50,60,70,80,90}`, 8 mics, decimated
44.1 → 16 kHz, one periodic-Hann periodogram over the strict rotor-on span of
`results/static_rig/single_rotor/dregon_singles.json` (9.8–36 s, 0.03–0.1 Hz
bins). Likelihood band 30 Hz–7.2 kHz (0.45 sr, clear of the decimator).
Carrier frozen at the measured core speed (±0.003 rev/s).

Model and priors: explainer tbl-fit-priors, bench column. Two deviations, both
forced by the recordings:

1. **Ambient.** The explainer reads the ambient "before the span minus 0.5 s".
   On every DREGON recording that segment holds the armed motor idling at
   about −64 to −69 dB, 3–8 dB above the stopped-rotor room (−72 dB, band level,
   mic median). The fit reads the ambient on every run within 2 dB of the
   recording's 5th-percentile level, at least 1 s from the span (Welch, 1 s
   segments).
2. **Wind** is one level per setting and mic, not per mic: the downwash grows
   with the throttle (Motor1 windy mics 5, 6: −43/−40 dB at 50 % to −24/−21 dB
   at 90 %, lower envelope at 30–90 Hz).

Pedestal convention: `sigma2_k` is the variance of `ln A` (what
`single_rotor.amplitude_jitter` measures) and the lag factor
`exp[sigma2 (e^{-2 pi gamma |tau|} - 1)]` keeps `10^{p/10}` the line's mean
POWER. The explainer's eq-aero writes `e^{g - sigma2/2}`, which keeps the mean
AMPLITUDE (power `e^{sigma2}` too high); `render_planted` uses the power
convention.

Optimiser: Adam (lr 0.02) then strong-Wolfe L-BFGS, then a fresh-memory L-BFGS
restart with half the budget; converged iff the restart gains < 1e-4 nats per
cell (the fit.py rule). Four restarts per motor, log-normal jitter (sd 0.8) on
the shaft and pedestal seeds; lowest objective kept.

Seeds: shaft from the width-law quick fit (`results/static_rig/single_rotor/ou/`;
prior median where it fell into the random-walk corner, λ < 3), pedestal from
`amplitude_jitter.json`, profile from the measured line levels moved to
80 rev/s with α = 5.5, floor from the lower envelope at the 14 control points,
pair EQ from the other four settings (leave-one-out, 200 Hz Gaussian).

### Acceptance (explainer § "Checks before acceptance", fixed before any fit)

| Check | Criterion |
|---|---|
| Planted synthetic | σ_ν, λ within 20 %; σ²_k within 30 % on lines ≥ 15 dB over floor; profile within 1 dB; α^a within 0.3 |
| Quick-fit agreement | on the 16 recordings where the quick fit held: \|ln σ_ν − ln σ_ν,quick\| ≤ 0.3, λ within ×1.5, profile within 2 dB on lines ≥ 10 dB over floor |
| No random-walk corner | M1_90, M2_80, M3_90, M4_90: λ in 5–30 1/s, HWHM at k = 80 within ×1.5 of the free per-order width |
| Exponents | α^a in 4.5–6.5; α^m in −3…+2; the BPF line (k = 2) on its own law within 1 dB rms over the ladder |
| Wind | Motor1: the profile with the windy mics (5, 6) excluded changes by < 0.5 dB |
| Cost | ≤ 10 min per setting |

Goodness of fit (reported, not gated): per band, the mean of `I/M` and the
median over ln 2 (both 1 for a correct model), and the line-bin power ratio
(data/model within ±1 Hz of each line) by order group.

### Forward-model validation (before any fit)

`render_planted` draws the seed model of the Motor1 ladder in the time domain
(both families on one OU shaft via `single_rotor.synthesise`, the pedestal as
a log-amplitude OU, shaped Gaussian floor, wind and ambient). Its periodogram
against the lag-domain expected periodogram: mean `I/M` 0.976–1.017 in every
band 30 Hz–7.2 kHz on all five settings, and the line-bin power within
−1.05…+0.79 dB by order group (one realisation). The forward model, its units,
the pedestal convention and the two-family sum agree with the generative
process.

## Results

## Conclusion
