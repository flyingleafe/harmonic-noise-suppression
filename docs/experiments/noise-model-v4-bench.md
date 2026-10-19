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

### Jobs

Cost: one objective + gradient over the Motor1 ladder (6.63 M cells) takes
9.6 s on the laptop CPU and 1.26 s on a Kaggle T4 (`nv4-bench-gpusmoke-kg-e1dc2c`,
HEAD `96c15409`); Motors 2–4 have 10 s spans on every setting and cost about a
third. Schedule: Adam 1500 steps (lr 0.02), L-BFGS 400 iterations, restart 200
(the explainer's 200-iteration L-BFGS raised because the round-4 flight fits
never met the restart test at 150). Two seeds per T4, both T4s of a Kaggle
session in parallel.

| job | HEAD | content |
|---|---|---|
| `nv4-bench-m1-7180f6` | `77bbfb36` | Motor1, restarts 0–3 |
| `nv4-bench-m234-3492bc` | `77bbfb36` | Motors 2–4, restarts 0–3 |
| `nv4-bench-aux-49cd61` | `77bbfb36` | Motor1 without the windy mics 5, 6 (restarts 0–3); planted Motor2 (truth = the seed model); Motor1 L-BFGS only (800 iterations, no Adam) |

## Results

### Motor1 (`nv4-bench-m1-7180f6`, `results/noise_v4/bench/fits/Motor1.json`)

**The optimiser converged; the fit is not the bench physics.** All four
restarts pass the restart test (gain 7.2–8.3e-5 nats/cell, 400 + 200 L-BFGS
iterations after 1500 Adam steps, about 18 min each on a T4) and agree to
4.5e-5 nats/cell; every fitted parameter agrees across restarts (σ_ν within 3 %,
α^a 2.80–2.90). The data/model ratio is near one almost everywhere (band means
0.86–1.21, 1.52–1.61 at 30–100 Hz on the 80/90 % settings). The selected point fails every
physical check:

| | fit | quick fit / measurement | criterion |
|---|---|---|---|
| σ_ν (rad/s), 50/60/70/80/90 % | 1.10 / 0.80 / 2.14 / 0.85 / 1.76 | 0.45 / 0.42 / 0.56 / 0.50 / – | \|Δ ln\| ≤ 0.3: 0/4 |
| λ (1/s) | 81 / 35 / 250 / 25 / 137 | 10.8 / 9.0 / 12.4 / 11.2 / – | ×1.5: 0/4; M1_90 in 5–30: no |
| α^a | 2.90 | 4.7 (Motor1, k ≤ 20), 6.2 (k = 2, pooled) | 4.5–6.5: fail |
| α^m | 1.40 | — | −3…+2: pass |
| BPF line on its law | 3.1 dB rms | — | ≤ 1 dB: fail |
| b0, b1 | 0.83, 1.52 | 2.35, 0.51 | — |
| profile vs quick reading (lines ≥ 10 dB) | median \|Δ\| 1.7–3.1 dB, 33–59 % within 2 dB | | within 2 dB |

**Mechanism: the line skirts replace the floor.** With λ = 25–250 1/s the
shaft OU is nearly white frequency noise, and every line is a Lorentzian whose
1/Δf² skirt reaches the neighbouring orders. The fitted floor spline sits at
−80 to −103 dB on Motor1_70 mic 0 at 68–685 Hz (the data between the lines:
−45 to −55 dB); in the
gaps half-way between orders the lines carry 45–90 % of the model power below
1 kHz and the floor 0–3 %. The skirts are shaped per mic and per line by the
scatter δ (2.4 dB) and the pair EQ, which the floor (one shape, one gain per
mic) cannot match. To fill the low band through skirts the fit raises the
low-order cores: at k = 1 on Motor1_70 mic 0 the model core is −12.8 dB
against −38.1 dB in the data, and the power within ±1 Hz of orders 1–4 is
7.6–10.6 dB above the data on every setting. The wind term follows: the calm
mics' wind rises from its −73 dB centre to −55 dB and the windy mics 5, 6 fall
from −33/−31 to −47/−53 dB (Motor1_70), so the wind no longer sits on the
windy mics. The priors cost little against 6.6 M cells: λ = 250 is 6 prior
sd from 12 (18 nats), the floor mean 5.8 dB under its centre.
<!-- source: results/noise_v4/bench/checks.json; restarts/Motor1__s{0-3}.json -->

### Motors 2–4 (`nv4-bench-m234-3492bc`)

Same outcome, and worse: no restart converges (0/4 on each motor; restart gain
above 1e-4/cell), the restarts disagree by 6.8e-4 (Motor2), 1.0e-3 (Motor3) and
5.3e-5 (Motor4) nats/cell, and λ runs further: 54–1503 (Motor2), 793–9058
(Motor3), 128–6923 1/s (Motor4), σ_ν 0.8–8.9 rad/s. α^a is 4.07 / 2.91 / 2.70
(Motor2/3/4), only Motor2 near the measurement; the BPF line misses its own
law by 1.7 / 2.9 / 3.4 dB rms. Power within ±1 Hz of orders 1–4 is 1.6–14.1 dB
above the data on every setting. Every fit fails quick-fit agreement on all
settings and λ ∈ 5–30 on its random-walk-corner recording.

**What the fits agree on is the phase diffusion** `D = 2 σ_ν² / λ` (rad²/s),
the only shaft quantity a Lorentzian line carries: 0.47–1.31× the quick fit's
D on the 16 recordings where the quick fit held (0.47–0.99× on all but
Motor1_80). The fit slides along the ridge of constant D to the
white-frequency-noise limit (λ → ∞), where a line has no Gaussian bend at high
order and keeps its 1/Δf² skirt. The width-law quick fit reads λ from exactly
that bend (k² widths to k ≈ 20, near-linear above), so the likelihood trades
the bend for skirts that build the floor. The ridge also explains the
non-convergence and the restart spread.
<!-- source: results/noise_v4/bench/checks.json; fits/Motor{2,3,4}.json -->

**The ridge is in the likelihood, not the prior.** Moving the Motor1_70 fit
along constant D (σ_ν = √(Dλ/2), everything else held) costs +125 550 nats at
λ = 10 against the λ = 320 optimum; Motor3_60 is monotone to λ = 9000. 73 % of
the Motor1_70 cost sits more than 5 Hz from any line (2–7.2 kHz mostly), where
the collapsed floor (−80 to −103 dB) leaves the gaps empty once the skirts go.
The scan holds the floor, wind and δ at the skirt solution, so it overstates
the gap; no prior of a few hundred nats can decide it either way.

### Round 2 — one shaft law per motor

The per-setting `(σ_ν, λ)` pair (10 free numbers per motor, log-normal priors)
is replaced by one speed law per ladder (`LadderModel.shaft`, schema
`noise-v4-fit/2`): σ_ν = 2π f_ref ε r^a, λ = λ_ref r^b, r = f / 80 rev/s.
ε is the relative rms speed jitter at f_ref, HalfNormal(0.005) in linear space;
a, b ~ N(1, 0.5); λ_ref carries no prior. a = b = 1 is the aero-damped motor
whose torque disturbance scales with its load: constant relative jitter,
damping ∝ speed, D ∝ speed. With a speed-independent electrical/ESC damping
term b falls below 1. The quick fit on its 16 good recordings, pooled with
per-motor intercepts, gives σ_ν ∝ f^0.60±0.19, λ ∝ f^0.70±0.23, D ∝ f^0.51±0.20,
over a 1.6× speed span (a check of the law's form, not a prior centre).
Along the ridge ε ∝ √λ, so ε's prior costs ~1.5 nats per 1000 1/s of λ_ref
on Motor1: it cannot hold the rate. What can is the tie. Two arms on Motor1
and Motor3: the free law, and the pinned arm `--pin-lam 12` (λ = 12 r,
i.e. 7–13 1/s, ε and a still free). The objective gap between them, with
floor, wind and δ refitted, decides between a prior and a model change.

## Conclusion
