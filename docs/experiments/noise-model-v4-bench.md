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

Jobs `nv4-law-motor1-a04b4d`, `nv4-law-motor3-17ccc4` (uni, one seed each,
1500 Adam + 400 + 200 L-BFGS; `results/noise_v4/bench_law/restarts/`):

| Arm | Total (nats) | vs per-setting R1 | ε | a | λ_ref (1/s) | b | λ over ladder | D (rad²/s) | α^a | conv |
|---|--:|--:|--:|--:|--:|--:|---|---|--:|:-:|
| Motor1 free law | −72 247 294 | +10 639 | 0.28 % | 0.86 | 98 | 0.96 | 61–107 | 0.028–0.044 | 2.83 | yes |
| Motor1 λ pinned | −72 233 155 | +24 778 | 0.17 % | 0.79 | 12 | 1 | 7.4–13.2 | 0.097–0.135 | 2.86 | yes |
| Motor3 free law | −29 901 065 | +3 838 | 2.6 % | −0.37 | 13 062 | −1.05 | 11 751–21 817 | 0.023–0.028 | 2.65 | no |
| Motor3 λ pinned | −29 876 716 | +28 187 | 0.15 % | 0.58 | 12 | 1 | 7.4–13.3 | 0.089–0.097 | 2.88 | no |

Pinning λ near the quick-fit value costs 14 139 nats on Motor1 (2.1e-3/cell)
and 24 350 on Motor3 (8.4e-3/cell) once everything else is refitted (Motor1_70:
5 677 nats against the scan's 125 550), still far beyond any prior. 75–80 % of the loss sits
more than 1.5 Hz from a line (Motor1: 11 508 of 14 139 nats; Motor3: 18 107
of 24 350), 1–7 kHz mostly: the data carry 1/Δf² wings on the high orders
that a single OU at λ ≈ 10 (corner 1.6 Hz, 1/Δf⁴ beyond it) cannot make.
On Motor1 the tie alone settles to a coherent law (a = 0.86, b = 0.96, wing
corner 10–17 Hz); on Motor3 it runs off (a, b three prior sd from 1). α^a
stays at 2.65–2.88 in every arm, independent of the shaft.
<!-- source: results/noise_v4/bench_law/restarts/Motor{1,3}__s0{,__lam12}.json -->

### Round 3 — the pedestal without its render cap

Why the floor does not take the inter-line power: there is no broadband floor
to take. Folded by distance to the nearest line (Motor1_70, 1–2 kHz, mic mean),
the data fall without a plateau from −34.6 dB on the line to −41.9 (2–3 Hz),
−49.1 (12–18 Hz) and −52.0 dB midway; the line-to-midway drop is 9–24 dB on
every setting at 0.3–2 kHz, and across the 8 mics the midway level tracks the
line level (r = 0.92, drop 14.7–18.5 dB per mic). The floor (one spline, scalar
mic gains) is flat in Δf and cannot follow the line EQ. The only line-attached
wide terms are the pedestal and the shaft; R2 fitted σ²_k at the 0.6 cap
(0.56–0.59 for k ≥ 30, γ 8–50 Hz), so the rest went to the shaft at large λ.
The cap is a render limit (explainer § am-pm, 2026-10-02), not a measurement.

`--no-sigma2-cap` (σ²_k = e^u, priors unchanged), free law and λ pinned at 12 r
(`nv4-nocap-motor{1,3}`, 400 + 200 L-BFGS), then the free law at 1200 + 600
with and without `--center-delta` (zero-mean δ per setting;
`nv4-nocap-long-motor{1,3}`):

| Arm | Total (nats) | vs capped free law | λ over ladder (1/s) | σ_ν (rad/s) | D vs quick | conv |
|---|--:|--:|---|---|---|:-:|
| Motor1 uncapped, 1200 | −72 319 049 | −71 755 | 12.4–29.6 | 0.49–0.83 | 0.85–1.08× | no (2.1e-4) |
| Motor1 uncapped, λ pinned, 400 | −72 311 462 | — | 7.4–13.2 | 0.46–0.71 | — | no |
| Motor1 uncapped + δ centred | −72 317 460 | −70 166 | 12.4–29.9 | 0.49–0.83 | same | no (2.6e-4) |
| Motor3 uncapped, 1200 | −29 944 760 | −43 695 | 18.8–32.0 | 0.54–0.61 | 0.57–0.88× | yes (9.2e-5) |
| Motor3 uncapped, λ pinned, 400 | −29 941 796 | — | 7.4–13.3 | 0.38–0.56 | — | no |
| Motor3 uncapped + δ centred | −29 943 254 | −42 189 | 17.6–32.8 | 0.55–0.60 | same | no (2.5e-4) |

Lifting the cap is worth 44–72 k nats, and the shaft returns to the quick
fit's regime: at equal schedule (400 + 200) the free law beats λ pinned at
12 r by 709 (Motor1) and 683 nats (Motor3), against 14–24 k with the cap.
σ²_k grows with order, 0.15 at k = 12, 0.5–0.9 at k = 16–20, 1.5 at k = 30,
2.1–2.9 at k = 40, 2.7–4.1 at k = 60–80 (a coherent share e^{−σ²} of 2–7 %
at high order), with γ_k 1.5–7 Hz (prior centre 1 Hz). The pedestal now draws
the wings: the Motor1_70 fold matches within 0.5 dB from the line to midway
except the 2–3 Hz bin (model 1.7 dB high), and the floor sits at −79.6 dB at 1–2 kHz, 27 dB
under the midway level — the bench shows no broadband floor there, and its
speed exponent b0 wanders (3.4–9.0) because nothing pins it.

The line speed exponent leaked into δ: in R2 and the uncapped fits the mean
δ over k ≤ 40 rises 3–4 dB across the ladder, so the effective exponent
(α^a plus the δ trend) is 4.0–4.5 at k ≤ 40 and 5.7–6.8 on the BPF, matching
the measurement (4.7/5.0 at k ≤ 20, 6.2 on the BPF), while α^a alone reads
2.7–3.8. Centring δ moves α^a to 4.12 (Motor1) / 3.69 (Motor3) at a cost of
~1.5 k nats; the effective exponents stay where they were, so the residual
trend is an order dependence of the exponent, not a free offset.
<!-- source: results/noise_v4/bench_law/restarts/Motor{1,3}__s0__nocap{,__long,__dc__long}.json, Motor{1,3}__s0__lam12__nocap.json -->

### Round 4 — the v4 spec: one model, two fits (2026-10-21)

Code at `b2de6db3` (bench, flight, renderer) and `453b847c` (flight: one
shared floor, shaft speed law). Spec: `docs/explainers/noise-model-v4.qmd`.

**A0, phase-pedestal width** (`nv4-a0-phasewidth-ff6078`, uni-cpu, 20
recordings). New instrument (`scripts/noise_v4_phase_width.py`): the old
20→50 ms jump was its band-flank floor. $c_W$ = p97.5/2.24: even orders
6.94 Hz (n = 387), odd 30.68 Hz (n = 94); median W 1.0–1.6 Hz at k ≤ 20,
8.7 Hz at k > 40 (even). Resolved lags only: 18.4 / 163 Hz.
<!-- source: results/noise_v4/phase_width/{summary.json,findings.md} -->

**Timing (A100).** Bench 15.3 M cells, 0.98–1.98 s per objective; L-BFGS
0.29 s per evaluation; one restart 1125–1139 s, so the bench fits on
uni-gpushort. Flight 4.75 M cells, 1.71 s; one restart ~910 s.

**Bench restarts s1, s2** (`nv4-bench-s1-deb00a`, `nv4-bench-s2-aeb4e9`,
κ = 0, Adam 1500 + L-BFGS 1200). Not converged (restart-pass gain 3.1e-3 /
2.9e-3 nats/cell; floor moved 3.5 / 6.4 dB, exponents 0.29 / 0.42). They end
662 k nats apart: α 5.37 / 4.22, β 1.00 / 0.15, b₀ 1.11 / 0.57, b₁ 0.89 / 1.54.
The shaft agrees: σ_ν(80 rev/s) 0.72 / 0.74 rad/s, a_σ 1.11 / 1.56, λ_ref
54.5 / 47.0 1/s, b_λ 2.54 / 2.45 (λ ×4.7 over the ladder). The floor
collapses below 300 Hz (control values −103 to −213 dB at 30–260 Hz against
−48 to −66 dB above 400 Hz): with κ = 0 nothing holds it there. The evidence
freeze plus the restart and release passes end 826 nats worse than the
L-BFGS pass (s1).
<!-- source: results/noise_v4/bench/restarts/dregon__kappa0__s{1,2}.json params, optimiser -->

**Flight, first launch: summed floor, flat shaft** (`nv4-flight-s01/s23`, 4
restarts). Not converged (polish gain 2.1–2.7e-4 nats/cell). Restarts agree
to 550 nats but the floor law is a rotor selector: b₀ 34.0–35.0, b₁ −6.7 to
−7.3, s₀ 5.6–6.6 (prior median 2.5e-3); at cruise s^34 gives RFront 97 % of
the floor, at standby the rotor floor is zero. α 4.04–4.07, β 7.96–8.68,
σ_ν 6.36–6.44 rad/s. The flat σ_ν smears order 84 by 86 Hz (sd): the fitted
expected periodogram is flat across the standby k = 84 lines, which stand
5.4–16.2 dB over their neighbours in 5.4 Hz bins. Cruise k = 84 (6.2–7.7 kHz)
is invisible at 44.1 kHz too. Decisions: one shared flight floor on the
rotor-mean speed; the flight shaft size keeps its speed law (a_σ ~ N(1, 2)),
rate pinned. Relaunched as group `nv4-flight2`.
<!-- source: results/noise_v4/flight_r0_sumfloor_flatshaft/restarts/*.json;
results/noise_v4/md2_motor_orders/fly125_{spectrogram,cruise_k84}_44k.png -->

**Next.** κ from a planted check whose truth has a measured floor below
400 Hz; then a longer bench schedule until the restart pass gains < 1e-4
nats/cell; then the flat-shaft refit.

### Round 5 — bounded priors, κ, converged bench fits (2026-10-22)

**Priors.** $W_\phi \le 24$ Hz, $\sigma^2_\phi \le 3$, one $c_W = 6.94$ Hz
for every order, $b(f) \ge 0$ on the band (`v4_priors`). Spec
`docs/explainers/noise-model-v4.qmd` § The priors.

**κ by the planted check.** Truth: the r5 s2 fit with its floor below
2.5 kHz replaced by the measured inter-line floor.

| κ | floor err ≥ 500 Hz, rms / max | mean err < 500 Hz | median $W$ err |
|---|---|---|---|
| 0 | 1.5 / 2.4 dB | +2.5 dB | 29 % |
| 100 | 1.4 / 2.1 dB | +16.5 dB | 35 % |
| 1000 | 1.8 / 4.3 dB | +19.5 dB | 45 % |
| 1e4 | diverged | | |

No κ meets the 1 dB rule. Below 500 Hz the bench floor and the bench wind
explain the same cells and the push only lifts the floor. Decision: κ = 0.
<!-- source: results/noise_v4/bench/planted/dregon__kappa0__r5__s2__floor2500__kappa{0,100}__r1.json -->

**Converged bench fits** (uni, 3 h): κ 0 / 10 / 100 at −158,957,322.8 /
−158,948,445.0 / −158,905,892.1 nats, 13 restart passes. In-sample check
(`nv4-insample-r5b-72401b`): mean |model − data| 1.6–3.2 dB per recording; at
0.4–2 kHz the floor sits 55–60 dB under the total, so the pedestals carry the
broadband level.
<!-- source: results/noise_v4/bench/restarts/dregon__kappa{0,10,100}__r5__s2__c1.json -->

### Round 6 — minimum, η, flight fits (2026-10-23)

**Saddle escape.** The κ = 0 fit `c1` passed the convergence test but had 38
directions of negative curvature (each on one coordinate: `log_gA`, `u_amp`,
`u_W`, `p_aero` of near-invisible orders; η came out NaN). `noise_v4_bench.py
escape` (Hessian, line search down each negative direction, convergence loop,
repeat; uni-gpushort, one Hessian 1290–1310 s):

| step | negative directions | gain | objective (nats) |
|---|---|---|---|
| c1 → c2 | 38 → 12 | 1835 | |
| c2 → c3 | 12 → 5 | 607.9 | −158,965,903.0 |
| c3 → c4 | 5 → 4 | 135.4 | −158,966,038.4 |
| c4 → c5 | 4 → 1 | 26.8 | −158,966,065.2 |
| c5 → c6 → c7 | 1 → 1 | 0 | −158,966,065.2 |

The last direction (eigenvalue −0.275, a quadratic gain of 0.034 nats at the
smallest trial step) lowers the objective at no trial step in either sign.
Rule adopted (`bench_eta.MINIMUM_RULE`): such a payload is a minimum, and the
direction is left out of $H^{-1}$ like a flat one. `c7` is the final bench fit.
Jobs `nv4-escape-k0g2-78e2a2`, `-k0g3-edff74`, `-k0g4-6580d2`,
`-k0g5-c7131a`, `nv4-eta-final-16b888`.
<!-- source: results/noise_v4/bench/restarts/dregon__kappa0__r5__s2__c{3..7}.json optimiser.saddle_escape -->

**η on the final fit** (block jackknife; $p_{\rm eff}$ = 655):

| blocks | 200 Hz | 400 Hz | 800 Hz | 1600 Hz | one per recording |
|---|---|---|---|---|---|
| η | 0.00152 | 0.00145 | 0.00140 | 0.00135 | 0.00132 |

η falls slowly with block size: the cells are strongly dependent, and the
dependence reaches across 1.6 kHz.
<!-- source: results/noise_v4/bench/eta/dregon__kappa0__r5__s2__c7.json -->

**Planted twin of the final fit** (truth `c6`, κ = 0, resumed from the truth;
`nv4-planted-final-3143af`): converged (1.3e-6 nats/cell). α (0.04), ε (4 %),
λ (5 %) recover; profiles on lines ≥ 10 dB: median error 0.35 dB, 88 % within
1 dB; $W_\phi$ on lines ≥ 15 dB: median 18 %, 71 % within 30 %; $b_0$ off by
1.0; the floor off by up to 55 dB (rms 16 dB), where the truth floor had
collapsed below 300 Hz.
<!-- source: results/noise_v4/bench/planted/dregon__kappa0__r5__s2__c6__kappa0__r1.json -->

**Defect found and fixed: the planted payload recorded the wrong pair EQ.**
The planted fit records −166,174,609.7 nats, but its `.fit.json` reloaded on
the same planted rig scored −166,140,705.1 (also inside one job, so not a
cross-node effect). Cause (`nv4-planted-diag-23ee04`): the fit resumed from
the truth payload, and `load_payload` set the model's pair EQ to the TRUTH's;
`bench_v4.payload` then wrote the EQ measured on the planted data
(`rig.eq_all`), up to 3.2 dB different. Loading the truth's EQ with the same
parameters reproduces −166,174,609.7 exactly. Fix (`5c13fc73`): the payload
writes the EQ the model computed with; the planted `.fit.json` is repaired
(truth EQ, `mic_response.repair` records it). On real data the two EQs are
the same measurement, so the bench chain c1–c7 is unaffected (its reload
reproduced the objective exactly). Caveat for the twin: it used the truth's
pair EQ (an oracle constant), not one measured on the planted data. The
recovery checks above are unchanged (they read the fitted model, not the
payload). Escape on the repaired twin (`nv4-planted-escape-61cfde`, kaggle,
500 min): the reload reproduces −166,174,609.7, so the fix holds. Four rounds
lower the objective by 4,713.9 nats to −166,179,323.0 (negative directions
57 → 32 → 12 → 8; min eigenvalue −0.53). The deadline stopped round 4 before
its line search, so the twin is not yet at a minimum and η did not run. The
continuation (`nv4-planted-escape2-c75cd0`, reused Hessian, 480 min) ran
five more rounds to −166,179,350.7 (`…r1.fit__c2.json`). Negative directions
8 → 4 → 8 → 6 → 6, min eigenvalue −0.53 → −0.31. They are all pedestal
amplitude, width and share coordinates (`u_amp`, `u_W`, `u_s2P`). The round
gains are 1.17, 0.10, 0.08 and 0.04 nats. On 15.3 M cells that is 8e-8 to
3e-9 per cell, under the convergence tolerance. The strict minimum rule
(no direction lowers the objective at any step) is never met, so η on the
twin is not reported yet.

**MD2 flight.** The converged joint fit (`md2_fly125__flight_v4__s0__r5c1`)
scored standby PIT 27.8 and parity ratio 3.57, so the per-regime fallback
applies. Two changes of procedure and prior: the phase pedestal is fitted per
order (`--free-phase-pedestal`; the held median filled the cruise valleys 10 dB
too high), and $\lvert\alpha^m\rvert \le 4$ (per-regime pools pushed it to
−40.8 / +17.5). Per-order pedestal groups move by the median over orders in
the convergence test. Converged pair: standby `…_standby_fpb__flight_v4__s0__r6c1`
(17 passes), cruise `…_cruise_fpb__flight_v4__s0__r6c3`
(`nv4-md2-cruise-fpg4-caa813`). Scores (`nv4-score-md2-r6c3-449bbb`):

| | v4 pair | nv3r4 |
|---|---|---|
| parity ratio | **0.481** | 0.515 |
| PIT standby / ramp / cruise | 0.359 / 3.309 / 0.700 | 0.369 / 3.604 / 0.701 |
| cruise LTAS | 1.516 dB | 1.541 dB |
| Listen LTAS, seed 0 | 0.97 dB | 0.95 dB |
| Listen LTAS, seeds 0–5 | 0.987 ± 0.054 | 0.975 ± 0.029 |

The Listen band errors are the same for both models (FLY125 fit, FLY124 clip);
accepted as on par (Dmitrii, 2026-10-23).

**DREGON flight stage** (procedure C of the spec). Scores against nv3r4's
PIT 1.680 / LTAS 2.384:

| stage | PIT | LTAS | job |
|---|---|---|---|
| no gains (bench κ = 0) | 15.8 | 2.384 | |
| + gain per mic | 19.57 | 2.47 | `nv4-score-mgnl-k0-ccc534` |
| + gain per rotor (bench κ = 10) | 61.24 | 2.60 | `nv4-score-lg-k10-ba7b42` |
| + gain per rotor and mic | 66.36 | 2.85 | `nv4-score-mg-k0-4f5cf2` |
| + coherence free (40 passes, not converged) | 29.4 | 3.20 | |
| probe: shaft ε 0.02 / 0.05 (no refit) | 21.7 / 43.1 | 2.33 / 2.31 | |

Local scorer probes (hovering and spinning windows): the scorer reads
absolute CQT dB; the scoring windows are 2–6 dB louder above 120 Hz than the
fit segments; v4 matches the fit segments, nv3r4 sits 3.5–4.5 dB above them on
mic 2. Lines +3 dB or floor −6 dB bring every mic back (2.01 / 2.16), still
above nv3r4 (1.43 / 1.61); with the four rotors' profiles averaged it is 8.3 /
10.3. Details and the comparison: `docs/explainers/noise-model-v4-dregon-gap.qmd`
§ 5–6.

**C-wind: the wind as a separate gusty term (2026-10-23).** Real DREGON
wind is gusty and independent per capsule: on the five scoring windows the
20–400 Hz level between the lines has an sd of 2.9–3.4 dB over 128 ms frames
(cross-mic correlation 0.07–0.24); the static wind of v4 and nv3r4 gives
1.3–1.5 dB (only the scatter of a steady noise). New wind kind `sc` (code
`bd6c7596`): per capsule one SC generator draw (`wind_sc.py`, speed
`offset + scale · Weibull(2)`, `rate` knots per second), level per mic by
Whittle with lines and floor frozen, gust law by simulated moments of
`floor_envelope` after a regression on the 1–4 kHz level (grid 5 × 5 × 5,
4 seeds, job `nv4-cwind-grid-e2ddae`, 11 min on uni-cpu). Chosen: offset 2,
scale 0.5 m/s, rate 2 Hz; renderer check 0.30 dB (tol 1). Payload
`results/noise_v4/flight/wind/…__scwind.json`. One gust law for all mics
under-reads the gust sd on mics 1, 2, 5, 6 (1.9–2.9 against 4.2–5.2 dB).
Full scorer (`nv4-score-cwind-4955be`, seeds 2001–2004): **PIT 9.62**
(k0 15.8, bar 1.680), **LTAS 1.952** (bar 2.384), **Listen 2.39** (bar
3.08). The gust wind beats nv3r4 on both spectral measures and fails PIT.

Local HPPNet probes (render seed 2001, all five windows, mean of 8 mics; bar
1.680; scripts `/tmp/hpp_gust_v3.py`, `hpp_odd.py`, `hpp_swap.py`, not kept):

| arm | HPPNet |
|---|---|
| real | 1.22 |
| nv3r4 (static wind) | 1.67 |
| nv3r4, static wind → SC gusts (default law, 3 / 8 knots), same band power | 1.73 / 1.80 |
| v4 k0 (static wind) | 17.57 |
| v4 k0, static wind → SC gusts, 8 knots, same band power | 2.30 |
| v4 C-wind (fitted gust law) | 15.57 |
| v4 k0 + gusts (8 knots), odd orders 3 / 5 / 7 raised by 10 / 14 / 7 dB | 7.27 |
| v4 k0 + gusts (8 knots), all lines +3 dB | 2.21 |
| v4 lines + nv3r4 floor and wind | 15.02 |
| nv3r4 lines + v4 floor + gusts (8 knots) | 2.03 |

Reading: the default SC law (gust sd 5.9 dB, twice the real one) hides the
gap; the fitted law does not. The weak odd orders and the motor family are
not the cause (raising or removing them makes it worse). The fragile part is
the v4 line set: nv3r4's lines survive v4's background, v4's lines do not
survive nv3r4's; the failures are whole rotors lost on single mics (mics 0
and 4 under C-wind), which points at the bench per-(rotor, mic) line balance
[INFERENCE].

**Speed-law search (2026-10-23).** A direct Sobol search (256 + 128 points)
over the line level $g_a$ and exponent $\alpha_a$, the motor level and
exponent, the floor level $g_f$ and $b_0$, on the C-wind payload, scored by
HPPNet PIT and the scorer LTAS on the five 8 s tuning segments (never the
scoring windows); code `e61f289d`, job `nv4-lawsearch-5f4381`, table
`results/noise_v4/flight/lawsearch/…__laws.table.json`. Tuning references:
real PIT 0.93; nv3r4 PIT 1.54, LTAS 1.72; C-wind PIT 5.57, LTAS 1.86. No
point beats nv3r4 on PIT; best overall 1.79 (LTAS 3.54), best with LTAS ≤
nv3r4 1.81 (LTAS 1.56; $g_a$ +2.2, $g_f$ −5.2 dB). Main effects: $g_a$
moves PIT by 14.6 rev/s and $g_f$ by 7.0; every exponent by ≤ 1.3 (noise).
The in-flight speed laws are not what is missing; the line-to-floor ratio is,
and it saturates at PIT ≈ 1.8.
Point A on the official scorer (held-out windows, `nv4-score-lawsa-1c2691`):
PIT 3.03 (C-wind 9.62, bar 1.680), LTAS 2.61 (C-wind 1.95, bar 2.384),
Listen 3.23 (C-wind 2.39, bar 3.08). The tuning gain transfers only in part
(PIT 1.81 on the tuning segments) and costs both spectral measures.

**Similarity battery (2026-10-23).** `experiments/noise_model/similarity.py`,
report `results/noise_v4/similarity/report.md` (render seed 2001, 8 mics;
noise floor = a real window's first half against its second half). On the
tuning segments: Itakura–Saito real‖arm nv3r4 0.170 (floor 0.175), C-wind
0.231, law point A 0.289, B 0.413; order prominence |Δ| nv3r4 0.71, C-wind
0.91 (floor 0.97); frame-level distribution shape (W1 centred) C-wind 0.94,
nv3r4 1.26 (floor 0.60); modulation spectrum C-wind 5.7, nv3r4 7.0 dB (floor
4.6); level sd still 1.0 dB too small for C-wind, 1.5 for nv3r4. Reading:
C-wind is closer than nv3r4 in time structure and per-mic spread, nv3r4 is
closer in line and order structure; the PIT-best law point B (+6 dB lines) is
the least realistic spectrally, so PIT alone rewards over-prominent lines. On
the 4 s scoring windows the half-split floor is large (LTAS 3.6) because the
real windows drift in level inside 4 s, and every model is 1.6–3.7 dB quieter
than real there (the level gap).

*Correction and time-aware measures (`f6ab4cf9`).* The half-split floor is
invalid (the halves run on different speed trajectories, so lines are
misplaced, and the level drifts); it is replaced by a seed floor (seed 2001
vs 2002 of the same payload on the same trajectory). The linear-bin IS is
time-blind and 65–72 % from 2–8 kHz; v4's IS deficit is all in 4–8 kHz
(0.25 vs 0.15 per bin), while v4 is closer than nv3r4 at 62–250 Hz. Frame
third-octave Fréchet distance (tuning; seed floors 9–14 dB²): nv3r4 362
(mean term 241, covariance 121), C-wind 308 (247, 61); scoring windows:
543 vs 466. Cross-band co-modulation (mean off-diagonal r of frame levels):
real 0.241 (scoring 0.444), nv3r4 0.067 (0.119), static v4 0.019 (0.039),
C-wind 0.124 (0.153). Time-aware measures favour C-wind; it carries half of
the real broadband co-modulation, nv3r4 a quarter.

**Alternating flight fit (2026-10-23/24).** Wide pool `dregon-flight-wide`
(18 windows, 144 s, the five tuning segments first), source the c7-lines C-wind
payload. Gate on the tuning segments: PIT ≤ 1.54 and every battery measure no
worse than nv3r4. Round 1, lines k ≤ 30 with ε free (`nv4-lines-r1-f076f0`,
not converged in 20 passes): ε 0.00145 → 0.049 (σ_ref 24.6 rad/s), PIT 8.78;
the same payload with the bench ε restored gives 1.84. ε absorbs the 0.5–0.7 Hz
label error and smears the high orders. Fix (`daae59f7`, `--hold-eps`): ε held
at the bench value, only k ≤ 10 free (a 0.7 Hz label error moves k = 10 by
7 Hz, inside one 7.8 Hz bin). Results (render seed 2001; tuning segments):

|Stage|Job|Converged|PIT|LTAS|Line / floor Δ (dB)|IS oct.|Fréchet (mean, cov)|Off-diag r (real 0.241)|
|---|---|---|---|---|---|---|---|---|
|nv3r4|—|—|1.54|1.72|+0.8 / +0.7|0.240|362 (241, 121)|0.067|
|R1 lines k ≤ 10|`nv4-lines-k10-a27434` + `c1-480a90`|yes (80 passes, 9e-8/cell)|2.18|2.60|+2.6 / +3.1|0.348|439 (383, 56)|0.153|
|Gust search|`nv4-gust-r1-7fa8a6`|175/175 points|1.97|2.40|+2.5 / +2.4|0.337|324 (260, 64)|0.240|
|R2 lines k ≤ 10|`nv4-lines-r2-608b60`|yes (30 passes, 1.4e-7/cell)|1.92|2.35|+2.4 / +2.5|0.327|311 (250, 61)|0.234|

Gust search: battery objective (sum of measure / seed-floor ratios) 113 → 92;
chosen law offset 2 m/s, scale 1 m/s, rate 8 Hz (rate on the grid edge, but the
objective is flat in rate: 91.7 / 93.5 / 94.1 at 8 / 4 / 2 Hz). R2 payload
`results/noise_v4/flight/lines_r2/…__c1__gust__wide__lines.json`. On the scoring
windows (report only) R1 was closer than nv3r4 on every spectral measure (LTAS
1.72 vs 2.38, IS 0.40 vs 1.55, Fréchet 312 vs 543). Reading: the gate fails
at the round cap (2). R2 beats nv3r4 on time structure (Fréchet, covariance
term, co-modulation close to real), but is 2.4 dB too loud on the tuning
segments and 0.9–1.2 dB too quiet on the scoring windows. One shared level law
cannot fit both window sets. It fits the mean of the 18 windows [INFERENCE].

*Second tracker (2026-10-24).* The same arms under HRSPNet
`rp_pyrseplstm_k64_alllab` (`best_real_median`, rp-protocol grid 0–159 rev/s,
164 bins; `scripts/noise_v4_pit_probe.py`, `e1f211e5`), tuning segments, seed
2001, mean PIT (rev/s): real 0.67, nv3r4 0.98, C-wind 1.67, R1 k ≤ 30 13.58
(bench ε restored: 1.71), R1 k ≤ 10 1.69, gust search 1.63, R2 1.63. HPPNet
on the same arms: real 0.93, nv3r4 1.54, C-wind 5.57, R2 1.92. The ranking
holds (nv3r4 is better than every v4 arm under both trackers), but the
C-wind collapse is HPPNet-specific (5.57 vs 1.67), and the alternating fit
changes HRSPNet PIT little (1.67 → 1.63).
Official scoring windows (report only; 4 s, seeds 2001–2004, job
`nv4-pit-trackers3-804ecd`, `results/noise_v4/pit_probe/scored_*.json`; the
HPPNet column reproduces the official nv3r4 1.680 and C-wind 9.62 exactly):

|Arm|HPPNet (bar 1.680)|HRSPNet k64|
|---|---|---|
|real|1.219|0.956|
|nv3r4|1.680|1.105|
|C-wind|9.618|1.926|
|R1 lines k ≤ 10|2.996|2.202|
|gust search|2.321|1.885|
|R2 lines k ≤ 10|2.239|1.861|

Under HRSPNet, nv3r4 is 0.15 rev/s from real and R2 0.91; the v4 gap is not
a property of HPPNet alone.

*Trim search on R2 (2026-10-24; Dmitrii: "adjust the floor and lines a
little until the gate is passed, at the smallest loss of spectral
similarity").* Gate: tuning PIT below nv3r4's 1.541. Selection: the
lowest mic-0 LTAS error. Law-search composition with an order-1 gain
`g_a1` (`2c9dba52`). Box: `g_a` −3…9, `g_a1` −3…9, `g_f` −9…3 dB. 256 + 128
Sobol points, job `nv4-trim-r2-dae303`, table `results/noise_v4/flight/trim/`.
Result: no feasible point. PIT floors at 1.635 near `g_a` +5…7, `g_a1`
+7…9, `g_f` −5…−7 dB (LTAS 2.8–4.5 there, against 2.36 for R2).
Direct-render probes at the trim (seed 2001, tuning, HPPNet):

|Change at the trim|Mean PIT|
|---|---|
|none|1.637|
|wind −10 dB|1.658|
|ε × 0.3|1.661|
|motor lines off|1.632|
|orders ≥ 11 −4 / +4 / +8 dB|1.655 / 1.644 / 1.987|
|orders ≥ 11 off|2.098|
|bench pedestal for k ≤ 10|1.684 (R2 untrimmed: 1.92 → 1.81)|

The R2 refit moved the k ≤ 10 pedestals far from the bench: σ²_φ for k = 1
went from 0.001 to 1.2 (8.8 Hz wide), and for k = 6–10 to the cap 3 at
24 Hz. The label error enters through the pedestal when ε is held. No
global level, width or wind change passes 1.635. Orders ≥ 11 carry
tracker information (off: +0.46). Their per-rotor pattern differs from
nv3r4's flight-fitted one (per-order rotor-pattern correlation −0.96…0.99).
Per-rotor line gains (`9d8f96c2`, job `nv4-trim-rotors-62814f`; box `g_a`
3…9, `g_a1` 3…11, `g_r1..4` −6…6, `g_f` −9…−3 dB; 512 + 256 points): 204
of 768 points pass the PIT gate (best 1.423). The chosen point, the lowest
mic-0 LTAS among them (`results/noise_v4/flight/trim_rotors/…__laws.json`),
has `g_a` +5.0, `g_a1` +7.5, rotor gains −4.9 / −4.8 / −3.1 / 0.0, `g_f`
−8.4 dB, which nets rotor lines +0.1 / +0.2 / +1.9 / +5.0 dB. Main effect:
`g_r4` (PIT spread 0.32). Tuning: PIT 1.516 (bar 1.541), mic-0 LTAS 1.50
(nv3r4 1.72), so the tuning gate passes. But the all-mic battery is much
worse than R2: Fréchet 741 vs 311 dB², IS octave-balanced 0.51 vs 0.33,
line Δ +3.3 dB, k = 1–2 peak error 5.8 vs 2.6 dB. Held-out windows
(`nv4-score-trim-1d6809`, `nv4-pit-trim-dddfd2`): HPPNet PIT 1.808 (bar
1.680; R2 2.239), LTAS 3.061 (bar 2.384), HRSPNet 1.617 (nv3r4 1.105).
Listen did not run because the render clips (peak 1.38). Reading: the
gate on five segments does not transfer, and mic-0 LTAS is a weak stand-in
for spectral similarity.

## Conclusion

- **Bench (step 2).** Converged and at a minimum: `dregon__kappa0__r5__s2__c7`,
  κ = 0 (no κ passes the planted 1 dB floor rule).
- **η (step 3).** 0.00152 (200 Hz blocks) to 0.00132 (per recording) on the
  final fit.
- **MD2 flight.** The converged per-regime pair beats nv3r4 on parity
  (0.481 vs 0.515) and on every regime and cruise LTAS; Listen is on par
  within render-seed noise.
- **DREGON flight.** Not reached: every converged stage on the frozen bench
  lines fails the PIT bar (best 15.8 vs 1.680; LTAS 2.384 equal).
