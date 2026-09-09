# Fitting the stochastic rotor-noise family to real recordings

**Status:** in progress — 2026-09-08 → . Branch `stochastic-fit`, worktree
`.worktrees/stochastic-fit`. Code: `src/experiments/stochastic_fit/`. Textbook
explainer of the method (model, objective, references, procedure):
`docs/explainers/stochastic-fit.qmd` (render with `quarto render`; the OJS
figure needs an HTTP origin).

## Motivation

Regressors trained on the stochastic synthetic family alone
(`data_processing.stochastic_rotor_noise`, policy `conf/online_mix/salv2_stoch.yaml`)
score 8.08 rev/s all-MAE on the real panel; with real audio in the mix,
2.67 (`docs/experiments/stochastic-transfer.md`). Every previous attempt moved
one sampler range at a time and measured transfer. None helped. The question
this campaign answers first is prior to all of them: **at its best parameters
for a given real recording, how much of that recording does the family
explain, and where does it fail?** Two hypotheses with opposite fixes: wrong
ranges (retune the sampler) vs wrong family (change the model).

The instrument is a maximum-a-posteriori fit of the family's own spectral
model to a recording's eight-channel Hann-2048/512 periodogram under the
Whittle likelihood, rotor speeds held fixed to the refined references
(`pi_kalman_refine`, the 2026-09-06 linewidth-audit recipe). Everything the
sampler draws is a parameter; the GP drifts are latents with their whitened
priors. Explained fraction = (floor-only − fit) / (floor-only − LOO), where
the LOO reference is a leave-one-out periodogram smoother scored with the
analytic exponential-mean bias removed — the score a correct model reaches.

## Data

Bundle on R2 `artifacts/stochastic-fit/clips/` (manifest.json, 54 clips):
27 four-second training crops with acoustic references (17 DREGON room2, 10
FLY125; `.worktrees/jhtr-refinement/results/jhtr/trajectory-linewidth/refs`),
23 eight-second `nosource` clips of the frozen validation split (8 DREGON
room1 with measured shaft rate, 15 FLY124; stopped, spin-up, ramps and cruise;
refined on `uni-cpu`, job `stochfit-prepare-32bde4`), 4 renderer control clips
on real FLY125/room2 trajectories with their planted parameters.

## Results

### The instrument passes its controls

Four renderer clips (planted parameters, default sampler): excess over the
LOO reference −0.004 ± 0.01 nats/cell, explained 1.01–1.07; identical at the
short (150/150/300/60) and long (300/300/800/150) optimizer schedules, so the
fit is converged; residual profiles along the lines flat, half-order residual
1.00, coherence statistic 1.02–1.09 (exponential). ~15 s per clip on a Kaggle
P100 (PyPI torch 2.7 cu126 — the Kaggle image's torch has no sm_60 kernels).

Two findings about the instrument itself:

- **LOO replicates must be two hops apart.** The first reference averaged the
  ±1/±2-hop neighbours; at a quarter-window hop the adjacent Hann frame's
  periodogram is correlated with the cell (power correlation ≈ 0.43), so the
  smoother predicted every cell better than the truth could and all four
  controls showed the same +0.245 nats/cell "excess". Replicates at ±2/±4
  hops (half-window shifts, correlation ≈ 0.03) put the controls at zero;
  wider spans (±4/±6, ±4/±8) drift negative (−0.03 to −0.09) as the GP
  drift breaks stationarity. `fit.LOO_OFFSETS = (-4, -2, 2, 4)`.
- **Width-law parameters are only partly identifiable in the family
  itself.** The renderer floors every half width at 0.6 bins (4.7 Hz), so
  `gamma0` is observable only through orders where `gamma0 + slope*k`
  exceeds the floor; slopes come back roughly (0.57/0.95/0.99/0.67 vs planted
  0.30/0.70/0.72/0.68 on one control) and a rotor whose lines are buried gets
  arbitrary widths at no likelihood cost. The MAP drift curves are shrunk
  ~2× (fitted std 1.3–3.3 dB vs planted 3.8–5.3 at the 3 dB prior). The
  likelihood-level questions are unaffected; parameter readouts are
  restricted to visible lines (`diagnostics.fitted_parameter_summary`).

### The full-skirt arm: half to two thirds (superseded by the realized-family arm below)

Variant `family` (the renderer's exact parametrization, reference carriers,
no correction), Kaggle job `stochfit-family-b9f69d`, all 48 clips with
turning rotors (the two stopped-rotor clips have no comb and no defined
fraction):

| group | n | explained: median (min–max) | excess nats/cell: median (min–max) | floor−LOO (median) |
|---|---:|---|---|---:|
| DREGON room2 (crops) | 17 | 0.45 (0.37–0.54) | 0.33 (0.30–0.36) | 0.62 |
| DREGON room1 (validation) | 7 | 0.66 (0.48–0.73) | 0.29 (0.28–0.42) | 0.91 |
| FLY125 (crops) | 10 | 0.68 (0.64–0.74) | 0.31 (0.27–0.33) | 0.99 |
| FLY124 (validation) | 14 | 0.70 (0.56–0.75) | 0.26 (0.17–0.68) | 0.86 |
| renderer controls | 4 | 1.02 (1.01–1.07) | −0.006 (−0.02–0.00) | 0.30 |

The leftover ≈ 0.3 nats/cell is nearly constant within a group — a
structural misfit, not a bad draw. **Read with the correction under
"Ablations": this arm carries full Lorentzian skirts the renderer never
produces; the realized family's excess is 0.13–0.16 (explained 0.74–0.87).** On the KL scale (`r − 1 − log r`, r =
true/model mean) 0.3 nats is what a 2× under-prediction or a 2.5×
over-prediction on *every* cell would cost; direction matters, and the
diagnostics below say which cells carry it.
Per-clip numbers: `results/stochastic_fit/summary_family.json` of the job.

### Where it fails (residual `I/M` of the family fit, representative clips)

- **Low-order line shape.** Orders 1–8, residual along the line in units of
  the fitted half width: FLY125_06 R = 0.75 / 0.93 / 1.66 / **2.12** / 2.02 /
  1.34 / 0.76 at −3γ…+3γ; FLY124_31 0.86 / 1.16 / 1.45 / 1.38 / 1.20 / 0.84 /
  0.78. Real low harmonics are much sharper than the family can render: the
  renderer's 0.6-bin width floor plus bin-centre sampling cannot make a
  sub-bin tone. Orders 9–64 are flat (1 ± 0.1) — the mid-order widths are
  fine; the "widths grow too fast with order" hypothesis is not what the
  data say there.
- **Fluctuation statistics.** Normalized variance of the residual at line
  centres 1.4–3.1 at orders 1–8 and 65+ (controls 1.02–1.09): heavier than
  exponential, i.e. intermittent — consistent with the earlier flicker
  measurement (real 4–5 dB vs stochastic 3.4). Residual autocorrelation at
  lag 4 hops 0.06–0.15 at low orders (controls 0.00): amplitude variation the
  3 dB / 1.5 s GP drift does not follow.
- **Microphones** (model-free, `family` job periodograms, 200–4000 Hz;
  spread across the 8 mics in dB, median over clips):

  | group | floor proxy | line proxy | overall level | floor − level | line − level |
  |---|---:|---:|---:|---:|---:|
  | FLY125 | 12.8 | 13.0 | 12.2 | 1.2 | 1.0 |
  | FLY124 | 10.6 | 10.7 | 10.1 | 1.0 | 0.6 |
  | DREGON room1 | 4.4 | 6.8 | 10.1 | 9.2 | 3.4 |
  | DREGON room2 | 5.1 | 6.6 | 8.5 | 8.0 | 3.5 |

  On Michael's rig the microphones differ by ~12 dB in level with floor and
  lines moving *together*; the family's per-microphone gain applies to the
  lines only, over a common floor, and cannot represent this. On DREGON the
  floor and line spreads are decoupled (8–9 dB), which the family cannot
  represent either. (The fitted model's per-mic floor residual is not
  quoted: at K ≈ 266 the line masks cover 98–100 % of cells and the remainder
  is a selection-biased sliver; the `mic_floor` ablation is the evidence.)
- Half-order residual 1.05–1.17 at orders 1–8: a little sub-harmonic content.

### Ablations (jobs `stochfit-var-crops-7c047a`, `stochfit-trunc-82ef61`, `stochfit-gauss-0456f6`, `stochfit-var-valid-016a03`)

Excess over the LOO reference, nats/cell: median [IQR] over clips, and the
*paired* difference against the realized family (`family_bucket_rps` on the
crops; `family_rps` on the validation rigs, where the bucket arm was not
run) with the count of clips improved.

**The baseline is the family as rendered, not as written.** `build_psd`
renders each Lorentzian over a power-of-two bucket of ≥ 5γ half widths and
renormalizes by the fixed 87.4 %, so a realized line has no skirt beyond
5–10 γ. The `family` arm with full 1/d² skirts overstates the misfit by
about half; `family_bucket*` reproduces the realized support and is the
reference below.

Crops (27 clips: FLY125 10, DREGON room2 17), median excess and paired Δ vs `family_bucket_rps`:

| arm | one change | FLY125 | Δ (improved) | room2 | Δ (improved) |
|---|---|---:|---:|---:|---:|
| `family` | full skirts, reference carriers | 0.307 | +0.170 (0/10) | 0.333 | +0.177 (0/17) |
| `family_rps` | full skirts + carrier correction | 0.299 | +0.158 (0/10) | 0.314 | +0.163 (0/17) |
| **`family_bucket_rps`** | **the realized family** | **0.133 [0.125, 0.160]** | — | **0.161 [0.139, 0.191]** | — |
| `family_trunc_rps` | clean ±5γ cut | 0.119 | −0.014 (9/10) | 0.126 (16 finite of 17; `hovering_nosource_room2_03` non-finite on retry too) | −0.032 (15/16) |
| `integrated` | exact bin integral | 0.299 | +0.159 (0/10) | 0.315 | +0.162 (0/17) |
| `sharp` | no width floor + integral | 0.271 | +0.131 (0/10) | 0.315 | +0.164 (0/17) |
| `free_gamma` | width free per order | 0.285 | +0.153 | 0.304 | +0.153 |
| `mic_floor` | per-mic floor offset | 0.290 | +0.159 | 0.319 | +0.156 |
| `drift6` | 6 dB drift prior | 0.297 | +0.160 | 0.310 | +0.162 |
| `extended` | all of the above, Lorentzian | 0.241 | +0.102 | 0.285 | +0.138 |
| **`gauss`** | **Gaussian line, equal HWHM** | **0.069 [0.059, 0.076]** | **−0.075 [−0.084, −0.054] (9/10)** | **0.079 [0.058, 0.092]** | **−0.074 [−0.099, −0.058] (16/17)** |
| `gauss_sharp` | + no width floor | 0.066 | −0.063 (10/10) | 0.086 | −0.068 (17/17) |
| `gauss_mic` | + per-mic floor | 0.065 | −0.069 (10/10) | 0.082 | −0.079 (17/17) |
| `gauss_drift6` | + 6 dB drift prior | 0.069 | −0.061 (10/10) | 0.083 | −0.074 (16/17) |
| `gauss_free` | + free width per order | 0.057 | −0.073 (10/10) | 0.055 | −0.098 (17/17) |
| `gauss_all` | all Gaussian relaxations | 0.060 | −0.069 (10/10) | 0.058 | −0.083 (17/17) |

(`integrated` ≡ `family_rps` to three decimals on every clip: the bin-centre
sampling of the renderer costs nothing at this resolution.)

Validation rigs (`stochfit-bucket-valid-ea26eb` added the realized-family
arm), median excess and paired Δ vs `family_bucket_rps`:

| arm | DREGON room1 (7) | Δ (improved) | FLY124 (14) | Δ (improved) |
|---|---:|---:|---:|---:|
| `family_rps` (full skirts) | 0.279 [0.273, 0.290] | +0.180 (0/7) | 0.257 [0.210, 0.267] | +0.128 (1/14) |
| **`family_bucket_rps`** | **0.114 [0.086, 0.117]** | — | **0.116 [0.083, 0.137]** | — |
| `extended` (Lorentzian) | 0.247 | +0.141 (0/7) | 0.203 | +0.086 (2/14) |
| **`gauss`** | **0.102 [0.070, 0.111]** | **−0.011 [−0.018, −0.010] (7/7)** | **0.066 [0.054, 0.072]** | **−0.056 [−0.065, −0.024] (14/14)** |

The single-change arms vs the full-skirt arm (`sharp` −0.005 / −0.012,
`mic_floor` +0.003 / −0.009, `speed_law` +0.006 / −0.000) buy nothing.
Explained fraction of the realized family: 0.87 on both validation rigs;
of the Gaussian: 0.89 (room1), 0.92 (FLY124).

Renderer controls (4 Lorentzian clips): `family_bucket_rps` −0.007,
`family_trunc_rps` −0.007, `gauss` −0.007 [−0.012, −0.005], paired Δ 0.000.
The negative control is therefore **uninformative**, not passed: on the
controls' wide lines the clipped tail is small against the floor, so the
shape arms are indistinguishable there. What the controls establish is
that no arm gains from the fitting machinery itself.

Readings. (1) The realized family explains 0.87 (FLY125, room1, FLY124) /
0.74 (room2) of the way from floor-only to correct. (2) The Gaussian line
improves on the realized family on 46/48 clips, but by a rig-dependent
amount: it halves the remaining excess on FLY125 (0.13 → 0.07), room2
(0.16 → 0.08) and FLY124 (0.12 → 0.07), while on DREGON room1 it buys only
−0.011 (0.114 → 0.102, 7/7) — room1's misfit is mostly *not* the line
shape. (FLY125_00 is one of the two non-improving clips, 0.078 vs 0.077.) (3) Within
the Gaussian family a free width per order buys another 0.01–0.02 on
room2 (17/17) and 0.01 on FLY125; sub-bin widths, per-mic floor and a
looser drift prior buy nothing beyond the Gaussian. (4) The speed law, on
the ramp-containing validation clips, buys nothing. (5) What is left — 0.055–0.07 nats/cell on FLY125, FLY124 and room2
(the Gaussian combinations ran on the crops only), 0.10 on room1 — is not
reachable by any spectral-shape relaxation tested; it is where the
second-order findings below point.

### Phase statistics along the lines (tested estimator, `phase_stats.py`)

Normalized complex cross-correlation of the STFT evaluated at each line's
exact centre (frame-start reference; reading the nearest bin flips the phase
by π at every bin change — the first ad-hoc pass did that and its numbers
are superseded), carrier phase advance removed, lags in hops of 32 ms;
isolated lines, runs ≥ 8 frames, pooled per band; the phase-scrambled null
beside it. A tone with Wiener phase (Lorentzian line) follows the
window-smoothed autocorrelation (`lorentzian_lag_prediction`, pinned by
`tests/experiments/test_stochastic_fit_phase_stats.py` together with the
white-noise overlap curve, the tone-in-noise plateau and a bin-crossing
chirp); a coherent tone plus noise plateaus at the tone's share of the bin
power.

| orders 1–8 | lag 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | null (lags ≥ 3) |
|---|---|---|---|---|---|---|---|---|---|
| FLY125 (8 clips) | 0.78 | 0.51 | 0.41 | 0.37 | 0.37 | 0.35 | 0.28 | 0.33 | 0.07–0.08 |
| FLY124 (7) | 0.77 | 0.47 | 0.41 | 0.38 | 0.33 | 0.33 | 0.33 | 0.34 | 0.07–0.10 |
| DREGON room2 (5) | 0.65 | 0.26 | 0.24 | 0.24 | 0.18 | 0.20 | 0.23 | 0.13 | 0.09–0.19 |
| renderer controls (2) | 0.66 | 0.21 | 0.16 | 0.15 | 0.17 | 0.24 | 0.09 | 0.09 | 0.08–0.16 |
| Lorentzian at the fitted γ (4.7 Hz) | 0.79 | 0.40 | 0.15 | 0.05 | 0.01 | 0 | 0 | 0 | |

Orders 9–24: DREGON room1 0.69 / 0.43 / 0.39 / 0.41 / 0.50 / 0.35 / 0.32 /
0.35 against a null of 0.62 / 0.24 / 0.14 / 0.17 / 0.25 / 0.13 / 0.20 /
0.22 (few isolated runs, hence the high null); Michael's 0.3–0.5 with
nulls 0.13–0.26; no band above 24 has enough isolated runs.

The null is white noise through the same window, frames and demodulation:
successive lagged products of a 75 %-overlap STFT are correlated, so a short
run's |Σ| is inflated beyond the iid value; a phase-scrambled null would
under-read that by ~30 %, which is why the null is matched white noise
(the first version of this estimator, and its 0.6 plateaus, are
superseded; that version also took |Σ| per run and added, a positive bias
growing with the number of short runs).

Reading: on both Michael's flights the low harmonics carry a coherent
component — a plateau of 0.33–0.37 out to half a second, four times the
null, where a Lorentzian of the fitted width is at zero by lag 6; the
tone's share of the bin power is of that order (the plateau is a lower
bound: any carrier error decorrelates long lags). DREGON room2's low
harmonics sit 0.05–0.1 above their null — at most a marginal coherent
share; room1's mid orders 0.1–0.25 above theirs. The renderer controls sit
at their null, as they must.

Centre regressions of `log R` on the rotor's in-clip speed deviation:
slopes −0.09…+0.15 nats per 1 % on every group/band, |corr| ≤ 0.09; on the
sub-bin offset +0.1…+0.4 with the same sign and size on the controls
(the model's bin-centre sampling). Neither explains the normalized
variance of 1.3–4; the in-clip speed range (±1–2 %) has little leverage on
the across-speed law, which the `speed_law` arm on the ramp clips tests.

### Interpretation: the phase-increment model, two regimes

The tracker's generative model (`phase_increment_tracker.py`, WP18 of
`rps-refine-precision.md`) is `θ_k = k·φ + b_k + ψ_k` with `φ` the shaft
phase (`r_label + δr`, the ~0.6 rev/s label-invisible jitter) and `b_k` a
per-harmonic phase diffusion (bench: HWHM 0.042 Hz/order, i.e.
`q_k = 4π·0.042·k` rad²/s). A tone with Wiener phase has an exactly
Lorentzian line (HWHM = q/4π) — the family's Cauchy shape is the transform
of `b_k`, not an extra assumption. But the *shaft* term is band-limited
(τ_c ≳ a frame): a tone whose frequency wanders slowly has a line equal to
the distribution of its instantaneous frequency — Gaussian, width ∝ k·σ_r
— the quasi-static FM regime; only when the phase variance accumulates
linearly (τ ≫ τ_c, or true diffusion) does the shape become Lorentzian
(width ∝ k²σ_r²τ_c). The general line is their convolution (Voigt). The
`gauss` result says real harmonics sit closer to the quasi-static regime at
this front end than the family's lines do: the realized family already
clips its skirts at 5–10 γ, and the Gaussian's faster fall-off still halves
the remaining excess. The per-harmonic diffusion is the only Lorentzian
left and is sub-bin below k ≈ 60.

WP18's caveats carry over: the rank-one (shaft/arrival-time) plus diagonal
covariance of the rate opinions explained only 12–27 % of the off-diagonal
energy and the common term was unresolved on DREGON / marginal on Michael's
and predominantly per-microphone — so the three-term decomposition is the
coordinate system for the generative model, not a validated covariance;
its loadings are to be fitted, not assumed.

## Conclusion (provisional — no modified renderer exists yet)

Answer to "wrong ranges or wrong family": the realized family already
explains three quarters (room2) to seven eighths (FLY125, room1, FLY124)
of what a correct spectral model would; what it misses is
structural, not a range. Two structural facts are established by paired
ablations and second-order statistics: (1) the harmonic line falls off
faster than the family's clipped Lorentzian — a Gaussian of equal half
width improves on the realized family on 46 of 48 clips, halving the
remaining misfit on FLY125, FLY124 and room2 but buying only 0.01 of
room1's 0.11; a free width per order adds a little on DREGON; (2) Michael's low harmonics are partly coherent tones
(plateau 0.33–0.37 vs null 0.08), DREGON's are not, and real amplitudes
fluctuate more than exponentially everywhere. Neither sub-bin widths, nor
per-microphone floors, nor a looser drift prior, nor the speed law, nor
bin integration moves the fit.

Physically (§ Interpretation) this is the quasi-static FM regime: coherent
tones on one slowly wandering shaft, whose within-frame frequency
distribution is Gaussian with width ∝ k, over a small per-harmonic
diffusion; the family renders independently phase-diffusing narrowband
noise per harmonic. The candidate generative change is therefore a
shared-shaft FM tone bank (`δr_ρ(t)` per rotor, `x_ρk = A_ρk cos(k φ_ρ +
b_ρk + ψ_ρk)`), with a coherent share that is a *range including zero*
(DREGON), a heavier-tailed / faster amplitude process, and a per-microphone
overall gain.

What this campaign has **not** yet done: implement that renderer, read its
ranges off the fitted distributions, and pass a three-part gate:

1. *Calibration.* The new renderer's samples on the real trajectories,
   refitted with the model that describes the new renderer, must sit at
   excess ≈ 0 — as every correctly specified control has so far. A
   non-zero control excess means the model, the reference or the renderer
   is wrong; it is never a realism target.
2. *Real-data misfit.* The extended model fitted to the 48 real clips must
   move from the realized family's 0.13–0.16 toward its own control-
   calibrated zero. The Gaussian arm's 0.055–0.07 is where that stands.
3. *Held-out summary distributions.* The renderer's draws must land inside
   the real clips' spread, per rig, on statistics the fit does not use:
   coherence-vs-lag per order band, amplitude normalized variance and
   residual ACF, per-microphone level/floor covariance, line profiles.

Then the transfer test (retrain S2, score the real panel). Until then the
claim "samples statistically more similar to DREGON / MD2" is a
prediction, not a result.

## Gotchas

- Kaggle output collection through omnirun fails silently above ~2 GB (the
  job shows `succeeded`, `omnirun pull` returns nothing, and the kernel may
  still be RUNNING on Kaggle when omnirun says done). Check with
  `uvx --from kaggle kaggle kernels status/output <ref>`; result files are now
  slim (float16 dB spectrum, periodogram rebuilt offline from the R2 clip).
- Kaggle's image torch has no sm_60 kernels; the slim snapshot uses a uv env
  with PyPI `torch==2.7.0` (`scripts/kaggle_slim_stochfit.sh`, branch
  `kaggle-slim-stochfit`, worktree `.worktrees/kaggle-slim-stochfit`).
