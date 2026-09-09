# One probabilistic rotor-noise model per rig: plan

**Date**: 2026-09-09 · **Branch**: `stochastic-fit` · **Status**: PROPOSED
(awaiting the user's reordering) · Evidence base:
`docs/experiments/stochastic-fit.md` (the fitting campaign),
`writing/reports/2026-07-18_dregon-analysis-and-generator-design` (per-rotor
identity, geometry), `docs/experiments/residual-attribution.md` (mic
structure of the floor), `docs/experiments/wind-channel-likelihood.md`
(the refuted wake gate), Bretthorst (1988).

## 1. The question

Fit, to DREGON and to Michael's rig separately, one generative model with
**rig-level fundamental parameters**, **per-rotor variation**, and
**per-clip nuisance**, such that (a) held-out real clips are explained as
well as by fully free per-clip fits, (b) the model's own samples calibrate
at zero misfit, (c) the samples fall inside the real clips' spread on
statistics the fit never sees. "Explains real data really well" is those
three numbers, not a visual.

Why this and not more per-clip fitting: the independent fits already
explain 0.74–0.87 of a correct spectral model, and their *scatter* across
clips of one rig is uninterpretable — a mixture of true variation,
non-identifiability (width law vs carrier correction) and optimizer
trade-offs (`stochastic-fit.md` § across-clip consistency). Bretthorst's
rule for this situation (p. 39): parameters that come out the same across
repeated experiments are real, those that differ are artefacts of fitting
the noise and must be removed from the model. A tied fit with held-out
predictive checks is that rule made quantitative.

## 2. What the data have said (compressed)

Spectral misfit of the realized family (excess over the correct-model
reference, nats/cell): FLY125 0.133, room2 0.161, room1 0.114, FLY124
0.116. Established structure, each by paired ablation or a tested
second-order statistic:

| fact | evidence | consequence for the model |
|---|---|---|
| Lines fall off faster than a clipped Lorentzian; Gaussian of equal HWHM wins on 46/48 clips; free per-order width adds a little on DREGON | `stochastic-fit.md` § ablations | Gaussian (quasi-static FM) line, width ∝ k·σ_r, over a per-harmonic Lorentzian diffusion (bench 0.042 Hz/order) |
| room1 (measured shaft rate) gains only 0.01 from the Gaussian | same | room1's remaining 0.10 is not line shape: amplitude process, microphone structure, floor |
| Michael's low orders partly coherent (0.33–0.37 vs null 0.08, flat to 0.5 s); DREGON ≤ marginal | § phase statistics | coherent share as a rig-level *range including zero* |
| amplitude normalized variance 1.4–3.1 at k ≤ 8 and k ≥ 65; drift τ pinned at the knot floor (0.5–0.8 s) on every rig | § where it fails, § consistency | faster and heavier-tailed amplitude process than the 3 dB / 1.5 s GP |
| Michael's mics differ by ~12 dB with floor and lines moving together; DREGON's floor and line spreads decoupled by 8–9 dB | § where it fails | Michael's: one per-mic gain on everything; DREGON: per-mic floor separate from per-(mic, rotor) line gains |
| DREGON 50–500 Hz floor: std 3.3–3.4 dB (null 0.55), lag-1 between the 0.1 s and 0.5 s controls, cross-mic corr 0.06, no resolved heavy tail; low-band line cells co-move with it (0.7–0.8); > 500 Hz and Michael’s: common (0.7–0.9), σ 1–2 dB | § floor envelope (calibrated) | per-mic *independent* log-OU floor modulation in the low band on DREGON; a common slow term everywhere |
| four-motor bench floor exceeds the sum of single motors by +10.9 dB at 100–250 Hz; per-rotor low-band shares unidentifiable in flight (VIF 21–118) | `residual-attribution.md` | the floor is one rig process, not four rotor sources; do not fit per-rotor floors |
| physics-gated wake channel: gate predicts per-mic bench floor at ρ = 0.92 but uniform exposure beats it in training | `wind-channel-likelihood.md` | keep the floor's per-mic *statistics*, drop the spatial gate |
| bench rotors: rotor-specific profile 5.1 of 21.4 dB² of centred line variance, same on all mics (rotor×mic 1.9) and speeds; Motor2's low orders +3 dB richer | § bench (new) | per-rotor profile deviation δ_rk, constant across speed and mic, partially pooled |
| flight fits: per-rotor profile stable to 2–4 dB at k ≥ 9, 4–11 dB at k ≤ 8; rotors' profiles correlate 0.4–0.7 within a clip; rotor overall level wanders 2–6 dB clip to clip | § consistency | δ_rk is a rotor constant; low orders need the coherent/bursty terms; per-clip per-rotor level |
| tilt −5.5…−6.6 dB/oct ±0.7 on every rig; roll-off DREGON ≈ 1.0–1.2 vs Michael's 0.22–0.27; Michael's floor 26–36 dB under the lines | § consistency | rig constants; the sampler's ranges (tilt −9…−1, roll-off ≥ 0.4, floor ≥ −22) do not contain the data |
| width law and γ₀ scatter 3–10× between slices, trading against the carrier correction (1.5–3.3 rev/s rms) | § consistency | fit σ_r pooled per rig with telemetry carriers; the carrier correction becomes a tight clip nuisance |

## 3. The model

Levels: rig θ, rotor δ_r (r = 1..4), clip η_c; within a clip, microphone
m, frame t, frequency f. Expected periodogram

```
S_cm(t,f) = G_m · [ D_c(t,f) + Σ_r g_mr Σ_k P_crk(t) · L_rk(f − k·φ_cr(t)) ⊛ W ]
```

- **Floor** `D_c(t,f) = D_rig(f) · s̄_c(t)^p_floor · exp(b_c(t)) · exp(u_cm(t)·1[f < f_u])`:
  rig shape `D_rig(f)` (tilt + 14-knot log-frequency curve, rig level),
  the speed law, a common slow modulation `b_c` (GP, σ_b, τ_b rig-level),
  and — DREGON — a per-microphone independent log-OU `u_cm` (σ_u ≈ 3.5–4 dB,
  τ_u ≈ 0.1 s, corner `f_u` ≈ 500 Hz, all rig-level; Michael's σ_u → 0 by
  evidence). Per-mic floor gain `d_m` (rig) on DREGON.
- **Lines** `10 log P_crk(t) = A_k^rig + δ_rk + ℓ_cr + h_crk(t) + p_amp·10 log(r_cr(t)/80)`:
  rig profile `A_k` (roll-off + jitter pattern), rotor deviation
  `δ_rk ~ N(0, τ_δ²)` (partial pooling, τ_δ rig-level, expected ≈ 5 dB²
  from the bench), per-clip per-rotor level `ℓ_cr`, and the amplitude
  process `h_crk(t)`: Matérn-½ (OU) rather than squared-exponential, σ_h
  and τ_h rig-level, with a Student-t innovation option tested by
  evidence (the floor's tails are Gaussian; the lines' may not be).
- **Line shape** `L_rk`: Gaussian of width `k·σ_r` (quasi-static FM of one
  shaft, σ_r rig-level; the sampler realizes it as an OU `δr_ρ(t)` shared
  by a rotor's harmonics) convolved with a Lorentzian of HWHM `q·k` (per-
  harmonic diffusion, rig-level, prior at the bench 0.042 Hz/order), then
  the analysis window `W`. Per-rotor width scale `w_r` is a *tested*
  addition (M3 below), not assumed.
- **Carrier** `φ_cr(t) = r_cr^label(t) + o_cr(t)`: the clip's telemetry or
  refined rate plus a tight, slow correction (prior std 0.3 rev/s, 0.5 s
  knots) — a nuisance, marginalized in the sense of Bretthorst § 2.3, not
  a free width-law competitor.
- **Microphones**: Michael's — one gain `G_m` (rig) on everything;
  DREGON — `G_m = 1`, per-mic floor `d_m` and per-(mic, rotor) line gains
  `g_mr` (rig; the residual-attribution spatial basis, cosine 0.95–1.00
  across throttle). Cross-mic *coherence* of the lines is not in this
  likelihood (see § 4).
- **Coherent share** (Michael's low orders): in a per-cell Whittle
  likelihood a partly coherent tone is a non-exponential cell; two options
  to be decided by evidence, not now: a Rician cell likelihood with share
  `c_rk` (rig, range including zero), or Whittle throughout with the share
  read only from the held-out coherence statistic. Start with the second.

Rig-level: `D_rig(f), d_m, G_m, g_mr, p_floor, p_amp, σ_b, τ_b, σ_u, τ_u,
f_u, A_k, τ_δ, σ_h, τ_h, σ_r, q, (c_k)`. Rotor-level: `δ_rk, (w_r)`.
Clip-level: `s̄_c, b_c(t), u_cm(t), ℓ_cr, h_crk(t), o_cr(t)`.

## 4. Likelihood and fitting

**Likelihood.** Whittle per cell and microphone, on the fixed 2048/512
Hann grid (`sum I/M + log M`), with the bias-corrected leave-one-out
reference as the zero (`fit.py::loo_reference`, ±2/±4-hop replicates).
This is Bretthorst's construction with the deterministic sinusoid replaced
by a second-order stochastic line: the periodogram is the sufficient
statistic *only* under his six conditions (p. 20 — single stationary
frequency, white noise); random-phase finite-width lines and a colored
floor violate them, and Whittle's spectral likelihood (which he cites,
p. 110) is the correct replacement. What transfers unchanged: Gaussian
noise as the maximum-entropy, conservative choice (§ 2.2); nuisance
parameters removed by integration or, where the data determine them, by
estimation (§ 2.3, App. C); the residual-versus-expected diagnostic
(§ 3.6: `N d̄² − m h̄²` against `(N−m)σ²` is exactly our excess over the
LOO reference); and the **joint analysis of multiple measurements**
(§ 7.5): shared parameters across records with per-record amplitudes,
phases and noise levels beats averaging by orders of magnitude and stays
correct when the records differ in their nuisances. Our rig fit is § 7.5
with clips as records.

Multichannel Whittle (an `M×M` cross-spectral matrix per cell: rank-1 per
rotor line term over a diagonal floor) is the principled extension and
the only way to *fit* line coherence across microphones. Deferred: the
floor's incoherence makes the per-mic diagonal likelihood already right
for the floor, and the rig-level spectral parameters do not need cross-mic
phase. It enters if the held-out spatial statistics (§ 7) fail.

**Procedure** (one job per rig, GPU):

1. Warm start every rig-level parameter at the median of the independent
   fits; clip nuisances at their per-clip values; `δ_rk = 0`.
2. Joint MAP over all training clips of the rig with the shared
   parameters tied; the hyperpriors (`τ_δ, σ_h, τ_h, σ_u, τ_u`) fitted by
   type-II ML (the whitened-knot prior already in `CombSpectrum.prior`
   generalizes; the OU kernel replaces `se_cholesky` for `h` and `u`).
3. Score: excess over LOO on the training clips; then **held-out**: freeze
   rig and rotor parameters, refit only clip nuisances on clips of a
   recording the rig fit never saw (DREGON: fit room2 crops → test room1
   validation clips and vice versa; Michael's: FLY125 → FLY124 and vice
   versa), report predictive excess vs the independent fits' excess.
4. Model ladder, each step admitted by held-out ΔNLL and by the evidence
   approximation of Bretthorst ch. 5 (Laplace at the mode with proper
   priors — the Ockham factor of the added parameters; the rule of thumb
   p. 65: a parameter earns its place when its projection is ≳ 3σ):

   | step | adds | question |
   |---|---|---|
   | M0 | — | the independent fits (current numbers) |
   | M1 | tie every rig-level parameter; clip nuisances free | does one rig model suffice? |
   | M2 | `δ_rk` partially pooled | are rotor profiles real in flight? (bench says yes) |
   | M3 | per-rotor width scale `w_r`, per-rotor level | do rotors differ beyond their profile? |
   | M4 | per-mic independent floor OU `u_cm` (DREGON), `d_m`, `g_mr` | does the floor process close room1's gap? |
   | M5 | Gaussian⊛Lorentz line with pooled `σ_r`, `q`; tight carrier | is the width law identifiable once pooled? |
   | M6 | OU / Student-t amplitude process | does the amplitude statistic move? |

   Expected: M1 costs little at k ≥ 9 and something at k ≤ 8; M2 recovers
   ~5 dB² of it; M4 is room1's lever; M5 is the Gaussian's 0.05–0.07.
5. **Reproducibility test** (Bretthorst p. 39): fit each recording of a
   rig separately at the M5 level and compare the rig-level estimates;
   those that agree are rig constants, those that do not are demoted to
   clip level. This is the criterion for the final parameter split, not
   the table in § 3.

**Bench as an independent calibration experiment.** The 20 single-motor
files are the same model with `R = 1`, known rate, no collinearity: fit
`δ_rk`, `g_mr`, the width law per rotor. The flight-fitted `δ_rk` must
agree with the bench `δ_rk` (a rotor property survives the change of
experiment); the floor must *not* be transferred (four motors are +10.9 dB
over the sum of singles at 100–250 Hz; bench airframe state differs).

## 5. The gust / floor process

Measured (this campaign, calibrated instrument): DREGON's low-band floor
carries a per-microphone *independent* modulation, σ ≈ 3.3 dB net, with a
correlation time between 0.1 and 0.5 s, no resolved heavy tail, and the
low-band line cells at the same microphone move with it (0.7–0.8); above
500 Hz and on Michael's the modulation is common across mics and 1–2 dB.
No multi-second gusts in the 4–8 s clips. The process parameters (σ_u,
τ_u, whether it multiplies lines as well as floor) are M4's to fit, not
these observations'.
Open measurement before M4 is frozen: run `floor_envelope` on the full
DREGON flights (41–82 s, `DREGON_free-flight_nosource_room{1,2}`,
`hovering_nosource_room2`) to see whether the many-second, order-of-
magnitude bursts the tracker's `adaptive_floor` was written for exist,
and with what rate and duration; if they do, `u_cm` gets a second, slow
component (or a two-state envelope). The refuted physical wake gate is
not revived; the per-mic loading `d_m` is fitted, not predicted.

## 6. Sampler consequences

Every rig-level parameter becomes a per-rig *distribution* read off the
fit (posterior mode ± its Laplace width, plus the between-recording spread
from § 4.5), replacing the hand ranges in `StochasticRanges`. Structural
changes to `stochastic_rotor_noise`: `line_mode: "fm"` (shared-shaft OU
per rotor, Gaussian⊛Lorentz lines), per-rotor profile deviation with
`τ_δ` in place of `rotor_similarity`, OU amplitude process, per-mic
independent low-band floor OU (DREGON preset), per-mic overall gain on
everything (Michael's preset), fixed per-rig microphone patterns `d_m`,
`g_mr` instead of a fresh uniform draw per clip. Two rig presets, one
family.

## 7. Gates (unchanged from `stochastic-fit.md` § Conclusion)

1. Calibration: samples of the new renderer on the real trajectories,
   refitted with the model that describes the renderer → excess ≈ 0.
2. Real-data misfit: the rig model on held-out real clips, from
   0.11–0.16 toward zero.
3. Held-out summary distributions per rig: coherence-vs-lag per order band,
   amplitude normalized variance and residual ACF, floor-envelope
   statistics (σ, τ, cross-mic correlation, tails), per-mic level/floor
   covariance, line profiles and their per-rotor spread — the renderer's
   draws inside the real spread.

Then S2 retrain and the real panel.

## 8. Work packages

| WP | what | compute | output |
|---|---|---|---|
| 1 | floor statistics on the full DREGON flights (§ 5) | laptop, minutes | numbers into `stochastic-fit.md`; M4's shape |
| 2 | `fit_rig`: tied parameters across a clip batch, OU kernels, type-II hyperpriors, held-out scoring; controls (render a rig from planted rig parameters, recover them) | Kaggle P100 | calibrated instrument |
| 3 | ladder M1–M6 on both rigs, both held-out directions; evidence table | Kaggle, ~1 h per rig per step | the parameter split, the rig tables |
| 4 | bench calibration fit (`R = 1`) and the reproducibility comparison with WP3's `δ_rk`, `g_mr`, width law | Kaggle | rotor-level constants with two independent estimates |
| 5 | renderer changes (§ 6) and the rig presets | local | `line_mode: "fm"`, presets |
| 6 | gates 1–3 | Kaggle + laptop | the record's verdict |
| 7 | S2 unified retrain on the new stream; real-panel score | Vast | transfer answer |

Order: 1 ∥ 2 → 3 → 4 (bench can start after 2) → 5 → 6 → 7.

## 9. Risks

- Width law identifiability: even pooled, `σ_r` and `q` may trade with the
  carrier prior on refined (audio-derived) DREGON room2 carriers; room1's
  measured shaft rates and Michael's telemetry are the anchor — report
  `σ_r` from those first.
- Low orders: the coherent tone and the fast amplitude process both live
  at k ≤ 8; a Whittle fit cannot separate them — the held-out coherence
  statistic, not the fit, adjudicates the share.
- Compute: one clip fits in ~15 s on a P100; a rig batch of 24–27 clips
  with shared parameters is one process, minutes per ladder step; Kaggle's
  2 GB output limit forces the slim result format already in place.
- Bench → flight: rotor profiles transfer (source property), floors do not;
  do not let the bench floor enter the rig prior.
