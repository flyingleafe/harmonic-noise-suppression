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

### Across-clip consistency of the independent fits (realized-family arm, visible lines)

Every clip was fitted separately; nothing was tied across the clips of a
rig, and `profile_db`, `gamma0`/`slope` and `mic_gain_db` carry **no
prior** (`CombSpectrum.prior` regularizes only the GP knots and the carrier
correction). The scatter below therefore mixes three things it cannot
separate: genuine slice-to-slice variation, non-identifiability (widths
trade off against the carrier correction), and optimizer trade-offs. It is
descriptive; the question "does one rig-level model suffice?" is answered
only by the tied fit described at the end of this section. Median [10th,
90th percentile] across clips; the sampler's range beside it:

| parameter | room1 (7) | room2 (17) | FLY124 (14) | FLY125 (10) | sampler |
|---|---|---|---|---|---|
| floor tilt, dB/oct | −6.6 [−6.8, −5.3] | −6.1 [−7.2, −5.2] | −5.5 [−6.4, −4.8] | −5.9 [−6.2, −5.5] | −9…−1 |
| harmonic jitter, dB | 5.2 [4.9, 5.9] | 5.8 [4.6, 7.1] | 5.5 [4.4, 7.0] | 6.8 [6.2, 7.7] | 2–8 |
| drift std, dB | 4.3 [3.6, 5.7] | 3.7 [2.7, 4.5] | 3.5 [2.7, 5.3] | 2.6 [1.5, 3.3] | 0.5–6 |
| drift τ, s | 0.8 (at the knot floor) | 0.5 | 0.8 | 0.5 | 0.3–6 |
| drift common share | 0.02 | 0.02 | 0.02 | 0.01 | 0–1 |
| roll-off p | 0.98 [0.84, 1.04] | 1.20 [0.97, 1.52] | **0.27 [0.13, 0.57]** | **0.22 [−0.06, 0.46]** | 0.4–1.9 |
| floor under median line, dB | −15 [−29, −10] | −18 [−26, −10] | **−26 [−29, −21]** | **−36 [−44, −30]** | −22…−2 |
| floor shape std, dB | 10.5 [7.5, 25] | 7.9 [6.5, 9.8] | 15.5 [7.0, 22] | 17.0 [9.0, 19.5] | 2–9 |
| mic gain spread, dB | 11.7 | 9.2 | 12.0 | 12.7 | ≤ 12 |
| mic-gain pattern: corr. between clips / per-mic mean spread / its clip-to-clip std | 0.28 / 6.3 / 0.7 | 0.39 / 5.7 / 0.9 | **0.76 / 10.1 / 0.7** | **0.76 / 11.5 / 0.6** | fresh draw per clip |
| width slope, Hz/order | 0.44 [0.27, 0.70] | 0.75 [0.42, 2.1] | 0.79 [0.21, 1.4] | 0.59 [0.08, 0.78] | 0.05–0.8 |
| γ₀, Hz | 5.8 [4.3, 12.8] | 7.9 [2.4, 15] | 3.3 [0.1, 5.8] | 2.3 [0.0, 4.4] | 0.5–4 |
| carrier correction rms, rev/s | 1.6 [1.0, 2.8] | 1.6 [1.0, 3.5] | 3.3 [1.1, 4.0] | 1.7 [0.4, 2.8] | — |

Three classes. *Stable rig properties*: tilt (±0.7 dB on every rig; the
sampler's range is far wider than the data), jitter, drift std, the
absence of a common per-rotor drift (0.02 vs the sampler's 0–1), and on
Michael's rig the microphone pattern — the (mic × rotor) gain matrices of
different clips correlate 0.76 with a 10–12 dB ring spread and 0.6–0.7 dB
clip-to-clip scatter: a fixed per-rig microphone pattern, not a fresh
uniform draw. *Rig-specific but consistent*: roll-off (Michael's combs are
flat in order, p ≈ 0.2–0.3, below the sampler's floor) and the floor's
distance under the lines (Michael's −26 to −36 dB, outside the sampler's
range). *Not identifiable per clip*: the width law (10–90 % spans 0.2–2
Hz/order; widths trade off against the carrier correction, itself
1.5–3.3 rev/s rms and unstable) — a σ_r for the sampler must come from a
pooled per-rig fit with telemetry carriers, not from clip-level slopes.

Drift τ sits at the knot floor on every rig: the data want faster
amplitude variation than the 1.5 s prior allows, consistent with the
normalized-variance finding. The `gauss` arm shows the same stability
classes, except that its γ₀ on DREGON (13–19 Hz) and its floor level on
the crops absorb model mismatch at buried lines and are not physical.

The test this calls for (not yet run): a **tied fit** per rig — tilt,
microphone pattern, profile shape (roll-off, jitter pattern) and width law
shared across the rig's clips; line-level mean, drifts, floor level and
carrier correction free per clip — scored by the **predictive ΔNLL of
held-out slices** against the fully independent fit. Tying that costs ≈ 0
nats/cell on held-out clips means one rig-level model suffices and its
values are the sampler's; a cost that is a substantial fraction of the
0.11–0.16 excess means the slices genuinely differ and the sampler needs
within-rig ranges of that size. Parameter IQRs alone cannot decide this.

### DREGON bench: the four rotors' timbres are similar but distinct, as a source property (`bench.py`)

Revision of the 2026-07-18 report
(`writing/reports/2026-07-18_dregon-analysis-and-generator-design`, § "Are
the four rotors one source?"), which read one microphone (the nearest) at
nominal 70 Hz, twelve harmonics normalised to the fundamental, and found a
6.8 dB RMS inter-rotor spread that motivated the generator's per-rotor
sub-embeddings `z_r = z_drone + δz_r`. That figure used the nominal 70 Hz
grid while the acoustic rates are 67.7–69.5 rev/s (its own panel shows
rotor 2's lines off the grid from k = 3), and the fundamental it normalised
by sits in the wind band. Here: all 20 `Motor{1-4}_{50..90}` files, 20 s
each, Welch 2¹⁶ (0.67 Hz), comb refined per file (acoustic rates of the
forensics doc ±3 %), line = peak within ±1.5 Hz of `k f0`, per-mic level
removed (median over k ≤ 48) before any comparison.

| comparison (k = 2–48, centred) | RMS dB | correlation |
|---|---|---|
| same rotor, two microphones (median over mic pairs, per speed) | 4.0–4.4 | 0.90 |
| same rotor, two speeds (mic-averaged profile) | 4.0–5.0 | 0.84–0.92 |
| **two rotors, same microphone** (median over pairs, per speed) | **5.4–6.6** | **0.76–0.83** |
| two rotors, mic-averaged | 2.9–6.4 (pair 3–4 closest, 2–3 farthest) | 0.74–0.94 |

Variance decomposition of the centred line level `L[rotor, speed, mic, k]`
(21.4 dB² total after removing the common k-profile): rotor main effect
5.1, speed 4.4, rotor×speed 3.8, rotor×mic 1.9, mic main 1.0, remainder 5.2.
The rotor-specific profile is a *source* property — it is the same on all
eight microphones (rotor×mic 1.9 vs rotor 5.1) and largely the same at all
five speeds — and it is about a quarter of the centred variance; the
band means show it: k2–8 / k9–24 / k25–48 = +10.6/+2.1/−4.7 (Motor1),
+13.2/+3.8/−4.0 (Motor2), +9.1/+5.9/−4.0 (Motor3), +8.4/+4.4/−4.4 (Motor4)
dB. Motor2's richer low-order comb, the July report's headline, survives;
its 6.8 dB overstated the spread by the grid error. The per-rotor
microphone pattern is separately known to be stable across throttle
(cosine 0.95–1.00, `residual-attribution.md`). Together with the flight
fits (§ across-clip consistency: k ≥ 9 profile stable per rotor to 2–4 dB
across slices) this fixes the hierarchy for the sampler: one rig-level
profile, a per-rotor deviation of ~5 dB² that is constant across speed and
microphone, a per-clip level.

### DREGON's low-band floor: per-microphone-independent modulation; Michael's: common (`floor_envelope`)

No gust statistics existed (the wind-channel work measured the spatial
gate; the tracker's `adaptive_floor` absorbs bursts without measuring
them). Instrument: per frame and microphone, the median periodogram over
the *floor cells* of a band (±5 Hz around every `k r` of every rotor
excluded), dB re the clip median. **Calibration** (advisory 2026-09-09):
on the 2048/512 training grid four rotors near 80 rev/s leave no floor
cell below 500 Hz — the first version fell back to the whole band in
98–100 % of frames, so its 50–500 Hz numbers were band envelopes with the
lines in; discarded. On an 8192/1024 grid (1.95 Hz, 64 ms hop) floor
cells are 45–58 % of the band, fallback 0 %. Controls through the exact
estimator: white noise std 0.55 dB, p5/p95 ±0.9, lag-1 0.68 (the 87.5 %
overlap alone), cross-mic −0.04; planted independent log-OU of 4 dB reads
2.8 / 3.4 / 2.6 dB at τ = 0.1 / 0.5 / 2 s with lag-1 0.88 / 0.95 / 0.96 and
cross-mic ≈ 0; a common 4 dB OU reads cross-mic 0.79. So σ is recovered
to within ~25 %, the independent/common distinction is clean, τ is
bracketed only coarsely by lag-1 and run length. 48 real clips, medians:

| group | band | std | p5 / p95 | frames > +6 dB | run med / p90 (64 ms frames) | cross-mic corr | PC1 | lag-1 | floor–line corr |
|---|---|---:|---|---:|---|---:|---:|---:|---:|
| DREGON room1 | 50–500 Hz | 3.3 | −5.9 / +5.2 | 3.2 % | 4 / 7 | **0.06** | 0.58 | 0.93 | 0.70 |
| DREGON room2 | 50–500 Hz | 3.4 | −5.3 / +6.0 | 5.0 % | 2 / 4 | **0.06** | 0.46 | 0.90 | 0.79 |
| DREGON room1 / room2 | 500–2000 Hz | 1.8 / 1.4 | ±2–3 | 0 | – | 0.80 / 0.74 | 0.78 / 0.70 | 0.92 / 0.86 | 0.40 / 0.72 |
| FLY124 | 50–500 Hz | 2.3 | −3.1 / +4.0 | 0.5 % | 1 / 1 | 0.91 | 0.91 | 0.92 | 0.49 |
| FLY125 | 50–500 Hz | 1.7 | −2.4 / +2.2 | 0 | – | 0.79 | 0.78 | 0.81 | 0.00 |
| FLY124 / FLY125 | 500–2000 Hz | 1.8 / 1.2 | ±2–3 | 0 | – | 0.94 / 0.89 | 0.94 / 0.89 | 0.94 / 0.90 | 0.78 / 0.61 |

Readings (band-envelope observations; the process parameters are for the
hierarchical fit to estimate). (1) DREGON's 50–500 Hz floor fluctuates
with std 3.3–3.4 dB (null 0.55) and its per-mic envelopes are
**uncorrelated across microphones** (0.06; the same rig above 500 Hz is
0.74–0.80, Michael's 0.8–0.9 in both bands) — the time-domain counterpart
of the residual-attribution MSC finding (0.014 at 50–200 Hz): local flow
noise at each diaphragm. (2) Time scale: lag-1 0.90–0.93 sits between the
τ = 0.1 s (0.88) and 0.5 s (0.95) controls; excursions above +6 dB last
2–7 frames (0.13–0.45 s); no multi-second bursts in these 4–8 s clips —
whether the tracker's many-second gusts exist needs the 41–82 s flights.
(3) Tails: p5/p95 ≈ ±1.6–1.8 σ, 3–5 % of frames above +6 dB — consistent
with log-Gaussian; no heavy tail resolved. (4) **The low-band line cells
co-move with the floor at the same microphone on DREGON (0.70–0.79)** but
not on FLY125 (0.00): the per-mic modulation is not purely additive floor —
either it multiplies everything at that diaphragm, or DREGON's k ≤ 6 lines
are floor-dominated there (line − floor ≈ 3 dB on this rig at low orders);
the hierarchical fit must let `u_cm(t)` act on lines and floor and test
which. Michael's floor needs only a common term (0.8–0.9), σ 1–2 dB net,
slower (lag-1 0.81–0.94). The current sampler's floor drift (common to all
mics) has the wrong sharing for DREGON's low band.

### Coherent share per order: Michael's even orders are phase-locked tones, DREGON's lines are random-phase (`coherent.py`)

Bretthorst's general linear model on the refined carriers: for order `k`
and a segment of `T` s, the eight model functions `cos/sin(k φ_ρ(t))` of
the four rotors are fitted jointly by least squares (the Gram matrix
carries the near-collinear rotors), the explained fraction of the band
energy around the lines is the segment's coherent share, the same
functions at `k + ½` give the null. Controls on real FLY125 carriers:
phase-locked tones read 0.94–0.99 at k = 1–2 flat in `T` (0.7 / 0.5 / 0.3
at k = 4 / 8 / 12 — the band's noise share grows with `k`, so absolute
values are read against this ceiling), random-phase Lorentzian lines of
HWHM 0.5 Hz read 0.92 → 0.17 from 0.25 to 2 s at k = 1; null 0.05.
Real clips, median net share at T = 0.25 / 0.5 / 1 / 2 s:

| k | FLY125 (10) | FLY124 (14) | DREGON room1 (7) | DREGON room2 (17) |
|---|---|---|---|---|
| 1 | 0.76 / 0.75 / 0.77 / 0.77 | 0.73 / 0.68 / 0.64 / 0.57 | 0.83 / 0.67 / 0.43 / 0.20 | 0.83 / 0.47 / 0.23 / 0.10 |
| 2 | 0.66 / 0.85 / 0.86 / **0.82** | 0.70 / 0.79 / 0.76 / **0.73** | 0.76 / 0.55 / 0.28 / 0.17 | 0.65 / 0.33 / 0.15 / 0.07 |
| 3 | 0.38 / 0.37 / 0.25 / 0.12 | 0.32 / 0.19 / 0.12 / 0.07 | 0.34 / 0.19 / 0.11 / 0.04 | 0.45 / 0.20 / 0.10 / 0.05 |
| 4 | 0.52 / 0.46 / 0.39 / 0.33 | 0.44 / 0.38 / 0.29 / 0.25 | 0.34 / 0.19 / 0.12 / 0.05 | 0.37 / 0.19 / 0.08 / 0.04 |
| 5 | 0.19 / 0.11 / 0.04 / 0.03 | 0.17 / 0.09 / 0.03 / 0.02 | 0.04 / 0.03 / 0.01 / 0.01 | 0.26 / 0.12 / 0.06 / 0.03 |
| 6 | 0.56 / 0.48 / 0.43 / 0.40 | 0.34 / 0.23 / 0.13 / 0.09 | 0.15 / 0.10 / 0.04 / 0.02 | 0.18 / 0.10 / 0.04 / 0.02 |
| 7 | **−0.08 / −0.04 / −0.02 / −0.01** | −0.04 / −0.02 / 0 / 0 | 0.10 / 0.05 / 0.02 / 0.01 | 0.14 / 0.07 / 0.03 / 0.02 |
| 8 | 0.40 / 0.32 / 0.20 / 0.13 | 0.19 / 0.14 / 0.06 / 0.04 | 0.15 / 0.08 / 0.04 / 0.02 | 0.14 / 0.07 / 0.03 / 0.01 |
| 10 | 0.20 / 0.12 / 0.10 / 0.05 | 0.14 / 0.07 / 0.04 / 0.03 | 0.09 / 0.05 / 0.03 / 0.01 | 0.08 / 0.04 / 0.02 / 0.02 |
| 12 | 0.05 / 0.04 / 0.02 / 0.02 | 0.05 / 0.03 / 0.01 / 0.01 | 0.05 / 0.03 / 0.02 / 0.01 | 0.10 / 0.04 / 0.02 / 0.01 |

Readings. (1) **Michael's even orders are phase-locked tones**: k = 2 holds
0.82–0.86 out to 2 s (the tone control's shape), k = 4 and 6 decay slowly
(0.5 → 0.3–0.4), k = 8 by 1 s; k = 1 (shaft rate) is coherent at 0.6–0.77.
The **odd orders are nearly absent** (k = 7 at the null on both flights,
k = 5, 9, 11 ≤ 0.2 at 0.25 s): a two-blade rotor's blade-passing comb at
`2 r` with only imbalance at the odd shaft orders. This is the "phase
interference" of the low harmonics — four deterministic tones in one bin —
and it is what the earlier lag-coherence plateau (0.33–0.37, all orders
1–8 pooled) was averaging over. (2) **DREGON's lines are random-phase at
every order**, decaying with `T` exactly as the Lorentzian control: k = 1
0.83 → 0.20 (room1) / 0.10 (room2), i.e. coherence time ≈ 0.5 s
(HWHM ≈ 0.3 Hz at k = 1, broader than the bench's 0.04 Hz/order — flight
shaft jitter); odd and even orders alike. (3) Consequences: Michael's
preset renders even orders as phase-locked tones with a per-order
decoherence (`q_k`) read from these curves and odd orders at their
measured low level; DREGON's renders Lorentzian random-phase lines with
the flight coherence time. In the spectral fit, k ≤ 8 on Michael's is a
deterministic-tone regime where the exponential cell model is wrong by
construction; ladder verdicts are read at k ≥ 9 there.

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

## The rig model: ladder, verdicts, presets — 2026-09-09

Instrument: `rig.py` (`RigParams` / `ClipInRig` / `fit_rig` / `fit_heldout`),
one joint MAP over all clips of a rig with rig-level parameters tied,
rotor-level deviations partially pooled, clip nuisances free — Bretthorst's
joint analysis of multiple measurements with the Whittle likelihood. Every
ladder step is Gaussian lines with a carrier correction. Planted-rig control
(`controls.py`, real FLY125 carriers, exponential cells): width law,
microphone pattern (0.97–1.00) and the per-mic modulation (0.92–0.93, σ 3.4
vs 3.35 planted) come back; profile 0.59–0.86, rotor deviation 0.35–0.45,
floor tilt/shape not — identifiability limits at this comb density (the
bench gives δ_rk directly, the between-line instrument gives the floor); the
control's excess is −0.05…−0.11 because planted modulations faster than the
LOO reference's ±64 ms replicate window bias the reference high. Kaggle P100,
~4 min per config.

| config | DREGON room2 (17) → room1 (7) | Michael's FLY125 (10) → FLY124 (14) |
|---|---|---|
| M0 independent fits | 0.088 → 0.089 | 0.077 → 0.079 |
| M1 every rig-level parameter tied | 0.091 → 0.151 | 0.127 → 0.150 |
| M2 + rotor profile deviation δ_rk | 0.093 → 0.148 | 0.123 → 0.150 |
| M4 + per-mic floor gains / gain-all + per-mic low-band OU | 0.068 → 0.149 | 0.112 → 0.123 |
| **M5 + OU (rough, 0.5 s) amplitude process** | **−0.014 → 0.071** | **0.017 → 0.030** |

excess over LOO, nats/cell, train median → held-out median. Readings. (1)
Tying every rig-level parameter costs 0.003 in-sample on DREGON and 0.05 on
Michael's — one rig model per rig is adequate. (2) The rough amplitude
process is the largest single lever on both rigs, larger than the line
shape: on Michael's the M5g rig model predicts held-out FLY124 *better* than
independent per-clip fits (0.030 vs 0.079); on DREGON it halves the transfer
penalty (0.149 → 0.071; two room1 clips remain at 0.11 and 0.75). (3) δ_rk is
not identifiable in flight (M2 = M1), as the control said; the bench value
(2.2 dB) and the flight fit's 2.7–2.9 dB agree. Rig parameters (M5/M5g):
DREGON width law 3.7 + 0.37 k Hz × per-rotor scale 1.5–2.8, profile roll-off
0.68, even−odd +1.4 dB, mic line-gain spread 6 dB, per-mic floor ±3 dB,
tilt −5.2; Michael's 2.0 + 0.45 k × 1–3.3, roll-off ≈ 0 with even−odd +6 dB
(k = 2 at +22, odd orders −6…−11), gain-all spread ±4.5 dB, mic line-gain
spread 7.6 dB, tilt −3.1.

**Two shaft-jitter regimes.** Michael's high-order widths (0.45 k × 1–3.3 Hz)
and its k = 2 tone coherent for 2 s are consistent only with widths growing
as k²: a shaft jitter of σ ≈ 0.6 rev/s with a *millisecond* correlation time
(the diffusive regime; WP18's 0.6 rev/s, 16 ms) — HWHM ≈ 0.011 k² Hz gives
0.04 Hz at k = 2 and 18 Hz at k = 40. DREGON's ~0.5 s coherence at k = 1 and
0.55–1 Hz/order widths are the linear law: σ 0.8–1.2 rev/s, τ 0.15–0.35 s
(quasi-static). One mechanism, one knob per rig.

**Renderer** (`stochastic_rotor_noise.py`): `line_mode: "fm"` — tones on a
shared jittering shaft (OU per rotor) with a per-harmonic phase diffusion,
vectorized at 0.7 s per 8-mic 4-s clip; `rotor_delta_std_db` (replaces the
similarity mixing), `harm_gp_kernel: ou`, `umod_*` (independent per-mic
low-band log-OU on floor and lines), `mic_gain_all_db`, `mic_floor_std_db`,
`floor_shape_preset` (a measured curve the GP jitters around). Default
streams are bit-identical (degenerate ranges draw nothing).
**Presets**: `conf/online_mix/rig_fm_5050.yaml`, 50/50 DREGON / Michael's
plus the 0.2 silence arm; every range from the tables above.

**Gate 3 (held-out summary statistics), preset renders on real
trajectories.** Coherent share (net, k = 1/2/4/8 × T = 0.25/0.5/1/2 s):
Michael's preset k = 2: 0.68/0.77/0.78/0.65 (real 0.66/0.85/0.86/0.82),
k = 4: 0.39/0.52/0.55/0.24 (0.52/0.46/0.39/0.33), k = 8: 0.36/0.23/0.15/0.08
(0.40/0.32/0.20/0.13); DREGON preset (jitter 0.7–1.1) k = 1:
0.95/0.73/0.53/0.23 (0.83/0.47/0.23/0.10), k = 2: 0.80/0.54/0.27/0.15
(0.65/0.33/0.15/0.07), k = 4: 0.50/0.27/0.13/0.06 (0.37/0.19/0.08/0.04).
Floor envelope (std dB / cross-mic corr / lag-1): DREGON 50–500 Hz render
2.8 / 0.14 / 0.91 vs real 3.3 / −0.04 / 0.92, 500–2000 Hz 1.8 / 0.92 / 0.84
vs 1.6 / 0.71 / 0.90; Michael's 50–500 Hz 2.05 / 0.83 / 0.74 vs
1.8 / 0.81 / 0.79, 500–2000 Hz 1.2 / 0.90 / 0.72 vs 0.9 / 0.58 / 0.63.

**Gate 1 (calibration), committed presets.** 12 renders per preset on the
real trajectories (`prepare-presets`), refitted with the rig model (M5 /
M5g) on Kaggle: excess −0.129 (DREGON) / −0.082 (Michael's; mean −0.007).
Oracle decomposition on planted clips with a *known* spectrum: the true
spectrum itself scores −0.042…−0.061 against the LOO reference at these
modulation speeds (OU τ 0.3–1.5 s, umod τ 0.15–0.5 s) — the reference's
bias — and the fit's free clip nuisances take a further −0.03…−0.06 on
draws (the planted control's −0.05…−0.11). So the renders sit where a
correctly specified generator sits under this instrument. The real clips
under the same model sit at −0.014 / +0.017: relative to the oracle bias
they keep ≈ 0.05–0.10 nats/cell the model does not explain and the renders
do not carry — the residual realism gap, which no preset range closes; it
needs structure the spectral model lacks (coherent low-order tones in the
cell likelihood, the interference of four tones in one bin). The transfer
run (`rig_fm_scv2_unified`) is the test of whether what is captured is
enough.

## Population correction — 2026-09-09

The matched-RPS listening audit rejected the first fitted presets despite
their conditional Whittle scores: both rigs produced too many independently
prominent teeth; Michael's real comb has a smooth order decay and a strong
blade-pass region, while its draw kept conspicuous high orders. The cause is
statistical, not another range: M1--M5 maximized each clip nuisance and the
preset sampled hand ranges around those modes. `max_eta p(I|theta,eta)` is not
the prior-predictive `integral p(I|theta,eta)p(eta|theta)d eta`.

`population.py` now fits the latter with a native-PyTorch variational
objective over every renderer random effect at clip level. The static profile
is `mu_k + delta_rk + ell_cr + B_k z_cr`: persistent rotor identity,
marginalized line/floor level and low-rank correlated order modes. Basis rows
are centred so they cannot duplicate level. The population prior is expressed
in scale-invariant coordinates — order-2 line/floor ratio and rotor-level
contrasts — because the fit normalizes each clip's periodogram and absolute
level is unidentifiable. Whitened line/floor OU paths, DREGON microphone OU
and carrier corrections are marginalized too. Held-out scoring freezes the
population and reports both ELBO and importance-weighted predictive NLL/ESS.
`run.py popfit` is the P0--P3 recording-held-out ladder.

The planted **population** control (24 independent clips, 18 train / 6
held-out, rank 1) passes:

- order-profile covariance cosine 0.998;
- order-2 line/floor mean 23.67 dB recovered from 23.79 dB;
- line/floor population std 2.34 dB from 3.28 dB and rotor-contrast std
  3.72 dB from 3.00 dB;
- train / held-out excess over the correct Whittle reference +0.012 / +0.013
  nats/cell;
- median held-out importance ESS 3.75 of 8.

The bounded regression control uses 12 clips and admits sampling uncertainty
while pinning covariance recovery, held-out calibration and non-degenerate
importance weights (`test_stochastic_fit_population.py`). Full experiment
tests: 50 passed, 1 deselected; renderer tests: 28 passed.

### The first population ladder does not pass a waveform gate

The first `popgate` implementation compared prior profile draws with
`clip.params.profile_db` from the same candidate fit. That is circular as a
realism claim: the “real” side already obeys the candidate hierarchy. It is
retained only as `poplatent`, labelled `fitted-latent-diagnostic`.

`poprawgate` is the actual gate. It never refits either side. On each real
trajectory it renders one independent prior-predictive waveform and extracts
from real and rendered audio the same union-comb statistic: for order `k`, the
integrated power in the union of all four telemetry-predicted line windows
relative to neighbouring bins after masking every predicted line. Rotors are
aggregated because room1's nearby lines are almost never attributable to one
rotor without importing the candidate model. Non-positive excess is retained
as a censored observation. Train the classifier on room2-real versus
room2-render; score it only on room1-real versus room1-render.

DREGON P0--P3 marginal evidence is nearly flat. Held-out IW NLL/cell:
P0 −1.7488, P1 −1.7530, P2 −1.7569, P3 −1.7501. The tiny P2 optimum is not a
realism result. The first DREGON waveform diagnostics below are **superseded
hybrid-preset diagnostics** after the parity audit described below:

| draw | exploratory AUC | 90% coverage | median-curve IQR RMSE |
|---|---:|---:|---:|
| P0 | 1.000 | 0.116 | 2.91 |
| P2 | 1.000 | 0.201 | 2.69 |
| P0 + existing spectral recolour and random-RIR observation model | 1.000 | 0.167 | 3.05 |

The AUC is **not a DREGON model-selection gate**. Real DREGON carries wind
gusts that this model deliberately does not attempt to reproduce beyond the
already measured per-microphone low-band OU term; a classifier can detect
that missing channel trivially even when the comb is right. `poprawgate`
therefore refuses to run its classifier on DREGON groups. DREGON retains only
descriptive per-order comb curves/coverage. The classifier is used on
Michael's FLY125→FLY124, where gusts are not the dominant unmatched process,
with FLY103/FLY108 reserved for independent confirmation.

All DREGON fit/gate clips use the refined audio-aligned trajectories from the
August telemetry-refinement campaign: room2 crops carry their refined
references and the turning room1 clips were refined during bundle preparation.
`poprawgate` now rejects a turning DREGON clip with no refinement metadata.
Unrefined command telemetry is not a valid carrier for fit-realism conclusions.

**Fit→renderer parity correction.** The first waveform gates transferred the
fitted profile, line/floor population and floor curve but accidentally left
the policy's linewidth and random microphone draws in place. They did not
test the complete fitted population. `population_ranges` now transfers
per-rotor `gamma0`/slope/width scale, the fitted `(mic, rotor)` line gains,
per-mic floor gains and Michael's whole-signal gains; matched rendering also
uses the fitted line/floor speed exponents and static floor share. The FM
shaft-rate std is the fitted Gaussian high-order HWHM slope divided by
`sqrt(2 log 2)`, per rotor. The fitted width intercept remains in the spectral
normalization but is not invented as an FM process: it is dominated by the
analysis-window/carrier nuisance and shared-shaft FM has no order-independent
width.

The first descriptive self-control (four draws on both sides) was insufficient:
the real gate has one “real” draw and four comparison draws per carrier.
The exact control now renders eight draws on each FLY125/FLY124 carrier,
repeatedly assigns one versus four, and bootstraps whole carrier clusters.
Over 200 repetitions: point AUC median/95th 0.541/0.633, upper AUC
0.694/0.795, coverage median/5th 0.873/0.831, and curve error median/95th
0.159/0.201 IQR. The proposed upper-AUC limit 0.70 falsely rejected 44.5%;
the preregistered limit is therefore 0.80, which rejects 4%. The complete
Michael gate is calibrated before reading the real result. The old hybrid
table cannot admit visibility; rerun Michael's after parity first.

The hybrid mismatch mirrored the listening finding—low orders 1--4 weak and
many orders above 15 prominent—but cannot identify which omitted fitted
parameter caused it. One independent defect is valid: P2's first factor had
12.1 dB RMS because loading scale could grow while variational coordinates
shrunk. `RigParams` now uses unit-RMS mode directions plus an explicit positive
mode standard deviation with a 5 dB half-normal prior; P1--P3 must be rerun.

The training-only mean correction, default recolour/reverb, and Ledoit--Wolf
covariance probes were all evaluated through the same incomplete hybrid
renderer. Their numerical failures remain useful provenance but do **not**
reject those structures or admit the next visibility arm. Decision order is:
complete fit→renderer parity → planted waveform control → rerun Michael's
continuous P0--P3 → only then decide whether PV visibility/censoring is
necessary.

**Validation status correction.** Repeated room1 failures informed the
sequence above, so room1 is now development validation, not an untouched
confirmatory set, even though no room1 statistic entered parameter fitting.
The next candidate and stopping rule must be frozen from training-recording
CV. FLY103/FLY108 remain independent for Michael's. DREGON has no unused
noise-only recording in the current bundle; either prepare a genuinely unused
recording or label the DREGON realism result exploratory.

### First 50/50 rig-FM transfer run: decisive failure

The originally requested unchanged baseline completed after its interrupted
run was resumed from the R2 checkpoint: W&B `7rf8rng7`, revision `0aa5923`,
104 validation rounds / 52,000 optimizer steps, stopped by any-subset
saturation. Best scores (each at its own epoch):

| metric | best MAE (rev/s) | epoch |
|---|---:|---:|
| `real_r3` / real overall | 23.075 | 47 |
| real nosource | 16.840 | 47 |
| real source-present | 30.131 | 1 |
| synthetic overall | 8.600 | 30 |
| static no-mix / mix | 1.136 / 1.134 | 59 / 92 |

R4's comparable real score is 2.99. The first fitted-preset stream is
therefore 7.72× worse and fails the transfer objective. Static-subset
improvements kept resetting the controller until epoch 103; the run was not
censored early by its poor real score. This result belongs to the pre-parity,
hand-exported presets and is a baseline, not a test of the corrected
population renderer now under development.

## Decomposed-envelope fits, the width law, and the label error — 2026-09-10

Four things were fitted or decided here. Every one uses training recordings
only (FLY125 / dregon_room2); FLY124 and dregon_room1 stay held out, and
FLY103/FLY108 remain untouched.

### The profile population comes from the decomposed envelopes

`src/experiments/stochastic_fit/decomp_population.py` (`popdecomp`) fits
`mu_k + delta_rk + Bz + eps` to linewidth-matched Vold-Kalman amplitudes from
`decomp-frames-v2` (FLY125 only). The envelopes separate microphone, rotor and
order against the exact carrier the solve used, so the profile hierarchy is
identifiable there while it is not in the dense-comb spectral fit (M2 ≡ M1 in
the ladder). Rank selection uses held-out `score_samples` with rotors averaged
inside a time chunk (rotors of one chunk share the flight state) plus the
**one-standard-error rule**: rank 3 → rank 1 on Michael's, and on the planted
control rank 3 → rank 2 with the true rank 2 (basis cosine 0.9991).

### The amplitude process: one timescale, not two

`decomp_dynamics.py` (`popdyn`) fits the amplitude process from the same
envelopes. Only the band the decomposition passes untouched is used
(0.03–1.5 Hz for orders `k >= 4`, whose Vold-Kalman bandwidth is `k` Hz), and a
white term absorbs envelope estimation noise without being transferred.
Cross-validation is **leave-one-rotor-out**: the kernel is a rig-level
population and each rotor carries its own common component, so a held-out rotor
is independent in a way a held-out slice of time or order is not.

| arm | held-out Whittle / bin | SE |
|---|---:|---:|
| 1 OU + white | −2.2237 | 0.0524 |
| 2 OU + white | −2.2108 | 0.0517 |
| 3 OU + white | −2.2109 | 0.0518 |

The second component buys 0.013 nats against a fold SE of 0.052, so the
one-SE rule keeps **one** component: `std 2.90 dB, tau 0.82 s, cross-order
coherence 0.191`. The earlier "two-timescale ACF" reading does not survive
cross-validation, and the planned second renderer timescale is therefore
**rejected** — the renderer already expresses this exactly (`harm_gp_kernel:
ou`). The planted control recovers a genuine mixture (total std 4.80 vs 4.69,
coherences 0.324/0.055 vs 0.30/0.05), so the negative result is the data's,
not the instrument's.

Both speed laws were pinned at 2.5 and are now measured:

| quantity | fitted | previous |
|---|---:|---:|
| line power vs rps (dB/dB) | 3.93 ± 0.10 (clustered) | 2.5 |
| floor power vs rps (dB/dB) | 3.46, bootstrap q05 3.20 | 2.5 |
| floor static share | 0.0 | 0.0025 |

The floor law is fitted on the decomposition's *residual* — the recording with
every tracked line removed, which is what the renderer's floor stands for. Its
upper bootstrap tail is unidentified (the exponent and the static share trade
off along a ridge, so q95 sits at the bound); the per-band slopes run 1.0–5.0,
which is real heterogeneity a single exponent cannot carry. On the planted
control the speed exponent came back 4.40 against a true 4.10 with a clustered
SE of 0.11: a 180 s record with few independent speed excursions supports
about ±0.3, so read the fitted 3.93 that way.

### The width law is quasi-static (k^1), on both rigs

`Spec.fit_width_power` makes the exponent of `gamma = gamma0 + slope k^p` a
fitted rig scalar, so the regime is read off the likelihood instead of assumed.
Per-clip fits over the 27 training crops (`gauss` = p 1, `gauss_wp` = p fitted,
`gauss_wp2` = p 2), mean `excess_over_loo` (lower is better):

| rig | p = 1 | p fitted | p = 2 | Δ(p=2) ± SE |
|---|---:|---:|---:|---:|
| FLY125 (10 clips) | 0.0696 | 0.1056 | 0.1443 | +0.075 ± 0.018 |
| dregon_room2 (17) | 0.0759 | 0.1923 | 0.2037 | +0.128 ± 0.023 |

The quadratic law is 4–6 standard errors worse on both rigs, and the free
exponent lands at a median of 0.9 with per-clip scatter 0.44–3.6 (weakly
identified, and its extra parameter destabilizes the staged optimizer). So
`k^1` stands: the shaft's speed offset is **frozen inside an analysis window**.
Michael's preset previously carried `shaft_jitter_tau_s: [0.003, 0.008]`, a
diffusive `k^2` regime that was assumed, never measured; both rigs now use
`[0.3, 1.0]` s — above the 128 ms window, below the clip.

### The renderer had no label error; real recordings have ~1.1–1.4 rev/s of it

Every per-clip fit with `Spec.rps_offset` carries a per-rotor carrier
correction. Its static part, over training clips only
(`carrier_error.py`, `popcarrier`):

| rig | static offset std (rev/s) | 90% CI | within-clip |
|---|---:|---:|---:|
| Michael's (FLY125, 10 clips) | 1.100 | 0.71–1.40 | 0.90 |
| DREGON (room2, 17 clips) | 1.362 | 1.10–1.59 | 1.34 |

The estimate is a posterior mean under a zero-centred prior, so it is shrunk —
it understates the label error. The sub-clip structure is *not* reported: its
autocorrelation is the prior's own (knots every `rps_offset_dt_s`), so only the
static part is transferred. The magnitude is corroborated independently by the
blind tracking campaign, whose final PIT-MAE is 1.14 rev/s on fly124_cruise and
3.21 on dregon_ramp (`docs/experiments/rps-refine-precision.md`).

At order 64 a 1.1 rev/s offset displaces a line by ~70 Hz — nine analysis bins
— which is why real combs lose their top against a local floor while a comb
rendered exactly on the label does not (measured gap at k = 64: real −7.9 dB
vs synthetic +1.1 dB). `StochasticRanges.shaft_offset_rps` now draws one static
per-rotor offset, the comb rides `label + offset`, and the returned label stays
the caller's, so a synthetic clip carries the same label error a real one does.
A stopped rotor stays stopped.

This matters beyond the gate: R4's real validation MAE is 2.99 rev/s while the
real labels themselves are uncertain at 1.1–1.4 rev/s. Training on offset-free
synthetic data asks a regressor for a precision the real task does not define.

### Transfer plumbing

`popexport` writes a policy whose stochastic sources carry the fitted presets
through `raw_predictive.population_ranges` — the same function the gate renders
with, so an exported policy and a passed gate cannot drift apart. Two
corrections were needed there: the renderer sizes each clip's comb from its own
slowest turning rotor, so fitted profiles are held out to 200 orders past their
**measured support**, and that support stops at the onset of the fit's
unconstrained tail (DREGON's mean curve falls from −14 dB at k 91 to −140 dB by
k 106 — orders that sit above the fit band on every frame and are pure prior).

## Prior conditional-fit conclusion (superseded by the population correction)

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
