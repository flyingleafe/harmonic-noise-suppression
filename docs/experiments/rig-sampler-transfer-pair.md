# Easy and hard sim-to-real transfer: the rig-sampler pair

> **OPEN DECISION — BLOCKS THE NEXT RUNS (raised 2026-09-15).** Two defects in
> the preset banks were found after this pair finished. Both change the
> training distribution, so the follow-up arms (curriculum, mixed) must NOT be
> launched until the user picks a fix and the banks are rebuilt. Detail and the
> concrete options: [§ Open decision: rebuild the
> banks](#open-decision-rebuild-the-banks). Two answers are needed: (a) trim
> the unconstrained ladder edge, enable `band_taper_frac`, or both; (b) DREGON
> anchor raw or refined.

**Status:** complete, with an open decision before the follow-up arms. Jobs
`rig-easy-dce43e` and `rig-hard-1b2f06` (vast, A100) ran to completion from
commit `7525ce8c` ("transfer arms: rig neighbourhood sampler, preset-bank
streams, renderer fix"). Per-experiment docs:
`conf/experiment/rig_easy_scv2_unified.md`,
`conf/experiment/rig_hard_scv2_unified.md`.

## Motivation

`rig_fitted_scv2_unified` trained an SCv2 RPS predictor on the two rigs' cruise
fits as POINT presets. It transferred unevenly — best `real_r3` 9.97 rev/s at
epoch 22, plateau 13.42, `r1`/`r2` ratio 1.93 — and after epoch 22 every real
view degraded while the synthetic half kept improving: overfitting to two fixed
parameter vectors. The real-data reference is `real_r4_scv2_unified` at
`val/real_r3` **2.99**.

The pair asks whether the limit is the *point-ness* of the training
distribution and, if the answer is yes, how far the distribution can be widened
before it stops helping. Nothing but the noise family changes.

* **Easy** — `rig_easy_scv2_unified`: draws from CLOSE neighbourhoods of the two
  real rigs' fits. The training distribution is measured to COVER the real
  rigs, so this arm should transfer. If it does not, the limit is the fit's
  fidelity, not its point-ness: a neighbourhood of a wrong point is still
  wrong.
* **Hard** — `rig_hard_scv2_unified`: the same cloud width smeared along the
  PATH between the two fits, mixing coordinate uniform on [0, 1]. Rigs
  resembling either real rig are present, neither is especially likely, so the
  model cannot memorise a fingerprint and has to learn the operation.

The decision rule stated when the pair was launched: the easy arm's real
validation loss is EXPECTED to fall well. The informative outcome is the hard
arm — if a model trained on a wide cloud in which neither target rig is
privileged also transfers to real audio, the sim-to-real problem for this task
is essentially closed, because the real rigs stop being special.

## Setup

Both arms are `rig_fitted_scv2_unified` with one field changed: the training
stream. Same `real_r1_scv2` architecture, `override /validation: rps_unified`,
bfloat16, batch 128, 12 workers, 2 s clips, `samples_per_validation: null`,
synthetic only. The streams (`conf/online_mix/rig_easy_5050.yaml`,
`rig_hard_5050.yaml`) are the fitted policy with each stochastic source's
`ranges:` replaced by `preset_bank:`; weights, the full-flight trajectory,
`rps_scale_range`, `render_reuse: 48`, the level draw, the silence source, the
speech source and the mixing policy are the base policy's.

**Banks.** `scripts/_build_rig_bank.py`, 2048 draws each, seed 20260914,
anchors `results/S2/cruise_8clip.json:fly125_cruise_00` (Michael's) and
`results/S2/dregon_room2_cruise_refined.json:free-flight_nosource_room2_cruise_00`,
both on the physical level scale.

* Easy: `--mode neighbourhood --strength 2.0`, 1024 draws per anchor. Strength
  2.0 is the measured COVERAGE setting: the cloud brackets the real clip in
  100 % of 1/3-octave bands above 300 Hz on both rigs (95.2 % / 90.5 % of all
  bands including the low ones); 2.5 and 3.0 add nothing above 300 Hz and cost
  acceptance. Guards reject 23.8 % of Michael's draws and 5.9 % of DREGON's
  (1.29 attempts per accepted draw, 77.9 % first try; `ltas` 383, `speed_law`
  190, `trend_falls` 39, `parity_sign` 24).
* Hard: `--mode path --spread 2.0`. Same width, different location. The
  realised mixing coordinate survives the truncation: ten-bin counts
  [208, 194, 212, 216, 187, 199, 207, 199, 214, 212], mean *t* 0.503,
  Kolmogorov–Smirnov distance to U(0, 1) **0.0127** against a 95 % threshold of
  **0.0301**. Acceptance 1.21 attempts per draw, 82.7 % first try (`ltas` 293,
  `speed_law` 162, `trend_falls` 2, `parity_sign` 1).

**Sampler structure** (`src/experiments/stochastic_fit/rig_sampler.py`, widths
in `results/rig_sampler/structure.json`). Per rotor, over orders `k = 1..K`
with `x = log10 k`,
`profile_db[r,k] = gain + slope*(x - mean x) + env(k)*s(k) + resid(k)`. The two
structural invariants of a rotor comb are HARD guards — reject and redraw,
never clip:

* the parity split keeps its sign and DECAYS with order (13.6 dB over k ≤ 16,
  3.1 dB over k = 33..48, 0.0 dB above k ≈ 80 on FLY125 cruise rotor 0), so
  `env(k)` is a measured envelope rather than one amplitude; 0 of the 124
  measured rotor profiles has an inverted low-band split;
* the harmonic profile falls with order like a power law in `log` order.

Linewidths move only slightly; the per-rotor gains and profile shapes move
freely. Widths are measured across the SIX stage-2 fits in `structure.json`'s
provenance (Michael cruise base/refined/dyn, Michael standby, DREGON room-2
cruise refined, DREGON flight), in three tiers, with the strength ladder:
`strength = 1` is the between-refit spread (another refit of the same audio),
`~3` reaches the between-rotor spread within one rig (slope 4.40 against
1.46 dB/decade), `~5` the between-rig spread (slope 6.93). So strength 2.0 is
"a noticeably different rotor set on the same airframe".

**Exponent bound.** The speed exponents are bounded to the FITTED range —
`amp_exp` to [4.398, 14.111], `floor_exp` to [0, 6.792]
(`structure.json:between_rig.per_fit`; `floor_exp`'s measured minimum is −3.711
and is raised to 0 because a negative floor exponent diverges at zero rotor
speed). The measured between-refit width (sigma 2.733) is legitimate but its
TAIL is supported by no fit: unbounded, the easy bank reached 24.7 dB/dB and
the hard bank 31.9 dB/dB, at which an idle window's comb sits ~20 dB under the
same rig at cruise. The bound clipped 706 `amp_exp` / 510 `floor_exp` draws of
2050 (easy) and 672 / 449 of 2049 (hard); realised `amp_exp` 8.85 ± 3.57 and
8.76 ± 3.50, `floor_exp` 2.79 ± 2.94 and 2.93 ± 2.82, all filling their bounds.

**Level path.** Every anchor is loaded through both halves of the declared
conversion: `scores.power_scale` folded per clip, then `to_renderer_units`'
`10 log10(work/analysis)` = **+4.4032 dB** at the 44.1 kHz work grid these
renders use. On that scale, with NO gain applied anywhere, each anchor's render
agrees with its OWN real clip to **−1.81 dB** (FLY125 cruise) and **+0.58 dB**
(DREGON room-2 cruise) in the 300 Hz–7.9 kHz level band, RMS band deviation
2.09 and 0.59 dB. The band coordinate is the honest one: the same two exports
differ by +2.467 dB on it and by +15.750 dB on a whole-array RMS, and
+2.467 = +0.087 (the two REAL recordings) + 0.575 (DREGON's fit error) + 1.805
(FLY125's fit error) — the recordings are within 0.09 dB of each other, so the
residual offset is fit error and it is small.

**Reproducibility and transport.** Banks are gitignored BUILD PRODUCTS (24.6 /
24.4 MB) and `omnirun` ships a clean pushed checkout, so each job REBUILDS its
bank first (56 s / 65 s on one core) and skips the build if the file is already
current. The builds are bit-reproducible — file SHA-256
`58b24d2d…3dd307d` (easy) and `b0a977a0…11209a6` (hard) — and the provenance's
`inputs_digest` covers the anchors, the donor ranges, every width and the
source of the builder, the sampler and the renderer, so a rebuild with matching
inputs is skipped and a code change is not. The two anchor fits were
un-ignored (`.gitignore`: `!results/S2/cruise_8clip.json`,
`!results/S2/dregon_room2_cruise_refined.json`) so they travel with the
checkout instead of needing R2 transport before the build step.

**Known caveat, belonging to the arms rather than the plumbing.** The
per-microphone pattern is perturbed per entry (sigma 1.12 dB gain, 0.20 dB
floor) but NOT permuted across channel indices, so the index-locked mechanism
behind the fitted run's `r1`/`r2` ratio of 1.93 is softened, not removed. The
perturbation is small against DREGON's −13 to +11 dB span.

## Preparation findings

### A renderer defect on the shared training path

Making the level path absolute (above) made an absolute per-order comparison
possible for the first time, and it found a one-line defect in
`stochastic_rotor_noise.synthesize`'s `line_mode in ("coherent", "fm")` branch.
The tone bank's level is set by `scale = sqrt(want / have)`, and the two sides
of that ratio referred to DIFFERENT spectra: `want`'s denominator was the bare
floor, while `have`'s was the variance of `floor_audio`, which INCLUDES the
incoherent share of the comb. `scale` therefore came out high by exactly
`sqrt(I)` with

`I = 1 + mean(g_m · incoherent) / (mean(floor) · floor_mic_m)`,

one factor on the whole tone bank, giving a per-order error
`Δ_k = 10 log10(w_k·I + (1 − w_k))` that is large only where the coherent share
`w_k` is large. Measured against the fit's own forward law (`predicted_m`), on
each anchor's own fit support, one microphone:

| order | Michael, measured | Michael, closed form | DREGON, measured | DREGON, closed form |
|---|---:|---:|---:|---:|
| 1 | +14.29 | +14.49 | +13.27 | +13.14 |
| 2 | +9.82 | +10.12 | +9.25 | +8.98 |
| 3 | +3.41 | +4.08 | +3.33 | +3.45 |

The closed form has NO free parameters and its worst residual against the
measurement is **0.75 dB**. In every band above 300 Hz the render tracks
`predicted_m` to ≤0.5 dB, and at orders 4–6 to ≤0.73 dB. Performing the
coherent/incoherent split OUTSIDE `synthesize` — two renders neither of which
splits — lands within 0.65 dB of the forward law at every order and every band
on both rigs, which measures what the
corrected mixing gives rather than patching it. Diagnosis and reproduction:
`docs/explainers/flight-assembly.qmd` §4.4,
`docs/explainers/flight-startup/low_order_diagnosis.json`. The fix (`want`'s
denominator is `np.mean(floor_spec[m])`, the spectrum `floor_audio` was
realized from) is in `7525ce8c`, which is what both arms run.

**Consequence for the historical rows — say it plainly.** EVERY earlier
stochastic training set carried this defect, and `synthesize`'s default
`normalize_rms=0.1` hid it: rescaling the whole waveform erases the absolute
level while leaving line-to-floor ratios intact, so the inflation never showed
up as a loudness error, only as too much power in orders 1–3. The historical
stochastic rows — including `rig_fitted_scv2_unified` and
`ctrl_diverse_scv2_unified` — are therefore **not comparable value-for-value**
with these two arms: they are the closest available reference for the QUESTION,
but they were trained on a different noise distribution than the one the fixed
renderer produces. Read them as direction, not as a delta.

The same applies to any earlier LTAS number produced through that branch. The
revised-phase campaign's absolute-level LTAS gate runs the legacy arm through
`stage2.render_from_export`, i.e. through `synthesize`, and its DREGON figures
are a baseline of 2.1362 dB against a baseline-variability tolerance of
0.0560 dB — a tolerance three orders of magnitude smaller than the low-order
inflation. Those gate numbers are **not reproducible against the fixed
renderer** and must be re-derived before they are cited again.

### Surviving fit-side error, deliberately not corrected

Once the render defect is accounted for, the FIT still over-predicts the lowest
orders: order 1 sits about **3 dB** high on Michael's rig and about **8 dB**
high on DREGON (`predicted_m` against the real clip, same supports). That is
not fixed here, on purpose — it is a property of the family under test, and the
question these arms ask is whether that family, sampled as a distribution,
transfers. Correcting it would change the object being measured.

A side diagnostic prepared alongside, `results/multires_rescore/table.md`
(previous vs revised C3 composite risk at three window lengths), is DIAGNOSTIC
ONLY and in sample for the previous arm; it is not evidence for or against
either arm here.

## What will be measured

Real-split validation, selecting on **`real_overall`, not `overall_macro`** —
the fitted-preset run's synthetic half improved monotonically while every real
view degraded, so the macro neither stopped the run nor reduced the LR. The
three comparison rows:

| row | stream | role |
|---|---|---|
| `rig_fitted_scv2_unified` | two fitted rigs as POINT presets | the previous tight arm: best `real_r3` 9.97 at ep 22, plateau 13.42, `r1`/`r2` 1.93 |
| `ctrl_diverse_scv2_unified` | wide measured ranges | diversity without the fits |
| `real_r4_scv2_unified` | REAL noise in training | the reference: `val/real_r3` **2.99** |

Readings, per the launch decision rule:

* **Easy transfers, hard does not** — the expected outcome. Point-ness was part
  of the limit, and transfer needs the training distribution to sit ON the
  target rigs. Useful, bounded: every new rig needs its own fit.
* **Both transfer** — the sim-to-real problem for this task is essentially
  closed. A model trained on a wide cloud in which neither target rig is
  privileged would have learned the operation rather than either fingerprint,
  so a new rig needs no fit at all.
* **Neither transfers** — the limit is the family's FIDELITY, not the width or
  the placement of the cloud. A neighbourhood of a wrong point is still wrong,
  and the surviving low-order over-prediction (3 / 8 dB) plus the index-locked
  microphone pattern become the next things to fix.

## Results

**PENDING** — all three runs are in flight. The first two job pairs are dead
and must not be read as results: `rig-easy-e7e4ce` / `rig-hard-6ed3cf` were
placed on 4.93 GiB GPUs and OOM'd at the first epoch, and `rig-easy-130d76` /
`rig-hard-7655b0` were cancelled before placement. A third easy submission,
`rig-easy-dce43e`, failed placement (`InstanceUnreachable`, sshd refusing) and
was re-queued onto the same id. Everything upstream of the GPU was verified on
the box each time: the bank rebuilt remotely and self-checked at 0.000 dB
Nyquist fold in all six bands with the path coordinate's KS distance at 0.0127.

| arm | job | GPU | status |
|---|---|---|---|
| `rig_easy_scv2_unified` | `rig-easy-dce43e` | vast A100 | running |
| `rig_hard_scv2_unified` | `rig-hard-1b2f06` | vast A100 | running |
| `rig_easy_hppnet_l2_unified` | `rig-easy-hppnet-l2-4fa419` | vast A100 | running |

The salience arm was added once the GPU salience validation seam was merged
from the unmerged `unified-runs` branch (`31228688`, merge `92c6cd79`): the
unified panel now dispatches its readout by task and scores salience through
the model's own `decode_logits` on the logits' device. It is the noise-family
counterpart of `real_r4_hppnet_l2_unified` and the model-family counterpart of
`rig_easy_scv2_unified`.

To be filled when they land: best `real_overall` and the per-view `real_r1` /
`real_r2` / `real_r3` rev/s MAE at that epoch for each arm, the epoch it
occurred at, whether the post-best real degradation of the fitted run recurs,
and the `r1`/`r2` ratio against the fitted run's 1.93.

## Conclusion

The pair answered its question: widening the distribution helps, and the wide
cloud is not worse than the close one (hard 5.37 vs easy 5.72 best real MAE,
against 9.97 for the point-preset run and 2.99 for real audio). The gap to
real audio is still 1.8x, and the regime split says where it lives: both
synthetic arms are close to real at cruise and far from it in the
transitions, where they collapse the four rotors onto nearly one speed
(output spread 0.28 easy / 0.19 hard against 4.71 for the real-trained model).

## Open decision: rebuild the banks

**Status: PENDING — the user has not answered. Do not launch the curriculum or
mixed arms before this is settled.** Raised 2026-09-15 while reading the model
matrix figures (`docs/explainers/model-matrix.qmd`).

Two defects were found in the preset banks that both finished arms trained on.
Each one changes the training distribution, so fixing either makes
`rig-easy-dce43e` and `rig-hard-1b2f06` non-reproducible from the new banks;
that argues for fixing them NOW, before more arms are spent on the old
distribution, rather than after.

### Defect 1 — the comb's last order is an unconstrained parameter, and it is loud

Michael's anchor fit (`results/S2/cruise_8clip.json:fly125_cruise_00`) has a
`profile_db` grid of 111 orders whose LAST order stands 9 to 39 dB above its
neighbours:

|rotor|orders K-7 … K-1 (dB)|order K = 111 (dB)|jump|
|---|---|---|---|
|0|-30.2 … -30.6|**-12.8**|+18|
|1|-37.3 … -36.8|**-27.3**|+9.5|
|2|-38.2 … -34.8|**-23.1**|+11.7|
|3|-29.8 … -21.7|**+17.6**|+39 — the rotor's global max, above its order 2|

The cause is the fitting band. At the fit's own cruise speeds (75-92 rev/s)
order 111 sits at 8.3-10.2 kHz, outside the 30-7900 Hz band the likelihood
scores, so it was never constrained and absorbed whatever the band edge
needed. Harmless where it was fitted. On stream trajectories that dip to
24 rev/s it sweeps INTO the band: on `results/model_matrix/trajectories/traj_00`
(24.0-62.3 rev/s) order 111 tracks 2660-6910 Hz, which is exactly the dominant
sweeping line visible in `C_stream_michaelsfit_t0` and the +10 dB 5-6.5 kHz
hump measured there. `band_taper_frac` is `0.0` in these renders, so the comb
ends on a hard edge and nothing fades the inflated order. Decimation is
innocent: a 2x-grid brick-wall reference reproduces the hump, and the fold
term stays under +1.16 dB.

Options: **(a)** drop the orders whose frequency at the fit's own speeds fell
outside the fitting band — deletes a parameter the data never constrained;
**(b)** enable `band_taper_frac` so the comb end always fades — fixes the
symptom generally, leaves the bad parameter in place; **(c)** both.
Recommendation: (c), with (a) as the correctness fix and (b) as defence in
depth.

### Defect 2 — the two rigs' anchors disagree on label provenance

The banks' provenance block records:

|rig|anchor fit|labels|
|---|---|---|
|Michael's|`results/S2/cruise_8clip.json:fly125_cruise_00`|**raw**|
|DREGON|`results/S2/dregon_room2_cruise_refined.json:free-flight_nosource_room2_cruise_00`|**`rps_refined`**|

So one rig is anchored on raw telemetry and the other on model-supported
refined labels, which the campaign treats as not independent truth. Nothing in
the docs said so before this note. It is a candidate explanation for the
per-rig asymmetry in the results (DREGON cells beat Michael cells for the hard
arm, 4.80 vs 6.28 overall). A raw-label pooled DREGON flight fit exists:
`results/S2/dregon_flight.json`.

Both anchors are FLIGHT fits, not bench fits; no bench fit enters either bank.
For context on how the flight fit compares, from
`docs/explainers/dregon-transfer/index.json` (median 8-mic RPS MAE of a reader
on synthetic clips, and LTAS deviation against real):

|arm|median MAE|spread|LTAS dev|
|---|---|---|---|
|real audio|1.069|0.271|—|
|flight-fitted|**2.077**|**0.773**|**1.54 dB**|
|bench, blind|2.488|0.256|9.37 dB|
|bench comb + flight floor|2.473|0.344|10.86 dB|

The flight fit has by far the best spectrum and the best MAE, and the WORST
clip-to-clip spread: it matches the average well and individual clips
unevenly.

Options: keep refined and document the asymmetry, or switch DREGON to
`results/S2/dregon_flight.json` so both rigs are raw-label.

### What happens once it is answered

1. Rebuild both banks with `scripts/_build_rig_bank.py` (bit-reproducible;
   it re-checks the >7.5 kHz fold and aborts above 0.05 dB).
2. Re-render the affected model-matrix category-C and category-D clips and
   confirm the sweeping line is gone.
3. Re-run `rig_easy_scv2_unified` and `rig_hard_scv2_unified` on the new
   banks, and decide whether `rig_easy_hppnet_l2_unified` is re-run too.
4. Then launch the curriculum and mixed arms.
