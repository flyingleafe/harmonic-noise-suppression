# Stage 2, synthetic: when are four near-identical rotors separable?

Instrument `src/experiments/noise_model/multirotor.py`, driver
`scripts/noise_v2_multirotor_synth.py`, raw `sweep.json`, aggregates
`summary.json`, figures `fig_profile_error.png` and `fig_criterion.png`.
Every number below is read out of those two JSONs (git `2cef2543`).

**Provenance and convergence.** The planted truth is the committed
`bench_dregon_Motor{1..4}_70__bench.json` — carriers 70.1791 / 67.2551 /
69.5788 / 69.4941 rev/s, `sigma_nu` 0.1852 / 0.1803 / 0.1664 / 1.9355 rad/s,
`lam` 0.15716 / 0.00206 / 0.09028 / 0.00270 s⁻¹, and the `profile_db` rows
verbatim. **All four fits, and the `allMotors_70` fit whose floor shape and
carrier pattern are used, report `optimiser.converged = false`
(`which_converged = "none"`; final `grad_norm` 8321 / 375 / 42332 / 1080 and
3356).** Every statement below is therefore about recovering *those numbers*,
not about recovering a converged optimum.

4 rotors, 8 mics, 24 s at 16 kHz, 3 seeds (11/12/13), comb synthesised over
every order below the Nyquist, estimators scored on the ladder
`k = 1,2,3,4,6,8,12,16,24,32,48,64,80,96,104`. Carriers are the real
`allMotors_70` pattern (gaps 3.02 / 1.08 / 0.83 Hz) with the offsets from
their mean rescaled so only the minimum spacing `delta` moves.

## 1. The resolvability map

Median |profile error| in dB over the cells whose planted line clears 20 dB
SNR in its own analysis band (156 of 180 cells), and the ladder orders whose
median gated error exceeds 3 dB:

| δ (Hz) | E1 | E2-single | E2-multi | E2-multi (oracle steering) | E1 fails at k | E2-single fails at k | E2-multi fails at k |
|---|---|---|---|---|---|---|---|
| 0.01 | 6.23 | 3.41 | 7.04 | 1.58 | 1,3,4,6,8,…,104 (all but 2) | 6,8,12,24,64,96,104 | 8,12,16,24,32,48,64,80,96,104 |
| 0.05 | 6.17 | 3.16 | 5.02 | 1.09 | 1,3,6,8,12,24,48,64,80,96,104 | 6,8,12,24,64,80,96,104 | 8,12,24,32,48,64,80,96,104 |
| 0.20 | 4.12 | 2.83 | **1.78** | 0.36 | 8,24,32,48,64,80,96,104 | 8,12,24,48,64,80,96,104 | 48,64,80,96,104 |
| **0.83** | 3.58 | 2.20 | **0.71** | 0.40 | 24,32,48,64,80,96,104 | 24,48,64,80,96,104 | 64,96,104 |
| 3.00 | 4.63 | 0.95 | **0.37** | 0.19 | 12,16,24,32,48,64,80,96,104 | 24,32,48,64,80,96 | 48,64 |

Controls at δ = 0.83 Hz (same gate, 141 cells):

| control | E1 | E2-single | E2-multi | E2-multi (oracle) |
|---|---|---|---|---|
| baseline | 3.58 | 2.20 | 0.71 | 0.40 |
| steering identical across rotors | 3.24 | 2.41 | **2.01** | **2.39** |
| planted track drift doubled (0.48 Hz) | **7.88** | 2.31 | 0.71 | 0.29 |

At the real minimum spacing **0.83 Hz the per-rotor profiles ARE recoverable**:
E2-multi's median is 0.71 dB over the whole ladder, and at the strictest
40 dB SNR tier (69 cells) it is 0.89 dB with k = 96 the only failing order
(E2-single 3.65 dB, failing at k = 24, 48, 96; E1 11.95 dB, failing from
k = 24 up). E1 is the weakest estimator everywhere: at δ = 0.83 its median is
3.58 dB and **33 % of its gated cells collapse** (estimate floored to zero),
against 12 % for E2-single and 11 % for E2-multi.

## 2. Where each estimator fails, and why

* **E1 (full-length periodogram, global Voronoi cells, cluster total then
  split by mass).** Fails from k ≈ 24 upward at every δ, *including* δ = 3 Hz
  where the order-k spacing is 312 Hz at k = 104. The mechanism is not the
  order spacing but **accidental collisions in a 440-line comb**: four
  incommensurate carriers put ~440 lines into 7.2 kHz, mean spacing ~18 Hz,
  and above k ≈ 24 some line of some rotor lands inside another's linewidth.
  E1 then reports the equal split, which against a 20–40 dB profile contrast
  is a 10–20 dB error. `sweep.json:nearest_line_hz` records the distances;
  ladder cells with a neighbour inside 0.1 Hz number 61 (δ = 0.01), 17
  (δ = 0.05), 3 (δ = 0.2) and 0 at δ ≥ 0.83 — at the larger spacings the
  collisions sit at 0.1–5 Hz, still inside the line core.
* **E2-single (per-line demodulation, windowed LS against the known beat
  bases of every line in the band).** Beats E1 at every δ (2.20 vs 3.58 at
  δ = 0.83; 0.95 vs 4.63 at δ = 3). It fails where the LS window cannot hold
  a beat cycle of the nearest interferer without exceeding the coherence time
  (section 3).
* **E2-multi (one steering vector per rotor, all mics solved jointly).** Best
  estimator for δ ≥ 0.2 Hz: 0.71 dB at δ = 0.83, a factor 3 on E2-single and a
  factor 5 on E1. **It degrades below E2-single for δ ≤ 0.05 Hz** (5.02 and
  7.04 dB) because its steering must be *estimated* from a consecutive block
  of orders that are themselves unresolved there; the oracle variant, handed
  the planted steering, is the best estimator at every δ (1.09 and 1.58 dB).
  The spatial lever therefore survives small δ — the *estimate* of it does
  not.
* **Nothing recovers δ ≤ 0.05 Hz at 24 s beyond the low orders.** Even with
  oracle steering the median is 1.1–1.6 dB and k ≥ 64 fails.

## 3. The criterion — and where the data disagrees with the hypothesis

The hypothesised criterion `k·δ·τ_c(k) ≳ 1` holds, with **two corrections the
data forced**:

1. **It is flat in `k`, not increasing** (left panel of
   `fig_criterion.png`). Every fitted rotor has `lam·τ_c << 1`, so the shaft
   term is ballistic, `τ_c(k) = C/k`, and `k δ τ_c(k) = δ C` is
   order-independent. The worst-neighbour score is 0.033 (δ = 0.01), 0.163,
   0.650, **2.697** (δ = 0.83) and 9.747 (δ = 3), and over k = 1…104 it varies
   by at most 1.5 % for any rotor. Resolvability is therefore **not** an
   order-by-order property: the observed order dependence of the errors comes
   from line SNR and from comb collisions, not from the criterion.
2. **`τ_c` must be the within-record one.** `sigma_nu` is a *stationary*
   spread, but every fitted `lam` is ≤ 0.157 s⁻¹, so over one 24 s support the
   shaft error is a frozen carrier *offset* plus a small wander — and the
   bench fit estimates that offset (it refines, then fits, the carrier). The
   operative scale is `sigma_eff = sigma_nu·sqrt(min(1, 2·lam·T/3))`
   (`multirotor.within_record_spec`): 0.1852 / 0.0327 / 0.1664 / 0.4022 rad/s
   against the fitted 0.1852 / 0.1803 / 0.1664 / 1.9355. For `Motor4_70` that
   is a factor 4.8, and it is the difference between "0.83 Hz is
   unresolvable" (ensemble worst-neighbour score **0.604**) and "0.83 Hz is
   resolvable" (within-record score **2.697**). The ensemble width would have
   predicted failure where the sweep measures 0.71 dB.

The realised beat count `δ_near·T_w` is the operative coordinate (right panel
of `fig_criterion.png`): binned medians of |error| over all δ drop once it
passes 1 — to 3.4 dB (E1), 1.9 dB (E2-single), 1.1 dB (E2-multi) and 0.3 dB
(oracle) — and sit at 2.6–6 dB below it for every estimator.

**On E2-single's lever.** The hypothesis said E2-single beats E1 only through
track demodulation. The doubled-drift control confirms that lever
quantitatively: doubling the planted track drift (0.24 → 0.48 Hz) moves E1
from 3.58 to 7.88 dB and pushes its first failing order from 24 down to 8,
while E2-single (2.20 → 2.31) and E2-multi (0.71 → 0.71) do not move. But
E2-single already beats E1 at the *baseline* drift (2.20 vs 3.58), because the
per-line LS un-mixes overlapping lines that a periodogram can only split by
mass. Track demodulation is one of two levers, not the only one.

**On E2-multi's lever.** Confirmed by the identical-steering control: making
all four rotors share one steering vector costs E2-multi a factor 2.8
(0.71 → 2.01 dB) and the oracle a factor 6 (0.40 → 2.39 dB), and restores
failing orders 48 and 80. Spatial diversity is real and is worth ~4 dB of
profile accuracy at the real spacing.

## 4. The label-identifiability failure, reported separately

`offset_spread_ratio = max_r sigma_nu_r / (2 pi δ)` is 0.37 at δ = 0.83, 1.54
at δ = 0.2, 6.16 at δ = 0.05 and 30.8 at δ = 0.01. Above 1 the realised
per-record carrier offsets can exceed half the spacing and the refined
carriers can cross: what is then unidentifiable is *which* recovered profile
belongs to which rotor, not the set of profiles. The carrier refinement's own
worst-case error over seeds and rotors is 0.018 rev/s at δ = 3, 0.034 at
δ = 0.2, **0.249 at δ = 0.83** and 0.126–0.131 at δ ≤ 0.05 (clipped at the
half-spacing bound). The 0.249 outlier at δ = 0.83 is a single rotor-seed
whose harmonic sum locked onto the wrong side of the room bound; it is the one
mechanism in this study that would put a *wrong* profile on a named rotor
rather than a noisy one.

## 5. A defect in the committed profiles, found on the way

The `profile_db` rows of all four Motor*_70 fits put essentially all of their
power in orders whose line lies **above the fitted band**. With the band
capped at `min(7900, 0.45·sr) = 7200 Hz` the in-band order limit is
k = 102 / 107 / 103 / 103, and:

| rotor | total, all orders | total, in-band only | strongest in-band order |
|---|---|---|---|
| Motor1 | −35.0 dB | −80.0 dB | k = 56, −81.9 dB |
| Motor2 | −37.9 dB | −86.3 dB | k = 58, −89.2 dB |
| Motor3 | −35.8 dB | −87.0 dB | k = 56, −90.9 dB |
| Motor4 | −38.1 dB | −78.5 dB | k = 9, −82.2 dB |

i.e. 43–51 dB of each fitted comb sits in orders the likelihood never saw
(`k·f > 7200 Hz`), where the profile is prior-driven. Two consequences:

* the main sweep is unaffected **as a resolvability study** — it is run at a
  comb-to-floor of 80 dB precisely so the in-band lines clear the floor, and
  every verdict is gated on each line's own in-band SNR;
* **the `delta_0p83_real_snr` case is confounded and has to be re-run.** It
  sets the floor from the *all-orders* comb at the rig fit's measured ratio
  (4.51 dB), so the floor is ~45 dB too high relative to the in-band lines:
  only 3 of 180 ladder cells clear even a 6 dB line SNR (median line SNR
  −39.7 dB) and its estimator numbers (E1 6.70, E2-single 3.95, E2-multi
  14.70 dB, all on those 3 cells) measure the floor, not the separation.
  **What is missing is one re-run with the comb-to-floor ratio defined on
  in-band orders only**; the instrument needs no change, only
  `rig_comb_to_floor_db` and the generator's `k_cap` restricted to
  `floor(7200 / f_max)`. Stated rather than silently dropped.

## 6. Recommendation for stage 3 (a proposal for Main, not implemented)

The measurement says the four profiles at 0.83 Hz are recoverable to ≲1 dB up
to k ≈ 48 by a phase-aware multi-mic solve, and to ≲3 dB by the periodogram
only below k ≈ 24. The rig fit therefore does not need the estimator to
*replace* anything; it needs it where the periodogram likelihood is flat.

1. **Profile initialisation, not a frozen block.** Run
   `estimate_e2(multi=True)` on the real `allMotors_70` support and use its
   `(R, K)` output as the *initial value* of `profile_db` in the rig fit. It
   is a point estimate with a measured error budget, not a measurement with a
   likelihood, so freezing it would import its own 0.7–3 dB bias into every
   downstream number.
2. **An order-dependent prior width, taken from this sweep.** Give the rig
   fit's per-rotor profile a Gaussian prior centred on the E2-multi estimate
   with standard deviation equal to this study's measured per-order error at
   the matching SNR tier: ≈1 dB for k ≤ 16, ≈3 dB for 24 ≤ k ≤ 48, and **no
   prior at all** (the present free parameter) above k ≈ 64, where even the
   oracle estimator fails. That is the only part of the estimator output
   defensible as prior information.
3. **Fit the four carriers from the offsets `refine_offsets` returns**, and
   report `offset_spread_ratio` alongside the fit: at 0.83 Hz it is 0.37, so
   the labels are safe, but `Motor4`'s `sigma_nu = 1.9355 rad/s` is only a
   factor 2.7 below the level at which rotor labels stop being identifiable.
4. **Do not expect the rig fit to reproduce orders above k ≈ 102 at all** —
   they are outside the fitted band (section 5). Any "the rig fit does not
   reproduce the per-rotor fits" comparison must be restricted to in-band
   orders before it means anything.

## 7. Labelled hypotheses for the real four-motor transfer gap

Resolvability at 0.83 Hz is **not** the explanation — the sweep says the
profiles are recoverable there, by E2-multi to 0.71 dB and by E1 itself below
k ≈ 24. Alternatives, as hypotheses only:

* **[H1] Out-of-band profile mass.** Section 5: 43–51 dB of each single-motor
  fitted comb is in unconstrained orders, so a per-rotor-vs-rig comparison
  over all orders is comparing prior samples. Testable immediately by
  restricting the comparison to k ≤ 102.
* **[H2] Non-convergence.** All five fits report `converged = false` with grad
  norms 375–42332; the rig fit's `sigma_nu = 0.9979`, `lam = 128.98`,
  `sigma_eps = (0.3878, 0.2843)` resemble no single-motor fit's, which is what
  a different local basin looks like rather than what four combined rotors
  should look like.
* **[H3] Per-mic steering phase is not in the forward model.** v2 carries one
  mic line gain per (mic, rotor) and no per-order phase; this study measures
  the cross-mic phase structure as worth ~4 dB of profile accuracy
  (section 3). A rig fit that cannot use it is fitting four blurred lines with
  four amplitudes.
* **[H4] Accidental comb collisions.** Section 2: above k ≈ 24 some pair of
  the 440 lines is always inside a linewidth, and a periodogram likelihood
  cannot attribute those. This is a genuine limit of the *likelihood*, not of
  the data, and it is the strongest argument for entering the estimator output
  as a prior rather than trusting the rig fit's own high-order split.
