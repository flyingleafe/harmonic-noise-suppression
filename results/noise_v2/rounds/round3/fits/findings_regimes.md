# R3 PER-REGIME Michael's candidate — standby fit, cruise fit, carrier-gated composition

Written by `R3Standby`. Every number below is read out of a COMMITTED JSON and
carries the convergence status of the fit that produced it. **Neither fit
converged** (both stopped on the L-BFGS iteration cap, `which_converged:
"none"`), exactly as the pooled R3 fit did — so this candidate's numbers are
compared against the pooled one's on equal footing, and no number here is a
statement about a converged optimum.

Sources:

| artefact | path | job |
| --- | --- | --- |
| standby fit | `round3/fits/michaels_fly125_standby__flight.json` | `nv2-r3-michaels-standby--6c860e` (uni-cpu, 5 h wall granted) |
| cruise fit | `round3/fits/michaels_fly125_cruise__flight.json` | `nv2-r3-michaels-cruise-03bf4e` (uni-cpu) |
| score arm | `round3/render/arm_michaels_v2_regimes.json` | `nv2-r3-score-michaels-re-ac8bc5` (colab, GPU, probe sha256 `6e50e025…2877b1` verified in-job) |
| k = 2 comb/floor sanity | `round3/fits/regimes_comb_floor.json` | local, seed 2001, 8 mics |
| code | `supports.set_michaels_standby`, `render.render_noise_regimes`, `noise_v2_round_score.v2_regime_arm` | SHAs `a996e5e5`, `58e895fb`, scored at `d9bd1940` |

## Why

The pooled R3 fit (`michaels_fly125_all__flight.json`, not converged) buys its
ramp and cruise numbers with standby: equal-regime **2.334258** rev/s, cruise
0.693809 (legacy 0.755783) and ramp 3.145259 (legacy 8.007098) BETTER than the
legacy arm, but standby **3.163707** against the legacy **0.317103** — ten
times worse. One speed law over a 4.7x carrier span cannot serve both ends.
The previous generation never pooled them: the frozen evaluator rendered
standby from a STANDBY export and cruise (and the ramp) from the cruise export
(`noise_v2_round_score.LEGACY_BASELINE`), which is what R2's own render-regime
study already recommended reproducing
(`round2/render_regime/findings.md:150`).

## The standby pool: 3 disjoint 4 s windows, 12.0 s

`supports.set_michaels_standby` — every DISJOINT window the standby band
admits at the longest campaign length that admits at least three. FLY125 holds
ONE standby run, **1.775–15.270 s** (13.50 s), on `rps_refined` at the 200 Hz
telemetry grid `clips.windows` reads:

* at **8 s** the anchored grid admits **0** windows and the 1 s-stride
  candidates (2.0 … 7.0) all overlap — at most **ONE** disjoint window;
* at **4 s** the 1 s-stride candidates are 2.0 … 11.0, and packed
  earliest-first without overlap that is **THREE**: **2.0, 6.0, 10.0 s**.

So the pool is 3 x 4 s = **12.0 s** of standby material (the pooled fit had one
8 s window). All three keep every rotor inside `gates.REGIME_BANDS["standby"]`
= 20–45 rev/s for their whole span — asserted on the real label by
`tests/experiments/test_noise_model_supports.py::TestSpecsAndCache::
test_the_standby_pool_is_disjoint_and_inside_the_standby_band`, which also
re-derives the "8 s admits fewer than three" fact rather than trusting it. All
three lie inside spans already declared as Michael's TRAINING material
(`noise_v2_likelihood_window.TRAINING_SUPPORTS`: FLY125 @2+8 and @10+8 cover
2.0–18.0 s), so no held-out measurement moved. FLY124 is in neither pool: the
`michaels-cruise` set lists the five frozen FLY124 score windows so one cache
index describes the round, and `noise_v2_fit._specs:787` drops them.

## The two fits (both NOT converged)

| quantity | standby fit | cruise fit |
| --- | ---: | ---: |
| support / windows | `michaels_fly125_standby`, 3 x 4 s | `michaels_fly125_cruise`, 8 x 8 s |
| `k_max` | 130 | 81 |
| objective `whittle_nats` | **-6 998 258.027** | **-24 370 322.133** |
| `n_cells` | 749 952 | 2 064 384 |
| per cell | **-9.331608** | **-11.805130** |
| per band (comb / floor) | -6 822 120.5 / -176 137.6 | -23 684 580.6 / -685 741.6 |
| `sigma_nu` (rev/s) | **0.328062** | **2.926554** |
| `lam` | 0.5 (model constant, `priors.flight_lam_pin` = 0.5, `pinned_dynamics` null) | 0.5 (same) |
| `gamma_hz` rotor-mean at k = 1 / 2 / 4 / 8 | 0.6401 / 25.4376 / 0.0162 / 1.5229 | 2.3789 / 0.1201 / 3.6811 / 1.5525 |
| `gamma_hz` median over the whole (R, K) block | 1.5748 | 4.9122 |
| `gamma_hz` geometric mean | 1.8825 | 3.7099 |
| `gamma_low_order_check` | `fail_above_floor` (resolution floor 3.906 Hz) | `fail_k2_ramp` |
| `amp_exp` | **4.1265** (FREE) | 2.0 (PINNED at the prior median) |
| `floor_exp` | **4.9279** (FREE) | 2.0 (PINNED) |
| `floor_static_rel` | 5.889e-05 (FREE) | 2.500e-03 (PINNED) |
| `floor_mean_db` / `diagnostics.floor_level_db` | -40.3759 / **-40.1488** | -34.9946 / **-35.3460** |
| `floor_tilt_db_oct` | +0.3478 | -0.0406 |
| measured floor (diagnostics) | -40.3264 dB, 458/520 lines visible | -33.6235 dB, 262/324 lines visible |
| `speed_span` / span rule | **1.9278** > 1.5 -> `span_pins.pinned` EMPTY | **1.4360** < 1.5 -> `amp_exp`, `floor_exp`, `floor_static_rel` pinned |
| converged | **false**, `none`, grad 210.440, restart gain 6.057e-04 (tol 1e-04) | **false**, `none`, grad 1174.077, restart gain 2.995e-03 (tol 1e-04) |
| wall | **11 863.2 s** (3.30 h; Adam 4 412.5 s) | **2 917.9 s** (48.6 min; Adam 1 137.8 s) |

Two expectations from the submission note were WRONG and the fits say so:

1. **The standby pool does NOT span under 1.5x.** It spans **1.9278x** —
   `diagnostics.batch.carrier_min_rev_s` 20.930 to `carrier_max_rev_s` 40.350
   rev/s — because the four rotors themselves differ that much
   while the rig idles (the @10+4 window alone carries 30.99 / 34.54 / 40.15 /
   36.54 rev/s). So the standby fit identified its OWN speed law inside the
   standby band (`amp_exp` 4.13, `floor_exp` 4.93) instead of inheriting the
   prior medians, and the composition never asks it to extrapolate: above 45
   rev/s its weight is exactly zero.
2. **The CRUISE pool is the one the span rule pins.** 68.2–97.9 rev/s is
   1.436x, under the 1.5x threshold, so `amp_exp` = `floor_exp` = 2.0 and
   `floor_static_rel` = 2.5e-3 are constants at their prior medians. This is
   the R3 model's own guard against R1's cruise-only pathology (R1's `/1`
   cruise fit, with no span rule, took `amp_exp` 10.02 / `floor_exp` -10.19);
   and in this candidate the cruise fit's extrapolation to standby is never
   rendered either.

The first fit attempt at `--time 2h` (`nv2-r3-michaels-standby-28a220`) was
cancelled at 01:57Z because a 130-order pool runs Adam at ~11–16 steps/min:
1500 Adam steps plus the pooled job's own L-BFGS share could not fit in 2 h.
The resubmission changed ONLY `--time`.

## The composition rule (`render.render_noise_regimes`)

Per sample, with `w` = `data_processing.rps_gating.regime_weight` — the
previous generation's own regime rule, the smoothstep `3x^2 - 2x^3` in the
SLOWEST rotor's rate, exactly 0 at `STANDBY_MAX_RPS` = **45 rev/s** and
exactly 1 at `CRUISE_MIN_RPS` = **65 rev/s**, with its `settle_s` term
disabled (that term delays TRUST in a refined label after a spool-up; model
parameters have no such history, and keeping it would make the render depend
on how much of the recording came before the window):

    out = sqrt(1 - w) * standby_render + sqrt(w) * cruise_render

The two renders are drawn from INDEPENDENT streams spawned from the arm's one
seed (`render.regime_seeds`), so their powers add and the composed power is
exactly `(1 - w) P_standby + w P_cruise`: the standby fit alone below 45
rev/s, the cruise fit alone at or above 65, and a continuous interpolation of
the two regimes' LEVELS across the ramp band. A regime whose weight is zero
everywhere is not rendered at all, so only a window that actually crosses the
band costs two renders. The likelihood gate reads the matching
`render.expected_periodogram_regimes` — the same blend evaluated per frame at
the frame centre — so the predicted periodogram is the composition's, not one
fit's.

**What the legacy arm actually did**, for the record: its per-regime selection
was a HARD switch on the support's DECLARED regime label — standby took
`results/S2/standby.json`, and BOTH ramp and cruise took the cruise export
(`LEGACY_BASELINE`, `from_regime: cruise`) — which needs a label per window
and cannot render a track that crosses a band at all. The only per-CARRIER
interpolation in the previous generation is the `rps_gating` smoothstep, which
is what this composition reuses verbatim, thresholds included. So the ramp is
interpolated between the two fitted regimes here instead of being handed to
cruise whole.

`test_noise_model_core.py::test_regime_composition_crosses_the_bands_without_a_level_step`
pins it on a synthetic 35 -> 80 rev/s spool-up between two planted fits 12 dB
apart: outside the blend the composition IS each regime's own render (bitwise),
no two adjacent 64 ms blocks differ by more than 3 dB (measured 1.57 dB;
a hard switch at 45 rev/s gives 9.31 dB and one at the midpoint 11.01 dB), and
the block levels track `(1-w) P_s + w P_c` to within 1.5 dB (measured 0.6 dB;
the same sum from ONE shared stream deviates 2.94 dB, because correlated
phases add up to 3 dB of coherent gain mid-blend).

## HPPNet, against the legacy bar and the pooled R3 candidate

`arm_michaels_v2_regimes.json`, candidate `michaels_v2_r3_regimes`, seeds
2001–2004, 8 mics, probe digest verified in-job, cohort complete, all three
regimes present.

| quantity | per-regime (this) | pooled R3 | legacy | bar |
| --- | ---: | ---: | ---: | ---: |
| equal-regime mean PIT MAE (rev/s) | **1.621670** | 2.334258 | 3.026661 | <= 3.177994 -> **within, margin 1.556325** |
| ratio vs the legacy regime mean | **0.535795** | 0.771232 | 1.0 | <= 1.05 |
| standby | **0.785384** (ratio 2.4768) | 3.163707 (9.9769) | 0.317103 | — |
| ramp | **3.267119** (ratio 0.4080) | 3.145259 (0.3928) | 8.007098 | — |
| cruise | **0.812507** (ratio 1.0751) | 0.693809 (0.9180) | 0.755783 | — |
| proxy `ltas_abs_db`, michaels cruise | **1.281622** (margin **-0.061953**) | 1.886305 (-0.666637) | — | <= 1.219668 -> **FAIL** |
| proxy `mr_ltas` | 1.461482 | — | — | — |
| likelihood comb margin (nats/s) | **-1749.800** (below oracle ✓) | -862.83 | — | below oracle |
| likelihood floor margin (nats/s) | +368.038 | +201.56 | — | — |
| likelihood gate | **pass** | pass | — | — |

Per support (mean over the four seeds, seed spread beside it, real arm for
scale):

| support | regime | mean rps | candidate | seed spread | real arm | `ltas_abs_db` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FLY124@8+8 | standby | 36.39 | 0.599740 | 0.4656 | 0.398474 | 0.5012 |
| FLY124@16+8 | standby | 36.39 | 0.971028 | 1.7904 | 0.289978 | 0.3418 |
| FLY124@27.68+8 | ramp | 56.82 | 3.267119 | 0.3250 | 3.171148 | 3.6250 |
| FLY124@40+8 | cruise | 80.93 | 1.076823 | 0.2007 | 0.884854 | 0.4727 |
| FLY124@56+8 | cruise | 80.69 | 0.548191 | 0.1437 | 0.290248 | 2.0906 |

Total render wall 433 s for 5 supports x 4 seeds (the ramp support is the only
one that renders twice).

**The standby regime improves 4.03x on the pooled candidate** (3.163707 ->
0.785384) at a cost of 0.119 rev/s on cruise (0.693809 -> 0.812507) and 0.122
on the ramp (3.145259 -> 3.267119); the equal-regime mean falls from 2.334258
to 1.621670, i.e. the parity margin grows from 0.843736 to **1.556325** rev/s,
and the proxy — still a FAIL — moves from -0.667 dB to **-0.062 dB** of its
gate, which is now within one seed's LTAS spread (1.618 dB) of passing. The
standby regime is still 2.48x the legacy 0.317103: the legacy standby export is
fitted on the same 13.5 s of FLY125 standby and is a stationary per-clip model,
while this arm's standby fit is a NOT-CONVERGED 130-order flight fit whose
low-order widths sit above the window's 3.906 Hz resolution floor
(`gamma_low_order_check: fail_above_floor`).

`pass = false` / `parity_pass = false` for the arm record as a whole because it
carries ONE rig (the DREGON cohort is empty in a michaels-only arm, exactly as
in the pooled R3 and R2 michaels arms); the michaels frozen gate itself is
`frozen_gate_pass = true`.

## k = 2 comb/floor sanity of the composition (`regimes_comb_floor.json`)

`scripts/noise_v2_render_regime.py` is single-fit by construction (one
`--fit`, and its twelve hypothesis arms are `build(fit, cache)` closures over
that one payload), so it was NOT modified; instead its own measurement
functions (`study_supports`, `arm_diagnostics`, `comb_to_floor_db`) were run
on the composed render, seed 2001, 8 mics. `mean_db` over rotors, in dB:

| support | arm | k = 2 | k = 4 | k = 8 | band level (dB) | `ltas_mean_abs_db` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| standby (w ≡ 0) | real | 15.12 | 6.86 | 5.56 | -2.81 | — |
| | legacy | 16.42 | 8.83 | 6.34 | -3.36 | 0.590 |
| | **composed** | **17.07** | 10.49 | 6.59 | -3.00 | 0.501 |
| | cruise fit alone | 25.87 | 11.64 | 9.61 | +2.82 | 4.524 |
| ramp (w 0 -> 1, 16.18 % of samples in the blend) | real | 6.89 | 2.81 | 1.75 | +9.33 | — |
| | legacy | 6.20 | 3.41 | 4.05 | +6.35 | 4.188 |
| | **composed** | **8.89** | 1.89 | 3.17 | +7.28 | 3.625 |
| | standby fit alone | 1.39 | 1.70 | 2.97 | +9.41 | 2.381 |
| | cruise fit alone | 6.93 | 3.10 | 3.25 | +7.95 | 2.929 |
| cruise (w ≡ 1) | real | 25.33 | 11.02 | 2.44 | +12.16 | — |
| | legacy | 30.73 | 15.43 | 3.44 | +10.21 | 0.983 |
| | **composed** | **26.88** | 9.94 | 3.29 | +10.90 | 0.473 |
| | standby fit alone | 15.38 | 4.32 | 1.63 | +13.90 | 3.472 |

This is the composition doing its job, on real FLY124 carriers: on the standby
support the cruise fit is not rendered at all (it would sit **+10.75 dB** over
the real k = 2 contrast and 4.52 dB off in LTAS), on the cruise support the
standby fit is not rendered (it would sit **-9.95 dB** under it), and each
regime's own render is reproduced exactly where its weight is one. On the ramp
the composed contrast (8.89 dB) is between the real 6.89 and neither single
arm's, because 16.2 % of that window's samples are a level interpolation of
the two.

## What is left

* Both fits stopped on the iteration cap. A converged standby fit is the
  obvious next lever on the one regime still above legacy, and the restart
  gain (6.06e-04 nats/cell against a 1e-04 tolerance) says the standby
  objective was still moving when the cap hit.
* The proxy gate is -0.062 dB from passing on this candidate; its two cruise
  supports' own LTAS deviations are 0.473 and 2.091 dB, so the whole deficit
  sits in ONE support (FLY124@56+8).
* The round record is composed by whoever owns both rigs' arms:
  `--compose … --arm-job michaels_v2_r3_regimes=nv2-r3-score-michaels-re-ac8bc5`.
