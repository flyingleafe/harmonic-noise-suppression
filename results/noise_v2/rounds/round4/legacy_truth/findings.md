# R4 legacy-truth: what the LEGACY DREGON render carries that v2's does not

HPPNet PIT MAE over the three DREGON room-2 cruise score windows: real 1.074,
legacy 1.963, the round-3 v2 candidate 69.555 rev/s
(`results/noise_v2/rounds/round3/dregon_humps/widen.md`). Round 3 closed off
the easy explanations — the v2 render is faithful to its own forward model to
+0.05 dB in power-sum in every cell class, the fit is within +2..+4 dB of the
real periodogram uniformly, line width does not move the tracker and neither
does phase structure at the fitted level. This round takes the legacy render as
the DATA: what does it carry, can v2 express it, and which parameters carry it.

Full tables: `anatomy.json` / `anatomy.md` (A), `fits/` + `score_legacy_fit.json`
(B), `swap.json` (C). Figure: `prominence_ladder.png`.

## A. Anatomy: the three qualities that differ

Measured on the renders of the three score windows, seed 2001, 8 microphones,
order-tracked on each frame's own `motors_command` carrier
(`noise_v2_widen_dregon.line_width_db3`'s own statistic, reproduced to the
digit at k=1: real +6.16 / legacy +18.20 / v2 +7.36 dB on `free-flight`,
against widen.md's +6.16 / +18.20 / +7.36).

| quality | real | legacy | v2 R3 | verdict |
|---|---:|---:|---:|---|
| prominence over local floor, k=1 (dB, 3-window mean) | +5.77 | **+17.67** | +7.42 | **DIFFERS: legacy +10.2 dB over v2** |
| prominence over local floor, k=2 (dB) | +2.08 (2 windows) | **+7.08** | +2.79 | **DIFFERS: legacy +4.3 dB over v2** |
| prominence, k=3..8 (dB, mean over identified cells) | +0.86 (7) | +1.47 (14) | +0.59 (10) | differs by ~0.9 dB only |
| prominence, k=9..32 (dB, mean over identified cells) | +0.22 (31) | +0.47 (61) | +0.25 (62) | SAME, all at the floor |
| per-order comb level k=1, v2 units (dB) | — | **-24.07** | -46.04 | **DIFFERS: +21.98 dB** |
| per-order comb level k=2, v2 units (dB) | — | -30.32 | -51.57 | +21.24 dB |
| -3 dB line width k=1 (Hz) | 11.8-20.7 | 10.4-11.6 | 11.6-12.2 | SAME (all four-rotor spread) |
| measured local floor slope (dB/oct) | -4.68..-5.41 | -4.46..-4.90 | -4.54..-5.02 | SAME within 0.5 dB/oct |
| cross-order phase coherence B(1,2) | 0.079-0.129 | 0.124-0.215 | 0.097-0.131 | tracks LEVEL, not model (see below) |

**1. Low-order comb LEVEL is the difference, and it is a parameter-level fact,
not only a rendered one.** The legacy export's own per-order level, converted
into v2's units by the one stated conversion
(`revised_eval.legacy_profile_db_to_candidate`, -39.03 dB), is **+21.98 dB**
above the v2 flight fit's at k=1 and **+21.24 dB** at k=2 (and +28..+36 dB at
k=3..12, where neither model's line is visible). This is the same +21 dB the
round-3 level sweep found empirically, read off the two fitted parameter sets
directly. The v2 fit is not missing a gain stage: its `comb_gain_db` is
-6.92 dB and its k=1 `low_order_gain_db` is already +13.86 dB, and the gap
survives both. The rendered consequence is smaller than the parameter gap
because most of legacy's low-order power goes into the 13 Hz-wide pedestal
rather than into the needle: +10.2 dB of measured k=1 prominence, not +22.

**2. The width law differs by three orders of magnitude and does not show.**
Legacy renders a Lorentzian pedestal of `13.07 + 0.211 k` Hz half-width (13.3
Hz at k=1, 19.8 at k=32); v2's fitted `gamma_rk` is 0.0014 Hz at k=1 and 6.09
Hz at k=32. On a 4 s DREGON window neither is resolvable: the measured -3 dB
widths are 10.4-12.2 Hz for both and 11.8-20.7 Hz for the real clip, because at
k=1 every arm's "line" is the four-rotor spread convolved with the frame.

**3. The floor slope is the same; the floor MODEL is not.** Measured local
floor slope: real -4.68/-4.86/-5.41, legacy -4.67/-4.46/-4.90, v2
-4.54/-4.83/-5.02 dB/octave. The parameterisations disagree wildly
(`floor_tilt_db_oct` -2.10 legacy against -7.20 v2; `floor_exp` -3.71 against
+22.82; `floor_static_rel` 0.015 against 3.81) and land in the same place,
because the tilt is only one term of a 14-control-point shape and the speed
exponents are read at a 1.7x carrier span.

**4. Cross-order phase coherence is NOT a quality that separates them.** The
legacy renderer IS the more deterministic one by construction — its coherent
share `w_k = exp(-(k/1.696)^2)` (0.706 at k=1, 0.249 at k=2, 0.004 at k=4)
rides a tone bank at exactly `k * phi_r(t)` of the label track with
`shaft_jitter_rps = 0` and `phase_diffusion_hz_per_order = 0`, i.e. an exact
k-fold phase relation, while v2 gives every order an INDEPENDENT Wiener phase
on top of a shared integrated-OU shaft. Measured, that shows as
`|E[exp(i(phi_2k - 2 phi_k))]|` = 0.215/0.205/0.124 (legacy) against
0.115/0.131/0.097 (v2) and 0.096/0.129/0.079 (real) — but the CONTROL settles
it: v2 with nothing changed but the comb level (+21 dB) scores
**0.390/0.391/0.415**, far above legacy. The statistic measures comb-to-floor
contrast in the +-16 Hz band, not a phase-model difference; legacy's higher
value is a consequence of its 10 dB louder k=1 line. Above the first pair every
arm sits at the floor-dominated 0.05-0.09.

**Composition.** The two models reach a free-flight comb by different routes,
and this is why the level gap can exist at all. The legacy DREGON arm is
**fitted on the flight recording itself** — `results/S2/dregon_room2_cruise_refined.json`
holds one MAP fit per room-2 flight clip and the arm is identity-matched to the
scored recording, so there is no bench->flight transplant and the flight
periodogram alone sets every level. v2 transplants the comb from four DREGON
single-motor 70% BENCH fits (log-mean of rates/scales/widths, dB-mean of the
per-order profile) and lets the flight pool move only the floor, the mic gains,
one shared `comb_gain_db` and one gain per order for k <= 8. Both then scale
the comb by `(f_r/80)^amp_exp`: legacy `amp_exp` 8.756, which at these windows'
own mean carriers (80.4/80.9/79.3 rev/s) is +0.19/+0.43/-0.33 dB, and v2
`amp_exp` 0.000, exactly 0 dB. The speed law is therefore not the gap either.

## B. Can v2 express the legacy render? YES — in the mode that frees the profile

The legacy model was rendered (seed 2001, 8 mics, identity-matched per
recording) on the frozen `motors_command` labels of **exactly the five 8 s
windows the R3 v2 DREGON flight fit pooled**, and a v2 support was built from
that audio through the new `supports.synthetic_support` (same periodogram
route, same `.npz`, spec `synthetic:legacy_<name>`; `supports.json`). v2 was
then fitted to it in two modes, 4 restarts each as separate cluster jobs
(`submit_note.md`; 2 of 4 landed per mode inside the wall, the other two are
still running — the restart spread over the pair that landed is 5.1e-6 and
4.9e-6 nats/cell, i.e. at the convergence tolerance, and seed 2 is the
reported restart in both).

| fit | mode | converged | nats/cell | HPPNet mean | free-flight | hovering | updown |
|---|---|---|---:|---:|---:|---:|---:|
| real DREGON clip | — | — | — | **1.074** | 0.638 | 0.888 | 1.695 |
| legacy render (the bar) | — | — | — | **1.963** | 1.496 | 1.895 | 2.499 |
| v2 fitted to the REAL clip (R3) | `flight_floor_lowk` | False (`none`) | -8.5930 | **69.555** | 62.105 | 73.448 | 73.112 |
| v2 fitted to the LEGACY render | `flight_floor_lowk` | False (`none`) | -8.2083 | **5.560** | 8.481 | 3.915 | 4.283 |
| v2 fitted to the LEGACY render | `flight` (free profile) | False (`none`) | -8.3132 | **1.746** | 1.497 | 1.616 | 2.127 |

**PASS.** The free-profile fit to the legacy render scores 1.746 rev/s, under
the 2.19 bar and under the legacy render's own 1.963 — v2's generative form is
not what fails on DREGON. The R3 mode (`flight_floor_lowk`, bench comb frozen
above k=8) reaches 5.560: a 12.5x improvement on the same mode fitted to the
real clip, but not the bar, because the transplanted bench SHAPE above k=8 is
not the legacy comb's shape and only one global gain can move it.

What the fits took, against the R3 fit-to-real in the same mode:

| parameter | fit to REAL (R3) | fit to LEGACY, `flight_floor_lowk` | fit to LEGACY, `flight` |
|---|---:|---:|---:|
| `comb_gain_db` | -6.918 | **+0.797** | — (no transplant) |
| `low_order_gain_db` k=1..8 | +13.86, +3.67, -0.87, -4.89, +0.65, -9.20, -0.97, -5.70 | **+23.13, +8.74, +2.98, -3.20, +2.34, -5.88, +0.09, +2.70** | — |
| `profile_db` k=1..8, rotor-mean dB | -46.04, -51.57, -73.35, -69.87, -75.66, -74.89, -78.46, -73.11 | **-29.06, -38.78, -61.78, -60.47, -66.26, -63.85, -69.69, -56.99** | -32.32, -38.61, -44.91, -46.22, -50.07, -49.71, -55.42, -50.81 |
| `amp_exp` | 0.000 | 0.000 (pinned) | 3.356 |
| `sigma_nu` / `lam` | 0.7423 / 0.0608 (frozen) | 0.7423 / 0.0608 (frozen) | 5.842 / 0.500 |
| `gamma_hz` k=1 / 8 / 32 | 0.0014 / 0.685 / 6.09 (frozen) | 0.0014 / 0.685 / 6.09 (frozen) | 0.649 / 11.38 / 0.184 |
| `floor_mean_db` | -38.273 | -35.145 | -35.463 |
| `floor_tilt_db_oct` | -7.201 | -8.205 | -5.030 |
| `floor_exp` / `floor_static_rel` | 22.817 / 3.808 | 3.379 / 0.014 | 3.202 / 0.001 |

The comb the legacy render asks for is **+17.0 dB** at k=1 and **+12.8 dB** at
k=2 over the comb the real clip asks for, in v2's own `profile_db` — the same
+21 dB the parameter anatomy reads off the legacy export (§A.1), minus the
share legacy puts in its pedestal.

**Did the fit reproduce the +18 dB prominence? Yes.** 3-window mean prominence
over the local floor:

| arm | k=1 | k=2 | k=3 | k=4 | k=6 | k=8 | floor slope dB/oct |
|---|---:|---:|---:|---:|---:|---:|---:|
| legacy render | +17.67 | +7.08 | +1.74 | +2.42 | +1.19 | +0.86 | -4.68 |
| v2 fitted to REAL (R3) | +7.42 | +2.79 | +0.72 | +0.83 | +0.42 | +0.22 | -4.79 |
| v2 fitted to LEGACY, `flight_floor_lowk` | **+18.48** | **+8.16** | +0.78 | +0.90 | +0.54 | +0.88 | -4.95 |
| v2 fitted to LEGACY, `flight` | **+15.73** | **+6.22** | +1.92 | +2.37 | +1.50 | +0.43 | -4.87 |

Cell-class residual of each render against the LEGACY render's own periodogram
(2048/512, all 8 mics, 3-window mean of the median and of the power-sum ratio,
`noise_v2_render_dregon.cell_classes`):

| arm | comb k<=8 | comb 8<k<=40 | floor |
|---|---|---|---|
| v2 fitted to REAL (R3) | -1.22 / **-6.32** | -0.77 / -1.25 | -0.92 / -0.51 |
| v2 fitted to LEGACY, `flight_floor_lowk` | +1.92 / +1.86 | +0.62 / +0.34 | +0.44 / +0.93 |
| v2 fitted to LEGACY, `flight` | +0.70 / **+0.35** | -0.01 / +0.03 | -0.17 / +0.73 |

median / power-sum dB. The free-profile fit lands within 0.4 dB of the legacy
render in every class; the R3-mode fit within 1.9 dB; the R3 fit-to-real is
6.3 dB short in the low-order comb cells alone, with the floor matched — which
is exactly the deficit HPPNet reads.

## C. Attribution by parameter swap

`P_real` = the R3 `flight_floor_lowk` fit to the real clip, `P_legacy` = the
`flight_floor_lowk` fit to the legacy render (both the same parameterisation on
the same frozen bench comb, so the six groups below are **exhaustive**: the
`all` arm reproduces the target's parameters exactly, asserted in
`swap_prelim.json` as `swap.exhaustive: true`, and reproduces its PIT to the
digit). `profile_db` already carries both gains, so a gain swap re-folds the
difference into `profile_db` — the swap is a swap of the rendered comb, not of
a record field.

| swap | free-flight | hovering | updown | mean | vs its origin |
|---|---:|---:|---:|---:|---:|
| `P_real` (origin) | 62.105 | 73.448 | 73.112 | 69.555 | — |
| + `comb_gain` only | 34.453 | 41.699 | 44.939 | 40.364 | -29.19 |
| + `low_order` only | 50.810 | 55.213 | 68.538 | 58.187 | -11.37 |
| + `floor` only | 46.187 | 51.282 | 43.505 | 46.991 | -22.56 |
| + `mic` only | 70.729 | 71.949 | 70.439 | 71.039 | +1.48 |
| + `gamma` only | 62.105 | 73.448 | 73.112 | 69.555 | 0.00 |
| + `dynamics` only | 62.105 | 73.448 | 73.112 | 69.555 | 0.00 |
| + `all` | 8.481 | 3.915 | 4.283 | 5.560 | -63.99 |
| `P_legacy` (origin) | 8.481 | 3.915 | 4.283 | 5.560 | — |
| - `comb_gain` (to real) | 32.375 | 29.810 | 20.291 | 27.492 | +21.93 |
| - `low_order` (to real) | 37.122 | 36.588 | 23.135 | 32.282 | +26.72 |
| - `floor` (to real) | 39.406 | 27.945 | 40.848 | 36.066 | +30.51 |
| - `mic` (to real) | 10.862 | 14.300 | 3.371 | 9.511 | +3.95 |
| - `gamma` (to real) | 8.481 | 3.915 | 4.283 | 5.560 | 0.00 |
| - `dynamics` (to real) | 8.481 | 3.915 | 4.283 | 5.560 | 0.00 |
| - `all` (to real) | 62.105 | 73.448 | 73.112 | 69.555 | +63.99 |

**No single group carries it; three do, jointly.** `gamma_hz`, `sigma_nu` and
`lam` are exact no-ops — in `flight_floor_lowk` they arrive frozen from the
same bench mapping in both fits, so the width law and the shaft dynamics are
provably not the lever. `mic` is worth 1.5-4.0 rev/s. The three that matter are
`comb_gain_db`, `low_order_gain_db` and the floor block: each alone moves
P_real only from 69.6 to 40-59, and removing any ONE of them from P_legacy
costs 22-31 rev/s of the 64 rev/s total. What they share is the comb-to-floor
contrast; they are three ways of writing the same physical quantity.

The dB of low-order prominence each group adds to `P_real` (same estimator):

| swap | k=1 | k=2 | k=1 gain | HPPNet mean |
|---|---:|---:|---:|---:|
| `P_real` | +7.42 | +2.79 | — | 69.555 |
| + `comb_gain` | +13.97 | +8.73 | **+6.55** | 40.364 |
| + `low_order` | +15.45 | +6.43 | **+8.03** | 58.187 |
| + `floor` | +6.98 | +2.27 | **-0.44** | 46.991 |
| + `all` = `P_legacy` | +18.48 | +8.16 | **+11.06** | 5.560 |

Two readings this forces:

1. **Low-order prominence is necessary but not sufficient.** `low_order` alone
   buys MORE k=1 prominence than `comb_gain` alone (+8.03 against +6.55 dB) and
   scores WORSE (58.2 against 40.4). The difference is that `comb_gain_db`
   lifts every order to k=88 while `low_order_gain_db` lifts only k<=8: the
   tracker also needs the comb ABOVE k=8, i.e. above ~640 Hz, which the k<=8
   block cannot give it.
2. **The floor matters at constant low-order prominence.** The `floor` swap
   changes the k=1 prominence by -0.44 dB and still takes 22.6 rev/s off
   P_real. It lowers `floor_mean_db` by 3.1 dB, steepens the tilt by 1.0
   dB/octave and replaces `floor_exp` 22.8 / `floor_static_rel` 3.81 by 3.4 /
   0.014 — i.e. it changes the broadband floor the whole CQT sees, not the
   +-0.7 fbar annulus the order-tracked prominence is read against.

## (d) What this says about what triggers the tracker on DREGON-like data

A purely spectral statement, with no claim about HPPNet internals. On DREGON
cruise the tracker locks when the comb stands out of the floor **across the
band**, not at the first two orders only:

* it needs roughly **+17 dB** of k=1 and **+7 dB** of k=2 carrier-locked
  prominence over the local floor (legacy +17.67/+7.08, v2-fit-to-legacy
  +18.48/+8.16 and +15.73/+6.22, all tracked at 1.7-5.6 rev/s; the real clip's
  own +5.77/+2.08 is tracked at 1.07 because the real clip carries structure no
  render has);
* that prominence must extend to the HIGH orders too: at matched or better
  k=1 prominence, lifting only k<=8 scores 58.2 while lifting all k scores
  40.4;
* the broadband FLOOR level and slope are part of the same lever: -3.1 dB of
  floor mean and -1.0 dB/octave of tilt are worth 22.6 rev/s at unchanged
  low-order prominence;
* line WIDTH and shaft/line phase statistics are not the lever — the width law
  differs by three orders of magnitude between the two models with no measured
  width difference (§A.2), and in the swap `gamma_hz`, `sigma_nu` and `lam` are
  exact no-ops.

## (e) Recommendation for R5

Pin the comb-to-floor contrast, not the comb alone. Concretely: fit the DREGON
flight pool with `comb_gain_db` and `low_order_gain_db` under a prior centred
at the legacy-like values this round identified — `comb_gain_db` +0.80 dB and
`low_order_gain_db` +23.1, +8.7, +3.0, -3.2, +2.3, -5.9, +0.1, +2.7 dB, with
the floor block at `floor_mean_db` -35.1, `floor_tilt_db_oct` -8.2,
`floor_exp` 3.4, `floor_static_rel` 0.014 — and report the Whittle cost of the
pin explicitly. Prefer a mode that frees the per-order profile above k=8 (mode
`flight`, or `flight_floor_lowk` with `--low-orders` raised): the R3 mode is
1.9 dB from the legacy render in the low-order comb cells and 5.56 rev/s from
its score, and the free-profile mode is 0.4 dB and 1.75 rev/s.

**The honest caveat.** The real recording does not support these values. R3
measured the v2 fit-to-real to be within +2..+4 dB of the real periodogram
UNIFORMLY across comb and floor cell classes (`render_vs_model.md`), and this
round's cell-class read says the fit-to-real is only 6.3 dB below the LEGACY
render in the low-order comb cells while matching the floor — so a legacy-like
pin costs real-data likelihood in exchange for tracker realism. The legacy
export is not evidence about the physics either: it is a per-clip MAP fit OF
THE FLIGHT CLIP ITSELF with a coherent-needle line model, and its k=1 level is
+22 dB above what the transplanted bench comb implies. R5 should therefore
report the pin as a DELIBERATE trade — synthetic-data trackability bought with
a quantified Whittle penalty — and never as a better description of DREGON.
