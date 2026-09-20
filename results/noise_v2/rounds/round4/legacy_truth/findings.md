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
