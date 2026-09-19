# Noise model v2 — R3 pooled Michael's flight fit (`michaels_fly125_all`)

Fit `results/noise_v2/rounds/round3/fits/michaels_fly125_all__flight.json`,
schema `noise-v2-fit/2`, code SHA `01e1464bc0c5b424c3d65abca4e690f6ae6acc82`
(the JSON's `git` field), job `nv2-r3-michaels-flight-8197c4` (`uni-cpu`,
`--gpus 0 --cpus 8 --mem 48 --time 3h`), submitted 2026-09-18T22:51:11Z,
started 22:51:24Z, `succeeded` 00:26:50Z with `R3FLIGHT_DONE exit_build=0
exit_fit=0`. Every number below is read from that committed JSON; nothing is
recomputed except the nats/cell ratios and the prior intervals, both stated
with their inputs.

**CONVERGENCE — read this before any number below.** The fit is **NOT
converged**: `optimiser.converged = false`, `optimiser.which_converged =
"none"`, `lbfgs_restart_gain_per_cell = 9.0618e-03` nats/cell against
`tol_nats_per_cell = 1e-4` (91x over), `grad_norm = 5430.15`, `log_box_hits =
0`. This is the SAME failure mode as R2 (1.8505e-02, |grad| 4726.58) and R1
(1.8756e-02, |grad| 8555.3), halved once more in restart gain: the L-BFGS
restart is still improving when the 200 + 100 iteration budget runs out. Every
parameter, objective and render number quoted below is therefore a property of
**this non-converged point**, and the label is repeated at each table.

## Pool (10 windows, `michaels-all`, the R2 pool unchanged)

`params.supports` — 8 cruise + 1 standby + 1 ramp FLY125 fit windows of 8 s:

| window (8 s) | regime |
|---|---|
| `flight_michaels_FLY125@2.000+8_rps_refined` | **standby** |
| `flight_michaels_FLY125@11.588+8_rps_refined` | **ramp** |
| `flight_michaels_FLY125@{16,32,48,64,96,112,128,144}.000+8_rps_refined` | cruise (8) |

4 rotors, 8 mics, `k_max = 81`, `sr = 16000`, front end `n_fft/hop = 2048/512`
(work grid 64000 / 8192). `diagnostics.batch`: `n_frames_total = 2470`,
`frame_stride = 4`, `n_frames_used = 250`, `exposure_scale = 9.88`,
`carrier_min_rev_s = 20.9301`, `carrier_max_rev_s = 97.9103`, **`speed_span =
4.678x`** — bit-identical to R2's pool, as intended (same set, same SHA of
`supports.set_michaels_all`, rebuilt in-job).

`diagnostics.span_pins`: `speed_span = 4.678` against `threshold = 1.5`,
`pinned = []`, `values = {}` — **no speed law was pinned**, both exponents and
`floor_static_rel` are FITTED. That is the R3 span rule doing nothing here
because the pool is wide enough, which is exactly the check it exists for.

## Objective (NOT converged)

| quantity | value | cells | nats/cell |
|---|---:|---:|---:|
| total `objective.whittle_nats` | −33 523 981.1960 | 2 016 000 | **−16.62896** |
| comb band (> 300 Hz) | −32 619 473.6214 | 1 946 000 | −16.76232 |
| floor band (30–300 Hz) | −904 507.5746 | 70 000 | −12.92154 |

Band edges `objective.band_hz = [30.0, 7898.4375]`, `band_split_hz = 300.0`,
`exposure_scale = 9.88`. Against R2's non-converged fit on the same pool, same
cells and the same frame budget (−33 465 417.4347 / 2 016 000 = −16.59991):
**R3 is 0.02905 nats/cell BETTER overall** (comb −16.76232 vs −16.73468, i.e.
0.02763 better; floor −12.92154 vs −12.85314, 0.06840 better). Both points are
non-converged, so this is a comparison of two stopping points, not of two
optima — but R3 reaches a lower objective with FOUR free scalars fewer in the
dynamics (R1/R2's `sigma_eps_even/odd`, `lam_eps_even/odd` are gone) and the
per-line `gamma_rk` block in their place.

**No per-regime (per-support) objective breakdown exists in the JSON** —
schema `/2` records only the two-band split above. The per-regime Whittle would
need a re-evaluation pass over the 10 windows; it is not quoted rather than
invented.

## Dynamics, widths, exponents, floor (NOT converged)

| parameter | value | status |
|---|---:|---|
| `sigma_nu` | **1.23332 rad/s** | free — R2: 5.1716 (R1 cruise 8.3785) |
| `lam` | **0.5 /s** | **PINNED by the model in flight** — `priors.flight_lam_pin = 0.5`, and `optimiser.pinned_dynamics = null` because no `--pin` was passed: in flight `lam` has no Pyro site at all (`fit.seeds`/`fit._seed_params` take `MD.flight_lam`), so 0.5 is a CONSTANT, not a fitted-then-held value. Verified in the unit summary: the job printed `R3FLIGHT sigma_nu 1.23331800032961 lam 0.5 pinned None`. |
| `amp_exp` (comb speed exponent) | **4.16118** | free — R2: 3.90290 |
| `floor_exp` (floor speed exponent) | **2.00468** | free — R2: 3.22426 |
| `floor_static_rel` | 0.00028766 | free — R2: 0.00196440 |
| `floor_mean_db` | −38.79900 dB | free — R2: −38.18370 |
| `diagnostics.floor_level_db` | **−39.08072 dB** | derived (`floor_mean_db` + mean `mic_floor_db`) — R2: −38.27220 |
| `floor_tilt_db_oct` | **−0.75078 dB/oct** | free — R2: +0.30305 |

`gamma_hz` — the R3 per-line Lorentzian half-widths, `(4 rotors, 81 orders)`,
at the ladder (`diagnostics.gamma_hz_at`, Hz, rotor 0/1/2/3 and the geometric
mean over rotors):

| k | rotor 0 | rotor 1 | rotor 2 | rotor 3 | geo-mean | prior median `0.01k` | prior +2 sd |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.002428 | 2.98993 | 1.35307 | 2.37779 | 0.39094 | 0.01 | 0.0739 |
| 2 | 0.0000558 | 0.000387 | 1.19854 | 0.003375 | 0.00306 | 0.02 | 0.1478 |
| 4 | 0.38811 | 0.35166 | 9.07899 | 0.08931 | 0.57677 | 0.04 | 0.2956 |
| 8 | 3.09428 | 0.87506 | 0.33353 | 0.63349 | 0.86970 | 0.08 | 0.5911 |
| 16 | 9.11731 | 4.41216 | 1.13966 | 14.41193 | 5.06996 | 0.16 | 1.1822 |
| 32 | 28.89875 | 24.59408 | 1.82630 | 34.25424 | — | 0.32 | 2.3644 |

`diagnostics.gamma_low_order_check`: **verdict `pass`**, `passed = true`,
`k_checked = [1,2,3,4]`, `resolution_hz = 3.90625` (the 8 s window's `1/(2T)`
at this front end), `max_over_resolution = 2.3242` (rotor 2 at k = 4: 9.079 Hz
= 2.32 x the resolution, inside the factor-3 band), `k4_over_k1 = 159.84`. The
ramp ratio is large but NOT decisive by the rule R3Model fixed at `01e1464b`:
the floor test governs, and every low-order width sits within a factor of 3 of
a bin, so the widths at `k <= 4` are not resolvable structure and their ratio
is noise. The HIGH orders are another matter — from k = 16 up the widths run
4–34 Hz, an order of magnitude above the prior's +2 sd, i.e. the data (not the
prior) is asking for broad lines at high k.

Comb profile (`params.profile.profile_db`, 81 orders, dB): rotor 0 peak −29.57
at k = 2, median −57.70; rotor 1 peak −35.96 (k = 2), median −56.77; rotor 2
peak −41.02 (k = 2), median −62.50; rotor 3 peak −46.63 (k = 2), median −61.75.
`mic_gains_db` −5.95 … +3.89, `mic_floor_db` −2.52 … +1.93,
`mic_line_gain_db` −4.18 … +5.70 — all inside the ±6 dB prior sd.

**Prior-edge flags** (central 95 % of the R3 priors in the JSON's own `priors`
block): `sigma_nu = 1.2333` is ABOVE `LN(log 0.3, 0.5)`'s [0.1104, 0.8155] —
the model's own docstring calls a flight fit that wants more than 2 rad/s a
model error, so 1.23 is high but under that line, and it is 4.2x SMALLER than
R2's 5.17 on the same pool, which is the R3 line-width block taking work off
the shaft. `amp_exp = 4.1612` is just outside `N(2, 1)`'s [0.0, 4.0].
`floor_exp = 2.0047` sits essentially AT its prior median 2.0 (95 % [0.736,
5.437]) where R2 wanted 3.22. `floor_static_rel = 0.000288` is just below
`LN(log 0.0025, 1.0)`'s [0.000338, 0.018473].

## Optimiser and wall (NOT converged)

| field | value |
|---|---:|
| Adam | 1500 steps, lr 0.02, batch 8 frames, seed 0, `init_jitter = 0` |
| `adam_first_loss` → `adam_final_loss` | −31 970 386.25 → −32 720 644.70 |
| `adam_wall_s` | **2068.54 s** (34.5 min; R2 2159.84) |
| L-BFGS | 200 + 100 iterations, 64 frames, history 10, `strong_wolfe` |
| `lbfgs_loss_before` → `first_pass` → `after_restart` | −33 412 160.85 → −33 514 255.07 → −33 518 931.85 |
| `lbfgs_restart_gain_per_cell` | **9.0618e-03** vs tol 1e-4 → **NOT converged** |
| `lbfgs_evals` / `lbfgs_restart_evals` | 228 / 114 |
| `lbfgs_wall_s` | **3552.26 s** (59.2 min; R2 3550.26) |
| `grad_norm` | 5430.15 |
| `optimiser.wall_s` | **5620.80 s = 1.561 h** (job wall 1.59 h; R2 5710.10 s) |

Same optimiser budget as R2 to the step, same wall to within 1.6 %: the R3
parameterisation (81 x 4 widths instead of 4 `*_eps` scalars) costs nothing
measurable per iteration here.

## Render-regime sanity (k = 2 comb-to-floor), BEFORE any scoring

`scripts/noise_v2_render_regime.py --fit <this fit> --regimes
cruise,standby,ramp --figures --out
results/noise_v2/rounds/round3/render_regime_flight`, run locally on CPU
(301 s), seed 2001, 8 mics, mic 0, 30–7900 Hz.

The script DID accept schema `/2` — but only at `58ab33ca` ("noise-v2 R3
score: accept schema /2"). At the fit's own SHA `01e1464b` it died on
`scripts/noise_v2_render_regime.py:455`, a hard `!= "noise-v2-fit/1"` gate;
`scripts/noise_v2_round_score.py` carried the same one-line gate
(`FIT_SCHEMA`). Both now read `experiments.noise_model.READABLE_FIT_SCHEMAS`
= `("noise-v2-fit/2", "noise-v2-fit/1")`, the pair the renderer already
accepted. (Cosmetic, unfixed: the generated `findings.md` header still prints
the round-2 default record path; the record actually written is
`round3/render_regime_flight/render_regime.json`.)

| regime | real | **R3 v2** | R2 v2 | legacy | floor-only control (`v2_nocomb`) |
|---|---:|---:|---:|---:|---:|
| standby | **15.1 dB** | **22.9** | 22.0 | 16.4 | 9.7 |
| ramp | **6.9 dB** | **8.2** | 7.7 | 6.2 | 12.4 |
| cruise | **25.3 dB** | **26.8** | 25.7 | 30.7 | 9.2 |

k = 4 / k = 8 for the R3 v2 arm: standby 12.8 / 13.0 (real 6.9 / 5.6), ramp
5.4 / 3.7 (real 2.8 / 1.7), cruise 10.6 / 5.8 (real 11.0 / 2.4).

Read: the comb SURVIVES both extrapolated regimes (R1's cruise-only fit gave
4.4 dB at standby and −0.7 dB at ramp; anything near the `v2_nocomb` control
would mean the comb had collapsed). R3 now sits 7.8 dB OVER real at standby
(R2: 6.9 over), 1.3 dB over at ramp (R2: 0.8) and 1.5 dB over at cruise (R2:
0.4) — the R3 render is slightly COMBIER than R2's everywhere, consistently
about 1 dB. Level and LTAS, absolute, mic 0: standby offset +0.24 dB / LTAS
1.37 dB (R2: +0.04 / 1.13), ramp −3.48 / 3.48 (R2: −3.08 / 3.08), cruise
+0.55 / 1.00 (R2: +0.83 / 1.26). Cruise LTAS improves, standby and ramp give
up ~0.25 and ~0.40 dB.

Figures: `round3/render_regime_flight/{ltas,spectrogram}_{standby,ramp}.png`.

## Delta vs R2, in one paragraph

Same pool, same optimiser, same wall, both non-converged. R3 buys 0.029
nats/cell, drops `sigma_nu` from 5.17 to 1.23 rad/s (the line width now lives
in `gamma_rk`, not in the shaft), pulls `floor_exp` from 3.22 to its prior
median 2.00 and flips the floor tilt from +0.30 to −0.75 dB/oct, and pushes
`amp_exp` from 3.90 to 4.16. The k = 2 comb/floor sanity moves 0.5–1.1 dB
FURTHER from real in all three regimes while staying far from the no-comb
control. Whether that is better or worse for the gate is a question for the
HPPNet arm, not for this file.
