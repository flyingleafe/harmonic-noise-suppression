# Noise model v2 — R2 pooled Michael's flight fit (`michaels_fly125_all`)

Fit `results/noise_v2/rounds/round2/fits/michaels_fly125_all__flight.json`,
code SHA `c514bb507a4fdcd41a4bd658f4b66902848cc191` (`git` field of the JSON),
job `nv2-r2-michaels-flight-cb36cd` (`uni-cpu`, `--gpus 0 --cpus 8 --mem 48
--time 3h`), submitted 2026-09-18T07:51:24Z, started 07:51:40Z, `succeeded`
09:28:57Z, `exit_fit=0`. Every number below is read from that JSON (pool
geometry from `results/noise_v2/rounds/round2/supports/index.json`, set
`michaels-all`); nothing is recomputed.

**CONVERGENCE — read this before any number below.** The fit is **NOT
converged**: `optimiser.converged = false`, `optimiser.which_converged =
"none"`, `lbfgs_restart_gain_per_cell = 1.8505e-02` nats/cell against the
`tol_nats_per_cell = 1e-4` tolerance (185x over), `grad_norm = 4726.58`,
`log_box_hits = 0`. This is the same verdict as R1's cruise fit
(restart gain 1.8756e-02, |grad| 8555.3): the L-BFGS restart is still
improving the objective when the 200 + 100 iteration budget runs out. Every
parameter, objective and render number quoted below is therefore **a property
of THIS non-converged point**, not of a converged model — the label is repeated
at each table.

## Pool composition (10 windows, `michaels-all`, all NOT-converged-fit inputs)

| window (8 s) | regime | segment s | per-rotor carrier rev/s (mean) | window carrier min–max rev/s |
|---|---|---|---|---:|
| `flight_michaels_FLY125@2.000+8_rps_refined` | **standby** | 2.0–10.0 | 29.93 / 33.91 / 39.27 / 35.20 | 21.14–40.33 |
| `flight_michaels_FLY125@11.588+8_rps_refined` | **ramp** | 11.5875–19.5875 | 59.65 / 54.44 / 60.48 / 55.55 | 30.97–95.99 |
| `flight_michaels_FLY125@16.000+8_rps_refined` | cruise | 16.0–24.0 | 89.96 / 74.30 / 81.42 / 75.39 | 69.09–95.95 |
| `flight_michaels_FLY125@32.000+8_rps_refined` | cruise | 32.0–40.0 | 91.80 / 74.91 / 80.99 / 75.23 | 72.59–94.38 |
| `flight_michaels_FLY125@48.000+8_rps_refined` | cruise | 48.0–56.0 | 89.41 / 75.01 / 81.32 / 77.44 | 68.75–93.50 |
| `flight_michaels_FLY125@64.000+8_rps_refined` | cruise | 64.0–72.0 | 91.06 / 74.48 / 82.84 / 74.48 | 69.57–94.14 |
| `flight_michaels_FLY125@96.000+8_rps_refined` | cruise | 96.0–104.0 | 90.09 / 74.62 / 81.29 / 76.74 | 71.46–93.83 |
| `flight_michaels_FLY125@112.000+8_rps_refined` | cruise | 112.0–120.0 | 91.62 / 74.05 / 80.97 / 75.98 | 69.57–97.41 |
| `flight_michaels_FLY125@128.000+8_rps_refined` | cruise | 128.0–136.0 | 91.61 / 74.21 / 81.80 / 73.92 | 70.72–95.16 |
| `flight_michaels_FLY125@144.000+8_rps_refined` | cruise | 144.0–152.0 | 91.57 / 74.24 / 82.86 / 75.27 | 69.71–97.10 |

8 cruise + 1 standby + 1 ramp, 4 rotors, 8 mics, `k_max = 81`, `sr = 16000`,
`n_fft/hop = 2048/512` (work grid 64000 / 8192).

**Carrier range the objective actually saw** (`diagnostics.batch`):
`carrier_min_rev_s = 20.9301`, `carrier_max_rev_s = 97.9103` — a speed span of
**4.678x**, against R1's cruise-only 68.1834–97.9103 = **1.436x**. That is the
whole point of the pool: at 4.678x both speed exponents are INTERPOLATED over
the regimes that get rendered (standby ≈ 0.455 of the 80 rev/s reference, ramp
≈ 0.710, cruise ≈ 1.012), where R1 extrapolated both.

Frames: `n_frames_total = 2470`, `frame_stride = 4`, `n_frames_used = 250`
(R1 cruise: 1976 / 4 / 256), `exposure_scale = 9.88`.

## Objective (NOT converged)

| quantity | value | cells | nats/cell |
|---|---:|---:|---:|
| total `objective.whittle_nats` | −33 465 417.4347 | 2 016 000 | **−16.5999** |
| comb band (> 300 Hz) | −32 565 697.9450 | 1 946 000 | −16.7347 |
| floor band (30–300 Hz) | −899 719.4897 | 70 000 | −12.8531 |

Band edges `objective.band_hz = [30.0, 7898.4375]`, `band_split_hz = 300.0`.
R1's cruise fit for scale: −24 362 038.63 / 2 064 384 cells = −11.8012
nats/cell — the two are NOT comparable as model quality (different pool, 10
windows vs 8, different frames), they are recorded because the round record
quotes nats/cell.

**The JSON carries NO per-regime (per-support) objective breakdown** — the
only decomposition `noise-v2-fit/1` records is the two-band split above
(`objective.per_band`, `objective.n_cells_per_band`). The per-regime Whittle
would need a re-evaluation pass over the 10 windows, which this fit did not
run; it is not quoted here rather than being invented.

## The six dynamics, the two exponents, the floor (NOT converged)

| parameter | value | status |
|---|---:|---|
| `sigma_nu` | 5.1716 rev/s | free (R1 cruise: 8.3785) |
| `lam` | 0.5 /s | **PINNED** (R1Basin ridge coordinate, Main-approved) |
| `sigma_eps_even` | 0.0057388 | free (R1: 0.0016142) |
| `sigma_eps_odd` | 0.8493112 | free (R1: 0.5953105) |
| `lam_eps_even` | 2.0 /s | **PINNED** |
| `lam_eps_odd` | 75.6330709 /s | **PINNED** |
| `amp_exp` (comb speed exponent) | **3.9029** | free — R1 cruise: 10.0196 |
| `floor_exp` (floor speed exponent) | **+3.2243** | free, sampled in LOG space at this SHA so it cannot go negative — R1 cruise: **−10.1892** |
| `floor_mean_db` | −38.1837 dB | free (R1: −32.7099) |
| `diagnostics.floor_level_db` | **−38.2722 dB** | derived floor level (R1: −32.7911) |
| `floor_tilt_db_oct` | 0.30305 dB/oct | free (R1: 0.72670) |
| `floor_static_rel` | 0.0019644 | free (R1: 0.0049277) |

`optimiser.pinned_dynamics = {lam: 0.5, lam_eps: [2.0, 75.63307088769216]}`
confirms exactly the three pins the submit note declares, and nothing else.
`params.p = 1.0` (fixed), `params.carrier_rev_s = null` (flight mode tracks the
label). Prior on the floor exponent at this SHA is `log_floor_exp =
[0.6931472, 0.7]` (log-normal, mean 2.0) — the fitted +3.2243 is inside its
central 95 %; `amp_exp`'s prior is unchanged `N(2, 2)` and 3.9029 is inside it
too. The only prior-edge flag the driver raises is `sigma_nu = 5.1716` outside
the central 95 % of `log_sigma_nu = [−0.7985, 0.7]`.

Per-rotor comb profile (81 orders, dB): rotor 0 peak −30.33 (k = 2), median
−56.80; rotor 1 peak −36.27, median −60.03; rotor 2 peak −41.46, median
−61.72; rotor 3 peak −46.33, median −61.71. Mic line gains and
`mic_floor_db` stay inside ±7 dB (`mic_gains_db` −0.76 … +4.63,
`mic_floor_db` −3.07 … +2.57). `diagnostics.saturated_coherence` =
0.4861 / 0.9997 / 0.9993 / 0.9987 at k = 1 / 10 / 20 / 40.

## Optimiser and wall

| field | value |
|---|---:|
| Adam | 1500 steps, lr 0.02, batch 8 frames, seed 0 |
| `adam_first_loss` → `adam_final_loss` | −32 002 232.03 → −32 681 186.69 |
| `adam_wall_s` | **2159.84 s** (36.0 min) |
| L-BFGS | 200 + 100 iterations, 64 frames, history 10, `strong_wolfe` |
| `lbfgs_loss_before` → `lbfgs_loss_first_pass` → `lbfgs_loss_after_restart` | −33 374 544.43 → −33 449 901.60 → −33 459 451.89 |
| `lbfgs_restart_gain_per_cell` | **1.8505e-02** nats/cell vs tol 1e-4 → NOT converged |
| `lbfgs_evals` / `lbfgs_restart_evals` | 226 / 114 |
| `lbfgs_wall_s` | **3550.26 s** (59.2 min), 516 096 cells on 64 frames |
| `grad_norm` | 4726.58 |
| `optimiser.wall_s` | **5710.10 s = 1.586 h** (job wall 1.62 h) |

Within the submit note's 1.5–2 h envelope; the Adam half ran 36 min against
R1's 41 min despite the larger pool (the batch is 8 frames either way), and the
L-BFGS half 59 min against R1's 67 min.

## Render-regime sanity BEFORE scoring (k = 2 comb-to-floor)

`results/noise_v2/rounds/round2/render_regime_flight/render_regime.json`
(+ `findings.md`, 4 figures), laptop CPU, seed 2001, 8 mics, no `--probe`
(no HPPNet PIT in this pass — the PIT comes from the score arm). Run with the
fit above, i.e. **the NOT-converged point**; `fit.pool_speed_span = 4.678`,
`fit.pool_carrier_rev_s = [20.930, 97.910]` recorded in the record itself.

| regime | arm | band level dB (mic 0) | LTAS mean abs dB | **k = 2 comb/floor dB** | k = 4 | k = 8 |
|---|---|---:|---:|---:|---:|---:|
| standby | **real** | −2.81 | 0.00 | **15.1** | 6.9 | 5.6 |
| standby | legacy | −3.36 | 0.59 | 16.4 | 8.8 | 6.3 |
| standby | **v2 (this fit)** | **−3.56** | **1.13** | **22.0** | 10.7 | 6.6 |
| standby | v2 (R1 cruise fit, for contrast) | 29.51 | 34.14 | 4.4 | 1.9 | 1.5 |
| ramp | **real** | 9.33 | 0.00 | **6.9** | 2.8 | 1.7 |
| ramp | legacy | 6.35 | 4.19 | 6.2 | 3.4 | 4.0 |
| ramp | **v2 (this fit)** | **7.07** | **3.08** | **7.7** | 1.9 | 4.1 |
| ramp | v2 (R1 cruise fit, for contrast) | 25.90 | 17.11 | −0.7 | 1.7 | 1.7 |
| cruise | **real** | 12.16 | 0.00 | **25.3** | 11.0 | 2.4 |
| cruise | legacy | 10.21 | 0.98 | 30.7 | 15.4 | 3.4 |
| cruise | **v2 (this fit)** | **11.14** | **1.26** | **25.7** | 8.9 | 3.4 |
| cruise | v2 (R1 cruise fit, for contrast) | 10.37 | 0.59 | 26.4 | 11.7 | 3.0 |

(The R1 rows are `round2/render_regime/findings.md`, the same runner and seed
on the R1 cruise-only fit.)

**The defect the pool was supposed to fix is fixed, on this metric.** The two
hypotheses R2 opened with (H1 `amp_exp` collapses the comb outside cruise, H2
negative `floor_exp` explodes the floor as the rotors slow) are both closed by
the numbers above:

* standby went from a comb 10.7 dB BELOW the real clip's k = 2 ratio (4.4 vs
  15.1) to 6.9 dB ABOVE it (22.0 vs 15.1) — the comb now over-shoots rather
  than vanishing, and it over-shoots by less than the legacy arm over-shoots
  at cruise (30.7 vs 25.3);
* the standby level error collapsed from **+34.14 dB to +0.04 dB**
  (`ltas_level_offset_db`), LTAS mean abs from 34.14 dB to **1.13 dB** —
  better than the legacy arm's own cruise LTAS (0.98 dB) and within 0.54 dB of
  legacy's standby (0.59 dB);
* ramp: k = 2 7.7 vs real 6.9 (R1: −0.7), level offset −3.08 dB vs legacy
  −1.95 dB, LTAS 3.08 dB vs legacy 4.19 dB — the v2 render is now CLOSER to
  the real ramp clip than the legacy render is;
* cruise, the regime R1 already had, did not regress: k = 2 25.7 vs real 25.3
  (R1 26.4), LTAS 1.26 dB (R1 0.59 dB, legacy 0.98 dB).

The speed envelopes now move the right way and by a plausible amount: at
standby (`speed_mean` 0.455) `comb_envelope_db = −13.14`,
`floor_envelope_db = −10.79`, `comb − floor = −2.35` dB, against R1's
−32.67 / +36.96 / **−69.63** dB; at ramp (0.710) −5.77 / −4.74 / −1.02 dB
against R1's −14.61 / +15.46 / −30.07 dB. The 70 dB comb-to-floor
extrapolation error is gone; what is left is a ~2 dB tilt.

What this pass does NOT say: it carries no PIT MAE (no `--probe`), so it is
not a parity claim. The parity verdict is the score arm
(`nv2-r2-score-michaels-5099b4`, `round2/render/arm_michaels_v2.json`),
against Michael's PARITY equal-regime bar 3.177994 rev/s (legacy per-regime
0.317 standby / 8.007 ramp / 0.756 cruise) and proxy `ltas_abs_db` ≤ 1.219668
dB. And no number here or there upgrades the fit's convergence status: it is
`converged = false`, `which_converged = "none"`, restart gain 1.8505e-02
nats/cell vs 1e-4.

## Figures

* `results/noise_v2/rounds/round2/render_regime_flight/ltas_standby.png`
* `results/noise_v2/rounds/round2/render_regime_flight/spectrogram_standby.png`
* `results/noise_v2/rounds/round2/render_regime_flight/ltas_ramp.png`
* `results/noise_v2/rounds/round2/render_regime_flight/spectrogram_ramp.png`
