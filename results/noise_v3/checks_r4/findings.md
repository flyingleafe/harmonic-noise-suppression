# Noise model v3 — the section 3.5 checks

`scripts/noise_v3_checks.py` (render and measure only). Every number below is read from the JSON named in its section.

## (b) Held-out line-power spread and intermittency (`heldout/heldout.json`)

Wander estimator of `noise_v3_measure_wander.py` (rig-centred, block 0.5 s, τ at lag 1), window bootstrap median [5 %, 95 %]:

| rig | arm | windows | line tracks | σ_total dB | σ_d dB | σ_v dB | σ_u dB |
|---|---|---:|---:|---|---|---|---|
| dregon | real | 5 | 0 | — (no track) | — (no track) | — (no track) | 3.66 (3.62 [2.52, 4.04]) |
| dregon | v3 | 20 | 0 | — (no track) | — (no track) | — (no track) | 0.34 (0.34 [0.25, 0.42]) |
| dregon | v2 | 20 | 0 | — (no track) | — (no track) | — (no track) | 0.54 (0.54 [0.44, 0.63]) |
| michaels | real | 4 | 5 | 1.46 (1.46 [0.53, 1.50]) | 0.12 (0.12 [0.00, 0.12]) | 1.46 (1.46 [0.53, 1.50]) | 2.97 (2.96 [0.83, 2.97]) |
| michaels | v3 | 16 | 36 | 0.77 (0.77 [0.55, 0.92]) | 0.88 (0.88 [0.68, 1.00]) | 0.00 (0.00 [0.00, 0.00]) | 0.30 (0.29 [0.23, 0.34]) |
| michaels | v2 | 16 | 30 | 0.38 (0.38 [0.28, 0.44]) | 0.26 (0.26 [0.18, 0.30]) | 0.29 (0.28 [0.14, 0.37]) | 0.21 (0.21 [0.15, 0.26]) |

Mid-order lines (k 8-24, every rotor), per block: UNDER the floor = line power below the local floor (prominence < 3.01 dB); PRESENT = window prominence >= 6 dB; VISIBLE = >= 6 dB in >= 20 % of the window's blocks, and the disappearance rate is the share of a visible line's blocks under 6 dB. Window bootstrap median [5 %, 95 %]:

| rig | arm | lines | cells under floor | present-line blocks under floor | lines that appear and disappear | block sd of a present line dB | visible lines | disappearance rate |
|---|---|---:|---|---|---|---|---:|---|
| dregon | real | 340 (0 present) | 81.8 [78.1, 85.8] % | — [—, —] % | 10.9 [6.2, 15.3] % | — [—, —] | 10 | 68.8 [66.4, 72.5] % |
| dregon | v3 | 1360 (0 present) | 99.0 [98.3, 99.7] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | v2 | 1360 (0 present) | 92.2 [90.3, 94.1] % | — [—, —] % | 1.8 [0.0, 3.7] % | — [—, —] | 0 | — [—, —] % |
| michaels | real | 272 (26 present) | 80.3 [66.6, 94.5] % | 7.2 [4.1, 17.7] % | 22.1 [3.3, 39.8] % | 2.76 [2.57, 3.91] | 38 | 38.2 [32.9, 51.1] % |
| michaels | v3 | 1088 (121 present) | 79.2 [59.2, 99.5] % | 4.6 [2.0, 8.2] % | 16.8 [0.0, 34.2] % | 1.40 [1.23, 1.70] | 166 | 31.2 [28.6, 34.4] % |
| michaels | v2 | 1088 (108 present) | 79.1 [60.0, 98.1] % | 3.2 [1.6, 5.5] % | 17.0 [0.0, 32.8] % | 1.18 [1.14, 1.21] | 170 | 33.9 [29.9, 38.6] % |

Every measured order by wander order group (the groups `sigma_v_db_by_order` is measured on), the same statistics, window bootstrap median [5 %, 95 %]:

| rig | orders | arm | lines | cells under floor | present-line blocks under floor | lines that appear and disappear | block sd of a present line dB | visible lines | disappearance rate |
|---|---|---|---:|---|---|---|---|---:|---|
| dregon | 1-2 | real | 20 (0 present) | 58.8 [31.2, 87.5] % | — [—, —] % | 50.0 [15.0, 80.0] % | — [—, —] | 8 | 71.9 [68.8, 75.0] % |
| dregon | 1-2 | v3 | 80 (0 present) | 26.6 [22.7, 30.8] % | — [—, —] % | 17.5 [0.0, 52.5] % | — [—, —] | 0 | — [—, —] % |
| dregon | 1-2 | v2 | 80 (0 present) | 11.6 [7.2, 15.9] % | — [—, —] % | 8.8 [0.0, 16.2] % | — [—, —] | 3 | 75.0 [75.0, 75.0] % |
| dregon | 3-8 | real | 120 (0 present) | 87.0 [83.2, 91.6] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 3-8 | v3 | 480 (0 present) | 98.8 [97.5, 99.9] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 3-8 | v2 | 480 (0 present) | 97.8 [95.6, 100.0] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 9-24 | real | 320 (0 present) | 81.2 [77.6, 85.9] % | — [—, —] % | 11.6 [6.6, 15.9] % | — [—, —] | 10 | 68.8 [66.2, 71.9] % |
| dregon | 9-24 | v3 | 1280 (0 present) | 99.1 [98.2, 99.7] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 9-24 | v2 | 1280 (0 present) | 91.9 [90.0, 93.8] % | — [—, —] % | 2.0 [0.0, 5.9] % | — [—, —] | 0 | — [—, —] % |
| dregon | 25-60 | real | 720 (0 present) | 89.2 [84.8, 93.5] % | — [—, —] % | 6.8 [2.8, 12.4] % | — [—, —] | 10 | 72.7 [68.8, 75.0] % |
| dregon | 25-60 | v3 | 2880 (0 present) | 99.7 [99.5, 99.8] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 25-60 | v2 | 2880 (0 present) | 98.6 [98.2, 99.0] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 61+ | real | 753 (0 present) | 98.2 [97.3, 99.1] % | — [—, —] % | 1.6 [0.0, 3.2] % | — [—, —] | 2 | 75.0 [75.0, 75.0] % |
| dregon | 61+ | v3 | 3012 (0 present) | 99.8 [99.5, 99.9] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 61+ | v2 | 3012 (0 present) | 99.5 [99.4, 99.6] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| michaels | 1-2 | real | 10 (8 present) | 4.7 [1.3, 8.5] % | 0.0 [0.0, 0.0] % | 10.0 [0.0, 20.0] % | 2.27 [1.16, 3.36] | 9 | 5.6 [0.0, 10.3] % |
| michaels | 1-2 | v3 | 40 (32 present) | 11.2 [4.6, 17.3] % | 0.0 [0.0, 0.0] % | 5.0 [0.0, 10.0] % | 1.31 [1.16, 1.49] | 32 | 0.0 [0.0, 0.0] % |
| michaels | 1-2 | v2 | 40 (32 present) | 13.3 [8.1, 17.9] % | 0.0 [0.0, 0.0] % | 0.0 [0.0, 0.0] % | 0.87 [0.75, 1.14] | 32 | 0.0 [0.0, 0.0] % |
| michaels | 3-8 | real | 96 (22 present) | 62.2 [44.6, 78.1] % | 6.5 [1.0, 13.1] % | 25.0 [8.3, 46.9] % | 2.80 [2.13, 3.54] | 24 | 23.7 [18.8, 29.5] % |
| michaels | 3-8 | v3 | 384 (103 present) | 67.7 [37.2, 84.3] % | 2.4 [0.3, 4.2] % | 16.1 [3.6, 37.5] % | 1.53 [1.04, 1.97] | 123 | 17.1 [14.0, 19.9] % |
| michaels | 3-8 | v2 | 384 (102 present) | 68.1 [37.5, 99.8] % | 2.2 [0.9, 3.4] % | 16.4 [0.0, 32.8] % | 1.19 [0.86, 1.52] | 120 | 17.6 [13.3, 21.3] % |
| michaels | 9-24 | real | 256 (22 present) | 81.5 [68.5, 95.4] % | 8.2 [4.8, 20.0] % | 21.5 [1.2, 37.9] % | 2.96 [2.59, 3.88] | 34 | 40.4 [34.6, 54.4] % |
| michaels | 9-24 | v3 | 1024 (99 present) | 80.5 [61.6, 99.5] % | 5.0 [2.4, 8.5] % | 17.1 [0.0, 34.0] % | 1.41 [1.30, 1.41] | 142 | 34.6 [32.4, 37.2] % |
| michaels | 9-24 | v2 | 1024 (88 present) | 80.2 [62.4, 98.2] % | 3.8 [1.8, 6.8] % | 17.1 [0.0, 32.5] % | 1.37 [1.14, 1.43] | 146 | 37.2 [33.8, 41.5] % |
| michaels | 25-60 | real | 576 (1 present) | 96.1 [92.6, 99.6] % | 0.0 [0.0, 0.0] % | 3.5 [0.0, 6.9] % | 1.39 [1.39, 1.39] | 2 | 65.6 [65.6, 65.6] % |
| michaels | 25-60 | v3 | 2304 (0 present) | 97.0 [94.5, 99.9] % | — [—, —] % | 0.7 [0.0, 1.4] % | — [—, —] | 0 | — [—, —] % |
| michaels | 25-60 | v2 | 2304 (0 present) | 97.2 [95.0, 99.9] % | — [—, —] % | 0.7 [0.0, 1.3] % | — [—, —] | 0 | — [—, —] % |
| michaels | 61+ | real | 1588 (16 present) | 98.2 [97.9, 99.9] % | 5.1 [2.3, 7.8] % | 1.6 [0.0, 1.9] % | 1.42 [1.40, 1.42] | 23 | 25.8 [25.6, 26.0] % |
| michaels | 61+ | v3 | 6352 (105 present) | 96.8 [96.2, 100.0] % | 3.7 [3.0, 4.4] % | 2.2 [0.0, 2.7] % | 1.66 [1.60, 1.73] | 163 | 36.1 [36.1, 36.2] % |
| michaels | 61+ | v2 | 6352 (109 present) | 96.9 [96.2, 100.0] % | 7.4 [7.0, 7.8] % | 2.2 [0.0, 2.7] % | 2.04 [2.03, 2.19] | 149 | 30.4 [29.6, 31.2] % |

Figure: `heldout/prominence_hist.png`.

## (c) Tonality on rendered audio (`tonality/rendered.json`)

R4 `line_width_db3` (-3 dB width, Hz, 8192-point, resolution 1.95 Hz) at k = 1..8 and the order-6-dB counts on the window prominence ladder; renders: median over the pattern windows x 4 seeds.

| fit | arm | width k=1..8 Hz | orders >= 6 dB / rotor | highest >= 6 dB | prom k=1,2,4,8,16 dB |
|---|---|---|---:|---:|---|
| dregon | real | 10.4, 23.4, —, —, —, —, 9.0, 30.6 | 0.0 | 0 | —, 4.6, 2.9, 1.6, 1.8 |
| dregon | v3 | 10.3, 15.5, —, 36.2, —, 24.9, 10.6, 16.4 | 0.0 | 0 | —, 3.8, 0.9, 0.5, 0.9 |
| dregon | v2 | 10.3, 10.2, —, 9.3, —, —, 7.0, 10.4 | 0.0 | 0 | —, 4.1, 0.5, 0.4, 1.8 |
| michaels_cruise | real | 3.1, 4.2, —, 5.7, 18.9, 8.2, 58.8, 13.8 | 9.0 | 18 | —, 24.8, 12.0, 7.8, 4.3 |
| michaels_cruise | v3 | 2.7, 4.4, —, 6.4, 19.2, 8.2, 23.2, 13.3 | 6.6 | 14 | —, 22.2, 11.7, 7.5, 2.9 |
| michaels_cruise | v2 | —, 4.6, —, 7.6, 18.2, 10.3, —, 16.0 | 7.0 | 15 | —, 22.0, 10.3, 7.1, 2.0 |
| michaels_standby | real | —, 4.0, —, 4.4, 2.0, 4.1, 12.9, 8.5 | 3.1 | 88 | —, —, 1.1, 1.1, 1.3 |
| michaels_standby | v3 | 8.9, 4.8, —, 3.1, 5.9, 3.3, 6.8, 7.6 | 4.0 | 88 | —, —, -0.6, 0.7, 0.8 |
| michaels_standby | v2 | 11.9, 4.0, —, 2.9, 6.2, 3.3, 7.4, 5.2 | 3.9 | 86 | —, —, -0.5, 0.7, 0.8 |

Visible-order counts per rotor at 3 / 6 / 10 dB (rotor mean), per pattern window: `audit` = the tonality audit's estimator on the clip's time-mean periodogram (orders 1..k_max of the v3 fit); `ladder` = the wander estimator's window prominence. Renders: median over the 4 seeds.

| fit | pattern | arm | audit >= 3 / 6 / 10 dB | audit highest >= 6 dB | ladder >= 3 / 6 / 10 dB |
|---|---|---|---|---:|---|
| dregon | dregon_cruise_narrow | real | 7.5 / 1.0 / 0.0 | 1 | 4.8 / 0.0 / 0.0 |
| dregon | dregon_cruise_narrow | v3 | 3.9 / 2.0 / 0.0 | 2 | 1.0 / 0.0 / 0.0 |
| dregon | dregon_cruise_narrow | v2 | 3.0 / 2.0 / 0.0 | 2 | 2.0 / 0.0 / 0.0 |
| dregon | dregon_cruise_wide | real | 5.5 / 1.0 / 0.0 | 1 | 7.5 / 0.0 / 0.0 |
| dregon | dregon_cruise_wide | v3 | 2.2 / 1.9 / 0.0 | 2 | 1.0 / 0.0 / 0.0 |
| dregon | dregon_cruise_wide | v2 | 2.6 / 1.0 / 0.0 | 1 | 1.8 / 0.0 / 0.0 |
| michaels_cruise | michaels_cruise_narrow | real | 8.8 / 3.0 / 1.2 | 9 | 12.2 / 7.8 / 3.0 |
| michaels_cruise | michaels_cruise_narrow | v3 | 7.2 / 2.0 / 1.2 | 5 | 11.4 / 5.8 / 2.0 |
| michaels_cruise | michaels_cruise_narrow | v2 | 7.8 / 1.9 / 1.2 | 6 | 11.2 / 6.5 / 1.8 |
| michaels_cruise | michaels_cruise_wide | real | 11.8 / 6.8 / 3.2 | 11 | 16.5 / 10.2 / 5.0 |
| michaels_cruise | michaels_cruise_wide | v3 | 10.4 / 5.5 / 2.8 | 10 | 12.0 / 7.4 / 3.5 |
| michaels_cruise | michaels_cruise_wide | v2 | 10.5 / 5.5 / 2.8 | 11 | 12.1 / 7.1 / 3.2 |
| michaels_standby | michaels_standby_narrow | real | 7.0 / 3.5 / 2.8 | 86 | 4.8 / 3.2 / 2.2 |
| michaels_standby | michaels_standby_narrow | v3 | 8.0 / 4.0 / 2.1 | 86 | 7.2 / 4.9 / 1.5 |
| michaels_standby | michaels_standby_narrow | v2 | 8.9 / 3.9 / 2.0 | 84 | 6.4 / 4.8 / 1.1 |
| michaels_standby | michaels_standby_wide | real | 4.5 / 1.2 / 0.0 | 65 | 11.0 / 3.0 / 1.0 |
| michaels_standby | michaels_standby_wide | v3 | 5.1 / 2.4 / 1.2 | 57 | 5.9 / 2.9 / 0.8 |
| michaels_standby | michaels_standby_wide | v2 | 4.4 / 2.0 / 1.2 | 44 | 5.0 / 3.0 / 1.0 |

## (c) Tonality on the expectation (`tonality/fits.json`, `noise_v2_tonality_audit.py --fit`)

| pattern | payload | prom k=1,2,4,8,16 dB (mic median) | orders >= 3 / 6 / 10 dB / rotor | highest >= 6 dB | trend crosses floor at k | pedestal median dB |
|---|---|---|---|---:|---:|---:|
| dregon_cruise_narrow | v3 dregon | 8.3, 8.7, 4.1, 1.5, 2.0 | 7.2 / 2.0 / 0.0 | 2 | 89 | 3.3 |
| dregon_cruise_narrow | v2 anchor dregon | 9.1, 9.4, 3.2, 5.4, 8.5 | 23.2 / 9.5 / 2.0 | 37 | 89 | 8.5 |
| dregon_cruise_wide | v3 dregon | 7.8, 6.4, 2.4, 0.0, -0.0 | 3.2 / 1.5 / 0.0 | 2 | 89 | 3.1 |
| dregon_cruise_wide | v2 anchor dregon | 9.5, 7.0, 1.9, 1.4, 3.1 | 17.2 / 6.8 / 0.2 | 38 | 89 | 8.8 |
| michaels_cruise_narrow | v3 michaels | 3.6, 28.6, 9.4, 5.7, 1.9 | 15.5 / 9.8 / 4.0 | 22 | 79 | 3.1 |
| michaels_cruise_narrow | v2 anchor michaels | 2.0, 29.6, 8.1, 5.4, 1.7 | 15.0 / 8.0 / 4.2 | 20 | 73 | 5.7 |
| michaels_cruise_wide | v3 michaels | 0.8, 24.0, 12.0, 4.7, 3.5 | 16.8 / 9.2 / 4.0 | 20 | 81 | 3.7 |
| michaels_cruise_wide | v2 anchor michaels | -2.2, 25.7, 9.1, 4.6, 3.1 | 17.0 / 8.2 / 3.8 | 20 | 70 | 10.6 |
| michaels_standby_narrow | v3 michaels | 8.1, 10.5, 1.8, 1.2, 1.8 | 7.8 / 4.8 / 2.2 | 89 | 115 | 2.0 |
| michaels_standby_narrow | v2 anchor michaels_standby | 9.0, 11.5, 1.4, 0.7, 1.8 | 8.5 / 4.2 / 2.8 | 86 | 127 | 8.3 |
| michaels_standby_wide | v3 michaels | 13.1, 6.4, 2.7, 3.2, 2.0 | 11.5 / 5.0 / 1.8 | 92 | 116 | 0.0 |
| michaels_standby_wide | v2 anchor michaels_standby | 14.8, 7.0, 2.1, 2.2, 2.1 | 11.5 / 3.8 / 1.5 | 85 | 129 | 3.9 |

## (d) Parity gate (`parity/arm_*.json`, `noise_v2_round_score.py --fit`)

HPPNet PIT MAE on renders of the frozen supports (four frozen render seeds, 8 mics) against the legacy PARITY bar and DREGON's frozen STRETCH target; proxy `ltas_abs_db` against its closure-0.7 gate.

| arm | rig | PIT MAE rev/s (mean) | 95 % upper | Michael's ratio | parity bar | parity | stretch bar | stretch | proxy dB | proxy gate dB |
|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|
| `arm_dregon_v3.json` | dregon | 1.665586 | 2.028494 | — | 2.187786 | PASS | 1.897063 | PASS | 2.4220 | 1.9786 |
| `arm_michaels_v3.json` | michaels | 1.576687 | — | 0.5209 | 3.177994 | PASS | — | — | 1.2810 | 1.2197 |

## (e) Fitted latents against the measured wander (`latents/latents.json`)

Measured σ = rms of the fit's prior sd over the family (v: per order group); posterior variance and the MAP's expected sd / lag-1 = the OU prior seen through the measured block noise s² (Kalman smoother); correctly shrunk latents give fitted sd = MAP expected sd, sqrt(fitted ms + posterior var) = measured σ, fitted lag-1 = MAP expected lag-1.

| fit | family | values | fitted sd dB | MAP expected sd dB | sqrt(fitted ms + posterior var) dB | measured σ dB | sqrt(fitted var + s²) dB | fitted lag-1 | MAP expected lag-1 | measured ρ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
- dregon: (e) disabled — every wander sigma is 0 by construction (rig-only fit); no latent track was fitted and none is rendered, so check (e) does not apply
- michaels_cruise: (e) disabled — every wander sigma is 0 by construction (rig-only fit); no latent track was fitted and none is rendered, so check (e) does not apply
- michaels_standby: (e) disabled — every wander sigma is 0 by construction (rig-only fit); no latent track was fitted and none is rendered, so check (e) does not apply

The same check as second moments (dB²): fitted mean square + posterior variance against the measured σ², and at lag one the fitted lag-1 product + the posterior lag-1 covariance against ρσ² (per block pair).

| fit | family | fitted ms | + posterior var | = lag-0 sum | measured σ² | fitted lag-1 product | + posterior lag-1 cov | = lag-1 sum | measured ρσ² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

