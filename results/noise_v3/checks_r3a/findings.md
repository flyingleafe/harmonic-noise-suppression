# Noise model v3 — the section 3.5 checks

`scripts/noise_v3_checks.py` (render and measure only). Every number below is read from the JSON named in its section.

## (b) Held-out line-power spread and intermittency (`heldout/heldout.json`)

Wander estimator of `noise_v3_measure_wander.py` (rig-centred, block 0.5 s, τ at lag 1), window bootstrap median [5 %, 95 %]:

| rig | arm | windows | line tracks | σ_total dB | σ_d dB | σ_v dB | σ_u dB |
|---|---|---:|---:|---|---|---|---|
| dregon | real | 5 | 0 | — (no track) | — (no track) | — (no track) | 3.66 (3.62 [2.52, 4.04]) |
| dregon | v3 | 20 | 0 | — (no track) | — (no track) | — (no track) | 0.97 (0.95 [0.80, 1.06]) |
| dregon | v2 | 20 | 0 | — (no track) | — (no track) | — (no track) | 0.54 (0.54 [0.44, 0.63]) |
| michaels | real | 4 | 5 | 1.46 (1.46 [0.53, 1.50]) | 0.12 (0.12 [0.00, 0.12]) | 1.46 (1.46 [0.53, 1.50]) | 2.97 (2.96 [0.83, 2.97]) |
| michaels | v3 | 16 | 21 | 2.95 (2.83 [2.41, 3.33]) | 0.00 (0.00 [0.00, 0.56]) | 2.95 (2.82 [2.40, 3.32]) | 0.73 (0.71 [0.64, 0.77]) |
| michaels | v2 | 16 | 30 | 0.38 (0.38 [0.28, 0.44]) | 0.26 (0.26 [0.18, 0.30]) | 0.29 (0.28 [0.14, 0.37]) | 0.21 (0.21 [0.15, 0.26]) |

Mid-order lines (k 8-24, every rotor), per block: UNDER the floor = line power below the local floor (prominence < 3.01 dB); PRESENT = window prominence >= 6 dB; VISIBLE = >= 6 dB in >= 20 % of the window's blocks, and the disappearance rate is the share of a visible line's blocks under 6 dB. Window bootstrap median [5 %, 95 %]:

| rig | arm | lines | cells under floor | present-line blocks under floor | lines that appear and disappear | block sd of a present line dB | visible lines | disappearance rate |
|---|---|---:|---|---|---|---|---:|---|
| dregon | real | 340 (0 present) | 81.8 [78.1, 85.8] % | — [—, —] % | 10.9 [6.2, 15.3] % | — [—, —] | 10 | 68.8 [66.4, 72.5] % |
| dregon | v3 | 1360 (0 present) | 89.2 [88.4, 90.1] % | — [—, —] % | 5.6 [4.6, 6.5] % | — [—, —] | 26 | 67.8 [65.9, 69.9] % |
| dregon | v2 | 1360 (0 present) | 92.2 [90.3, 94.1] % | — [—, —] % | 1.8 [0.0, 3.7] % | — [—, —] | 0 | — [—, —] % |
| michaels | real | 272 (26 present) | 80.3 [66.6, 94.5] % | 7.2 [4.1, 17.7] % | 22.1 [3.3, 39.8] % | 2.76 [2.57, 3.91] | 38 | 38.2 [32.9, 51.1] % |
| michaels | v3 | 1088 (138 present) | 74.3 [60.7, 89.2] % | 9.9 [8.5, 11.9] % | 27.9 [4.2, 47.2] % | 2.67 [2.61, 2.83] | 197 | 40.5 [35.8, 45.9] % |
| michaels | v2 | 1088 (108 present) | 79.1 [60.0, 98.1] % | 3.2 [1.6, 5.5] % | 17.0 [0.0, 32.8] % | 1.18 [1.14, 1.21] | 170 | 33.9 [29.9, 38.6] % |

Every measured order by wander order group (the groups `sigma_v_db_by_order` is measured on), the same statistics, window bootstrap median [5 %, 95 %]:

| rig | orders | arm | lines | cells under floor | present-line blocks under floor | lines that appear and disappear | block sd of a present line dB | visible lines | disappearance rate |
|---|---|---|---:|---|---|---|---|---:|---|
| dregon | 1-2 | real | 20 (0 present) | 58.8 [31.2, 87.5] % | — [—, —] % | 50.0 [15.0, 80.0] % | — [—, —] | 8 | 71.9 [68.8, 75.0] % |
| dregon | 1-2 | v3 | 80 (0 present) | 22.8 [20.2, 25.6] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 1-2 | v2 | 80 (0 present) | 11.6 [7.2, 15.9] % | — [—, —] % | 8.8 [0.0, 16.2] % | — [—, —] | 3 | 75.0 [75.0, 75.0] % |
| dregon | 3-8 | real | 120 (0 present) | 87.0 [83.2, 91.6] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 3-8 | v3 | 480 (0 present) | 97.5 [96.6, 98.2] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 3-8 | v2 | 480 (0 present) | 97.8 [95.6, 100.0] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 9-24 | real | 320 (0 present) | 81.2 [77.6, 85.9] % | — [—, —] % | 11.6 [6.6, 15.9] % | — [—, —] | 10 | 68.8 [66.2, 71.9] % |
| dregon | 9-24 | v3 | 1280 (0 present) | 88.6 [87.8, 89.6] % | — [—, —] % | 5.9 [4.8, 6.9] % | — [—, —] | 26 | 67.8 [65.6, 70.0] % |
| dregon | 9-24 | v2 | 1280 (0 present) | 91.9 [90.0, 93.8] % | — [—, —] % | 2.0 [0.0, 5.9] % | — [—, —] | 0 | — [—, —] % |
| dregon | 25-60 | real | 720 (0 present) | 89.2 [84.8, 93.5] % | — [—, —] % | 6.8 [2.8, 12.4] % | — [—, —] | 10 | 72.7 [68.8, 75.0] % |
| dregon | 25-60 | v3 | 2880 (0 present) | 87.3 [86.1, 88.8] % | — [—, —] % | 7.9 [6.2, 9.7] % | — [—, —] | 38 | 72.1 [71.0, 73.0] % |
| dregon | 25-60 | v2 | 2880 (0 present) | 98.6 [98.2, 99.0] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 61+ | real | 753 (0 present) | 98.2 [97.3, 99.1] % | — [—, —] % | 1.6 [0.0, 3.2] % | — [—, —] | 2 | 75.0 [75.0, 75.0] % |
| dregon | 61+ | v3 | 3012 (0 present) | 89.6 [88.3, 90.9] % | — [—, —] % | 9.3 [7.5, 11.0] % | — [—, —] | 37 | 72.6 [72.0, 73.1] % |
| dregon | 61+ | v2 | 3012 (0 present) | 99.5 [99.4, 99.6] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| michaels | 1-2 | real | 10 (8 present) | 4.7 [1.3, 8.5] % | 0.0 [0.0, 0.0] % | 10.0 [0.0, 20.0] % | 2.27 [1.16, 3.36] | 9 | 5.6 [0.0, 10.3] % |
| michaels | 1-2 | v3 | 40 (32 present) | 8.9 [4.9, 12.5] % | 0.0 [0.0, 0.0] % | 2.5 [0.0, 5.0] % | 1.26 [1.15, 1.36] | 33 | 1.0 [0.0, 1.9] % |
| michaels | 1-2 | v2 | 40 (32 present) | 13.3 [8.1, 17.9] % | 0.0 [0.0, 0.0] % | 0.0 [0.0, 0.0] % | 0.87 [0.75, 1.14] | 32 | 0.0 [0.0, 0.0] % |
| michaels | 3-8 | real | 96 (22 present) | 62.2 [44.6, 78.1] % | 6.5 [1.0, 13.1] % | 25.0 [8.3, 46.9] % | 2.80 [2.13, 3.54] | 24 | 23.7 [18.8, 29.5] % |
| michaels | 3-8 | v3 | 384 (105 present) | 61.5 [39.8, 73.4] % | 5.7 [2.6, 8.7] % | 32.8 [22.7, 46.9] % | 3.08 [2.90, 3.34] | 125 | 25.4 [19.3, 34.6] % |
| michaels | 3-8 | v2 | 384 (102 present) | 68.1 [37.5, 99.8] % | 2.2 [0.9, 3.4] % | 16.4 [0.0, 32.8] % | 1.19 [0.86, 1.52] | 120 | 17.6 [13.3, 21.3] % |
| michaels | 9-24 | real | 256 (22 present) | 81.5 [68.5, 95.4] % | 8.2 [4.8, 20.0] % | 21.5 [1.2, 37.9] % | 2.96 [2.59, 3.88] | 34 | 40.4 [34.6, 54.4] % |
| michaels | 9-24 | v3 | 1024 (119 present) | 75.3 [62.0, 89.7] % | 10.5 [9.5, 12.0] % | 27.9 [3.5, 47.0] % | 2.67 [2.58, 2.83] | 175 | 42.2 [37.4, 47.7] % |
| michaels | 9-24 | v2 | 1024 (88 present) | 80.2 [62.4, 98.2] % | 3.8 [1.8, 6.8] % | 17.1 [0.0, 32.5] % | 1.37 [1.14, 1.43] | 146 | 37.2 [33.8, 41.5] % |
| michaels | 25-60 | real | 576 (1 present) | 96.1 [92.6, 99.6] % | 0.0 [0.0, 0.0] % | 3.5 [0.0, 6.9] % | 1.39 [1.39, 1.39] | 2 | 65.6 [65.6, 65.6] % |
| michaels | 25-60 | v3 | 2304 (10 present) | 86.6 [78.0, 96.4] % | 20.6 [17.9, 27.1] % | 19.2 [3.1, 32.4] % | 2.87 [2.78, 2.97] | 57 | 67.3 [67.0, 68.3] % |
| michaels | 25-60 | v2 | 2304 (0 present) | 97.2 [95.0, 99.9] % | — [—, —] % | 0.7 [0.0, 1.3] % | — [—, —] | 0 | — [—, —] % |
| michaels | 61+ | real | 1588 (16 present) | 98.2 [97.9, 99.9] % | 5.1 [2.3, 7.8] % | 1.6 [0.0, 1.9] % | 1.42 [1.40, 1.42] | 23 | 25.8 [25.6, 26.0] % |
| michaels | 61+ | v3 | 6352 (64 present) | 94.1 [90.5, 95.3] % | 14.9 [13.9, 62.5] % | 9.1 [6.6, 16.6] % | 3.87 [3.60, 4.50] | 121 | 46.6 [44.0, 72.6] % |
| michaels | 61+ | v2 | 6352 (109 present) | 96.9 [96.2, 100.0] % | 7.4 [7.0, 7.8] % | 2.2 [0.0, 2.7] % | 2.04 [2.03, 2.19] | 149 | 30.4 [29.6, 31.2] % |

Figure: `heldout/prominence_hist.png`.

## (c) Tonality on rendered audio (`tonality/rendered.json`)

R4 `line_width_db3` (-3 dB width, Hz, 8192-point, resolution 1.95 Hz) at k = 1..8 and the order-6-dB counts on the window prominence ladder; renders: median over the pattern windows x 4 seeds.

| fit | arm | width k=1..8 Hz | orders >= 6 dB / rotor | highest >= 6 dB | prom k=1,2,4,8,16 dB |
|---|---|---|---:|---:|---|
| dregon | real | 10.4, 23.4, —, —, —, —, 9.0, 30.6 | 0.0 | 0 | —, 4.6, 2.9, 1.6, 1.8 |
| dregon | v3 | 9.7, 17.5, —, —, —, 13.4, 10.9, 13.2 | 0.0 | 0 | —, 4.2, 0.8, 0.2, 1.2 |
| dregon | v2 | 10.3, 10.2, —, 9.3, —, —, 7.0, 10.4 | 0.0 | 0 | —, 4.1, 0.5, 0.4, 1.8 |
| michaels_cruise | real | 3.1, 4.2, —, 5.7, 18.9, 8.2, 58.8, 13.8 | 9.0 | 18 | —, 24.8, 12.0, 7.8, 4.3 |
| michaels_cruise | v3 | 2.5, 4.9, —, 7.8, 19.8, 9.3, 40.4, 15.1 | 8.8 | 20 | —, 23.0, 10.9, 7.0, 3.9 |
| michaels_cruise | v2 | —, 4.6, —, 7.6, 18.2, 10.3, —, 16.0 | 7.0 | 15 | —, 22.0, 10.3, 7.1, 2.0 |
| michaels_standby | real | —, 4.0, —, 4.4, 2.0, 4.1, 12.9, 8.5 | 3.1 | 88 | —, —, 1.1, 1.1, 1.3 |
| michaels_standby | v3 | 8.7, 4.1, 0.9, 2.8, 3.5, 3.5, 6.5, 4.6 | 2.1 | 104 | —, —, -0.8, 1.1, 0.9 |
| michaels_standby | v2 | 11.9, 4.0, —, 2.9, 6.2, 3.3, 7.4, 5.2 | 3.9 | 86 | —, —, -0.5, 0.7, 0.8 |

Visible-order counts per rotor at 3 / 6 / 10 dB (rotor mean), per pattern window: `audit` = the tonality audit's estimator on the clip's time-mean periodogram (orders 1..k_max of the v3 fit); `ladder` = the wander estimator's window prominence. Renders: median over the 4 seeds.

| fit | pattern | arm | audit >= 3 / 6 / 10 dB | audit highest >= 6 dB | ladder >= 3 / 6 / 10 dB |
|---|---|---|---|---:|---|
| dregon | dregon_cruise_narrow | real | 7.5 / 1.0 / 0.0 | 1 | 4.8 / 0.0 / 0.0 |
| dregon | dregon_cruise_narrow | v3 | 4.5 / 2.0 / 0.4 | 2 | 2.0 / 0.0 / 0.0 |
| dregon | dregon_cruise_narrow | v2 | 3.0 / 2.0 / 0.0 | 2 | 2.0 / 0.0 / 0.0 |
| dregon | dregon_cruise_wide | real | 5.5 / 1.0 / 0.0 | 1 | 7.5 / 0.0 / 0.0 |
| dregon | dregon_cruise_wide | v3 | 3.6 / 2.0 / 0.0 | 2 | 4.4 / 0.0 / 0.0 |
| dregon | dregon_cruise_wide | v2 | 2.6 / 1.0 / 0.0 | 1 | 1.8 / 0.0 / 0.0 |
| michaels_cruise | michaels_cruise_narrow | real | 8.8 / 3.0 / 1.2 | 9 | 12.2 / 7.8 / 3.0 |
| michaels_cruise | michaels_cruise_narrow | v3 | 8.5 / 4.2 / 1.5 | 13 | 17.0 / 8.2 / 2.9 |
| michaels_cruise | michaels_cruise_narrow | v2 | 7.8 / 1.9 / 1.2 | 6 | 11.2 / 6.5 / 1.8 |
| michaels_cruise | michaels_cruise_wide | real | 11.8 / 6.8 / 3.2 | 11 | 16.5 / 10.2 / 5.0 |
| michaels_cruise | michaels_cruise_wide | v3 | 11.2 / 6.6 / 3.0 | 14 | 21.0 / 9.2 / 4.0 |
| michaels_cruise | michaels_cruise_wide | v2 | 10.5 / 5.5 / 2.8 | 11 | 12.1 / 7.1 / 3.2 |
| michaels_standby | michaels_standby_narrow | real | 7.0 / 3.5 / 2.8 | 86 | 4.8 / 3.2 / 2.2 |
| michaels_standby | michaels_standby_narrow | v3 | 22.4 / 4.0 / 1.9 | 84 | 8.8 / 2.2 / 0.8 |
| michaels_standby | michaels_standby_narrow | v2 | 8.9 / 3.9 / 2.0 | 84 | 6.4 / 4.8 / 1.1 |
| michaels_standby | michaels_standby_wide | real | 4.5 / 1.2 / 0.0 | 65 | 11.0 / 3.0 / 1.0 |
| michaels_standby | michaels_standby_wide | v3 | 7.9 / 2.5 / 1.4 | 46 | 9.5 / 1.6 / 0.8 |
| michaels_standby | michaels_standby_wide | v2 | 4.4 / 2.0 / 1.2 | 44 | 5.0 / 3.0 / 1.0 |

## (c) Tonality on the expectation (`tonality/fits.json`, `noise_v2_tonality_audit.py --fit`)

| pattern | payload | prom k=1,2,4,8,16 dB (mic median) | orders >= 3 / 6 / 10 dB / rotor | highest >= 6 dB | trend crosses floor at k | pedestal median dB |
|---|---|---|---|---:|---:|---:|
| dregon_cruise_narrow | v3 dregon | 8.8, 9.7, 4.1, 2.3, 2.3 | 10.2 / 2.5 / 0.0 | 12 | 88 | 7.2 |
| dregon_cruise_narrow | v2 anchor dregon | 9.1, 9.4, 3.2, 5.4, 8.5 | 23.2 / 9.5 / 2.0 | 37 | 89 | 8.5 |
| dregon_cruise_wide | v3 dregon | 8.5, 7.4, 2.6, 0.5, 0.5 | 6.8 / 2.0 / 0.0 | 2 | 88 | 5.9 |
| dregon_cruise_wide | v2 anchor dregon | 9.5, 7.0, 1.9, 1.4, 3.1 | 17.2 / 6.8 / 0.2 | 38 | 89 | 8.8 |
| michaels_cruise_narrow | v3 michaels | 5.3, 30.2, 9.1, 4.3, 1.9 | 17.5 / 9.8 / 4.2 | 23 | 82 | 5.9 |
| michaels_cruise_narrow | v2 anchor michaels | 2.0, 29.6, 8.1, 5.4, 1.7 | 15.0 / 8.0 / 4.2 | 20 | 73 | 5.7 |
| michaels_cruise_wide | v3 michaels | 1.6, 26.1, 12.0, 2.8, 4.0 | 18.5 / 10.8 / 4.2 | 25 | 82 | 6.6 |
| michaels_cruise_wide | v2 anchor michaels | -2.2, 25.7, 9.1, 4.6, 3.1 | 17.0 / 8.2 / 3.8 | 20 | 70 | 10.6 |
| michaels_standby_narrow | v3 michaels | 13.2, 11.1, 1.6, 1.2, 2.3 | 8.8 / 4.0 / 2.0 | 86 | 114 | 1.7 |
| michaels_standby_narrow | v2 anchor michaels_standby | 9.0, 11.5, 1.4, 0.7, 1.8 | 8.5 / 4.2 / 2.8 | 86 | 127 | 8.3 |
| michaels_standby_wide | v3 michaels | 20.5, 6.7, 2.5, 3.8, 2.4 | 10.0 / 3.0 / 2.2 | 85 | 115 | 0.0 |
| michaels_standby_wide | v2 anchor michaels_standby | 14.8, 7.0, 2.1, 2.2, 2.1 | 11.5 / 3.8 / 1.5 | 85 | 129 | 3.9 |

## (d) Parity gate (`parity/arm_*.json`, `noise_v2_round_score.py --fit`)

HPPNet PIT MAE on renders of the frozen supports (four frozen render seeds, 8 mics) against the legacy PARITY bar and DREGON's frozen STRETCH target; proxy `ltas_abs_db` against its closure-0.7 gate.

| arm | rig | PIT MAE rev/s (mean) | 95 % upper | Michael's ratio | parity bar | parity | stretch bar | stretch | proxy dB | proxy gate dB |
|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|
| `arm_dregon_v3.json` | dregon | 1.709914 | 2.074563 | — | 2.187786 | PASS | 1.897063 | PASS | 2.8266 | 1.9786 |
| `arm_michaels_v3.json` | michaels | 1.674130 | — | 0.5531 | 3.177994 | PASS | — | — | 2.3701 | 1.2197 |

## (e) Fitted latents against the measured wander (`latents/latents.json`)

Measured σ = rms of the fit's prior sd over the family (v: per order group); posterior variance and the MAP's expected sd / lag-1 = the OU prior seen through the measured block noise s² (Kalman smoother); correctly shrunk latents give fitted sd = MAP expected sd, sqrt(fitted ms + posterior var) = measured σ, fitted lag-1 = MAP expected lag-1.

| fit | family | values | fitted sd dB | MAP expected sd dB | sqrt(fitted ms + posterior var) dB | measured σ dB | sqrt(fitted var + s²) dB | fitted lag-1 | MAP expected lag-1 | measured ρ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dregon | d | 320 | 1.87 | 0.61 | 2.00 | 0.94 | 2.19 | 0.41 | 0.30 | 0.19 |
| dregon | v | 28160 | 3.92 | 4.81 | 4.05 | 4.92 | 4.08 | 0.50 | 0.55 | 0.50 |
| dregon | v_present | 32 | 7.94 | 7.09 | 8.02 | 7.18 | 8.02 | 0.32 | 0.58 | 0.56 |
| dregon | u | 80 | 1.63 | 2.81 | 1.66 | 2.83 | 1.67 | 0.48 | 0.91 | 0.90 |
| dregon | uj | 1120 | 2.73 | 1.49 | 2.75 | 1.52 | 2.75 | 0.79 | 0.73 | 0.70 |
| dregon | v k1-2 | 640 | 0.00 | 0.00 | 0.00 | 0.00 | 1.14 | — | — | 0.51 |
| dregon | v k3-8 | 1920 | 0.00 | 0.00 | 0.00 | 0.00 | 1.14 | — | — | 0.51 |
| dregon | v k9-24 | 5120 | 3.87 | 2.80 | 4.00 | 2.98 | 4.03 | 0.62 | 0.59 | 0.53 |
| dregon | v k25-60 | 11520 | 4.05 | 3.75 | 4.19 | 3.91 | 4.21 | 0.49 | 0.48 | 0.45 |
| dregon | v k61+ | 8960 | 4.33 | 7.09 | 4.47 | 7.18 | 4.47 | 0.46 | 0.58 | 0.56 |
| dregon | v_present k61+ | 32 | 7.94 | 7.09 | 8.02 | 7.18 | 8.02 | 0.32 | 0.58 | 0.56 |
| michaels_cruise | d | 512 | 0.70 | 0.25 | 0.84 | 0.52 | 1.33 | 0.52 | 0.67 | 0.42 |
| michaels_cruise | v | 41472 | 3.76 | 4.15 | 3.91 | 4.28 | 3.93 | 0.65 | 0.66 | 0.62 |
| michaels_cruise | v_present | 4240 | 4.21 | 3.59 | 4.33 | 3.72 | 4.36 | 0.64 | 0.65 | 0.61 |
| michaels_cruise | u | 128 | 2.91 | 1.70 | 2.93 | 1.73 | 2.93 | 0.84 | 0.80 | 0.78 |
| michaels_cruise | uj | 1792 | 2.67 | 1.60 | 2.69 | 1.63 | 2.69 | 0.78 | 0.69 | 0.67 |
| michaels_cruise | v k1-2 | 1024 | 0.30 | 0.12 | 0.42 | 0.32 | 1.18 | 0.91 | 0.90 | 0.65 |
| michaels_cruise | v k3-8 | 3072 | 4.67 | 4.04 | 4.79 | 4.17 | 4.81 | 0.68 | 0.69 | 0.65 |
| michaels_cruise | v k9-24 | 8192 | 4.78 | 3.64 | 4.90 | 3.79 | 4.92 | 0.67 | 0.60 | 0.56 |
| michaels_cruise | v k25-60 | 18432 | 3.32 | 3.45 | 3.47 | 3.61 | 3.50 | 0.64 | 0.67 | 0.62 |
| michaels_cruise | v k61+ | 10752 | 3.49 | 5.58 | 3.65 | 5.69 | 3.67 | 0.61 | 0.67 | 0.65 |
| michaels_cruise | v_present k1-2 | 528 | 0.32 | 0.12 | 0.44 | 0.32 | 1.18 | 0.91 | 0.90 | 0.65 |
| michaels_cruise | v_present k3-8 | 1776 | 4.66 | 4.04 | 4.78 | 4.17 | 4.79 | 0.67 | 0.69 | 0.65 |
| michaels_cruise | v_present k9-24 | 1904 | 4.38 | 3.64 | 4.50 | 3.79 | 4.52 | 0.62 | 0.60 | 0.56 |
| michaels_cruise | v_present k25-60 | 32 | 2.33 | 3.45 | 2.55 | 3.61 | 2.59 | 0.52 | 0.67 | 0.62 |
| michaels_standby | d | 96 | 0.91 | 0.24 | 1.02 | 0.52 | 1.46 | 0.57 | 0.67 | 0.42 |
| michaels_standby | v | 12480 | 3.24 | 4.74 | 3.41 | 4.86 | 3.43 | 0.62 | 0.67 | 0.63 |
| michaels_standby | v_present | 328 | 6.57 | 5.58 | 6.66 | 5.69 | 6.67 | 0.75 | 0.67 | 0.65 |
| michaels_standby | u | 24 | 0.90 | 1.70 | 0.96 | 1.73 | 0.97 | 0.53 | 0.80 | 0.78 |
| michaels_standby | uj | 336 | 1.64 | 1.60 | 1.67 | 1.63 | 1.67 | 0.74 | 0.69 | 0.67 |
| michaels_standby | v k1-2 | 192 | 0.31 | 0.12 | 0.43 | 0.32 | 1.18 | 0.84 | 0.90 | 0.65 |
| michaels_standby | v k3-8 | 576 | 3.06 | 4.04 | 3.24 | 4.17 | 3.27 | 0.51 | 0.69 | 0.65 |
| michaels_standby | v k9-24 | 1536 | 3.40 | 3.64 | 3.56 | 3.79 | 3.59 | 0.54 | 0.60 | 0.56 |
| michaels_standby | v k25-60 | 3456 | 2.81 | 3.45 | 2.99 | 3.61 | 3.03 | 0.67 | 0.67 | 0.62 |
| michaels_standby | v k61+ | 6720 | 3.46 | 5.58 | 3.63 | 5.69 | 3.64 | 0.63 | 0.67 | 0.65 |
| michaels_standby | v_present k61+ | 328 | 6.57 | 5.58 | 6.66 | 5.69 | 6.67 | 0.75 | 0.67 | 0.65 |

The same check as second moments (dB²): fitted mean square + posterior variance against the measured σ², and at lag one the fitted lag-1 product + the posterior lag-1 covariance against ρσ² (per block pair).

| fit | family | fitted ms | + posterior var | = lag-0 sum | measured σ² | fitted lag-1 product | + posterior lag-1 cov | = lag-1 sum | measured ρσ² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dregon | d | 3.48 | 0.52 | 4.00 | 0.89 | 1.42 | 0.06 | 1.48 | 0.17 |
| dregon | v | 15.38 | 1.06 | 16.44 | 24.24 | 7.78 | 0.04 | 7.82 | 12.87 |
| dregon | v_present | 63.08 | 1.24 | 64.32 | 51.51 | 20.74 | 0.02 | 20.77 | 29.05 |
| dregon | u | 2.66 | 0.10 | 2.77 | 8.02 | 1.28 | 0.01 | 1.29 | 7.21 |
| dregon | uj | 7.46 | 0.10 | 7.56 | 2.31 | 5.98 | 0.01 | 5.99 | 1.61 |
| dregon | v k1-2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| dregon | v k3-8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| dregon | v k9-24 | 14.95 | 1.05 | 16.00 | 8.86 | 9.29 | 0.09 | 9.38 | 4.70 |
| dregon | v k25-60 | 16.39 | 1.16 | 17.55 | 15.25 | 8.08 | 0.05 | 8.13 | 6.79 |
| dregon | v k61+ | 18.73 | 1.24 | 19.97 | 51.51 | 8.74 | 0.02 | 8.77 | 29.05 |
| dregon | v_present k61+ | 63.08 | 1.24 | 64.32 | 51.51 | 20.74 | 0.02 | 20.77 | 29.05 |
| michaels_cruise | d | 0.49 | 0.21 | 0.70 | 0.28 | 0.25 | 0.07 | 0.32 | 0.12 |
| michaels_cruise | v | 14.17 | 1.09 | 15.26 | 18.29 | 9.14 | 0.07 | 9.21 | 11.47 |
| michaels_cruise | v_present | 17.74 | 0.98 | 18.72 | 13.86 | 11.33 | 0.07 | 11.40 | 8.41 |
| michaels_cruise | u | 8.46 | 0.10 | 8.57 | 3.00 | 6.85 | 0.01 | 6.86 | 2.33 |
| michaels_cruise | uj | 7.10 | 0.11 | 7.21 | 2.67 | 5.46 | 0.01 | 5.47 | 1.78 |
| michaels_cruise | v k1-2 | 0.09 | 0.09 | 0.18 | 0.10 | 0.08 | 0.05 | 0.14 | 0.07 |
| michaels_cruise | v k3-8 | 21.83 | 1.11 | 22.94 | 17.40 | 14.78 | 0.08 | 14.85 | 11.27 |
| michaels_cruise | v k9-24 | 22.87 | 1.11 | 23.98 | 14.39 | 15.24 | 0.07 | 15.31 | 8.06 |
| michaels_cruise | v k25-60 | 10.99 | 1.07 | 12.06 | 13.01 | 6.99 | 0.09 | 7.08 | 8.12 |
| michaels_cruise | v k61+ | 12.16 | 1.18 | 13.34 | 32.32 | 7.43 | 0.05 | 7.47 | 20.94 |
| michaels_cruise | v_present k1-2 | 0.10 | 0.09 | 0.19 | 0.10 | 0.10 | 0.05 | 0.15 | 0.07 |
| michaels_cruise | v_present k3-8 | 21.70 | 1.11 | 22.81 | 17.40 | 14.37 | 0.08 | 14.44 | 11.27 |
| michaels_cruise | v_present k9-24 | 19.15 | 1.11 | 20.26 | 14.39 | 11.75 | 0.07 | 11.82 | 8.06 |
| michaels_cruise | v_present k25-60 | 5.42 | 1.07 | 6.49 | 13.01 | 2.84 | 0.09 | 2.93 | 8.12 |
| michaels_standby | d | 0.83 | 0.22 | 1.05 | 0.28 | 0.42 | 0.08 | 0.49 | 0.12 |
| michaels_standby | v | 10.48 | 1.13 | 11.61 | 23.58 | 6.53 | 0.06 | 6.59 | 15.04 |
| michaels_standby | v_present | 43.21 | 1.19 | 44.39 | 32.32 | 34.01 | 0.05 | 34.05 | 20.94 |
| michaels_standby | u | 0.82 | 0.11 | 0.92 | 3.00 | 0.46 | 0.01 | 0.47 | 2.33 |
| michaels_standby | uj | 2.68 | 0.11 | 2.79 | 2.67 | 1.89 | 0.01 | 1.89 | 1.78 |
| michaels_standby | v k1-2 | 0.10 | 0.09 | 0.19 | 0.10 | 0.08 | 0.05 | 0.13 | 0.07 |
| michaels_standby | v k3-8 | 9.39 | 1.11 | 10.51 | 17.40 | 4.79 | 0.08 | 4.87 | 11.27 |
| michaels_standby | v k9-24 | 11.59 | 1.12 | 12.71 | 14.39 | 6.39 | 0.07 | 6.46 | 8.06 |
| michaels_standby | v k25-60 | 7.87 | 1.08 | 8.95 | 13.01 | 5.21 | 0.09 | 5.30 | 8.12 |
| michaels_standby | v k61+ | 11.96 | 1.19 | 13.15 | 32.32 | 7.57 | 0.05 | 7.62 | 20.94 |
| michaels_standby | v_present k61+ | 43.21 | 1.19 | 44.39 | 32.32 | 34.01 | 0.05 | 34.05 | 20.94 |

