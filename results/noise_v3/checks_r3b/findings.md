# Noise model v3 — the section 3.5 checks

`scripts/noise_v3_checks.py` (render and measure only). Every number below is read from the JSON named in its section.

## (b) Held-out line-power spread and intermittency (`heldout/heldout.json`)

Wander estimator of `noise_v3_measure_wander.py` (rig-centred, block 0.5 s, τ at lag 1), window bootstrap median [5 %, 95 %]:

| rig | arm | windows | line tracks | σ_total dB | σ_d dB | σ_v dB | σ_u dB |
|---|---|---:|---:|---|---|---|---|
| dregon | real | 5 | 0 | — (no track) | — (no track) | — (no track) | 3.66 (3.62 [2.52, 4.04]) |
| dregon | v3 | 20 | 0 | — (no track) | — (no track) | — (no track) | 0.96 (0.94 [0.80, 1.06]) |
| dregon | v2 | 20 | 0 | — (no track) | — (no track) | — (no track) | 0.54 (0.54 [0.44, 0.63]) |
| michaels | real | 4 | 5 | 1.46 (1.46 [0.53, 1.50]) | 0.12 (0.12 [0.00, 0.12]) | 1.46 (1.46 [0.53, 1.50]) | 2.97 (2.96 [0.83, 2.97]) |
| michaels | v3 | 16 | 23 | 3.03 (2.94 [2.53, 3.38]) | 0.00 (0.00 [0.00, 0.56]) | 3.03 (2.93 [2.53, 3.38]) | 0.66 (0.66 [0.61, 0.69]) |
| michaels | v2 | 16 | 30 | 0.38 (0.38 [0.28, 0.44]) | 0.26 (0.26 [0.18, 0.30]) | 0.29 (0.28 [0.14, 0.37]) | 0.21 (0.21 [0.15, 0.26]) |

Mid-order lines (k 8-24, every rotor), per block: UNDER the floor = line power below the local floor (prominence < 3.01 dB); PRESENT = window prominence >= 6 dB; VISIBLE = >= 6 dB in >= 20 % of the window's blocks, and the disappearance rate is the share of a visible line's blocks under 6 dB. Window bootstrap median [5 %, 95 %]:

| rig | arm | lines | cells under floor | present-line blocks under floor | lines that appear and disappear | block sd of a present line dB | visible lines | disappearance rate |
|---|---|---:|---|---|---|---|---:|---|
| dregon | real | 340 (0 present) | 81.8 [78.1, 85.8] % | — [—, —] % | 10.9 [6.2, 15.3] % | — [—, —] | 10 | 68.8 [66.4, 72.5] % |
| dregon | v3 | 1360 (0 present) | 87.3 [86.2, 88.4] % | — [—, —] % | 6.8 [5.7, 7.8] % | — [—, —] | 42 | 66.1 [63.9, 68.1] % |
| dregon | v2 | 1360 (0 present) | 92.2 [90.3, 94.1] % | — [—, —] % | 1.8 [0.0, 3.7] % | — [—, —] | 0 | — [—, —] % |
| michaels | real | 272 (26 present) | 80.3 [66.6, 94.5] % | 7.2 [4.1, 17.7] % | 22.1 [3.3, 39.8] % | 2.76 [2.57, 3.91] | 38 | 38.2 [32.9, 51.1] % |
| michaels | v3 | 1088 (138 present) | 74.3 [60.3, 89.6] % | 9.3 [7.9, 11.3] % | 28.1 [6.0, 45.2] % | 2.61 [2.53, 2.78] | 202 | 40.8 [35.5, 46.6] % |
| michaels | v2 | 1088 (108 present) | 79.1 [60.0, 98.1] % | 3.2 [1.6, 5.5] % | 17.0 [0.0, 32.8] % | 1.18 [1.14, 1.21] | 170 | 33.9 [29.9, 38.6] % |

Every measured order by wander order group (the groups `sigma_v_db_by_order` is measured on), the same statistics, window bootstrap median [5 %, 95 %]:

| rig | orders | arm | lines | cells under floor | present-line blocks under floor | lines that appear and disappear | block sd of a present line dB | visible lines | disappearance rate |
|---|---|---|---:|---|---|---|---|---:|---|
| dregon | 1-2 | real | 20 (0 present) | 58.8 [31.2, 87.5] % | — [—, —] % | 50.0 [15.0, 80.0] % | — [—, —] | 8 | 71.9 [68.8, 75.0] % |
| dregon | 1-2 | v3 | 80 (0 present) | 21.4 [18.8, 24.2] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 1-2 | v2 | 80 (0 present) | 11.6 [7.2, 15.9] % | — [—, —] % | 8.8 [0.0, 16.2] % | — [—, —] | 3 | 75.0 [75.0, 75.0] % |
| dregon | 3-8 | real | 120 (0 present) | 87.0 [83.2, 91.6] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 3-8 | v3 | 480 (0 present) | 98.7 [98.1, 99.1] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 3-8 | v2 | 480 (0 present) | 97.8 [95.6, 100.0] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 9-24 | real | 320 (0 present) | 81.2 [77.6, 85.9] % | — [—, —] % | 11.6 [6.6, 15.9] % | — [—, —] | 10 | 68.8 [66.2, 71.9] % |
| dregon | 9-24 | v3 | 1280 (0 present) | 86.6 [85.4, 87.9] % | — [—, —] % | 7.3 [6.1, 8.4] % | — [—, —] | 42 | 66.1 [63.9, 68.1] % |
| dregon | 9-24 | v2 | 1280 (0 present) | 91.9 [90.0, 93.8] % | — [—, —] % | 2.0 [0.0, 5.9] % | — [—, —] | 0 | — [—, —] % |
| dregon | 25-60 | real | 720 (0 present) | 89.2 [84.8, 93.5] % | — [—, —] % | 6.8 [2.8, 12.4] % | — [—, —] | 10 | 72.7 [68.8, 75.0] % |
| dregon | 25-60 | v3 | 2880 (1 present) | 87.2 [85.9, 88.6] % | 50.0 [50.0, 50.0] % | 7.7 [5.9, 9.5] % | 4.60 [4.60, 4.60] | 38 | 73.4 [71.1, 74.7] % |
| dregon | 25-60 | v2 | 2880 (0 present) | 98.6 [98.2, 99.0] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 61+ | real | 753 (0 present) | 98.2 [97.3, 99.1] % | — [—, —] % | 1.6 [0.0, 3.2] % | — [—, —] | 2 | 75.0 [75.0, 75.0] % |
| dregon | 61+ | v3 | 3012 (0 present) | 86.5 [85.4, 87.9] % | — [—, —] % | 11.9 [9.1, 14.9] % | — [—, —] | 49 | 73.0 [71.7, 73.8] % |
| dregon | 61+ | v2 | 3012 (0 present) | 99.5 [99.4, 99.6] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| michaels | 1-2 | real | 10 (8 present) | 4.7 [1.3, 8.5] % | 0.0 [0.0, 0.0] % | 10.0 [0.0, 20.0] % | 2.27 [1.16, 3.36] | 9 | 5.6 [0.0, 10.3] % |
| michaels | 1-2 | v3 | 40 (32 present) | 9.2 [4.9, 13.1] % | 0.0 [0.0, 0.0] % | 5.0 [5.0, 5.0] % | 1.26 [1.20, 1.41] | 33 | 1.0 [0.0, 1.9] % |
| michaels | 1-2 | v2 | 40 (32 present) | 13.3 [8.1, 17.9] % | 0.0 [0.0, 0.0] % | 0.0 [0.0, 0.0] % | 0.87 [0.75, 1.14] | 32 | 0.0 [0.0, 0.0] % |
| michaels | 3-8 | real | 96 (22 present) | 62.2 [44.6, 78.1] % | 6.5 [1.0, 13.1] % | 25.0 [8.3, 46.9] % | 2.80 [2.13, 3.54] | 24 | 23.7 [18.8, 29.5] % |
| michaels | 3-8 | v3 | 384 (106 present) | 61.6 [39.3, 74.0] % | 5.7 [2.6, 8.6] % | 31.0 [21.6, 45.8] % | 3.12 [2.90, 3.36] | 125 | 25.7 [17.9, 36.0] % |
| michaels | 3-8 | v2 | 384 (102 present) | 68.1 [37.5, 99.8] % | 2.2 [0.9, 3.4] % | 16.4 [0.0, 32.8] % | 1.19 [0.86, 1.52] | 120 | 17.6 [13.3, 21.3] % |
| michaels | 9-24 | real | 256 (22 present) | 81.5 [68.5, 95.4] % | 8.2 [4.8, 20.0] % | 21.5 [1.2, 37.9] % | 2.96 [2.59, 3.88] | 34 | 40.4 [34.6, 54.4] % |
| michaels | 9-24 | v3 | 1024 (118 present) | 75.3 [61.8, 90.0] % | 9.9 [8.8, 11.5] % | 28.2 [4.9, 45.2] % | 2.68 [2.53, 2.81] | 180 | 42.8 [37.4, 48.6] % |
| michaels | 9-24 | v2 | 1024 (88 present) | 80.2 [62.4, 98.2] % | 3.8 [1.8, 6.8] % | 17.1 [0.0, 32.5] % | 1.37 [1.14, 1.43] | 146 | 37.2 [33.8, 41.5] % |
| michaels | 25-60 | real | 576 (1 present) | 96.1 [92.6, 99.6] % | 0.0 [0.0, 0.0] % | 3.5 [0.0, 6.9] % | 1.39 [1.39, 1.39] | 2 | 65.6 [65.6, 65.6] % |
| michaels | 25-60 | v3 | 2304 (11 present) | 87.0 [78.9, 96.1] % | 22.7 [18.8, 29.7] % | 18.1 [3.0, 30.3] % | 2.96 [2.96, 3.04] | 47 | 65.8 [65.3, 67.6] % |
| michaels | 25-60 | v2 | 2304 (0 present) | 97.2 [95.0, 99.9] % | — [—, —] % | 0.7 [0.0, 1.3] % | — [—, —] | 0 | — [—, —] % |
| michaels | 61+ | real | 1588 (16 present) | 98.2 [97.9, 99.9] % | 5.1 [2.3, 7.8] % | 1.6 [0.0, 1.9] % | 1.42 [1.40, 1.42] | 23 | 25.8 [25.6, 26.0] % |
| michaels | 61+ | v3 | 6352 (61 present) | 94.2 [90.7, 95.4] % | 13.7 [12.7, 14.5] % | 9.3 [6.7, 16.9] % | 3.81 [3.59, 4.03] | 125 | 48.0 [45.1, 75.0] % |
| michaels | 61+ | v2 | 6352 (109 present) | 96.9 [96.2, 100.0] % | 7.4 [7.0, 7.8] % | 2.2 [0.0, 2.7] % | 2.04 [2.03, 2.19] | 149 | 30.4 [29.6, 31.2] % |

Figure: `heldout/prominence_hist.png`.

## (c) Tonality on rendered audio (`tonality/rendered.json`)

R4 `line_width_db3` (-3 dB width, Hz, 8192-point, resolution 1.95 Hz) at k = 1..8 and the order-6-dB counts on the window prominence ladder; renders: median over the pattern windows x 4 seeds.

| fit | arm | width k=1..8 Hz | orders >= 6 dB / rotor | highest >= 6 dB | prom k=1,2,4,8,16 dB |
|---|---|---|---:|---:|---|
| dregon | real | 10.4, 23.4, —, —, —, —, 9.0, 30.6 | 0.0 | 0 | —, 4.6, 2.9, 1.6, 1.8 |
| dregon | v3 | 9.9, 17.9, —, —, —, 22.2, 15.6, 16.6 | 0.0 | 0 | —, 4.2, 0.8, 0.6, 1.8 |
| dregon | v2 | 10.3, 10.2, —, 9.3, —, —, 7.0, 10.4 | 0.0 | 0 | —, 4.1, 0.5, 0.4, 1.8 |
| michaels_cruise | real | 3.1, 4.2, —, 5.7, 18.9, 8.2, 58.8, 13.8 | 9.0 | 18 | —, 24.8, 12.0, 7.8, 4.3 |
| michaels_cruise | v3 | 2.5, 5.0, —, 8.0, 19.9, 9.4, 40.4, 15.0 | 8.6 | 19 | —, 23.0, 11.3, 7.2, 4.0 |
| michaels_cruise | v2 | —, 4.6, —, 7.6, 18.2, 10.3, —, 16.0 | 7.0 | 15 | —, 22.0, 10.3, 7.1, 2.0 |
| michaels_standby | real | —, 4.0, —, 4.4, 2.0, 4.1, 12.9, 8.5 | 3.1 | 88 | —, —, 1.1, 1.1, 1.3 |
| michaels_standby | v3 | 8.7, 4.1, 0.8, 2.8, 3.5, 3.5, 5.9, 4.6 | 2.2 | 104 | —, —, -0.9, 1.1, 1.0 |
| michaels_standby | v2 | 11.9, 4.0, —, 2.9, 6.2, 3.3, 7.4, 5.2 | 3.9 | 86 | —, —, -0.5, 0.7, 0.8 |

Visible-order counts per rotor at 3 / 6 / 10 dB (rotor mean), per pattern window: `audit` = the tonality audit's estimator on the clip's time-mean periodogram (orders 1..k_max of the v3 fit); `ladder` = the wander estimator's window prominence. Renders: median over the 4 seeds.

| fit | pattern | arm | audit >= 3 / 6 / 10 dB | audit highest >= 6 dB | ladder >= 3 / 6 / 10 dB |
|---|---|---|---|---:|---|
| dregon | dregon_cruise_narrow | real | 7.5 / 1.0 / 0.0 | 1 | 4.8 / 0.0 / 0.0 |
| dregon | dregon_cruise_narrow | v3 | 4.4 / 2.0 / 0.0 | 2 | 2.1 / 0.0 / 0.0 |
| dregon | dregon_cruise_narrow | v2 | 3.0 / 2.0 / 0.0 | 2 | 2.0 / 0.0 / 0.0 |
| dregon | dregon_cruise_wide | real | 5.5 / 1.0 / 0.0 | 1 | 7.5 / 0.0 / 0.0 |
| dregon | dregon_cruise_wide | v3 | 4.4 / 2.0 / 0.0 | 2 | 5.9 / 0.1 / 0.0 |
| dregon | dregon_cruise_wide | v2 | 2.6 / 1.0 / 0.0 | 1 | 1.8 / 0.0 / 0.0 |
| michaels_cruise | michaels_cruise_narrow | real | 8.8 / 3.0 / 1.2 | 9 | 12.2 / 7.8 / 3.0 |
| michaels_cruise | michaels_cruise_narrow | v3 | 8.2 / 4.2 / 1.5 | 12 | 17.4 / 8.5 / 3.2 |
| michaels_cruise | michaels_cruise_narrow | v2 | 7.8 / 1.9 / 1.2 | 6 | 11.2 / 6.5 / 1.8 |
| michaels_cruise | michaels_cruise_wide | real | 11.8 / 6.8 / 3.2 | 11 | 16.5 / 10.2 / 5.0 |
| michaels_cruise | michaels_cruise_wide | v3 | 11.0 / 6.5 / 3.5 | 14 | 20.5 / 9.4 / 4.2 |
| michaels_cruise | michaels_cruise_wide | v2 | 10.5 / 5.5 / 2.8 | 11 | 12.1 / 7.1 / 3.2 |
| michaels_standby | michaels_standby_narrow | real | 7.0 / 3.5 / 2.8 | 86 | 4.8 / 3.2 / 2.2 |
| michaels_standby | michaels_standby_narrow | v3 | 22.2 / 4.1 / 1.9 | 84 | 9.5 / 2.4 / 0.8 |
| michaels_standby | michaels_standby_narrow | v2 | 8.9 / 3.9 / 2.0 | 84 | 6.4 / 4.8 / 1.1 |
| michaels_standby | michaels_standby_wide | real | 4.5 / 1.2 / 0.0 | 65 | 11.0 / 3.0 / 1.0 |
| michaels_standby | michaels_standby_wide | v3 | 7.5 / 2.4 / 1.4 | 45 | 9.8 / 1.6 / 0.8 |
| michaels_standby | michaels_standby_wide | v2 | 4.4 / 2.0 / 1.2 | 44 | 5.0 / 3.0 / 1.0 |

## (c) Tonality on the expectation (`tonality/fits.json`, `noise_v2_tonality_audit.py --fit`)

| pattern | payload | prom k=1,2,4,8,16 dB (mic median) | orders >= 3 / 6 / 10 dB / rotor | highest >= 6 dB | trend crosses floor at k | pedestal median dB |
|---|---|---|---|---:|---:|---:|
| dregon_cruise_narrow | v3 dregon | 9.0, 9.4, 4.1, 1.9, 3.0 | 10.5 / 2.5 / 0.0 | 12 | 88 | 8.7 |
| dregon_cruise_narrow | v2 anchor dregon | 9.1, 9.4, 3.2, 5.4, 8.5 | 23.2 / 9.5 / 2.0 | 37 | 89 | 8.5 |
| dregon_cruise_wide | v3 dregon | 8.5, 7.1, 2.6, 0.5, 0.7 | 8.2 / 2.0 / 0.0 | 2 | 88 | 7.4 |
| dregon_cruise_wide | v2 anchor dregon | 9.5, 7.0, 1.9, 1.4, 3.1 | 17.2 / 6.8 / 0.2 | 38 | 89 | 8.8 |
| michaels_cruise_narrow | v3 michaels | 5.4, 30.5, 9.2, 4.3, 1.9 | 17.5 / 9.8 / 4.8 | 23 | 82 | 6.2 |
| michaels_cruise_narrow | v2 anchor michaels | 2.0, 29.6, 8.1, 5.4, 1.7 | 15.0 / 8.0 / 4.2 | 20 | 73 | 5.7 |
| michaels_cruise_wide | v3 michaels | 1.8, 26.2, 11.8, 2.9, 4.1 | 19.0 / 10.2 / 4.2 | 22 | 82 | 6.8 |
| michaels_cruise_wide | v2 anchor michaels | -2.2, 25.7, 9.1, 4.6, 3.1 | 17.0 / 8.2 / 3.8 | 20 | 70 | 10.6 |
| michaels_standby_narrow | v3 michaels | 13.1, 11.1, 1.6, 1.1, 2.5 | 8.8 / 3.8 / 2.0 | 86 | 108 | 1.4 |
| michaels_standby_narrow | v2 anchor michaels_standby | 9.0, 11.5, 1.4, 0.7, 1.8 | 8.5 / 4.2 / 2.8 | 86 | 127 | 8.3 |
| michaels_standby_wide | v3 michaels | 20.5, 6.7, 2.5, 3.8, 2.6 | 9.5 / 3.0 / 2.0 | 85 | 115 | 0.0 |
| michaels_standby_wide | v2 anchor michaels_standby | 14.8, 7.0, 2.1, 2.2, 2.1 | 11.5 / 3.8 / 1.5 | 85 | 129 | 3.9 |

## (d) Parity gate (`parity/arm_*.json`, `noise_v2_round_score.py --fit`)

HPPNet PIT MAE on renders of the frozen supports (four frozen render seeds, 8 mics) against the legacy PARITY bar and DREGON's frozen STRETCH target; proxy `ltas_abs_db` against its closure-0.7 gate.

| arm | rig | PIT MAE rev/s (mean) | 95 % upper | Michael's ratio | parity bar | parity | stretch bar | stretch | proxy dB | proxy gate dB |
|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|
| `arm_dregon_v3.json` | dregon | 1.700280 | 2.047514 | — | 2.187786 | PASS | 1.897063 | PASS | 2.5606 | 1.9786 |
| `arm_michaels_v3.json` | michaels | 1.591524 | — | 0.5258 | 3.177994 | PASS | — | — | 2.4837 | 1.2197 |

## (e) Fitted latents against the measured wander (`latents/latents.json`)

Measured σ = rms of the fit's prior sd over the family (v: per order group); posterior variance and the MAP's expected sd / lag-1 = the OU prior seen through the measured block noise s² (Kalman smoother); correctly shrunk latents give fitted sd = MAP expected sd, sqrt(fitted ms + posterior var) = measured σ, fitted lag-1 = MAP expected lag-1.

| fit | family | values | fitted sd dB | MAP expected sd dB | sqrt(fitted ms + posterior var) dB | measured σ dB | sqrt(fitted var + s²) dB | fitted lag-1 | MAP expected lag-1 | measured ρ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dregon | d | 320 | 1.78 | 0.61 | 1.92 | 0.94 | 2.12 | 0.38 | 0.30 | 0.19 |
| dregon | v | 28160 | 4.12 | 4.81 | 4.24 | 4.92 | 4.27 | 0.52 | 0.55 | 0.50 |
| dregon | v_present | 32 | 8.92 | 7.09 | 8.99 | 7.18 | 8.99 | 0.43 | 0.58 | 0.56 |
| dregon | u | 80 | 1.38 | 2.81 | 1.42 | 2.83 | 1.42 | 0.61 | 0.91 | 0.90 |
| dregon | uj | 1120 | 1.85 | 1.49 | 1.88 | 1.52 | 1.88 | 0.48 | 0.73 | 0.70 |
| dregon | v k1-2 | 640 | 0.00 | 0.00 | 0.00 | 0.00 | 1.14 | — | — | 0.51 |
| dregon | v k3-8 | 1920 | 0.00 | 0.00 | 0.00 | 0.00 | 1.14 | — | — | 0.51 |
| dregon | v k9-24 | 5120 | 3.89 | 2.80 | 4.02 | 2.98 | 4.05 | 0.60 | 0.59 | 0.53 |
| dregon | v k25-60 | 11520 | 4.15 | 3.75 | 4.28 | 3.91 | 4.30 | 0.50 | 0.48 | 0.45 |
| dregon | v k61+ | 8960 | 4.75 | 7.09 | 4.87 | 7.18 | 4.88 | 0.51 | 0.58 | 0.56 |
| dregon | v_present k61+ | 32 | 8.92 | 7.09 | 8.99 | 7.18 | 8.99 | 0.43 | 0.58 | 0.56 |
| michaels_cruise | d | 512 | 0.75 | 0.25 | 0.88 | 0.52 | 1.36 | 0.56 | 0.67 | 0.42 |
| michaels_cruise | v | 41472 | 3.72 | 4.15 | 3.87 | 4.28 | 3.89 | 0.64 | 0.66 | 0.62 |
| michaels_cruise | v_present | 4240 | 4.29 | 3.59 | 4.41 | 3.72 | 4.44 | 0.66 | 0.65 | 0.61 |
| michaels_cruise | u | 128 | 2.14 | 1.70 | 2.16 | 1.73 | 2.16 | 0.86 | 0.80 | 0.78 |
| michaels_cruise | uj | 1792 | 2.35 | 1.60 | 2.38 | 1.63 | 2.38 | 0.68 | 0.69 | 0.67 |
| michaels_cruise | v k1-2 | 1024 | 0.40 | 0.12 | 0.49 | 0.32 | 1.20 | 0.92 | 0.90 | 0.65 |
| michaels_cruise | v k3-8 | 3072 | 4.85 | 4.04 | 4.96 | 4.17 | 4.98 | 0.69 | 0.69 | 0.65 |
| michaels_cruise | v k9-24 | 8192 | 4.71 | 3.64 | 4.83 | 3.79 | 4.85 | 0.66 | 0.60 | 0.56 |
| michaels_cruise | v k25-60 | 18432 | 3.30 | 3.45 | 3.46 | 3.61 | 3.49 | 0.64 | 0.67 | 0.62 |
| michaels_cruise | v k61+ | 10752 | 3.34 | 5.58 | 3.51 | 5.69 | 3.53 | 0.57 | 0.67 | 0.65 |
| michaels_cruise | v_present k1-2 | 528 | 0.39 | 0.12 | 0.49 | 0.32 | 1.20 | 0.92 | 0.90 | 0.65 |
| michaels_cruise | v_present k3-8 | 1776 | 4.81 | 4.04 | 4.93 | 4.17 | 4.95 | 0.69 | 0.69 | 0.65 |
| michaels_cruise | v_present k9-24 | 1904 | 4.39 | 3.64 | 4.52 | 3.79 | 4.54 | 0.62 | 0.60 | 0.56 |
| michaels_cruise | v_present k25-60 | 32 | 2.32 | 3.45 | 2.54 | 3.61 | 2.58 | 0.52 | 0.67 | 0.62 |
| michaels_standby | d | 96 | 0.90 | 0.24 | 1.02 | 0.52 | 1.45 | 0.57 | 0.67 | 0.42 |
| michaels_standby | v | 12480 | 3.17 | 4.74 | 3.34 | 4.86 | 3.36 | 0.59 | 0.67 | 0.63 |
| michaels_standby | v_present | 328 | 6.62 | 5.58 | 6.71 | 5.69 | 6.72 | 0.75 | 0.67 | 0.65 |
| michaels_standby | u | 24 | 1.15 | 1.70 | 1.19 | 1.73 | 1.20 | 0.60 | 0.80 | 0.78 |
| michaels_standby | uj | 336 | 1.37 | 1.60 | 1.41 | 1.63 | 1.41 | 0.56 | 0.69 | 0.67 |
| michaels_standby | v k1-2 | 192 | 0.32 | 0.12 | 0.44 | 0.32 | 1.18 | 0.84 | 0.90 | 0.65 |
| michaels_standby | v k3-8 | 576 | 2.95 | 4.04 | 3.13 | 4.17 | 3.16 | 0.45 | 0.69 | 0.65 |
| michaels_standby | v k9-24 | 1536 | 3.38 | 3.64 | 3.54 | 3.79 | 3.57 | 0.53 | 0.60 | 0.56 |
| michaels_standby | v k25-60 | 3456 | 2.78 | 3.45 | 2.97 | 3.61 | 3.00 | 0.65 | 0.67 | 0.62 |
| michaels_standby | v k61+ | 6720 | 3.36 | 5.58 | 3.53 | 5.69 | 3.54 | 0.60 | 0.67 | 0.65 |
| michaels_standby | v_present k61+ | 328 | 6.62 | 5.58 | 6.71 | 5.69 | 6.72 | 0.75 | 0.67 | 0.65 |

The same check as second moments (dB²): fitted mean square + posterior variance against the measured σ², and at lag one the fitted lag-1 product + the posterior lag-1 covariance against ρσ² (per block pair).

| fit | family | fitted ms | + posterior var | = lag-0 sum | measured σ² | fitted lag-1 product | + posterior lag-1 cov | = lag-1 sum | measured ρσ² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dregon | d | 3.18 | 0.52 | 3.69 | 0.89 | 1.21 | 0.06 | 1.27 | 0.17 |
| dregon | v | 16.95 | 1.06 | 18.01 | 24.24 | 8.90 | 0.04 | 8.95 | 12.87 |
| dregon | v_present | 79.52 | 1.24 | 80.76 | 51.51 | 35.38 | 0.02 | 35.40 | 29.05 |
| dregon | u | 1.91 | 0.10 | 2.02 | 8.02 | 1.17 | 0.01 | 1.18 | 7.21 |
| dregon | uj | 3.42 | 0.10 | 3.52 | 2.31 | 1.64 | 0.01 | 1.65 | 1.61 |
| dregon | v k1-2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| dregon | v k3-8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| dregon | v k9-24 | 15.14 | 1.05 | 16.19 | 8.86 | 9.16 | 0.09 | 9.25 | 4.70 |
| dregon | v k25-60 | 17.19 | 1.16 | 18.34 | 15.25 | 8.66 | 0.05 | 8.71 | 6.79 |
| dregon | v k61+ | 22.52 | 1.24 | 23.76 | 51.51 | 11.62 | 0.02 | 11.64 | 29.05 |
| dregon | v_present k61+ | 79.52 | 1.24 | 80.76 | 51.51 | 35.38 | 0.02 | 35.40 | 29.05 |
| michaels_cruise | d | 0.56 | 0.21 | 0.78 | 0.28 | 0.31 | 0.07 | 0.38 | 0.12 |
| michaels_cruise | v | 13.86 | 1.09 | 14.94 | 18.29 | 8.82 | 0.07 | 8.89 | 11.47 |
| michaels_cruise | v_present | 18.43 | 0.98 | 19.42 | 13.86 | 12.06 | 0.07 | 12.13 | 8.41 |
| michaels_cruise | u | 4.56 | 0.10 | 4.66 | 3.00 | 3.77 | 0.01 | 3.78 | 2.33 |
| michaels_cruise | uj | 5.53 | 0.11 | 5.64 | 2.67 | 3.68 | 0.01 | 3.68 | 1.78 |
| michaels_cruise | v k1-2 | 0.16 | 0.09 | 0.24 | 0.10 | 0.14 | 0.05 | 0.20 | 0.07 |
| michaels_cruise | v k3-8 | 23.48 | 1.11 | 24.58 | 17.40 | 16.20 | 0.08 | 16.27 | 11.27 |
| michaels_cruise | v k9-24 | 22.23 | 1.11 | 23.34 | 14.39 | 14.61 | 0.07 | 14.68 | 8.06 |
| michaels_cruise | v k25-60 | 10.88 | 1.07 | 11.95 | 13.01 | 6.91 | 0.09 | 7.00 | 8.12 |
| michaels_cruise | v k61+ | 11.14 | 1.18 | 12.32 | 32.32 | 6.40 | 0.05 | 6.44 | 20.94 |
| michaels_cruise | v_present k1-2 | 0.15 | 0.09 | 0.24 | 0.10 | 0.14 | 0.05 | 0.19 | 0.07 |
| michaels_cruise | v_present k3-8 | 23.18 | 1.11 | 24.29 | 17.40 | 15.88 | 0.08 | 15.96 | 11.27 |
| michaels_cruise | v_present k9-24 | 19.29 | 1.11 | 20.41 | 14.39 | 11.97 | 0.07 | 12.04 | 8.06 |
| michaels_cruise | v_present k25-60 | 5.38 | 1.07 | 6.46 | 13.01 | 2.81 | 0.09 | 2.89 | 8.12 |
| michaels_standby | d | 0.82 | 0.22 | 1.03 | 0.28 | 0.41 | 0.08 | 0.49 | 0.12 |
| michaels_standby | v | 10.02 | 1.13 | 11.15 | 23.58 | 5.98 | 0.06 | 6.05 | 15.04 |
| michaels_standby | v_present | 43.87 | 1.19 | 45.06 | 32.32 | 34.35 | 0.05 | 34.40 | 20.94 |
| michaels_standby | u | 1.31 | 0.11 | 1.42 | 3.00 | 0.84 | 0.01 | 0.85 | 2.33 |
| michaels_standby | uj | 1.87 | 0.11 | 1.98 | 2.67 | 0.96 | 0.01 | 0.96 | 1.78 |
| michaels_standby | v k1-2 | 0.10 | 0.09 | 0.19 | 0.10 | 0.08 | 0.05 | 0.14 | 0.07 |
| michaels_standby | v k3-8 | 8.70 | 1.11 | 9.82 | 17.40 | 3.89 | 0.08 | 3.97 | 11.27 |
| michaels_standby | v k9-24 | 11.44 | 1.12 | 12.56 | 14.39 | 6.14 | 0.07 | 6.21 | 8.06 |
| michaels_standby | v k25-60 | 7.74 | 1.08 | 8.81 | 13.01 | 4.93 | 0.09 | 5.02 | 8.12 |
| michaels_standby | v k61+ | 11.27 | 1.19 | 12.45 | 32.32 | 6.84 | 0.05 | 6.89 | 20.94 |
| michaels_standby | v_present k61+ | 43.87 | 1.19 | 45.06 | 32.32 | 34.35 | 0.05 | 34.40 | 20.94 |

