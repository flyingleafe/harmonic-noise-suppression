# Noise model v3 — the section 3.5 checks

`scripts/noise_v3_checks.py` (render and measure only). Every number below is read from the JSON named in its section.

## (b) Held-out line-power spread and intermittency (`heldout/heldout.json`)

Wander estimator of `noise_v3_measure_wander.py` (rig-centred, block 0.5 s, τ at lag 1), window bootstrap median [5 %, 95 %]:

| rig | arm | windows | line tracks | σ_total dB | σ_d dB | σ_v dB | σ_u dB |
|---|---|---:|---:|---|---|---|---|
| dregon | real | 5 | 0 | — (no track) | — (no track) | — (no track) | 3.66 (3.62 [2.52, 4.04]) |
| dregon | v3 | 20 | 21 | 4.74 (4.54 [3.89, 4.86]) | 28.29 (2.04 [0.00, 28.29]) | 0.00 (3.84 [0.00, 4.80]) | 0.81 (0.80 [0.68, 0.90]) |
| dregon | v2 | 20 | 0 | — (no track) | — (no track) | — (no track) | 0.54 (0.54 [0.44, 0.63]) |
| michaels | real | 4 | 5 | 1.46 (1.46 [0.53, 1.50]) | 0.12 (0.12 [0.00, 0.12]) | 1.46 (1.46 [0.53, 1.50]) | 2.97 (2.96 [0.83, 2.97]) |
| michaels | v3 | 16 | 20 | 3.93 (3.72 [3.32, 3.99]) | 0.00 (0.00 [0.00, 0.00]) | 3.93 (3.72 [3.32, 3.99]) | 1.53 (1.47 [1.23, 1.66]) |
| michaels | v2 | 16 | 30 | 0.38 (0.38 [0.28, 0.44]) | 0.26 (0.26 [0.18, 0.30]) | 0.29 (0.28 [0.14, 0.37]) | 0.21 (0.21 [0.15, 0.26]) |

Mid-order lines (k 8-24, every rotor), per block: UNDER the floor = line power below the local floor (prominence < 3.01 dB); PRESENT = window prominence >= 6 dB; VISIBLE = >= 6 dB in >= 20 % of the window's blocks, and the disappearance rate is the share of a visible line's blocks under 6 dB. Window bootstrap median [5 %, 95 %]:

| rig | arm | lines | cells under floor | present-line blocks under floor | lines that appear and disappear | block sd of a present line dB | visible lines | disappearance rate |
|---|---|---:|---|---|---|---|---:|---|
| dregon | real | 340 (0 present) | 81.8 [78.1, 85.8] % | — [—, —] % | 10.9 [6.2, 15.3] % | — [—, —] | 10 | 68.8 [66.4, 72.5] % |
| dregon | v3 | 1360 (90 present) | 74.0 [72.9, 75.1] % | 16.5 [14.0, 19.0] % | 24.8 [22.9, 26.5] % | 3.44 [3.39, 3.58] | 218 | 57.2 [56.1, 58.2] % |
| dregon | v2 | 1360 (0 present) | 92.2 [90.3, 94.1] % | — [—, —] % | 1.8 [0.0, 3.7] % | — [—, —] | 0 | — [—, —] % |
| michaels | real | 272 (26 present) | 80.3 [66.6, 94.5] % | 7.2 [4.1, 17.7] % | 22.1 [3.3, 39.8] % | 2.76 [2.57, 3.91] | 38 | 38.2 [32.9, 51.1] % |
| michaels | v3 | 1088 (135 present) | 74.7 [61.4, 89.7] % | 14.9 [12.2, 26.4] % | 35.6 [13.1, 54.8] % | 3.40 [3.21, 3.87] | 193 | 45.9 [43.0, 55.6] % |
| michaels | v2 | 1088 (108 present) | 79.1 [60.0, 98.1] % | 3.2 [1.6, 5.5] % | 17.0 [0.0, 32.8] % | 1.18 [1.14, 1.21] | 170 | 33.9 [29.9, 38.6] % |

Every measured order by wander order group (the groups `sigma_v_db_by_order` is measured on), the same statistics, window bootstrap median [5 %, 95 %]:

| rig | orders | arm | lines | cells under floor | present-line blocks under floor | lines that appear and disappear | block sd of a present line dB | visible lines | disappearance rate |
|---|---|---|---:|---|---|---|---|---:|---|
| dregon | 1-2 | real | 20 (0 present) | 58.8 [31.2, 87.5] % | — [—, —] % | 50.0 [15.0, 80.0] % | — [—, —] | 8 | 71.9 [68.8, 75.0] % |
| dregon | 1-2 | v3 | 80 (0 present) | 6.6 [5.3, 7.7] % | — [—, —] % | 2.5 [0.0, 5.0] % | — [—, —] | 7 | 75.0 [75.0, 75.0] % |
| dregon | 1-2 | v2 | 80 (0 present) | 11.6 [7.2, 15.9] % | — [—, —] % | 8.8 [0.0, 16.2] % | — [—, —] | 3 | 75.0 [75.0, 75.0] % |
| dregon | 3-8 | real | 120 (0 present) | 87.0 [83.2, 91.6] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 3-8 | v3 | 480 (0 present) | 95.0 [94.3, 95.7] % | — [—, —] % | 0.2 [0.0, 0.6] % | — [—, —] | 0 | — [—, —] % |
| dregon | 3-8 | v2 | 480 (0 present) | 97.8 [95.6, 100.0] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 9-24 | real | 320 (0 present) | 81.2 [77.6, 85.9] % | — [—, —] % | 11.6 [6.6, 15.9] % | — [—, —] | 10 | 68.8 [66.2, 71.9] % |
| dregon | 9-24 | v3 | 1280 (90 present) | 73.4 [72.3, 74.5] % | 16.5 [14.0, 19.0] % | 26.2 [24.1, 28.1] % | 3.44 [3.39, 3.55] | 218 | 57.2 [56.1, 58.2] % |
| dregon | 9-24 | v2 | 1280 (0 present) | 91.9 [90.0, 93.8] % | — [—, —] % | 2.0 [0.0, 5.9] % | — [—, —] | 0 | — [—, —] % |
| dregon | 25-60 | real | 720 (0 present) | 89.2 [84.8, 93.5] % | — [—, —] % | 6.8 [2.8, 12.4] % | — [—, —] | 10 | 72.7 [68.8, 75.0] % |
| dregon | 25-60 | v3 | 2880 (95 present) | 85.3 [84.9, 85.7] % | 17.9 [13.7, 23.3] % | 16.2 [14.9, 17.6] % | 3.75 [3.36, 4.39] | 225 | 56.6 [53.4, 59.5] % |
| dregon | 25-60 | v2 | 2880 (0 present) | 98.6 [98.2, 99.0] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 61+ | real | 753 (0 present) | 98.2 [97.3, 99.1] % | — [—, —] % | 1.6 [0.0, 3.2] % | — [—, —] | 2 | 75.0 [75.0, 75.0] % |
| dregon | 61+ | v3 | 3012 (0 present) | 96.2 [95.9, 96.5] % | — [—, —] % | 1.6 [1.2, 1.8] % | — [—, —] | 5 | 72.5 [68.8, 75.0] % |
| dregon | 61+ | v2 | 3012 (0 present) | 99.5 [99.4, 99.6] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| michaels | 1-2 | real | 10 (8 present) | 4.7 [1.3, 8.5] % | 0.0 [0.0, 0.0] % | 10.0 [0.0, 20.0] % | 2.27 [1.16, 3.36] | 9 | 5.6 [0.0, 10.3] % |
| michaels | 1-2 | v3 | 40 (32 present) | 6.7 [3.9, 9.3] % | 0.0 [0.0, 0.0] % | 12.5 [10.0, 15.0] % | 1.73 [1.51, 1.84] | 36 | 4.7 [3.0, 6.3] % |
| michaels | 1-2 | v2 | 40 (32 present) | 13.3 [8.1, 17.9] % | 0.0 [0.0, 0.0] % | 0.0 [0.0, 0.0] % | 0.87 [0.75, 1.14] | 32 | 0.0 [0.0, 0.0] % |
| michaels | 3-8 | real | 96 (22 present) | 62.2 [44.6, 78.1] % | 6.5 [1.0, 13.1] % | 25.0 [8.3, 46.9] % | 2.80 [2.13, 3.54] | 24 | 23.7 [18.8, 29.5] % |
| michaels | 3-8 | v3 | 384 (128 present) | 53.1 [37.1, 61.6] % | 10.8 [8.2, 13.2] % | 57.6 [56.0, 59.9] % | 4.18 [3.90, 4.37] | 160 | 34.4 [30.3, 40.7] % |
| michaels | 3-8 | v2 | 384 (102 present) | 68.1 [37.5, 99.8] % | 2.2 [0.9, 3.4] % | 16.4 [0.0, 32.8] % | 1.19 [0.86, 1.52] | 120 | 17.6 [13.3, 21.3] % |
| michaels | 9-24 | real | 256 (22 present) | 81.5 [68.5, 95.4] % | 8.2 [4.8, 20.0] % | 21.5 [1.2, 37.9] % | 2.96 [2.59, 3.88] | 34 | 40.4 [34.6, 54.4] % |
| michaels | 9-24 | v3 | 1024 (111 present) | 76.5 [63.4, 91.2] % | 15.4 [13.4, 37.5] % | 33.9 [9.2, 55.1] % | 3.34 [3.19, 4.07] | 164 | 47.4 [44.8, 64.6] % |
| michaels | 9-24 | v2 | 1024 (88 present) | 80.2 [62.4, 98.2] % | 3.8 [1.8, 6.8] % | 17.1 [0.0, 32.5] % | 1.37 [1.14, 1.43] | 146 | 37.2 [33.8, 41.5] % |
| michaels | 25-60 | real | 576 (1 present) | 96.1 [92.6, 99.6] % | 0.0 [0.0, 0.0] % | 3.5 [0.0, 6.9] % | 1.39 [1.39, 1.39] | 2 | 65.6 [65.6, 65.6] % |
| michaels | 25-60 | v3 | 2304 (14 present) | 91.0 [86.3, 96.7] % | 32.1 [31.9, 32.5] % | 14.7 [2.6, 25.6] % | 3.69 [2.94, 4.04] | 51 | 66.7 [66.2, 67.6] % |
| michaels | 25-60 | v2 | 2304 (0 present) | 97.2 [95.0, 99.9] % | — [—, —] % | 0.7 [0.0, 1.3] % | — [—, —] | 0 | — [—, —] % |
| michaels | 61+ | real | 1588 (16 present) | 98.2 [97.9, 99.9] % | 5.1 [2.3, 7.8] % | 1.6 [0.0, 1.9] % | 1.42 [1.40, 1.42] | 23 | 25.8 [25.6, 26.0] % |
| michaels | 61+ | v3 | 6352 (98 present) | 96.4 [96.0, 97.9] % | 12.9 [12.6, 13.2] % | 4.4 [2.5, 5.1] % | 3.25 [3.21, 3.31] | 177 | 46.3 [46.3, 46.3] % |
| michaels | 61+ | v2 | 6352 (109 present) | 96.9 [96.2, 100.0] % | 7.4 [7.0, 7.8] % | 2.2 [0.0, 2.7] % | 2.04 [2.03, 2.19] | 149 | 30.4 [29.6, 31.2] % |

Figure: `heldout/prominence_hist.png`.

## (c) Tonality on rendered audio (`tonality/rendered.json`)

R4 `line_width_db3` (-3 dB width, Hz, 8192-point, resolution 1.95 Hz) at k = 1..8 and the order-6-dB counts on the window prominence ladder; renders: median over the pattern windows x 4 seeds.

| fit | arm | width k=1..8 Hz | orders >= 6 dB / rotor | highest >= 6 dB | prom k=1,2,4,8,16 dB |
|---|---|---|---:|---:|---|
| dregon | real | 10.4, 23.4, —, —, —, —, 30.7, 30.2 | 0.0 | 0 | —, 4.6, 2.9, 1.6, 1.8 |
| dregon | v3 | 9.8, 17.4, —, 20.0, —, —, 12.3, 15.9 | 0.8 | 14 | —, 4.7, 0.4, 1.1, 2.4 |
| dregon | v2 | 10.3, 10.2, —, 9.4, —, —, 6.8, 10.3 | 0.0 | 0 | —, 4.1, 0.5, 0.4, 1.8 |
| michaels_cruise | real | 3.1, 4.2, —, 5.7, 18.9, 8.2, 58.8, 13.8 | 9.0 | 18 | —, 24.8, 12.0, 7.8, 4.3 |
| michaels_cruise | v3 | 2.6, 4.9, —, 7.7, 17.3, 10.3, 16.5, 16.6 | 8.5 | 19 | —, 24.1, 10.7, 7.0, 3.4 |
| michaels_cruise | v2 | —, 4.6, —, 7.6, 18.2, 10.3, —, 16.0 | 7.0 | 15 | —, 22.0, 10.3, 7.1, 2.0 |
| michaels_standby | real | —, 4.0, —, 4.4, 2.0, 4.1, 12.9, 8.5 | 3.1 | 88 | —, —, 1.1, 1.1, 1.3 |
| michaels_standby | v3 | 9.0, 3.8, 2.1, 3.9, 3.6, 7.4, 7.7, 9.8 | 5.2 | 140 | —, —, -1.8, 2.1, 1.4 |
| michaels_standby | v2 | 11.9, 4.0, —, 2.9, 6.2, 3.3, 7.4, 5.2 | 3.9 | 86 | —, —, -0.5, 0.7, 0.8 |

Visible-order counts per rotor at 3 / 6 / 10 dB (rotor mean), per pattern window: `audit` = the tonality audit's estimator on the clip's time-mean periodogram (orders 1..k_max of the v3 fit); `ladder` = the wander estimator's window prominence. Renders: median over the 4 seeds.

| fit | pattern | arm | audit >= 3 / 6 / 10 dB | audit highest >= 6 dB | ladder >= 3 / 6 / 10 dB |
|---|---|---|---|---:|---|
| dregon | dregon_cruise_narrow | real | 7.5 / 1.0 / 0.0 | 1 | 4.8 / 0.0 / 0.0 |
| dregon | dregon_cruise_narrow | v3 | 6.4 / 2.1 / 0.0 | 5 | 5.8 / 0.8 / 0.0 |
| dregon | dregon_cruise_narrow | v2 | 3.0 / 2.0 / 0.0 | 2 | 2.0 / 0.0 / 0.0 |
| dregon | dregon_cruise_wide | real | 5.5 / 1.0 / 0.0 | 1 | 7.5 / 0.0 / 0.0 |
| dregon | dregon_cruise_wide | v3 | 4.4 / 2.2 / 0.2 | 4 | 7.4 / 0.8 / 0.0 |
| dregon | dregon_cruise_wide | v2 | 2.6 / 1.0 / 0.0 | 1 | 1.8 / 0.0 / 0.0 |
| michaels_cruise | michaels_cruise_narrow | real | 8.8 / 3.0 / 1.2 | 9 | 12.2 / 7.8 / 3.0 |
| michaels_cruise | michaels_cruise_narrow | v3 | 8.1 / 4.1 / 2.1 | 12 | 16.9 / 7.9 / 3.6 |
| michaels_cruise | michaels_cruise_narrow | v2 | 7.8 / 1.9 / 1.2 | 6 | 11.2 / 6.5 / 1.8 |
| michaels_cruise | michaels_cruise_wide | real | 11.8 / 6.8 / 3.2 | 11 | 16.5 / 10.2 / 5.0 |
| michaels_cruise | michaels_cruise_wide | v3 | 11.9 / 5.6 / 3.0 | 12 | 19.0 / 8.9 / 3.8 |
| michaels_cruise | michaels_cruise_wide | v2 | 10.5 / 5.5 / 2.8 | 11 | 12.1 / 7.1 / 3.2 |
| michaels_standby | michaels_standby_narrow | real | 7.0 / 3.5 / 2.8 | 86 | 4.8 / 3.2 / 2.2 |
| michaels_standby | michaels_standby_narrow | v3 | 18.0 / 6.5 / 3.2 | 85 | 14.0 / 5.8 / 1.9 |
| michaels_standby | michaels_standby_narrow | v2 | 8.9 / 3.9 / 2.0 | 84 | 6.4 / 4.8 / 1.1 |
| michaels_standby | michaels_standby_wide | real | 4.5 / 1.2 / 0.0 | 65 | 11.0 / 3.0 / 1.0 |
| michaels_standby | michaels_standby_wide | v3 | 9.1 / 3.0 / 1.4 | 77 | 13.1 / 4.9 / 1.4 |
| michaels_standby | michaels_standby_wide | v2 | 4.4 / 2.0 / 1.2 | 44 | 5.0 / 3.0 / 1.0 |

## (c) Tonality on the expectation (`tonality/fits.json`, `noise_v2_tonality_audit.py --fit`)

| pattern | payload | prom k=1,2,4,8,16 dB (mic median) | orders >= 3 / 6 / 10 dB / rotor | highest >= 6 dB | trend crosses floor at k | pedestal median dB |
|---|---|---|---|---:|---:|---:|
| dregon_cruise_narrow | v3 dregon | 9.0, 9.2, 1.8, 4.0, 4.3 | 10.5 / 5.8 / 0.0 | 22 | 89 | 3.1 |
| dregon_cruise_narrow | v2 anchor dregon | 9.1, 9.4, 3.2, 5.4, 8.5 | 23.2 / 9.5 / 2.0 | 37 | 89 | 8.5 |
| dregon_cruise_wide | v3 dregon | 9.4, 7.1, 1.0, 0.9, 0.6 | 8.2 / 3.0 / 0.0 | 28 | 89 | 3.4 |
| dregon_cruise_wide | v2 anchor dregon | 9.5, 7.0, 1.9, 1.4, 3.1 | 17.2 / 6.8 / 0.2 | 38 | 89 | 8.8 |
| michaels_cruise_narrow | v3 michaels | 5.5, 31.8, 9.7, 6.0, 1.7 | 17.8 / 9.5 / 4.5 | 23 | 61 | 2.3 |
| michaels_cruise_narrow | v2 anchor michaels | 2.0, 29.6, 8.1, 5.4, 1.7 | 15.0 / 8.0 / 4.2 | 20 | 73 | 5.7 |
| michaels_cruise_wide | v3 michaels | 2.5, 27.2, 12.3, 5.0, 3.3 | 19.5 / 10.5 / 3.8 | 20 | 63 | 2.6 |
| michaels_cruise_wide | v2 anchor michaels | -2.2, 25.7, 9.1, 4.6, 3.1 | 17.0 / 8.2 / 3.8 | 20 | 70 | 10.6 |
| michaels_standby_narrow | v3 michaels | 14.1, 8.4, 1.3, 2.0, 1.7 | 8.8 / 4.0 / 2.0 | 86 | 101 | 1.4 |
| michaels_standby_narrow | v2 anchor michaels_standby | 9.0, 11.5, 1.4, 0.7, 1.8 | 8.5 / 4.2 / 2.8 | 86 | 127 | 8.3 |
| michaels_standby_wide | v3 michaels | 21.5, 5.5, 1.6, 4.9, 2.2 | 13.0 / 3.8 / 1.2 | 90 | 108 | 0.2 |
| michaels_standby_wide | v2 anchor michaels_standby | 14.8, 7.0, 2.1, 2.2, 2.1 | 11.5 / 3.8 / 1.5 | 85 | 129 | 3.9 |

## (d) Parity gate (`parity/arm_*.json`, `noise_v2_round_score.py --fit`)

HPPNet PIT MAE on renders of the frozen supports (four frozen render seeds, 8 mics) against the legacy PARITY bar and DREGON's frozen STRETCH target; proxy `ltas_abs_db` against its closure-0.7 gate.

| arm | rig | PIT MAE rev/s (mean) | 95 % upper | Michael's ratio | parity bar | parity | stretch bar | stretch | proxy dB | proxy gate dB |
|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|
| `arm_dregon_v3.json` | dregon | 1.686967 | 2.033397 | — | 2.187786 | PASS | 1.897063 | PASS | 2.4707 | 1.9786 |
| `arm_michaels_v3.json` | michaels | 2.329048 | — | 0.7695 | 3.177994 | PASS | — | — | 2.1683 | 1.2197 |

## (e) Fitted latents against the measured wander (`latents/latents.json`)

Measured σ = rms of the fit's prior sd over the family (v: per order group); posterior variance and the MAP's expected sd / lag-1 = the OU prior seen through the measured block noise s² (Kalman smoother); correctly shrunk latents give fitted sd = MAP expected sd, sqrt(fitted ms + posterior var) = measured σ, fitted lag-1 = MAP expected lag-1.

| fit | family | values | fitted sd dB | MAP expected sd dB | sqrt(fitted ms + posterior var) dB | measured σ dB | sqrt(fitted var + s²) dB | fitted lag-1 | MAP expected lag-1 | measured ρ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dregon | d | 320 | 2.32 | 1.69 | 2.50 | 1.94 | 2.58 | 0.61 | 0.56 | 0.46 |
| dregon | v | 28160 | 5.30 | 4.41 | 5.40 | 4.52 | 5.42 | 0.77 | 0.65 | 0.61 |
| dregon | v_present | 32 | 12.79 | 4.59 | 12.83 | 4.72 | 12.84 | 0.87 | 0.59 | 0.56 |
| dregon | u | 80 | 2.09 | 2.45 | 2.12 | 2.48 | 2.12 | 0.64 | 0.73 | 0.72 |
| dregon | uj | 1120 | 7.03 | 3.97 | 7.04 | 3.99 | 7.04 | 0.97 | 0.93 | 0.92 |
| dregon | v k1-2 | 640 | 0.00 | 0.00 | 0.00 | 0.00 | 1.14 | — | — | 0.51 |
| dregon | v k3-8 | 1920 | 0.00 | 0.00 | 0.00 | 0.00 | 1.14 | — | — | 0.51 |
| dregon | v k9-24 | 5120 | 5.16 | 4.07 | 5.26 | 4.21 | 5.28 | 0.71 | 0.67 | 0.64 |
| dregon | v k25-60 | 11520 | 6.21 | 4.87 | 6.30 | 4.98 | 6.31 | 0.79 | 0.68 | 0.65 |
| dregon | v k61+ | 8960 | 4.84 | 4.59 | 4.96 | 4.72 | 4.98 | 0.78 | 0.59 | 0.56 |
| dregon | v_present k61+ | 32 | 12.79 | 4.59 | 12.83 | 4.72 | 12.84 | 0.87 | 0.59 | 0.56 |
| michaels_cruise | d | 512 | 2.37 | 0.89 | 2.46 | 1.12 | 2.63 | 0.92 | 0.87 | 0.70 |
| michaels_cruise | v | 41472 | 5.16 | 4.28 | 5.26 | 4.40 | 5.28 | 0.84 | 0.76 | 0.72 |
| michaels_cruise | v_present | 4240 | 5.37 | 5.12 | 5.46 | 5.22 | 5.48 | 0.76 | 0.77 | 0.75 |
| michaels_cruise | u | 128 | 4.00 | 3.41 | 4.01 | 3.42 | 4.01 | 0.91 | 0.89 | 0.88 |
| michaels_cruise | uj | 1792 | 6.18 | 3.61 | 6.19 | 3.63 | 6.19 | 0.94 | 0.87 | 0.86 |
| michaels_cruise | v k1-2 | 1024 | 1.03 | 0.34 | 1.10 | 0.52 | 1.53 | 0.98 | 0.97 | 0.84 |
| michaels_cruise | v k3-8 | 3072 | 7.01 | 5.79 | 7.09 | 5.89 | 7.10 | 0.84 | 0.80 | 0.78 |
| michaels_cruise | v k9-24 | 8192 | 6.48 | 5.18 | 6.57 | 5.29 | 6.58 | 0.80 | 0.73 | 0.70 |
| michaels_cruise | v k25-60 | 18432 | 4.65 | 4.03 | 4.76 | 4.16 | 4.79 | 0.83 | 0.78 | 0.73 |
| michaels_cruise | v k61+ | 10752 | 4.42 | 3.57 | 4.54 | 3.72 | 4.56 | 0.92 | 0.73 | 0.68 |
| michaels_cruise | v_present k1-2 | 528 | 1.18 | 0.34 | 1.24 | 0.52 | 1.64 | 0.99 | 0.97 | 0.84 |
| michaels_cruise | v_present k3-8 | 1776 | 5.96 | 5.79 | 6.05 | 5.89 | 6.06 | 0.78 | 0.80 | 0.78 |
| michaels_cruise | v_present k9-24 | 1904 | 5.53 | 5.18 | 5.63 | 5.29 | 5.65 | 0.73 | 0.73 | 0.70 |
| michaels_cruise | v_present k25-60 | 32 | 1.60 | 4.03 | 1.90 | 4.16 | 1.96 | 0.12 | 0.78 | 0.73 |
| michaels_standby | d | 96 | 1.19 | 0.89 | 1.38 | 1.12 | 1.65 | 0.63 | 0.87 | 0.70 |
| michaels_standby | v | 12480 | 3.57 | 4.03 | 3.71 | 4.16 | 3.75 | 0.79 | 0.75 | 0.71 |
| michaels_standby | v_present | 328 | 6.34 | 3.57 | 6.42 | 3.72 | 6.44 | 0.85 | 0.73 | 0.68 |
| michaels_standby | u | 24 | 2.12 | 3.41 | 2.15 | 3.42 | 2.15 | 0.89 | 0.89 | 0.88 |
| michaels_standby | uj | 336 | 2.73 | 3.61 | 2.75 | 3.63 | 2.76 | 0.86 | 0.87 | 0.86 |
| michaels_standby | v k1-2 | 192 | 1.09 | 0.33 | 1.16 | 0.52 | 1.57 | 0.98 | 0.97 | 0.84 |
| michaels_standby | v k3-8 | 576 | 5.12 | 5.79 | 5.23 | 5.89 | 5.24 | 0.81 | 0.80 | 0.78 |
| michaels_standby | v k9-24 | 1536 | 4.35 | 5.18 | 4.48 | 5.29 | 4.50 | 0.67 | 0.73 | 0.70 |
| michaels_standby | v k25-60 | 3456 | 3.07 | 4.03 | 3.24 | 4.16 | 3.27 | 0.73 | 0.78 | 0.73 |
| michaels_standby | v k61+ | 6720 | 3.49 | 3.57 | 3.64 | 3.72 | 3.67 | 0.84 | 0.73 | 0.68 |
| michaels_standby | v_present k61+ | 328 | 6.34 | 3.57 | 6.42 | 3.72 | 6.44 | 0.85 | 0.73 | 0.68 |

The same check as second moments (dB²): fitted mean square + posterior variance against the measured σ², and at lag one the fitted lag-1 product + the posterior lag-1 covariance against ρσ² (per block pair).

| fit | family | fitted ms | + posterior var | = lag-0 sum | measured σ² | fitted lag-1 product | + posterior lag-1 cov | = lag-1 sum | measured ρσ² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dregon | d | 5.37 | 0.88 | 6.25 | 3.75 | 3.21 | 0.12 | 3.33 | 1.71 |
| dregon | v | 28.06 | 1.05 | 29.11 | 20.46 | 21.93 | 0.05 | 21.99 | 12.60 |
| dregon | v_present | 163.51 | 1.17 | 164.68 | 22.25 | 147.32 | 0.05 | 147.37 | 12.51 |
| dregon | u | 4.36 | 0.11 | 4.47 | 6.13 | 2.79 | 0.00 | 2.79 | 4.42 |
| dregon | uj | 49.39 | 0.11 | 49.50 | 15.88 | 48.27 | 0.00 | 48.28 | 14.62 |
| dregon | v k1-2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| dregon | v k3-8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| dregon | v k9-24 | 26.58 | 1.12 | 27.70 | 17.69 | 19.02 | 0.08 | 19.09 | 11.25 |
| dregon | v k25-60 | 38.52 | 1.16 | 39.68 | 24.84 | 30.73 | 0.06 | 30.79 | 16.08 |
| dregon | v k61+ | 23.46 | 1.17 | 24.64 | 22.25 | 18.54 | 0.05 | 18.59 | 12.51 |
| dregon | v_present k61+ | 163.51 | 1.17 | 164.68 | 22.25 | 147.32 | 0.05 | 147.37 | 12.51 |
| michaels_cruise | d | 5.61 | 0.46 | 6.07 | 1.26 | 5.19 | 0.19 | 5.38 | 0.89 |
| michaels_cruise | v | 26.64 | 1.06 | 27.70 | 19.38 | 22.63 | 0.09 | 22.72 | 13.99 |
| michaels_cruise | v_present | 28.79 | 1.02 | 29.81 | 27.25 | 21.83 | 0.07 | 21.91 | 20.29 |
| michaels_cruise | u | 15.98 | 0.11 | 16.09 | 11.72 | 14.62 | 0.00 | 14.63 | 10.30 |
| michaels_cruise | uj | 38.18 | 0.11 | 38.29 | 13.15 | 35.98 | 0.00 | 35.98 | 11.34 |
| michaels_cruise | v k1-2 | 1.05 | 0.15 | 1.21 | 0.27 | 1.05 | 0.11 | 1.16 | 0.22 |
| michaels_cruise | v k3-8 | 49.17 | 1.14 | 50.30 | 34.70 | 41.92 | 0.07 | 41.99 | 27.02 |
| michaels_cruise | v k9-24 | 42.00 | 1.15 | 43.15 | 27.95 | 33.75 | 0.07 | 33.82 | 19.69 |
| michaels_cruise | v k25-60 | 21.63 | 1.06 | 22.69 | 17.32 | 18.28 | 0.10 | 18.38 | 12.70 |
| michaels_cruise | v k61+ | 19.53 | 1.05 | 20.58 | 13.82 | 18.15 | 0.10 | 18.25 | 9.42 |
| michaels_cruise | v_present k1-2 | 1.39 | 0.15 | 1.55 | 0.27 | 1.40 | 0.11 | 1.51 | 0.22 |
| michaels_cruise | v_present k3-8 | 35.49 | 1.14 | 36.63 | 34.70 | 27.77 | 0.07 | 27.84 | 27.02 |
| michaels_cruise | v_present k9-24 | 30.58 | 1.15 | 31.73 | 27.95 | 22.33 | 0.07 | 22.39 | 19.69 |
| michaels_cruise | v_present k25-60 | 2.56 | 1.06 | 3.62 | 17.32 | 0.31 | 0.10 | 0.41 | 12.70 |
| michaels_standby | d | 1.42 | 0.48 | 1.90 | 1.26 | 0.79 | 0.20 | 0.98 | 0.89 |
| michaels_standby | v | 12.74 | 1.06 | 13.80 | 17.28 | 10.06 | 0.10 | 10.16 | 12.27 |
| michaels_standby | v_present | 40.22 | 1.06 | 41.28 | 13.82 | 35.09 | 0.10 | 35.19 | 9.42 |
| michaels_standby | u | 4.50 | 0.11 | 4.61 | 11.72 | 3.90 | 0.00 | 3.90 | 10.30 |
| michaels_standby | uj | 7.47 | 0.11 | 7.59 | 13.15 | 6.27 | 0.00 | 6.27 | 11.34 |
| michaels_standby | v k1-2 | 1.18 | 0.16 | 1.34 | 0.27 | 1.18 | 0.12 | 1.30 | 0.22 |
| michaels_standby | v k3-8 | 26.18 | 1.14 | 27.33 | 34.70 | 21.31 | 0.07 | 21.38 | 27.02 |
| michaels_standby | v k9-24 | 18.93 | 1.15 | 20.09 | 27.95 | 12.91 | 0.07 | 12.97 | 19.69 |
| michaels_standby | v k25-60 | 9.42 | 1.07 | 10.49 | 17.32 | 6.75 | 0.10 | 6.85 | 12.70 |
| michaels_standby | v k61+ | 12.20 | 1.06 | 13.26 | 13.82 | 10.40 | 0.10 | 10.51 | 9.42 |
| michaels_standby | v_present k61+ | 40.22 | 1.06 | 41.28 | 13.82 | 35.09 | 0.10 | 35.19 | 9.42 |

