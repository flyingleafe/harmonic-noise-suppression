# Noise model v3 — the section 3.5 checks

`scripts/noise_v3_checks.py` (render and measure only). Every number below is read from the JSON named in its section.

## (a) Prior predictive (`prior/prior.json`)

8 draws of every rig site from each fit's recorded v3 prior, rendered on a fixed 4 s held-out trajectory (8 mics).

| fit | max γ/(0.01k) | share of lines > 5 γ0k | max γ Hz (k ≤ 8) | max σ_ν rad/s | drawn shape sd / σ_B (max) | drawn max abs / σ_B (max) | σ_B dB | render floor shape sd dB (min / med / max) | real floor shape sd dB | render width Hz k=1..8 (max) | real width Hz k=1..8 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| dregon | 11.80 | 9.3 % | 0.594 | 1.55 | 1.48 | 3.01 | 17.60 | 2.76 / 4.67 / 7.67 | 9.38 | 16.2, 19.5, 5.9, 6.0, 5.9, 7.4, 6.8, 10.6 | 12.0, —, —, —, —, —, —, — |
| michaels_cruise | 11.26 | 9.1 % | 0.567 | 1.08 | 1.63 | 3.24 | 3.95 | 2.26 / 3.95 / 5.74 | 4.55 | 6.3, 8.4, 14.7, 7.7, 22.0, 47.6, 32.6, 20.8 | —, 5.1, —, 6.5, —, 12.4, 51.4, 14.9 |
| michaels_standby | 11.66 | 9.4 % | 0.492 | 1.41 | 1.53 | 3.07 | 13.01 | 3.15 / 9.28 / 13.65 | 3.56 | 8.6, 4.0, 4.8, 4.0, 4.3, 8.1, 7.8, 14.1 | —, 3.0, —, 3.3, 1.7, 7.3, 6.0, 4.9 |

## (b) Held-out line-power spread and intermittency (`heldout/heldout.json`)

Wander estimator of `noise_v3_measure_wander.py` (rig-centred, block 0.5 s, τ at lag 1), window bootstrap median [5 %, 95 %]:

| rig | arm | windows | line tracks | σ_total dB | σ_d dB | σ_v dB | σ_u dB |
|---|---|---:|---:|---|---|---|---|
| dregon | real | 5 | 0 | — (no track) | — (no track) | — (no track) | 3.66 (3.62 [2.52, 4.04]) |
| dregon | v3 | 20 | 16 | 3.40 (3.34 [2.83, 3.64]) | 0.00 (0.00 [0.00, 0.00]) | 3.40 (3.34 [2.83, 3.64]) | 0.88 (0.86 [0.73, 0.98]) |
| dregon | v2 | 20 | 0 | — (no track) | — (no track) | — (no track) | 0.54 (0.54 [0.44, 0.63]) |
| michaels | real | 4 | 5 | 1.46 (1.46 [0.53, 1.50]) | 0.12 (0.12 [0.00, 0.12]) | 1.46 (1.46 [0.53, 1.50]) | 2.97 (2.96 [0.83, 2.97]) |
| michaels | v3 | 16 | 22 | 3.16 (3.05 [2.59, 3.47]) | 0.00 (0.00 [0.00, 0.66]) | 3.16 (3.04 [2.59, 3.45]) | 0.83 (0.81 [0.71, 0.90]) |
| michaels | v2 | 16 | 30 | 0.38 (0.38 [0.28, 0.44]) | 0.26 (0.26 [0.18, 0.30]) | 0.29 (0.28 [0.14, 0.37]) | 0.21 (0.21 [0.15, 0.26]) |

Mid-order lines (k 8-24, every rotor), per block: UNDER the floor = line power below the local floor (prominence < 3.01 dB); PRESENT = window prominence >= 6 dB; VISIBLE = >= 6 dB in >= 20 % of the window's blocks, and the disappearance rate is the share of a visible line's blocks under 6 dB. Window bootstrap median [5 %, 95 %]:

| rig | arm | lines | cells under floor | present-line blocks under floor | lines that appear and disappear | block sd of a present line dB | visible lines | disappearance rate |
|---|---|---:|---|---|---|---|---:|---|
| dregon | real | 340 (0 present) | 81.8 [78.1, 85.8] % | — [—, —] % | 10.9 [6.2, 15.3] % | — [—, —] | 10 | 68.8 [66.4, 72.5] % |
| dregon | v3 | 1360 (3 present) | 87.9 [87.2, 88.5] % | 58.3 [50.0, 62.5] % | 7.2 [6.2, 8.2] % | 4.63 [4.63, 4.96] | 37 | 66.2 [63.3, 68.4] % |
| dregon | v2 | 1360 (0 present) | 92.2 [90.3, 94.1] % | — [—, —] % | 1.8 [0.0, 3.7] % | — [—, —] | 0 | — [—, —] % |
| michaels | real | 272 (26 present) | 80.3 [66.6, 94.5] % | 7.2 [4.1, 17.7] % | 22.1 [3.3, 39.8] % | 2.76 [2.57, 3.91] | 38 | 38.2 [32.9, 51.1] % |
| michaels | v3 | 1088 (118 present) | 76.6 [61.4, 93.8] % | 8.6 [7.0, 10.8] % | 27.6 [4.4, 44.1] % | 2.62 [2.52, 2.76] | 181 | 42.9 [38.9, 58.5] % |
| michaels | v2 | 1088 (108 present) | 79.1 [60.0, 98.1] % | 3.2 [1.6, 5.5] % | 17.0 [0.0, 32.8] % | 1.18 [1.14, 1.21] | 170 | 33.9 [29.9, 38.6] % |

Every measured order by wander order group (the groups `sigma_v_db_by_order` is measured on), the same statistics, window bootstrap median [5 %, 95 %]:

| rig | orders | arm | lines | cells under floor | present-line blocks under floor | lines that appear and disappear | block sd of a present line dB | visible lines | disappearance rate |
|---|---|---|---:|---|---|---|---|---:|---|
| dregon | 1-2 | real | 20 (0 present) | 58.8 [31.2, 87.5] % | — [—, —] % | 50.0 [15.0, 80.0] % | — [—, —] | 8 | 71.9 [68.8, 75.0] % |
| dregon | 1-2 | v3 | 80 (0 present) | 8.9 [7.8, 10.0] % | — [—, —] % | 1.2 [0.0, 3.8] % | — [—, —] | 3 | 75.0 [75.0, 75.0] % |
| dregon | 1-2 | v2 | 80 (0 present) | 11.6 [7.2, 15.9] % | — [—, —] % | 8.8 [0.0, 16.2] % | — [—, —] | 3 | 75.0 [75.0, 75.0] % |
| dregon | 3-8 | real | 120 (0 present) | 87.0 [83.2, 91.6] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 3-8 | v3 | 480 (0 present) | 97.4 [97.0, 97.9] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 3-8 | v2 | 480 (0 present) | 97.8 [95.6, 100.0] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 9-24 | real | 320 (0 present) | 81.2 [77.6, 85.9] % | — [—, —] % | 11.6 [6.6, 15.9] % | — [—, —] | 10 | 68.8 [66.2, 71.9] % |
| dregon | 9-24 | v3 | 1280 (3 present) | 87.5 [86.9, 88.1] % | 58.3 [50.0, 62.5] % | 7.7 [6.6, 8.8] % | 4.63 [4.63, 4.96] | 37 | 66.4 [63.3, 68.5] % |
| dregon | 9-24 | v2 | 1280 (0 present) | 91.9 [90.0, 93.8] % | — [—, —] % | 2.0 [0.0, 5.9] % | — [—, —] | 0 | — [—, —] % |
| dregon | 25-60 | real | 720 (0 present) | 89.2 [84.8, 93.5] % | — [—, —] % | 6.8 [2.8, 12.4] % | — [—, —] | 10 | 72.7 [68.8, 75.0] % |
| dregon | 25-60 | v3 | 2880 (61 present) | 92.0 [91.6, 92.3] % | 16.2 [11.9, 21.6] % | 6.0 [5.6, 6.6] % | 3.69 [3.36, 4.21] | 94 | 44.4 [41.1, 47.1] % |
| dregon | 25-60 | v2 | 2880 (0 present) | 98.6 [98.2, 99.0] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| dregon | 61+ | real | 753 (0 present) | 98.2 [97.3, 99.1] % | — [—, —] % | 1.6 [0.0, 3.2] % | — [—, —] | 2 | 75.0 [75.0, 75.0] % |
| dregon | 61+ | v3 | 3012 (0 present) | 94.6 [94.0, 95.2] % | — [—, —] % | 2.1 [1.3, 3.3] % | — [—, —] | 4 | 75.0 [75.0, 75.0] % |
| dregon | 61+ | v2 | 3012 (0 present) | 99.5 [99.4, 99.6] % | — [—, —] % | 0.0 [0.0, 0.0] % | — [—, —] | 0 | — [—, —] % |
| michaels | 1-2 | real | 10 (8 present) | 4.7 [1.3, 8.5] % | 0.0 [0.0, 0.0] % | 10.0 [0.0, 20.0] % | 2.27 [1.16, 3.36] | 9 | 5.6 [0.0, 10.3] % |
| michaels | 1-2 | v3 | 40 (32 present) | 9.7 [5.6, 13.5] % | 0.0 [0.0, 0.0] % | 5.0 [0.0, 10.0] % | 1.38 [1.15, 1.47] | 33 | 1.0 [0.0, 1.9] % |
| michaels | 1-2 | v2 | 40 (32 present) | 13.3 [8.1, 17.9] % | 0.0 [0.0, 0.0] % | 0.0 [0.0, 0.0] % | 0.87 [0.75, 1.14] | 32 | 0.0 [0.0, 0.0] % |
| michaels | 3-8 | real | 96 (22 present) | 62.2 [44.6, 78.1] % | 6.5 [1.0, 13.1] % | 25.0 [8.3, 46.9] % | 2.80 [2.13, 3.54] | 24 | 23.7 [18.8, 29.5] % |
| michaels | 3-8 | v3 | 384 (121 present) | 56.2 [34.9, 67.2] % | 5.7 [3.1, 8.1] % | 36.2 [30.2, 45.3] % | 3.29 [3.15, 3.42] | 145 | 27.5 [23.6, 35.2] % |
| michaels | 3-8 | v2 | 384 (102 present) | 68.1 [37.5, 99.8] % | 2.2 [0.9, 3.4] % | 16.4 [0.0, 32.8] % | 1.19 [0.86, 1.52] | 120 | 17.6 [13.3, 21.3] % |
| michaels | 9-24 | real | 256 (22 present) | 81.5 [68.5, 95.4] % | 8.2 [4.8, 20.0] % | 21.5 [1.2, 37.9] % | 2.96 [2.59, 3.88] | 34 | 40.4 [34.6, 54.4] % |
| michaels | 9-24 | v3 | 1024 (97 present) | 78.5 [63.7, 95.2] % | 9.4 [8.3, 11.0] % | 26.9 [2.3, 44.9] % | 2.59 [2.51, 2.73] | 153 | 44.9 [41.2, 68.8] % |
| michaels | 9-24 | v2 | 1024 (88 present) | 80.2 [62.4, 98.2] % | 3.8 [1.8, 6.8] % | 17.1 [0.0, 32.5] % | 1.37 [1.14, 1.43] | 146 | 37.2 [33.8, 41.5] % |
| michaels | 25-60 | real | 576 (1 present) | 96.1 [92.6, 99.6] % | 0.0 [0.0, 0.0] % | 3.5 [0.0, 6.9] % | 1.39 [1.39, 1.39] | 2 | 65.6 [65.6, 65.6] % |
| michaels | 25-60 | v3 | 2304 (6 present) | 92.3 [88.5, 97.0] % | 25.0 [21.9, 31.2] % | 10.5 [1.7, 17.8] % | 2.58 [2.57, 2.80] | 36 | 68.6 [68.2, 69.2] % |
| michaels | 25-60 | v2 | 2304 (0 present) | 97.2 [95.0, 99.9] % | — [—, —] % | 0.7 [0.0, 1.3] % | — [—, —] | 0 | — [—, —] % |
| michaels | 61+ | real | 1588 (16 present) | 98.2 [97.9, 99.9] % | 5.1 [2.3, 7.8] % | 1.6 [0.0, 1.9] % | 1.42 [1.40, 1.42] | 23 | 25.8 [25.6, 26.0] % |
| michaels | 61+ | v3 | 6352 (159 present) | 95.1 [94.4, 97.8] % | 21.3 [19.7, 23.1] % | 6.9 [3.2, 7.9] % | 4.21 [4.12, 4.26] | 223 | 48.6 [48.6, 48.6] % |
| michaels | 61+ | v2 | 6352 (109 present) | 96.9 [96.2, 100.0] % | 7.4 [7.0, 7.8] % | 2.2 [0.0, 2.7] % | 2.04 [2.03, 2.19] | 149 | 30.4 [29.6, 31.2] % |

Figure: `heldout/prominence_hist.png`.

## (c) Tonality on rendered audio (`tonality/rendered.json`)

R4 `line_width_db3` (-3 dB width, Hz, 8192-point, resolution 1.95 Hz) at k = 1..8 and the order-6-dB counts on the window prominence ladder; renders: median over the pattern windows x 4 seeds.

| fit | arm | width k=1..8 Hz | orders >= 6 dB / rotor | highest >= 6 dB | prom k=1,2,4,8,16 dB |
|---|---|---|---:|---:|---|
| dregon | real | 10.4, 23.4, —, —, —, —, 30.7, 30.2 | 0.0 | 0 | —, 4.6, 2.9, 1.6, 1.8 |
| dregon | v3 | 9.5, 17.6, —, —, —, —, 17.6, 17.6 | 0.5 | 14 | —, 4.7, 0.4, 1.0, 1.2 |
| dregon | v2 | 10.3, 10.2, —, 9.4, —, —, 6.8, 10.3 | 0.0 | 0 | —, 4.1, 0.5, 0.4, 1.8 |
| michaels_cruise | real | 3.1, 4.2, —, 5.7, 18.9, 8.2, 58.8, 13.8 | 9.0 | 18 | —, 24.8, 12.0, 7.8, 4.3 |
| michaels_cruise | v3 | 2.7, 4.9, —, 8.1, 16.6, 10.7, 32.7, 15.4 | 7.9 | 16 | —, 23.3, 10.8, 7.4, 3.0 |
| michaels_cruise | v2 | —, 4.6, —, 7.6, 18.2, 10.3, —, 16.0 | 7.0 | 15 | —, 22.0, 10.3, 7.1, 2.0 |
| michaels_standby | real | —, 4.0, —, 4.4, 2.0, 4.1, 12.9, 8.5 | 3.1 | 88 | —, —, 1.1, 1.1, 1.3 |
| michaels_standby | v3 | 10.7, 3.8, 2.2, 3.2, 3.5, 6.3, 5.9, 10.1 | 4.5 | 149 | —, —, -1.5, 1.5, 1.0 |
| michaels_standby | v2 | 11.9, 4.0, —, 2.9, 6.2, 3.3, 7.4, 5.2 | 3.9 | 86 | —, —, -0.5, 0.7, 0.8 |

Visible-order counts per rotor at 3 / 6 / 10 dB (rotor mean), per pattern window: `audit` = the tonality audit's estimator on the clip's time-mean periodogram (orders 1..k_max of the v3 fit); `ladder` = the wander estimator's window prominence. Renders: median over the 4 seeds.

| fit | pattern | arm | audit >= 3 / 6 / 10 dB | audit highest >= 6 dB | ladder >= 3 / 6 / 10 dB |
|---|---|---|---|---:|---|
| dregon | dregon_cruise_narrow | real | 7.5 / 1.0 / 0.0 | 1 | 4.8 / 0.0 / 0.0 |
| dregon | dregon_cruise_narrow | v3 | 3.8 / 2.0 / 0.0 | 2 | 3.6 / 0.5 / 0.0 |
| dregon | dregon_cruise_narrow | v2 | 3.0 / 2.0 / 0.0 | 2 | 2.0 / 0.0 / 0.0 |
| dregon | dregon_cruise_wide | real | 5.5 / 1.0 / 0.0 | 1 | 7.5 / 0.0 / 0.0 |
| dregon | dregon_cruise_wide | v3 | 3.2 / 2.0 / 0.2 | 2 | 3.8 / 0.4 / 0.0 |
| dregon | dregon_cruise_wide | v2 | 2.6 / 1.0 / 0.0 | 1 | 1.8 / 0.0 / 0.0 |
| michaels_cruise | michaels_cruise_narrow | real | 8.8 / 3.0 / 1.2 | 9 | 12.2 / 7.8 / 3.0 |
| michaels_cruise | michaels_cruise_narrow | v3 | 8.5 / 3.1 / 1.4 | 11 | 15.5 / 7.5 / 2.9 |
| michaels_cruise | michaels_cruise_narrow | v2 | 7.8 / 1.9 / 1.2 | 6 | 11.2 / 6.5 / 1.8 |
| michaels_cruise | michaels_cruise_wide | real | 11.8 / 6.8 / 3.2 | 11 | 16.5 / 10.2 / 5.0 |
| michaels_cruise | michaels_cruise_wide | v3 | 11.4 / 5.9 / 3.4 | 12 | 16.4 / 8.6 / 3.8 |
| michaels_cruise | michaels_cruise_wide | v2 | 10.5 / 5.5 / 2.8 | 11 | 12.1 / 7.1 / 3.2 |
| michaels_standby | michaels_standby_narrow | real | 7.0 / 3.5 / 2.8 | 86 | 4.8 / 3.2 / 2.2 |
| michaels_standby | michaels_standby_narrow | v3 | 18.6 / 5.2 / 2.2 | 88 | 12.1 / 5.5 / 1.8 |
| michaels_standby | michaels_standby_narrow | v2 | 8.9 / 3.9 / 2.0 | 84 | 6.4 / 4.8 / 1.1 |
| michaels_standby | michaels_standby_wide | real | 4.5 / 1.2 / 0.0 | 65 | 11.0 / 3.0 / 1.0 |
| michaels_standby | michaels_standby_wide | v3 | 8.0 / 2.6 / 1.4 | 69 | 11.1 / 4.2 / 1.8 |
| michaels_standby | michaels_standby_wide | v2 | 4.4 / 2.0 / 1.2 | 44 | 5.0 / 3.0 / 1.0 |

## (c) Tonality on the expectation (`tonality/fits.json`, `noise_v2_tonality_audit.py --fit`)

| pattern | payload | prom k=1,2,4,8,16 dB (mic median) | orders >= 3 / 6 / 10 dB / rotor | highest >= 6 dB | trend crosses floor at k | pedestal median dB |
|---|---|---|---|---:|---:|---:|
| dregon_cruise_narrow | v3 dregon | 9.0, 9.3, 1.9, 3.2, 2.8 | 7.0 / 3.2 / 0.0 | 22 | 89 | 2.5 |
| dregon_cruise_narrow | v2 anchor dregon | 9.1, 9.4, 3.2, 5.4, 8.5 | 23.2 / 9.5 / 2.0 | 37 | 89 | 8.5 |
| dregon_cruise_wide | v3 dregon | 9.4, 7.2, 1.1, 0.7, 0.4 | 5.2 / 2.5 / 0.0 | 15 | 89 | 2.8 |
| dregon_cruise_wide | v2 anchor dregon | 9.5, 7.0, 1.9, 1.4, 3.1 | 17.2 / 6.8 / 0.2 | 38 | 89 | 8.8 |
| michaels_cruise_narrow | v3 michaels | 5.0, 31.4, 9.6, 5.7, 1.7 | 18.0 / 9.8 / 4.2 | 23 | 62 | 2.5 |
| michaels_cruise_narrow | v2 anchor michaels | 2.0, 29.6, 8.1, 5.4, 1.7 | 15.0 / 8.0 / 4.2 | 20 | 73 | 5.7 |
| michaels_cruise_wide | v3 michaels | 2.0, 26.9, 12.2, 4.9, 3.4 | 18.8 / 11.0 / 3.8 | 20 | 64 | 2.8 |
| michaels_cruise_wide | v2 anchor michaels | -2.2, 25.7, 9.1, 4.6, 3.1 | 17.0 / 8.2 / 3.8 | 20 | 70 | 10.6 |
| michaels_standby_narrow | v3 michaels | 14.1, 8.4, 1.3, 2.0, 1.7 | 8.8 / 4.0 / 2.0 | 86 | 101 | 1.4 |
| michaels_standby_narrow | v2 anchor michaels_standby | 9.0, 11.5, 1.4, 0.7, 1.8 | 8.5 / 4.2 / 2.8 | 86 | 127 | 8.3 |
| michaels_standby_wide | v3 michaels | 21.5, 5.5, 1.6, 4.9, 2.2 | 13.0 / 3.8 / 1.2 | 90 | 108 | 0.2 |
| michaels_standby_wide | v2 anchor michaels_standby | 14.8, 7.0, 2.1, 2.2, 2.1 | 11.5 / 3.8 / 1.5 | 85 | 129 | 3.9 |

## (d) Parity gate (`parity/arm_*.json`, `noise_v2_round_score.py --fit`)

HPPNet PIT MAE on renders of the frozen supports (four frozen render seeds, 8 mics) against the legacy PARITY bar and DREGON's frozen STRETCH target; proxy `ltas_abs_db` against its closure-0.7 gate.

| arm | rig | PIT MAE rev/s (mean) | 95 % upper | Michael's ratio | parity bar | parity | stretch bar | stretch | proxy dB | proxy gate dB |
|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|
| `arm_dregon_v3.json` | dregon | 1.716079 | 2.062063 | — | 2.187786 | PASS | 1.897063 | PASS | 2.4302 | 1.9786 |
| `arm_michaels_v3.json` | michaels | 1.686297 | — | 0.5571 | 3.177994 | PASS | — | — | 2.3656 | 1.2197 |

## (e) Fitted latents against the measured wander (`latents/latents.json`)

Measured σ = rms of the fit's prior sd over the family (v: per order group); posterior variance and the MAP's expected sd / lag-1 = the OU prior seen through the measured block noise s² (Kalman smoother); correctly shrunk latents give fitted sd = MAP expected sd, sqrt(fitted ms + posterior var) = measured σ, fitted lag-1 = MAP expected lag-1.

| fit | family | values | fitted sd dB | MAP expected sd dB | sqrt(fitted ms + posterior var) dB | measured σ dB | sqrt(fitted var + s²) dB | fitted lag-1 | MAP expected lag-1 | measured ρ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dregon | d | 320 | 1.80 | 0.61 | 1.94 | 0.94 | 2.13 | 0.52 | 0.30 | 0.19 |
| dregon | v | 28160 | 4.40 | 4.81 | 4.52 | 4.92 | 4.55 | 0.64 | 0.55 | 0.50 |
| dregon | v_present | 32 | 8.40 | 7.09 | 8.48 | 7.18 | 8.48 | 0.49 | 0.58 | 0.56 |
| dregon | u | 80 | 2.45 | 2.81 | 2.48 | 2.83 | 2.48 | 0.74 | 0.91 | 0.90 |
| dregon | uj | 1120 | 3.97 | 1.49 | 3.99 | 1.52 | 3.99 | 0.91 | 0.73 | 0.70 |
| dregon | v k1-2 | 640 | 0.00 | 0.00 | 0.00 | 0.00 | 1.14 | — | — | 0.51 |
| dregon | v k3-8 | 1920 | 0.00 | 0.00 | 0.00 | 0.00 | 1.14 | — | — | 0.51 |
| dregon | v k9-24 | 5120 | 4.08 | 2.80 | 4.21 | 2.98 | 4.24 | 0.67 | 0.59 | 0.53 |
| dregon | v k25-60 | 11520 | 4.87 | 3.75 | 4.98 | 3.91 | 5.00 | 0.67 | 0.48 | 0.45 |
| dregon | v k61+ | 8960 | 4.58 | 7.09 | 4.72 | 7.18 | 4.72 | 0.59 | 0.58 | 0.56 |
| dregon | v_present k61+ | 32 | 8.40 | 7.09 | 8.48 | 7.18 | 8.48 | 0.49 | 0.58 | 0.56 |
| michaels_cruise | d | 512 | 1.02 | 0.25 | 1.12 | 0.52 | 1.53 | 0.78 | 0.67 | 0.42 |
| michaels_cruise | v | 41472 | 4.28 | 4.15 | 4.40 | 4.28 | 4.42 | 0.76 | 0.66 | 0.62 |
| michaels_cruise | v_present | 4240 | 4.49 | 3.59 | 4.59 | 3.72 | 4.63 | 0.69 | 0.65 | 0.61 |
| michaels_cruise | u | 128 | 3.41 | 1.70 | 3.42 | 1.73 | 3.43 | 0.89 | 0.80 | 0.78 |
| michaels_cruise | uj | 1792 | 3.61 | 1.60 | 3.63 | 1.63 | 3.63 | 0.88 | 0.69 | 0.67 |
| michaels_cruise | v k1-2 | 1024 | 0.42 | 0.12 | 0.52 | 0.32 | 1.21 | 0.94 | 0.90 | 0.65 |
| michaels_cruise | v k3-8 | 3072 | 5.80 | 4.04 | 5.89 | 4.17 | 5.91 | 0.80 | 0.69 | 0.65 |
| michaels_cruise | v k9-24 | 8192 | 5.18 | 3.64 | 5.29 | 3.79 | 5.30 | 0.73 | 0.60 | 0.56 |
| michaels_cruise | v k25-60 | 18432 | 4.03 | 3.45 | 4.16 | 3.61 | 4.19 | 0.77 | 0.67 | 0.62 |
| michaels_cruise | v k61+ | 10752 | 3.56 | 5.58 | 3.72 | 5.69 | 3.73 | 0.74 | 0.67 | 0.65 |
| michaels_cruise | v_present k1-2 | 528 | 0.40 | 0.12 | 0.50 | 0.32 | 1.20 | 0.94 | 0.90 | 0.65 |
| michaels_cruise | v_present k3-8 | 1776 | 5.00 | 4.04 | 5.11 | 4.17 | 5.13 | 0.72 | 0.69 | 0.65 |
| michaels_cruise | v_present k9-24 | 1904 | 4.63 | 3.64 | 4.75 | 3.79 | 4.76 | 0.66 | 0.60 | 0.56 |
| michaels_cruise | v_present k25-60 | 32 | 1.54 | 3.45 | 1.86 | 3.61 | 1.91 | 0.10 | 0.67 | 0.62 |
| michaels_standby | d | 96 | 0.91 | 0.24 | 1.02 | 0.52 | 1.45 | 0.60 | 0.67 | 0.42 |
| michaels_standby | v | 12480 | 3.65 | 4.74 | 3.80 | 4.86 | 3.82 | 0.71 | 0.67 | 0.63 |
| michaels_standby | v_present | 328 | 7.02 | 5.58 | 7.11 | 5.69 | 7.11 | 0.81 | 0.67 | 0.65 |
| michaels_standby | u | 24 | 1.74 | 1.70 | 1.77 | 1.73 | 1.77 | 0.87 | 0.80 | 0.78 |
| michaels_standby | uj | 336 | 1.96 | 1.60 | 1.99 | 1.63 | 1.99 | 0.83 | 0.69 | 0.67 |
| michaels_standby | v k1-2 | 192 | 0.42 | 0.12 | 0.51 | 0.32 | 1.21 | 0.92 | 0.90 | 0.65 |
| michaels_standby | v k3-8 | 576 | 4.51 | 4.04 | 4.63 | 4.17 | 4.65 | 0.77 | 0.69 | 0.65 |
| michaels_standby | v k9-24 | 1536 | 3.74 | 3.64 | 3.88 | 3.79 | 3.91 | 0.61 | 0.60 | 0.56 |
| michaels_standby | v k25-60 | 3456 | 2.82 | 3.45 | 3.01 | 3.61 | 3.04 | 0.66 | 0.67 | 0.62 |
| michaels_standby | v k61+ | 6720 | 3.97 | 5.58 | 4.11 | 5.69 | 4.13 | 0.74 | 0.67 | 0.65 |
| michaels_standby | v_present k61+ | 328 | 7.02 | 5.58 | 7.11 | 5.69 | 7.11 | 0.81 | 0.67 | 0.65 |

The same check as second moments (dB²): fitted mean square + posterior variance against the measured σ², and at lag one the fitted lag-1 product + the posterior lag-1 covariance against ρσ² (per block pair).

| fit | family | fitted ms | + posterior var | = lag-0 sum | measured σ² | fitted lag-1 product | + posterior lag-1 cov | = lag-1 sum | measured ρσ² |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dregon | d | 3.23 | 0.52 | 3.75 | 0.89 | 1.65 | 0.06 | 1.71 | 0.17 |
| dregon | v | 19.40 | 1.06 | 20.46 | 24.24 | 12.56 | 0.04 | 12.60 | 12.87 |
| dregon | v_present | 70.59 | 1.24 | 71.83 | 51.51 | 34.19 | 0.02 | 34.21 | 29.05 |
| dregon | u | 6.02 | 0.10 | 6.13 | 8.02 | 4.41 | 0.01 | 4.42 | 7.21 |
| dregon | uj | 15.78 | 0.10 | 15.88 | 2.31 | 14.62 | 0.01 | 14.62 | 1.61 |
| dregon | v k1-2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| dregon | v k3-8 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| dregon | v k9-24 | 16.64 | 1.05 | 17.69 | 8.86 | 11.16 | 0.09 | 11.25 | 4.70 |
| dregon | v k25-60 | 23.69 | 1.16 | 24.84 | 15.25 | 16.03 | 0.05 | 16.08 | 6.79 |
| dregon | v k61+ | 21.01 | 1.24 | 22.25 | 51.51 | 12.48 | 0.02 | 12.51 | 29.05 |
| dregon | v_present k61+ | 70.59 | 1.24 | 71.83 | 51.51 | 34.19 | 0.02 | 34.21 | 29.05 |
| michaels_cruise | d | 1.05 | 0.21 | 1.26 | 0.28 | 0.81 | 0.07 | 0.89 | 0.12 |
| michaels_cruise | v | 18.29 | 1.09 | 19.38 | 18.29 | 13.91 | 0.07 | 13.99 | 11.47 |
| michaels_cruise | v_present | 20.12 | 0.98 | 21.11 | 13.86 | 13.91 | 0.07 | 13.98 | 8.41 |
| michaels_cruise | u | 11.61 | 0.10 | 11.72 | 3.00 | 10.29 | 0.01 | 10.30 | 2.33 |
| michaels_cruise | uj | 13.04 | 0.11 | 13.15 | 2.67 | 11.33 | 0.01 | 11.34 | 1.78 |
| michaels_cruise | v k1-2 | 0.18 | 0.09 | 0.27 | 0.10 | 0.17 | 0.05 | 0.22 | 0.07 |
| michaels_cruise | v k3-8 | 33.59 | 1.11 | 34.70 | 17.40 | 26.94 | 0.08 | 27.02 | 11.27 |
| michaels_cruise | v k9-24 | 26.84 | 1.11 | 27.95 | 14.39 | 19.62 | 0.07 | 19.69 | 8.06 |
| michaels_cruise | v k25-60 | 16.24 | 1.07 | 17.32 | 13.01 | 12.62 | 0.09 | 12.70 | 8.12 |
| michaels_cruise | v k61+ | 12.64 | 1.18 | 13.82 | 32.32 | 9.37 | 0.05 | 9.42 | 20.94 |
| michaels_cruise | v_present k1-2 | 0.16 | 0.09 | 0.25 | 0.10 | 0.15 | 0.05 | 0.20 | 0.07 |
| michaels_cruise | v_present k3-8 | 25.00 | 1.11 | 26.11 | 17.40 | 17.98 | 0.08 | 18.05 | 11.27 |
| michaels_cruise | v_present k9-24 | 21.40 | 1.11 | 22.52 | 14.39 | 14.16 | 0.07 | 14.23 | 8.06 |
| michaels_cruise | v_present k25-60 | 2.37 | 1.07 | 3.45 | 13.01 | 0.25 | 0.09 | 0.34 | 8.12 |
| michaels_standby | d | 0.82 | 0.22 | 1.04 | 0.28 | 0.43 | 0.08 | 0.50 | 0.12 |
| michaels_standby | v | 13.34 | 1.13 | 14.46 | 23.58 | 9.50 | 0.06 | 9.56 | 15.04 |
| michaels_standby | v_present | 49.30 | 1.19 | 50.48 | 32.32 | 40.83 | 0.05 | 40.88 | 20.94 |
| michaels_standby | u | 3.02 | 0.11 | 3.12 | 3.00 | 2.61 | 0.01 | 2.62 | 2.33 |
| michaels_standby | uj | 3.85 | 0.11 | 3.96 | 2.67 | 3.13 | 0.01 | 3.14 | 1.78 |
| michaels_standby | v k1-2 | 0.17 | 0.09 | 0.26 | 0.10 | 0.16 | 0.05 | 0.22 | 0.07 |
| michaels_standby | v k3-8 | 20.37 | 1.11 | 21.48 | 17.40 | 15.81 | 0.08 | 15.89 | 11.27 |
| michaels_standby | v k9-24 | 13.96 | 1.12 | 15.08 | 14.39 | 8.71 | 0.07 | 8.78 | 8.06 |
| michaels_standby | v k25-60 | 7.96 | 1.08 | 9.04 | 13.01 | 5.11 | 0.09 | 5.20 | 8.12 |
| michaels_standby | v k61+ | 15.73 | 1.19 | 16.92 | 32.32 | 11.66 | 0.05 | 11.71 | 20.94 |
| michaels_standby | v_present k61+ | 49.30 | 1.19 | 50.48 | 32.32 | 40.83 | 0.05 | 40.88 | 20.94 |

