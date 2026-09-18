# Noise model v2 — round 2 fits

22 fit JSON(s) under `results/noise_v2/rounds/round2/fits`. Every number below is read from a `noise-v2-fit/1` payload in that directory; nothing is recomputed here.

The tables down to `Prior edges and non-convergence` are the generator's output (`scripts/noise_v2_fit.py findings`, re-run in the main checkout over this directory; byte-identical to the copy the fit job wrote, except for this header and this note). The sections after them are appended: they read the same payloads and the `peak_scale_db` conversion documented in `round1/bench_diag/findings.md`, and every input is named where it is used.

## Per-support parameters

| support | mode | R | k_max | sigma_nu | lam | sigma_eps even/odd | lam_eps even/odd | carrier rev/s | floor dB | whittle nats | comb | floor | cells | conv | wall s |
|---|---|--:|--:|--:|--:|---|---|---|--:|--:|--:|--:|--:|:-:|--:|
| `bench_dregon_Motor1_50` | bench | 1 | 130 | 5.1243 | 5203.232 | 0.6896 / 0.1818 | 0.723 / 1025.083 | 49.015 | -59.88 | -2.12226e+07 | -2.04139e+07 | -808756 | 1809712 | y | 1478 |
| `bench_dregon_Motor1_60` | bench | 1 | 130 | 6.3814 | 3520.263 | 0.0000 / 0.0102 | 0.000 / 0.035 | 58.766 | -53.31 | -1.62063e+07 | -1.56024e+07 | -603860 | 1474496 | N | 1888 |
| `bench_dregon_Motor1_70` | bench | 1 | 117 | 4.9929 | 571.660 | 0.0008 / 0.0179 | 0.015 / 0.119 | 68.314 | -50.20 | -1.88559e+07 | -1.81717e+07 | -684262 | 1817744 | N | 2026 |
| `bench_dregon_Motor1_80` | bench | 1 | 102 | 1.1556 | 4.079 | 0.1074 / 0.0815 | 111.336 / 0.934 | 78.101 | -49.16 | -6.69071e+06 | -6.45115e+06 | -239566 | 671688 | y | 926 |
| `bench_dregon_Motor1_90` | bench | 1 | 90 | 2.1089 | 24.245 | 0.0904 / 0.1520 | 256.893 / 0.536 | 88.166 | -48.61 | -7.0296e+06 | -6.78917e+06 | -240431 | 733640 | N | 626 |
| `bench_dregon_Motor2_50` | bench | 1 | 130 | 0.4482 | 73.869 | 0.1757 / 0.2149 | 6.958 / 282.377 | 48.435 | -58.44 | -8.18808e+06 | -7.91199e+06 | -276085 | 703808 | N | 562 |
| `bench_dregon_Motor2_60` | bench | 1 | 130 | 2.1238 | 654.844 | 0.2855 / 0.6765 | 0.384 / 0.129 | 58.144 | -54.06 | -7.59132e+06 | -7.32024e+06 | -271081 | 687752 | N | 528 |
| `bench_dregon_Motor2_70` | bench | 1 | 118 | 4.4216 | 587.099 | 0.0358 / 0.1031 | 0.075 / 0.041 | 67.552 | -51.02 | -6.92499e+06 | -6.70673e+06 | -218262 | 676856 | N | 470 |
| `bench_dregon_Motor2_80` | bench | 1 | 103 | 6.5147 | 1217.042 | 0.1492 / 0.1252 | 0.102 / 0.213 | 77.210 | -49.46 | -7.08173e+06 | -6.85638e+06 | -225349 | 712416 | N | 354 |
| `bench_dregon_Motor2_90` | bench | 1 | 92 | 1.5611 | 0.267 | 0.0806 / 0.2464 | 15.016 / 324.713 | 86.695 | -50.30 | -6.31127e+06 | -6.09835e+06 | -212921 | 668824 | N | 367 |
| `bench_dregon_Motor3_50` | bench | 1 | 130 | 102.2640 | 378739.048 | 0.0772 / 0.0530 | 0.282 / 0.206 | 49.135 | -54.22 | -7.67778e+06 | -7.41211e+06 | -265671 | 678576 | N | 309 |
| `bench_dregon_Motor3_60` | bench | 1 | 130 | 1.3772 | 28.474 | 0.0349 / 0.0996 | 0.224 / 0.615 | 58.949 | -51.16 | -7.54989e+06 | -7.33768e+06 | -212210 | 721592 | y | 488 |
| `bench_dregon_Motor3_70` | bench | 1 | 116 | 143.1799 | 667449.128 | 4.3998 / 0.0630 | 0.008 / 0.192 | 68.634 | -53.95 | -6.87171e+06 | -6.63105e+06 | -240659 | 681440 | N | 330 |
| `bench_dregon_Motor3_80` | bench | 1 | 101 | 1.4588 | 12.430 | 0.0838 / 0.0000 | 2980.511 / 0.000 | 78.814 | -49.71 | -6.84373e+06 | -6.60182e+06 | -241916 | 710120 | N | 521 |
| `bench_dregon_Motor3_90` | bench | 1 | 90 | 1.7157 | 2.831 | 0.1336 / 0.0681 | 174.340 / 51.621 | 88.244 | -53.42 | -6.28078e+06 | -6.07592e+06 | -204859 | 676856 | N | 501 |
| `bench_dregon_Motor4_50` | bench | 1 | 130 | 0.7860 | 66.955 | 0.0963 / 0.1368 | 3349.647 / 208.127 | 49.806 | -61.61 | -7.82851e+06 | -7.53823e+06 | -290281 | 690616 | N | 361 |
| `bench_dregon_Motor4_60` | bench | 1 | 130 | 18.3330 | 16285.743 | 0.0015 / 2.9872 | 0.000 / 0.074 | 59.741 | -51.92 | -7.16659e+06 | -6.92743e+06 | -239161 | 678576 | N | 669 |
| `bench_dregon_Motor4_70` | bench | 1 | 115 | 1.1517 | 0.075 | 0.2368 / 0.1622 | 1.475 / 495.272 | 69.333 | -53.17 | -6.88247e+06 | -6.6323e+06 | -250174 | 687752 | N | 419 |
| `bench_dregon_Motor4_80` | bench | 1 | 100 | 1.5888 | 1.469 | 0.1219 / 0.0809 | 245.034 / 0.442 | 79.375 | -52.03 | -6.36352e+06 | -6.133e+06 | -230520 | 671688 | N | 731 |
| `bench_dregon_Motor4_90` | bench | 1 | 89 | 1.5807 | 1.050 | 0.1068 / 0.0574 | 208.744 / 30.986 | 89.172 | -52.27 | -6.23061e+06 | -6.02387e+06 | -206746 | 667672 | N | 542 |
| `bench_dregon_allMotors_70` | bench | 4 | 114 | 0.9960 | 109.752 | 0.3999 / 0.2406 | 10.824 / 11.171 | 64.636, 67.660, 68.736, 69.565 | -37.92 | -1.60442e+07 | -1.57697e+07 | -274497 | 1973192 | N | 1432 |
| `dregon_room2_floor` | flight_floor_only | 4 | 88 | 7.7676 | 359.826 | 0.0735 / 0.0659 | 0.060 / 0.823 | label | -36.69 | -1.76977e+07 | -1.751e+07 | -187667 | 2056320 | N | 2480 |

## Population median [IQR]

| set | n | sigma_nu | lam | sigma_eps even | sigma_eps odd | lam_eps even | lam_eps odd |
|---|--:|---|---|---|---|---|---|
| DREGON single-motor bench (all throttles) | 20 | 1.912 [1.438, 5.439] | 70.41 [3.767, 1793] | 0.1016 [0.06686, 0.1558] | 0.1014 [0.0616, 0.1671] | 1.099 [0.09552, 182.9] | 0.489 [0.1267, 90.75] |
| flight | 1 | 7.768 [7.768, 7.768] | 359.8 [359.8, 359.8] | 0.07352 [0.07352, 0.07352] | 0.06585 [0.06585, 0.06585] | 0.06017 [0.06017, 0.06017] | 0.823 [0.823, 0.823] |

## Four-motor validation

`motor_allMotors_70` against the four `Motor{1-4}_70` single-motor fits: the ratio of the four-rotor value to the geometric mean of the per-rotor ones and the per-rotor spread, as read off the fits. This is a raw parameter comparison only — what the four-motor support does or does not validate is measured in its own workstream, not here.

| parameter | four-motor | per-rotor geo-mean | ratio | per-rotor min/max |
|---|--:|--:|--:|---|
| `sigma_nu` | 0.996 | 7.768 | 0.128 | 1.152 / 143.2 |
| `lam` | 109.8 | 359.8 | 0.305 | 0.07483 / 6.674e+05 |
| `sigma_eps_even` | 0.3999 | 0.07352 | 5.439 | 0.0007834 / 4.4 |
| `sigma_eps_odd` | 0.2406 | 0.06585 | 3.653 | 0.01785 / 0.1622 |
| `lam_eps_even` | 10.82 | 0.06017 | 179.890 | 0.007873 / 1.475 |
| `lam_eps_odd` | 11.17 | 0.823 | 13.572 | 0.04073 / 495.3 |

## Multi-start restarts

Each support was fitted from several starts: start 0 from the data-driven initialisation, the others from a log-normal perturbation of the dynamics init (`OptimSpec.init_jitter`; a bare seed change is a no-op because a bench fit is deterministic). The REPORTED fit above is the start with the lowest polished objective. `best-median` and `best-worst` are that objective's advantage over the median and the worst start, per observed cell, against the same 0.0001 nats/cell tolerance the convergence test uses; `starts agree` is yes only when even the worst start is inside it. The dynamics columns are min / median / max over the starts.

| support | starts | best nats/cell | best-median | best-worst | starts agree | L-BFGS conv | sigma_nu | lam | sigma_eps even | sigma_eps odd | lam_eps even | lam_eps odd |
|---|--:|--:|--:|--:|:-:|:-:|---|---|---|---|---|---|
| `bench_dregon_Motor1_50` | 4 | -11.7271 | 0.00734 | 0.00735 | N | y | 5.12 / 49.6 / 96.7 | 5.2e+03 / 1.78e+05 / 6.82e+05 | 0.0828 / 0.0831 / 0.69 | 0.0204 / 0.0341 / 0.182 | 0.723 / 180 / 182 | 0.217 / 0.362 / 1.03e+03 |
| `bench_dregon_Motor1_60` | 4 | -10.9911 | 0.000425 | 0.00122 | N | N | 3.46 / 3.72 / 6.38 | 1.02e+03 / 1.21e+03 / 3.52e+03 | 9.06e-13 / 2.86e-09 / 2.72e-07 | 2.72e-08 / 0.0204 / 0.0534 | 2.1e-08 / 3.44e-07 / 4.35e-05 | 1.02e-05 / 0.111 / 0.305 |
| `bench_dregon_Motor1_70` | 4 | -10.3733 | 0.000321 | 0.00065 | N | N | 4.83 / 4.92 / 5.05 | 534 / 565 / 605 | 3.29e-05 / 0.000288 / 0.000783 | 0.0079 / 0.0177 / 0.0222 | 0.00202 / 0.0167 / 0.0672 | 0.0865 / 0.128 / 0.326 |
| `bench_dregon_Motor1_80` | 4 | -9.9610 | 1.79e-05 | 4.75e-05 | y | y | 1.15 / 1.16 / 1.16 | 3.93 / 4.12 / 4.16 | 0.107 / 0.107 / 0.108 | 0.0815 / 0.0976 / 0.164 | 110 / 111 / 111 | 0.227 / 0.45 / 0.934 |
| `bench_dregon_Motor1_90` | 4 | -9.5818 | 0.000537 | 0.00122 | N | N | 2.07 / 2.1 / 2.13 | 19.9 / 22.7 / 24.4 | 0.0899 / 0.0912 / 0.0927 | 0.0198 / 0.0285 / 0.152 | 246 / 255 / 266 | 0.163 / 0.518 / 0.855 |
| `bench_dregon_Motor2_50` | 4 | -11.6340 | 1.48e-05 | 8.54e-05 | y | N | 0.448 / 0.451 / 0.454 | 66.9 / 71.3 / 73.9 | 0.149 / 0.155 / 0.176 | 0.215 / 0.215 / 0.216 | 6.96 / 9.2 / 10.1 | 281 / 284 / 287 |
| `bench_dregon_Motor2_60` | 4 | -11.0379 | 0.00359 | 0.00527 | N | N | 1.55 / 1.81 / 2.12 | 247 / 367 / 655 | 0.0214 / 0.123 / 0.285 | 0.0681 / 0.107 / 0.676 | 0.0267 / 0.146 / 0.384 | 0.129 / 0.172 / 0.201 |
| `bench_dregon_Motor2_70` | 4 | -10.2311 | 0.00114 | 0.0194 | N | N | 1.7 / 4.04 / 4.92 | 31.5 / 494 / 780 | 0.00964 / 0.0343 / 0.0863 | 0.0342 / 0.106 / 1.38 | 0.0542 / 0.0953 / 0.119 | 0.0239 / 0.0477 / 36.9 |
| `bench_dregon_Motor2_80` | 4 | -9.9404 | 4.02e-05 | 0.000195 | N | N | 6.22 / 6.5 / 7.16 | 1.09e+03 / 1.21e+03 / 1.5e+03 | 0.022 / 0.153 / 0.547 | 0.0488 / 0.109 / 0.125 | 0.0735 / 0.101 / 0.133 | 0.0723 / 0.184 / 0.353 |
| `bench_dregon_Motor2_90` | 4 | -9.4364 | 0.00297 | 0.00562 | N | N | 1.53 / 1.55 / 1.56 | 0.0871 / 0.255 / 0.269 | 0.0694 / 0.0784 / 0.0916 | 0.234 / 0.239 / 0.246 | 10.9 / 16.8 / 21.5 | 325 / 431 / 526 |
| `bench_dregon_Motor3_50` | 4 | -11.3145 | 4e-05 | 7.15e-05 | y | N | 102 / 108 / 113 | 3.79e+05 / 4.28e+05 / 4.63e+05 | 0.0482 / 0.0772 / 0.0865 | 0.0344 / 0.0448 / 0.053 | 0.219 / 0.319 / 0.634 | 0.206 / 0.284 / 0.495 |
| `bench_dregon_Motor3_60` | 4 | -10.4628 | 4.78e-05 | 0.000152 | N | y | 1.38 / 1.39 / 1.4 | 28.5 / 29.3 / 30.1 | 0.00108 / 0.0197 / 0.0349 | 0.0176 / 0.081 / 0.0996 | 0.00954 / 0.12 / 0.422 | 0.0769 / 0.432 / 0.986 |
| `bench_dregon_Motor3_70` | 4 | -10.0841 | 7.85e-05 | 0.000313 | N | N | 75.8 / 112 / 143 | 1.86e+05 / 4.02e+05 / 6.67e+05 | 0.492 / 4.08 / 5.78 | 0.0388 / 0.0503 / 0.063 | 0.00457 / 0.009 / 0.16 | 0.131 / 0.187 / 0.265 |
| `bench_dregon_Motor3_80` | 4 | -9.6374 | 0.00246 | 0.0568 | N | N | 1.17 / 1.44 / 1.46 | 10.4 / 12.1 / 13.4 | 0.0838 / 0.086 / 3.15 | 2.14e-06 / 0.00864 / 0.0267 | 0.0728 / 1.05e+03 / 2.98e+03 | 0.000392 / 0.122 / 0.359 |
| `bench_dregon_Motor3_90` | 4 | -9.2793 | 0.000111 | 0.0201 | N | N | 1.72 / 1.72 / 156 | 2.7 / 2.87 / 5.25e+05 | 0.133 / 0.134 / 10.1 | 0.0194 / 0.069 / 0.0707 | 0.00371 / 172 / 175 | 0.0326 / 49.2 / 51.6 |
| `bench_dregon_Motor4_50` | 4 | -11.3355 | 0.00167 | 0.00213 | N | N | 0.637 / 0.794 / 0.843 | 31.4 / 69 / 77.1 | 0.0837 / 0.284 / 0.719 | 0.137 / 0.192 / 0.198 | 0.197 / 17.5 / 3.35e+03 | 208 / 535 / 578 |
| `bench_dregon_Motor4_60` | 4 | -10.5612 | 0.0136 | 0.0228 | N | N | 3.51 / 8.73 / 18.3 | 861 / 4.17e+03 / 1.63e+04 | 0.0015 / 0.0116 / 0.0262 | 0.0705 / 0.451 / 2.99 | 0.000283 / 0.00152 / 0.011 | 0.0383 / 0.0904 / 0.305 |
| `bench_dregon_Motor4_70` | 4 | -10.0072 | 0.0175 | 0.0196 | N | N | 1.15 / 6.03 / 6.57 | 0.0748 / 1e+03 / 1.09e+03 | 0.218 / 1.24 / 7.44 | 0.00309 / 0.0217 / 0.162 | 0.00973 / 0.307 / 1.48 | 0.0593 / 0.0989 / 495 |
| `bench_dregon_Motor4_80` | 4 | -9.4739 | 0.000224 | 0.000288 | N | N | 1.59 / 1.59 / 1.59 | 1.45 / 1.49 / 1.52 | 0.122 / 0.122 / 0.122 | 0.0484 / 0.0578 / 0.0809 | 239 / 240 / 245 | 0.442 / 3.11 / 8.31 |
| `bench_dregon_Motor4_90` | 4 | -9.3318 | 0.000103 | 0.000154 | N | N | 1.58 / 1.58 / 1.58 | 1.05 / 1.12 / 1.22 | 0.106 / 0.107 / 0.107 | 0.0559 / 0.0568 / 0.0574 | 208 / 209 / 210 | 31 / 31.5 / 31.9 |
| `bench_dregon_allMotors_70` | 4 | -8.1311 | 0.000239 | 0.000355 | N | N | 0.996 / 1.23 / 1.44 | 110 / 169 / 214 | 0.4 / 0.429 / 0.432 | 0.241 / 0.245 / 0.249 | 7.77 / 8.41 / 10.8 | 11.2 / 13.1 / 13.3 |

3 of 21 supports have every start inside the tolerance. The widest disagreement is `bench_dregon_Motor3_80` at 0.0568 nats/cell, 568 times the tolerance. `lam` alone spans a factor of 1.95 (median over supports) and up to 1.94e+05 across the starts of one support: on this evidence the R1 bench MAP problem is multi-modal, and a number computed from a single start is a draw from that multiplicity rather than an estimate.

## Fitted `lam_eps` against the measured OU verdict

The long-lag study (`results/noise_v2/decoherence_long`) found a per-order ceiling only at k = 4 and k = 8, with tau_c 2.15 s and 1.12 s, i.e. lam_eps between 0.465 and 0.893 /s; k = 2 was still growing at 5 s and k >= 16 sat over the estimator's ceiling, so the verdict is order-conditional and the model is NOT being asked to reproduce a universal OU. The question here is only whether the fitted rates land anywhere near the orders where a ceiling was measured.

| set | n | lam_eps even median [IQR] | lam_eps odd median [IQR] | within a factor of 3 of the measured window |
|---|--:|---|---|--:|
| DREGON single-motor bench (all throttles) | 20 | 1.099 [0.09552, 182.9] | 0.489 [0.1267, 90.75] | 12 / 40 |
| flight | 1 | 0.06017 [0.06017, 0.06017] | 0.823 [0.823, 0.823] | 1 / 2 |

## Prior edges and non-convergence

- `bench_dregon_Motor1_50`: lam = 5203.23 > 30 (label-chain regime, see the explainer's third consequence of the shaft prior)
- `bench_dregon_Motor1_50`: sigma_nu = 5.1243 outside the prior's central 95 %
- `bench_dregon_Motor1_60`: lam = 3520.26 > 30 (label-chain regime, see the explainer's third consequence of the shaft prior)
- `bench_dregon_Motor1_60`: sigma_nu = 6.3814 outside the prior's central 95 %
- `bench_dregon_Motor1_60`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor1_70`: lam = 571.66 > 30 (label-chain regime, see the explainer's third consequence of the shaft prior)
- `bench_dregon_Motor1_70`: sigma_nu = 4.9929 outside the prior's central 95 %
- `bench_dregon_Motor1_70`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor1_90`: sigma_nu = 2.1089 outside the prior's central 95 %
- `bench_dregon_Motor1_90`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor2_50`: lam = 73.87 > 30 (label-chain regime, see the explainer's third consequence of the shaft prior)
- `bench_dregon_Motor2_50`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor2_60`: lam = 654.84 > 30 (label-chain regime, see the explainer's third consequence of the shaft prior)
- `bench_dregon_Motor2_60`: sigma_nu = 2.1238 outside the prior's central 95 %
- `bench_dregon_Motor2_60`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor2_70`: lam = 587.10 > 30 (label-chain regime, see the explainer's third consequence of the shaft prior)
- `bench_dregon_Motor2_70`: sigma_nu = 4.4216 outside the prior's central 95 %
- `bench_dregon_Motor2_70`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor2_80`: lam = 1217.04 > 30 (label-chain regime, see the explainer's third consequence of the shaft prior)
- `bench_dregon_Motor2_80`: sigma_nu = 6.5147 outside the prior's central 95 %
- `bench_dregon_Motor2_80`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor2_90`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor3_50`: lam = 378739.05 > 30 (label-chain regime, see the explainer's third consequence of the shaft prior)
- `bench_dregon_Motor3_50`: sigma_nu = 102.2640 outside the prior's central 95 %
- `bench_dregon_Motor3_50`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor3_70`: lam = 667449.13 > 30 (label-chain regime, see the explainer's third consequence of the shaft prior)
- `bench_dregon_Motor3_70`: sigma_nu = 143.1799 outside the prior's central 95 %
- `bench_dregon_Motor3_70`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor3_80`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor3_90`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor4_50`: lam = 66.96 > 30 (label-chain regime, see the explainer's third consequence of the shaft prior)
- `bench_dregon_Motor4_50`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor4_60`: lam = 16285.74 > 30 (label-chain regime, see the explainer's third consequence of the shaft prior)
- `bench_dregon_Motor4_60`: sigma_nu = 18.3330 outside the prior's central 95 %
- `bench_dregon_Motor4_60`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor4_70`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor4_80`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor4_90`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_allMotors_70`: lam = 109.75 > 30 (label-chain regime, see the explainer's third consequence of the shaft prior)
- `bench_dregon_allMotors_70`: NOT converged (L-BFGS restart still improving)
- `dregon_room2_floor`: lam = 359.83 > 30 (label-chain regime, see the explainer's third consequence of the shaft prior)
- `dregon_room2_floor`: sigma_nu = 7.7676 outside the prior's central 95 %
- `dregon_room2_floor`: NOT converged (L-BFGS restart still improving)

## In-band comb evidence, per support

`profile_db` is a per-order line VARIANCE, not a periodogram level (`results/noise_v2/rounds/round1/bench_diag/findings.md`, section 0). The model's own unit-line response puts the line's PEAK BIN above `profile_db` by `peak_scale_db(k)`, documented there at the fitted dynamics of `Motor1_70` as +52.05 (k = 1), +50.96 (k = 2), +49.53 (k = 5), +46.88 (k = 10), +43.93 (k = 20), +40.93 (k = 40), +37.93 (k = 80) dB. The k = 4 and k = 8 values used below, +49.88 and +47.73 dB, are log-k interpolations of that row. The response is support-specific — the same table gives +49.85 / +47.37 / +41.99 / +36.45 dB at `Motor1_80`'s fitted dynamics — so `peak above floor` below is that documented conversion applied at ONE anchor: good to a few dB, not to 0.1 dB. It is not recomputed per fit, because `peak_scale_db` is a BenchDiag probe and the fit payload does not carry it.

- `profile_db` k = j: mean over rotors of `params.profile.profile_db[:, j-1]`.
- `floor`: `diagnostics.floor_level_db` = `params.floor.floor_mean_db + mean(params.floor.mic_floor_db)`, the mic-averaged convention.
- `peak above floor` k = j: `profile_db(j) + mean(params.profile.mic_line_gain_db) + peak_scale_db(j) - floor`.
- `comb` / `floor` nats/cell: `objective.per_band` over `objective.n_cells_per_band`, i.e. the >= 300 Hz comb band and the < 300 Hz floor band OF THE SAME FIT. They are not a comb-versus-no-comb contrast: R2 ran no per-support floor-only counterfactual and no fit payload records a comb objective gain, so identification rests on the conversion above.

| support | pass | in-window margin dB | orders | floor dB | profile_db k=1/2/4/8 | peak above floor k=1/2/4/8 dB | comb nats/cell | floor nats/cell | comb identified |
|---|:-:|--:|--:|--:|---|---|--:|--:|:-:|
| `bench_dregon_Motor1_50` | y | 8.32 | 130 | -59.88 | -70.1 / -69.2 / -75.6 / -72.8 | +43.5 / +43.2 / +35.7 / +36.4 | -11.722 | -11.868 | y |
| `bench_dregon_Motor1_60` | y | 10.85 | 130 | -53.31 | -54.2 / -53.7 / -63.5 / -64.5 | +51.6 / +51.0 / +40.1 / +37.0 | -10.996 | -10.876 | y |
| `bench_dregon_Motor1_70` | **N** | 1.70 | 117 | -50.20 | -52.3 / -50.0 / -57.9 / -58.9 | +50.3 / +51.5 / +42.5 / +39.4 | -10.388 | -9.997 | y |
| `bench_dregon_Motor1_80` | y | 2.94 | 102 | -49.16 | -51.2 / -53.7 / -61.6 / -57.4 | +50.7 / +47.1 / +38.1 / +40.1 | -9.980 | -9.473 | y |
| `bench_dregon_Motor1_90` | **N** | 1.82 | 90 | -48.61 | -49.5 / -49.4 / -61.8 / -58.7 | +51.0 / +50.0 / +36.5 / +37.4 | -9.616 | -8.704 | y |
| `bench_dregon_Motor2_50` | y | 12.02 | 130 | -58.44 | -64.9 / -66.2 / -71.9 / -74.2 | +47.4 / +45.0 / +38.3 / +33.8 | -11.681 | -10.420 | y |
| `bench_dregon_Motor2_60` | y | 14.69 | 130 | -54.06 | -56.4 / -53.1 / -62.0 / -65.4 | +50.0 / +52.2 / +42.2 / +36.7 | -11.060 | -10.468 | y |
| `bench_dregon_Motor2_70` | y | 2.40 | 118 | -51.02 | -43.1 / -47.6 / -57.8 / -56.0 | +60.7 / +55.2 / +43.9 / +43.5 | -10.296 | -8.563 | y |
| `bench_dregon_Motor2_80` | y | 4.01 | 103 | -49.46 | -40.9 / -44.7 / -59.5 / -46.1 | +61.2 / +56.3 / +40.5 / +51.8 | -10.001 | -8.401 | y |
| `bench_dregon_Motor2_90` | y | 3.02 | 92 | -50.30 | -52.0 / -46.5 / -59.7 / -59.8 | +52.7 / +57.2 / +42.9 / +40.6 | -9.475 | -8.455 | y |
| `bench_dregon_Motor3_50` | y | 7.53 | 130 | -54.22 | -64.4 / -62.0 / -72.9 / -69.3 | +42.7 / +44.0 / +32.0 / +33.5 | -11.350 | -10.397 | y |
| `bench_dregon_Motor3_60` | y | 6.37 | 130 | -51.16 | -54.6 / -51.4 / -63.9 / -66.6 | +48.7 / +50.8 / +37.2 / +32.4 | -10.567 | -7.811 | y |
| `bench_dregon_Motor3_70` | y | 7.64 | 116 | -53.95 | -52.1 / -55.7 / -61.2 / -58.3 | +55.8 / +51.1 / +44.5 / +45.3 | -10.112 | -9.377 | y |
| `bench_dregon_Motor3_80` | y | 2.27 | 101 | -49.71 | -49.4 / -50.6 / -58.8 / -61.0 | +52.9 / +50.6 / +41.4 / +37.0 | -9.660 | -9.048 | y |
| `bench_dregon_Motor3_90` | y | 2.85 | 90 | -53.42 | -48.2 / -50.9 / -59.9 / -59.1 | +60.1 / +56.4 / +46.3 / +44.9 | -9.328 | -8.037 | y |
| `bench_dregon_Motor4_50` | y | 8.41 | 130 | -61.61 | -65.6 / -61.8 / -67.9 / -66.6 | +52.7 / +55.4 / +48.2 / +47.3 | -11.342 | -11.161 | y |
| `bench_dregon_Motor4_60` | y | 10.74 | 130 | -51.92 | -61.9 / -54.9 / -66.2 / -63.9 | +44.2 / +50.2 / +37.7 / +37.9 | -10.608 | -9.360 | y |
| `bench_dregon_Motor4_70` | **N** | 1.28 | 115 | -53.17 | -56.1 / -53.7 / -59.3 / -62.0 | +51.3 / +52.6 / +45.9 / +41.0 | -10.021 | -9.661 | y |
| `bench_dregon_Motor4_80` | y | 2.13 | 100 | -52.03 | -55.0 / -52.5 / -60.0 / -54.8 | +52.4 / +53.8 / +45.3 / +48.3 | -9.488 | -9.116 | y |
| `bench_dregon_Motor4_90` | **N** | 1.93 | 89 | -52.27 | -48.3 / -49.0 / -59.4 / -55.8 | +59.0 / +57.2 / +45.7 / +47.1 | -9.375 | -8.225 | y |
| `bench_dregon_allMotors_70` | y | 3.78 | 114 | -37.92 | -41.5 / -52.1 / -60.5 / -66.9 | +49.2 / +37.5 / +28.0 / +19.4 | -8.305 | -3.694 | y |

**21 of 21** supports put the modelled line peak above their own fitted floor at every one of k = 1, 2, 4, 8. The smallest margin anywhere in the set is +19.4 dB (`bench_dregon_allMotors_70`, k = 8); the k = 1 margins run +42.7 to +61.2 dB, median +51.3 dB. On this evidence the R2 comb is identified on all 21 windows, and the R1 failure mode — a profile whose conversion lands at or under the floor because the window held post-spin-down silence — occurs nowhere in R2.

Not evidence either way, recorded to forestall misreading: `diagnostics.saturated_coherence` is the MODEL-implied long-lag coherence `lag.saturated_coherence(k; sigma_eps)` at the fitted `sigma_eps`, not a measured inter-mic coherence. At k = 1 it spans 0.0001 to 1.0000 over the 21 fits, and it collapses below 0.9 on `bench_dregon_Motor2_60`, `bench_dregon_Motor4_60` — a statement about the fitted jitter, not about whether a comb is present.

## The four flagged recordings

`Motor1_70`, `Motor1_90`, `Motor4_70` and `Motor4_90` fail the rev-2 in-window band-power margin rule at the 2.0 dB threshold (1.28-1.93 dB, every one within 0.8 dB of it; `results/noise_v2/rounds/round2/supports/findings.md`) and were fitted anyway.

| support | margin dB | peak above floor k=1/2/4/8 dB | comb nats/cell | conv | starts conv | best-worst nats/cell | lam | lam max/min over starts |
|---|--:|---|--:|:-:|--:|--:|--:|--:|
| `bench_dregon_Motor1_70` | 1.70 | +50.3 / +51.5 / +42.5 / +39.4 | -10.388 | N | 0/4 | 0.00065 | 571.7 | 1.13 |
| `bench_dregon_Motor1_90` | 1.82 | +51.0 / +50.0 / +36.5 / +37.4 | -9.616 | N | 0/4 | 0.00122 | 24.24 | 1.22 |
| `bench_dregon_Motor4_70` | 1.28 | +51.3 / +52.6 / +45.9 / +41.0 | -10.021 | N | 0/4 | 0.0196 | 0.07483 | 1.46e+04 |
| `bench_dregon_Motor4_90` | 1.93 | +59.0 / +57.2 / +45.7 / +47.1 | -9.375 | N | 0/4 | 0.000154 | 1.05 | 1.16 |

All four carry an identified comb by the conversion above — k = 1 peaks +50.3 to +59.0 dB over their own floors, inside the set's range — so a low band-power margin is not evidence of an absent comb: that statistic is measured on a 1.5 Hz demodulation band while these fits carry 89-117 orders. What the four do show is the round's worst conditioning: `bench_dregon_Motor4_70` spans a factor of 1.46e+04 in `lam` across its four starts, and all four supports are 0/4 converged. Their PARAMETERS are draws from a multi-modal MAP surface; their COMB is not in doubt.

## Against round 1

The same 21 supports were fitted in R1 on the rev-1 windows. The WINDOWS CHANGED — the rev-2 rule puts every window on the motor — so per-support `nats/cell` is not comparable across rounds (different data, different cell counts: `bench_dregon_Motor1_70` had 418304 cells in R1 against 1817744 in R2). What is comparable is the conditioning and the comb-versus-floor geometry.

| quantity | R1 (`round1/fits`) | R2 (`round2/fits`) |
|---|---|---|
| DREGON bench supports | 21 | 21 |
| L-BFGS converged, reported fit | 1 / 21 | 3 / 21 |
| all four starts inside 1e-4 nats/cell | 1 / 21 | 3 / 21 |
| best-worst nats/cell: median [min, max] | 0.00243 [6.64e-05, 0.0448] | 0.00122 [4.75e-05, 0.0568] |
| `lam` max/min over starts: median [max] | 2.38 [1.26e+04] | 1.95 [1.94e+05] |
| fitted floor dB: min / max | -76.38 / -36.97 | -61.61 / -37.92 |
| k = 1 peak above floor dB: min / median / max | +7.9 / +39.3 / +61.2 | +42.7 / +51.3 / +61.2 |
| whittle nats/cell, median (NOT comparable, different windows) | -15.918 | -10.084 |

- **Windows.** 11 of the 21 R1 fits sat on a floor more than 15 dB below their R2 counterpart (`Motor1_70` -76.1 -> -50.2 dB, `Motor2_60` -76.0 -> -54.1 dB, `Motor2_70` -76.3 -> -51.0 dB, `Motor2_80` -76.2 -> -49.5 dB, `Motor2_90` -76.3 -> -50.3 dB, `Motor3_50` -74.4 -> -54.2 dB, `Motor3_60` -75.5 -> -51.2 dB, `Motor3_70` -73.0 -> -53.9 dB, `Motor3_90` -76.4 -> -53.4 dB, `Motor4_70` -74.8 -> -53.2 dB, `Motor4_80` -75.6 -> -52.0 dB): those are the post-spin-down windows the rev-2 rule replaced. R2 floors span -61.6 to -37.9 dB, a 23.7 dB range against 39.4 dB in R1.
- **Comb geometry.** The weakest R1 k = 1 line peak over its own floor was +7.9 dB (`bench_dregon_Motor2_60`) and 4 R1 fits were under +20 dB; in R2 the minimum is +42.7 dB and the median moved +12.0 dB. The comb the round-1 diagnosis said was missing for want of a motor-on window is present everywhere on the rev-2 windows.
- **Restart spread.** The median best-worst gap fell from 0.00243 to 0.00122 nats/cell (factor 2.0), but the worst support did not improve: 0.0448 at `bench_dregon_Motor1_70` in R1 against 0.0568 at `bench_dregon_Motor3_80` in R2, 568 times the tolerance. Supports with all four starts inside the tolerance: 3 of 21 in R2 against 1 of 21 in R1.
- **`lam` spread.** The median spread over starts fell from 2.38 to 1.95, but the tail survives: 3 R2 supports still span more than a factor of 100 (worst `bench_dregon_Motor3_90` at 1.94e+05) against 4 in R1 (worst 1.26e+04). The shaft rate remains the least identified parameter of the round.
- **Convergence.** 3 of 21 reported fits pass the L-BFGS restart test in R2 against 1 of 21 in R1. That is a change in kind, not in degree: 18 of 21 R2 fits are still stopped by the iteration budget, and every number quoted from them carries its status in the tables above.

## Provenance

- Bench fits: job `nv2-r2-dregon-restarts2-3b1ffe` (uni-cpu, succeeded), 21 dregon-bench supports x 4 starts on the R2 supports (`results/noise_v2/rounds/round2/supports/index.json`).
- Floor-only flight fit: job `nv2-r2-dregon-floor2-01c39a` (uni-cpu, succeeded), `dregon_room2_floor__flight_floor_only.json`, comb frozen from the four Motor*_70 R2 bench fits by the log-mean rule recorded in that payload's `frozen_from` block (`frozen_comb_from`, `frozen_comb_paths`, `rule`, `profile_orders` 118 from per-fit 117/118/116/115). The two jobs' 21 bench payloads and 84 restart payloads are byte-identical; the floor job's outputs are the superset and are what is committed here.
- Restart reduction: `scripts/noise_v2_fit.py:reduce_restarts`, re-run in the main checkout over the committed `restarts/` payloads; it reproduced all 21 committed bench payloads byte-for-byte (`restarts/reduce.json`).
- The `dregon_room2_floor` flight floor-only fit carries no `restarts` block: it is a single start (`optimiser.seed` 0), NOT converged.
