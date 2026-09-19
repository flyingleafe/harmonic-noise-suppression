# Noise model v2 — round 3 fits (Model R3)

22 fit JSON(s) under `results/noise_v2/rounds/round3/fits`. Every number below is read from a `noise-v2-fit/2` payload in that directory (a `/1` payload's per-order OU is mapped onto its equivalent Lorentzian width first); nothing is recomputed here. The `gamma` columns are the rotor-max width in Hz at that order.

The tables down to `Prior edges and non-convergence` are the generator's output
(`scripts/noise_v2_fit.py findings`, re-run in the main checkout over this
directory; byte-identical to the copy the fit job wrote, except that the job ran
before `michaels_fly125_all__flight.json` was committed, so that row is new).
The sections after them are appended by `R3Bench` and read the same payloads;
every input is named where it is used. `michaels_fly125_all` is `R3Michaels`'
flight fit, not part of this bench refit — it appears because the generator
globs the directory, and it is excluded from every appended table below.

## Per-support parameters

| support | mode | R | k_max | sigma_nu | lam | g(k=1) | g(k=2) | g(k=4) | g(k=8) | g(k=16) | g(k=32) | low-k check | pinned | carrier rev/s | floor dB | whittle nats | comb | floor | cells | conv | wall s |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|---|---|--:|--:|--:|--:|--:|:-:|--:|
| `bench_dregon_Motor1_50` | bench | 1 | 130 | 0.4025 | 24.706 | 8.687e-05 | 0.004329 | 0.0009361 | 0.6106 | 1.627 | 4.326 | pass | — | 49.015 | -64.95 | -2.12466e+07 | -2.04363e+07 | -810329 | 1809712 | y | 1355 |
| `bench_dregon_Motor1_60` | bench | 1 | 130 | 0.4734 | 9.500 | 5.252e-05 | 5.923e-05 | 0.01442 | 0.553 | 0.9315 | 3.217 | pass | — | 58.766 | -61.25 | -1.62744e+07 | -1.56484e+07 | -626044 | 1474496 | y | 1425 |
| `bench_dregon_Motor1_70` | bench | 1 | 117 | 0.8478 | 0.358 | 0.004788 | 0.00543 | 0.04924 | 0.5157 | 1.402 | 6.735 | fail_k2_ramp | — | 68.314 | -52.80 | -1.89237e+07 | -1.82228e+07 | -700915 | 1817744 | N | 1742 |
| `bench_dregon_Motor1_80` | bench | 1 | 102 | 0.8819 | 2.005 | 0.001645 | 0.000874 | 0.01989 | 2.142 | 1.28 | 5.872 | pass | — | 78.101 | -50.67 | -6.70935e+06 | -6.466e+06 | -243347 | 671688 | y | 813 |
| `bench_dregon_Motor1_90` | bench | 1 | 90 | 1.1569 | 0.375 | 0.001969 | 0.007777 | 0.07757 | 1.21 | 1.084 | 5.61 | pass | — | 88.166 | -50.85 | -7.04939e+06 | -6.80321e+06 | -246177 | 733640 | y | 801 |
| `bench_dregon_Motor2_50` | bench | 1 | 130 | 0.6431 | 165.336 | 5.341e-05 | 0.0005282 | 0.014 | 0.0952 | 3.218 | 1.691 | pass | — | 48.435 | -58.69 | -8.19289e+06 | -7.91588e+06 | -277005 | 703808 | y | 791 |
| `bench_dregon_Motor2_60` | bench | 1 | 130 | 0.3964 | 0.117 | 0.00035 | 0.008 | 0.01492 | 0.04653 | 1.208 | 35.17 | pass | — | 58.144 | -57.17 | -7.66282e+06 | -7.38362e+06 | -279193 | 687752 | y | 833 |
| `bench_dregon_Motor2_70` | bench | 1 | 118 | 0.7786 | 0.014 | 0.0007938 | 0.005462 | 0.02034 | 0.749 | 0.8424 | 9.103 | pass | — | 67.552 | -53.09 | -6.99536e+06 | -6.76168e+06 | -233684 | 676856 | y | 779 |
| `bench_dregon_Motor2_80` | bench | 1 | 103 | 0.7817 | 1.724 | 0.001264 | 0.01012 | 0.03128 | 21.02 | 1.006 | 7.628 | pass | — | 77.210 | -50.61 | -7.13518e+06 | -6.89808e+06 | -237107 | 712416 | y | 784 |
| `bench_dregon_Motor2_90` | bench | 1 | 92 | 1.3047 | 0.012 | 0.001409 | 0.005368 | 0.03138 | 0.3474 | 0.9987 | 5.171 | pass | — | 86.695 | -48.09 | -6.33047e+06 | -6.12705e+06 | -203417 | 668824 | N | 709 |
| `bench_dregon_Motor3_50` | bench | 1 | 130 | 0.6901 | 87.367 | 0.0002607 | 0.0005809 | 0.03647 | 0.1859 | 0.6623 | 5.361 | pass | — | 49.135 | -60.73 | -7.71429e+06 | -7.43509e+06 | -279195 | 678576 | N | 772 |
| `bench_dregon_Motor3_60` | bench | 1 | 130 | 0.5304 | 0.387 | 0.0005787 | 0.002592 | 0.01577 | 0.1656 | 1.061 | 9.547 | pass | — | 58.949 | -60.47 | -7.67807e+06 | -7.40775e+06 | -270320 | 721592 | N | 830 |
| `bench_dregon_Motor3_70` | bench | 1 | 116 | 0.5896 | 0.355 | 0.0006063 | 0.00267 | 0.02836 | 0.3039 | 1.265 | 5.8 | pass | — | 68.634 | -55.12 | -6.90122e+06 | -6.65893e+06 | -242289 | 681440 | y | 775 |
| `bench_dregon_Motor3_80` | bench | 1 | 101 | 0.8925 | 0.036 | 0.0004551 | 0.004949 | 0.0378 | 0.2469 | 1.486 | 4.242 | pass | — | 78.814 | -51.10 | -6.87247e+06 | -6.62759e+06 | -244880 | 710120 | y | 750 |
| `bench_dregon_Motor3_90` | bench | 1 | 90 | 1.0974 | 0.014 | 0.0001092 | 0.005789 | 0.03337 | 0.3086 | 1.347 | 12.06 | pass | — | 88.244 | -55.29 | -6.30106e+06 | -6.09368e+06 | -207385 | 676856 | N | 770 |
| `bench_dregon_Motor4_50` | bench | 1 | 130 | 1.7723 | 1370.163 | 0.0005915 | 0.001037 | 0.02733 | 0.4442 | 4.011 | 4.478 | pass | — | 49.806 | -62.80 | -7.83551e+06 | -7.54366e+06 | -291849 | 690616 | y | 836 |
| `bench_dregon_Motor4_60` | bench | 1 | 130 | 0.4964 | 0.122 | 0.003133 | 0.002533 | 0.0202 | 0.06803 | 1.085 | 7.348 | pass | — | 59.741 | -56.44 | -7.26403e+06 | -6.99123e+06 | -272804 | 678576 | N | 798 |
| `bench_dregon_Motor4_70` | bench | 1 | 115 | 0.7803 | 0.008 | 0.001528 | 0.002324 | 0.01108 | 1.876 | 1.333 | 3.861 | pass | — | 69.333 | -56.61 | -6.91634e+06 | -6.66541e+06 | -250933 | 687752 | N | 752 |
| `bench_dregon_Motor4_80` | bench | 1 | 100 | 1.1978 | 0.057 | 0.003139 | 0.002678 | 0.01914 | 5.832 | 2.286 | 3.734 | pass | — | 79.375 | -55.17 | -6.38092e+06 | -6.14945e+06 | -231467 | 671688 | y | 833 |
| `bench_dregon_Motor4_90` | bench | 1 | 89 | 1.1416 | 0.013 | 0.0007535 | 0.00416 | 0.01848 | 0.1833 | 0.9983 | 3.115 | pass | — | 89.172 | -49.55 | -6.25099e+06 | -6.04331e+06 | -207684 | 667672 | N | 797 |
| `bench_dregon_allMotors_70` | bench | 4 | 114 | 0.5484 | 44.951 | 12.77 | 0.2287 | 0.5722 | 2.213 | 5.428 | 39.92 | fail_above_floor | — | 64.636, 67.660, 68.736, 69.565 | -37.67 | -1.61005e+07 | -1.58228e+07 | -277731 | 1973192 | N | 3576 |
| `michaels_fly125_all` | flight | 4 | 81 | 1.2333 | 0.500 | 2.99 | 1.199 | 9.079 | 3.094 | 14.41 | 34.25 | pass | — | label | -39.08 | -3.3524e+07 | -3.26195e+07 | -904508 | 2016000 | N | 5621 |

## Population median [IQR]

| set | n | sigma_nu | lam | gamma log-mean over orders (Hz) |
|---|--:|---|---|---|
| DREGON single-motor bench (all throttles) | 20 | 0.781 [0.5748, 1.108] | 0.3568 [0.03064, 3.879] | 3.363 [2.633, 4.095] |
| flight | 1 | 1.233 [1.233, 1.233] | 0.5 [0.5, 0.5] | 51.33 [51.33, 51.33] |

## Four-motor validation

`motor_allMotors_70` against the four `Motor{1-4}_70` single-motor fits: the ratio of the four-rotor value to the geometric mean of the per-rotor ones and the per-rotor spread, as read off the fits. This is a raw parameter comparison only — what the four-motor support does or does not validate is measured in its own workstream, not here.

| parameter | four-motor | per-rotor geo-mean | ratio | per-rotor min/max |
|---|--:|--:|--:|---|
| `sigma_nu` | 0.5484 | 0.7423 | 0.739 | 0.5896 / 0.8478 |
| `lam` | 44.95 | 0.0608 | 739.319 | 0.007933 / 0.3583 |
| `gamma_log_mean` | 1.861 | 3.41 | 0.546 | 2.503 / 4.3 |

## Multi-start restarts

Each support was fitted from several starts: start 0 from the data-driven initialisation, the others from a log-normal perturbation of the dynamics init (`OptimSpec.init_jitter`; a bare seed change is a no-op because a bench fit is deterministic). The REPORTED fit above is the start with the lowest polished objective. `best-median` and `best-worst` are that objective's advantage over the median and the worst start, per observed cell, against the same 0.0001 nats/cell tolerance the convergence test uses; `starts agree` is yes only when even the worst start is inside it. The dynamics columns are min / median / max over the starts.

| support | starts | best nats/cell | best-median | best-worst | starts agree | L-BFGS conv | sigma_nu | lam | gamma log-mean |
|---|--:|--:|--:|--:|:-:|:-:|---|---|---|
| `bench_dregon_Motor1_50` | 4 | -11.7403 | 1.04e-05 | 3e-05 | y | y | 0.403 / 0.409 / 0.413 | 24.7 / 25.5 / 25.9 | 5.1 / 5.13 / 5.15 |
| `bench_dregon_Motor1_60` | 4 | -11.0373 | 3.76e-05 | 0.000517 | N | y | 0.468 / 0.471 / 0.475 | 9.11 / 9.26 / 9.5 | 3 / 3.08 / 3.18 |
| `bench_dregon_Motor1_70` | 4 | -10.4105 | 0.000232 | 0.000804 | N | N | 0.825 / 0.848 / 0.868 | 0.141 / 0.312 / 0.38 | 3.6 / 3.91 / 4.02 |
| `bench_dregon_Motor1_80` | 4 | -9.9888 | 0.0033 | 0.00645 | N | y | 0.818 / 0.851 / 0.884 | 1.14 / 1.54 / 2 | 2.04 / 2.69 / 3.14 |
| `bench_dregon_Motor1_90` | 4 | -9.6088 | 2.38e-06 | 0.000154 | N | y | 1.16 / 1.16 / 1.16 | 0.317 / 0.371 / 0.389 | 2.67 / 2.71 / 3.03 |
| `bench_dregon_Motor2_50` | 4 | -11.6408 | 4.19e-05 | 4.67e-05 | y | y | 0.639 / 0.645 / 0.648 | 163 / 166 / 166 | 3.96 / 4 / 4.13 |
| `bench_dregon_Motor2_60` | 4 | -11.1418 | 2.7e-05 | 0.000125 | N | y | 0.394 / 0.396 / 0.397 | 0.0534 / 0.0827 / 0.117 | 4.28 / 4.34 / 4.6 |
| `bench_dregon_Motor2_70` | 4 | -10.3351 | 0.000159 | 0.000173 | N | y | 0.771 / 0.772 / 0.779 | 0.0123 / 0.014 / 0.0153 | 2.48 / 2.56 / 2.63 |
| `bench_dregon_Motor2_80` | 4 | -10.0155 | 1.88e-05 | 2.76e-05 | y | y | 0.781 / 0.782 / 0.782 | 1.69 / 1.72 / 1.72 | 2.68 / 2.73 / 2.83 |
| `bench_dregon_Motor2_90` | 4 | -9.4651 | 4e-05 | 7.34e-05 | y | N | 1.3 / 1.31 / 1.31 | 0.00992 / 0.0117 / 0.0139 | 2.08 / 2.33 / 2.76 |
| `bench_dregon_Motor3_50` | 4 | -11.3684 | 0.000905 | 0.0012 | N | N | 0.668 / 0.683 / 0.69 | 77.7 / 81.6 / 87.4 | 4.43 / 4.56 / 5.35 |
| `bench_dregon_Motor3_60` | 4 | -10.6405 | 5.52e-06 | 4.34e-05 | y | N | 0.53 / 0.531 / 0.531 | 0.329 / 0.363 / 0.408 | 4.29 / 4.33 / 4.38 |
| `bench_dregon_Motor3_70` | 4 | -10.1274 | 3.34e-06 | 6.44e-06 | y | y | 0.588 / 0.589 / 0.59 | 0.248 / 0.32 / 0.355 | 4.29 / 4.3 / 4.38 |
| `bench_dregon_Motor3_80` | 4 | -9.6779 | 0.000714 | 0.000745 | N | y | 0.889 / 0.892 / 0.893 | 0.00909 / 0.0222 / 0.0361 | 2.13 / 2.29 / 3.14 |
| `bench_dregon_Motor3_90` | 4 | -9.3093 | 0.000936 | 0.000995 | N | N | 1.08 / 1.08 / 1.1 | 0.0143 / 0.0313 / 0.0485 | 3.96 / 4.06 / 4.21 |
| `bench_dregon_Motor4_50` | 4 | -11.3457 | 5.99e-05 | 0.000156 | N | y | 1.65 / 1.78 / 1.95 | 1.2e+03 / 1.38e+03 / 1.66e+03 | 3.46 / 3.61 / 4.38 |
| `bench_dregon_Motor4_60` | 4 | -10.7048 | 0.000152 | 0.000269 | N | N | 0.496 / 0.496 / 0.498 | 0.104 / 0.112 / 0.122 | 3.55 / 3.62 / 3.92 |
| `bench_dregon_Motor4_70` | 4 | -10.0564 | 0.00048 | 0.0063 | N | N | 0.78 / 0.789 / 0.8 | 0.00793 / 0.00883 / 0.0103 | 2.61 / 3.1 / 3.15 |
| `bench_dregon_Motor4_80` | 4 | -9.4998 | 0.000944 | 0.000972 | N | y | 1.2 / 1.2 / 1.21 | 0.0566 / 0.0731 / 0.0888 | 1.87 / 2.18 / 2.36 |
| `bench_dregon_Motor4_90` | 4 | -9.3624 | 0.000111 | 0.0043 | N | N | 1.14 / 1.14 / 1.15 | 0.00822 / 0.0126 / 0.0157 | 1.44 / 1.53 / 1.62 |
| `bench_dregon_allMotors_70` | 4 | -8.1596 | 0.000123 | 0.000408 | N | N | 0.548 / 0.554 / 0.564 | 45 / 47.4 / 49.5 | 1.6 / 1.76 / 1.86 |

6 of 21 supports have every start inside the tolerance. The widest disagreement is `bench_dregon_Motor1_80` at 0.00645 nats/cell, 65 times the tolerance. `lam` alone spans a factor of 1.3 (median over supports) and up to 3.97 across the starts of one support: on this evidence the R1 bench MAP problem is multi-modal, and a number computed from a single start is a draw from that multiplicity rather than an estimate.

## The low-order `gamma_rk` check

R3's one possible degeneracy: in the Brownian limit the shaft term is Lorentzian too, so a `gamma_rk` ramping as `k^2` would absorb it. PASS is every fitted width at `k <= 4` sitting within a factor of 3 of the window's resolution floor `1 / (2 T)`; `fail_k2_ramp` is `gamma_4 / gamma_1 >= 8`, and then `lam` takes the long-lag pin instead of the bench value.

| support | verdict | max gamma(k<=4) / resolution | gamma_4 / gamma_1 |
|---|:-:|--:|--:|
| `bench_dregon_Motor1_50` | pass | 0.273 | 10.8 |
| `bench_dregon_Motor1_60` | pass | 0.742 | 275 |
| `bench_dregon_Motor1_70` | fail_k2_ramp | 3.12 | 10.3 |
| `bench_dregon_Motor1_80` | pass | 0.466 | 12.1 |
| `bench_dregon_Motor1_90` | pass | 1.98 | 39.4 |
| `bench_dregon_Motor2_50` | pass | 0.344 | 262 |
| `bench_dregon_Motor2_60` | pass | 0.358 | 42.6 |
| `bench_dregon_Motor2_70` | pass | 0.48 | 25.6 |
| `bench_dregon_Motor2_80` | pass | 0.777 | 24.7 |
| `bench_dregon_Motor2_90` | pass | 0.732 | 22.3 |
| `bench_dregon_Motor3_50` | pass | 0.863 | 140 |
| `bench_dregon_Motor3_60` | pass | 0.397 | 27.2 |
| `bench_dregon_Motor3_70` | pass | 0.814 | 46.8 |
| `bench_dregon_Motor3_80` | pass | 0.936 | 83 |
| `bench_dregon_Motor3_90` | pass | 0.952 | 305 |
| `bench_dregon_Motor4_50` | pass | 0.658 | 46.2 |
| `bench_dregon_Motor4_60` | pass | 0.478 | 6.45 |
| `bench_dregon_Motor4_70` | pass | 0.266 | 7.25 |
| `bench_dregon_Motor4_80` | pass | 0.448 | 6.1 |
| `bench_dregon_Motor4_90` | pass | 0.43 | 24.5 |
| `bench_dregon_allMotors_70` | fail_above_floor | 879 | 4.75 |
| `michaels_fly125_all` | pass | 2.32 | 160 |

20 of 22 fits pass. Failing: `bench_dregon_Motor1_70`, `bench_dregon_allMotors_70`

## Prior edges and non-convergence

- `bench_dregon_Motor1_50`: lam = 24.71 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_Motor1_70`: low-order gamma check fail_k2_ramp
- `bench_dregon_Motor1_70`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor2_50`: lam = 165.34 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_Motor2_60`: lam = 0.12 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_Motor2_70`: lam = 0.01 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_Motor2_90`: lam = 0.01 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_Motor2_90`: sigma_nu = 1.3047 outside the prior's central 95 % [0.074, 1.22]
- `bench_dregon_Motor2_90`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor3_50`: lam = 87.37 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_Motor3_50`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor3_60`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor3_80`: lam = 0.04 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_Motor3_90`: lam = 0.01 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_Motor3_90`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor4_50`: lam = 1370.16 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_Motor4_50`: sigma_nu = 1.7723 outside the prior's central 95 % [0.074, 1.22]
- `bench_dregon_Motor4_60`: lam = 0.12 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_Motor4_60`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor4_70`: lam = 0.01 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_Motor4_70`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_Motor4_80`: lam = 0.06 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_Motor4_90`: lam = 0.01 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_Motor4_90`: NOT converged (L-BFGS restart still improving)
- `bench_dregon_allMotors_70`: lam = 44.95 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_allMotors_70`: low-order gamma check fail_above_floor
- `bench_dregon_allMotors_70`: NOT converged (L-BFGS restart still improving)
- `michaels_fly125_all`: sigma_nu = 1.2333 outside the prior's central 95 % [0.11, 0.815]
- `michaels_fly125_all`: NOT converged (L-BFGS restart still improving)

## R3 per support: widths to k = 64, the low-order check, the restarts

`gamma(k)` is the max over rotors of `params.gamma_hz` at that order (the generator's table above stops at k = 32; k = 64 is added here). `over res` is `diagnostics.gamma_low_order_check.max_over_resolution` — the largest fitted width at k <= 4 in units of the window's resolution floor 1 / (2 T) — and `g4/g1` is the ramp ratio, read only where the floor test fails. `best-worst` is `restarts.best_minus_worst_per_cell` against the 1e-4 nats/cell tolerance; `starts conv` counts `restarts.converged`.

| support | sigma_nu | lam | g(k=1) | g(k=2) | g(k=4) | g(k=8) | g(k=16) | g(k=32) | g(k=64) | verdict | over res | g4/g1 | best-worst | starts conv | conv |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|--:|--:|--:|:-:|:-:|
| `bench_dregon_Motor1_50` | 0.4025 | 24.71 | 8.687e-05 | 0.004329 | 0.0009361 | 0.6106 | 1.627 | 4.326 | 16.96 | pass | 0.273 | 10.8 | 3e-05 | 4/4 | y |
| `bench_dregon_Motor1_60` | 0.4734 | 9.5 | 5.252e-05 | 5.923e-05 | 0.01442 | 0.553 | 0.9315 | 3.217 | 9.547 | pass | 0.742 | 275 | 0.000517 | 4/4 | y |
| `bench_dregon_Motor1_70` | 0.8478 | 0.3583 | 0.004788 | 0.00543 | 0.04924 | 0.5157 | 1.402 | 6.735 | 40.03 | fail_k2_ramp | 3.12 | 10.3 | 0.000804 | 3/4 | N |
| `bench_dregon_Motor1_80` | 0.8819 | 2.005 | 0.001645 | 0.000874 | 0.01989 | 2.142 | 1.28 | 5.872 | 21.89 | pass | 0.466 | 12.1 | 0.00645 | 3/4 | y |
| `bench_dregon_Motor1_90` | 1.157 | 0.3755 | 0.001969 | 0.007777 | 0.07757 | 1.21 | 1.084 | 5.61 | 11.5 | pass | 1.98 | 39.4 | 0.000154 | 4/4 | y |
| `bench_dregon_Motor2_50` | 0.6431 | 165.3 | 5.341e-05 | 0.0005282 | 0.014 | 0.0952 | 3.218 | 1.691 | 18.42 | pass | 0.344 | 262 | 4.67e-05 | 2/4 | y |
| `bench_dregon_Motor2_60` | 0.3964 | 0.1172 | 0.00035 | 0.008 | 0.01492 | 0.04653 | 1.208 | 35.17 | 22.69 | pass | 0.358 | 42.6 | 0.000125 | 4/4 | y |
| `bench_dregon_Motor2_70` | 0.7786 | 0.01353 | 0.0007938 | 0.005462 | 0.02034 | 0.749 | 0.8424 | 9.103 | 15.93 | pass | 0.48 | 25.6 | 0.000173 | 4/4 | y |
| `bench_dregon_Motor2_80` | 0.7817 | 1.724 | 0.001264 | 0.01012 | 0.03128 | 21.02 | 1.006 | 7.628 | 24.65 | pass | 0.777 | 24.7 | 2.76e-05 | 4/4 | y |
| `bench_dregon_Motor2_90` | 1.305 | 0.01197 | 0.001409 | 0.005368 | 0.03138 | 0.3474 | 0.9987 | 5.171 | 28.7 | pass | 0.732 | 22.3 | 7.34e-05 | 0/4 | N |
| `bench_dregon_Motor3_50` | 0.6901 | 87.37 | 0.0002607 | 0.0005809 | 0.03647 | 0.1859 | 0.6623 | 5.361 | 22.82 | pass | 0.863 | 140 | 0.0012 | 3/4 | N |
| `bench_dregon_Motor3_60` | 0.5304 | 0.3868 | 0.0005787 | 0.002592 | 0.01577 | 0.1656 | 1.061 | 9.547 | 18.7 | pass | 0.397 | 27.2 | 4.34e-05 | 0/4 | N |
| `bench_dregon_Motor3_70` | 0.5896 | 0.3553 | 0.0006063 | 0.00267 | 0.02836 | 0.3039 | 1.265 | 5.8 | 24.97 | pass | 0.814 | 46.8 | 6.44e-06 | 4/4 | y |
| `bench_dregon_Motor3_80` | 0.8925 | 0.03608 | 0.0004551 | 0.004949 | 0.0378 | 0.2469 | 1.486 | 4.242 | 18.26 | pass | 0.936 | 83 | 0.000745 | 3/4 | y |
| `bench_dregon_Motor3_90` | 1.097 | 0.0143 | 0.0001092 | 0.005789 | 0.03337 | 0.3086 | 1.347 | 12.06 | 23.13 | pass | 0.952 | 305 | 0.000995 | 0/4 | N |
| `bench_dregon_Motor4_50` | 1.772 | 1370 | 0.0005915 | 0.001037 | 0.02733 | 0.4442 | 4.011 | 4.478 | 3.248 | pass | 0.658 | 46.2 | 0.000156 | 3/4 | y |
| `bench_dregon_Motor4_60` | 0.4964 | 0.1221 | 0.003133 | 0.002533 | 0.0202 | 0.06803 | 1.085 | 7.348 | 11.8 | pass | 0.478 | 6.45 | 0.000269 | 1/4 | N |
| `bench_dregon_Motor4_70` | 0.7803 | 0.007933 | 0.001528 | 0.002324 | 0.01108 | 1.876 | 1.333 | 3.861 | 26.79 | pass | 0.266 | 7.25 | 0.0063 | 2/4 | N |
| `bench_dregon_Motor4_80` | 1.198 | 0.05664 | 0.003139 | 0.002678 | 0.01914 | 5.832 | 2.286 | 3.734 | 27.03 | pass | 0.448 | 6.1 | 0.000972 | 1/4 | y |
| `bench_dregon_Motor4_90` | 1.142 | 0.0126 | 0.0007535 | 0.00416 | 0.01848 | 0.1833 | 0.9983 | 3.115 | 40.98 | pass | 0.43 | 24.5 | 0.0043 | 0/4 | N |
| `bench_dregon_allMotors_70` | 0.5484 | 44.95 | 12.77 | 0.2287 | 0.5722 | 2.213 | 5.428 | 39.92 | 48.41 | fail_above_floor | 879 | 4.75 | 0.000408 | 0/4 | N |

## Against round 2, support by support

The windows are the SAME (`--set dregon-bench` is unchanged since R2; the R3 rebuild's `index.json` matches R2's over all 21 supports up to float ULP), so `nats/cell` IS comparable here, unlike the R1-to-R2 comparison. R2 is the `noise-v2-fit/1` law (per-order OU jitter, `sigma_eps`/`lam_eps`); R3 is `noise-v2-fit/2` (shaft OU plus a free Lorentzian width `gamma_hz` per line). A LOWER `nats/cell` is a better fit of the same data by a different model with a different parameter count, and the `delta` column is that difference, not a likelihood-ratio test.

| support | R3 nats/cell | R2 nats/cell | delta | R3 conv | R2 conv | R3 sigma_nu | R2 sigma_nu | R3 lam | R2 lam | R3 wall s | R2 wall s |
|---|--:|--:|--:|:-:|:-:|--:|--:|--:|--:|--:|--:|
| `bench_dregon_Motor1_50` | -11.7403 | -11.7271 | -0.0132 | y | y | 0.4025 | 5.124 | 24.71 | 5203 | 1355 | 1478 |
| `bench_dregon_Motor1_60` | -11.0373 | -10.9911 | -0.0462 | y | N | 0.4734 | 6.381 | 9.5 | 3520 | 1425 | 1888 |
| `bench_dregon_Motor1_70` | -10.4105 | -10.3733 | -0.0373 | N | N | 0.8478 | 4.993 | 0.3583 | 571.7 | 1742 | 2026 |
| `bench_dregon_Motor1_80` | -9.9888 | -9.9610 | -0.0277 | y | y | 0.8819 | 1.156 | 2.005 | 4.079 | 813 | 926 |
| `bench_dregon_Motor1_90` | -9.6088 | -9.5818 | -0.0270 | y | N | 1.157 | 2.109 | 0.3755 | 24.24 | 801 | 626 |
| `bench_dregon_Motor2_50` | -11.6408 | -11.6340 | -0.0068 | y | N | 0.6431 | 0.4482 | 165.3 | 73.87 | 791 | 562 |
| `bench_dregon_Motor2_60` | -11.1418 | -11.0379 | -0.1040 | y | N | 0.3964 | 2.124 | 0.1172 | 654.8 | 833 | 528 |
| `bench_dregon_Motor2_70` | -10.3351 | -10.2311 | -0.1040 | y | N | 0.7786 | 4.422 | 0.01353 | 587.1 | 779 | 470 |
| `bench_dregon_Motor2_80` | -10.0155 | -9.9404 | -0.0750 | y | N | 0.7817 | 6.515 | 1.724 | 1217 | 784 | 354 |
| `bench_dregon_Motor2_90` | -9.4651 | -9.4364 | -0.0287 | N | N | 1.305 | 1.561 | 0.01197 | 0.2666 | 709 | 367 |
| `bench_dregon_Motor3_50` | -11.3684 | -11.3145 | -0.0538 | N | N | 0.6901 | 102.3 | 87.37 | 3.787e+05 | 772 | 309 |
| `bench_dregon_Motor3_60` | -10.6405 | -10.4628 | -0.1776 | N | y | 0.5304 | 1.377 | 0.3868 | 28.47 | 830 | 488 |
| `bench_dregon_Motor3_70` | -10.1274 | -10.0841 | -0.0433 | y | N | 0.5896 | 143.2 | 0.3553 | 6.674e+05 | 775 | 330 |
| `bench_dregon_Motor3_80` | -9.6779 | -9.6374 | -0.0405 | y | N | 0.8925 | 1.459 | 0.03608 | 12.43 | 750 | 521 |
| `bench_dregon_Motor3_90` | -9.3093 | -9.2793 | -0.0300 | N | N | 1.097 | 1.716 | 0.0143 | 2.831 | 770 | 501 |
| `bench_dregon_Motor4_50` | -11.3457 | -11.3355 | -0.0101 | y | N | 1.772 | 0.786 | 1370 | 66.96 | 836 | 361 |
| `bench_dregon_Motor4_60` | -10.7048 | -10.5612 | -0.1436 | N | N | 0.4964 | 18.33 | 0.1221 | 1.629e+04 | 798 | 669 |
| `bench_dregon_Motor4_70` | -10.0564 | -10.0072 | -0.0492 | N | N | 0.7803 | 1.152 | 0.007933 | 0.07483 | 752 | 419 |
| `bench_dregon_Motor4_80` | -9.4998 | -9.4739 | -0.0259 | y | N | 1.198 | 1.589 | 0.05664 | 1.469 | 833 | 731 |
| `bench_dregon_Motor4_90` | -9.3624 | -9.3318 | -0.0305 | N | N | 1.142 | 1.581 | 0.0126 | 1.05 | 797 | 542 |
| `bench_dregon_allMotors_70` | -8.1596 | -8.1311 | -0.0285 | N | N | 0.5484 | 0.996 | 44.95 | 109.8 | 3576 | 1432 |

## Round summary, R3 against R2

| quantity | R2 (`round2/fits`) | R3 (`round3/fits`) |
|---|---|---|
| DREGON bench supports | 21 | 21 |
| L-BFGS converged, reported fit | 3 / 21 | 12 / 21 |
| all four starts inside 0.0001 nats/cell | 3 / 21 | 6 / 21 |
| best-worst nats/cell: median [min, max] | 0.00122 [4.75e-05, 0.0568] | 0.000269 [6.44e-06, 0.00645] |
| `lam` max/min over starts: median [max] | 1.95 [1.94e+05] | 1.3 [3.97] |
| `sigma_nu`: min / median / max | 0.4482 / 1.716 / 143.2 | 0.3964 / 0.7803 / 1.772 |
| `lam` /s: min / median / max | 0.07483 / 73.87 / 6.674e+05 | 0.007933 / 0.3583 / 1370 |
| low-order `gamma` check passed | n/a (`/1` law) | 19 / 21 |
| whittle nats/cell, median (SAME windows) | -10.0841 | -10.1274 |
| total wall s over the bench fits | 15529 | 21520 |

- **Objective.** R3 is lower (better) on 21 of 21 supports, higher on 0; the per-support delta runs -0.1776 to -0.0068 nats/cell, median -0.0373.
- **Conditioning.** Converged reported fits 3/21 -> 12/21; supports with all four starts inside 0.0001 nats/cell 3/21 -> 6/21; worst restart spread 0.0568 -> 0.00645 nats/cell.
- **Rates.** `sigma_nu` moved from 1.716 to 0.7803 rad/s (median) and `lam` from 73.87 to 0.3583 /s: the free per-line width takes the line broadening the R2 law could only buy with a fast shaft.

## The two low-order `gamma` failures

19 of the 21 bench fits pass the low-order check. The two that do not are named
here because the verdict governs whether that support's fitted `lam` may be
carried forward (on a fail, the long-lag pin is taken instead of the bench
value), and because neither failure is the degeneracy the check exists to catch.

| support | verdict | max gamma(k<=4) / resolution | gamma_4 / gamma_1 | resolution Hz | sigma_nu | lam | conv |
|---|:-:|--:|--:|--:|--:|--:|:-:|
| `bench_dregon_Motor1_70` | `fail_k2_ramp` | 3.12 | 10.3 | 0.01578 | 0.8478 | 0.3583 | N |
| `bench_dregon_allMotors_70` | `fail_above_floor` | 879 | 4.75 | 0.01453 | 0.5484 | 44.95 | N |

- `bench_dregon_Motor1_70` sits just over the factor-3 floor band (3.12) with a
  ramp of 10.3 against the 8 threshold, i.e. it fails on both legs by a hair.
  It is one of R2's four flagged low-margin recordings and it is 0/4 converged
  in R3 as it was in R2.
- `bench_dregon_allMotors_70` is the four-rotor support and fails on the FLOOR
  leg by three orders of magnitude (879 x resolution), with a ramp of 4.75 —
  under the ramp threshold, so this is NOT the `k^2` shaft-absorbing mode. Its
  `gamma(k=1)` is 12.77 Hz against 0.0002-0.013 Hz on the single-motor supports:
  with four carriers frozen at their own labels, whatever mismatch remains
  between the four label tracks and the four true speeds is absorbed by the
  k = 1 width. The other 20 supports carry one rotor and one carrier and show
  nothing of the kind.

## Provenance

- Bench fits: job `nv2-r3-bench-9ebc3b` (uni-cpu, 16 cpus, `--time 3h`,
  succeeded, `R3BENCH_DONE exit_build=0 exit_fit=0`), code SHA
  `01e1464bc0c5b424c3d65abca4e690f6ae6acc82`, submitted from the detached
  worktree `.worktrees/submit-R3Bench`. 21 `dregon-bench` supports x 4 starts
  (`--seeds 4`, `--init-jitter` 0.8 on starts 1-3), `--jobs 16 --threads 1`.
  Started 2026-09-18T22:52:05Z, ended 2026-09-19T01:03:37Z (2 h 11 min).
- Windows: rebuilt in-job with `scripts/noise_v2_supports.py build --set
  dregon-bench`. Checked equal to R2's: `round2/supports/index.json` against
  the rebuild's `round3/supports/index.json` agrees on all 21 names, specs and
  window rules, differing only in float-ULP wobble of `level_db`,
  `level_deficit_db`, `residual_max_abs_hz` and a few `npz_bytes`. That is why
  the R3-to-R2 `nats/cell` comparison above is a comparison and not a coincidence.
- Restart reduction: `scripts/noise_v2_fit.py:reduce_restarts`, re-run in the
  main checkout over the fetched `restarts/` payloads. 16 of 21 reproduced the
  job's own reduction BYTE-for-byte; the other 5
  (`Motor1_50`, `Motor1_70`, `Motor1_90`, `Motor3_80`, `Motor4_70`) differ only
  in the last ULP of `restarts.params.gamma_hz.log_mean` (that field is
  `exp(mean(log gamma))`, evaluated on a different CPU) — every selected
  restart, fitted parameter, objective, convergence flag, ladder VALUE and
  `best_minus_worst_per_cell` reproduced exactly. Recorded in
  `restarts/reduce.json`; the COMMITTED payloads are the job's own.
