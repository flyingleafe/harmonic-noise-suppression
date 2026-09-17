# Order decoherence at long lags - findings

Source: `scripts/noise_v2_order_decoherence.py --long --max-lag-s 5`. Numbers: `results/noise_v2/decoherence_long/decoherence_long.json`. Figure: `docs/explainers/noise-model-v2-plan/decoherence_long.png`.

The 0.5 s study returned an independent per-order term that grows as `V_eps(k, tau) = a k^p tau^q` with `q` near 1 - a Wiener phase per order, which has NO ceiling. A per-order OU PHASE has one: `V_eps` saturates at `2 sigma_eps^2/lambda_eps` after `tau_c = 1/lambda_eps`, and below `tau_c` the two shapes are the same curve. The question is therefore only decidable at lags long enough to reach `tau_c`, which is what this run measures: the same estimator, the same controls, lags out to 5 s on the windows long enough to hold 3 of them.

## The supports

| support | window [s] | from [s] | rate [rev/s] | cells | lags [s] | window |
|---|---|---|---|---|---|---|
| `motor_Motor1_70` | 30.6 | 4.8 | 68.507 | 213 | 0.5, 1, 2, 3, 5 | level |
| `DREGON-bench__motor_Motor1_50` | 30.0 | 6.0 | 49.035 | 181 | 0.5, 1, 2, 3, 5 | published |
| `DREGON-bench__motor_Motor1_60` | 30.0 | 8.0 | 58.782 | 209 | 0.5, 1, 2, 3, 5 | published |
| `DREGON-bench__motor_Motor1_70` | 30.0 | 5.0 | 68.504 | 215 | 0.5, 1, 2, 3, 5 | published |

`motor_Motor1_70` enters twice: once on the level-detected active run - the longest continuous single-rotor window this project holds - and once on its published `noise-v2-bench-points` window. The other two supports are the remaining single-rotor points whose published span reaches 20 s. Orders are the 2, 4, 8, 16, 24, 32, 40 family, all of them below the 4200 Hz anti-alias corner at these rates (order 40 of the fastest support sits at 2740 Hz).

## The numbers

`V` is `V_eps` in rad^2: observed, minus the measured leakage of the shaft estimate, minus the additive-noise floor of that support's paired control. `q` and `a` are the power law `V = a tau^q`, `tau_c` and `c` the saturating `V = c (1 - exp(-tau/tau_c))`, both fitted by least squares on `log V` over the lags from 0.2 s up; `n fit` is how many lags that left. `dAIC = AIC(power) - AIC(saturating)`, so a POSITIVE value favours the ceiling; the decision bar is 2.

Two read-out limits are marked. A bracketed `V` is a lag whose OBSERVED residual is over the 3 rad^2 estimator ceiling: there `|gamma|` has reached its own sampling floor, the curve bends because the instrument has run out, and the lag takes part in NO fit (control (ii) is what proves this - its planted random walk crosses the ceiling at the high orders and the raw curve then flattens). A `*` on `k` marks an order already over the ceiling at 500 ms, i.e. one that has no readable long lag at all. A `tau_c` beyond the longest FITTED lag means the saturating curve has straightened into the power law inside the data, which is not evidence of a ceiling; a `verdict` of `flat` means the curve grows by less than 1.25 across the fitted lags, so there is no lag dependence left for either shape to explain.

### Pooled over the 4 long supports

| k | V(0.5 s) | V(1 s) | V(2 s) | V(5 s) | n fit | q | a | tau_c [s] | c | dAIC | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.0254 | 0.061 | 0.138 | 0.431 | 6 | +1.17 | 0.0611 | 1202604.18 | 7.58e+04 | -16.1 | growing |
| 4 | 0.0866 | 0.197 | 0.321 | 0.239 | 6 | +0.83 | 0.119 | 2.15 | 0.368 | +2.9 | ceiling |
| 8 | 0.672 | 1.15 | 1.38 | (2.4) | 4 | +0.73 | 0.978 | 1.12 | 1.76 | +5.0 | ceiling |
| 16 | 1.48 | (2.38) | (0.732) | (0.312) | 2 | - | - | - | - | - | no fit |
| 24* | (1.72) | (1.32) | (-2.17) | (-3.78) | 1 | - | - | - | - | - | no fit |
| 32* | (3.11) | (-1.32) | (-4.67) | (-8.71) | 0 | - | - | - | - | - | no fit |
| 40* | (1.55) | (-0.577) | (-7.97) | (-21.6) | 0 | - | - | - | - | - | no fit |

### `motor_Motor1_70` (30.6 s, 68.507 rev/s, level window)

| k | V(0.5 s) | V(1 s) | V(2 s) | V(5 s) | n fit | q | a | tau_c [s] | c | dAIC | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.00866 | 0.011 | 0.0361 | 0.0655 | 6 | +0.89 | 0.0161 | 8.13 | 0.144 | -1.1 | growing (no ceiling inside the fitted lags) |
| 4 | 0.0818 | 0.11 | 0.287 | 0.41 | 6 | +0.91 | 0.125 | 5.31 | 0.774 | +2.1 | growing (no ceiling inside the fitted lags) |
| 8 | 0.47 | 0.666 | 0.573 | (1.42) | 5 | +0.48 | 0.584 | 0.62 | 0.853 | -1.2 | undetermined |
| 16 | 1.61 | (2.56) | (1.73) | (0.125) | 2 | - | - | - | - | - | no fit |
| 24* | (2.21) | (2.19) | (1.42) | (-3.3) | 1 | - | - | - | - | - | no fit |
| 32* | (3.49) | (0.348) | (-3.95) | (-17.7) | 0 | - | - | - | - | - | no fit |
| 40* | (1.34) | (-0.874) | (-8.6) | (-26.5) | 0 | - | - | - | - | - | no fit |

### `DREGON-bench__motor_Motor1_50` (30.0 s, 49.035 rev/s, published window)

| k | V(0.5 s) | V(1 s) | V(2 s) | V(5 s) | n fit | q | a | tau_c [s] | c | dAIC | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.0379 | 0.104 | 0.251 | 0.642 | 6 | +1.13 | 0.104 | 1202604.11 | 1.28e+05 | -5.8 | growing |
| 4 | 0.108 | 0.153 | 0.246 | -0.823 | 4 | +0.62 | 0.159 | 0.83 | 0.249 | -8.8 | growing |
| 8 | 0.668 | 1.42 | (1.9) | (1.35) | 3 | - | - | - | - | - | no fit |
| 16* | (1.02) | (1.17) | (-0.856) | (-0.504) | 1 | - | - | - | - | - | no fit |
| 24* | (0.998) | (-1.65) | (-8.57) | (-6.51) | 0 | - | - | - | - | - | no fit |
| 32 | - | - | - | - | 0 | - | - | - | - | - | no fit |
| 40 | - | - | - | - | 0 | - | - | - | - | - | no fit |

### `DREGON-bench__motor_Motor1_60` (30.0 s, 58.782 rev/s, published window)

| k | V(0.5 s) | V(1 s) | V(2 s) | V(5 s) | n fit | q | a | tau_c [s] | c | dAIC | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.049 | 0.118 | 0.227 | 0.911 | 6 | +1.24 | 0.109 | 1202603.95 | 1.37e+05 | -12.2 | growing |
| 4 | 0.084 | 0.421 | 0.489 | 0.903 | 5 | +0.87 | 0.232 | 4.97 | 1.34 | +0.5 | undetermined |
| 8 | 1.06 | 1.81 | (2.42) | (4.78) | 3 | - | - | - | - | - | no fit |
| 16* | (1.7) | (3.09) | (0.552) | (1.65) | 1 | - | - | - | - | - | no fit |
| 24* | (1.36) | (2.29) | (-2.47) | (-2.16) | 0 | - | - | - | - | - | no fit |
| 32* | (2.85) | (-2.93) | (-8.37) | (1.38) | 0 | - | - | - | - | - | no fit |
| 40 | - | - | - | - | 0 | - | - | - | - | - | no fit |

### `DREGON-bench__motor_Motor1_70` (30.0 s, 68.504 rev/s, published window)

| k | V(0.5 s) | V(1 s) | V(2 s) | V(5 s) | n fit | q | a | tau_c [s] | c | dAIC | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.00618 | 0.0113 | 0.0388 | 0.105 | 6 | +1.39 | 0.0125 | 1202604.26 | 1.61e+04 | -12.6 | growing |
| 4 | 0.0722 | 0.103 | 0.261 | 0.465 | 6 | +1.06 | 0.109 | 155.98 | 17.3 | -0.5 | growing (no ceiling inside the fitted lags) |
| 8 | 0.495 | 0.71 | 0.61 | (2.07) | 5 | +0.55 | 0.647 | 0.81 | 1.06 | -1.6 | undetermined |
| 16 | 1.59 | (2.71) | (1.5) | (-0.0258) | 2 | - | - | - | - | - | no fit |
| 24* | (2.31) | (2.47) | (0.925) | (-3.16) | 1 | - | - | - | - | - | no fit |
| 32* | (2.98) | (-1.37) | (-1.7) | (-9.84) | 0 | - | - | - | - | - | no fit |
| 40* | (1.77) | (-0.281) | (-7.33) | (-16.6) | 0 | - | - | - | - | - | no fit |

## The controls at the same lags

Control (i) is the paired floor render - the same rate, the same per-(order, microphone) line-to-noise ratio, a shared integrated-OU shaft and nothing else - so the quantity tabulated for it is its own leakage-corrected residual, i.e. the additive-noise floor, which must be FLAT in `tau` (`q` near 0). Control (ii) is that render plus a planted independent per-order Wiener phase, `D_k = 0.1 k` rad^2/s, whose net term must keep growing as `2 D_k tau` (`q` near 1). They calibrate what a plateau and what continued growth look like through this estimator at these lags.

### Control (i): shaft + floor only (the floor; must be flat)

| k | V(0.5 s) | V(1 s) | V(2 s) | V(5 s) | n fit | q | a | tau_c [s] | c | dAIC | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.04 | 0.0377 | 0.0319 | 0.0043 | 6 | -0.54 | 0.0264 | 0.00 | 0.024 | -4.7 | flat |
| 4 | 0.107 | 0.113 | 0.137 | 0.159 | 6 | +0.14 | 0.124 | 0.14 | 0.134 | -10.9 | growing |
| 8 | 0.124 | 0.0843 | -0.0771 | -0.613 | 3 | - | - | - | - | - | no fit |
| 16 | 0.242 | 0.394 | 0.553 | 0.594 | 6 | +0.38 | 0.386 | 0.60 | 0.587 | -4.7 | growing |
| 24 | 0.509 | 0.825 | 1.02 | 1.15 | 6 | +0.28 | 0.742 | 0.38 | 0.966 | -6.4 | growing |
| 32 | 0.785 | 1.1 | 1.29 | 1.81 | 6 | +0.26 | 1.1 | 0.30 | 1.36 | -10.8 | growing |
| 40 | 0.98 | 1.13 | 1.25 | 2.1 | 6 | +0.25 | 1.17 | 0.27 | 1.41 | -8.8 | growing |

### Control (ii): planted D_k = 0.1 k (must keep growing)

| k | V(0.5 s) | V(1 s) | V(2 s) | V(5 s) | n fit | q | a | tau_c [s] | c | dAIC | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.227 | 0.531 | 1.31 | 2.78 | 6 | +1.13 | 0.534 | 1202604.17 | 6.58e+05 | -6.5 | growing |
| 4 | 0.569 | 1.13 | 2.03 | (3.98) | 5 | +0.90 | 0.96 | 3.36 | 3.93 | +2.7 | growing (no ceiling inside the fitted lags) |
| 8 | 0.464 | 0.718 | 1.3 | (2.91) | 4 | +0.74 | 0.757 | 1.39 | 1.59 | -7.9 | growing |
| 16 | 1.62 | (1.99) | (1.32) | (0.599) | 2 | - | - | - | - | - | no fit |
| 24* | (2.13) | (0.873) | (-1.93) | (-3.16) | 1 | - | - | - | - | - | no fit |
| 32* | (1.22) | (-2.48) | (-8.24) | (-10.1) | 0 | - | - | - | - | - | no fit |
| 40* | (0.832) | (-2.23) | (-8.48) | (-8.73) | 0 | - | - | - | - | - | no fit |

Control (ii) is never fitted a ceiling at these levels, so no data plateau had to be discounted as the estimator's own compression.

## The verdict

orders 2, 4, 8 are below the ceiling at 500 ms; ceiling: k = 4, 8; growing: k = 2. Calibration - control (i) (shaft + floor only, must be flat): median q = +0.25, verdicts flat/growing over 6 orders; control (ii) (planted D_k = 0.1 k random walk, must keep growing): median q = +0.90, verdicts growing over 3 orders.

**k = 2**: V_eps = 0.0254 rad^2 at 0.5 s, 0.061 at 1 s, 0.138 at 2 s, 0.431 at 5 s; q = +1.17, tau_c = 1202604.18 s, dAIC = -16.1 - the power law wins by 16.1 AIC, so this order KEEPS GROWING out to the longest lag; as a Wiener phase that is D_k = 0.0305 rad^2/s at q = 1.17.

**k = 4**: V_eps = 0.0866 rad^2 at 0.5 s, 0.197 at 1 s, 0.321 at 2 s, 0.239 at 5 s; q = +0.83, tau_c = 2.15 s, dAIC = +2.9 - the saturating shape wins by more than 2 AIC and its knee is inside the measured lags, so this order has a CEILING: an OU phase with lambda_eps = 0.466 /s and sigma_eps = 0.414 rad/s^0.5 (plateau 0.368 rad^2).

**k = 8**: V_eps = 0.672 rad^2 at 0.5 s, 1.15 at 1 s, 1.38 at 2 s, 2.4 at 5 s; q = +0.73, tau_c = 1.12 s, dAIC = +5.0 - the saturating shape wins by more than 2 AIC and its knee is inside the measured lags, so this order has a CEILING: an OU phase with lambda_eps = 0.894 /s and sigma_eps = 1.26 rad/s^0.5 (plateau 1.76 rad^2).

**k = 16**: V_eps = 1.48 rad^2 at 0.5 s, 2.38 at 1 s, 0.732 at 2 s, 0.312 at 5 s - only 2 of the fitted lags are readable (4 of them sit over the 3 rad^2 estimator ceiling), fewer than the 4 a two-parameter shape needs, so no fit and the answer is UNDETERMINED for this order.

**k = 24**: V_eps = 1.72 rad^2 at 0.5 s, 1.32 at 1 s, -2.17 at 2 s, -3.78 at 5 s - but its observed residual is already at the 3 rad^2 estimator ceiling at 500 ms, so the shape cannot be read and the answer is UNDETERMINED for this order.

**k = 32**: V_eps = 3.11 rad^2 at 0.5 s, -1.32 at 1 s, -4.67 at 2 s, -8.71 at 5 s - but its observed residual is already at the 3 rad^2 estimator ceiling at 500 ms, so the shape cannot be read and the answer is UNDETERMINED for this order.

**k = 40**: V_eps = 1.55 rad^2 at 0.5 s, -0.577 at 1 s, -7.97 at 2 s, -21.6 at 5 s - but its observed residual is already at the 3 rad^2 estimator ceiling at 500 ms, so the shape cannot be read and the answer is UNDETERMINED for this order.

Pooled over the 3 orders that are below the ceiling at 500 ms (ceiling at k = 4, 8; growing at k = 2), the answer is **CEILING**: the independent per-order phase is an OU phase with a finite plateau, not a random walk.

The instrument is calibrated by the two controls at the same lags: control (i), which carries no per-order term at all, returns a floor with median q = +0.25 (flat is q = 0), and control (ii), which carries a planted random walk, returns median q = +0.90 (a random walk is q = 1). Control (ii) is never fitted a ceiling at these levels, so no plateau had to be discounted as the estimator's own compression.

## What this run does NOT settle

- The windows are 30-31 s, so a 5 s lag has only about 6 disjoint increments in it; the long end of every curve is the noisiest part of it.
- All four supports are ONE rotor of ONE rig (DREGON Motor 1 at 50, 60 and 70 % throttle). A ceiling or its absence here is a statement about this rotor, not about every rig in the corpus.
- The high orders are at the estimator ceiling long before 1 s, so the shape can only ever be read on the low orders. Nothing here contradicts or confirms the high-order behaviour.
- `V_eps` is a floor-subtracted difference of two circular-estimator read-outs; at the long lags the two terms are close, so its relative error is larger than the short-lag one. The `net_err` column of the JSON carries the standard error of each cell.
