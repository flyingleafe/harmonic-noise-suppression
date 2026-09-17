# Order decoherence on the DREGON bench - findings

Source: `scripts/noise_v2_order_decoherence.py`. Numbers: `results/noise_v2/decoherence/decoherence.json`. Figures: `docs/explainers/noise-model-v2-plan/decoherence_*.png`.

The model under test splits the phase increment of order `k` over a lag `tau` into a SHARED part and an INDEPENDENT part:

    dphi_k(t; tau) = k * dtheta(t; tau) + eps_k(t; tau)

`dtheta` is everything that scales with the order - the shaft speed error and any propagation term tied to it. `eps_k` is what the tied model cannot represent. `V_shared(tau) = Var[dtheta]` and `V_eps(k, tau) = Var[eps_k]` are measured on 12 single-motor bench recordings, 8 microphones, orders 1-40, on a 200.5 Hz phase grid.

## The numbers

| k | V_eps(50 ms) | V_eps(500 ms) | Var[k dtheta] (50 ms) | ratio |
|---|---|---|---|---|
| 1 | -0.01804 | 0.002418 | 0.0007861 | -22.9 |
| 2 | 0.006432 | -0.04142 | 0.003144 | 2.05 |
| 3 | 0.1604 | 0.1564 | 0.007075 | 22.7 |
| 4 | 0.05412 | 0.1093 | 0.01258 | 4.3 |
| 5 | 0.548 | 0.4646 | 0.01965 | 27.9 |
| 6 | -0.007225 | 0.1664 | 0.0283 | -0.255 |
| 7 | 0.3753 | 0.2645 | 0.03852 | 9.74 |
| 8 | 0.01306 | 0.7434 | 0.05031 | 0.26 |
| 9 | 0.4682 | 0.3468 | 0.06367 | 7.35 |
| 10 | 0.1671 | 0.6495 | 0.07861 | 2.13 |
| 11 | 0.2468 | 0.7536 | 0.09512 | 2.59 |
| 12 | 0.1496 | 1.599 | 0.1132 | 1.32 |
| 13 | 0.2553 | 1.017 | 0.1329 | 1.92 |
| 14 | 0.129 | 1.62 | 0.1541 | 0.837 |
| 15 | 0.9982 | 1.75 | 0.1769 | 5.64 |
| 16 | 0.1986 | 2.082 | 0.2012 | 0.987 |
| 17 | 1.166 | 2.257 | 0.2272 | 5.13 |
| 18 | 0.3888 | 2.415 | 0.2547 | 1.53 |
| 19 | 1.316 | 2.353 | 0.2838 | 4.64 |
| 20 | 0.3826 | 2.409 | 0.3144 | 1.22 |
| 21 | 0.3358 | 2.059 | 0.3467 | 0.969 |
| 22 | 0.4807 | 2.585 | 0.3805 | 1.26 |
| 23 | 1.838 | 3.986 | 0.4159 | 4.42 |
| 24 | 0.5834 | 2.813 | 0.4528 | 1.29 |
| 25 | 1.441 | 3.108 | 0.4913 | 2.93 |
| 26 | 0.7778 | 3.118 | 0.5314 | 1.46 |
| 27 | 1.972 | 5.096 | 0.5731 | 3.44 |
| 28 | 0.8705 | 2.805 | 0.6163 | 1.41 |
| 29 | 1.447 | 2.532 | 0.6611 | 2.19 |
| 30 | 0.9326 | 3.298 | 0.7075 | 1.32 |
| 31 | 1.57 | 6.959 | 0.7555 | 2.08 |
| 32 | 0.9003 | 2.864 | 0.805 | 1.12 |
| 33 | 1.753 | 6.187 | 0.8561 | 2.05 |
| 34 | 1.095 | 2.869 | 0.9087 | 1.2 |
| 35 | 3.206 | 15.95 | 0.963 | 3.33 |
| 36 | 0.9624 | 3.437 | 1.019 | 0.945 |
| 37 | 1.401 | 3.915 | 1.076 | 1.3 |
| 38 | 1.274 | 2.815 | 1.135 | 1.12 |
| 39 | 1.012 | 0.9699 | 1.196 | 0.847 |
| 40 | 1.333 | 4.342 | 1.258 | 1.06 |

`V_eps` is the net value: observed, minus the measured leakage of the shaft estimate, minus the additive-noise floor of the paired control. `Var[k dtheta]` is `k^2 V_shared(50 ms)` with `V_shared(50 ms) = 7.861e-04` rad^2 (`V_shared(500 ms) = 1.667e-02` rad^2). `ratio` is `V_eps(50 ms) / Var[k dtheta](50 ms)` - how much of the per-order phase noise the tied model misses. A dash is an order with no gated cell at that lag; a non-positive entry is an order whose residual sits AT the additive-noise floor, so no independent term is resolved there.

## The power law

`V_eps(k, tau) = a k^p tau^q` by log-log least squares over k in [4, 40] and tau in [10, 500] ms, 91 resolvable cells, rms log residual 0.637. A cell is resolvable when its order clears the gate on at least 5 of the 8 microphones in most recordings, its observed residual is below the 3.0 rad^2 estimator ceiling, and its recording-mean net value stands 2 standard errors ACROSS recordings clear of zero:

- **a = 0.09148** (95 % CI 0.04659 - 0.3827)
- **p = 1.082** (95 % CI 0.635 - 1.436)
- **q = 0.434** (95 % CI 0.364 - 0.741)

The CI is a 2000-draw bootstrap over the 12 recordings. Without the floor subtraction the same fit gives a = 0.3061, p = 0.843, q = 0.336; the floor is flat in `tau`, so leaving it in flattens `q` and steepens `p`.

The rotor has two blades, so its sound sits in the EVEN orders and the odd orders are what blade-to-blade dissimilarity leaves over - and the two families do not carry the same independent phase noise. Each resolvable odd order carries 3.7 times the independent phase noise of the mean of its two even neighbours at 50 ms (10 such orders, median 0.773 against 0.229 rad^2), and 2.2 times at 500 ms. That split, not scatter, is a good part of the 0.64 rms log residual of the joint fit. Fitted separately:

- **even orders** (43 cells, rms 0.255): a = 0.03411 (0.01678 - 0.1326), p = 1.709 (1.129 - 2.245), q = 0.884 (0.632 - 1.348)

- **odd orders** (48 cells, rms 0.683): a = 0.202 (0.04808 - 0.9457), p = 0.900 (0.248 - 1.438), q = 0.464 (0.326 - 0.672)

## Is the residual independent?

Cross-order correlation of `eps_k` at 50 ms, over 630 order pairs (rows de-rotated with one shaft estimate, columns with a disjoint one, so the shaft-estimate error cannot manufacture a correlation):

- mean off-diagonal `rho` = **0.028** (mean |rho| = 0.038)
- same parity (odd-odd and even-even) = 0.036, cross parity = 0.019, `|k-l| = 2` = 0.060

**Verdict:** independent across orders: mean |rho| = 0.038 is inside the 0.10 bar and the odd/even blocks do not separate, so one Wiener phase per order with no cross-order structure is the right generator.

Cross-microphone correlation of `eps_k` at the same order and lag, averaged over the 28 microphone pairs: mean over k = 4-24 is **0.207**.

**Verdict:** mixed: cross-microphone rho = 0.207 sits between the independent bar (0.10) and the shared bar (0.50), so part of the per-order phase noise is source-side and part is path-side.

## Controls

Every control is rendered at the rate, the order count and the per-(order, microphone) line-to-noise ratio of its own recording and is pushed through the identical pipeline. Control (i) carries ONLY a shared integrated-OU shaft (sigma_nu = 1.77 rad/s, lambda = 5.48 1/s, the acoustic bench fit of `results/noise_v2/shaft/findings.md`) plus a white floor; control (ii) is that same render plus an independent per-order Wiener with `D_k = 0.1 k` rad^2/s, and is corrected with control (i) as its own floor, so the pair is internally consistent.

Both controls are run TWICE: once on the shaft each recording actually has (`sigma_nu` fitted per recording from its own measured `V_shared` at fixed `lambda = 5.48`, which comes out at 0.31 - 0.65 rad/s, median 0.45), and once on the literally mandated `sigma_nu = 1.77`. The MATCHED pair is the headline and the matched floor is what is subtracted from the data, because the published bench fit absorbed the per-order decoherence into the shared term and so over-states the real shared phase noise by about a factor three - and a control whose shaft is three times too large is not a valid floor: its low-order lines are broader, its unwrapped low-order phase slips more, and its shaft-estimate error comes out several times larger, which above order 20 drives the leakage correction past the observed residual. The nominal pair is reported alongside, and its behaviour is itself the evidence for that statement.

- **Control (i) returns the floor.** On the matched shaft its `V_eps` is flat in `tau` to within 18% between 50 and 500 ms - it carries no `tau` dependence at all, which is what an additive-noise floor must look like - and its level is 0.96 times the analytic phase floor `2 log(1 + 1/SNR)` of the measured line-to-noise ratio, cell by cell. Below 50 ms the floor is genuinely smaller, because the noise inside the band is correlated over about 1/(2 band) = 16 ms and the two samples of a short increment share part of it. No independent term is invented where none exists. The floor's own log-log slope against `k` is +0.04 - no `k^2` shape, so the leakage correction has not left a shaft-shaped term behind for the subtraction to remove from the data. On the nominal shaft the same three read-outs are 21%, 0.94 and -0.44, and its floor goes NEGATIVE at 33 of 37 orders against 32 on the matched shaft - that is the out-of-range behaviour, in one number.
- **Control (ii) is recovered.** Over the 162 resolvable cells of the fit domain the recovered `V_eps` is 0.67 times the planted `2 D_k tau` (interquartile 0.47 - 0.86), and the power-law fit of the recovered values gives a = 0.2923 (planted 0.20), p = 0.850 (planted 1), q = 1.094 (planted 1). On the nominal shaft the recovery ratio is 1.23 over 157 cells.
- **The subtraction returns zero when there is nothing to find.** A SECOND matched floor render - same shaft amplitude and same per-cell SNR, independent noise, blade phases and shaft realisation - put through the same subtraction against the first gives 1 resolvable cells over the fit domain against 91 for the bench, and a median |net| of 0.03754 rad^2 against 0.7485 rad^2. The residual the bench shows is not the subtraction's own noise.

| k | tau [ms] | planted 2 D_k tau | recovered | ratio | floor (ctrl i) | floor analytic |
|---|---|---|---|---|---|---|
| 2 | 50 | 0.02 | 0.008855 | 0.443 | 0.2109 | 0.1794 |
| 4 | 50 | 0.04 | 0.0531 | 1.33 | 0.1852 | 0.2457 |
| 8 | 50 | 0.08 | 0.07832 | 0.979 | 0.26 | 0.2584 |
| 16 | 50 | 0.16 | 0.1032 | 0.645 | 0.2357 | 0.2573 |
| 24 | 50 | 0.24 | 0.1764 | 0.735 | 0.3721 | 0.5449 |
| 32 | 50 | 0.32 | 0.19 | 0.594 | 0.4603 | 0.7396 |
| 40 | 50 | 0.4 | 0.06594 | 0.165 | 0.3512 | 0.9395 |
| 2 | 500 | 0.2 | 0.1394 | 0.697 | 0.2566 | 0.1794 |
| 4 | 500 | 0.4 | 0.3453 | 0.863 | 0.1658 | 0.2457 |
| 8 | 500 | 0.8 | 0.9098 | 1.14 | -0.2762 | 0.2584 |
| 16 | 500 | 1.6 | 1.22 | 0.762 | -0.8787 | 0.2573 |
| 24 | 500 | 2.4 | 1.278 | 0.533 | -2.27 | 0.5449 |
| 32 | 500 | 3.2 | 0.6653 | 0.208 | -4.436 | 0.7396 |
| 40 | 500 | 4 | -2.22 | -0.555 | -7.647 | 0.9395 |

Control (ii) plants a per-order Wiener that is SHARED across microphones, so its cross-microphone `rho` is high (0.28 over k = 4-24) while control (i), whose only per-order term is independent sensor noise, gives 0.00. The two controls bracket the cross-microphone test, and the measured bench value is read against them.

## What this means

The independent per-order term is not a correction, it is the dominant part of the harmonic phase noise on this bench. At 50 ms the tied model predicts `k^2 V_shared = 0.0503` rad^2 at order 8 and 1.26 rad^2 at order 40, while the measured independent term is 0.0131 and 1.33 rad^2 - a factor 0 and 1 more than the shared term carries, and the shared term only catches up above order 6. It does scale with the order, but far more slowly than the `k^2` of a tied phase: `V_eps = 0.0915 k^1.08 tau^0.43` rad^2, with p = 1.08 (95 % CI 0.64 - 1.44) and q = 0.43 (95 % CI 0.36 - 0.74), i.e. close to a Wiener phase per order with a diffusion roughly proportional to `k`, `D_k = 0.0457 k^1.08` rad^2/s. At 500 ms the term has grown to 0.743 rad^2 at order 8 and 4.34 rad^2 at order 40, which is decoherence, not jitter: a harmonic that has lost 4.34 rad^2 of phase over half a second cannot be rendered from the shaft label alone. The size is not one number per order either: the odd orders of this two-bladed rotor carry 3.7 times the independent phase noise of the even orders, so a generator needs the weak, dissimilarity-driven family to be noisier than the blade-passing family. It is partly source-side and partly path-side: cross-microphone rho = 0.21, so about that fraction of the per-order variance belongs to the rotor and the rest to each path.

## How the numbers were made

- **Window.** Eleven of the twelve files are 25 s long and the motor runs for only about 11 s of that; `motor_Motor1_70` runs for 31 s. The window is the longest stretch of the level track within 6 dB of its 90th percentile, trimmed by 0.5 s at each end and capped at 24 s. A fixed 24 s window from t = 3 s is half room noise, and the uniform phase of the silent half alone produces a linearly growing increment variance at every order.

- **Demodulation.** Constant carrier per recording, refined with `residual_frequency` at orders (10, 40) so that every line up to 40 sits within about 1 Hz of band centre; band `+-0.4 * rate`, decimated to 200.5 Hz. Audio is anti-aliased and decimated by 4 first (Nyquist 5512 Hz, the highest analysed line is 3.6 kHz); this reproduces the full-rate SNR of every cell to three decimals and makes the ladder four times cheaper.

- **Gate.** A cell is used when the peak of its `|z_k|^2` spectrum stands 10 dB above the median of the same spectrum inside the band; weights are that ratio. An order enters the fit only if it clears that bar on at least 5 of the 8 microphones: a per-cell gate on its own selects the luckiest channel of a marginal order, whose SNR reads high because its own noise fluctuated low, and that under-states the cell's floor. The weak odd orders of this two-bladed rotor are what the bar removes.

- **Why the variance is circular, not unwrapped.** In a `+-0.4 rate` band the line-to-noise power ratio of a high order on this bench is a few dB, so `np.unwrap` takes the wrong branch a few times per hundred samples. Each slip adds `(2 pi)^2` and the slip count random-walks, so the unwrapped variance grows linearly in `tau` whatever the phase does: it reads 16 rad^2 at order 1 and 500 ms, for a line 0.1 Hz wide. The headline variance is `-2 log |gamma|` of the amplitude-weighted circular mean of the de-rotated increment phasor, which cannot slip. `v_unwrap` in the JSON is the mandated unwrapped estimator, kept so the size of the artefact is on record.

- **Leakage.** `dtheta_hat` is a weighted sum of the same noisy increments, so its error `e` reappears in every residual as `k^2 Var[e]`. `Var[e]` is measured: the shaft band is split into 3 disjoint order groups, the three independent shaft estimates are differenced pairwise, and `Var[e]` follows from the three pairwise variances. Orders inside the shaft band are de-rotated leave-group-out. The correction is 44% of the observed residual at order 40 and 50 ms.

- **Leave-band-out.** The headline shaft estimate already uses only k <= 10: above that the per-sample phase is noise-dominated and its unwrapped phase slips. The literal all-gated-orders estimate of step 4 is reported alongside as `v_all_orders` and `v_shaft_all_orders`; it gives `V_shared(50 ms) = 6.468e-04` rad^2 against 7.861e-04 for the low band, because `k^2` in the normal equations hands most of the weight to orders whose phase is noise.

- **Ceiling.** `-2 log |gamma|` stops being informative once the residual has decohered; cells with `V_obs > 3.0` rad^2 are reported but excluded from the fit. Control (ii) fixes that bar: the planted `D_k` is returned wherever `V_obs` is under it and is lost above it.

## Per recording

| recording | rate [rev/s] | survey [rev/s] | window [s] | start [s] | band [Hz] | cells | V_shared(50 ms) | V_eps(k=8, 50 ms) | median slip [%] |
|---|---|---|---|---|---|---|---|---|---|
| motor_Motor1_70 | 68.501 | 68.30 | 24.0 | 4.8 | 27.4 | 219 | 6.673e-04 | 0.0557 | 1.64 |
| motor_Motor1_80 | 78.245 | 78.06 | 10.7 | 4.5 | 31.3 | 251 | 1.088e-03 | 0.091 | 1.40 |
| motor_Motor1_90 | 87.967 | 88.18 | 11.8 | 4.9 | 35.2 | 245 | 8.393e-04 | 0.169 | 1.23 |
| motor_Motor2_70 | 67.692 | 67.52 | 10.8 | 4.3 | 27.1 | 290 | 3.816e-04 | -0.0382 | 1.76 |
| motor_Motor2_80 | 77.337 | 77.18 | 11.3 | 4.7 | 30.9 | 255 | 7.647e-04 | -0.0407 | 1.58 |
| motor_Motor2_90 | 86.943 | 86.70 | 10.7 | 5.8 | 34.8 | 259 | 7.362e-04 | 0.0602 | 1.41 |
| motor_Motor3_70 | 68.697 | 68.56 | 10.8 | 5.4 | 27.5 | 259 | 5.244e-04 | 0.0399 | 1.82 |
| motor_Motor3_80 | 78.641 | 78.82 | 11.3 | 4.5 | 31.5 | 269 | 7.134e-04 | 0.0426 | 1.47 |
| motor_Motor3_90 | 88.508 | 88.24 | 10.8 | 5.3 | 35.4 | 239 | 1.350e-03 | 0.0401 | 1.41 |
| motor_Motor4_70 | 69.540 | 69.31 | 10.9 | 4.3 | 27.8 | 300 | 4.685e-04 | 0.114 | 1.82 |
| motor_Motor4_80 | 79.637 | 79.38 | 10.7 | 5.8 | 31.9 | 292 | 1.141e-03 | -0.266 | 1.54 |
| motor_Motor4_90 | 89.409 | 89.14 | 10.6 | 6.0 | 35.8 | 303 | 7.597e-04 | -0.11 | 1.66 |
