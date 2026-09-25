# Noise model v3: the wander hyperparameters, measured on the real windows

`scripts/noise_v3_measure_wander.py` at `8f07160ce150`; core `src/experiments/noise_model/wander.py`. The measurement of `docs/explainers/noise-model-v3-wander.qmd` section 3.3a. No model is evaluated and nothing is fitted to a likelihood.

**What is measured.** Per window and per block (0.25 / 0.5 / 1.0 s) on the 2048/512 flight front end (16 kHz, periodic Hann; R4's 8192-point frames are longer than a block): the ORDER-TRACKED line level of every rotor order -- the 3 bins nearest the frame's own label carrier `k f_r(n)` (98-100 % of a Hann line at any sub-bin offset), block-averaged, minus the local floor (q25 of the cells 3-12 bins either side) -- and the comb-masked floor level of every band (every order +-1 bin, lines >= 10 dB +-3 bins). The model's speed laws (line `(f/80)^2`, floor `mean_i (f_i/80)^2`, the prior centres every short-span pool is pinned at) are removed per block. Everything is the MIC MEAN of the dB levels.

**Centring.** The v3 fit shares `p_ik` and the floor across its pool, so its block latents carry EVERY departure from the rig mean at the block's speed -- slow drift inside a flight and the offsets between windows alike. The primary estimate therefore centres each line (each floor band) on its mean over all windows of the same regime (DREGON: flight; Michael's: cruise, standby) and takes lag products inside windows only. The `within window` rows centre every window on its own mean: the wander inside a flight alone, blind to anything slower than the window.

**Estimator.** The explainer's three lines, with the centring put back: under an OU of `(sigma^2, rho = e^{-T_b/tau})` plus independent block noise, every pooled centred lag product has an exact expectation linear in `sigma^2` (`wander.fit_ou`); `sigma^2` solves lag 0 and `tau` the lag-1 product, or the lags 1-4 products (both reported). `sigma_d` is the lag-0 CROSS moment of two distinct lines of the same rotor -- the explainer's 'covariance between two orders of one rotor in the same block', no noise term -- with `tau_d` over lags 1-4 (a pair's lag-1 product alone is too small against its scatter); `sigma_v` from the auto moments minus the fitted rotor-common part; `sigma_u` likewise from cross moments of two distinct floor bands, `sigma_uj` the rest. The explainer's raw arithmetic (`Var_b(y) - s^2`, `Cov_1 / sigma^2`, within-window, no centring correction) is reported as `naive`: on 4-8 s windows (8-16 blocks) it is biased low in both sigma and tau whenever tau is not short against the window (synthetic check in `tests/experiments/test_noise_model_wander.py`).

**Block noise.** The explainer's `s^2 = (10/ln10)^2 psi_1(n_eff)` with `n_eff = frames x bins x lineFrac^2` assumes independent exponential cells. On this front end the frames overlap 75 % and adjacent Hann bins correlate, so a 15 x 3 rectangle of Gaussian-noise cells carries the information of ~45/3.1 independent ones; and a narrow line of steady amplitude -- the model's line -- has no exponential scatter of its own. The estimates therefore use the MEASURED block noise: for lines, the variance of the block mean from the scatter of the frames inside the block (lag 0-3 autocovariance, the overlap's reach, centring-corrected); for the floor, whose cells are Gaussian noise, the exponential law at the exact correlated-cell count of the surviving cells, times the measured mic-coherence count for the mic mean. The explainer's count is a robustness row.

## Windows

| rig | set | windows | duration s | carrier rev/s (min-max of window means) | recordings |
| --- | --- | --- | --- | --- | --- |
| dregon | dregon_score | 5 | 4 | 79.4-80.9 | 5 |
| dregon | dregon_fit | 5 | 8 | 80.5-81.2 | 5 |
| dregon | dregon_extra | 40 | 4 | 78.7-82.4 | 6 |
| michaels | michaels_cruise | 8 | 8 | 80.3-81.0 | 1 |
| michaels | michaels_standby | 3 | 4 | 33.6-35.6 | 1 |

## The per-line sd on every valid track (the contract's sigma_v, schema 2)

**Why it changed.** Schema 1 read `sigma_v` off the RESOLVABLE lines alone (block prominence >= 6 dB in >= 80 % of the blocks, section 'Resolvable lines'). The intermittency diagnostic (`results/noise_v3/intermittency/findings.md`, `93a8c14111b8`) looked at the tracks that rule leaves out and found them a Gaussian in dB of LARGER sd (on DREGON growing with the order) -- dregon: 767 intermittent and 17983 always-under tracks against 44 resolvable; the intermittent tracks' median own block sd 1.48 dB at k 1-2, 1.70 dB at k 3-8, 2.07 dB at k 9-24, 2.62 dB at k 25-60, 4.35 dB at k 61+; the forward model with the floor held fitted 5.03 dB to them (the resolvable lines' 1.48 dB as they are); lag-1 autocorrelation 0.23 observed against 0.15 for that model; michaels: 262 intermittent and 5236 always-under tracks against 167 resolvable; the intermittent tracks' median own block sd 1.81 dB at k 1-2, 4.22 dB at k 3-8, 3.36 dB at k 9-24, 2.24 dB at k 25-60, 4.82 dB at k 61+; the forward model with the floor held fitted 5.81 dB to them (the resolvable lines' 1.98 dB as they are); lag-1 autocorrelation 0.38 observed against 0.44 for that model. 'Prominent in >= 80 % of the blocks' is a selection on low scatter: it keeps the steady lines, and a sd read off them is biased low for every line the v3 fit carries.

**Estimator.** Every classified track -- valid (in band, on the grid) in >= 80 % of its window's 0.5 s blocks, resolvable, intermittent or always-under -- is read through its block prominence with the diagnostic's test (b) forward model (`wander.ProminenceModel`): the line over its local floor is `mu + d + v + e + o` dB, `v` the per-line OU being measured, `d` the rotor-common OU held at the resolvable lines' `(sigma_d, tau_d)` (section 'The estimates'), `e` white block noise at the measured median line `s^2`, `o` the speed law minus the track's MEASURED local floor deviation (the line held in absolute level, the floor as it was); the floor cells add a line-free cell-over-floor ratio (every block of the always-under tracks with median < 1 dB), and `10 log10(10^(line/10) + ratio)` is the prominence. Each track's `mu` is matched to its median; each track is centred on its own mean and the model's draws alike, so `sigma` is the OU's STATIONARY sd. A track weighs the square of its line fraction at its median prominence, `(1 - 10^(-P/10))^2` -- the slope of the prominence in the line level: a track at 0 dB carries no line and weighs 0, one at 3 dB a quarter. `sigma_v` solves the weighted lag-0 sum and `tau_v` the lags 1-4 (`wander.fit_prominence_ou`, common random numbers over a (sigma, tau) table), per order group and pooled. A group keeps its own `tau_v` when the window bootstrap pins it (95 % / 5 % under 4x, not on the grid edge); otherwise its `sigma_v` is refitted at the pooled `tau_v`. Intervals: window bootstrap (within regime), 5-95 %, with `d` held at its point estimate.

Pooled, old beside new (sd dB, tau s):

| rig | schema 1: sigma_v (resolvable lines, rig-centred) | tau_v | tracks res / inter / under (weighted) | schema 2: sigma_v pooled | tau_v | same estimator, clear tracks only (median >= 3 dB) | same estimator, resolvable tracks only | sigma_d held / tau_d | sigma_total = sqrt(v^2 + d^2) | lag-1 autocorrelation observed / model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon | 1.15 [0.00, 1.54] | 0.91 [0.29, 26.87] | 44 / 767 / 17983 (15185) | **3.66** [3.25, 3.98] | 0.75 [0.59, 0.91] | 3.60 [3.11, 4.02] (1457 tracks) | 3.00 [1.91, 3.89] | 0.94 / 0.30 | 3.78 | 0.17 / 0.18 |
| michaels | 1.91 [1.28, 2.47] | 1.26 [0.84, 1.85] | 167 / 262 / 5236 (3026) | **3.78** [3.22, 4.41] | 1.15 [0.57, 1.85] | 3.68 [3.09, 4.38] (466 tracks) | 2.34 [1.80, 2.98] | 0.52 / 0.58 | 3.81 | 0.40 / 0.45 |

**Old to new in two steps:** dregon 1.15 (schema 1, 32 resolvable line tracks) -> 3.00 (all-track estimator, resolvable tracks) -> 3.66 dB (every valid track); michaels 1.91 (schema 1, 30 dominant line tracks) -> 2.34 (all-track estimator, resolvable tracks) -> 3.78 dB (every valid track). The first step is everything the two estimators do differently on the same tracks: schema 1 keeps only the line set's tracks with a median measured block sd <= 2 dB, treats blocks under 3 dB prominence or noisier than 4 dB as missing, reads the line level (cells minus floor, mic mean of dB) and centres on the regime mean; schema 2 keeps every valid block, reads the prominence with the floor modelled and centres per track. The second step is the resolvable rule alone.

By order group -- the contract's `sigma_v_db_by_order` / `tau_v_s_by_order`. The schema-1 column is the resolvable lines' own auto sd per group (section 'Breakdown', all of a line's wander, `d` included; n/a where no resolvable line was used).

| rig | orders | tracks res / inter / under (weighted) | schema 1: auto sd | schema 2: sigma_v | tau_v (source) | own tau: sigma_v / tau_v [tau 5-95 %] | clear tracks only, pooled tau: sigma_v (tracks) | lag-1 autocorrelation observed / model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon | 1-2 | 8 / 85 / 107 (200) | 0.82 | **0.00** [0.00, 0.00] | 0.75 [0.59, 0.91] (pooled) | 0.00 / 0.11 [0.05, 7.99] | 0.00 [0.00, 0.00] (177) | 0.16 / 0.17 |
| dregon | 3-8 | 0 / 32 / 1168 (1163) | n/a | **0.00** [0.00, 0.99] | 0.75 [0.59, 0.91] (pooled) | 0.00 / 0.13 [0.05, 27.13] | 0.00 [0.00, 0.72] (164) | 0.08 / 0.09 |
| dregon | 9-24 | 22 / 308 / 2870 (2601) | 1.54 | **2.98** [2.70, 3.25] | 0.79 [0.58, 1.07] (group) | 2.98 / 0.79 [0.58, 1.07] | 2.63 [2.35, 2.92] (638) | 0.20 / 0.18 |
| dregon | 25-60 | 6 / 186 / 7008 (5701) | 1.01 | **3.91** [3.70, 4.12] | 0.62 [0.51, 0.77] (group) | 3.91 / 0.62 [0.51, 0.77] | 3.81 [3.49, 4.08] (342) | 0.13 / 0.14 |
| dregon | 61+ | 8 / 156 / 6830 (5520) | 8.91 | **7.18** [6.43, 7.88] | 0.87 [0.60, 1.17] (group) | 7.18 / 0.87 [0.60, 1.17] | 7.21 [6.43, 7.87] (136) | 0.20 / 0.23 |
| michaels | 1-2 | 32 / 2 / 0 (34) | 0.47 | **0.32** [0.00, 1.05] | 1.15 [0.57, 1.85] (pooled) | 0.62 / 16.91 [0.05, 50.00] | 0.32 [0.00, 1.05] (34) | 0.42 / 0.38 |
| michaels | 3-8 | 63 / 55 / 142 (189) | 3.08 | **4.17** [3.30, 5.16] | 1.15 [0.57, 1.85] (pooled) | 16.00 / 50.00 [0.56, 50.00] sigma_max | 4.13 [3.23, 5.14] (121) | 0.45 / 0.47 |
| michaels | 9-24 | 54 / 145 / 505 (429) | 1.35 | **3.79** [3.30, 4.19] | 0.86 [0.42, 1.35] (group) | 3.79 / 0.86 [0.42, 1.35] | 3.77 [3.24, 4.27] (207) | 0.35 / 0.42 |
| michaels | 25-60 | 0 / 22 / 1562 (904) | n/a | **3.61** [3.26, 4.06] | 1.06 [0.67, 1.65] (group) | 3.61 / 1.06 [0.67, 1.65] | 3.06 [2.53, 3.47] (55) | 0.28 / 0.32 |
| michaels | 61+ | 18 / 38 / 3027 (1470) | n/a | **5.69** [4.30, 6.99] | 1.15 [0.57, 1.85] (pooled) | 6.96 / 2.41 [1.40, 6.27] | 5.67 [4.34, 6.99] (49) | 0.47 / 0.29 |

### Against the intermittency diagnostic

The diagnostic's own block sd is the RAW scatter of the intermittent tracks' block prominence around their mean -- block noise, the floor's deviation and `d` included, compressed where the line sits under its floor -- so it checks the ORDERING of `sigma_v` over the groups, not its level. Its like-for-like number is the forward model it fitted to the same tracks (one OU for the whole line, `d + v`, at the resolvable lines' `tau_total`, the floor held), against schema 2's `sqrt(sigma_v^2 + sigma_d^2)`. Schema 2's lag-1 is over every weighted track, the diagnostic's over its intermittent tracks.

| rig | orders | diagnostic: intermittent tracks | diagnostic: their median own block sd | schema 1: auto sd (resolvable) | schema 2: sigma_v [5-95 %] | schema 2: sqrt(sigma_v^2 + sigma_d^2) |
| --- | --- | --- | --- | --- | --- | --- |
| dregon | 1-2 | 86 | 1.48 | 0.82 | 0.00 [0.00, 0.00] | 0.94 |
| dregon | 3-8 | 32 | 1.70 | n/a | 0.00 [0.00, 0.99] | 0.94 |
| dregon | 9-24 | 308 | 2.07 | 1.54 | 2.98 [2.70, 3.25] | 3.12 |
| dregon | 25-60 | 185 | 2.62 | 1.01 | 3.91 [3.70, 4.12] | 4.02 |
| dregon | 61+ | 156 | 4.35 | 8.91 | 7.18 [6.43, 7.88] | 7.24 |
| michaels | 1-2 | 2 | 1.81 | 0.47 | 0.32 [0.00, 1.05] | 0.61 |
| michaels | 3-8 | 55 | 4.22 | 3.08 | 4.17 [3.30, 5.16] | 4.20 |
| michaels | 9-24 | 145 | 3.36 | 1.35 | 3.79 [3.30, 4.19] | 3.83 |
| michaels | 25-60 | 22 | 2.24 | n/a | 3.61 [3.26, 4.06] | 3.64 |
| michaels | 61+ | 38 | 4.82 | n/a | 5.69 [4.30, 6.99] | 5.71 |

| rig | diagnostic: resolvable sigma_total as is | diagnostic: fitted sigma, floor held (d + v) | schema 2: sqrt(sigma_v^2 + sigma_d^2), pooled | diagnostic: lag-1 observed / fitted model [5-95 %] | schema 2: lag-1 observed / model |
| --- | --- | --- | --- | --- | --- |
| dregon | 1.48 | 5.03 | 3.78 | 0.23 / 0.15 [0.12, 0.17] | 0.17 / 0.18 |
| michaels | 1.98 | 5.81 | 3.81 | 0.38 / 0.44 [0.41, 0.46] | 0.40 / 0.45 |

### Block length

The same estimate at every block length, `s^2` and `(sigma_d, tau_d)` held at that block length's measurement; tracks classified at it. The JSON carries 0.5 s. `sigma_v` is a stationary sd and `tau_v` a time, so neither should move with the block beyond the block average's smoothing of a short `tau`.

| rig | block s | tracks res / inter / under (weighted) | s^2 held dB^2 | sigma_d / tau_d held | sigma_v pooled [5-95 %] | tau_v [5-95 %] | sigma_v by group 1-2 / 3-8 / 9-24 / 25-60 / 61+ | lag-1 observed / model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon | 0.25 | 25 / 622 / 18293 (12747) | 3.90 | 0.92 / 0.13 | 3.86 [3.57, 4.13] | 0.56 [0.44, 0.69] | 0.00 / 0.00 / 3.24 / 4.42 / 7.59 | 0.30 / 0.29 |
| dregon | 0.5 | 44 / 767 / 17983 (15185) | 1.30 | 0.94 / 0.30 | 3.66 [3.25, 3.98] | 0.75 [0.59, 0.91] | 0.00 / 0.00 / 2.98 / 3.91 / 7.18 | 0.17 / 0.18 |
| dregon | 1 | 59 / 1078 / 17466 (16519) | 0.74 | 0.59 / 0.02 | 3.55 [3.04, 4.00] | 1.52 [1.16, 2.01] | 0.00 / 0.00 / 2.81 / 3.50 / 6.91 | -0.08 / -0.08 |
| michaels | 0.25 | 137 / 294 / 5254 (2212) | 2.42 | 0.58 / 0.30 | 3.96 [3.54, 4.37] | 0.71 [0.48, 0.98] | 0.00 / 4.40 / 4.08 / 4.13 / 5.54 | 0.53 / 0.53 |
| michaels | 0.5 | 167 / 262 / 5236 (3026) | 1.29 | 0.52 / 0.58 | 3.78 [3.22, 4.41] | 1.15 [0.57, 1.85] | 0.32 / 4.17 / 3.79 / 3.61 / 5.69 | 0.40 / 0.45 |
| michaels | 1 | 172 / 294 / 5172 (3581) | 0.79 | 0.58 / 0.02 | 3.48 [2.72, 4.68] | 1.96 [0.70, 4.60] | 0.00 / 3.78 / 3.64 / 3.47 / 6.20 | 0.24 / 0.25 |

**dregon:** pooled `sigma_v` 3.55-3.86 dB over 0.25-1 s blocks; `tau_v` 0.56 / 0.75 / 1.52 s -- it GROWS with the block: more than one OU time scale (a fast part the longer blocks average away), or too few blocks per window at the longest block (4 in the shortest window) to pin it. The contract's `tau_v` is the 0.5 s value, the block the v3 fit's latents live on.

**michaels:** pooled `sigma_v` 3.48-3.96 dB over 0.25-1 s blocks; `tau_v` 0.71 / 1.15 / 1.96 s -- it GROWS with the block: more than one OU time scale (a fast part the longer blocks average away), or too few blocks per window at the longest block (4 in the shortest window) to pin it. The contract's `tau_v` is the 0.5 s value, the block the v3 fit's latents live on.

## Resolvable lines

A (window, rotor, order) track is RESOLVABLE when its block-level prominence (mic-summed line cells over mic-summed local floor) is >= 6 dB in >= 80 % of the window's 0.5 s blocks, and ROTOR-DOMINANT when >= 80 % of its cells' line power is its own (window-mean line powers of every other rotor's orders times their Hann capture, per frame): with four rotors a few rev/s apart, another rotor's order sits within a bin or two of most lines. A track is USED when, in addition, its median measured block sd is under 2 dB (explainer section 1.5 (2): a noisier order's latent is prior-dominated); blocks under 3 dB prominence or over 4 dB measured sd are missing values. The line set is fixed at the chosen block length and used at every block length; the rotor-dominant set is used when its same-rotor pairs share >= 100 blocks, else every resolvable line.

Counts are `resolvable / rotor-dominant / resolvable & sd <= 2 dB / dominant & sd <= 2 dB`.

| rig | orders measured / window | tracks | distinct lines | line set used |
| --- | --- | --- | --- | --- |
| dregon | 105 | 44 / 4 / 32 / 2 | 21 / 4 / 16 / 2 | resolvable |
| michaels | 150 | 167 / 30 / 142 / 30 | 48 / 12 / 42 / 12 | dominant |

By order group (tracks):

| rig | 1-2 | 3-8 | 9-24 | 25-60 | 61+ |
| --- | --- | --- | --- | --- | --- |
| dregon | 8 / 0 / 8 / 0 | 0 / 0 / 0 / 0 | 22 / 2 / 20 / 1 | 6 / 1 / 3 / 1 | 8 / 1 / 1 / 0 |
| michaels | 32 / 8 / 32 / 8 | 63 / 10 / 56 / 10 | 54 / 12 / 41 / 12 | 0 / 0 / 0 / 0 | 18 / 0 / 13 / 0 |

By rotor (tracks):

- dregon: rotor 1: 20 / 1 / 15 / 1, rotor 2: 11 / 1 / 7 / 1, rotor 3: 6 / 0 / 6 / 0, rotor 4: 7 / 2 / 4 / 0
- michaels: rotor 1: 67 / 27 / 63 / 27, rotor 2: 42 / 0 / 33 / 0, rotor 3: 28 / 3 / 20 / 3, rotor 4: 30 / 0 / 26 / 0

Lines (rotor:order, windows):

- dregon: 1:2 (4 / 0 / 4 / 0), 1:12 (1 / 0 / 1 / 0), 1:13 (1 / 0 / 1 / 0), 1:14 (4 / 0 / 4 / 0), 1:18 (1 / 0 / 1 / 0), 1:42 (5 / 1 / 3 / 1), 1:70 (4 / 0 / 1 / 0), 2:13 (1 / 0 / 1 / 0), 2:14 (4 / 1 / 4 / 1), 2:16 (1 / 0 / 1 / 0), 2:21 (2 / 0 / 1 / 0), 2:78 (1 / 0 / 0 / 0), 2:80 (1 / 0 / 0 / 0), 2:81 (1 / 0 / 0 / 0), 3:2 (4 / 0 / 4 / 0), 3:14 (2 / 0 / 2 / 0), 4:13 (1 / 0 / 1 / 0), 4:16 (1 / 0 / 1 / 0), 4:21 (3 / 1 / 2 / 0), 4:48 (1 / 0 / 0 / 0), 4:80 (1 / 1 / 0 / 0)
- michaels: 1:2 (8 / 8 / 8 / 8), 1:4 (8 / 1 / 8 / 1), 1:6 (8 / 6 / 8 / 6), 1:8 (8 / 0 / 7 / 0), 1:10 (8 / 3 / 8 / 3), 1:12 (5 / 1 / 5 / 1), 1:14 (3 / 2 / 3 / 2), 1:16 (3 / 1 / 3 / 1), 1:18 (3 / 1 / 3 / 1), 1:20 (3 / 2 / 3 / 2), 1:22 (2 / 1 / 2 / 1), 1:24 (1 / 1 / 1 / 1), 1:84 (2 / 0 / 2 / 0), 1:94 (2 / 0 / 1 / 0), 1:95 (1 / 0 / 0 / 0), 1:98 (1 / 0 / 0 / 0), 1:99 (1 / 0 / 1 / 0), 2:2 (8 / 0 / 8 / 0), 2:4 (8 / 0 / 8 / 0), 2:5 (5 / 0 / 3 / 0), 2:6 (3 / 0 / 2 / 0), 2:8 (3 / 0 / 2 / 0), 2:10 (6 / 0 / 5 / 0), 2:12 (4 / 0 / 1 / 0), 2:14 (1 / 0 / 0 / 0), 2:75 (1 / 0 / 1 / 0), 2:84 (2 / 0 / 2 / 0), 2:89 (1 / 0 / 1 / 0), 3:2 (8 / 0 / 8 / 0), 3:4 (5 / 3 / 5 / 3), 3:7 (2 / 0 / 1 / 0), 3:9 (5 / 0 / 2 / 0), 3:11 (3 / 0 / 1 / 0), 3:18 (1 / 0 / 1 / 0), 3:22 (1 / 0 / 1 / 0), 3:65 (1 / 0 / 0 / 0), 3:72 (1 / 0 / 1 / 0), 3:73 (1 / 0 / 0 / 0), 4:2 (8 / 0 / 8 / 0), 4:4 (8 / 0 / 8 / 0), 4:5 (3 / 0 / 2 / 0), 4:6 (2 / 0 / 2 / 0), 4:10 (3 / 0 / 1 / 0), 4:12 (1 / 0 / 1 / 0), 4:14 (1 / 0 / 0 / 0), 4:71 (1 / 0 / 1 / 0), 4:79 (1 / 0 / 1 / 0), 4:84 (2 / 0 / 2 / 0)

## Block noise

| rig | block s | line blocks | line s^2 measured (median) | line s^2 explainer (median) | measured / explainer (median) | Gaussian-cell overlap factor (3-bin line rectangle) | floor s^2 measured | floor s^2 explainer | floor equiv / cells | floor mic-coherence count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon | 0.25 | 508 | 3.903 | 1.210 | 2.92 | 3.02 | 0.229 | 0.226 | 0.38 | 2.73 |
| dregon | 0.5 | 256 | 1.297 | 0.588 | 2.23 | 3.13 | 0.116 | 0.111 | 0.37 | 2.74 |
| dregon | 1.0 | 128 | 0.739 | 0.287 | 2.50 | 3.19 | 0.058 | 0.055 | 0.36 | 2.73 |
| michaels | 0.25 | 948 | 2.418 | 0.925 | 2.14 | 3.02 | 0.240 | 0.151 | 0.42 | 2.04 |
| michaels | 0.5 | 477 | 1.289 | 0.457 | 2.53 | 3.13 | 0.120 | 0.073 | 0.40 | 2.15 |
| michaels | 1.0 | 240 | 0.786 | 0.227 | 3.23 | 3.19 | 0.059 | 0.036 | 0.40 | 2.13 |

**What lag 0 hides** (chosen block length, rig-centred). A fit from lags 1-4 alone leaves a lag-0 excess, the nugget. For a PAIR of tracks it is block noise the two SHARE (a broadband event lifting every line and band of a block together), which would inflate the lag-0 covariance the common parts `d` and `u` are read from: a positive nugget is that inflation, a negative one says the OU shape over-predicts lag 0 from lags 1-4 (no sign of shared noise). For a track with itself it is the block noise the data show beyond the MEASURED one: near zero validates the within-block noise estimate.

| rig | line pairs (same rotor): nugget dB^2 | floor band pairs: nugget dB^2 | line auto from lags 1-4: sd dB / tau s | its nugget over the measured noise dB^2 | measured line noise, mean dB^2 |
| --- | --- | --- | --- | --- | --- |
| dregon | -0.150 | -0.166 | 2.15 / 0.34 | -2.318 | 1.621 |
| michaels | -0.748 | 0.313 | 1.96 / 1.05 | 0.025 | 1.429 |

## The estimates

sd in dB, tau in s. `d`: rotor-common line wander; `v`: per-line residual; `u`: floor level; `uj`: floor colour residual per control band; `tot`: all of a line's wander (d + v under one OU). Rig-centred (per regime), exact centred-moment fits, measured block noise; sigma of `d`/`u` at lag 0 and their tau over lags 1-4, sigma of `v`/`uj`/`tot` at lag 0 minus the noise and their tau at lag 1.

### Window bootstrap at 0.5 s

Windows resampled with replacement within each regime; median [5 %, 95 %] and the share of draws in which a sigma clips to zero. `lag 1` and `lags 1-4` are the two tau estimators of the explainer (the common parts use lags 1-4 in both).

| rig / tau from | sigma_d | tau_d | sigma_v | tau_v | sigma_u | tau_u | sigma_uj | tau_uj | sigma_tot | tau_tot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon / lag 1 | 0.89 [0.26, 1.28] (0 in 2 %) | 0.24 [0.02, 0.54] | 1.12 [0.00, 1.54] (0 in 9 %) | 0.79 [0.29, 26.87] | 2.81 [2.45, 3.17] | 4.66 [2.99, 7.61] | 1.51 [1.42, 1.60] | 1.38 [0.95, 1.98] | 1.42 [1.07, 1.66] | 0.46 [0.02, 0.92] |
| dregon / lags 1-4 | 0.89 [0.26, 1.28] (0 in 2 %) | 0.24 [0.02, 0.54] | 1.10 [0.00, 1.49] (0 in 9 %) | 0.46 [0.29, 1.05] | 2.81 [2.45, 3.17] | 4.66 [2.99, 7.61] | 1.51 [1.42, 1.61] | 1.74 [1.29, 2.26] | 1.41 [1.07, 1.64] | 0.39 [0.12, 0.59] |
| michaels / lag 1 | 0.49 [0.01, 0.63] (0 in 5 %) | 0.51 [0.02, 500.00] | 1.87 [1.28, 2.47] | 1.15 [0.84, 1.85] | 1.69 [1.23, 2.12] | 1.81 [1.19, 2.99] | 1.59 [1.47, 1.68] | 1.16 [0.81, 1.83] | 1.94 [1.39, 2.51] | 1.09 [0.84, 1.39] |
| michaels / lags 1-4 | 0.49 [0.01, 0.63] (0 in 5 %) | 0.51 [0.02, 500.00] | 1.86 [1.25, 2.47] | 0.96 [0.57, 1.33] | 1.69 [1.23, 2.12] | 1.81 [1.19, 2.99] | 1.61 [1.47, 1.69] | 1.92 [1.33, 2.94] | 1.93 [1.39, 2.51] | 0.93 [0.58, 1.20] |

### Three block lengths

| rig / block s | sigma_d | tau_d | sigma_v | tau_v | sigma_u | tau_u | sigma_uj | tau_uj | sigma_tot | tau_tot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon / 0.25 | 0.92 | 0.13 | 0.71 | 500.00 | 2.85 | 4.35 | 1.62 | 0.77 | 1.48 | 500.00 |
| dregon / 0.5 | 0.94 | 0.30 | 1.15 | 0.91 | 2.83 | 4.72 | 1.52 | 1.38 | 1.48 | 0.57 |
| dregon / 1.0 | 0.59 | 0.02 | 0.84 | 0.93 | 2.81 | 4.23 | 1.43 | 7.96 | 1.02 | 0.67 |
| michaels / 0.25 | 0.58 | 0.30 | 1.88 | 0.67 | 1.75 | 1.46 | 1.69 | 1.15 | 1.96 | 0.61 |
| michaels / 0.5 | 0.52 | 0.58 | 1.91 | 1.26 | 1.73 | 1.98 | 1.63 | 1.23 | 1.98 | 1.18 |
| michaels / 1.0 | 0.58 | 0.02 | 1.80 | 1.87 | 1.78 | 2.31 | 1.53 | 1.94 | 1.88 | 1.55 |

The same, WITHIN-WINDOW centring (the wander inside a flight only):

| rig / block s | sigma_d | tau_d | sigma_v | tau_v | sigma_u | tau_u | sigma_uj | tau_uj | sigma_tot | tau_tot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon / 0.25 | 0.83 | 0.02 | 9.11 | 500.00 | 2.20 | 2.64 | 1.30 | 0.48 | 18.18 | 500.00 |
| dregon / 0.5 | 0.87 | 0.24 | 1.37 | 1.24 | 2.35 | 3.60 | 1.24 | 0.87 | 1.53 | 0.61 |
| dregon / 1.0 | 0.46 | 0.02 | 0.81 | 0.23 | 2.36 | 4.27 | 1.52 | 3.48 | 0.93 | 0.21 |
| michaels / 0.25 | 0.54 | 0.22 | 1.95 | 0.77 | 1.58 | 1.35 | 1.38 | 0.68 | 2.01 | 0.67 |
| michaels / 0.5 | 0.51 | 0.53 | 2.13 | 1.68 | 1.56 | 1.68 | 1.34 | 0.93 | 2.16 | 1.49 |
| michaels / 1.0 | 0.55 | 0.02 | 2.25 | 3.30 | 1.75 | 2.85 | 1.26 | 1.44 | 2.16 | 2.32 |

The explainer's literal arithmetic (within-window, no centring correction) on the same tracks:

| rig / block s | line Var_b - s^2 -> sd | line tau lag-1 | line tau log-lin 1-4 | same-rotor cov -> sd_d | rotor-mean track tau lag-1 | floor Var_b - s^2 -> sd | floor cross-band cov -> sd_u | floor tau lag-1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon / 0.25 | 0.96 | n/a | 0.18 | 0.83 | 0.73 | 1.86 | 1.42 | 0.67 |
| dregon / 0.5 | 1.39 | 0.19 | n/a | 0.86 | n/a | 1.77 | 1.40 | 0.70 |
| dregon / 1.0 | 0.93 | n/a | n/a | 0.46 | n/a | 1.69 | 1.38 | 0.54 |
| michaels / 0.25 | 1.88 | 0.47 | 0.46 | 0.54 | 0.93 | 1.86 | 1.36 | 1.06 |
| michaels / 0.5 | 1.86 | 0.74 | 0.59 | 0.49 | 0.67 | 1.79 | 1.31 | 0.97 |
| michaels / 1.0 | 1.75 | 0.75 | n/a | 0.55 | 0.67 | 1.75 | 1.35 | 0.73 |

### Clips at zero

Per TRACK (one line or band in one window), the explainer's `Var_b(y) - s^2 <= 0`; per rig LINE (rotor, order pooled over its windows), the naive and the centred estimates; and whether any pooled fit clipped.

| rig / block s | line tracks clipped (measured noise) | floor tracks clipped | pooled fits clipped |
| --- | --- | --- | --- |
| dregon / 0.25 | 9 / 29 | 1 / 600 | none |
| dregon / 0.5 | 8 / 32 | 2 / 600 | none |
| dregon / 1.0 | 14 / 32 | 10 / 599 | none |
| michaels / 0.25 | 2 / 29 | 3 / 96 | none |
| michaels / 0.5 | 6 / 30 | 3 / 87 | none |
| michaels / 1.0 | 4 / 30 | 3 / 82 | none |

- dregon: rig lines pooled over windows: 1 / 16 clip with the naive arithmetic, 1 / 16 with the centred fit; with the EXPLAINER's noise count, 4 / 32 line tracks and 7 / 600 floor tracks clip.
- michaels: rig lines pooled over windows: 1 / 12 clip with the naive arithmetic, 1 / 12 with the centred fit; with the EXPLAINER's noise count, 8 / 30 line tracks and 6 / 87 floor tracks clip.

## Robustness at the chosen block length

| rig / variant | sigma_d | tau_d | sigma_v | tau_v | sigma_u | tau_u | sigma_uj | tau_uj | sigma_tot | tau_tot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon / tau at lag 1 (primary) | 0.94 | 0.30 | 1.15 | 0.91 | 2.83 | 4.72 | 1.52 | 1.38 | 1.48 | 0.57 |
| dregon / tau over lags 1-4 | 0.94 | 0.30 | 1.12 | 0.56 | 2.83 | 4.72 | 1.52 | 1.78 | 1.46 | 0.48 |
| dregon / explainer noise count | 0.94 | 0.30 | 1.50 | 0.42 | 2.83 | 4.72 | 1.50 | 1.47 | 1.77 | 0.38 |
| dregon / lines dominant (2 tracks) | 0.00 | n/a | 0.57 | 0.57 | 2.83 | 4.72 | 1.52 | 1.38 | 0.57 | 0.57 |
| dregon / orders <= 24 (28 tracks) | 0.96 | 0.24 | 0.99 | 1.19 | 2.83 | 4.72 | 1.52 | 1.38 | 1.36 | 0.52 |
| dregon / speed exponent 0.00 | 0.94 | 0.30 | 1.18 | 0.90 | 2.83 | 4.72 | 1.52 | 1.38 | 1.50 | 0.57 |
| dregon / speed exponent 7.89 | 0.99 | 0.34 | 1.07 | 0.99 | 2.83 | 4.72 | 1.52 | 1.38 | 1.44 | 0.58 |
| dregon / single mic, min | 0.00 | 0.21 | 0.97 | 0.56 | 2.81 | 1.86 | 2.16 | 0.70 | 1.65 | 0.51 |
| dregon / single mic, median | 0.90 | 0.48 | 1.87 | 1.18 | 3.15 | 3.34 | 2.56 | 1.02 | 2.00 | 1.07 |
| dregon / single mic, max | 1.44 | 500.00 | 2.18 | 500.00 | 4.05 | 4.41 | 2.97 | 1.53 | 2.48 | 2.17 |
| dregon / floor on 1/3-oct bands | n/a | n/a | n/a | n/a | 2.82 | 4.47 | 1.59 | 1.06 | n/a | n/a |
| michaels / tau at lag 1 (primary) | 0.52 | 0.58 | 1.91 | 1.26 | 1.73 | 1.98 | 1.63 | 1.23 | 1.98 | 1.18 |
| michaels / tau over lags 1-4 | 0.52 | 0.58 | 1.90 | 1.09 | 1.73 | 1.98 | 1.65 | 2.19 | 1.97 | 1.04 |
| michaels / explainer noise count | 0.52 | 0.58 | 2.12 | 0.78 | 1.73 | 1.98 | 1.65 | 1.17 | 2.18 | 0.76 |
| michaels / lines resolvable (142 tracks) | 1.38 | 1.74 | 2.24 | 1.52 | 1.73 | 1.98 | 1.63 | 1.23 | 2.63 | 1.57 |
| michaels / orders <= 24 (30 tracks) | 0.52 | 0.58 | 1.91 | 1.26 | 1.73 | 1.98 | 1.63 | 1.23 | 1.98 | 1.18 |
| michaels / speed exponent 0.00 | 0.59 | 0.54 | 1.93 | 1.26 | 1.73 | 1.98 | 1.63 | 1.23 | 2.01 | 1.15 |
| michaels / speed exponent 6.11 | 0.49 | 0.71 | 1.89 | 1.21 | 1.73 | 1.98 | 1.63 | 1.23 | 1.95 | 1.16 |
| michaels / single mic, min | 0.41 | 0.55 | 1.99 | 0.94 | 1.69 | 1.66 | 1.68 | 0.94 | 2.04 | 0.88 |
| michaels / single mic, median | 0.62 | 0.96 | 2.09 | 1.21 | 1.76 | 2.00 | 1.80 | 1.19 | 2.18 | 1.18 |
| michaels / single mic, max | 1.15 | 5.19 | 2.14 | 500.00 | 2.01 | 2.57 | 2.02 | 1.40 | 2.31 | 6.67 |
| michaels / floor on 1/3-oct bands | n/a | n/a | n/a | n/a | 1.70 | 1.91 | 1.56 | 1.30 | n/a | n/a |

- dregon: within-window speed exponent of the line levels (pooled slope of dB on 10 log10 f over 256 blocks, speed spread 0.05 dB): 7.89 (the model's is 2).
- michaels: within-window speed exponent of the line levels (pooled slope of dB on 10 log10 f over 477 blocks, speed spread 0.07 dB): 6.11 (the model's is 2).

## Breakdown

Per rotor (auto = all of a line's wander; common = same-rotor pairs), per order group, per floor band group; sd dB / tau s / pairs. Centred lag-1 fit, measured noise.

| rig | slice | sd | tau | pairs |
| --- | --- | --- | --- | --- |
| dregon | rotor 1 auto | 1.68 | 0.76 | 7 |
| dregon | rotor 2 auto | 1.21 | 0.29 | 4 |
| dregon | rotor 3 auto | 1.16 | 0.58 | 2 |
| dregon | rotor 4 auto | 1.48 | 0.29 | 3 |
| dregon | rotor 1 common | 1.00 | 0.38 | 42 |
| dregon | rotor 2 common | 1.06 | 0.25 | 12 |
| dregon | rotor 3 common | 0.97 | 0.11 | 2 |
| dregon | rotor 4 common | 0.00 | n/a | 6 |
| dregon | orders 1-2 auto | 0.82 | 500.00 | 2 |
| dregon | orders 25-60 auto | 1.01 | 0.16 | 1 |
| dregon | orders 61+ auto | 8.91 | 7.30 | 1 |
| dregon | orders 9-24 auto | 1.54 | 0.44 | 12 |
| dregon | lines of DIFFERENT rotors, pairs | 0.83 | 0.68 | 166 |
| dregon | floor 500-2k auto | 3.42 | 3.57 | 3 |
| dregon | floor <500 Hz auto | 3.74 | 3.23 | 5 |
| dregon | floor >=2k auto | 2.19 | 2.57 | 4 |
| michaels | rotor 1 auto | 1.48 | 1.53 | 11 |
| michaels | rotor 3 auto | 4.43 | 0.95 | 1 |
| michaels | rotor 1 common | 0.52 | 0.58 | 110 |
| michaels | orders 1-2 auto | 0.47 | 4.48 | 1 |
| michaels | orders 3-8 auto | 3.08 | 1.05 | 3 |
| michaels | orders 9-24 auto | 1.35 | 2.11 | 8 |
| michaels | lines of DIFFERENT rotors, pairs | 0.98 | 0.23 | 22 |
| michaels | floor 500-2k auto | 2.70 | 2.18 | 4 |
| michaels | floor <500 Hz auto | 3.28 | 1.43 | 4 |
| michaels | floor >=2k auto | 1.32 | 1.68 | 8 |

## Verdict

**dregon.** 32 line tracks used (44 resolvable, 4 rotor-dominant; set: resolvable). Line level around the rig mean: sd 1.48 dB [1.07, 1.66], tau 0.57 s, against a measured block noise of sd 1.27 dB; floor level: sd_u 2.83 dB [2.45, 3.17], tau_u 4.72 s [2.99, 7.61]; floor colour: sd_uj 1.52 dB [1.42, 1.60], tau_uj 1.38 s [0.95, 1.98]. Split of the line wander: rotor-common sd_d 0.94 dB [0.26, 1.28] (41 % of the variance), tau_d 0.30 s [0.02, 0.54]; per line on these lines (schema 1) sd_v 1.15 dB [0.00, 1.54], tau_v 0.91 s [0.29, 26.87]. The correlation times of d, v are NOT pinned by these windows (bootstrap 95 % over 5 % above 4x); the sigmas are. Across block lengths 0.25 / 0.5 / 1.0 s the line sd reads 1.48, 1.48, 1.02 dB and the floor level sd 2.85, 2.83, 2.81 dB. Against the legacy sliders: line sd below harm_gp_std_db [3.0, 4.5], line tau inside harm_gp_tau_s [0.3, 0.8], common share above harm_coherence [0.0, 0.1]; floor sd above floor_gp_std_db [1.0, 2.0], floor tau above floor_gp_tau_s [1.0, 4.0]. **The contract's per-line sd (schema 2), on all 15185 weighted valid tracks:** sd_v 3.66 dB [3.25, 3.98] pooled, tau_v 0.75 s [0.59, 0.91]; by order group k 1-2 0.00 dB [0.00, 0.00] at tau 0.75 s (pooled); k 3-8 0.00 dB [0.00, 0.99] at tau 0.75 s (pooled); k 9-24 2.98 dB [2.70, 3.25] at tau 0.79 s (group); k 25-60 3.91 dB [3.70, 4.12] at tau 0.62 s (group); k 61+ 7.18 dB [6.43, 7.88] at tau 0.87 s (group). The same estimator on the resolvable tracks alone reads 3.00 dB; lag-1 autocorrelation of the prominence 0.17 observed, 0.18 under the fit.

**michaels.** 30 line tracks used (167 resolvable, 30 rotor-dominant; set: dominant). Line level around the rig mean: sd 1.98 dB [1.39, 2.51], tau 1.18 s, against a measured block noise of sd 1.20 dB; floor level: sd_u 1.73 dB [1.23, 2.12], tau_u 1.98 s [1.19, 2.99]; floor colour: sd_uj 1.63 dB [1.47, 1.68], tau_uj 1.23 s [0.81, 1.83]. Split of the line wander: rotor-common sd_d 0.52 dB [0.01, 0.63] (7 % of the variance), tau_d 0.58 s [0.02, 500.00]; per line on these lines (schema 1) sd_v 1.91 dB [1.28, 2.47], tau_v 1.26 s [0.84, 1.85]. The correlation times of d are NOT pinned by these windows (bootstrap 95 % over 5 % above 4x); the sigmas are. Across block lengths 0.25 / 0.5 / 1.0 s the line sd reads 1.96, 1.98, 1.88 dB and the floor level sd 1.75, 1.73, 1.78 dB. Against the legacy sliders: line sd below harm_gp_std_db [2.9002381507162363, 2.9002381507162363], line tau above harm_gp_tau_s [0.8179731392136839, 0.8179731392136839], common share below harm_coherence [0.19148940999351152, 0.19148940999351152]; floor sd inside floor_gp_std_db [1.0, 2.0], floor tau inside floor_gp_tau_s [1.0, 4.0]. **The contract's per-line sd (schema 2), on all 3026 weighted valid tracks:** sd_v 3.78 dB [3.22, 4.41] pooled, tau_v 1.15 s [0.57, 1.85]; by order group k 1-2 0.32 dB [0.00, 1.05] at tau 1.15 s (pooled); k 3-8 4.17 dB [3.30, 5.16] at tau 1.15 s (pooled); k 9-24 3.79 dB [3.30, 4.19] at tau 0.86 s (group); k 25-60 3.61 dB [3.26, 4.06] at tau 1.06 s (group); k 61+ 5.69 dB [4.30, 6.99] at tau 1.15 s (pooled). The same estimator on the resolvable tracks alone reads 2.34 dB; lag-1 autocorrelation of the prominence 0.40 observed, 0.45 under the fit.


Legacy sliders (`conf/online_mix/rig_fitted_5050.yaml`, `ranges` of the two stochastic sources): dregon: harm_gp_std_db [3.0, 4.5], harm_gp_tau_s [0.3, 0.8], harm_coherence [0.0, 0.1], floor_gp_std_db [1.0, 2.0], floor_gp_tau_s [1.0, 4.0]; michaels: harm_gp_std_db [2.9002381507162363, 2.9002381507162363], harm_gp_tau_s [0.8179731392136839, 0.8179731392136839], harm_coherence [0.19148940999351152, 0.19148940999351152], floor_gp_std_db [1.0, 2.0], floor_gp_tau_s [1.0, 4.0].

## Figures

- `example_tracks.png`
- `autocovariance_vs_lag.png`

Detail: `wander_detail.json` (every number above).
