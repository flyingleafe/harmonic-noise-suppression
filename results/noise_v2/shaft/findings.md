# Shaft-phase noise, measured — findings

Source: `scripts/noise_v2_shaft_phase.py`. Numbers: `results/noise_v2/shaft/shaft.json`. Figures: `docs/explainers/noise-model-v2-plan/shaft_*.png`.

The model under test is `V_k(tau) = k^2 V_theta(tau) + 2 D_k tau` with `V_theta(tau) = 2 sigma_nu^2 [tau/lam - (1-exp(-lam tau))/lam^2]`. Integrated-OU means the speed error `nu` is correlated over `1/lam`; Wiener means `nu` is white, so `V_theta` is linear in `tau` at every lag. The integrated-OU curve has local log-log slope 1.5 at `lam tau = 2.1491`, which is how a measured lag is turned into `lam`.

## Verdict per rig

| rig / support | instrument | verdict | sigma_nu | lam | 1/lam [s] | D_theta | tau(slope 1.5) | dAIC OU-white | D_q/D_theta |
|---|---|---|---|---|---|---|---|---|---|
| neurobem_quad | telemetry 400 Hz | quasi-static (corner below every fit band; only sigma^2/lam identified) | - | - | - | - | 0.0113 | -1080202 | 0.00 |
| blackbird_quad | telemetry 187 Hz | quasi-static (corner below every fit band; only sigma^2/lam identified) | - | - | - | - | 0.00539 | -19991 | 0.00 |
| vid_m100 | telemetry 1014 Hz | integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only) | 33.1 | 2.39 | 0.419 | 467 | 0.0126 | -153348 | 0.00 |
| nanobench_cf21b | telemetry 100 Hz | integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only) | 144 | 0.674 | 1.48 | 3.08e+04 | 0.0105 | -33300 | 0.00 |
| pitcn_quad | telemetry 100 Hz | integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only) | 80.4 | 2.74 | 0.366 | 2.44e+03 | 0.0671 | -65919 | 0.00 |
| dregon_room1 | telemetry 1001 Hz | integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only) | 8.82 | 14.4 | 0.0693 | 5.6 | 0.00788 | -378933 | 0.00 |
| dregon_room1_command (aux) | telemetry 1001 Hz | integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only) | 8.01 | 2.84 | 0.352 | 22.2 | 0.00748 | -443845 | 0.00 |
| michaels | telemetry 29 Hz | wiener | - | - | - | - | - | +736 | 0.00 |
| dregon_bench | acoustics (12/12 supports) | integrated-OU | 1.77 | 5.48 | 0.182 | 0.599 | 0.392 | n/a | n/a |
| michaels_fly125 | acoustics (0/4 supports) | unidentified (saturated: V_theta exceeds 0.15 rad^2 at the shortest usable lag, so the harmonic is blurred rather than a line) | - | - | - | - | - | n/a | n/a |

Figure: `shaft_summary_table.png`.

## What the telemetry says

- **neurobem_quad** (rps, 400 Hz, 247 flights, 2479 s analysed). Verdict **quasi-static (corner below every fit band; only sigma^2/lam identified)**: the Lorentzian fit gives `sigma_nu = - rad/s`, `lam = 0.0898 1/s` (corner 0.0143 Hz, fit band 0.5-160 Hz), `D_theta = sigma^2/lam = 3.91e+06 rad^2/s`, and `AIC(OU) - AIC(white) = -1080202` (BIC -1080197). The structure function crosses slope 1.5 at `tau = 0.0113 s`, i.e. `lam = 190 1/s`. The ladder step is 0.00156 rev/s, the value is held for 0.0% of samples and changes at 400 Hz, so `D_q = 1.01e-08 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion.
- **blackbird_quad** (rps, 187 Hz, 1 flights, 199 s analysed). Verdict **quasi-static (corner below every fit band; only sigma^2/lam identified)**: the Lorentzian fit gives `sigma_nu = - rad/s`, `lam = 6.46 1/s` (corner 1.03 Hz, fit band 0.5-74.8 Hz), `D_theta = sigma^2/lam = 20.7 rad^2/s`, and `AIC(OU) - AIC(white) = -19991` (BIC -19986). The structure function crosses slope 1.5 at `tau = 0.00539 s`, i.e. `lam = 399 1/s`. The ladder step is 0.0167 rev/s, the value is held for 13.1% of samples and changes at 162 Hz, so `D_q = 2.81e-06 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion.
- **vid_m100** (rps, 1014 Hz, 4 flights, 312 s analysed). Verdict **integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only)**: the Lorentzian fit gives `sigma_nu = 33.1 rad/s`, `lam = 2.39 1/s` (corner 0.38 Hz, fit band 0.5-406 Hz), `D_theta = sigma^2/lam = 467 rad^2/s`, and `AIC(OU) - AIC(white) = -153348` (BIC -153341). The structure function crosses slope 1.5 at `tau = 0.0126 s`, i.e. `lam = 170 1/s`. The ladder step is 0.0167 rev/s, the value is held for 3.8% of samples and changes at 976 Hz, so `D_q = 4.68e-07 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion.
- **nanobench_cf21b** (rps, 100 Hz, 15 flights, 668 s analysed). Verdict **integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only)**: the Lorentzian fit gives `sigma_nu = 144 rad/s`, `lam = 0.674 1/s` (corner 0.107 Hz, fit band 0.5-40 Hz), `D_theta = sigma^2/lam = 3.08e+04 rad^2/s`, and `AIC(OU) - AIC(white) = -33300` (BIC -33297). The structure function crosses slope 1.5 at `tau = 0.0105 s`, i.e. `lam = 204 1/s`. The ladder step is 0.278 rev/s, the value is held for 0.0% of samples and changes at 100 Hz, so `D_q = 0.00127 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion.
- **pitcn_quad** (rps, 100 Hz, 68 flights, 3345 s analysed). Verdict **integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only)**: the Lorentzian fit gives `sigma_nu = 80.4 rad/s`, `lam = 2.74 1/s` (corner 0.435 Hz, fit band 0.5-40 Hz), `D_theta = sigma^2/lam = 2.44e+03 rad^2/s`, and `AIC(OU) - AIC(white) = -65919` (BIC -65915). The structure function crosses slope 1.5 at `tau = 0.0671 s`, i.e. `lam = 32.1 1/s`. The ladder step is 0.0667 rev/s, the value is held for 2.3% of samples and changes at 97.7 Hz, so `D_q = 7.48e-05 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion.
- **dregon_room1** (motors_measured, 1001 Hz, 5 flights, 239 s analysed). Verdict **integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only)**: the Lorentzian fit gives `sigma_nu = 8.82 rad/s`, `lam = 14.4 1/s` (corner 2.3 Hz, fit band 0.5-400 Hz), `D_theta = sigma^2/lam = 5.6 rad^2/s`, and `AIC(OU) - AIC(white) = -378933` (BIC -378927). The structure function crosses slope 1.5 at `tau = 0.00788 s`, i.e. `lam = 273 1/s`. The ladder step is 0.013 rev/s, the value is held for 95.4% of samples and changes at 46.4 Hz, so `D_q = 5.99e-06 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion.
- **dregon_room1_command** (motors_command, 1001 Hz, 5 flights, 241 s analysed). Verdict **integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only)**: the Lorentzian fit gives `sigma_nu = 8.01 rad/s`, `lam = 2.84 1/s` (corner 0.452 Hz, fit band 0.5-400 Hz), `D_theta = sigma^2/lam = 22.2 rad^2/s`, and `AIC(OU) - AIC(white) = -443845` (BIC -443839). The structure function crosses slope 1.5 at `tau = 0.00748 s`, i.e. `lam = 288 1/s`. The ladder step is 9.92e-05 rev/s, the value is held for 58.0% of samples and changes at 421 Hz, so `D_q = 3.84e-11 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion. This rig is AUXILIARY: it is a command track, not a speed measurement.
- **michaels** (rps, 29 Hz, 2 flights, 0 s analysed). Verdict **wiener**: the Lorentzian fit gives `sigma_nu = - rad/s`, `lam = 173 1/s` (corner 27.5 Hz, fit band 0.5-11.7 Hz), `D_theta = sigma^2/lam = 0.0256 rad^2/s`, and `AIC(OU) - AIC(white) = +736` (BIC +739). The structure function crosses slope 1.5 at `tau = - s`, i.e. `lam = - 1/s`. The ladder step is 0.0168 rev/s, the value is held for 50.6% of samples and changes at 14.4 Hz, so `D_q = 3.21e-05 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion.

Figures: `shaft_structure_functions.png`, `shaft_speed_psd.png`.

## What the acoustics say

- **dregon_bench**: 12 of 12 supports identified. Verdict **integrated-OU**: `sigma_nu = 1.77 rad/s` (range 0.23 - 3.2), `lam = 5.48 1/s` (range 0.31 - 16.2), `D_theta = 0.599 rad^2/s`. Model votes by AIC: {'ou_free_D': 12}. The per-order diffusion `D_k` is NOT IDENTIFIED: the fitted `2 D_k tau` at the shortest lag is 1.3x the measured `V` of the same order (NOT identified: the fitted 2 D_k tau at the shortest lag exceeds the measured V of that order, so D_k is absorbing model error), and the planted control independently returns a planted `D_k prop k` as flat. Read `sigma_nu`, `lam` and `D_theta` from this instrument; do NOT read `D_k` or its `k` exponent from it.
- **michaels_fly125**: 0 of 4 supports identified. Verdict **unidentified (saturated: V_theta exceeds 0.15 rad^2 at the shortest usable lag, so the harmonic is blurred rather than a line)**: `sigma_nu = - rad/s` (range - - -), `lam = - 1/s` (range - - -), `D_theta = - rad^2/s`. Model votes by AIC: None. The per-order diffusion `D_k` is NOT IDENTIFIED: the fitted `2 D_k tau` at the shortest lag is -x the measured `V` of the same order (None), and the planted control independently returns a planted `D_k prop k` as flat. Read `sigma_nu`, `lam` and `D_theta` from this instrument; do NOT read `D_k` or its `k` exponent from it.

Figures: `shaft_acoustic_Vk.png`, `shaft_Dk_vs_k.png`.

## The residual ABOVE the label band - the number the acoustic model needs

A rotor-speed label carried at the working rate 31.25 Hz holds nothing above 15.625 Hz. Everything above is phase noise a renderer must GENERATE, not read. Two variants isolate it: a zero-phase Butterworth high-pass at 16 Hz (a clean band edge, both fitted models multiplied by its |H|^4 power response) and the exact label residual, the native series minus the same series resampled to 31.25 Hz and back. `preferred` is the model the Gamma/Whittle AIC picks on that residual: `white` means the residual speed error is white, so the phase it drives is WIENER with `V_theta = 2 D_theta tau`; `OU tail` means a Lorentzian roll-off survives inside the band. `V_theta(tau)` is the MODEL-FREE structure function of the residual itself, measured per rotor and averaged over rotors, `V(tau) = 2 int S_nu(f)/(2 pi f)^2 [1 - cos(2 pi f tau)] df` over the measured Welch bins. The fitted Lorentzian's own `V_theta` is NOT this number: it integrates the model over all frequencies, including the band the trend removal just deleted, and where the corner sits below the band that extrapolation overstates `V_theta(1 s)` by five orders of magnitude. Rows marked `corner BELOW band` therefore print no sigma, lam or D_theta - on those the band sees the f^-2 tail alone and only the residual columns are measurements.

| rig | variant | fs [Hz] | band [Hz] | preferred | dAIC OU-white | nu_res RMS [rad/s] | S(10 ms) | S(100 ms) | S(1 s) | S_max | slope(5 ms) | slope(50 ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| neurobem_quad | 16 Hz high-pass | 400 | 16-160 | OU tail, corner BELOW band | -4312722 | 6.84 | 3.178e-03 | 4.569e-03 | 4.555e-03 | 7.977e-03 | 1.92 | -0.27 |
| neurobem_quad | label residual | 400 | 0.5-160 | OU tail | -4212613 | 9.67 | 6.940e-03 | 7.382e-02 | 5.826e-01 | 4.171e+00 | 1.95 | 0.62 |
| blackbird_quad | 16 Hz high-pass | 187 | 16-74.8 | OU tail | -71748 | 1.71 | 2.155e-04 | 2.426e-04 | 2.460e-04 | 3.980e-04 | 1.47 | -0.29 |
| blackbird_quad | label residual | 187 | 0.5-74.8 | white (Wiener phase) | -40832 | 2.75 | 6.771e-04 | 1.105e-02 | 7.992e-02 | 7.111e-01 | 1.78 | 1.06 |
| vid_m100 | 16 Hz high-pass | 1014 | 16-406 | OU tail, corner BELOW band | -629499 | 1.65 | 1.440e-04 | 2.760e-04 | 2.498e-04 | 4.420e-04 | 1.64 | -1.37 |
| vid_m100 | label residual | 1014 | 0.5-406 | OU tail | -799385 | 2.74 | 5.626e-04 | 5.869e-03 | 4.410e-02 | 2.829e-01 | 1.87 | 0.25 |
| nanobench_cf21b | 16 Hz high-pass | 100 | 16-40 | OU tail, corner BELOW band | -136374 | 5.03 | 3.052e-03 | 6.187e-03 | 5.618e-03 | 1.097e-02 | 1.53 | -3.88 |
| nanobench_cf21b | label residual | 100 | 0.5-40 | OU tail | -74782 | 8.38 | 9.141e-03 | 4.521e-02 | 4.690e-02 | 7.057e-02 | 1.64 | -0.80 |
| pitcn_quad | 16 Hz high-pass | 100 | 16-40 | OU tail, corner BELOW band | -263355 | 8.41 | 7.078e-03 | 8.201e-03 | 8.448e-03 | 1.353e-02 | 0.93 | -0.43 |
| pitcn_quad | label residual | 100 | 0.5-40 | OU tail | -110093 | 11.3 | 1.270e-02 | 1.018e-01 | 8.220e-01 | 6.397e+00 | 1.39 | 0.39 |
| dregon_room1 | 16 Hz high-pass | 1001 | 16-400 | OU tail, corner BELOW band | -1559227 | 2.59 | 4.309e-04 | 5.058e-04 | 5.007e-04 | 8.588e-04 | 1.71 | 0.23 |
| dregon_room1 | label residual | 1001 | 0.5-400 | OU tail | -581991 | 3.48 | 1.003e-03 | 1.960e-02 | 2.106e-01 | 1.896e+00 | 1.85 | 1.18 |
| dregon_room1_command | 16 Hz high-pass | 1001 | 16-400 | OU tail, corner BELOW band | -1621996 | 0.598 | 2.431e-05 | 3.105e-05 | 2.961e-05 | 4.996e-05 | 1.65 | -0.13 |
| dregon_room1_command | label residual | 1001 | 0.5-400 | OU tail | -1988605 | 0.828 | 6.453e-05 | 1.164e-03 | 1.045e-02 | 7.370e-02 | 1.83 | 1.06 |
| michaels | 16 Hz high-pass | 29 | - | skipped: corner 16.0 Hz is at or above 0.45 fs = 13.14 Hz | - | - | - | - | - | - | - |
| michaels | label residual | 29 | 0.5-11.7 | white (Wiener phase) | +1155 | 1.12 | 1.586e-03 | 1.634e-03 | 2.802e-03 | 1.561e-02 | -0.04 | -0.01 |

A rig whose 16 Hz row is `skipped` has its own Nyquist below that corner (Michael's log runs at ~29.4 Hz), so only its label-residual row speaks.

The 16 Hz residual SATURATES: its structure function reaches S_max = 5.0e-05-1.4e-02 rad^2 and stops growing, with a local slope already down to -3.88-0.23 at 50 ms, so the phase it drives is a BOUNDED wobble, not a random walk: at order k the standing deviation is k sqrt(S_max/2) = k x 0.005-0.082 rad.
The label-residual variant does NOT saturate - it keeps growing past 50 ms with a local slope of -0.80-1.18 and reaches S(1 s) = 2.80e-03-8.22e-01 rad^2. That growth is NOT band-limited content: the resampler here is LINEAR INTERPOLATION both ways (native onto a 31.25 Hz grid with `np.interp`, then back onto the native grid with `np.interp`), and linear interpolation has a lossy PASSBAND - it attenuates and phase-distorts frequencies well below its own Nyquist - so the residual carries low-frequency passband error as well as the out-of-band content. Read the low-frequency growth of the label-residual rows as the resampler's own error, and take the BOUNDED-wobble number from the 16 Hz rows, whose band edge is clean.

## Trend removal changes the answer - read lambda from the fitted corner

There are two ways to turn the data into `lam` and they DISAGREE, so both are printed. `tau(slope 1.5)` is read off the structure function of the FILTERED series. That series saturates at about `1/(2 pi f_hp)`, so the crossing moves with the high-pass and is not a filter-free property of the rig: pooled over rotors it reads 0.135-0.349 s under the 0.5 Hz high-pass (`lam from slope` 6.16-16.0 1/s, which does contain `lam_ref = 6`), 0.024-0.083 s under the 2 Hz high-pass, and 0.61-3.41 s with a linear detrend only. The apparent agreement of the 0.5 Hz column with `lam_ref` is therefore partly set by the filter corner and must NOT be quoted as a measurement of the shaft time constant.

The identifiable quantity is the FITTED corner, because both rival models are multiplied by the known `|H|^4` power response of the zero-phase high-pass before fitting, so the filter is in the model rather than in the answer. Where that corner lands inside the fit band (`corner in band` = yes) the fitted `lam` is a measurement; where it does not, only the tail amplitude is identified and `lam` is a ridge coordinate. On the 0.5 Hz variant the in-band rigs give `lam` = 1.41 (michaels), 1.67 (vid_m100), 5.50 (pitcn_quad), 8.63 (dregon_room1_command) and 33.1 1/s (dregon_room1) - a spread of a factor 23 across rigs, which brackets `lam_ref = 6` but does not endorse it as universal.

| rig | variant | trend removal | tau(slope 1.5) [s] | lam from slope [1/s] | fitted lam [1/s] | fitted corner [Hz] | corner in band |
|---|---|---|---|---|---|---|---|
| neurobem_quad | 16 | butterworth | 0.0113 | 190 | 0.0898 | 0.0143 | no |
| neurobem_quad | 0.5 | butterworth | 0.275 | 7.82 | 0.0142 | 0.00226 | no |
| neurobem_quad | 2 | butterworth | 0.0371 | 57.9 | 25.4 | 4.05 | yes |
| neurobem_quad | detrend | linear detrend per segment | 3.41 | 0.631 | 0.0141 | 0.00224 | no |
| neurobem_quad | label_residual | native minus resample to 31.25 Hz and back | 0.0177 | 122 | 128 | 20.4 | yes |
| blackbird_quad | 16 | butterworth | 0.00539 | 399 | 6.46 | 1.03 | yes |
| blackbird_quad | 0.5 | butterworth | 0.338 | 6.36 | 0.00532 | 0.000846 | no |
| blackbird_quad | 2 | butterworth | 0.083 | 25.9 | 0.00505 | 0.000803 | no |
| blackbird_quad | detrend | linear detrend per segment | 0.999 | 2.15 | 0.00291 | 0.000464 | no |
| blackbird_quad | label_residual | native minus resample to 31.25 Hz and back | 0.0178 | 121 | 214 | 34 | yes |
| vid_m100 | 16 | butterworth | 0.0126 | 170 | 24.2 | 3.86 | no |
| vid_m100 | 0.5 | butterworth | 0.135 | 16 | 1.67 | 0.266 | no |
| vid_m100 | 2 | butterworth | 0.0779 | 27.6 | 0.0334 | 0.00532 | no |
| vid_m100 | detrend | linear detrend per segment | 2.27 | 0.95 | 2.39 | 0.38 | yes |
| vid_m100 | label_residual | native minus resample to 31.25 Hz and back | 0.0185 | 116 | 153 | 24.3 | yes |
| nanobench_cf21b | 16 | butterworth | 0.0105 | 204 | 0.11 | 0.0175 | no |
| nanobench_cf21b | 0.5 | butterworth | 0.177 | 12.1 | 0.0381 | 0.00606 | no |
| nanobench_cf21b | 2 | butterworth | 0.0717 | 30 | 0.0579 | 0.00922 | no |
| nanobench_cf21b | detrend | linear detrend per segment | 0.674 | 3.19 | 0.674 | 0.107 | yes |
| nanobench_cf21b | label_residual | native minus resample to 31.25 Hz and back | 0.0156 | 138 | 129 | 20.6 | yes |
| pitcn_quad | 16 | butterworth | 0.0671 | 32.1 | 5.21 | 0.83 | no |
| pitcn_quad | 0.5 | butterworth | 0.349 | 6.16 | 5.5 | 0.876 | yes |
| pitcn_quad | 2 | butterworth | 0.0237 | 90.6 | 46.7 | 7.44 | yes |
| pitcn_quad | detrend | linear detrend per segment | 0.641 | 3.35 | 2.74 | 0.435 | yes |
| pitcn_quad | label_residual | native minus resample to 31.25 Hz and back | - | - | 169 | 26.9 | yes |
| dregon_room1 | 16 | butterworth | 0.00788 | 273 | 4.78 | 0.76 | no |
| dregon_room1 | 0.5 | butterworth | 0.198 | 10.9 | 33.1 | 5.26 | yes |
| dregon_room1 | 2 | butterworth | 0.0351 | 61.4 | 51.9 | 8.26 | yes |
| dregon_room1 | detrend | linear detrend per segment | 0.627 | 3.49 | 14.4 | 2.3 | yes |
| dregon_room1 | label_residual | native minus resample to 31.25 Hz and back | 0.0161 | 133 | 485 | 77.2 | yes |
| dregon_room1_command | 16 | butterworth | 0.00748 | 288 | 10.5 | 1.67 | no |
| dregon_room1_command | 0.5 | butterworth | 0.23 | 9.34 | 8.63 | 1.37 | yes |
| dregon_room1_command | 2 | butterworth | 0.0633 | 34 | 8.55 | 1.36 | no |
| dregon_room1_command | detrend | linear detrend per segment | 0.881 | 2.48 | 2.84 | 0.452 | yes |
| dregon_room1_command | label_residual | native minus resample to 31.25 Hz and back | 0.0179 | 120 | 38.4 | 6.11 | yes |
| michaels | 0.5 | butterworth | 0.222 | 9.68 | 1.41 | 0.224 | no |
| michaels | 2 | butterworth | 0.0364 | 59.1 | 5.69 | 0.906 | no |
| michaels | detrend | linear detrend per segment | 0.611 | 3.53 | 2.59 | 0.412 | yes |
| michaels | label_residual | native minus resample to 31.25 Hz and back | - | - | 173 | 27.5 | no |

## Against the C3 fitted values

| rig | C3 sigma [rad/s] | C3 D [rad^2/s] | measured D_theta (telemetry) | measured D_theta (acoustics) | sigma at lam_ref=6 (telemetry) | sigma at lam_ref=6 (acoustics) | C3 sigma / measured |
|---|---|---|---|---|---|---|---|
| DREGON airframe | 3.1 | 845 | 5.6 | 0.599 | 5.8 | 1.9 | tel 0.5x / ac 1.6x |
| Michael's M100 | 4.21 | 1.63 | 0.0256 | - | 0.392 | - | tel 10.7x / ac -x |

## Caveats

- **Sample-and-hold.** DREGON `motors_measured` holds its value for 95.4% of native samples and changes at 46.4 Hz, so its spectrum above that rate is its own staircase, not rotor dynamics. The same check is reported for every rig in `shaft.json` (`quantisation.held_fraction`, `quantisation.update_rate_hz`).
- **Quantisation.** `D_q` is the diffusion a white rounding error of the measured ladder step would produce if it were renewed at the observed update rate: `D_q = (2 pi step)^2 / (24 f_update)`. Where `D_q/D_theta` approaches 1 the diffusive tail of `S(tau)` is an artefact of the telemetry channel, not shaft physics. Both `D_q` variants (logging rate and update rate) are in `shaft.json`.
- **High-pass choice.** The slow trend is removed with a zero-phase order-4 Butterworth high-pass at 16 Hz (headline) and 0.5/2 Hz (reported alongside). A high-pass at `f_hp` makes `theta` stationary, so `S(tau)` SATURATES above `tau ~ 1/(2 pi f_hp)` instead of growing; the fitted OU curve drawn on the same figure passes through the same filter, which is why data and model can be compared at all. The PSD fit multiplies both models by the known `|H|^4` power response, so the corner bin is not read as a real roll-off.
- **Acoustic bandwidth.** One harmonic cannot be separated from its neighbours faster than one shaft revolution, so the acoustic lag grid starts at the frame rate of a window of 6 revolutions, not at 1 ms; the requested 1 ms lag is unreachable with any harmonic-isolating demodulation and the script reports the window and frame rate it actually used per support (`window_s`, `frame_rate_hz`).
- **No acoustic trend removal, and none possible.** A constant rate error is INVISIBLE to this estimator: it multiplies the baseband by `exp(-i 2 pi k d t)`, so the lagged product gains only the t-independent factor `exp(-i 2 pi k d tau)` and `|sum_t z(t+tau) z*(t)|` does not move. So no rate refinement is applied, and the acoustic estimate carries NO analogue of the telemetry high-pass: a slow drift of the mean speed is measured, not removed. The planted control in `shaft.json` (`planted_control`) confirms the invariance: a coherence-maximising rate search returned -0.28 rev/s on an exactly known rate and corrupted every `V_k`, which is why it was removed.
- **Estimator ceiling.** The coherence estimator can only resolve `V` while the debiased squared coherence stands clear of its own MEASURED floor, which is read off the half-order flank band (same window, frames and noise, no line). Cells past the ceiling are reported as `null`, never as a saturated value; `g2_floor` and `g2_se` in `shaft.json` give the floor and the error per cell.
- **Finite SNR.** Additive noise costs the baseband coherence a lag-INDEPENDENT `2 log(1 + 1/SNR)`, so every joint fit carries a free non-negative offset `c_k` per order; without it a weak order inflates `D_k`.
