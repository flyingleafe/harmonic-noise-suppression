# Shaft-phase noise, measured — findings

Source: `scripts/noise_v2_shaft_phase.py`. Numbers: `results/noise_v2/shaft/shaft.json`. Figures: `docs/explainers/noise-model-v2-plan/shaft_*.png`.

The model under test is `V_k(tau) = k^2 V_theta(tau) + 2 D_k tau` with `V_theta(tau) = 2 sigma_nu^2 [tau/lam - (1-exp(-lam tau))/lam^2]`. Integrated-OU means the speed error `nu` is correlated over `1/lam`; Wiener means `nu` is white, so `V_theta` is linear in `tau` at every lag. The integrated-OU curve has local log-log slope 1.5 at `lam tau = 2.1491`, which is how a measured lag is turned into `lam`.

## Verdict per rig

| rig / support | instrument | verdict | sigma_nu | lam | 1/lam [s] | D_theta | tau(slope 1.5) | dAIC OU-white | D_q/D_theta |
|---|---|---|---|---|---|---|---|---|---|
| neurobem_quad | telemetry 400 Hz | quasi-static (corner below every fit band; only sigma^2/lam identified) | 949 | 0.0142 | 70.3 | 6.37e+07 | 0.275 | -4102119 | 0.00 |
| blackbird_quad | telemetry 187 Hz | quasi-static (corner below every fit band; only sigma^2/lam identified) | 933 | 0.00532 | 188 | 2.01e+08 | 0.338 | -308890 | 0.00 |
| vid_m100 | telemetry 1014 Hz | integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only) | 33.1 | 2.39 | 0.419 | 467 | 0.135 | -905943 | 0.00 |
| nanobench_cf21b | telemetry 100 Hz | integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only) | 144 | 0.674 | 1.48 | 3.08e+04 | 0.177 | -211273 | 0.00 |
| pitcn_quad | telemetry 100 Hz | integrated-OU | 56.5 | 5.5 | 0.182 | 610 | 0.349 | -831922 | 0.00 |
| dregon_room1 | telemetry 1001 Hz | integrated-OU | 6.64 | 33.1 | 0.0302 | 1.2 | 0.198 | -811997 | 0.00 |
| dregon_room1_command (aux) | telemetry 1001 Hz | integrated-OU | 5.27 | 8.63 | 0.116 | 3.18 | 0.23 | -1192057 | 0.00 |
| michaels | telemetry 29 Hz | integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only) | 12.5 | 2.59 | 0.387 | 60.2 | 0.222 | -12020 | 0.00 |
| dregon_bench | acoustics (0/12 supports) | unidentified (saturated: V_theta exceeds 0.15 rad^2 at the shortest usable lag, so the harmonic is blurred rather than a line) | - | - | - | - | - | n/a | n/a |
| michaels_fly125 | acoustics (0/4 supports) | unidentified (saturated: V_theta exceeds 0.15 rad^2 at the shortest usable lag, so the harmonic is blurred rather than a line) | - | - | - | - | - | n/a | n/a |

Figure: `shaft_summary_table.png`.

## What the telemetry says

- **neurobem_quad** (rps, 400 Hz, 247 flights, 2479 s analysed). Verdict **quasi-static (corner below every fit band; only sigma^2/lam identified)**: the Lorentzian fit gives `sigma_nu = 949 rad/s`, `lam = 0.0142 1/s` (corner 0.00226 Hz, fit band 0.5-160 Hz), `D_theta = sigma^2/lam = 6.37e+07 rad^2/s`, and `AIC(OU) - AIC(white) = -4102119` (BIC -4102114). The structure function crosses slope 1.5 at `tau = 0.275 s`, i.e. `lam = 7.82 1/s`. The ladder step is 0.00156 rev/s, the value is held for 0.0% of samples and changes at 400 Hz, so `D_q = 1.01e-08 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion.
- **blackbird_quad** (rps, 187 Hz, 1 flights, 199 s analysed). Verdict **quasi-static (corner below every fit band; only sigma^2/lam identified)**: the Lorentzian fit gives `sigma_nu = 933 rad/s`, `lam = 0.00532 1/s` (corner 0.000846 Hz, fit band 0.5-74.8 Hz), `D_theta = sigma^2/lam = 2.01e+08 rad^2/s`, and `AIC(OU) - AIC(white) = -308890` (BIC -308885). The structure function crosses slope 1.5 at `tau = 0.338 s`, i.e. `lam = 6.36 1/s`. The ladder step is 0.0167 rev/s, the value is held for 13.1% of samples and changes at 162 Hz, so `D_q = 2.81e-06 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion.
- **vid_m100** (rps, 1014 Hz, 4 flights, 312 s analysed). Verdict **integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only)**: the Lorentzian fit gives `sigma_nu = 33.1 rad/s`, `lam = 2.39 1/s` (corner 0.38 Hz, fit band 0.5-406 Hz), `D_theta = sigma^2/lam = 467 rad^2/s`, and `AIC(OU) - AIC(white) = -905943` (BIC -905936). The structure function crosses slope 1.5 at `tau = 0.135 s`, i.e. `lam = 16 1/s`. The ladder step is 0.0167 rev/s, the value is held for 3.8% of samples and changes at 976 Hz, so `D_q = 4.68e-07 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion.
- **nanobench_cf21b** (rps, 100 Hz, 15 flights, 668 s analysed). Verdict **integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only)**: the Lorentzian fit gives `sigma_nu = 144 rad/s`, `lam = 0.674 1/s` (corner 0.107 Hz, fit band 0.5-40 Hz), `D_theta = sigma^2/lam = 3.08e+04 rad^2/s`, and `AIC(OU) - AIC(white) = -211273` (BIC -211269). The structure function crosses slope 1.5 at `tau = 0.177 s`, i.e. `lam = 12.1 1/s`. The ladder step is 0.278 rev/s, the value is held for 0.0% of samples and changes at 100 Hz, so `D_q = 0.00127 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion.
- **pitcn_quad** (rps, 100 Hz, 68 flights, 3345 s analysed). Verdict **integrated-OU**: the Lorentzian fit gives `sigma_nu = 56.5 rad/s`, `lam = 5.5 1/s` (corner 0.876 Hz, fit band 0.5-40 Hz), `D_theta = sigma^2/lam = 610 rad^2/s`, and `AIC(OU) - AIC(white) = -831922` (BIC -831918). The structure function crosses slope 1.5 at `tau = 0.349 s`, i.e. `lam = 6.16 1/s`. The ladder step is 0.0667 rev/s, the value is held for 2.3% of samples and changes at 97.7 Hz, so `D_q = 7.48e-05 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion.
- **dregon_room1** (motors_measured, 1001 Hz, 5 flights, 239 s analysed). Verdict **integrated-OU**: the Lorentzian fit gives `sigma_nu = 6.64 rad/s`, `lam = 33.1 1/s` (corner 5.26 Hz, fit band 0.5-400 Hz), `D_theta = sigma^2/lam = 1.2 rad^2/s`, and `AIC(OU) - AIC(white) = -811997` (BIC -811990). The structure function crosses slope 1.5 at `tau = 0.198 s`, i.e. `lam = 10.9 1/s`. The ladder step is 0.013 rev/s, the value is held for 95.4% of samples and changes at 46.4 Hz, so `D_q = 5.99e-06 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion.
- **dregon_room1_command** (motors_command, 1001 Hz, 5 flights, 241 s analysed). Verdict **integrated-OU**: the Lorentzian fit gives `sigma_nu = 5.27 rad/s`, `lam = 8.63 1/s` (corner 1.37 Hz, fit band 0.5-400 Hz), `D_theta = sigma^2/lam = 3.18 rad^2/s`, and `AIC(OU) - AIC(white) = -1192057` (BIC -1192051). The structure function crosses slope 1.5 at `tau = 0.23 s`, i.e. `lam = 9.34 1/s`. The ladder step is 9.92e-05 rev/s, the value is held for 58.0% of samples and changes at 421 Hz, so `D_q = 3.84e-11 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion. This rig is AUXILIARY: it is a command track, not a speed measurement.
- **michaels** (rps, 29 Hz, 2 flights, 227 s analysed). Verdict **integrated-OU (corner under the 0.5 Hz high-pass; located detrend-only)**: the Lorentzian fit gives `sigma_nu = 12.5 rad/s`, `lam = 2.59 1/s` (corner 0.412 Hz, fit band 0.5-11.7 Hz), `D_theta = sigma^2/lam = 60.2 rad^2/s`, and `AIC(OU) - AIC(white) = -12020` (BIC -12016). The structure function crosses slope 1.5 at `tau = 0.222 s`, i.e. `lam = 9.68 1/s`. The ladder step is 0.0168 rev/s, the value is held for 50.6% of samples and changes at 14.4 Hz, so `D_q = 3.21e-05 rad^2/s` = 0.00 of the measured `D_theta`; quantisation floor is below the measured diffusion.

Figures: `shaft_structure_functions.png`, `shaft_speed_psd.png`.

## What the acoustics say

- **dregon_bench**: 0 of 12 supports identified. Verdict **unidentified (saturated: V_theta exceeds 0.15 rad^2 at the shortest usable lag, so the harmonic is blurred rather than a line)**: `sigma_nu = - rad/s` (range - - -), `lam = - 1/s` (range - - -), `D_theta = - rad^2/s`. The per-order diffusion scales as `D_k ~ k^-` (range - - -). Model votes by AIC: None.
- **michaels_fly125**: 0 of 4 supports identified. Verdict **unidentified (saturated: V_theta exceeds 0.15 rad^2 at the shortest usable lag, so the harmonic is blurred rather than a line)**: `sigma_nu = - rad/s` (range - - -), `lam = - 1/s` (range - - -), `D_theta = - rad^2/s`. The per-order diffusion scales as `D_k ~ k^-` (range - - -). Model votes by AIC: None.

Figures: `shaft_acoustic_Vk.png`, `shaft_Dk_vs_k.png`.

## Against the C3 fitted values

| rig | C3 sigma [rad/s] | C3 D [rad^2/s] | measured D_theta (telemetry) | measured D_theta (acoustics) | sigma at lam_ref=6 (telemetry) | sigma at lam_ref=6 (acoustics) | C3 sigma / measured |
|---|---|---|---|---|---|---|---|
| DREGON airframe | 3.1 | 845 | 1.2 | - | 2.68 | - | tel 1.2x / ac -x |
| Michael's M100 | 4.21 | 1.63 | 60.2 | - | 19 | - | tel 0.2x / ac -x |

## Caveats

- **Sample-and-hold.** DREGON `motors_measured` holds its value for 95.4% of native samples and changes at 46.4 Hz, so its spectrum above that rate is its own staircase, not rotor dynamics. The same check is reported for every rig in `shaft.json` (`quantisation.held_fraction`, `quantisation.update_rate_hz`).
- **Quantisation.** `D_q` is the diffusion a white rounding error of the measured ladder step would produce if it were renewed at the observed update rate: `D_q = (2 pi step)^2 / (24 f_update)`. Where `D_q/D_theta` approaches 1 the diffusive tail of `S(tau)` is an artefact of the telemetry channel, not shaft physics. Both `D_q` variants (logging rate and update rate) are in `shaft.json`.
- **High-pass choice.** The slow trend is removed with a zero-phase order-4 Butterworth high-pass at 0.5 Hz (headline) and 2 Hz (reported alongside). A high-pass at `f_hp` makes `theta` stationary, so `S(tau)` SATURATES above `tau ~ 1/(2 pi f_hp)` instead of growing; the fitted OU curve drawn on the same figure passes through the same filter, which is why data and model can be compared at all. The PSD fit multiplies both models by the known `|H|^4` power response, so the corner bin is not read as a real roll-off.
- **Acoustic bandwidth.** One harmonic cannot be separated from its neighbours faster than one shaft revolution, so the acoustic lag grid starts at the frame rate of a window of 6 revolutions, not at 1 ms; the requested 1 ms lag is unreachable with any harmonic-isolating demodulation and the script reports the window and frame rate it actually used per support (`window_s`, `frame_rate_hz`).
- **No acoustic trend removal, and none possible.** A constant rate error is INVISIBLE to this estimator: it multiplies the baseband by `exp(-i 2 pi k d t)`, so the lagged product gains only the t-independent factor `exp(-i 2 pi k d tau)` and `|sum_t z(t+tau) z*(t)|` does not move. So no rate refinement is applied, and the acoustic estimate carries NO analogue of the telemetry high-pass: a slow drift of the mean speed is measured, not removed. The planted control in `shaft.json` (`planted_control`) confirms the invariance: a coherence-maximising rate search returned -0.28 rev/s on an exactly known rate and corrupted every `V_k`, which is why it was removed.
- **Estimator ceiling.** The coherence estimator can only resolve `V` while the debiased squared coherence stands clear of its own MEASURED floor, which is read off the half-order flank band (same window, frames and noise, no line). Cells past the ceiling are reported as `null`, never as a saturated value; `g2_floor` and `g2_se` in `shaft.json` give the floor and the error per cell.
- **Finite SNR.** Additive noise costs the baseband coherence a lag-INDEPENDENT `2 log(1 + 1/SNR)`, so every joint fit carries a free non-negative offset `c_k` per order; without it a weak order inflates `D_k`.
