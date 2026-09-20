# R4 legacy-truth A: anatomy of the legacy DREGON fit against v2 R3

Windows free-flight, hovering, updown; seed 2001, 8 mics; legacy export `results/S2/dregon_room2_cruise_refined.json` (identity-matched per recording), v2 flight fit `results/noise_v2/rounds/round3/fits/dregon_room2_floor__flight_floor_lowk.json` (`converged` False); git `9504fe5c13cdea78c8ba372ca6cf67f8883a2df8`.

## Per-order comb level (dB)

Legacy `profile_db` is a band weight in periodogram*Hz and v2's is a tone mean square; the column `legacy (v2 units)` applies the ONE stated conversion `revised_eval.legacy_profile_db_to_candidate` (+10 log10(2/16000) = -39.03 dB). `legacy needle` is that level times the coherent share `w_k`; the rest of the order's power is a Rayleigh pedestal of half-width `gamma_rk`. Rotor-mean in linear power.

| k | legacy (own units) | legacy (v2 units) | w_k | legacy needle (v2 units) | v2 flight profile | v2 low-order gain | legacy - v2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | +14.96 | -24.07 | 0.7064 | -25.58 | -46.04 | +13.86 | +21.98 |
| 2 | +8.71 | -30.32 | 0.2490 | -36.36 | -51.57 | +3.67 | +21.24 |
| 3 | +1.66 | -37.37 | 0.0438 | -50.96 | -73.35 | -0.87 | +35.97 |
| 4 | +1.86 | -37.17 | 0.0038 | -61.32 | -69.87 | -4.89 | +32.70 |
| 5 | -1.61 | -40.64 | 0.0002 | -78.38 | -75.66 | +0.65 | +35.02 |
| 6 | -1.54 | -40.57 | 0.0000 | -94.90 | -74.89 | -9.20 | +34.32 |
| 7 | -4.67 | -43.70 | 0.0000 | -117.66 | -78.46 | -0.97 | +34.76 |
| 8 | -3.03 | -42.06 | 0.0000 | -138.66 | -73.11 | -5.70 | +31.04 |
| 9 | -4.67 | -43.70 | 0.0000 | -163.70 | -71.98 | — | +28.27 |
| 10 | -4.52 | -43.55 | 0.0000 | -163.55 | -71.49 | — | +27.94 |
| 11 | -10.11 | -49.14 | 0.0000 | -169.14 | -65.62 | — | +16.48 |
| 12 | -8.22 | -47.25 | 0.0000 | -167.25 | -66.90 | — | +19.65 |
| 13 | -9.24 | -48.27 | 0.0000 | -168.27 | -65.33 | — | +17.06 |
| 14 | -8.06 | -47.09 | 0.0000 | -167.09 | -67.97 | — | +20.88 |
| 15 | -14.49 | -53.52 | 0.0000 | -173.52 | -75.19 | — | +21.68 |
| 16 | -6.98 | -46.01 | 0.0000 | -166.01 | -70.33 | — | +24.32 |
| 17 | -18.31 | -57.34 | 0.0000 | -177.34 | -81.18 | — | +23.84 |
| 18 | -8.17 | -47.20 | 0.0000 | -167.20 | -71.16 | — | +23.96 |
| 19 | -11.90 | -50.93 | 0.0000 | -170.93 | -76.83 | — | +25.90 |
| 20 | -8.62 | -47.65 | 0.0000 | -167.65 | -71.38 | — | +23.73 |
| 21 | -10.56 | -49.59 | 0.0000 | -169.59 | -70.06 | — | +20.48 |
| 22 | -11.37 | -50.40 | 0.0000 | -170.40 | -71.43 | — | +21.03 |
| 23 | -20.09 | -59.12 | 0.0000 | -179.12 | -78.89 | — | +19.77 |
| 24 | -10.99 | -50.02 | 0.0000 | -170.02 | -71.86 | — | +21.84 |
| 25 | -28.38 | -67.41 | 0.0000 | -187.41 | -80.44 | — | +13.03 |
| 26 | -11.53 | -50.56 | 0.0000 | -170.56 | -71.90 | — | +21.34 |
| 27 | -21.96 | -60.99 | 0.0000 | -180.99 | -77.69 | — | +16.70 |
| 28 | -11.91 | -50.94 | 0.0000 | -170.94 | -72.41 | — | +21.47 |
| 29 | -30.71 | -69.74 | 0.0000 | -189.74 | -78.20 | — | +8.45 |
| 30 | -13.77 | -52.80 | 0.0000 | -172.80 | -74.06 | — | +21.25 |
| 31 | -23.03 | -62.06 | 0.0000 | -182.06 | -84.25 | — | +22.19 |
| 32 | -14.26 | -53.29 | 0.0000 | -173.29 | -73.16 | — | +19.87 |

Both families scale the comb by `(f_r / 80)^amp_exp`, so at a window's own mean carrier the legacy comb carries an extra +0.19 dB (fbar 80.4 rev/s, free-flight), +0.43 dB (fbar 80.9 rev/s, hovering), -0.33 dB (fbar 79.3 rev/s, updown) and the v2 comb exactly 0 dB (`amp_exp` fitted at 0).

## Line width, coherence, floor and speed law

| quantity | legacy | v2 R3 flight |
|---|---|---|
| line half-width gamma at k=1 (Hz) | 13.279 | 0.0014 |
| line half-width gamma at k=2 (Hz) | 13.490 | 0.0037 |
| line half-width gamma at k=4 (Hz) | 13.913 | 0.0237 |
| line half-width gamma at k=8 (Hz) | 14.759 | 0.6851 |
| line half-width gamma at k=16 (Hz) | 16.450 | 1.1878 |
| line half-width gamma at k=32 (Hz) | 19.832 | 6.0870 |
| line model | needle (share w_k) + Rayleigh pedestal, k_half 1.696 | ONE Wiener-phase line per order, no split |
| shaft model | label track only (shaft_jitter 0, phase diffusion 0) | integrated OU, sigma_nu 0.7423, lam 0.0608 |
| amp_exp (line power ~ (f/80)^a) | 8.756 | 0.000 |
| floor_exp | -3.711 | 22.817 |
| floor_static_rel | 0.0149 | 3.8075 |
| floor_mean_db | -27.68 | -38.27 |
| floor_tilt_db_oct | -2.103 | -7.201 |
| comb_gain_db | — (no transplant) | -6.918 |

## MEASURED on the render: prominence over the local floor (dB)

Order-tracked on each frame's own label carrier, 8192-point (1.95 Hz per bin), mic- and rotor-averaged; the local floor is the two-sided 0.45-0.7 fbar annulus median. This is `noise_v2_widen_dregon.line_width_db3`'s own statistic, checked against it at k=1 in `anatomy.json` (`estimator_check`).
An order whose band peak does not clear its own local floor carries no identifiable prominence and is `—`.

| window | arm | k=1 | k=2 | k=3 | k=4 | k=6 | k=8 | k=12 | k=16 | k=24 | k=32 | floor slope dB/oct |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free-flight` | `real` | +6.16 | +2.29 | +1.39 | +0.91 | +1.16 | +0.73 | — | +0.10 | — | +0.01 | -4.68 |
| `free-flight` | `legacy` | +18.20 | +9.26 | +2.00 | +3.60 | +2.58 | +0.90 | +0.01 | +0.45 | +0.88 | +0.30 | -4.67 |
| `free-flight` | `v2` | +7.36 | +5.18 | +1.18 | +1.41 | +0.82 | +0.22 | +0.34 | +0.28 | +0.17 | +0.37 | -4.54 |
| `free-flight` | `v2_plus21db` | +26.96 | +24.08 | +3.69 | +7.06 | +4.20 | +2.06 | +1.90 | +1.82 | +1.76 | +1.61 | -1.71 |
| `hovering` | `real` | +7.53 | +1.86 | +0.34 | — | +0.10 | — | — | — | — | — | -4.86 |
| `hovering` | `legacy` | +18.75 | +7.02 | +2.18 | +2.24 | +0.98 | +0.91 | +0.46 | +0.49 | +0.35 | +0.51 | -4.46 |
| `hovering` | `v2` | +7.45 | +2.27 | +0.26 | +0.24 | +0.02 | — | +0.44 | +0.15 | +0.15 | +0.09 | -4.83 |
| `hovering` | `v2_plus21db` | +26.83 | +21.11 | +3.24 | +6.83 | +3.65 | +2.11 | +1.95 | +1.60 | +1.26 | +1.32 | -1.88 |
| `updown` | `real` | +3.61 | — | — | — | — | — | — | — | +0.39 | +0.39 | -5.41 |
| `updown` | `legacy` | +16.06 | +4.95 | +1.03 | +1.43 | +0.02 | +0.77 | +0.31 | +0.56 | — | +0.24 | -4.90 |
| `updown` | `v2` | +7.46 | +0.92 | — | — | — | — | +0.33 | +0.22 | +0.20 | +0.12 | -5.02 |
| `updown` | `v2_plus21db` | +27.18 | +19.64 | +3.16 | +6.19 | +2.72 | +2.10 | +1.73 | +1.39 | +1.94 | +0.85 | -1.91 |

## MEASURED: -3 dB width and cross-order phase coherence

`|E[exp(i(phi_2k - 2 phi_k))]|` on the +-16 Hz demodulated envelopes of the label's own carriers, averaged over 4 rotors and 8 microphones; 1 = the two orders keep a fixed phase relation over the window, 0 = they do not. `adj` is round 3's adjacent-pair statistic on the same envelopes.

| window | arm | w(k=1) | w(k=2) | w(k=4) | w(k=8) | w(k=16) | B(1,2) | B(2,4) | B(3,6) | B(4,8) | adj |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free-flight` | `real` | 12.0 | — | — | — | 39.6 | 0.096 | 0.075 | 0.057 | 0.066 | 0.070 |
| `free-flight` | `legacy` | 10.4 | 21.3 | 43.2 | 39.3 | 9.8 | 0.215 | 0.061 | 0.052 | 0.057 | 0.091 |
| `free-flight` | `v2` | 11.6 | 20.1 | — | 3.9 | 9.5 | 0.115 | 0.057 | 0.059 | 0.067 | 0.083 |
| `free-flight` | `v2_plus21db` | 10.2 | 17.5 | 5.8 | 6.5 | 12.2 | 0.390 | 0.253 | 0.091 | 0.072 | 0.156 |
| `hovering` | `real` | 11.8 | — | — | — | — | 0.129 | 0.085 | 0.054 | 0.061 | 0.072 |
| `hovering` | `legacy` | 11.0 | 23.3 | 43.1 | 34.5 | 12.4 | 0.205 | 0.062 | 0.062 | 0.067 | 0.095 |
| `hovering` | `v2` | 12.2 | 7.5 | — | — | 8.8 | 0.131 | 0.069 | 0.060 | 0.058 | 0.082 |
| `hovering` | `v2_plus21db` | 10.4 | 17.8 | 5.6 | 6.2 | 11.9 | 0.391 | 0.248 | 0.079 | 0.063 | 0.155 |
| `updown` | `real` | 20.7 | — | — | — | — | 0.079 | 0.076 | 0.059 | 0.055 | 0.076 |
| `updown` | `legacy` | 11.6 | 25.0 | 43.4 | 28.5 | 6.8 | 0.124 | 0.050 | 0.057 | 0.066 | 0.085 |
| `updown` | `v2` | 11.6 | 3.0 | — | — | 8.0 | 0.097 | 0.053 | 0.068 | 0.058 | 0.083 |
| `updown` | `v2_plus21db` | 10.4 | 19.6 | 6.8 | 7.5 | 12.6 | 0.415 | 0.254 | 0.098 | 0.068 | 0.174 |

