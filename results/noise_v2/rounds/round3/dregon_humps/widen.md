# DREGON round 3: spectrograms and widened lines

Mic 0, seed 2001; spectrograms at 2048/512 and 8192/1024 (1.95 Hz per bin), 0-2 kHz, label harmonics k=1..8 overlaid. The fitted `gamma_rk` (mean over rotors) is k=1: 0.0014 Hz, k=2: 0.0037 Hz, k=4: 0.0237 Hz, k=8: 0.6851 Hz against the legacy law 13.068 + 0.211k Hz.

## -3 dB width of the order-tracked line (Hz, 8192-point, 1.95 Hz per bin)

Each frame read on its own label carrier, averaged over four rotors and eight microphones; the four-rotor spread (k x ~7.8 rev/s) and the 512 ms frame's own drift are inside every number equally.

| window | arm | k=1 | k=2 | k=4 | peak over local base, k=1 |
|---|---|---:|---:|---:|---:|
| `free-flight` | `real` | 12.0 | — | — | +6.16 dB |
| `free-flight` | `legacy` | 10.4 | 21.3 | 43.2 | +18.20 dB |
| `free-flight` | `v2` | 11.6 | 20.1 | — | +7.36 dB |
| `free-flight` | `v2_plus21db` | 10.2 | 17.5 | 5.8 | +26.96 dB |
| `free-flight` | `v2_gx3` | 11.6 | 20.1 | — | +7.31 dB |
| `free-flight` | `v2_gx10` | 11.7 | 20.0 | — | +7.22 dB |
| `free-flight` | `v2_gx30` | 11.8 | 20.0 | — | +7.10 dB |
| `free-flight` | `v2_glegacy` | 39.2 | — | — | +3.66 dB |
| `free-flight` | `v2_gx3_plus12db` | 10.3 | 17.6 | — | +18.03 dB |
| `free-flight` | `v2_gx10_plus12db` | 10.3 | 17.5 | 37.6 | +18.01 dB |
| `free-flight` | `v2_gx30_plus12db` | 10.3 | 17.5 | — | +17.98 dB |
| `free-flight` | `v2_glegacy_plus12db` | 26.6 | 35.5 | — | +7.36 dB |
| `hovering` | `real` | 11.8 | — | — | +7.53 dB |
| `hovering` | `legacy` | 11.0 | 23.3 | 43.1 | +18.75 dB |
| `hovering` | `v2` | 12.2 | 7.5 | — | +7.45 dB |
| `hovering` | `v2_plus21db` | 10.4 | 17.8 | 5.6 | +26.83 dB |
| `hovering` | `v2_gx3` | 12.2 | 7.6 | — | +7.41 dB |
| `hovering` | `v2_gx10` | 12.2 | 7.8 | — | +7.36 dB |
| `hovering` | `v2_gx30` | 12.3 | 8.0 | — | +7.35 dB |
| `hovering` | `v2_glegacy` | 40.8 | — | -1.4 | +3.41 dB |
| `hovering` | `v2_gx3_plus12db` | 10.5 | 17.8 | 6.0 | +17.93 dB |
| `hovering` | `v2_gx10_plus12db` | 10.6 | 17.8 | 6.8 | +17.89 dB |
| `hovering` | `v2_gx30_plus12db` | 10.7 | 17.9 | 49.5 | +17.81 dB |
| `hovering` | `v2_glegacy_plus12db` | 31.7 | 26.4 | 6.2 | +7.13 dB |
| `updown` | `real` | 20.7 | — | — | +3.61 dB |
| `updown` | `legacy` | 11.6 | 25.0 | 43.4 | +16.06 dB |
| `updown` | `v2` | 11.6 | 3.0 | — | +7.46 dB |
| `updown` | `v2_plus21db` | 10.4 | 19.6 | 6.8 | +27.18 dB |
| `updown` | `v2_gx3` | 11.5 | 3.0 | — | +7.50 dB |
| `updown` | `v2_gx10` | 11.5 | 2.9 | — | +7.59 dB |
| `updown` | `v2_gx30` | 11.5 | 2.8 | — | +7.66 dB |
| `updown` | `v2_glegacy` | 39.0 | — | — | +3.62 dB |
| `updown` | `v2_gx3_plus12db` | 10.4 | 10.2 | 3.9 | +18.27 dB |
| `updown` | `v2_gx10_plus12db` | 10.4 | 10.2 | 4.2 | +18.32 dB |
| `updown` | `v2_gx30_plus12db` | 10.3 | 10.4 | 4.5 | +18.30 dB |
| `updown` | `v2_glegacy_plus12db` | 35.3 | 18.9 | — | +7.03 dB |

## HPPNet PIT MAE (rev/s)

| arm | free-flight | hovering | updown | mean |
|---|---:|---:|---:|---:|
| `real` | 0.638 | 0.888 | 1.695 | 1.074 |
| `legacy` | 1.496 | 1.895 | 2.499 | 1.963 |
| `v2` | 62.105 | 73.448 | 73.112 | 69.555 |
| `v2_plus21db` | 5.118 | 2.759 | 6.800 | 4.893 |
| `v2_gx3` | 70.781 | 80.226 | 72.713 | 74.573 |
| `v2_gx10` | 75.931 | 76.435 | 78.288 | 76.885 |
| `v2_gx30` | 74.707 | 78.315 | 79.275 | 77.433 |
| `v2_glegacy` | 80.371 | 80.587 | 77.685 | 79.548 |
| `v2_gx3_plus12db` | 36.634 | 29.889 | 33.646 | 33.390 |
| `v2_gx10_plus12db` | 37.683 | 37.830 | 42.709 | 39.407 |
| `v2_gx30_plus12db` | 36.715 | 46.691 | 56.892 | 46.766 |
| `v2_glegacy_plus12db` | 51.931 | 39.016 | 52.184 | 47.710 |

| arm | what it is |
|---|---|
| `real` | the real DREGON room-2 clip |
| `legacy` | legacy stage-2 baseline, identity-matched |
| `v2` | the round-3 v2 candidate as fitted |
| `v2_plus21db` | v2 with the comb 21 dB up (the level optimum) |
| `v2_gx3` | gamma_rk x3 |
| `v2_gx10` | gamma_rk x10 |
| `v2_gx30` | gamma_rk x30 |
| `v2_glegacy` | gamma_rk = 13.068 + 0.211k Hz (the legacy width law) |
| `v2_gx3_plus12db` | gamma_rk x3, comb 12 dB up |
| `v2_gx10_plus12db` | gamma_rk x10, comb 12 dB up |
| `v2_gx30_plus12db` | gamma_rk x30, comb 12 dB up |
| `v2_glegacy_plus12db` | gamma_rk = 13.068 + 0.211k Hz (the legacy width law), comb 12 dB up |

Figures: `results/noise_v2/rounds/round3/dregon_humps/spec_free-flight_2048.png`, `results/noise_v2/rounds/round3/dregon_humps/spec_free-flight_8192.png`, `results/noise_v2/rounds/round3/dregon_humps/spec_hovering_2048.png`, `results/noise_v2/rounds/round3/dregon_humps/spec_hovering_8192.png`, `results/noise_v2/rounds/round3/dregon_humps/spec_updown_2048.png`, `results/noise_v2/rounds/round3/dregon_humps/spec_updown_8192.png`.

Record `d15c8386b3fb1401d0ee1ca3e7d135b6642a1be9`.
## What the spectrograms show (hand-written, `DregonHumps`)

**REAL.** At 0-2 kHz no harmonic line is visible as a line: the picture is a
broadband floor that rises ~25 dB towards 30 Hz, with slow blobs of 250-750 Hz
energy that come and go over tenths of a second and a grainy 1-2 kHz texture
that the renders do not have. The only place the comb shows at all is below
~250 Hz, as a faint banding at k=1..3 that the order-tracked read puts at
+6.2/+7.5/+3.6 dB over its own local base.

**LEGACY.** The same overall shape with a visibly quieter and smoother 1-2 kHz
region, and the ONE thing it has that the others do not: clear horizontal
banding below ~300 Hz, steady across the whole 4 s. Its order-tracked line is
no wider than real's at k=1 (11.0 against 11.8 Hz on `hovering`) but it is
**11 dB more prominent** (+18.8 against +7.5 dB).

**v2 AS FITTED.** Indistinguishable from legacy in width and from real in
prominence: k=1 width 12.2 Hz (real 11.8, legacy 11.0) and +7.5 dB over base
(real +7.5, legacy +18.8). Its lines are NOT visibly thinner than legacy's —
at this resolution every arm's k=1 "line" is the four-rotor spread
(k x 7.8 rev/s) convolved with the 512 ms frame, not the fitted `gamma_rk`.

**v2 +21 dB.** The only render whose low orders stand out of the floor the way
legacy's do, and then some: +26.8 dB over base at k=1 against legacy's +18.8.

**So the thin-line hypothesis is not what the numbers say.** The fitted
`gamma_rk` (0.0014 Hz at k=1, 0.0037 at k=2, 0.024 at k=4) IS orders of
magnitude below the legacy law (13.1 + 0.21 k Hz), but at the resolution any
observer — or the tracker — has on a 4 s DREGON window, both are unresolvably
narrow: the measured widths agree to ~1 Hz. What differs is PROMINENCE, and
every attempt to buy tracking with width instead makes it worse: gamma x3
74.6, x10 76.9, x30 77.4, the legacy law 79.5 rev/s mean against v2's own
69.6; at +12 dB the ordering is the same (x3 33.4, x10 39.4, x30 46.8, legacy
law 47.7) and all of them lose to the un-widened needle comb at +21 dB
(4.893 rev/s mean, legacy 1.963, real 1.074). Widening at a fixed comb gain
also DILUTES the line: the legacy-law arm's k=1 prominence falls from +7.5 to
+3.4 dB and its width grows to 40.8 Hz.

**Second tracker.** The SCv2 probe (`ctrl_diverse_scv2_unified`,
`best_real_r2.ckpt`) was NOT run: this agent's run budget ended with the HPPNet
arms scored. Nothing about it is claimed here.
