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
| `free-flight` | `v2_gx10_plus12db` | 10.3 | 17.5 | 37.6 | +18.01 dB |
| `free-flight` | `v2_glegacy_plus12db` | 26.6 | 35.5 | — | +7.36 dB |
| `hovering` | `real` | 11.8 | — | — | +7.53 dB |
| `hovering` | `legacy` | 11.0 | 23.3 | 43.1 | +18.75 dB |
| `hovering` | `v2` | 12.2 | 7.5 | — | +7.45 dB |
| `hovering` | `v2_plus21db` | 10.4 | 17.8 | 5.6 | +26.83 dB |
| `hovering` | `v2_gx10_plus12db` | 10.6 | 17.8 | 6.8 | +17.89 dB |
| `hovering` | `v2_glegacy_plus12db` | 31.7 | 26.4 | 6.2 | +7.13 dB |
| `updown` | `real` | 20.7 | — | — | +3.61 dB |
| `updown` | `legacy` | 11.6 | 25.0 | 43.4 | +16.06 dB |
| `updown` | `v2` | 11.6 | 3.0 | — | +7.46 dB |
| `updown` | `v2_plus21db` | 10.4 | 19.6 | 6.8 | +27.18 dB |
| `updown` | `v2_gx10_plus12db` | 10.4 | 10.2 | 4.2 | +18.32 dB |
| `updown` | `v2_glegacy_plus12db` | 35.3 | 18.9 | — | +7.03 dB |

## HPPNet PIT MAE (rev/s)

| arm | free-flight | hovering | updown | mean |
|---|---:|---:|---:|---:|
| `real` | 0.983 | 1.314 | 2.546 | 1.614 |
| `legacy` | 1.273 | 1.547 | 1.963 | 1.594 |
| `v2` | 11.270 | 11.376 | 11.245 | 11.297 |
| `v2_plus21db` | 2.375 | 2.422 | 3.089 | 2.629 |
| `v2_gx10_plus12db` | 1.418 | 1.874 | 2.244 | 1.845 |
| `v2_glegacy_plus12db` | 2.237 | 2.298 | 2.138 | 2.224 |

| arm | what it is |
|---|---|
| `real` | the real DREGON room-2 clip |
| `legacy` | legacy stage-2 baseline, identity-matched |
| `v2` | the round-3 v2 candidate as fitted |
| `v2_plus21db` | v2 with the comb 21 dB up (the level optimum) |
| `v2_gx10_plus12db` | gamma_rk x10, comb 12 dB up |
| `v2_glegacy_plus12db` | gamma_rk = 13.068 + 0.211k Hz (the legacy width law), comb 12 dB up |

Record `1255f43841c3bf9be8227f1b0409baadf25301e2`.
