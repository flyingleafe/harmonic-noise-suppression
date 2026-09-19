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

## Second tracker `real_r4_scv2_unified` / `best_real_r2.ckpt` (rev/s)

Same renders, same eight microphones, same regime support; sha256 `93f62d19c9022f43dcf9e1e53d0a1b65c26aee3d6dcb70223f988675a9686749`. The rate REGRESSOR's own `rps_pred` track is read directly and PIT-assigned, where the HPPNet column reads salience layers by peak + parabola (`scripts/_synthetic_probe.py::score`).

| arm | free-flight | hovering | updown | mean | HPPNet mean |
|---|---:|---:|---:|---:|---:|
| `real` | 0.983 | 1.314 | 2.546 | 1.614 | 1.074 |
| `legacy` | 1.273 | 1.547 | 1.963 | 1.594 | 1.963 |
| `v2` | 11.270 | 11.376 | 11.245 | 11.297 | 69.555 |
| `v2_plus21db` | 2.375 | 2.422 | 3.089 | 2.629 | 4.893 |
| `v2_gx10_plus12db` | 1.418 | 1.874 | 2.244 | 1.845 | 39.407 |
| `v2_glegacy_plus12db` | 2.237 | 2.298 | 2.138 | 2.224 | 47.710 |

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

**Second tracker: the render's penalty is 6x smaller for a regressor than for
HPPNet, but the ORDERING is the same.** `real_r4_scv2_unified` /
`best_real_r2.ckpt` (the real-trained SCv2 regressor; validation round 24,
`real_r2` 3.43, `real_r3` 3.12 rev/s; sha256 `93f62d19…6749`) was run on the
same six arms, the same renders (the merge refuses unless every arm's measured
-3 dB line width matches bit for bit), the same eight microphones and the same
regime support. It reads its own `rps_pred` track rather than salience layers
— the regressor branch added to `scripts/_synthetic_probe.py::score`, which
leaves the HPPNet path numerically identical (it reproduces this file's real
column, 0.6381291290613775 / 0.8884082796921247 / 1.694930322906428, to the
last digit).

Read against HPPNet (means over the three windows): real 1.61 against 1.07,
legacy 1.59 against 1.96, **v2 as fitted 11.30 against 69.56**, v2 +21 dB 2.63
against 4.89, gamma x10 at +12 dB 1.85 against 39.41, the legacy width law at
+12 dB 2.22 against 47.71. Three things follow.

1. **The defect is real and both trackers see it.** v2 as fitted is the worst
   arm for both, by 7x over its own real clip for SCv2 (11.30 against 1.61) and
   by 65x for HPPNet. No tracker mistakes the as-fitted render for real audio.
2. **The SIZE of the penalty is tracker-specific.** HPPNet's 70 rev/s is a
   total loss of lock — it is a salience model that needs the comb to stand out
   of the floor and the v2 comb does not. The regressor degrades to 11 rev/s
   instead: it regresses a rate from the whole spectrum and keeps a usable,
   heavily biased estimate when the comb is buried. Any claim of the form "the
   render costs N rev/s" must therefore name its tracker; only the ORDERING is
   portable.
3. **The re-levelled and widened arms are nearly transparent to the
   regressor.** gamma x10 at +12 dB scores 1.85 against real 1.61 and legacy
   1.59 — inside the real clip's own window-to-window spread (0.98-2.55) — while
   HPPNet still puts it at 39.4. So the level lever that R3 identified fixes
   the regressor almost completely and the salience tracker only partly; the
   remaining HPPNet gap at +21 dB (4.89 against real 1.07) is the part of the
   comb's SHAPE, not its level, that is still wrong.

Commands, both from this checkout:

```
python scripts/noise_v2_widen_dregon.py --probe \
  --probe-experiment real_r4_scv2_unified --probe-ckpt best_real_r2.ckpt \
  --arms real,legacy,v2,v2_plus21db,v2_gx10_plus12db,v2_glegacy_plus12db \
  --stem widen_scv2 --out results/noise_v2/rounds/round3/dregon_humps
python scripts/noise_v2_widen_dregon.py --stem widen \
  --out results/noise_v2/rounds/round3/dregon_humps \
  --merge-second results/noise_v2/rounds/round3/dregon_humps/widen_scv2.json
```

The earlier record in this file that the SCv2 column was blocked applied to
`ctrl_diverse_scv2_unified` and to the salience-only PIT path. Both are
superseded: the path now scores regressors, and `ctrl_diverse` was dropped
because it is the synthetic-diverse CONTROL arm and does not track real DREGON
at all — measured here at 41.5 / 46.8 / 49.0 rev/s on the three REAL clips,
consistent with its own validation (`real_nosource` 14.3, `real_r3` 20.2,
`real_r2` 29.0 rev/s measured on the frozen valid split), so it is not a "best
SCv2" and no column from it is reported.
