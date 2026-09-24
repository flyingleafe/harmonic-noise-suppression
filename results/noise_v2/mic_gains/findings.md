# Per-microphone level deviations on the real windows: one gain, or not?

`scripts/_mic_gain_rank.py` at `a329f05bf872`.

Measurement only -- no model is evaluated and nothing is fitted to a likelihood. `L[window, mic, band]` is the 1/3-octave level (30-7900 Hz) of every real window of each rig, in four variants: `full`, `floor` (every bin within +-2 Hz -- one bin on the flight front end -- of a rotor order at the window's own per-frame carriers removed) and `comb` (the order-tracked line peaks, power-averaged over the orders inside a band) and `floor_q25` (the 0.25 quantile of the same unmasked cells, a floor a stray tonal cannot lift). `D = L - mean_over_mics(L)` is the per-mic deviation, and the models fitted to it are the indicator partitions `const` (`g[mic]`), `per_band` (`g[mic, band]`), `per_regime` (`g[mic, regime]`; on the bench a regime is one MOTOR), `per_recording` (`g[mic, recording]`) and `per_window` (`g[mic, window]`). Each is fitted by least squares, which for an indicator partition is the group mean.

## Windows

| set | windows | mics | duration | carrier rev/s (min-max of window means) | recordings |
| --- | --- | --- | --- | --- | --- |
| dregon/dregon_score | 5 | 8 | 4 s | 79.4-80.9 | 5 |
| dregon/dregon_fit | 5 | 8 | 8 s | 80.5-81.2 | 5 |
| dregon/dregon_extra | 40 | 8 | 4 s | 78.7-82.4 | 6 |
| michaels/michaels_cruise | 8 | 8 | 8 s | 80.3-81.0 | 1 (FLY125) |
| michaels/michaels_standby | 3 | 8 | 4 s | 33.6-35.6 | 1 (FLY125) |
| dregon_bench/bench_Motor1 | 5 | 8 | 7.29/12.79/21/25.71/41 s | 49.0-88.2 | 5 |
| dregon_bench/bench_Motor2 | 5 | 8 | 7.1/11.66/11.8/12.27/12.42 s | 48.4-86.7 | 5 |
| dregon_bench/bench_Motor3 | 5 | 8 | 11.8/11.83/11.88/12.38/12.58 s | 49.1-88.2 | 5 |
| dregon_bench/bench_Motor4 | 5 | 8 | 11.64/11.71/11.83/11.99/12.04 s | 49.8-89.2 | 5 |

Band grid: 24 kept 1/3-octave bands on the flight front end (df = 7.812 Hz). Bands whose every bin is occupied by an order in every frame have no floor and are reported as `n/a`.

`dregon` and `michaels` are the two REAL RIGS -- the flight windows of a complete eight-microphone array on a flying drone. `dregon_bench` is AUXILIARY: one motor at a time on a bench, four different source positions pooled, present only because it is the only DREGON material with a real rotor-speed span. Its rank test is reported for completeness and answers a different question (source geometry), not `can the channel gains be normalised once`.

## Summary

| set | windows | floor `const` var. expl. | floor `const` residual dB | floor `per_band` residual dB | comb-vs-floor r | excess <500 Hz spread dB | excess >500 Hz spread dB | speed span oct | max speed slope dB/oct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon | 50 | 35.9% | 3.35 | 2.14 | 0.846 | 4.94 | 1.44 | 0.07 | unidentified |
| michaels | 11 | 75.7% | 1.62 | 0.87 | 0.996 | 2.44 | 3.70 | 1.27 | 1.13 (3/8 sig.) |
| dregon_bench | 20 | 4.5% | 6.84 | 6.70 | 0.747 | 2.28 | 1.69 | 0.85 | 3.00 (0/8 sig.) |

## dregon

### Rank test

| variant | model | params | var. explained | residual RMS dB | held-out RMS dB | 30-300 | 300-1k | 1k-3k | 3k-8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | const | 8 | 35.3% | 2.99 | 3.00 | 3.55 | 2.43 | 2.65 | 2.45 |
| full | per_band | 192 | 73.6% | 1.91 | 1.95 | 2.46 | 2.04 | 0.86 | 0.82 |
| full | per_regime | 8 | 35.3% | 2.99 | 3.00 | 3.55 | 2.43 | 2.65 | 2.45 |
| full | per_recording | 48 | 39.4% | 2.90 | 2.96 | 3.41 | 2.21 | 2.68 | 2.49 |
| full | per_window | 400 | 47.3% | 2.70 | n/a | 3.04 | 1.76 | 2.78 | 2.69 |
| floor | const | 8 | 35.9% | 3.35 | 3.36 | 3.86 | 2.94 | 3.04 | 2.86 |
| floor | per_band | 192 | 73.8% | 2.14 | 2.19 | 2.75 | 2.43 | 0.88 | 0.83 |
| floor | per_regime | 8 | 35.9% | 3.35 | 3.36 | 3.86 | 2.94 | 3.04 | 2.86 |
| floor | per_recording | 48 | 39.2% | 3.26 | 3.33 | 3.73 | 2.73 | 3.07 | 2.90 |
| floor | per_window | 400 | 47.2% | 3.04 | n/a | 3.25 | 2.28 | 3.20 | 3.13 |
| comb | const | 8 | 31.2% | 2.67 | 2.68 | 4.04 | 2.37 | 1.95 | 1.73 |
| comb | per_band | 160 | 70.7% | 1.74 | 1.78 | 2.66 | 1.92 | 0.86 | 0.83 |
| comb | per_regime | 8 | 31.2% | 2.67 | 2.68 | 4.04 | 2.37 | 1.95 | 1.73 |
| comb | per_recording | 48 | 36.2% | 2.57 | 2.61 | 3.87 | 2.18 | 1.97 | 1.77 |
| comb | per_window | 400 | 43.4% | 2.42 | n/a | 3.59 | 1.78 | 2.02 | 1.93 |
| floor_q25 | const | 8 | 36.9% | 3.40 | 3.41 | 4.10 | 2.73 | 3.04 | 2.71 |
| floor_q25 | per_band | 192 | 76.0% | 2.10 | 2.15 | 2.84 | 2.14 | 0.80 | 0.50 |
| floor_q25 | per_regime | 8 | 36.9% | 3.40 | 3.41 | 4.10 | 2.73 | 3.04 | 2.71 |
| floor_q25 | per_recording | 48 | 39.1% | 3.34 | 3.41 | 4.00 | 2.58 | 3.05 | 2.78 |
| floor_q25 | per_window | 400 | 46.7% | 3.13 | n/a | 3.51 | 2.20 | 3.21 | 3.04 |

`held-out RMS` is leave-one-window-out: the model is fitted on the other windows and asked to predict the held-out one. `per_window` has no out-of-sample prediction by construction, hence `n/a`.

Total deviation RMS (the thing being explained): full 3.72 dB, floor 4.18 dB, comb 3.22 dB, floor_q25 4.28 dB.

### The eight gains (`const` model), dB relative to the mic mean

| quantity | mic 1 | mic 2 | mic 3 | mic 4 | mic 5 | mic 6 | mic 7 | mic 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full gain | +1.45 | +3.65 | -2.68 | -1.85 | +2.89 | -0.76 | -0.84 | -1.86 |
| floor gain | +2.04 | +3.66 | -2.50 | -2.66 | +3.45 | -0.85 | -0.60 | -2.54 |
| comb gain | +0.45 | +3.31 | -3.12 | -0.67 | +1.75 | -0.18 | -1.12 | -0.41 |
| floor_q25 gain | +2.26 | +3.55 | -2.75 | -2.61 | +3.76 | -0.81 | -0.88 | -2.52 |
| full sd over bands | 2.44 | 1.99 | 1.34 | 2.68 | 3.50 | 1.62 | 0.79 | 3.13 |
| floor sd over bands | 2.85 | 1.73 | 1.23 | 3.21 | 3.94 | 1.65 | 0.59 | 3.67 |
| comb sd over bands | 2.02 | 2.93 | 1.21 | 2.25 | 3.31 | 1.57 | 1.25 | 2.33 |
| floor_q25 sd over bands | 2.99 | 1.72 | 1.47 | 3.18 | 4.35 | 1.62 | 0.84 | 3.65 |
| full sd over windows | 1.46 | 1.33 | 1.29 | 0.77 | 1.55 | 1.09 | 1.43 | 1.32 |
| floor sd over windows | 1.52 | 1.42 | 1.39 | 0.88 | 1.82 | 1.30 | 1.47 | 1.36 |
| comb sd over windows | 1.23 | 1.15 | 1.09 | 0.62 | 1.30 | 0.82 | 1.29 | 1.36 |
| floor_q25 sd over windows | 1.64 | 1.41 | 1.32 | 0.62 | 1.90 | 1.07 | 1.51 | 0.94 |
| full sd over recordings | 0.97 | 0.73 | 0.63 | 0.35 | 0.73 | 0.64 | 0.85 | 0.95 |
| floor sd over recordings | 0.91 | 0.77 | 0.66 | 0.21 | 0.92 | 0.69 | 0.90 | 0.80 |
| comb sd over recordings | 0.84 | 0.65 | 0.58 | 0.35 | 0.55 | 0.43 | 0.76 | 1.12 |
| floor_q25 sd over recordings | 1.12 | 0.61 | 0.40 | 0.29 | 0.71 | 0.41 | 0.80 | 0.51 |
| comb minus floor | -1.59 | -0.36 | -0.62 | +1.99 | -1.69 | +0.67 | -0.52 | +2.12 |

comb-vs-floor gain correlation over the 8 channels: r = 0.846; RMS difference 1.37 dB; largest |difference| 2.12 dB.

Contrast the comb/floor split actually has (mic- and window-mean `comb - floor`): 30-300 +6.10 dB, 300-1k +3.57 dB, 1k-3k +2.70 dB, 3k-8k +2.17 dB. The order mask covers 73% of the (frame, bin) cells on average, so where the contrast is small the two variants are largely reading the same spectrum.

Robustness of the floor: the `floor_q25` gains (a 0.25 quantile of the SAME unmasked cells instead of their power mean) correlate with the `floor` gains at r = 0.998, RMS difference 0.20 dB, largest 0.32 dB; its low-band spread over channels is 5.16 dB against 1.43 dB above 500 Hz (power-mean floor: 4.94 / 1.44 dB).

### Residual RMS of the `const` model per window (floor variant), dB

| window | residual RMS dB |
| --- | --- |
| flight_dregon_rectangle_nosource_room2@1511905731.953+8_motors_command | 4.96 |
| flight_dregon_free-flight_nosource_room2@1512727425.205+4_motors_command | 4.60 |
| flight_dregon_free-flight_nosource_room1@1519672662.061+4_motors_command | 4.17 |
| flight_dregon_spinning_nosource_room2@1511905215.335+4_motors_command | 4.14 |
| flight_dregon_free-flight_nosource_room1@1519672674.061+4_motors_command | 4.14 |
| flight_dregon_free-flight_nosource_room1@1519672670.061+4_motors_command | 3.83 |
| flight_dregon_free-flight_nosource_room1@1519672686.061+4_motors_command | 3.81 |
| flight_dregon_free-flight_nosource_room1@1519672678.061+4_motors_command | 3.79 |
| ... 42 more | <= 3.75 |

### Wind test (floor excess)

Window rotor-mean carriers span 78.7-82.4 rev/s (0.07 octaves).
That is under the 0.25-octave minimum this report demands of a speed axis, so the `slope` rows below are UNIDENTIFIED -- they are tabled only to show that their standard errors are of the same size.

| quantity | mic 1 | mic 2 | mic 3 | mic 4 | mic 5 | mic 6 | mic 7 | mic 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| excess 30-300 Hz | +4.68 | +5.45 | -1.86 | -5.72 | +6.69 | -2.27 | -0.32 | -6.66 |
| excess 300-1k Hz | +2.89 | +2.96 | -4.20 | -2.89 | +5.40 | -1.34 | -1.40 | -1.41 |
| excess 1k-3k Hz | -1.29 | +2.20 | -2.36 | +0.81 | -1.23 | +0.96 | -0.36 | +1.28 |
| excess 3k-8k Hz | -1.11 | +2.49 | -2.23 | +0.61 | -1.12 | +1.06 | -0.64 | +0.94 |
| excess < 500 Hz | +4.63 | +5.17 | -2.27 | -5.61 | +6.75 | -2.23 | -0.52 | -5.92 |
| excess > 500 Hz | -0.43 | +2.35 | -2.76 | +0.18 | +0.19 | +0.54 | -0.70 | +0.63 |
| slope <500 Hz dB/oct | +50.13 | +83.13 | +49.73 | -43.90 | +50.13 | -52.69 | -35.82 | -100.71 |
| slope stderr | 33.32 | 30.42 | 31.86 | 22.53 | 40.10 | 32.01 | 33.15 | 30.45 |
| slope full band dB/oct | +24.17 | +58.55 | +22.58 | -20.14 | +29.82 | -25.53 | -28.68 | -60.78 |

Spread over channels: 4.94 dB below 500 Hz against 1.44 dB above it. Groups (one free intercept each in the within-group fit): flight x50; largest within-group speed span 0.07 octaves, below the 0.25-octave minimum, so the within-group slope is unidentified and not tabled.

### Verdict

- (i) full: the rank-one `g[mic]` model explains 35.3% of the deviation with a 2.99 dB residual (3.00 dB held out), against 73.6% / 1.91 dB (1.95 dB held out) for `g[mic, band]`, 35.3% / 2.99 dB for `g[mic, regime]`, 39.4% / 2.90 dB for `g[mic, recording]` and 47.3% / 2.70 dB for `g[mic, window]` -- one constant per channel is NOT enough; the deviation is frequency-dependent (a per-channel transfer function, not a gain), and the richer per-band description also predicts a held-out window better.
- (i) floor: the rank-one `g[mic]` model explains 35.9% of the deviation with a 3.35 dB residual (3.36 dB held out), against 73.8% / 2.14 dB (2.19 dB held out) for `g[mic, band]`, 35.9% / 3.35 dB for `g[mic, regime]`, 39.2% / 3.26 dB for `g[mic, recording]` and 47.2% / 3.04 dB for `g[mic, window]` -- one constant per channel is NOT enough; the deviation is frequency-dependent (a per-channel transfer function, not a gain), and the richer per-band description also predicts a held-out window better.
- (ii) comb and floor gains correlate at r = 0.846 with an RMS difference of 1.37 dB (largest 2.12 dB) -- the lines and the floor do NOT see the same per-channel gain; the split has +6.1 dB of comb-over-floor contrast at 30-300 Hz and +2.2 dB at 3-8 kHz, with 73% of the cells masked.
- (iii) the floor excess spreads 4.94 dB over channels below 500 Hz against 1.44 dB above it (5.16 / 1.43 dB on the quantile floor), which IS the low-frequency concentration wind/self-noise has; the speed axis is NOT identified on these windows (0.07 octaves of carrier span, pooled slopes up to 100.71 dB/oct at standard errors up to 40.10 dB/oct), so the speed question is left to the sets that do have a span.

## michaels

### Rank test

| variant | model | params | var. explained | residual RMS dB | held-out RMS dB | 30-300 | 300-1k | 1k-3k | 3k-8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | const | 8 | 80.3% | 1.47 | 1.49 | 1.60 | 0.79 | 1.16 | 2.02 |
| full | per_band | 192 | 91.4% | 0.97 | 1.07 | 1.09 | 0.57 | 1.04 | 0.96 |
| full | per_regime | 16 | 81.6% | 1.42 | 1.44 | 1.63 | 0.77 | 0.95 | 1.89 |
| full | per_recording | 8 | 80.3% | 1.47 | 1.49 | 1.60 | 0.79 | 1.16 | 2.02 |
| full | per_window | 88 | 82.7% | 1.38 | n/a | 1.57 | 0.72 | 0.90 | 1.87 |
| floor | const | 8 | 75.7% | 1.62 | 1.63 | 1.84 | 1.26 | 1.10 | 2.01 |
| floor | per_band | 192 | 93.1% | 0.87 | 0.97 | 0.92 | 0.56 | 0.90 | 0.96 |
| floor | per_regime | 16 | 76.7% | 1.59 | 1.60 | 1.83 | 1.30 | 1.05 | 1.91 |
| floor | per_recording | 8 | 75.7% | 1.62 | 1.63 | 1.84 | 1.26 | 1.10 | 2.01 |
| floor | per_window | 88 | 77.9% | 1.54 | n/a | 1.76 | 1.28 | 1.01 | 1.90 |
| comb | const | 8 | 82.6% | 1.43 | 1.45 | 1.61 | 0.81 | 1.19 | 1.90 |
| comb | per_band | 184 | 92.4% | 0.94 | 1.06 | 0.99 | 0.63 | 1.10 | 0.98 |
| comb | per_regime | 16 | 84.8% | 1.33 | 1.35 | 1.62 | 0.83 | 0.92 | 1.72 |
| comb | per_recording | 8 | 82.6% | 1.43 | 1.45 | 1.61 | 0.81 | 1.19 | 1.90 |
| comb | per_window | 88 | 85.8% | 1.29 | n/a | 1.58 | 0.76 | 0.84 | 1.69 |
| floor_q25 | const | 8 | 69.8% | 1.79 | 1.80 | 1.94 | 1.52 | 1.35 | 2.18 |
| floor_q25 | per_band | 192 | 92.1% | 0.92 | 1.02 | 0.91 | 0.62 | 0.98 | 1.06 |
| floor_q25 | per_regime | 16 | 70.2% | 1.78 | 1.79 | 1.93 | 1.53 | 1.36 | 2.14 |
| floor_q25 | per_recording | 8 | 69.8% | 1.79 | 1.80 | 1.94 | 1.52 | 1.35 | 2.18 |
| floor_q25 | per_window | 88 | 71.4% | 1.74 | n/a | 1.87 | 1.51 | 1.31 | 2.13 |

`held-out RMS` is leave-one-window-out: the model is fitted on the other windows and asked to predict the held-out one. `per_window` has no out-of-sample prediction by construction, hence `n/a`.

Total deviation RMS (the thing being explained): full 3.31 dB, floor 3.29 dB, comb 3.42 dB, floor_q25 3.25 dB.

### The eight gains (`const` model), dB relative to the mic mean

| quantity | mic 1 | mic 2 | mic 3 | mic 4 | mic 5 | mic 6 | mic 7 | mic 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full gain | +0.77 | -5.84 | +0.78 | +1.24 | +4.40 | +2.12 | -0.38 | -3.09 |
| floor gain | +0.57 | -5.26 | +0.61 | +1.27 | +4.51 | +2.03 | -0.45 | -3.28 |
| comb gain | +0.64 | -6.08 | +1.04 | +1.60 | +4.47 | +2.20 | -0.53 | -3.34 |
| floor_q25 gain | +0.60 | -4.55 | +0.49 | +1.31 | +4.50 | +1.81 | -0.70 | -3.46 |
| full sd over bands | 1.46 | 1.15 | 0.88 | 1.68 | 0.78 | 0.68 | 0.87 | 1.14 |
| floor sd over bands | 1.52 | 2.31 | 0.88 | 1.86 | 1.04 | 1.35 | 1.40 | 1.32 |
| comb sd over bands | 1.49 | 1.04 | 0.88 | 1.94 | 0.88 | 0.71 | 0.88 | 1.69 |
| floor_q25 sd over bands | 1.59 | 2.93 | 1.07 | 1.74 | 0.83 | 1.66 | 1.17 | 1.23 |
| full sd over windows | 0.16 | 1.19 | 0.28 | 0.14 | 0.29 | 0.41 | 0.67 | 0.32 |
| floor sd over windows | 0.16 | 0.90 | 0.59 | 0.52 | 0.37 | 0.46 | 0.71 | 0.46 |
| comb sd over windows | 0.27 | 1.42 | 0.27 | 0.26 | 0.36 | 0.56 | 0.65 | 0.21 |
| floor_q25 sd over windows | 0.30 | 0.75 | 0.50 | 0.47 | 0.29 | 0.29 | 0.57 | 0.39 |
| full sd over recordings | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| floor sd over recordings | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| comb sd over recordings | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| floor_q25 sd over recordings | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| comb minus floor | +0.08 | -0.82 | +0.43 | +0.33 | -0.04 | +0.16 | -0.07 | -0.06 |

comb-vs-floor gain correlation over the 8 channels: r = 0.996; RMS difference 0.35 dB; largest |difference| 0.82 dB.

Contrast the comb/floor split actually has (mic- and window-mean `comb - floor`): 30-300 +12.28 dB, 300-1k +9.92 dB, 1k-3k +3.42 dB, 3k-8k +2.23 dB. The order mask covers 81% of the (frame, bin) cells on average, so where the contrast is small the two variants are largely reading the same spectrum.

Robustness of the floor: the `floor_q25` gains (a 0.25 quantile of the SAME unmasked cells instead of their power mean) correlate with the `floor` gains at r = 0.996, RMS difference 0.29 dB, largest 0.71 dB; its low-band spread over channels is 2.17 dB against 3.67 dB above 500 Hz (power-mean floor: 2.44 / 3.70 dB).

### Residual RMS of the `const` model per window (floor variant), dB

| window | residual RMS dB |
| --- | --- |
| flight_michaels_FLY125@32.000+8_rps_refined | 1.83 |
| flight_michaels_FLY125@128.000+8_rps_refined | 1.80 |
| flight_michaels_FLY125@64.000+8_rps_refined | 1.77 |
| flight_michaels_FLY125@112.000+8_rps_refined | 1.75 |
| flight_michaels_FLY125@96.000+8_rps_refined | 1.62 |
| flight_michaels_FLY125@144.000+8_rps_refined | 1.52 |
| flight_michaels_FLY125@48.000+8_rps_refined | 1.50 |
| flight_michaels_FLY125@2.000+4_rps_refined | 1.45 |
| ... 3 more | <= 1.43 |

### Wind test (floor excess)

Window rotor-mean carriers span 33.6-81.0 rev/s (1.27 octaves).

| quantity | mic 1 | mic 2 | mic 3 | mic 4 | mic 5 | mic 6 | mic 7 | mic 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| excess 30-300 Hz | +1.64 | -3.86 | -0.16 | -0.76 | +4.05 | +1.91 | -0.41 | -2.42 |
| excess 300-1k Hz | +0.74 | -4.23 | +0.60 | +1.36 | +4.52 | +1.79 | -1.17 | -3.61 |
| excess 1k-3k Hz | +0.97 | -6.73 | +0.78 | +1.73 | +4.86 | +2.40 | -0.62 | -3.39 |
| excess 3k-8k Hz | -1.84 | -6.64 | +1.60 | +3.73 | +5.42 | +2.82 | -0.28 | -4.80 |
| excess < 500 Hz | +1.43 | -3.49 | -0.09 | -0.47 | +4.06 | +1.82 | -0.63 | -2.65 |
| excess > 500 Hz | +0.03 | -6.50 | +1.06 | +2.36 | +5.03 | +2.46 | -0.56 | -3.88 |
| slope <500 Hz dB/oct | +0.22 | +0.40 | -0.74 | -1.12 | -0.46 | -0.13 | +1.13 | +0.71 |
| slope stderr | 0.19 | 0.65 | 0.11 | 0.09 | 0.30 | 0.33 | 0.47 | 0.49 |
| slope full band dB/oct | +0.13 | -0.96 | -0.98 | -0.84 | +0.37 | +0.73 | +1.05 | +0.51 |

Spread over channels: 2.44 dB below 500 Hz against 3.70 dB above it. Groups (one free intercept each in the within-group fit): cruise x8, standby x3; largest within-group speed span 0.08 octaves, below the 0.25-octave minimum, so the within-group slope is unidentified and not tabled.

### Verdict

- (i) full: the rank-one `g[mic]` model explains 80.3% of the deviation with a 1.47 dB residual (1.49 dB held out), against 91.4% / 0.97 dB (1.07 dB held out) for `g[mic, band]`, 81.6% / 1.42 dB for `g[mic, regime]`, 80.3% / 1.47 dB for `g[mic, recording]` and 82.7% / 1.38 dB for `g[mic, window]` -- one constant per channel is NOT enough; the deviation is frequency-dependent (a per-channel transfer function, not a gain), and the richer per-band description also predicts a held-out window better.
- (i) floor: the rank-one `g[mic]` model explains 75.7% of the deviation with a 1.62 dB residual (1.63 dB held out), against 93.1% / 0.87 dB (0.97 dB held out) for `g[mic, band]`, 76.7% / 1.59 dB for `g[mic, regime]`, 75.7% / 1.62 dB for `g[mic, recording]` and 77.9% / 1.54 dB for `g[mic, window]` -- one constant per channel is NOT enough; the deviation is frequency-dependent (a per-channel transfer function, not a gain), and the richer per-band description also predicts a held-out window better.
- (ii) comb and floor gains correlate at r = 0.996 with an RMS difference of 0.35 dB (largest 0.82 dB) -- the same per-channel sensitivity moves the lines and the floor, as a gain must; the split has +12.3 dB of comb-over-floor contrast at 30-300 Hz and +2.2 dB at 3-8 kHz, with 81% of the cells masked.
- (iii) the floor excess spreads 2.44 dB over channels below 500 Hz against 3.70 dB above it (2.17 / 3.67 dB on the quantile floor), i.e. it is NOT concentrated at low frequency; its pooled speed slope reaches 1.13 dB per octave (RMS 0.71, largest standard error 0.65) over a 1.27-octave span, with 3/8 channels beyond two standard errors, so the excess DOES track rotor speed, though weakly (at most 1.13 dB/oct, 1.4 dB over the whole span) -- not the wind/self-noise signature.

## dregon_bench

### Rank test

| variant | model | params | var. explained | residual RMS dB | held-out RMS dB | 30-300 | 300-1k | 1k-3k | 3k-8k |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full | const | 8 | 6.3% | 6.20 | 6.36 | 8.99 | 3.31 | 2.13 | 3.08 |
| full | per_band | 192 | 10.9% | 6.05 | 6.37 | 8.83 | 3.19 | 1.62 | 2.96 |
| full | per_regime | 32 | 46.4% | 4.69 | 4.78 | 6.01 | 3.22 | 3.94 | 3.06 |
| full | per_recording | 160 | 50.0% | 4.53 | n/a | 5.68 | 3.15 | 4.04 | 3.12 |
| full | per_window | 160 | 50.0% | 4.53 | n/a | 5.68 | 3.15 | 4.04 | 3.12 |
| floor | const | 8 | 4.5% | 6.84 | 7.05 | 9.82 | 4.29 | 2.28 | 3.16 |
| floor | per_band | 192 | 8.4% | 6.70 | 7.05 | 9.70 | 4.17 | 1.59 | 2.97 |
| floor | per_regime | 32 | 53.8% | 4.76 | 4.88 | 5.61 | 3.41 | 4.74 | 3.79 |
| floor | per_recording | 160 | 58.3% | 4.52 | n/a | 5.08 | 3.23 | 4.89 | 3.89 |
| floor | per_window | 160 | 58.3% | 4.52 | n/a | 5.08 | 3.23 | 4.89 | 3.89 |
| comb | const | 8 | 24.4% | 3.50 | 3.54 | 5.55 | 2.69 | 2.15 | 3.09 |
| comb | per_band | 176 | 34.9% | 3.25 | 3.56 | 5.03 | 2.59 | 1.95 | 3.01 |
| comb | per_regime | 32 | 38.5% | 3.16 | 3.20 | 5.09 | 2.53 | 2.15 | 2.29 |
| comb | per_recording | 160 | 41.2% | 3.09 | n/a | 4.97 | 2.47 | 2.13 | 2.21 |
| comb | per_window | 160 | 41.2% | 3.09 | n/a | 4.97 | 2.47 | 2.13 | 2.21 |
| floor_q25 | const | 8 | 3.9% | 6.99 | 7.22 | 9.93 | 4.73 | 2.28 | 3.48 |
| floor_q25 | per_band | 192 | 7.0% | 6.88 | 7.24 | 9.84 | 4.67 | 1.59 | 3.36 |
| floor_q25 | per_regime | 32 | 58.2% | 4.61 | 4.76 | 5.40 | 3.28 | 4.73 | 3.64 |
| floor_q25 | per_recording | 160 | 63.1% | 4.33 | n/a | 4.82 | 2.99 | 4.88 | 3.69 |
| floor_q25 | per_window | 160 | 63.1% | 4.33 | n/a | 4.82 | 2.99 | 4.88 | 3.69 |

`held-out RMS` is leave-one-window-out: the model is fitted on the other windows and asked to predict the held-out one. `per_window` has no out-of-sample prediction by construction, hence `n/a`.

Total deviation RMS (the thing being explained): full 6.41 dB, floor 7.00 dB, comb 4.03 dB, floor_q25 7.13 dB.

### The eight gains (`const` model), dB relative to the mic mean

| quantity | mic 1 | mic 2 | mic 3 | mic 4 | mic 5 | mic 6 | mic 7 | mic 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full gain | -1.27 | +3.75 | -1.66 | +0.20 | +0.54 | -0.08 | -1.36 | -0.12 |
| floor gain | -0.75 | +3.57 | -1.30 | -0.21 | +0.89 | -0.49 | -1.07 | -0.63 |
| comb gain | -2.12 | +3.80 | -2.40 | +1.28 | -1.00 | +0.93 | -1.54 | +1.05 |
| floor_q25 gain | -0.51 | +3.40 | -1.12 | -0.30 | +0.91 | -0.59 | -0.89 | -0.90 |
| full sd over bands | 1.09 | 1.30 | 0.92 | 1.13 | 2.36 | 1.29 | 1.21 | 1.47 |
| floor sd over bands | 0.85 | 1.09 | 0.95 | 1.33 | 2.48 | 1.53 | 0.92 | 1.48 |
| comb sd over bands | 1.82 | 1.96 | 1.35 | 1.15 | 2.76 | 0.97 | 1.61 | 1.63 |
| floor_q25 sd over bands | 0.84 | 0.93 | 0.93 | 1.30 | 2.17 | 1.34 | 0.79 | 1.41 |
| full sd over windows | 4.58 | 4.80 | 4.91 | 2.88 | 6.04 | 3.72 | 4.00 | 2.95 |
| floor sd over windows | 5.64 | 5.90 | 5.87 | 3.21 | 7.55 | 4.36 | 4.77 | 3.46 |
| comb sd over windows | 1.47 | 1.91 | 1.80 | 1.74 | 1.72 | 1.70 | 1.68 | 1.59 |
| floor_q25 sd over windows | 6.00 | 6.50 | 6.18 | 3.45 | 8.01 | 4.70 | 5.00 | 3.72 |
| full sd over recordings | 4.58 | 4.80 | 4.91 | 2.88 | 6.04 | 3.72 | 4.00 | 2.95 |
| floor sd over recordings | 5.64 | 5.90 | 5.87 | 3.21 | 7.55 | 4.36 | 4.77 | 3.46 |
| comb sd over recordings | 1.47 | 1.91 | 1.80 | 1.74 | 1.72 | 1.70 | 1.68 | 1.59 |
| floor_q25 sd over recordings | 6.00 | 6.50 | 6.18 | 3.45 | 8.01 | 4.70 | 5.00 | 3.72 |
| comb minus floor | -1.36 | +0.23 | -1.09 | +1.49 | -1.89 | +1.42 | -0.47 | +1.68 |

comb-vs-floor gain correlation over the 8 channels: r = 0.747; RMS difference 1.32 dB; largest |difference| 1.89 dB.

Contrast the comb/floor split actually has (mic- and window-mean `comb - floor`): 30-300 +31.35 dB, 300-1k +24.49 dB, 1k-3k +11.89 dB, 3k-8k +7.27 dB. The order mask covers 6% of the (frame, bin) cells on average, so where the contrast is small the two variants are largely reading the same spectrum.

Robustness of the floor: the `floor_q25` gains (a 0.25 quantile of the SAME unmasked cells instead of their power mean) correlate with the `floor` gains at r = 0.994, RMS difference 0.17 dB, largest 0.27 dB; its low-band spread over channels is 2.16 dB against 1.55 dB above 500 Hz (power-mean floor: 2.28 / 1.69 dB).

### Residual RMS of the `const` model per window (floor variant), dB

| window | residual RMS dB |
| --- | --- |
| bench_dregon_Motor3_90 | 8.83 |
| bench_dregon_Motor3_60 | 8.64 |
| bench_dregon_Motor4_90 | 8.03 |
| bench_dregon_Motor4_70 | 7.88 |
| bench_dregon_Motor3_70 | 7.82 |
| bench_dregon_Motor4_60 | 7.53 |
| bench_dregon_Motor4_80 | 7.36 |
| bench_dregon_Motor1_90 | 7.08 |
| ... 12 more | <= 7.01 |

### Wind test (floor excess)

Window rotor-mean carriers span 48.4-89.2 rev/s (0.88 octaves).

| quantity | mic 1 | mic 2 | mic 3 | mic 4 | mic 5 | mic 6 | mic 7 | mic 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| excess 30-300 Hz | -0.16 | +3.78 | -0.35 | -1.57 | +3.32 | -1.89 | -1.01 | -2.13 |
| excess 300-1k Hz | -1.08 | +3.86 | -1.83 | +0.16 | +0.88 | -0.61 | -1.54 | +0.15 |
| excess 1k-3k Hz | -1.48 | +2.98 | -2.18 | +1.30 | -2.28 | +1.45 | -0.69 | +0.90 |
| excess 3k-8k Hz | -0.91 | +3.38 | -1.95 | +0.81 | -1.20 | +0.73 | -1.10 | +0.23 |
| excess < 500 Hz | -0.31 | +3.99 | -0.58 | -1.39 | +3.16 | -1.85 | -1.24 | -1.79 |
| excess > 500 Hz | -1.20 | +3.14 | -2.03 | +0.97 | -1.37 | +0.87 | -0.90 | +0.53 |
| slope <500 Hz dB/oct | +1.71 | -1.51 | +0.43 | +0.77 | -0.93 | -1.02 | +1.29 | -0.74 |
| slope stderr | 7.73 | 7.81 | 7.86 | 4.04 | 10.01 | 5.63 | 6.14 | 4.31 |
| slope full band dB/oct | +0.97 | -0.61 | +0.24 | +0.26 | -0.40 | -0.60 | +0.58 | -0.44 |
| slope <500 Hz within-group | +3.00 | -1.45 | +0.25 | +0.10 | -2.47 | -1.00 | +1.51 | +0.05 |
| within-group stderr | 3.58 | 1.63 | 2.17 | 1.42 | 1.73 | 2.37 | 2.42 | 1.38 |

Spread over channels: 2.28 dB below 500 Hz against 1.69 dB above it. Groups (one free intercept each in the within-group fit): bench_Motor1 x5, bench_Motor2 x5, bench_Motor3 x5, bench_Motor4 x5; largest within-group speed span 0.85 octaves.

### Verdict

- (i) full: the rank-one `g[mic]` model explains 6.3% of the deviation with a 6.20 dB residual (6.36 dB held out), against 10.9% / 6.05 dB (6.37 dB held out) for `g[mic, band]`, 46.4% / 4.69 dB for `g[mic, regime]`, 50.0% / 4.53 dB for `g[mic, recording]` and 50.0% / 4.53 dB for `g[mic, window]` -- one constant per channel is NOT enough; the deviation is time-varying (not a property of the channel), though the per-band description does not survive holding a window out.
- (i) floor: the rank-one `g[mic]` model explains 4.5% of the deviation with a 6.84 dB residual (7.05 dB held out), against 8.4% / 6.70 dB (7.05 dB held out) for `g[mic, band]`, 53.8% / 4.76 dB for `g[mic, regime]`, 58.3% / 4.52 dB for `g[mic, recording]` and 58.3% / 4.52 dB for `g[mic, window]` -- one constant per channel is NOT enough; the deviation is time-varying (not a property of the channel), though the per-band description does not survive holding a window out.
- (ii) comb and floor gains correlate at r = 0.747 with an RMS difference of 1.32 dB (largest 1.89 dB) -- the lines and the floor do NOT see the same per-channel gain; the split has +31.3 dB of comb-over-floor contrast at 30-300 Hz and +7.3 dB at 3-8 kHz, with 6% of the cells masked.
- (iii) the floor excess spreads 2.28 dB over channels below 500 Hz against 1.69 dB above it (2.16 / 1.55 dB on the quantile floor), i.e. it is NOT concentrated at low frequency; its within-motor speed slope reaches 3.00 dB per octave (RMS 1.60, largest standard error 3.58) over a 0.85-octave span, with 0/8 channels beyond two standard errors, so the excess is consistent with NO speed dependence (the span bounds every channel's slope at 10.16 dB/oct) -- not the wind/self-noise signature.

## Figures

- `deviation_heatmap_dregon.png`
- `deviation_heatmap_michaels.png`
- `deviation_heatmap_dregon_bench.png`
- `floor_excess_vs_speed_dregon.png`
- `floor_excess_vs_speed_michaels.png`
- `floor_excess_vs_band.png`
