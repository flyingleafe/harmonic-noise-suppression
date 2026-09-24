# DREGON's intermittent rotor lines: label, amplitude or floor?

`scripts/_dregon_intermittency.py` at `93a8c14111b8`; numbers in `intermittency.json`. A laptop diagnostic, no model is fitted. The wander measurement (`results/noise_v3/wander/findings.md`) read the line sigma (1.48 dB on DREGON) off RESOLVABLE tracks only; this looks at the tracks it left out, on the same windows (`_mic_gain_rank.rig_specs(extra=True)`), the same 2048/512 front end and the same block line measurement (`wander.measure_lines`, 0.5 s blocks).

**Definitions.** A (window, rotor, k) track is classified when it is valid (in band, on the grid) in >= 80% of the window's blocks: RESOLVABLE when its block prominence (mic-summed 3 label-tracked cells over the mic-summed local q25 floor) is >= 6 dB in >= 80% of its valid blocks, INTERMITTENT at 20%-80%, ALWAYS-UNDER below 20%. An ABSENT block is one under 6 dB. DREGON's label is the commanded speed (`motors_command`); Michael's is the audio-refined `rps_refined` -- the control.

## Counts

| rig | windows | blocks | resolvable | intermittent | always-under | off grid | intermittent blocks | of them absent | distinct lines | rotor-dominant (own share >= 0.8) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon | 50 | 440 | 44 | 767 | 17983 | 2166 | 6280 | 3840 | 156 | 60 |
| michaels | 11 | 152 | 167 | 262 | 5236 | 927 | 3884 | 2115 | 105 | 29 |

Intermittent tracks by order group and rotor:

| rig | k 1-2 | k 3-8 | k 9-24 | k 25-60 | k 61+ | rotor 1 | rotor 2 | rotor 3 | rotor 4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon | 86 | 32 | 308 | 185 | 156 | 175 | 182 | 204 | 206 |
| michaels | 2 | 55 | 145 | 22 | 38 | 58 | 59 | 68 | 77 |

- dregon, the lines intermittent in the most windows (rotor:order): 3:2 (27), 1:2 (22), 4:2 (22), 4:13 (19), 4:14 (17), 3:14 (16), 2:2 (15), 3:12 (15), 4:16 (14), 1:14 (12), 1:69 (12), 3:42 (12), 1:16 (11), 3:16 (11), 3:70 (11)
- michaels, the lines intermittent in the most windows (rotor:order): 3:20 (7), 4:12 (7), 1:22 (6), 1:5 (6), 3:13 (6), 4:7 (6), 4:8 (6), 1:14 (5), 1:16 (5), 1:18 (5), 1:20 (5), 2:20 (5), 3:11 (5), 3:7 (5), 4:10 (5)

## The three tests, as counterfactuals

Each mechanism is removed in turn and the intermittent tracks re-measured: the share of their absent blocks that return (>= 6 dB), of their present blocks that go, of the tracks still intermittent by the same rule, and the present/absent switches between consecutive blocks against the observed. A mechanism that causes the intermittency leaves few tracks intermittent once removed: a carrier that fixes a wrong label returns absent blocks without losing present ones; a floor held still may leave a line steadily present or steadily under (seen only in floor dips). The +-3-bin peak search picks the best of seven positions per block, so its row is an upper bound, not an estimate.

| rig | counterfactual | absent -> present | present -> absent | still intermittent | now resolvable | now always-under | switches / observed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dregon | peak +-3 bins, any | 27.5 % | 0.0 % | 83.4 % | 16.6 % | 0.0 % | 0.87 |
| dregon | peak +-3 bins, own | 11.6 % | 0.0 % | 95.3 % | 4.7 % | 0.0 % | 1.00 |
| dregon | carrier from other lines | 7.4 % | 13.7 % | 85.7 % | 2.5 % | 11.9 % | 0.93 |
| dregon | label: window mean | 12.9 % | 45.2 % | 61.0 % | 2.0 % | 37.0 % | 0.75 |
| dregon | label: 1 s moving mean | 7.5 % | 20.6 % | 81.4 % | 1.6 % | 17.1 % | 0.88 |
| dregon | floor at mean: local | 6.6 % | 22.5 % | 77.8 % | 1.6 % | 20.6 % | 0.89 |
| dregon | floor at mean: band u+u_j | 8.2 % | 18.5 % | 86.2 % | 1.0 % | 12.8 % | 0.98 |
| dregon | floor at mean: level u | 4.1 % | 13.2 % | 86.0 % | 1.0 % | 12.9 % | 0.94 |
| michaels | peak +-3 bins, any | 41.5 % | 0.0 % | 61.1 % | 38.9 % | 0.0 % | 0.73 |
| michaels | peak +-3 bins, own | 11.8 % | 0.0 % | 96.6 % | 3.4 % | 0.0 % | 1.05 |
| michaels | carrier from other lines | 6.5 % | 12.9 % | 78.6 % | 2.3 % | 19.1 % | 0.94 |
| michaels | label: window mean | 14.1 % | 40.1 % | 60.7 % | 2.7 % | 36.6 % | 0.81 |
| michaels | label: 1 s moving mean | 8.7 % | 23.9 % | 76.7 % | 1.5 % | 21.8 % | 0.86 |
| michaels | floor at mean: local | 9.6 % | 15.5 % | 83.2 % | 1.9 % | 14.9 % | 0.95 |
| michaels | floor at mean: band u+u_j | 6.7 % | 8.6 % | 93.5 % | 1.5 % | 5.0 % | 0.98 |
| michaels | floor at mean: level u | 5.9 % | 8.5 % | 92.7 % | 1.5 % | 5.7 % | 1.00 |

By order group, the tracks still intermittent after each counterfactual (and the median of the intermittent tracks' own block sd):

| rig | orders | intermittent tracks | own block sd dB | peak +-3 bins, own | carrier from other lines | label: 1 s moving mean | floor at mean: local | floor at mean: band u+u_j |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon | 1-2 | 86 | 1.48 | 98.8 % (0.0 % under) | 97.7 % (2.3 % under) | 97.7 % (1.2 % under) | 44.2 % (53.5 % under) | 82.6 % (17.4 % under) |
| dregon | 3-8 | 32 | 1.70 | 100.0 % (0.0 % under) | 100.0 % (0.0 % under) | 93.8 % (6.2 % under) | 59.4 % (40.6 % under) | 81.2 % (18.8 % under) |
| dregon | 9-24 | 308 | 2.07 | 95.5 % (0.0 % under) | 90.3 % (7.1 % under) | 83.1 % (15.3 % under) | 81.2 % (17.2 % under) | 85.7 % (12.3 % under) |
| dregon | 25-60 | 185 | 2.62 | 93.5 % (0.0 % under) | 78.9 % (16.8 % under) | 74.1 % (24.3 % under) | 77.8 % (20.0 % under) | 84.9 % (14.1 % under) |
| dregon | 61+ | 156 | 4.35 | 94.2 % (0.0 % under) | 75.0 % (23.1 % under) | 75.0 % (23.1 % under) | 93.6 % (5.8 % under) | 91.7 % (8.3 % under) |
| michaels | 1-2 | 2 | 1.81 | 100.0 % (0.0 % under) | 100.0 % (0.0 % under) | 100.0 % (0.0 % under) | 100.0 % (0.0 % under) | 100.0 % (0.0 % under) |
| michaels | 3-8 | 55 | 4.22 | 92.7 % (0.0 % under) | 87.3 % (7.3 % under) | 87.3 % (9.1 % under) | 94.5 % (5.5 % under) | 96.4 % (3.6 % under) |
| michaels | 9-24 | 145 | 3.36 | 96.6 % (0.0 % under) | 84.1 % (13.8 % under) | 75.2 % (24.1 % under) | 77.2 % (19.3 % under) | 91.0 % (6.2 % under) |
| michaels | 25-60 | 22 | 2.24 | 100.0 % (0.0 % under) | 86.4 % (13.6 % under) | 45.5 % (54.5 % under) | 77.3 % (22.7 % under) | 90.9 % (9.1 % under) |
| michaels | 61+ | 38 | 4.82 | 100.0 % (0.0 % under) | 39.5 % (60.5 % under) | 84.2 % (13.2 % under) | 92.1 % (7.9 % under) | 100.0 % (0.0 % under) |

## (a) Label displacement

- **Peak search** (the prominence of `measure_lines` on the periodogram shifted by -3..+3 bins, per block): a recovered peak belongs to ANOTHER rotor when that rotor's nearest order (block-mean label) lies within 1 bin of the peak bin and nearer than the own label. The null is the same search on the always-under tracks.
- **Carrier from the rotor's other lines**: per block, the offset (grid 0.05 rev/s, up to 1 rev/s or half the gap to the nearest rotor) that maximises the mean prominence of the rotor's OTHER resolvable or intermittent orders >= 5, applied to the track (leave-one-out: no selection on the track itself).
- **Smoothed label**: the rotor's window-mean carrier, and a centred 1 s moving mean of the frame carriers (a motor's inertia low-passes the command).

| rig | tracks | absent blocks | any peak >= 6 dB | on another rotor's line | own | own at +-3 (search edge) | own shift rev/s, median [IQR] | present blocks whose peak moves (>= 1 dB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon | intermittent | 3840 | 27.5 % | 15.9 % | 11.6 % | 63.7 % | 0.56 [0.37, 1.12] | 31.7 % |
| dregon | always-under (null) | 157390 | 2.1 % | 1.1 % | 0.9 % | 63.5 % | 0.65 [0.37, 1.30] |  |
| michaels | intermittent | 2115 | 41.5 % | 29.7 % | 11.8 % | 79.1 % | 1.46 [1.07, 2.34] | 38.1 % |
| michaels | always-under (null) | 62458 | 2.8 % | 1.6 % | 1.1 % | 92.4 % | 1.80 [1.02, 7.81] |  |

Own-peak offsets of the recovered absent blocks (bins, -3..+3):

- dregon intermittent: -3: 156, -2: 63, -1: 51, 0: 0, 1: 14, 2: 33, 3: 127
- dregon always-under: -3: 514, -2: 241, -1: 102, 0: 0, 1: 71, 2: 131, 3: 433
- michaels intermittent: -3: 98, -2: 23, -1: 10, 0: 0, 1: 10, 2: 9, 3: 99
- michaels always-under: -3: 406, -2: 19, -1: 8, 0: 0, 1: 10, 2: 17, 3: 246

- dregon: the carrier re-estimate exists for 88.4 % of the absent blocks; its |offset| is 0.25 rev/s median on absent blocks and 0.20 on present ones (zero on 14.5 % of absent). The label sits > 1 bin off its window mean in 63.5 % of absent and 63.0 % of present blocks; another rotor's order lies within 1 bin of the label in 53.0 % of absent and 57.6 % of present blocks.
- michaels: the carrier re-estimate exists for 100.0 % of the absent blocks; its |offset| is 0.20 rev/s median on absent blocks and 0.15 on present ones (zero on 15.7 % of absent). The label sits > 1 bin off its window mean in 67.5 % of absent and 60.2 % of present blocks; another rotor's order lies within 1 bin of the label in 49.1 % of absent and 60.8 % of present blocks.

## (b) Distribution shape

The Gaussian hypothesis, forward-modelled to the observable: per track, the line level over the local floor is an OU in dB with the resolvable lines' `(sigma_total, tau_total)` (`results/noise_v3/wander/wander_detail.json`) plus the rig's measured block noise, the floor cells add the empirical cell-over-floor ratio of line-free blocks (always-under tracks with window-median prominence < 1 dB), and the sum is read as block prominence; each track's level is set so the model's median prominence is the track's. Three versions: (1) that, at the resolvable lines' sigma -- the literal test; (2) the same with the track's MEASURED local floor deviation subtracted block by block (the line held in absolute level, the floor as it was: what v3's line wander plus its floor predicts); (3) as (2) with the line sigma at which the model's centred sd matches the observed one -- the line's OWN wander once the floor is accounted, and a shape test at matched width. Three track sets: the intermittent ones; NEAR-THRESHOLD -- every classified track whose median block prominence is 4-8 dB, whatever its class, a set chosen blind to the scatter the 20-80 % rule selects on; and the resolvable ones, the calibration (the wander sigma was measured on them). 200 draws each; `p` is the share of draws at least as extreme as the observed on the side it departs to (floor 1/(draws + 1)).

**dregon, intermittent** (767 tracks, 6280 blocks; block noise 1.30 dB^2, tau 0.57 s):

| statistic | observed | (1) sigma 1.48 dB [5, 50, 95 %] | p | (2) sigma 1.48 dB + floor | p | (3) sigma fitted 5.03 dB + floor | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sd of block prominence around the track mean, dB | 2.95 | 1.18 [1.16, 1.21] | 0.005 | 1.48 [1.46, 1.51] | 0.005 | 2.96 [2.89, 3.02] | 0.423 |
| excess kurtosis (centred) | 1.05 | 0.61 [0.35, 1.24] | 0.109 | 0.59 [0.42, 0.90] | 0.015 | 0.90 [0.71, 1.13] | 0.134 |
| skewness (centred) | 0.32 | 0.25 [0.17, 0.36] | 0.164 | 0.24 [0.17, 0.32] | 0.060 | 0.58 [0.51, 0.64] | 0.005 |
| lag-1 autocorrelation (centred, pooled) | 0.23 | 0.01 [-0.01, 0.03] | 0.005 | 0.13 [0.11, 0.15] | 0.005 | 0.15 [0.12, 0.17] | 0.005 |
| present/absent switches per block step | 0.36 | 0.22 [0.21, 0.23] | 0.005 | 0.24 [0.23, 0.25] | 0.005 | 0.30 [0.29, 0.31] | 0.005 |
| spread of the tracks' own sds (sd / mean over tracks) | 0.49 | 0.32 [0.30, 0.34] | 0.005 | 0.36 [0.35, 0.38] | 0.005 | 0.41 [0.40, 0.43] | 0.005 |
| blocks >= 6 dB | 38.9 % | 28.5 % [27.7 %, 29.4 %] | 0.005 | 30.8 % [30.1 %, 31.5 %] | 0.005 | 36.8 % [35.5 %, 37.9 %] | 0.005 |
| blocks < 3 dB | 27.1 % | 21.3 % [20.6 %, 21.9 %] | 0.005 | 23.3 % [22.7 %, 23.9 %] | 0.005 | 32.9 % [31.9 %, 34.0 %] | 0.005 |
| blocks > 6 dB under the track mean | 1.9 % | 0.0 % [0.0 %, 0.0 %] | 0.005 | 0.0 % [0.0 %, 0.0 %] | 0.005 | 1.2 % [0.9 %, 1.4 %] | 0.005 |
| BIC(1) - BIC(2 Gaussians), centred | 167.1 | -13.6 [-40.2, 31.2] | 0.005 | -5.4 [-36.3, 37.0] | 0.005 | 305.0 [247.0, 374.3] | 0.005 |
| BIC(1) - BIC(2 Gaussians), raw | -51.8 | -37.9 [-61.3, -9.7] | 0.174 | -5.5 [-28.1, 26.4] | 0.005 | 910.3 [800.4, 999.4] | 0.005 |
| tracks intermittent by the 20-80 % rule | 100.0 % | 37.9 % [36.1 %, 39.8 %] | 0.005 | 47.5 % [45.2 %, 49.5 %] | 0.005 | 62.5 % [60.0 %, 64.7 %] | 0.005 |

Two-Gaussian fits: centred means -0.56 / 0.82 dB, sds 2.24 / 3.61, weights 0.60 / 0.40; raw means 3.16 / 6.19 dB, sds 2.75 / 3.07, weights 0.41 / 0.59. Co-movement of two tracks of one window: mean r 0.15 for the same rotor (3302 pairs), 0.12 for different rotors (10130 pairs); the v3 model's rotor-common share of a line's block variance is 0.26 (different rotors 0). A track's own block sd, median by order group: k 1-2 1.48 dB (86), k 3-8 1.70 dB (32), k 9-24 2.07 dB (308), k 25-60 2.62 dB (185), k 61+ 4.35 dB (156).

**dregon, near_threshold** (744 tracks, 6208 blocks; block noise 1.30 dB^2, tau 0.57 s):

| statistic | observed | (1) sigma 1.48 dB [5, 50, 95 %] | p | (2) sigma 1.48 dB + floor | p | (3) sigma fitted 3.36 dB + floor | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sd of block prominence around the track mean, dB | 2.28 | 1.20 [1.18, 1.22] | 0.005 | 1.54 [1.51, 1.56] | 0.005 | 2.30 [2.25, 2.34] | 0.289 |
| excess kurtosis (centred) | 2.05 | 0.27 [0.09, 0.65] | 0.005 | 0.35 [0.19, 0.53] | 0.005 | 0.38 [0.22, 0.58] | 0.005 |
| skewness (centred) | -0.01 | 0.21 [0.16, 0.30] | 0.005 | 0.22 [0.16, 0.28] | 0.005 | 0.42 [0.37, 0.49] | 0.005 |
| lag-1 autocorrelation (centred, pooled) | 0.19 | 0.03 [0.01, 0.05] | 0.005 | 0.13 [0.12, 0.15] | 0.005 | 0.15 [0.13, 0.17] | 0.005 |
| present/absent switches per block step | 0.31 | 0.28 [0.27, 0.29] | 0.005 | 0.31 [0.29, 0.32] | 0.239 | 0.34 [0.33, 0.35] | 0.005 |
| spread of the tracks' own sds (sd / mean over tracks) | 0.52 | 0.29 [0.28, 0.31] | 0.005 | 0.33 [0.32, 0.34] | 0.005 | 0.33 [0.31, 0.34] | 0.005 |
| blocks >= 6 dB | 33.3 % | 30.8 % [29.8 %, 31.7 %] | 0.005 | 34.2 % [33.3 %, 35.1 %] | 0.055 | 38.5 % [37.4 %, 39.8 %] | 0.005 |
| blocks < 3 dB | 16.7 % | 6.3 % [5.8 %, 6.8 %] | 0.005 | 9.6 % [9.1 %, 10.3 %] | 0.005 | 18.3 % [17.3 %, 19.2 %] | 0.005 |
| blocks > 6 dB under the track mean | 1.3 % | 0.0 % [0.0 %, 0.0 %] | 0.005 | 0.0 % [0.0 %, 0.0 %] | 0.005 | 0.2 % [0.1 %, 0.2 %] | 0.005 |
| BIC(1) - BIC(2 Gaussians), centred | 384.2 | -23.8 [-44.1, 6.8] | 0.005 | -20.4 [-42.8, 4.7] | 0.005 | 126.6 [85.0, 174.6] | 0.005 |
| BIC(1) - BIC(2 Gaussians), raw | 392.9 | 208.6 [155.4, 270.6] | 0.005 | 142.1 [101.5, 180.2] | 0.005 | 418.4 [357.2, 484.6] | 0.264 |
| tracks intermittent by the 20-80 % rule | 68.7 % | 47.5 % [45.2 %, 49.9 %] | 0.005 | 60.4 % [57.5 %, 63.2 %] | 0.005 | 71.4 % [68.7 %, 74.2 %] | 0.060 |

Two-Gaussian fits: centred means -0.44 / 0.23 dB, sds 3.17 / 1.57, weights 0.35 / 0.65; raw means 5.10 / 5.11 dB, sds 3.39 / 1.62, weights 0.42 / 0.58. Co-movement of two tracks of one window: mean r 0.17 for the same rotor (2769 pairs), 0.12 for different rotors (8304 pairs); the v3 model's rotor-common share of a line's block variance is 0.26 (different rotors 0). A track's own block sd, median by order group: k 1-2 1.29 dB (135), k 3-8 1.31 dB (42), k 9-24 1.79 dB (335), k 25-60 2.25 dB (164), k 61+ 3.83 dB (68).

**dregon, resolvable** (44 tracks, 352 blocks; block noise 1.30 dB^2, tau 0.57 s):

| statistic | observed | (1) sigma 1.48 dB [5, 50, 95 %] | p | (2) sigma 1.48 dB + floor | p | (3) sigma fitted 2.52 dB + floor | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sd of block prominence around the track mean, dB | 2.20 | 1.41 [1.32, 1.50] | 0.005 | 1.65 [1.55, 1.77] | 0.005 | 2.21 [2.03, 2.35] | 0.483 |
| excess kurtosis (centred) | 3.31 | 0.02 [-0.37, 0.57] | 0.005 | 0.00 [-0.36, 0.54] | 0.005 | -0.02 [-0.37, 0.43] | 0.005 |
| skewness (centred) | -0.02 | 0.10 [-0.11, 0.31] | 0.204 | 0.05 [-0.15, 0.29] | 0.264 | 0.17 [-0.04, 0.38] | 0.080 |
| lag-1 autocorrelation (centred, pooled) | 0.19 | 0.04 [-0.04, 0.12] | 0.005 | 0.09 [0.02, 0.17] | 0.020 | 0.12 [0.03, 0.19] | 0.055 |
| present/absent switches per block step | 0.13 | 0.13 [0.09, 0.17] | 0.488 | 0.16 [0.12, 0.19] | 0.124 | 0.21 [0.17, 0.26] | 0.005 |
| spread of the tracks' own sds (sd / mean over tracks) | 0.57 | 0.28 [0.23, 0.33] | 0.005 | 0.29 [0.24, 0.34] | 0.005 | 0.29 [0.24, 0.34] | 0.005 |
| blocks >= 6 dB | 92.6 % | 91.2 % [88.6 %, 94.0 %] | 0.214 | 88.9 % [86.4 %, 91.5 %] | 0.025 | 83.2 % [79.5 %, 86.9 %] | 0.005 |
| blocks < 3 dB | 1.1 % | 0.0 % [0.0 %, 0.3 %] | 0.005 | 0.0 % [0.0 %, 0.6 %] | 0.005 | 0.9 % [0.0 %, 1.7 %] | 0.388 |
| blocks > 6 dB under the track mean | 1.7 % | 0.0 % [0.0 %, 0.0 %] | 0.005 | 0.0 % [0.0 %, 0.0 %] | 0.005 | 0.0 % [0.0 %, 0.6 %] | 0.005 |
| BIC(1) - BIC(2 Gaussians), centred | 54.3 | -17.6 [-19.8, -11.9] | 0.005 | -17.9 [-19.9, -12.4] | 0.005 | -16.5 [-19.7, -8.6] | 0.005 |
| BIC(1) - BIC(2 Gaussians), raw | 92.4 | 12.0 [-5.0, 32.0] | 0.005 | 2.3 [-10.3, 19.3] | 0.005 | -2.3 [-11.7, 14.5] | 0.005 |
| tracks intermittent by the 20-80 % rule | 0.0 % | 18.2 % [9.1 %, 27.3 %] | 0.005 | 25.0 % [15.9 %, 34.1 %] | 0.005 | 38.6 % [27.3 %, 52.3 %] | 0.005 |

Two-Gaussian fits: centred means -0.15 / 0.07 dB, sds 3.57 / 1.20, weights 0.30 / 0.70; raw means 8.09 / 11.27 dB, sds 1.60 / 4.69, weights 0.82 / 0.18. Co-movement of two tracks of one window: mean r 0.07 for the same rotor (29 pairs), 0.11 for different rotors (71 pairs); the v3 model's rotor-common share of a line's block variance is 0.26 (different rotors 0). A track's own block sd, median by order group: k 1-2 1.10 dB (8), k 3-8 n/a dB (0), k 9-24 1.53 dB (22), k 25-60 1.65 dB (6), k 61+ 3.79 dB (8).

**michaels, intermittent** (262 tracks, 3884 blocks; block noise 1.29 dB^2, tau 1.18 s):

| statistic | observed | (1) sigma 1.98 dB [5, 50, 95 %] | p | (2) sigma 1.98 dB + floor | p | (3) sigma fitted 5.81 dB + floor | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sd of block prominence around the track mean, dB | 3.76 | 1.52 [1.49, 1.55] | 0.005 | 2.07 [2.02, 2.12] | 0.005 | 3.76 [3.61, 3.88] | 0.493 |
| excess kurtosis (centred) | 0.12 | 0.60 [0.36, 1.00] | 0.005 | 0.84 [0.62, 1.12] | 0.005 | 0.73 [0.46, 1.00] | 0.005 |
| skewness (centred) | -0.10 | 0.18 [0.06, 0.28] | 0.005 | 0.12 [0.04, 0.20] | 0.005 | 0.52 [0.43, 0.59] | 0.005 |
| lag-1 autocorrelation (centred, pooled) | 0.38 | 0.26 [0.24, 0.29] | 0.005 | 0.37 [0.35, 0.40] | 0.323 | 0.44 [0.41, 0.46] | 0.010 |
| present/absent switches per block step | 0.34 | 0.21 [0.19, 0.22] | 0.005 | 0.23 [0.22, 0.24] | 0.005 | 0.25 [0.23, 0.26] | 0.005 |
| spread of the tracks' own sds (sd / mean over tracks) | 0.34 | 0.28 [0.26, 0.30] | 0.005 | 0.36 [0.34, 0.39] | 0.015 | 0.38 [0.35, 0.41] | 0.005 |
| blocks >= 6 dB | 45.5 % | 42.7 % [41.4 %, 44.2 %] | 0.005 | 42.9 % [41.7 %, 44.2 %] | 0.005 | 44.2 % [42.2 %, 46.3 %] | 0.184 |
| blocks < 3 dB | 26.7 % | 17.4 % [16.6 %, 18.3 %] | 0.005 | 19.9 % [19.0 %, 20.7 %] | 0.005 | 30.3 % [28.7 %, 32.1 %] | 0.005 |
| blocks > 6 dB under the track mean | 5.5 % | 0.0 % [0.0 %, 0.1 %] | 0.005 | 0.4 % [0.3 %, 0.6 %] | 0.005 | 3.8 % [3.2 %, 4.5 %] | 0.005 |
| BIC(1) - BIC(2 Gaussians), centred | -41.8 | -28.9 [-47.6, 1.8] | 0.154 | -32.0 [-46.1, -4.2] | 0.149 | 150.0 [97.3, 204.6] | 0.005 |
| BIC(1) - BIC(2 Gaussians), raw | -39.9 | 24.1 [-13.4, 79.4] | 0.010 | 115.5 [65.4, 167.4] | 0.005 | 563.5 [463.8, 667.8] | 0.005 |
| tracks intermittent by the 20-80 % rule | 100.0 % | 39.7 % [35.9 %, 43.5 %] | 0.005 | 49.6 % [45.8 %, 53.1 %] | 0.005 | 62.2 % [57.6 %, 66.0 %] | 0.005 |

Two-Gaussian fits: centred means -2.48 / 2.08 dB, sds 3.05 / 2.94, weights 0.46 / 0.54; raw means 2.47 / 7.45 dB, sds 3.54 / 3.42, weights 0.40 / 0.60. Co-movement of two tracks of one window: mean r 0.17 for the same rotor (827 pairs), 0.17 for different rotors (2631 pairs); the v3 model's rotor-common share of a line's block variance is 0.05 (different rotors 0). A track's own block sd, median by order group: k 1-2 1.81 dB (2), k 3-8 4.22 dB (55), k 9-24 3.36 dB (145), k 25-60 2.24 dB (22), k 61+ 4.82 dB (38).

**michaels, near_threshold** (192 tracks, 2932 blocks; block noise 1.29 dB^2, tau 1.18 s):

| statistic | observed | (1) sigma 1.98 dB [5, 50, 95 %] | p | (2) sigma 1.98 dB + floor | p | (3) sigma fitted 4.70 dB + floor | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sd of block prominence around the track mean, dB | 3.28 | 1.53 [1.49, 1.57] | 0.005 | 2.05 [2.00, 2.11] | 0.005 | 3.29 [3.17, 3.40] | 0.483 |
| excess kurtosis (centred) | 0.56 | 0.28 [0.06, 0.52] | 0.020 | 0.47 [0.25, 0.79] | 0.269 | 0.40 [0.17, 0.69] | 0.194 |
| skewness (centred) | -0.16 | 0.22 [0.13, 0.31] | 0.005 | 0.17 [0.08, 0.25] | 0.005 | 0.45 [0.36, 0.53] | 0.005 |
| lag-1 autocorrelation (centred, pooled) | 0.36 | 0.28 [0.25, 0.32] | 0.005 | 0.38 [0.35, 0.41] | 0.129 | 0.43 [0.40, 0.46] | 0.005 |
| present/absent switches per block step | 0.35 | 0.27 [0.25, 0.29] | 0.005 | 0.29 [0.27, 0.30] | 0.005 | 0.28 [0.26, 0.30] | 0.005 |
| spread of the tracks' own sds (sd / mean over tracks) | 0.38 | 0.25 [0.22, 0.27] | 0.005 | 0.31 [0.29, 0.34] | 0.005 | 0.30 [0.28, 0.34] | 0.005 |
| blocks >= 6 dB | 45.3 % | 44.6 % [42.8 %, 46.5 %] | 0.249 | 45.3 % [43.6 %, 46.9 %] | 0.483 | 47.6 % [45.3 %, 49.5 %] | 0.060 |
| blocks < 3 dB | 20.4 % | 7.0 % [5.9 %, 7.9 %] | 0.005 | 10.6 % [9.6 %, 11.6 %] | 0.005 | 20.8 % [19.1 %, 22.6 %] | 0.383 |
| blocks > 6 dB under the track mean | 3.9 % | 0.0 % [0.0 %, 0.0 %] | 0.005 | 0.3 % [0.1 %, 0.4 %] | 0.005 | 2.0 % [1.6 %, 2.5 %] | 0.005 |
| BIC(1) - BIC(2 Gaussians), centred | -25.3 | -18.9 [-33.4, 2.2] | 0.318 | -27.4 [-40.6, -9.8] | 0.433 | 68.4 [33.5, 104.3] | 0.005 |
| BIC(1) - BIC(2 Gaussians), raw | -36.4 | 42.0 [10.9, 81.5] | 0.005 | 81.7 [43.5, 131.2] | 0.005 | 258.7 [200.9, 325.2] | 0.005 |
| tracks intermittent by the 20-80 % rule | 89.6 % | 54.7 % [48.9 %, 59.4 %] | 0.005 | 67.7 % [62.5 %, 72.9 %] | 0.005 | 74.5 % [68.8 %, 79.2 %] | 0.005 |

Two-Gaussian fits: centred means -2.20 / 1.30 dB, sds 3.00 / 2.70, weights 0.37 / 0.63; raw means 3.61 / 7.09 dB, sds 3.17 / 2.96, weights 0.43 / 0.57. Co-movement of two tracks of one window: mean r 0.26 for the same rotor (453 pairs), 0.21 for different rotors (1505 pairs); the v3 model's rotor-common share of a line's block variance is 0.05 (different rotors 0). A track's own block sd, median by order group: k 1-2 1.81 dB (2), k 3-8 3.33 dB (38), k 9-24 3.03 dB (111), k 25-60 1.77 dB (24), k 61+ 4.89 dB (17).

**michaels, resolvable** (167 tracks, 2528 blocks; block noise 1.29 dB^2, tau 1.18 s):

| statistic | observed | (1) sigma 1.98 dB [5, 50, 95 %] | p | (2) sigma 1.98 dB + floor | p | (3) sigma fitted 2.46 dB + floor | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sd of block prominence around the track mean, dB | 2.84 | 1.88 [1.83, 1.94] | 0.005 | 2.59 [2.52, 2.66] | 0.005 | 2.84 [2.76, 2.92] | 0.498 |
| excess kurtosis (centred) | 1.09 | 0.02 [-0.12, 0.18] | 0.005 | 0.34 [0.17, 0.52] | 0.005 | 0.23 [0.04, 0.48] | 0.005 |
| skewness (centred) | -0.46 | 0.06 [-0.02, 0.15] | 0.005 | -0.11 [-0.19, -0.04] | 0.005 | -0.06 [-0.13, 0.03] | 0.005 |
| lag-1 autocorrelation (centred, pooled) | 0.37 | 0.32 [0.28, 0.35] | 0.015 | 0.41 [0.38, 0.43] | 0.030 | 0.42 [0.39, 0.44] | 0.005 |
| present/absent switches per block step | 0.07 | 0.03 [0.02, 0.04] | 0.005 | 0.05 [0.04, 0.06] | 0.005 | 0.06 [0.05, 0.07] | 0.010 |
| spread of the tracks' own sds (sd / mean over tracks) | 0.40 | 0.24 [0.21, 0.26] | 0.005 | 0.30 [0.27, 0.32] | 0.005 | 0.29 [0.26, 0.31] | 0.005 |
| blocks >= 6 dB | 94.4 % | 98.1 % [97.5 %, 98.6 %] | 0.005 | 95.9 % [95.2 %, 96.6 %] | 0.005 | 95.1 % [94.1 %, 95.9 %] | 0.129 |
| blocks < 3 dB | 1.3 % | 0.0 % [0.0 %, 0.1 %] | 0.005 | 0.3 % [0.2 %, 0.5 %] | 0.005 | 0.5 % [0.2 %, 0.7 %] | 0.005 |
| blocks > 6 dB under the track mean | 3.0 % | 0.0 % [0.0 %, 0.2 %] | 0.005 | 1.5 % [1.1 %, 1.9 %] | 0.005 | 2.0 % [1.6 %, 2.4 %] | 0.005 |
| BIC(1) - BIC(2 Gaussians), centred | 76.6 | -33.0 [-40.1, -24.0] | 0.005 | -37.4 [-41.4, -28.3] | 0.005 | -39.4 [-41.8, -33.5] | 0.005 |
| BIC(1) - BIC(2 Gaussians), raw | 726.5 | 1256.9 [1146.4, 1381.7] | 0.005 | 903.8 [811.8, 999.9] | 0.005 | 794.9 [697.7, 880.6] | 0.129 |
| tracks intermittent by the 20-80 % rule | 0.0 % | 2.4 % [0.6 %, 4.2 %] | 0.015 | 5.4 % [3.0 %, 7.8 %] | 0.005 | 7.2 % [4.2 %, 9.6 %] | 0.005 |

Two-Gaussian fits: centred means -1.45 / 0.76 dB, sds 3.36 / 2.17, weights 0.34 / 0.66; raw means 11.26 / 25.16 dB, sds 3.76 / 5.44, weights 0.79 / 0.21. Co-movement of two tracks of one window: mean r 0.27 for the same rotor (387 pairs), 0.18 for different rotors (1128 pairs); the v3 model's rotor-common share of a line's block variance is 0.05 (different rotors 0). A track's own block sd, median by order group: k 1-2 1.85 dB (32), k 3-8 3.12 dB (63), k 9-24 2.48 dB (54), k 25-60 n/a dB (0), k 61+ 1.51 dB (18).

## (c) Floor masking

Three floors, each centred on its own mean over the track's blocks (window for `u`): `local` -- the prominence's own denominator (q25 of the cells 3-12 bins either side, mic-summed; it shares the floor estimate's noise with the prominence, which biases its r slightly negative); `u+u_j` -- `measure_floor`'s comb-masked control band that holds the line, speed law removed (the v3 floor at the line); `u` -- the mean over bands of their deviations (the floor level latent). Pearson r of block prominence against the floor per track (>= 5 blocks) and pooled within tracks with a 90 % window bootstrap; the share of blocks whose floor sits > +1 sigma_u over its mean (sigma_u from `results/noise_v3/wander/<rig>.json`).

| rig | floor | tracks | r per track, median [IQR] | r < 0 | r < -0.5 | r pooled within [90 %] | slope dB/dB | floor sd within track dB | floor > +1 sigma_u: absent / present / all | floor above its mean: absent / present |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dregon (sigma_u 2.83) | local | 767 | -0.47 [-0.75, 0.04] | 73.5 % | 47.8 % | -0.25 [-0.31, -0.20] | -0.49 | 1.47 | 5.0 % / 1.4 % / 3.6 % | 58.2 % / 33.9 % |
| dregon (sigma_u 2.83) | uj | 767 | -0.19 [-0.51, 0.22] | 62.2 % | 25.9 % | -0.12 [-0.17, -0.08] | -0.23 | 1.53 | 4.2 % / 2.6 % / 3.6 % | 50.5 % / 40.4 % |
| dregon (sigma_u 2.83) | u | 767 | -0.24 [-0.53, 0.13] | 67.9 % | 28.4 % | -0.16 [-0.22, -0.12] | -0.43 | 1.13 | 2.7 % / 1.4 % / 2.1 % | 53.3 % / 38.2 % |
| michaels (sigma_u 1.73) | local | 262 | -0.47 [-0.68, -0.13] | 81.7 % | 46.6 % | -0.36 [-0.44, -0.27] | -0.67 | 2.03 | 19.6 % / 9.3 % / 14.9 % | 60.9 % / 33.5 % |
| michaels (sigma_u 1.73) | uj | 216 | -0.15 [-0.43, 0.17] | 63.0 % | 18.5 % | -0.13 [-0.22, -0.03] | -0.25 | 1.89 | 12.8 % / 9.5 % / 11.3 % | 51.0 % / 43.4 % |
| michaels (sigma_u 1.73) | u | 262 | -0.18 [-0.44, 0.04] | 71.8 % | 19.1 % | -0.19 [-0.25, -0.13] | -0.59 | 1.21 | 7.5 % / 3.5 % / 5.7 % | 51.9 % / 40.1 % |

- dregon: the block prominence against the rotor's own block label speed, pooled within tracks: r = 0.07.
- michaels: the block prominence against the rotor's own block label speed, pooled within tracks: r = 0.03.

## Verdict

**(b) the line amplitude itself (wider Gaussian) dominates DREGON's intermittency.** DREGON has 767 intermittent tracks (3840 absent blocks) against 44 resolvable and 17983 always-under. (a) Label: no unbiased carrier re-estimate removes it; the best, 'label: window mean', returns 12.9 % of the absent blocks, takes 45.2 % of the present ones and leaves 61.0 % of the tracks intermittent. The +-3-bin search reaches 6 dB in 27.5 % of absent blocks, 15.9 % of them on another rotor's line and 11.6 % elsewhere (always-under null 0.9 %); on Michael's audio-refined label that rate is 11.8 %. (c) Floor: the prominence against the floor under the line (median per-track r -0.47, 47.8 % of tracks below -0.5; pooled within tracks -0.25, so the floor carries 6.1 % of the pooled scatter), but holding the floor at its mean ('floor at mean: local') leaves 77.8 % of the tracks intermittent (20.6 % become steadily under: lines seen only in floor dips); by order group it matters most at the bottom -- holding the local floor leaves 44.2 % of the k 1-2 tracks intermittent (53.5 % steadily under), against 77.8 %-93.6 % at k >= 9. (b) Amplitude: around its track mean the prominence scatters by 2.95 dB; a Gaussian at the resolvable lines' sigma 1.48 dB predicts 1.18 dB, 1.48 with the measured local floor added, and would call only 47.5 % of these tracks intermittent; the near-threshold tracks, chosen on their median alone (744 tracks), scatter by 2.28 dB against 1.54, so the excess is not the 20-80 % rule selecting noisy tracks; and their own sds spread by 0.52 (sd / mean) against 0.33 for one width-matched sigma (3.36 dB): the lines differ in how much they wander. The line's own sigma, fitted with the floor in, is 5.03 dB against 2.52 dB for the resolvable lines in the same observable. At that width the shape reads 'wider Gaussian': excess kurtosis 1.05 vs 0.90 (p 0.134); no second mode (raw BIC gain of two Gaussians -52 vs 910 for the Gaussian, the two components 1.04 sds apart); blocks > 6 dB under the track mean 1.9 % vs 1.2 %, blocks under 3 dB 27.1 % vs 32.9 %; lag-1 autocorrelation 0.23 vs 0.15. Michael's control (262 intermittent tracks, audio-refined label): the best label re-estimate leaves 60.7 % intermittent, the floor held 83.2 %; the prominence scatters by 3.76 dB against 2.07 for its resolvable lines' sigma 1.98 dB with the floor, the line sigma fitted is 5.81 dB against 2.46 for its resolvable lines, and the shape reads 'wider Gaussian'. Implication: a Gaussian OU is SUFFICIENT in form -- neither a switching component nor the carrier deviation track is called for -- but ONE rig sigma_v is not: the lines near the 6 dB threshold wander by 5.03 dB where the resolvable ones need 2.52 dB in the same observable, and the intermittent tracks' own block sd runs 1.48 dB at k 1-2, 1.70 dB at k 3-8, 2.07 dB at k 9-24, 2.62 dB at k 25-60, 4.35 dB at k 61+: sigma_v has to depend on the line (its order group, or its level over the floor), not be pinned at the resolvable lines' value.

## Figures

- `example_tracks.png`
- `prominence_hist.png`
- `counterfactuals.png`
- `peak_search_offsets.png`
- `floor_scatter.png`

Detail: `intermittency.json`.
