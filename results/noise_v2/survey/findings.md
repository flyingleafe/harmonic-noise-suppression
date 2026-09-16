# Corpus survey for noise model v2: which drone-noise recordings become fit points

Scripts: `scripts/noise_v2_bench_speed.py` (per-recording speed reading),
`scripts/noise_v2_corpus_survey.py` (corpus verdicts, tables, figure, ingestion
manifest). Data: `bench_speeds_remote.json` (job `nv2-bench-remote-e1e19f`,
uni-cpu, 229 recordings), `bench_speeds_local.json` (the two corpora whose raw
tree exists only in `data/`, 31 recordings), `survey.json`, `survey_table.md`.
Figures: `bench_speeds_dregon_law.png`, `bench_speeds_gates.png`,
`survey_speeds.png`. Dataset: `noise-v2-bench-points`, manifest
`src/data_processing/noise_v2_bench_points.json`.

## The estimator and the tolerance rule

One speed per rotor per recording, from a long stationary window:

1. The motor-ON part is the set of 1 s blocks inside 12 dB of the loudest
   block; the window is the longest run of those blocks inside 4 dB of the
   active median, up to 16 s. This step is load-bearing: a DREGON bench file is
   25 s long but holds only 11-35 s of motor, and its silent tail carries a
   fixed 88 rev/s room tone that the harmonic sum locks onto (measured on
   `motor_Motor3_50`: 49.1 rev/s inside the run, 88.3 rev/s outside it).
2. Welch spectrum, median-averaged, per channel, then averaged over channels;
   resolution target 0.5 Hz with at least 4 half-overlapped segments.
3. Harmonic-sum score over 15-150 rev/s: the mean line prominence over the
   broadband floor at `k*f`, `k = 1..40` below 3 kHz, clipped at 30 dB.
4. Octave mapping: halve while the ODD harmonics of the half clear 6 dB on half
   of the tested orders (`k = 1,3,5,7,9,11`). 6 of the 20 DREGON bench readings
   are halved this way; the rest are already the shaft rate.
5. Per-rotor split: peaks inside +-5 % of `k*f` at orders 8-32, converted back
   to fundamentals. This resolves 4 rotors on `motor_allMotors_70`
   (65.23/67.61/68.76 rev/s) and 2-4 rotors on most DroneAudioSet recordings.

**Tolerance rule, as applied.** A recording is usable when (a) the harmonic-sum
peak clears the best rival comb OUTSIDE its own harmonic family by >= 3 dB,
(b) the two disjoint halves of the window agree to <= 1 rev/s, and (c) the
window is >= 8 s. The published tolerance is
`max(half-to-half difference, 0.5*df, 0.25) rev/s` — 0.25 rev/s on 72 of the 73
fit points, 0.56 rev/s on the ChuMS run.

Two deviations from the literal wording, both measured:

- *Rival exclusion is the harmonic family, not only the octave.* With only
  `f/2, f, 2f` excluded, the runner-up of a clean bench recording is its own
  third sub/super-harmonic — the same comb read every third tooth. That
  rejected `motor_Motor1_50`, whose reading is right to 0.09 rev/s (margin
  2.22 dB octave-only against 8.26 dB family-excluded). Every `m/n`, `m,n <= 4`
  is therefore excluded; both margins are stored per row
  (`margin_db`, `margin_octave_only_db`).
- *The margin is measured on the comb family score* `max(score(f), score(2f))`.
  A two-bladed rotor puts its energy on the blade-pass line, so scoring at the
  shaft rate dilutes the margin with the missing odd teeth: on
  `drone2_low_50cm_M_down_File3` the shaft-rate score is 7.39 dB while the
  blade-pass score of the same rotor is 11.99 dB.

The two halves are also octave-folded before the +-1 rev/s test, because an 8 s
half has weaker odd-harmonic evidence than the 16 s window it came from; the
fold is recorded per row (`half_octave_folded`).

## Cross-checks (numbers)

**DREGON single-motor bench against the validated throttle law**
(`rate = 0.975*throttle + 0.37 rev/s`, `bench_speeds_dregon_law.png`):
20 of 20 single-motor recordings usable, mean |error| **0.555 rev/s**, max
1.42 rev/s, RMS 0.703 rev/s. Per-throttle means 49.04 / 58.83 / 68.42 / 78.36 /
88.07 rev/s against the campaign's 49.11 / 58.98 / 68.53 / 78.37 / 88.17. An
independent refit of my own readings gives **slope 0.97577, intercept
0.2395 rev/s, R^2 0.997444, residual RMS 0.699 rev/s** against the campaign's
0.975 / 0.37 / 0.99788 / 0.63 — the residual is the documented motor-to-motor
spread, not estimator error.

**SPCUP19 AGH single rotors against the accepted blind readings**: 7 of 8
usable. Six agree to <= 0.27 rev/s (mean |error| **0.148 rev/s**: take 0
79.82 vs 79.71, take 2 113.54 vs 113.78, take 3 97.48 vs 97.43, take 5 98.08 vs
97.81, take 6 96.78 vs 96.59, take 7 96.78 vs 96.81). Take 1 reads **132.30 vs
66.33 rev/s, a factor 1.995** — the odd harmonics at 66 rev/s do not clear the
6 dB rule, and the blind campaign already flagged this take (`fvk_ratio_double`
1.021, the weakest odd-harmonic evidence of its set). Mean |error| over all
seven is therefore 9.55 rev/s, carried entirely by that one octave. Take 4
(77.66 vs 77.74) is rejected by the margin rule at 1.62 dB.

**KAIST rotating machine (out-of-domain control, stated 3010 RPM =
50.167 rev/s)**: 0 of 5 usable. The readings are +113.8 % / +113.7 % / -54.5 % /
+21.5 % / +140.9 % off the nominal and every one is refused by the margin rule
(margins -7.75 to +8.07 dB with the rival comb winning on three). This is the
corpus where the blind campaign recorded a FALSE ACCEPT (`0Nm_BPFO_*` at
1.82x nominal); the margin rule refuses all five here.

**DroneAudioSet against the paper's stated spectral lines** (arXiv:2510.15383
Fig. 7: 168/235 Hz for D_large low/high, 156/259 Hz for D_small low/high).
Cell medians over the fit points, and the blade-pass line they imply:

| cell | n | median rev/s | 2f (Hz) | paper (Hz) | error |
|---|---|---|---|---|---|
| drone1 low | 1 | 83.44 | 166.9 | 168 (D_large low) | -0.7 % |
| drone1 high | 9 | 118.88 | 237.8 | 235 (D_large high) | +1.2 % |
| drone2 high | 16 | 129.39 | 258.8 | 259 (D_small high) | -0.1 % |
| drone2 low | 19 | 106.63 | 213.3 | 156 (D_small low) | **+36.7 %** |

Three of the four cells match the paper to <= 1.2 %, which **identifies the
anonymous path tokens**: `drone1` is D_large (DJI F450, 9.4 in props) and
`drone2` is D_small (DJI F330, 8 in props). It also settles what the paper's
marked "fundamental" is: the blade-pass line at twice the shaft rate, not the
shaft rate (168 Hz of shaft would be 10 080 RPM on a 9.4 in propeller). The
`drone2 low` cell is the one disagreement: the measured line sits at 213 Hz on
19 recordings from 3 mic groups and 2 mic distances, so the disagreement is in
the published figure or in the session it was drawn from, not in one reading.

## Verdicts

Full table with locations, licences and reasons: `survey_table.md`. Counts:

| corpus | recordings read | usable | verdict |
|---|---|---|---|
| DREGON single-motor bench | 21 | 20 | USABLE |
| SPCUP19 AGH single rotors | 8 | 7 | USABLE |
| DroneAudioSet drone-only | 168 | 50 | PARTLY USABLE |
| SPCUP19 ChuMS propeller rig | 9 | 1 | PARTLY USABLE |
| SPCUP19 static / hover (4 rigs) | 18 | 0 | REJECTED |
| KAIST-rotating-acoustic | 5 | 0 | CONTROL ONLY |
| drone_audio (yes_drone) | 24 | 0 | REJECTED |
| zenodo_drone_noises | 7 | 1 | REJECTED (no rig identity) |
| AVQ | — | — | REJECTED (free flight) |
| DronePrint / MAVD / DroneNoise DB / ESC-50 | — | — | REJECTED (access/type) |

**New corpus found inside SPCUP19.** The ChuMS team package is not 216
unlabelled arrays (as the published frames expose it) but a **propeller test
rig**: `UAV_rotor_recordings.mat` holds 9 runs — 1, 2 or 3 propellers, three
repeats each — of 73-75 s on **8 calibrated microphones** with their positions
in the file (`MicPositions`, mm) and a per-mic OASPL of 88.9-98.4 dB. No RPM is
recorded anywhere in it. Only `3prop_repeat2` clears the margin rule
(76.92/77.26/77.51 rev/s, margin 3.26 dB); the other 8 runs read
-1.26 to +2.63 dB.

**Why the four-rotor static corpora fail.** `motor_allMotors_70` is the control
that shows it is geometry, not the estimator: the same rig and throttle that
give a 7.5-8.2 dB margin on one motor give **1.41 dB** with four, because a
candidate rate between the rotors collects teeth from all of them. All 18 SPCUP
static/hover windows fail the same way (margins -0.53 to +2.03 dB), which
reproduces the blind campaign's SPCUP static result (no window cleared its
off-comb null) with a different instrument.

**Rejections with their reason, by count** (181 rejected recordings):
118 DroneAudioSet recordings (margin below 3 dB — mostly the `M_up` mic group,
whose above-drone position sees several blade-pass lines 16 Hz apart rather
than one comb), 24 drone_audio clips (1.02 s window against the 8 s rule, and
margins -0.71 to +0.72 dB), 18 SPCUP static, 8 ChuMS, 6 zenodo, 5 KAIST,
1 DREGON (`motor_allMotors_70`), 1 SPCUP AGH take.

**Cell-consistency drop.** Five usable DroneAudioSet readings sit at a factor
0.50-0.51 of their own (rig, throttle) cell median — an octave failure on that
recording, not a different speed. They are excluded from the fit points and
listed in `survey.json:cell_octave_outliers`; 50 usable DroneAudioSet readings
become 45 fit points.

## What the fit points are

**73 fit points** over **12 distinct rig/condition points**:

- `dregon_mikrokopter_single_motor` at 50/60/70/80/90 % throttle — 20 points,
  5 conditions, 1 rotor each.
- `spcup_AGH` single rotor — 7 points, 1 condition, 1 rotor each, 66-114 rev/s.
- `daset_drone1` (DJI F450) low/high and `daset_drone2` (DJI F330) low/high —
  45 points, 4 conditions, 2-4 resolved rotors each.
- `spcup_ChuMS_3prop_bench` — 1 point, 3 propellers.

Median tolerance 0.25 rev/s on every corpus except ChuMS (0.56 rev/s). The
`zenodo_drone_noises` reading that passes the gate (n118, 110.5-111.4 rev/s,
margin 3.38 dB) is deliberately NOT a fit point: the corpus records no rig, so
a per-rig noise parameter cannot be attached to it.

Published as `noise-v2-bench-points` (`tdframe-v1`): per point, the stationary
audio at its native rate with every channel kept (<= 30 s), `meta.rig`,
`meta.corpus`, `meta.speed_rev_s` (one entry per resolved rotor),
`meta.speed_source`, `meta.speed_tolerance`, plus the source recording id and
offset. The labels are estimator output, so the committed manifest is hashed
into the derivation spec: a re-estimation mints a new dataset identity instead
of silently republishing different labels.

## Limits worth carrying into the model work

1. **One speed per rotor is available on 12 rig/condition points only, and 5 of
   them are the same DREGON motor at 5 throttles.** The genuinely new rigs are
   two DJI airframes (F450, F330) at two throttles each, the AGH single rotor,
   and the Dotterel 3-propeller rig.
2. **The per-rotor spread is small where it is resolvable**: 0.3-3.5 rev/s on
   DroneAudioSet cells, 3.5 rev/s on `motor_allMotors_70`. A shared shaft-speed
   error across rotors (the v2 nu_r construction) is not contradicted by these
   points, but they constrain the MEAN speed only — they carry no time-resolved
   speed track at all.
3. **The odd-harmonic octave test is the single point of failure.** It carries
   1 SPCUP take and 5 DroneAudioSet recordings into a factor-2 error; the cell
   median catches the DroneAudioSet ones because a cell has 9-20 recordings,
   and nothing catches the SPCUP take because its cell has 8 different speeds.
4. **No new corpus adds telemetry.** Every extra point is an estimate with a
   0.25 rev/s tolerance, against DREGON's and Michael's measured tracks; use
   them for rig-to-rig transfer of the noise parameters, not for anything that
   needs a speed trajectory.
