# Corpus survey for noise model v2: which drone-noise recordings become fit points

Second pass. It corrects two defects of the first pass: AVQ was rejected as
"free flight" without reading its specification table, and the rival-margin
rule counted a neighbouring ROTOR of the same rig as a rival comb.

Scripts: `scripts/noise_v2_bench_speed.py` (per-recording reading, figures),
`scripts/noise_v2_corpus_survey.py` (corpus verdicts, tables, figures,
ingestion manifest). Readings: `bench_speeds_dload.json` (job
`nv2-dload2-80fd43`, uni-cpu — DREGON bench, SPCUP19, KAIST, AVQ: 68 rows),
`bench_speeds_remote.json` (job `nv2-bench-v2-9de4ae`, uni-cpu —
DroneAudioSet, ChuMS: 177 rows), `bench_speeds_local.json` (the two corpora
whose raw tree exists only in `data/`: 31 rows). Tables: `survey.json`,
`survey_table.md`. Figures: `bench_speeds_dregon_law.png`,
`bench_speeds_gates.png`, `survey_speeds.png`,
`survey_multirotor_split.png`. Dataset: `noise-v2-bench-points`, manifest
`src/data_processing/noise_v2_bench_points.json`.

Reproduce (no audio is read for the figures):

```bash
PYTHONPATH=src python scripts/noise_v2_bench_speed.py --figures-from \
    results/noise_v2/survey/bench_speeds_{dload,remote,local}.json
PYTHONPATH=src python scripts/noise_v2_corpus_survey.py --speeds \
    results/noise_v2/survey/bench_speeds_{dload,remote,local}.json \
    --manifest-out src/data_processing/noise_v2_bench_points.json
```

## The estimator and the tolerance rule

One speed per rotor per recording, from a long stationary window.

1. **Stationary window.** The motor-ON part is the set of 1 s blocks inside
   12 dB of the loudest block; the window is the longest run of those blocks
   inside 4 dB of the active median, up to 16 s on a single-rotor rig and up
   to **30 s in multi-rotor mode**. This step is load-bearing: a DREGON bench
   file is 25-45 s long but holds 11-35 s of motor, and its silent tail
   carries a fixed 88 rev/s room tone that the harmonic sum locks onto
   (`motor_Motor3_50`: 49.1 rev/s inside the run, 88.3 rev/s outside it).
2. **Welch spectrum**, median-averaged, per channel, then averaged over
   channels. Single-rotor mode keeps the 0.5 Hz resolution target
   (df = 0.3365 Hz at 44.1 kHz); multi-rotor mode uses a fixed 4 s periodic
   Hann segment, so df = 0.25 Hz on every corpus and the first resolvable
   harmonic order is a number, not a guess.
3. **Harmonic sum** over 15-150 rev/s: the mean line prominence over the
   broadband floor at `k*f`, `k = 1..40` below 3 kHz, clipped at 30 dB. On a
   multi-rotor rig the argmax is the MEAN comb rate `f̄`.
4. **Octave mapping**: halve while the ODD harmonics of the half clear 6 dB on
   half of the tested orders (`k = 1,3,5,7,9,11`). 6 of the 20 DREGON bench
   readings are halved this way; the rest are already the shaft rate.
5. **Rival margin.** The accepted comb-family score `max(score(f), score(2f))`
   minus the best RIVAL. A rival is a candidate that is not a small-rational
   `m/n` relative of the accepted rate (`m, n <= 4`, 11 ratios — the same comb
   read every n-th tooth) and, in multi-rotor mode, not inside **±6 % of the
   accepted rate**: a rotor 1-3 rev/s from `f̄` is 1.4-4.3 % away at
   70 rev/s, so the first pass scored one rotor of the rig against another.
   Both margins are stored per row (`margin_db` with the neighbourhood
   excluded, `margin_family_only_db` without it).
6. **Per-rotor split** (multi-rotor mode only). The split runs on the LINE
   comb — the comb with a tooth at every order, which on a two-bladed rotor is
   the blade-pass line at `2f`, not the shaft rate. Without that choice the
   test is meaningless: on `drone2_low_50cm_M_down_File3` the dominant rotor
   appears at shaft orders 8, 10, 12, ..., 28 (even only, longest consecutive
   run 1) and at blade-pass orders 4, 5, 6, ..., 14 (run 11). Orders start at
   `k_min = ceil(2 * 4 * df * scale / 0.3)` — two periodic-Hann main lobes
   (4 df = 1 Hz at T = 4 s) for a 0.3 rev/s rotor-to-rotor difference, so
   `k_min = 7` on an `as_found` comb and 4 blade-pass orders (= shaft order 8)
   on a halved one. Inside `k*f̄ ± 6 %` up to 4 peaks 6 dB above the band's
   LOCAL MEDIAN are converted to rates `peak/k`; a rate is a resolved rotor
   only if it repeats within 0.3 rev/s across **≥ 3 consecutive** usable
   orders. Fewer resolved rates than rotors is reported as
   `multiplicity_unresolved`, not as a failure.

**Tolerance rule, as applied.** A recording is usable when (a) the margin of
item 5 is ≥ 3 dB, (b) the two disjoint halves of the window agree on `f̄` to
≤ 1 rev/s, and (c) the window is ≥ 8 s. The published tolerance is
`max(half-to-half difference, 0.5*df, 0.25) rev/s` — 0.25 rev/s on 42 of the
78 fit points, median 0.25 rev/s, worst 0.88 rev/s.

## Validation

**DREGON single-motor bench against the validated throttle law**
(`rate = 0.975*throttle + 0.37 rev/s`, `bench_speeds_dregon_law.png`):
**20 of 20** usable, mean |error| **0.5552 rev/s**, max 1.42, RMS 0.703 —
the first pass reproduced exactly (single-rotor mode is unchanged by
construction: the 16 s window, the resolution-target segment and the
family-only rival rule all stand, because a single-rotor rig has no
neighbouring rotor to mistake for a rival and nothing to split).
Per-throttle means 49.04 / 58.83 / 68.42 / 78.36 / 88.07 rev/s against the
campaign's 49.11 / 58.98 / 68.53 / 78.37 / 88.17. Independent refit of these
readings: slope 0.97577, intercept 0.2395 rev/s, R² 0.997444, residual RMS
0.699 rev/s.

**`motor_allMotors_70`, the four-rotor control on the same rig and throttle.**
Still rejected, and the rule change did not move it: margin **0.771 dB with
the ±6 % neighbourhood excluded and 0.771 dB without it** — the limiting rival
sits at 112.72 rev/s, which is 1.642 × 68.66, i.e. **5/3**, not a neighbour.
The split resolves one rotor line at 68.74 rev/s with support at orders
8-16, 18, 21, 24 (longest consecutive run 7); the next two clusters, 67.66 and
69.57 rev/s, appear at 8 orders each but never in 3 consecutive ones, so
`n_resolved = 1` and `multiplicity_unresolved = true`. The first pass reported
three rotors at 65.23/67.61/68.76 rev/s; those came from ONE order (16), where
the same band still gives 65.22 / 67.59 / 68.84 rev/s. The repeat requirement
is what removed them.

**SPCUP19 AGH single rotors against the accepted blind readings**: 7 of 8
usable, unchanged. Six agree to ≤ 0.27 rev/s (take 0 79.82 vs 79.71, take 2
113.54 vs 113.78, take 3 97.48 vs 97.43, take 5 98.08 vs 97.81, take 6 96.78
vs 96.59, take 7 96.78 vs 96.81). Take 1 reads 132.30 against 66.33 rev/s, a
factor 1.995 (the odd harmonics at 66 rev/s do not clear the 6 dB rule); take
4 (77.66 vs 77.74) is refused by the margin rule at 1.62 dB. Mean |error| over
all 8 takes is 8.37 rev/s, carried entirely by that one octave.

**KAIST rotating machine (out-of-domain control, stated 3010 RPM =
50.167 rev/s)**: 0 of 5 usable, unchanged. The readings are +113.8 / +113.7 /
−54.5 / +21.5 / +140.9 % off nominal.

**DroneAudioSet against the paper's stated lines** (arXiv:2510.15383 Fig. 7).
Cell medians over the fit points and the blade-pass line they imply:

| cell | n | median rev/s | 2f (Hz) | paper (Hz) | error |
|---|---|---|---|---|---|
| drone1 low | 2 | 85.52 | 171.0 | 168 (D_large low) | +1.8 % |
| drone1 high | 13 | 118.82 | 237.6 | 235 (D_large high) | +1.1 % |
| drone2 high | 15 | 129.56 | 259.1 | 259 (D_small high) | +0.05 % |
| drone2 low | 22 | 106.60 | 213.2 | 156 (D_small low) | **+36.7 %** |

Three of four cells match to ≤ 1.8 %, which identifies the anonymous path
tokens (`drone1` = D_large, DJI F450, 9.4 in props; `drone2` = D_small, DJI
F330, 8 in props) and settles that the paper's marked line is the blade-pass
line at twice the shaft rate. The `drone2 low` cell is the one disagreement:
the measured line sits at 213 Hz on 22 recordings from 3 mic groups and 2 mic
distances, so the disagreement is in the published figure, not in one reading.

## AVQ: the corrected verdict

The AVQ specification table (`webspace.eecs.qmul.ac.uk/lin.wang/demo/avq.html`,
column `Type` = EO and column `Drone` = "constant") marks **four** of the
twelve sequences as ego-noise only at a CONSTANT throttle setting: S1 seq1
50 % (120 s), S1 seq2 100 % (120 s), S1 seq3 150 % (40 s), S2 seq1 100 %
(210 s). One throttle per recording and no manoeuvre is the bench-class
definition used here, so the first pass's "free flight, not a bench/static
candidate by type" was wrong about these four. S2 seq2 is ego-noise only but
DYNAMIC throttle; S2 seq5/seq6 are constant 100 % but speech+ego-noise
MIXTURES; the remaining five are speech with the drone muted or dynamic. Those
eight stay excluded by type. The blind campaign's octave warning for AVQ
(median `fvk_ratio_double` 1.044) was measured on the flight sequences.

Read from `dload:AVQ` (8 ch, 44.1 kHz) in consecutive 30 s windows, one
reading per window:

| sequence | throttle | windows | usable | f̄ per window (rev/s) | margin dB |
|---|---|---|---|---|---|
| S1_seq1 | 50 % | 4 | 2 | 78.00, **78.00**, 39.46, **77.79** | 5.57, 4.68, 2.23, 4.37 |
| S1_seq2 | 100 % | 4 | 2 | **97.78**, 16.33, 97.34, **98.24** | 3.79, −0.29, 2.35, 4.80 |
| S1_seq3 | 150 % | 1 | 0 | 111.38 | 3.33 |
| S2_seq1 | 100 % | 7 | 0 | 88.98, 45.48, 45.25, 93.48, 45.03, 92.52, 93.12 | 2.19 … −0.81 |

The three S1 settings order correctly and monotonically — 78.0 / 97.8 /
111.4 rev/s for 50 / 100 / 150 % — which corroborates that the table's
throttle column is a real setpoint. S1_seq1 window 2 (39.46 rev/s) and
S1_seq2 window 1 (16.33 rev/s) are octave/lock failures refused by the margin
and stability gates. S1_seq3 clears the margin (3.33 dB) but fails the
stability gate (half-to-half 1.5 rev/s > 1) and its 40 s sequence yields only
one 30 s window. S2_seq1 is the same 100 % setting in the other session and
never clears 3 dB; its windows alternate between ~93 and ~45 rev/s (the
octave), so the session itself is not stationary at the reading level.

**AVQ therefore adds 4 fit points over 2 rig/condition cells** (S1 at 50 % and
at 100 %), each with `n_rotors = 4`, `n_resolved` 1 or 2 and
`multiplicity_unresolved = true`.

## What the multi-rotor correction actually did (and what it did not)

Measured over the 240 multi-rotor readings (`survey.json:multi_rotor`):

- The ±6 % neighbourhood exclusion **moved the limiting rival on 46 of 240**
  readings, median gain **0.51 dB**, maximum 2.33 dB, and **flipped exactly
  one recording** over the 3 dB bar by itself
  (`drone1_low_25cm_M_center_File5`, 1.582 → 3.555 dB).
- Of the 183 rejected multi-rotor readings, **1** has its limiting rival
  inside ±6 % of `f̄` (none can, after the fix), 17 inside ±8 %, 30 inside
  ±10 %. The premise that a neighbouring rotor is normally the scored rival is
  therefore not what the spectra show.
- **162 of the 183** rejected readings have their limiting rival at a DEEPER
  `m/n` relative of the accepted comb (`max(m, n)` between 5 and 12) — the
  same comb read every n-th tooth, which the published `m, n <= 4` exclusion
  does not cover. 18 of 18 SPCUP static windows, 101 of 116 DroneAudioSet,
  1 of 1 DREGON `allMotors`.
- That exclusion depth is not the cause either. A counterfactual pass with
  `--family-max 8` (43 ratios instead of 11; job `nv2-fam8-8cc561`, DREGON +
  SPCUP + KAIST + AVQ, 68 readings) **flips no verdict at all**: 31 usable
  either way, SPCUP static median margin −0.01 → +0.04 dB, DREGON law error
  0.5552 rev/s unchanged, and the KAIST control still refuses all 5.

The measured cause of the four-rotor rejection is that **the comb itself
collapses**, not that the wrong candidate is called a rival. On the same rig
at the same throttle, DREGON 70 %:

| recording | rotors | comb score | margin |
|---|---|---|---|
| `motor_Motor1_70` | 1 | 15.83 dB | 9.75 dB |
| `motor_Motor2_70` | 1 | 17.82 dB | 5.56 dB |
| `motor_Motor3_70` | 1 | 15.34 dB | 8.20 dB |
| `motor_Motor4_70` | 1 | 16.72 dB | 7.27 dB |
| `motor_allMotors_70` | 4 | **6.43 dB** | **0.77 dB** |

The 18 SPCUP static windows score 1.54-6.31 dB (median ≈ 4.2 dB) against
7.38-15.86 dB for the 8 AGH single-rotor takes, and their margins are
−0.58 … +2.00 dB (median −0.01). With four rotors the tonal energy of each
rotor and its prominence over a four-times-higher broadband floor both fall,
so NO candidate comb is distinguished and no rival-exclusion rule can create a
3 dB margin. The DroneAudioSet recordings that do pass are the ones where one
rotor dominates the mic: median comb score 9.10 dB for the 52 usable against
6.04 dB for the 116 rejected.

**Cost of the 30 s multi-rotor window.** The ChuMS propeller rig runs at
uncontrolled propeller speeds and drifts, so a longer window smears its lines:
`3prop_repeat2` reads 1.88 dB at 30 s against **3.93 dB at 16 s**
(`--multi-window-s 16`, where it also resolves two propellers at 76.84 /
77.32 rev/s). It was the corpus's only fit point in the first pass, so ChuMS
drops from 1 usable to 0. The other five multi-propeller runs improve
(2prop_repeat1 0.33 → 0.42, 3prop_repeat1 −0.03 → 0.70, 2prop_repeat2
−0.50 → 0.99, 3prop_repeat3 0.12 → 0.96) and 2prop_repeat3 falls 2.63 → 2.06;
none of them reaches 3 dB either way.

## Per-rig rotor multiplicity

`n_res` is the number of rotor rates the split resolved; `n_rotors` is how
many run (`survey.json:corpora[].stats.n_resolved_by_rig`, usable readings
only, figure `survey_multirotor_split.png`).

| rig | rotors | usable | n_res histogram | unresolved | max spread rev/s |
|---|---|---|---|---|---|
| `dregon_mikrokopter_single_motor` | 1 | 20 | 1: 20 | 0 | 0 |
| `spcup_AGH` (single rotor) | 1 | 7 | 1: 7 | 0 | 0 |
| `daset_drone1` | 4 | 15 | 1: 7, 2: 7, 3: 1 | 15 | 4.82 |
| `daset_drone2` | 4 | 37 | 1: 33, 2: 4 | 37 | 0.43 |
| `avq_quadrotor` | 4 | 4 | 1: 3, 2: 1 | 4 | 0.37 |
| `zenodo_unknown_n117` | 4 | 1 | 1: 1 | 1 | 0 |

No usable multi-rotor recording resolves four rotors: 51 of the 78 fit points
carry `multiplicity_unresolved = true`, 12 resolve two rates and 1 resolves
three. The rotors of a bolted-down airframe at a fixed throttle are therefore
coincident to within the 0.3 rev/s tolerance on all but 13 readings — and
where they are not (`daset_drone1`, spread up to 4.8 rev/s) the spread is
large enough to matter for a per-rotor model.

## Verdicts

Full table with locations, licences and per-recording reasons:
`survey_table.md`. Counts, first pass → this pass:

| corpus | read | usable (pass 1 → 2) | verdict |
|---|---|---|---|
| DREGON single-motor bench | 21 | 20 → 20 | USABLE |
| SPCUP19 AGH single rotors | 8 | 7 → 7 | USABLE |
| DroneAudioSet drone-only | 168 | 50 → 52 | PARTLY USABLE |
| **AVQ constant-throttle ego-noise** | **16 windows / 4 sequences** | **— → 4** | **USABLE** |
| SPCUP19 ChuMS propeller rig | 9 | 1 → 0 | PARTLY USABLE (no point now) |
| SPCUP19 static / hover (4 rigs) | 18 | 0 → 0 | REJECTED |
| KAIST-rotating-acoustic | 5 | 0 → 0 | CONTROL ONLY |
| drone_audio (`yes_drone`, 24 sampled) | 24 | 0 → 0 | REJECTED |
| zenodo_drone_noises | 7 | 1 → 1 | REJECTED (no rig identity) |
| DronePrint / MAVD / DroneNoise DB / ESC-50 | — | — | REJECTED (access/type) |

**Usable rig/condition points: 12 → 13** counted as the survey counts them
(including the one `zenodo` reading that has no rig identity and is therefore
not a fit point), i.e. **11 → 12 fit-point cells**: +2 AVQ (S1 at 50 %, S1 at
100 %), −1 ChuMS 3-propeller, the 5 DREGON throttles, the AGH single rotor and
the 4 DroneAudioSet cells unchanged.

**Rejections with their reason** (192 rejected readings): 116 DroneAudioSet
(margin below 3 dB — mostly the `M_up` mic group above the airframe, whose
readings also lock below the 15 rev/s floor), 24 drone_audio clips (1.02 s
window against the 8 s rule), 18 SPCUP static/hover (margins −0.58 … +2.00 dB;
the comb collapse above), 12 AVQ windows (8 below 3 dB, S1_seq3 on the
stability gate, the rest octave/lock failures), 9 ChuMS runs, 6 zenodo, 5
KAIST (the control), 1 DREGON (`motor_allMotors_70`), 1 SPCUP AGH take (take
4, 1.62 dB).

**Cell-consistency drop.** Five usable DroneAudioSet readings sit at a factor
0.50-0.51 of their own (rig, throttle) cell median — an octave failure on that
recording, not a different speed. They are listed in
`survey.json:cell_octave_outliers` and excluded from the fit points, so 52
usable DroneAudioSet readings become 47 fit points.

## What the fit points are

**78 fit points** over **12 rig/condition cells** (first pass: 73 over 11):

- `dregon_mikrokopter_single_motor` at 50/60/70/80/90 % throttle — 20 points,
  5 cells, 1 rotor each.
- `spcup_AGH` single rotor — 7 points, 1 cell, 1 rotor each, 78-132 rev/s.
- `daset_drone1` (DJI F450) low/high and `daset_drone2` (DJI F330) low/high —
  47 points, 4 cells, 1-3 resolved rotors of 4.
- `avq_quadrotor` at 50 % and 100 % — 4 points, 2 cells, 1-2 resolved rotors
  of 4, 77.8-98.2 rev/s.

Per point: the stationary audio at its native rate with every channel kept
(≤ 30 s), `meta.rig`, `meta.corpus`, `meta.speed_rev_s` (one entry per
resolved rotor), `meta.speed_source`, `meta.speed_tolerance`,
`label.n_rotors_resolved`, `label.multiplicity_unresolved`, plus the source
recording id and offset. The labels are estimator output, so the committed
manifest is hashed into the derivation spec (`gen.manifest`,
`recipe_version` 2): a re-estimation mints a new dataset identity instead of
silently republishing different labels.

## Limits worth carrying into the model work

1. **One speed per rotor is available on 12 rig/condition cells only, and 5 of
   them are the same DREGON motor at 5 throttles.** The genuinely new rigs are
   two DJI airframes (F450, F330) at two throttles each, the AGH single rotor
   and the AVQ quadrotor at two throttle settings.
2. **Rotor multiplicity is mostly unresolved**: 51 of 78 points give one rate
   for four rotors. The per-rotor spread, where it resolves, is 0.37-4.82
   rev/s. The points constrain the MEAN speed; none of them carries a
   time-resolved speed track.
3. **Four-rotor static rigs are not readable by a single-comb harmonic sum at
   all**, and this pass proves it is not the rival-exclusion rule: neither the
   ±6 % neighbourhood nor an `m, n <= 8` family depth changes a single verdict,
   while the comb score itself falls 15.3-17.8 dB → 6.4 dB from one motor to
   four on the same rig and throttle. A reading for those 18 SPCUP windows
   needs a multi-comb model (one comb per rotor, fitted jointly), not a looser
   gate.
4. **The odd-harmonic octave test remains the single point of failure.** It
   carries 1 SPCUP take, 5 DroneAudioSet recordings and 2 AVQ windows into a
   factor-2 error. The cell median catches the DroneAudioSet ones (9-22
   recordings per cell) and the per-window repeats catch the AVQ ones; nothing
   catches the SPCUP take, whose cell holds 8 different speeds.
5. **No new corpus adds telemetry.** Every extra point is an estimate with a
   0.25 rev/s tolerance (median), against DREGON's and Michael's measured
   tracks.
