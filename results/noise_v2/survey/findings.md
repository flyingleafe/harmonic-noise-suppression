# Corpus survey for noise model v2: which drone-noise recordings become fit points

Second pass. It corrects three things in the first pass: AVQ was rejected as
"free flight" without reading its specification table, the rival-margin rule
counted a neighbouring ROTOR of the same rig as a rival comb, and — the change
with the largest effect — **the 3 dB harmonic-sum margin was the wrong
usability gate for a multi-rotor recording**. A multi-rotor rig is now judged
on LINE EVIDENCE.

Scripts: `scripts/noise_v2_bench_speed.py` (per-recording reading, figures),
`scripts/noise_v2_corpus_survey.py` (corpus verdicts, tables, figures,
ingestion manifest). Readings: `bench_speeds_dload.json` (job
`nv2-gate-dload-3319dc`, uni-cpu — DREGON bench, SPCUP19, KAIST, AVQ: 68
rows), `bench_speeds_remote.json` (job `nv2-gate-daset-b3efb4`, uni-cpu —
DroneAudioSet: 168 rows), `bench_speeds_chums16.json` (ChuMS at
`--multi-window-s 16`, local: 9 rows), `bench_speeds_local.json` (the two
corpora whose raw tree exists only in `data/`, local: 31 rows). Tables:
`survey.json`, `survey_table.md`. Figures: `bench_speeds_dregon_law.png`,
`bench_speeds_gates.png`, `survey_speeds.png`,
`survey_multirotor_split.png`. Dataset: `noise-v2-bench-points`, manifest
`src/data_processing/noise_v2_bench_points.json`.

Reproduce (no audio is read for the figures):

```bash
PYTHONPATH=src python scripts/noise_v2_bench_speed.py --figures-from \
    results/noise_v2/survey/bench_speeds_{dload,remote,chums16,local}.json
PYTHONPATH=src python scripts/noise_v2_corpus_survey.py --speeds \
    results/noise_v2/survey/bench_speeds_{dload,remote,chums16,local}.json \
    --manifest-out src/data_processing/noise_v2_bench_points.json
```

## The estimator

1. **Stationary window.** The motor-ON part is the set of 1 s blocks inside
   12 dB of the loudest block; the window is the longest run of those blocks
   inside 4 dB of the active median, up to 16 s on a single-rotor rig and up to
   30 s in multi-rotor mode. This step is load-bearing: a DREGON bench file is
   25-45 s long but holds 11-35 s of motor, and its silent tail carries a fixed
   88 rev/s room tone that the harmonic sum locks onto (`motor_Motor3_50`:
   49.1 rev/s inside the run, 88.3 rev/s outside it).
2. **Welch spectrum**, median-averaged per channel, then averaged over
   channels. Single-rotor mode keeps the 0.5 Hz resolution target
   (df = 0.3365 Hz at 44.1 kHz); multi-rotor mode uses a fixed 4 s periodic
   Hann segment, so df = 0.25 Hz on every corpus.
3. **Harmonic sum** over 15-150 rev/s: the mean line prominence over the
   broadband floor at `k*f`, `k = 1..40` below 3 kHz, clipped at 30 dB. On a
   multi-rotor rig the argmax is the MEAN comb rate `f̄`.
4. **Octave mapping**: halve while the ODD harmonics of the half clear 6 dB on
   half of the tested orders (`k = 1,3,5,7,9,11`). 6 of the 20 DREGON
   single-motor readings are halved this way.
5. **Rival margin** (reported for every row; gated on for single-rotor rigs
   only, see below): the accepted comb-family score
   `max(score(f), score(2f))` minus the best RIVAL, where a rival is neither a
   small-rational `m/n` relative of the accepted rate (`m, n <= 4`, 11 ratios
   — the same comb read every n-th tooth) nor, in multi-rotor mode, inside
   **±6 %** of it (a rotor 1-3 rev/s from `f̄` is 1.4-4.3 % away at 70 rev/s,
   so the first pass scored one rotor of the rig against another). Both are
   stored: `margin_db` (neighbourhood excluded) and `margin_family_only_db`.
6. **Per-rotor split** (multi-rotor mode). The split runs on the LINE comb —
   the comb with a tooth at every order, which on a two-bladed rotor is the
   blade-pass line, not the shaft rate. Without that choice the test is
   meaningless: on `drone2_low_50cm_M_down_File3` the dominant rotor appears at
   even shaft orders only, but at blade-pass orders 4-14 without a gap. Orders
   start at `k_min = ceil(2 * 4 * df * scale / 0.3)` — two periodic-Hann main
   lobes (4 df = 1 Hz at T = 4 s) for a 0.3 rev/s rotor-to-rotor difference, so
   `k_min` = 7 on an `as_found` comb and 4 blade-pass orders (= shaft order 8)
   on a halved one. Inside `k*f̄ ± 6 %` up to 4 peaks 6 dB above the band's
   LOCAL MEDIAN become rate candidates `peak/k`; a rate is a resolved rotor
   when it repeats within 0.3 rev/s at **≥ 3 orders, consecutive or not** (a
   rotor line buried under a neighbour at one order is still that rotor at the
   next).

## The usability gate

**Single-rotor rig (unchanged).** Usable iff the margin of item 5 is ≥ 3 dB,
the two window halves agree on the rate to ≤ 1 rev/s, and the window is ≥ 8 s.

**Multi-rotor rig (new — line evidence).** Usable iff

* `f̄` is stable to ≤ 1 rev/s between the two disjoint window halves, AND
* a qualifying peak (6 dB over the band's local median inside `k*f̄ ± 6 %`)
  exists at **≥ 3 orders** of the line comb, consecutive or not, AND
* the window is ≥ 8 s.

The margin is still computed and stored, but it is **not** used, because it
measures rotor count rather than readability. The measurement that forced this
change, on one rig at one throttle:

| recording | rotors | comb score | margin |
|---|---|---|---|
| `motor_Motor1_70` | 1 | 15.83 dB | 9.75 dB |
| `motor_Motor2_70` | 1 | 17.82 dB | 5.56 dB |
| `motor_Motor3_70` | 1 | 15.34 dB | 8.20 dB |
| `motor_Motor4_70` | 1 | 16.72 dB | 7.27 dB |
| `motor_allMotors_70` | 4 | **6.43 dB** | **0.77 dB** |

With four rotors the tonal energy per rotor and its prominence over a
four-times-higher broadband floor both fall, so no single comb is
distinguished. No rival-exclusion rule can repair that: the ±6 % neighbourhood
exclusion moved the limiting rival on 44 of 240 multi-rotor readings (median
gain 0.556 dB, max 2.33 dB) and flipped one reading over the old 3 dB bar, and
a `--family-max 8` counterfactual (43 `m/n` ratios instead of 11, job
`nv2-fam8-8cc561`) flipped **no** verdict at all while leaving the SPCUP static
median margin at +0.04 dB. The line-evidence gate asks the question that can be
answered instead: are the rotor lines actually there, at enough orders, above
the local floor?

**Octave evidence no longer refuses a reading.** The odd shaft orders of the
accepted rate are counted (`odd_order_evidence_n`); below 2 orders over 6 dB
the row is admitted and flagged `octave_unresolved` (37 of the 148 usable
readings). The octave MAPPING itself is unchanged, and deliberately so:
forcing `f̄ = line comb / 2` on an `as_found` comb would halve
`motor_allMotors_70` to 34.3 rev/s against its validated 68.6 rev/s from the
throttle law. The factor-2 risk is instead caught downstream by the cell-median
check (below).

**Speeds are published per rotor.** `speed_rev_s` now holds one entry per
rotor: the resolved rates first, then `f̄` for every rotor the split could not
separate, with `resolved_mask` marking which is which
(`meta.resolved_mask` / `label.n_rotors_resolved` /
`label.multiplicity_unresolved` in the dataset).

Tolerance: `max(half-to-half difference, 0.5*df, 0.25) rev/s`, median
0.25 rev/s over the 135 fit points.

## Validation

**DREGON single-motor bench against the validated throttle law**
(`rate = 0.975*throttle + 0.37 rev/s`, `bench_speeds_dregon_law.png`): 20 of
20 usable, mean |error| **0.5552 rev/s**, max 1.42, RMS 0.703, refit slope
0.97577 / intercept 0.2395 / R² 0.997444 — **identical to the first pass**.
Single-rotor mode is untouched by every change above, which is what makes the
comparison a regression test rather than a coincidence.

**`motor_allMotors_70`, the four-rotor control on the same rig and throttle.**
Now **usable**, and it resolves **four rotors at 64.65 / 67.66 / 68.74 /
69.57 rev/s** (mean 67.66, spread 4.92 rev/s) against the single-motor law
value 68.6 rev/s at 70 % throttle. Line evidence: peaks at **27 orders** of the
line comb; half-to-half 0.8 rev/s; margin 0.771 dB (reported, not used);
`octave_unresolved` false. The first pass reported 65.23 / 67.61 / 68.76 rev/s
from one order (16) and rejected the recording.

**SPCUP19 AGH single rotors against the accepted blind readings**: 7 of 8
usable, unchanged. Six agree to ≤ 0.27 rev/s; take 1 reads 132.30 against
66.33 rev/s (a factor 1.995, the known octave failure) and take 4 is refused by
the single-rotor margin rule at 1.62 dB.

**KAIST rotating machine (out-of-domain control, stated 3010 RPM =
50.167 rev/s)**: 0 of 5 usable, unchanged. Note what this control now does and
does not test: KAIST is a SINGLE-rotor corpus, so it exercises the unchanged
3 dB margin gate, not the new line-evidence gate. The falsification checks for
the new gate are the DREGON throttle law (which the four-rotor control now
reproduces to 0.94 rev/s on its mean) and the cell-median consistency check.

**DroneAudioSet against the paper's stated lines** (arXiv:2510.15383 Fig. 7).
Cell medians over the fit points and the blade-pass line they imply:

| cell | n | median rev/s | 2f (Hz) | paper (Hz) | error |
|---|---|---|---|---|---|
| drone1 low | 17 | 84.96 | 169.9 | 168 (D_large low) | +1.1 % |
| drone1 high | 25 | 118.68 | 237.4 | 235 (D_large high) | +1.0 % |
| drone2 high | 28 | 126.03 | 252.1 | 259 (D_small high) | −2.7 % |
| drone2 low | 28 | 105.83 | 211.7 | 156 (D_small low) | **+35.7 %** |

Three of four cells still match to ≤ 2.7 % on 17-28 readings each (the first
pass had 1-22 per cell), which keeps the identification of the anonymous path
tokens (`drone1` = D_large DJI F450, `drone2` = D_small DJI F330) and the
reading of the paper's marked line as the blade-pass line at twice the shaft
rate. `drone2 high` loses 2.6 percentage points against the first pass because
the looser gate admits more factor-2 failures into the cell; `drone2 low`
remains the one real disagreement with the figure.

## AVQ: the corrected verdict

The AVQ specification table
(`webspace.eecs.qmul.ac.uk/lin.wang/demo/avq.html`, column `Type` = EO and
column `Drone` = "constant") marks FOUR of the twelve sequences as ego-noise
only at a CONSTANT throttle setting: S1 seq1 50 % (120 s), S1 seq2 100 %
(120 s), S1 seq3 150 % (40 s), S2 seq1 100 % (210 s). One throttle per
recording and no manoeuvre is the bench-class definition used here, so the
first pass's "free flight, not a bench/static candidate by type" was wrong
about these four. S2 seq2 is ego-noise only at DYNAMIC throttle; S2 seq5/seq6
are constant 100 % but speech+ego-noise MIXTURES; the other five are speech
with the drone muted or dynamic. Those eight stay excluded by type. The blind
campaign's octave warning for AVQ (median `fvk_ratio_double` 1.044) was
measured on the flight sequences.

Read from `dload:AVQ` (8 ch, 44.1 kHz) in consecutive 30 s windows, one
reading per window:

| sequence | throttle | windows | usable | f̄ per window (rev/s) |
|---|---|---|---|---|
| S1_seq1 | 50 % | 4 | 3 | 78.00, **78.00**, **39.46**, **77.79** |
| S1_seq2 | 100 % | 4 | 2 | **97.78**, 16.33, 97.34, **98.24** |
| S1_seq3 | 150 % | 1 | 0 | 111.38 (margin 3.33 dB but half-to-half 1.5 rev/s) |
| S2_seq1 | 100 % | 7 | 4 | 88.98, **45.48**, 45.25, **93.48**, **45.03**, **92.52**, 93.12 |

The three S1 settings order monotonically — 78.0 / 97.8 / 111.4 rev/s for
50 / 100 / 150 % — which corroborates that the table's throttle column is a
real setpoint. Three of the nine usable windows are factor-2 failures
(39.46 rev/s in the 50 % cell, 45.48 and 45.03 in the 100 % cell); the
cell-median check removes all three, so AVQ contributes **6 fit points over 2
cells** (S1 at 50 %: 2 points; S1 100 % + S2 100 %: 4 points).

## Verdicts

Full table with locations, licences and per-recording reasons:
`survey_table.md`. Usable readings, first pass → this pass:

| corpus | read | usable (pass 1 → 2) | verdict |
|---|---|---|---|
| DREGON bench (single + all-motors) | 21 | 20 → **21** | USABLE |
| SPCUP19 AGH single rotors | 8 | 7 → 7 | USABLE |
| DroneAudioSet drone-only | 168 | 50 → **98** | PARTLY USABLE |
| SPCUP19 static / hover (4 rigs) | 18 | 0 → **7** | PARTLY USABLE |
| SPCUP19 ChuMS propeller rig (16 s window) | 9 | 1 → **2** | PARTLY USABLE |
| AVQ constant-throttle ego-noise | 16 windows | — → **9** | USABLE |
| KAIST-rotating-acoustic | 5 | 0 → 0 | CONTROL ONLY |
| drone_audio (`yes_drone`, 24 sampled) | 24 | 0 → 0 | REJECTED (1.02 s clips) |
| zenodo_drone_noises | 7 | 1 → 4 | REJECTED (no rig identity) |
| DronePrint / MAVD / DroneNoise DB / ESC-50 | — | — | REJECTED (access/type) |

**Totals: 276 readings, 148 usable (79), 135 fit points (73), 18 fit-point
cells (11)**; the survey's own rig/condition count, which also counts the
rig-less `zenodo` readings, is 22 (12).

Fit-point cells, points each:

- `dregon_mikrokopter_single_motor` at 50/60/70/80/90 % — 4 each, 20 total.
- `dregon_mikrokopter_quad` at 70 % — 1 (the four-rotor control, 4 rotors
  resolved).
- `spcup_AGH` single rotor — 7; `spcup_AGH` stationary (4 rotors) — 5.
- `spcup_Diagonal_Unloading` stationary — 1; `spcup_Idea_ssu` stationary — 1.
- `spcup_ChuMS_2prop_bench` — 1; `spcup_ChuMS_3prop_bench` — 1.
- `daset_drone1` low/high — 16 + 23; `daset_drone2` low/high — 27 + 26.
- `avq_quadrotor` at 50 % — 2; at 100 % — 4.

The SPCUP19 static verdict changes from REJECTED to PARTLY USABLE: 7 of 18
windows now clear the line-evidence gate (peaks at 3-20 orders), all of them
flagged `octave_unresolved`, with `f̄` 50.4-142.5 rev/s. Four AGH static
windows agree on ~50.4 rev/s and resolve 2-4 rotors (spread up to 5.5 rev/s);
`AGH__static_clean__1` (142.5 rev/s, lines at only 3 orders) and
`AGH__static_corrupted__1` (111.3) do not agree with them, so the AGH
stationary cell is internally inconsistent by a factor near 2-3 and should be
treated as the weakest cell in the set.

**ChuMS is read with a 16 s window** (`--multi-window-s 16`): the rig runs at
uncontrolled propeller speeds and drifts, so the 30 s multi-rotor window smears
its lines — `3prop_repeat2` gives 1.88 dB of margin and no line cluster at 30 s
against 3.93 dB and three resolved propellers (77.08 / 77.52 / 78.07 rev/s) at
16 s. `2prop_repeat3` resolves two (82.63 / 83.03 rev/s).

**Rejections** (128 readings): 70 DroneAudioSet (lines at fewer than 3 orders —
mostly the `M_up` mic group above the airframe, whose readings also lock below
the 15 rev/s floor), 24 drone_audio clips (1.02 s window against the 8 s rule),
11 SPCUP static, 7 ChuMS, 7 AVQ windows, 5 KAIST (the control), 3 zenodo, 1
SPCUP AGH take.

**Cell-consistency drop.** Nine usable readings sit at a factor 0.48-0.52 of
their own (rig, condition) cell median — a factor-2 failure on that recording,
not a different speed: 6 DroneAudioSet and 3 AVQ windows
(`survey.json:cell_octave_outliers`). They are excluded from the fit points, so
148 usable readings become 135 fit points. This check is now the main defence
against the relaxed octave rule, and it only works where a cell holds ≥ 3
readings — the SPCUP stationary and ChuMS cells do not.

## Per-rig rotor multiplicity

`n_res` is the number of rotor rates the split resolved
(`survey.json:corpora[].stats.n_resolved_by_rig`, usable readings only, figure
`survey_multirotor_split.png`). Over the 135 fit points: 76 resolve one rate,
42 two, 10 three and **7 four**; 99 carry `multiplicity_unresolved = true`
(rotors coincident within 0.3 rev/s, so `speed_rev_s` repeats `f̄` for them).
The first pass resolved at most three and never four.

## Limits worth carrying into the model work

1. **18 fit-point cells, 5 of them the same DREGON motor at 5 throttles.** The
   genuinely distinct rigs are the MikroKopter (single motor and four-motor),
   two DJI airframes (F450, F330) at two throttles each, the AGH quadrotor
   (single rotor and stationary), two more SPCUP team rigs, the Dotterel
   propeller rig at 2 and 3 propellers, and the AVQ quadrotor at two settings.
2. **The multi-rotor gate buys quantity at the cost of octave certainty.** 37
   of 148 usable readings carry `octave_unresolved`, and 9 factor-2 failures
   had to be removed by the cell median. Any use of these labels should read
   `label.multiplicity_unresolved` and `meta.octave_unresolved` and treat the
   single-cell rigs (SPCUP stationary, ChuMS) as the weakest.
3. **Per-rotor spread, where it resolves**: 0.37-5.5 rev/s (largest on the AGH
   stationary windows and `motor_allMotors_70`, 4.92 rev/s). The points
   constrain the mean and the spread; none carries a time-resolved speed track.
4. **No new corpus adds telemetry.** Every extra point is an estimate with a
   0.25 rev/s median tolerance, against DREGON's and Michael's measured tracks.
