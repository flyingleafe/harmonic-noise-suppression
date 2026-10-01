# Static rig profiles — rotor speeds and harmonic profiles of stationary recordings

**Status:** in progress (2026-10-01 –)

## Motivation

The universal generative rotor-noise prior needs to know which rig
configurations it must cover. Stationary recordings (drone fixed, rotors at
constant speed, no other source) are the cleanest evidence: one long spectrum
per recording resolves every rotor's shaft rate, and with the rates fixed the
line power of every harmonic, per rotor and per microphone, is a direct
measurement of the rig's profile. Where the rig geometry is known, the
per-microphone levels test whether a rotor behaves as a point source with
spherical spreading.

**Hypothesis.** On a stationary recording the full-record spectrum separates
the rotors (higher orders resolve rotors the low orders merge), so per-rotor
shaft rates and per-harmonic, per-microphone line powers can be measured on
every rig we hold; they define the range of configurations the universal
prior must cover.

**MVP.** Speeds + per-order line powers on the whole census, one `uni-cpu`
job; profiles per rig; the point-source test on the rigs with geometry.

**Kill gate (before any profile is trusted).** Single-rotor speeds match the
DREGON throttle law (`0.975·throttle + 0.37` rev/s) within 1 rev/s on 20/20
runs and the AGH single-rotor readings; the octave decision is right on all of
them; the two halves of each record agree. Failing it stops the campaign at
the speed stage.

**Budget.** Two working days.

## Census (drone-noise-only, stationary)

| Rig | Dataset | Recordings | Rotors on | Channels |
|---|---|---|---|---|
| DREGON MikroKopter quad | `DREGON-frames` | `motor_Motor{1-4}_{50..90}` (20) + `motor_allMotors_70` | 1 / 4 | 8 |
| AGH quad (model not stated) | `SPCUP19-frames` | `ego-noise__mic_array__0..4` (5) | 4 | 8 |
| AGH quad, reference mic | `SPCUP19-frames` | `ego-noise__single_rotors__0..7` (8) | 1 | 1 |
| MikroKopter MK EASY Quadro V3 (KU Leuven) | `SPCUP19-frames` | 2 takes | 4 | 8 (ch6 dead) |
| YH-19HW toy quad (Maverick) | `SPCUP19-frames` | `Maverick__4` | 4 | 3 |
| AVQ quadrotor | `AVQ` | `S1_seq1/2/3`, `S2_seq1` | 4 | 8 |
| DJI F450 / DJI F330 | DroneAudioSet `drone-only` (HF parquet) | 168 files (up/down arrays, centre mic) | 4 | 8 / 1 |

Excluded: anything with another source (AGH `static_corrupted`, AVQ mixtures,
DroneAudioSet `drone-with-source`), varying-speed takes (AVQ `S2_seq2`), and
airborne recordings.

## Method

Code: `src/experiments/static_rig/spectra.py`, driver `scripts/static_rig.py`
(`fetch` → node-local npy, `run` → one gridrun JSON per recording), run on
`uni-cpu`.

1. Motor-on span (0.5 s blocks within 6 dB of the 90th-percentile level,
   longest run, 1 s trimmed each end).
2. Welch spectrum per channel, 16 s Hann segments (`df` ≈ 0.0625 Hz; halved
   segments on short records), running-median floor (20 Hz); channels combined
   as the mean line-over-floor prominence (dB).
3. Comb rate: best mean prominence over orders 1–12.
4. Speed spectrum `H(s)` = mean over all orders `j` of the prominence at
   `j·s` (clipped 0–30 dB), on a 0.0005 rev/s grid within ±8 % of the comb
   rate. Shaft wander widens order `j` by `j` in Hz, i.e. a constant width in
   rev/s, while the resolution improves as `df/j`; summing orders sharpens and
   stabilises each rotor's peak. Peaks over `median + max(1 dB, 25 % of the
   tallest)`, shoulders within 2 half-widths of a taller peak dropped, at most
   `n_rotors` kept; each refined by a one-to-one assignment of spectral peaks to
   (rotor, order) and a weighted fit `p ≈ j·s`.
5. Octave decision on the refined speeds: `s/2` (or `s/3`) is the shaft rate
   when the orders it adds carry lines ≥ 1 dB over control positions half an
   order away (median over rotors).
6. Uncertainty: the same read on each half of the record.
7. Line power per rotor, order and channel: PSD summed over ±4 bins around
   `j·s`, floor removed; lines of two rotors within 8 bins flagged merged.

### Synthetic check (local, 25 s, 16 kHz, 8 ch, OU shaft wander)

| Case (true rev/s) | Wander (Hz std) | Read |
|---|---|---|
| 64.65 / 67.66 / 68.74 / 69.57 | 0.07 | 64.646 / 67.655 / 68.717 / 69.537 |
| same | 0.28 | 64.501 / 67.599 / 68.667 / 69.612 |
| 80.0 / 80.3 / 80.6 / 80.9 | 0.07 | 79.983 / 80.298 / 80.572 / 80.892 |
| 80.0 / 80.1 / 80.2 / 80.3 | 0.07 | one peak, 80.14 (unresolved) |
| 90.0 / 90.0 / 91.0 / 92.5 | 0.07 | 89.998 / 90.977 / 92.463 (3 distinct) |
| single 2-blade 49.1 / 3-blade 97.6 | 0 | 49.100 / 97.600 |

### Local smoke on DREGON

- `motor_Motor1_50`: 49.039 rev/s (halves 48.985 / 49.045); throttle law
  `0.975·50 + 0.37` = 49.12.
- `motor_allMotors_70`: comb at the blade-pass rate 131.43, octave check
  +1.29 dB → shaft; rotors 67.664 / 68.747 / 69.564 (the noise-v2 survey read
  67.66 / 68.74 / 69.57 + 64.65). The fourth rotor is not resolved: in the
  full-record spectrum only 2 of the first 40 orders of 64.65 rev/s carry a
  line ≥ 6 dB (median 1.8 dB, against 11–13 orders and 3.7–4.0 dB for the other
  three), and its speed-spectrum peak is 0.4 dB over the median whichever way
  the channels are pooled (mean dB, pooled linear, max, top-3). The survey's
  64.65 was itself a weak read: margin 0.771 dB, halves 0.8 rev/s apart
  (`noise_v2_bench_points.json`). Whether two rotors share one of the three
  speeds is a question for the line powers (+3 dB for two coincident rotors).

## Results (round 1, `uni-cpu` job `static-rig-speeds-a5a5e5`, commit `8c23b9e5`)

209 recordings read (outputs in `results/static_rig/speeds/`, not committed:
147 MB of unit JSON + 23 MB of spectra).

**Kill gate as stated: FAILED** — 16 of 20 DREGON single-motor readings are
within 1 rev/s of the common law, not 20/20 (Motor2_80 −1.07, Motor2_90 −1.19,
Motor4_80 +1.24, Motor4_90 +1.28). The follow-up below shows the misses are
systematic per-motor offsets, not read errors; the gate is recorded as failed
regardless.

**Constant-speed recordings: the full-record read is precise.**

- DREGON single motors (20): the two halves of each record agree to
  0.116 rev/s at most (median 0.022). Against the common throttle law the
  error is mean 0.562, max 1.277 rev/s — but that law averages four motors that
  differ: per motor, a straight line in throttle fits every reading to
  ≤ 0.123 rev/s (Motor1 0.9727·t + 0.398, max residual 0.023; Motor2 0.9609·t
  + 0.439, 0.098; Motor3 0.9812·t + 0.136, 0.088; Motor4 0.9898·t + 0.337,
  0.123). Motor2 runs 0.7–1.2 rev/s below the law, Motor4 0.7–1.3 above.
- DREGON `allMotors_70`: 67.677 / 68.747 / 69.564 rev/s (halves agree to
  0.07); the fourth rotor is not resolved (see the smoke note above).
- AGH single rotors (8): 159.688, 132.626, 113.789, 97.406, 77.722 (rotor 4 at
  settings 275…125), 97.793 / 96.620 / 96.809 (rotors 3/2/1 at 160); halves
  agree to ≤ 0.06. Matches `AGH.yaml`'s 159.5 / 132.6 / 113.7 / 97.6 / 77.7.

**Multi-rotor rigs other than DREGON: the premise of constant speed fails.**

- AVQ `S1_seq2` ("constant 100 %"): four rotor tracks are clearly separate in a
  4 s STFT, but they DRIFT over the 120 s — one near 97–98 Hz, three between 79
  and 89 Hz that converge and cross (e.g. 89 → 83 Hz)
  (`results/static_rig/avq_S1_seq2_tracks.png`). In the full-record spectrum
  each rotor is a plateau several Hz wide whose width grows with the order
  (79–88 Hz at order 1, 583–591 Hz at order 6;
  `avq_S1_seq2_fullrecord_zoom.png`). 3 of the 4 AVQ takes gave no rotor.
- KU Leuven Team 1: the rotors fall from ~120 to ~106 Hz over the 16 s
  (`results/static_rig/spcup_tracks.png`).
- Maverick 4: weak, gappy lines (toy quad).
- DroneAudioSet: comb rates scatter between 15 and 250 rev/s within a design
  cell and most files give no rotor; not inspected further (its audio stays
  off the laptop).

So "stationary" (drone fixed, throttle constant) does not mean constant shaft
speed on these rigs; a full-record spectrum cannot be read there, and the
per-rotor speeds must be tracked in time.

**Point-source level test, DREGON single motors (from the round-1 line powers).**
385 harmonic lines (orders ≤ 40, ≥ 6 dB over the floor on all 8 mics; 3080
observations). Model per line: level = line constant + mic gain − α·20·log10 r
(r = rotor-hub-to-mic distance; the 1/r effect spans 4.65–4.85 dB across the
8 mics for each rotor). Across-mic RMS residual: 4.09 dB with neither term,
3.61 dB with mic gains only, 3.67 dB with distance only (α = 1.30), 3.46 dB
with both (α = 0.87; mic gains −2.09 … +2.62 dB). By order: 5.58 dB (1–4),
2.74 (5–10), 2.92 (11–20), 3.11 (21–40). Spreading is the right size (α near
1) but explains little: the per-mic level is dominated by something the
point-source model lacks (directivity, near-field interference, scattering by
the frame), worst at the lowest orders. Magnitude only; the phase test needs
complex amplitudes per mic.

## Round 2 — windowed tracking (2026-10-01)

Drifting rigs are tracked, not read from one spectrum
(`experiments/static_rig/tracks.py`): rotors seeded by the in-tree
`tracking.comb_seed.seed_from_gram` (1 s windows, hop 0.5 s, slew 3 rev/s/s,
40 orders, 4 restarts) on the most line-like channel, refined on all channels
by `tracking.vk_tracking.vk_track` (40 orders, 1 Hz envelopes); harmonics read
off the final VK envelopes — line power = mean `|x|²/2` minus floor density ×
envelope bandwidth, per order and mic, plus SNR and inter-mic phase.
DroneAudioSet takes are grouped (up ring, down ring, centre mic of one run).
Report: `scripts/static_rig_report.py` → `results/static_rig/report/`.

Track gate: recording VK residual ratio < 0.8; the track distinct, ≥ 5 orders
at median-mic SNR ≥ 6 dB, speed std < 3 rev/s, < 20 % of frames within
0.5 rev/s of the search-band edges.

- **Control (constant rigs):** 28 single-rotor recordings (DREGON 20, AGH 8):
  tracked vs full-record speed median 0.018, max 0.251 rev/s.
- **DREGON `allMotors_70`:** the tracker fails (residual ratio 2.57; four
  rotors within 2 rev/s); its speeds stay the full-record ones (3 of 4).
- **AVQ:** tracks follow the drifting rotors (overlay checked on `S1_seq2`;
  one identity swap at ~103 s after a crossing). Valid tracks: `S1_seq1` 1/4,
  `S1_seq2` 3/4, `S1_seq3` 4/4, `S2_seq1` 4/4 (rerun at full length, job
  `static-rig-combined-3a33f0`: means 87.1 / 93.0 / 90.0 / 68.1 rev/s,
  residual ratio 0.16).
- **KU Leuven:** Team 1 3/4, Team 2 1/4 valid; three tracks crowd one ridge
  (rotors falling ~117 → 105 Hz).
- **Maverick 4:** 0/4 (residual ratio 0.976).
- **AGH arrays:** 0 usable rotors in either band (70–260 or the 3-blade shaft
  band 40–100; residual ratios 0.88–0.92, tracks parked at band edges). Their
  lines only start at 150–820 Hz depending on the take, so the low orders that
  fix the shaft rate are missing; a shaft-rate harmonic sieve (40–170 rev/s,
  24 orders) on the round-1 spectra finds no dominant comb either (best
  scores within ~1 dB of the runners-up). Unresolved.

**Point-source test (tracked envelopes, valid tracks).** Model per line:
level = line constant + mic gain − α·20·log10 r; mic cells at SNR ≥ 6 dB, a
line needs ≥ 60 % of the mics. Cross-fitted by order parity: on the even
orders each recording's track → rotor-position permutation (best pure 1/r)
and the gains and α are fitted, the odd orders are scored under them, and
back; a track with no training line is dropped. Phase of the scored lines:
measured phase relative to the tracker's reference mic (loudest raw cell,
itself at SNR ≥ 6 dB) vs −2πf(r_c − r_ref)/c at coherence ≥ 0.8, against a
mic-permuted null.

| Rig | Scored lines | 1/r spread (dB) | Held-out RMS: none / gains / 1/r / both (dB) | α (two folds) | Phase error, median rad (null) |
|---|---|---|---|---|---|
| DREGON | 683 | 4.68 | 3.80 / 3.27 / 3.41 / 3.22 | 1.03, 0.33 | 0.59 (1.14) |
| AVQ | 173 | 4.53 | 3.62 / 3.27 / 3.34 / 3.03 | 1.67, 0.66 | 0.52 (1.07) |
| KU Leuven | 36 | 3.45 | 1.90 / 2.01 / 2.17 / 2.24 | 1.00, 1.02 | 0.70 (0.57) |

Spherical spreading accounts for only 0.3–0.4 dB of the 3.6–3.8 dB
across-mic scatter on DREGON and AVQ, and α is unstable between folds there; the phase
follows the point-source delay pattern far better than chance (about half the
null error) with ~0.5–0.6 rad median error. KU Leuven fails out of sample:
with the permutation chosen on the other parity its phase error (0.70 rad) is
worse than the null (0.57); a first pass that chose the permutation on all
orders showed 0.47 rad, an in-sample artefact. The point source is a usable
first-order delay model on DREGON and AVQ and a poor level model everywhere:
per-mic level is dominated by something else (directivity / near field /
scattering) `[INFERENCE]`.

**Combined profiles (rotors not all resolved).** Per recording, the total
comb of all rotors is read from the full-record spectrum
(`spectra.comb_band_powers`, `static_rig.py combined`): order k integrates
[k·s_lo − w, k·s_hi + w] minus the median density of 10 Hz flanks; orders stop
where the band would reach the next order's. Band: the candidate tracks'
5th–95th percentiles when ≥ 1 track is valid, else the round-1 speed modes
(±0.5 %), else the round-1 comb rate ± 8 %; AGH arrays only from `AGH.yaml`'s
take-4 blade-pass lines / 3; DroneAudioSet only from tracks. Per-rotor =
total − 10·log10 4 (symmetric). Median-mic level by order, dB re the loudest
measured order (cells at SNR ≥ 6 dB; `–` = no such cell):

| Recording | Verdict | Band (rev/s), source | Orders read / measured | Level by order 1, 2, 3, … |
|---|---|---|---|---|
| DREGON `allMotors_70` | unresolved 0/4 | 67.3–69.9, round-1 modes | 18 / 9 | 0 −1 −15 −12 −17 −15 −21 −17 −15 −16 −15 −15 −8 −15 −21 −17 – −19 |
| AVQ `S1_seq1` | partial 1/4 | 77.3–81.6, tracks | 13 / 8 | −19 0 −21 −13 – −16 – −15 – −15 – −17 – |
| AVQ `S1_seq2` | partial 3/4 | 79.4–98.5, tracks | 3 / 3 | −19 0 −21 |
| KU Leuven Team 1 | partial 3/4 | 104.3–117.4, tracks | 6 / 5 | −9 0 −13 −2 −13 −11 |
| KU Leuven Team 2 | partial 1/4 | 100.0–115.0, tracks | 5 / 2 | −9 −2 −13 0 – |
| Maverick 4 | unresolved 0/4 | 33.6–39.4, round-1 comb | 2 / 1 | −11 0 |
| AGH array 4 (setting 20) | unresolved 0/4 | 69.9–72.5, AGH.yaml BPF/3 | 18 / 8 | – – −11 −11 −17 – −18 −6 −14 −19 – – −12 – 0 −7 – −3 |
| AGH arrays 0–3 | unresolved 0/4 | none identifiable | – | – |

`allMotors_70`'s band holds the three resolved rotors only (the fourth,
64.65 rev/s, shows a line at 2 of 40 orders in round 1), so its symmetric
per-rotor level assumes four equal rotors over a three-rotor total. Wide
drifting bands leave few orders (AVQ `S1_seq2`: 3). On AGH array 4 the
shaft-order profile peaks at multiples of 3 only partly (orders 15 and 18
loudest, 8 and 16 next), so its shaft assignment stays `[INFERENCE]`.

