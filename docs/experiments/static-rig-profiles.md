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

## Results

Pending: the `uni-cpu` run over the census.
