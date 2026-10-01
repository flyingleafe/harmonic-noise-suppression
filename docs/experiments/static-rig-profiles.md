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


## Round 3 — joint multichannel search, line physics, single-rotor reading (2026-10-01)

Motivation: the round-2 tracker (hand-picked bands, single-mic seeding, VK)
missed rotors a human reads off the spectrogram (AVQ `S1_seq1`: 54 and 59
rev/s). Replaced by a Bretthorst-style joint search, then — on the user's
direction — stripped back to the simplest reading on the single-rotor case.

**Joint search** (`experiments/static_rig/joint_speeds.py`,
`scripts/static_rig_joint.py` → `results/static_rig/joint/`): per channel
Student-t/g-prior evidence of a comb, product over channels, spike-and-slab
line presence; dyadic windows split while the gain beats an Occam cost
(~12 nats/rotor); joint R-rotor modes = combinations of 1-D peaks (BPF must
carry ≥ 0.3 of the strongest harmonic, speeds within 1.7×, 2nd-order
intermodulation tones in the model), linked by Viterbi with a Hungarian
permutation cost. AVQ S1 (8 ch, ~100 s per 120 s recording):

| seq | A | B | C | D | all four found |
|---|---|---|---|---|---|
| seq1 (50 %) | 51.7–54.3 (11/14 windows) | 55.0–55.9 (10/14) | 57.2→61.2 (14/14) | 77.6–79.1 (14/14) | 7/14 |
| seq2 (100 %) | 79.3–82.3 (16/16) | 81.8–84.9 (13/16) | 85.0–88.2 (12/16) | 97.1–98.4 (16/16) | 9/16 |
| seq3 (150 %) | 88.5–89.3 (5/7) | 93.0–93.4 (5/7) | 95.4–96.9 (5/7) | 109.9–112.0 (7/7) | 5/7 |

Validated on 20 DREGON singles against round 1 (median 0.016, max 0.043
rev/s) and on halves (median 0.016, max 0.034; old method 0.022/0.116);
Laplace sd ~1e-5 is 1000× overconfident. The published OT multi-pitch
baseline (arXiv 2508.02471) resolves all four AVQ rotors in 6 of 450 frames.

**AVQ profiles, full band.** The first reading (3 kHz cap, 10 Hz running-median
floor, window-constant speeds) missed what the spectrogram shows: D's order 42
is 11–16 dB above noise in all three sequences (3284/4108/4651 Hz — a motor
order, 6 × 7 pole pairs `[INFERENCE]`), and the 7 s window-mean speed is off by
up to 0.3 % inside the window (±7–10 Hz at k = 42). With 0.5 s block-refined
tracks, a 20th-percentile floor over ±60 Hz and a collision mask
(`scripts/static_rig_scratch/avq_refine.py`): D orders 2–24 (even) at 5–42 dB
and k = 42; A/B/C only to k ≈ 8 (50 %), 16 (100 %), 10 (150 %). The 3.28 and
~5 kHz "clusters" are D k42 and D k63–66 with companions 4–8 Hz apart (modulation
sidebands `[INFERENCE, untested]`), not other rotors' harmonics — with a line
every ~15 Hz above 2 kHz, nearest-harmonic assignment is meaningless.

**DREGON singles — what a stationary rotor is**
(`results/static_rig/single_rotor/dregon_singles.json`,
`scripts/static_rig_scratch/dregon_*.py`). Strict span: BPF line within 1 % of
its plateau. Windy mics (downwash): Motor1 ch 5, 6; Motor2 ch 0; Motor3 ch 1, 2;
Motor4 ch 4.

- *Mean speed*: core peaks of orders 2–7 agree to ±0.001–0.005 rev/s, mics to
  ≤ 0.003: two decimals solid, the third ±3. The k = 2 core is unresolved
  (HWHM 0.7–0.9 of 1/T): no slow wander above ~0.004 rev/s over 7–35 s.
- *Shaft jitter* as phase drift from s̄·t: 0.010 rev after 0.1 s on every
  recording (the measurement floor), **0.015–0.038 rev (5–14°) after 1 s**,
  0.02–0.075 after 3 s; growth ∝ √τ ⇒ a phase random walk, i.e. OU frequency
  noise with memory ≪ 0.1 s — only D = 2σ_ν²/λ is identifiable. Lines are
  Lorentzian, HWHM γ_k = π k² D, tails 1/Δf²; the skirts of neighbours make a
  floor from k ≈ 15–20 (M1_70). No throttle trend; Motor1 wanders most.
- *Amplitude jitter*: 2–4 dB window-to-window on every line, not explainable by
  speed (≤ 0.1 dB). It is inflow-turbulence loading modulation: a pedestal of
  fixed width in Hz at every order (flat in k, ~ −20 dB, HWHM ≈ 0.3–0.5 Hz on
  M1_70), while the FM pedestal grows ∝ k². One Lorentzian cannot carry both:
  a shape fit returns the AM width at every order (2–3e-3 "D"); the phase-drift
  D (2–14e-4) is noise-inflated; resolved cores give D ≈ 0.3–2.5e-4 from 1–13
  orders. Derivation and physics:
  `writing/reports/2026-10-01_harmonic-line-am-fm/` (two Lorentzians, widths
  add; read the air below k_× = √(γ_m/πD), the shaft above).
- *Profile reading* (`experiments/static_rig/single_rotor.py`): with s̄ and D
  given, a_k = mean of the periodogram over ±max(γ_k, 1/T) ÷ the same mean of
  the known line shape (the peak when unresolved, peak·πγ_kT/1.5 when
  resolved); no floor, no fit; orders 1–150. Reconstruction = one diffusing
  shaft + harmonics at √a_k, random phases. Peaks match within ~1 dB
  (M1_70 ch 0: k1/2/4/8/16/24/32/42 real −85.2/−51.0/−59.8/−68.2/−76.0/−77.8/
  −84.5/−70.5 vs synth −85.1/−50.5/−58.7/−68.4/−76.0/−79.1/−83.3/−73.5 dB).
  Over/under-estimation across recordings tracks the reading window, i.e.
  floor leaking into wide windows when D is too large — not a D ∝ s law
  (that hypothesis gives a worse spectral fit on 17/20). Explorer:
  `notebooks/single_rotor_explainer.ipynb` (`plots.spectrum_viewer` with the
  new `overlay=`: both spectra on one pane, spectrogram toggle, audio).

Open: read D from the cores above k_× and the AM parameters (σ_m², λ_m) from
the pedestals below it; coherence of the AM across orders; then back to the
multi-rotor rigs with the same reading.

**Whittle fit and direct amplitude jitter (2026-10-01, laptop, ~5 min for
all 20).** `single_rotor.whittle_fit`: two-Lorentzian line
`a_k[L_γk + σ_m² L_(γk+γm)] ⊛ K + F_k` on the full-span Hann periodogram
(F_k = local 20th percentile, a constant), amplitudes per line in closed
form, grid + local search over (D, γ_m, σ_m²) on the non-windy mics;
`results/static_rig/single_rotor/whittle/`. `single_rotor.amplitude_jitter`:
0.25 s frames, peak of ±1 bin at k·s̄ → ln A_k(t) per order and mic; variance
minus the noise part = σ²_m,k, integral correlation time → γ_m,k, cross-order
correlation; `.../amplitude_jitter.json`.

- D_Whittle (×10⁻⁴, 50→90 %): M1 9.2/8.4/10.9/9.2/11.9; M2 2.2/3.0/6.0/
  12.9/7.7; M3 7.1/3.9/10.9/10.0/10.0; M4 4.6/3.9/3.6/4.6/3.9. Sharply
  peaked, within 1.5× of D_phase on 15/20: D ≈ 1e-3 (Motor1, M2/M3 ≥ 70 %)
  and 3–5e-4 (Motor4, M2/M3 ≤ 60 %) stand; the "D_core ≈ 1e-4" reading from
  1–13 orders was the unreliable one.
- The spectral AM pair is not constrained: γ_m at the grid floor on 19/20,
  σ_m² scattered 0.02–0.56.
- Direct per-order: σ_m² = 0.005–0.08 at k ≤ 8 (σ_m ≈ 0.1–0.3), 0.1–0.17 at
  k = 9–30 (partly frame scalloping of the FM), 0.02–0.11 above; γ_m ≈
  0.2–0.6 Hz at k ≤ 8 (memory 0.3–0.8 s), ~1 Hz at mid orders, frame-rate-
  limited above k ≈ 30. Cross-order correlation of ln A_k(t) by separation
  (`am_corr_lag.py`, `am_corr_lag.json`): Δk = 1 ≈ 0, **Δk = 2 ≈ +0.1–0.3**,
  Δ4/Δ6 +0.05–0.2, ~0 beyond Δ10; odd separations ≈ 0 — the even (blade-pass)
  and odd (shaft-rate) families breathe separately, with a short correlation
  length of a few orders inside each family. Independent per-line AM is the
  zeroth order; an AR(1)-in-k (ρ ≈ 0.25 per same-parity step) the refinement.
- Cross-mic correlation of the same order's ln A_k(t): 0.3–0.8 (M1_70 0.79,
  M3_70 0.75, M2_80 0.56, M4_80 0.34 at k ≤ 8) — the AM is mostly a source
  property (the rest is the per-mic pattern of each line breathing, see the
  mic section below). The measured per-order (σ²_m,k, γ_m,k) enter the
  reconstruction as one log-amplitude OU envelope per order shared by the
  mics; the per-mic residual is dropped. Table-D profiles are stored in
  `results/static_rig/single_rotor/profile/` (the notebook reads, never
  recomputes); synthesis is a blocked float32 cosine-bank matmul (~1–2 s per
  recording).

**Width law → OU shaft (2026-10-01).** Real-vs-model spectra for all 20
(`line_widths.py`, `results/static_rig/single_rotor/figs/`) showed the
random-walk Whittle lines 2–4× too wide above k ≈ 30 and 30–63 harmonics per
recording zeroed by the amplitude solver. A free per-order Lorentzian HWHM
(`width_law.py`) grows as k^1.4–1.9: quadratic to k ≈ 20 (π k² D with the
Whittle D), then bending towards linear (γ/k ≈ 0.03–0.08 Hz/order) — the OU
shaft-speed signature (Lorentzian while the coherence time exceeds 1/λ,
Gaussian 1.18 k σ_ν beyond). Quick fit (`ou_quick.py`, `.../ou/`): σ_ν =
0.04–0.09 rev/s (0.05–0.15 % of s̄), λ ≈ 9–15 s⁻¹ (memory 0.06–0.13 s) on
16/20; M1_90, M2_80, M3_90, M4_90 fall to λ ≈ 0.5–2 (random-walk corner,
still too wide at k = 80). Width residual sd 0.5–1.0 in ln: even orders are
3–10× wider than odd in the free fit (odd lines are weak, partly a low-SNR
bias), and at k ≈ 20 a resolution-sharp spike survives on top of the hump —
the shaft's own coherence time there is > 1 s, so the hump is not accumulated
FM; model question left open. Notebook model 2 = OU shaft + peak amplitudes
with the OU shape + measured per-order AM; no zeroed orders, RMS matches.

### Microphone structure of the line levels (2026-10-01)

Question: can the per-(rotor, mic) harmonic profiles be factorised into a
rotor profile and a per-mic response, X_rm[k] = X_r[k]·H_m[k]? Data: OU
profiles of the 20 singles, non-windy mics, dB; scripts in
`scripts/static_rig_scratch/` (`mic_factor.py`, `mic_symmetry.py`,
`mic_eq.py`, `mic_eq_fine.py`, `line_coincidence.py`, `mic_variance.py`,
`pattern_stability.py`); figures `results/static_rig/single_rotor/figs/`.
Geometry (`sources.dregon.get_geometry`): rotors at z = +0.19 m over the cube
centre, mics on two layers z = ±0.041 m, rotor–mic distances 0.22–0.40 m,
rotor radius ≈ 0.12 m — every mic is in the near field above ~1.5 kHz.

1. **Rank-1-per-harmonic factorisation fails.** Mic-to-mic spread of a
   rotor's line levels is 3.6–4.0 dB rms. Rotor profile alone leaves 3.7 dB,
   × scalar mic gain 3.4 dB, × per-harmonic H_m[k] 3.1 dB (70 %; 60/80 %
   alike). H_m is small (1–1.7 dB sd over k) and explains ~0.6 dB; the
   residual is pair-specific (rotor, mic, k).
2. **The pair residual is not smooth in k**: autocorrelation along k flat at
   +0.2…0.3 from Δk = 1 to 8 (a smooth directivity would be ≈ 1 at Δk = 1);
   ~25 % broad component, ~75 % per-order scatter ≈ 2 dB; odd orders 2.6 dB,
   even 2.1 dB.
3. **The rig's symmetry does not reproduce it.** The only mic-to-mic symmetry
   is the 180° rotation about z (ch c ↔ c+4; pairs Motor1↔Motor3,
   Motor2↔Motor4; matches the windy sets {5,6}↔{1,2}, {0}↔{4}; x-offset 2.3
   cm). Matched-mic difference spread 2.8 dB (M1/M3) and 3.2 dB (M2/M4) vs
   3.1/3.9 same-index and 3.4/4.2 random pairings — matched is best but only
   at the ~10th percentile of random, and equal to one rotor's own mic-to-mic
   spread (2.7/3.2 dB): matching the symmetric mic removes nothing.
4. **A fixed per-pair transfer function exists and lives on the frequency
   axis.** R_rm(f) = P_rm(f) / mean_m' P_rm'(f) (Welch, 20 Hz smoothing,
   300–6000 Hz; the rotor's emission cancels whatever its speed law) is the
   same curve at 50–90 % duty on the frequency axis (mean cross-duty corr
   +0.54…+0.60 per rotor, all 26 pairs) and not on the order axis (+0.18…
   +0.24). Its size is modest: sd 1.4–2.7 dB over frequency per mic; broad
   shapes (tilts, a step at ~4 kHz, notches).
5. **The fine (per-line) part is not a fine-grained RTF.** A room transfer
   function has features ≈ 4/T₆₀ Hz wide, finer than a harmonic spacing, so
   a per-line scatter is what an RTF would produce — but (a) the gaps between
   harmonics are only 0–6 dB above the rotor-off floor (pre-spin-up segment),
   so they cannot probe it (`mic_eq_fine.py`: fine part 0.7 dB sd, cross-duty
   correlation flat vs smoothing 2.7→50 Hz), and (b) on the lines themselves
   (`line_coincidence.py`) orders of two duties whose frequencies coincide
   within 2 Hz agree no better (corr +0.45, rms diff 3.8 dB) than orders
   10–40 Hz apart (+0.46) or the same order at the other speed (+0.48). The
   per-line term is tied neither to frequency nor to order.
6. **Variance accounting** (`mic_variance.py`, held-out duty, orders ≤ 60):
   raw 3.64 dB; per-mic sensitivity only → 3.07 (29 %); per-pair RTF → 2.70
   (45 %); pair + mic → 2.70 (mic is inside the pair term: the two EQs
   correlate +0.81, pair 2.4 dB sd of which 2.0 mic-common, 1.4 rotor-
   specific). Explained share flat for EQ widths 50–400 Hz: the reproducible
   part is broad. Remaining ≈ 2.7 dB per line.
7. **Within-run vs between-run** (`pattern_stability.py`): first vs second
   half of a span differ by 1.73 dB rms (corr +0.89; run mean good to ~0.9
   dB), other duty by 3.71 dB (+0.48). Odd/even wander alike (1.69/1.77),
   high orders less (1.4 dB at k 31–60). ≈ 2.4 dB therefore changes
   genuinely from run to run; ≈ 1 dB breathes within a run (the mic-specific
   part of the AM: cross-mic correlation of ln A_k 0.3–0.8).

Reading. The array is in the rotor's near field: each harmonic's field
across the mics is an interference pattern of an extended source (blade
sections, two blades, body scattering). Its broad, speed-independent part is
the pair RTF; its fine part depends on the loading distribution at the
operating point, so at a new speed neither the same frequency (different
order of a different loading) nor the same order (different wavelength and
loading) reproduces it. [INFERENCE: mechanism; the magnitudes and the
"neither axis" fact are measured.] Whether the per-run term varies smoothly
on a finer speed grid or jumps is unknown (10 % duty steps).

Model statement per (rotor, mic, line), for the generative prior:
mean rotor profile (shared by mics) + fixed pair EQ (≈ 2.4 dB sd, smooth in
f, ~2/3 mic-common) + per-run draw (≈ 2.4 dB, independent per line, ρ ≈ 0.25
between same-parity neighbours) + within-run wander (≈ 1 dB over 15 s, the
mic-specific AM). Together with the OU shaft (σ_ν, λ) and the per-order AM
envelopes this is the current form of the single-rotor noise model.

## Generic drone prior (2026-10-02)

`src/experiments/noise_model/drone_prior.py` turns the measurements above
into a prior over rigs (`sample_prior` → a `noise-v3-fit/1` payload the
renderer accepts; `trajectory_windows`, `gated_draws`). Driven by
`notebooks/prior_rigs_explainer.ipynb` / `prior_rigs_lab.py`; draws cached
in `results/prior_rigs/<name>/` (payloads with `_prior` provenance) and
`.cache/prior_rigs/<name>/` (audio). Laws (`PriorSpec`), each sourced:

| Block | Law | Source |
|---|---|---|
| Comb, even orders | line over floor: `k2 − slope·log10(k/2)`, floored at a tail; `k2 ~ N(26, 5)` dB per rotor, `slope ~ N(25, 5)` dB/dec, `tail ~ N(0, 2)` dB; `k = 2` + N(8, 4) | Michael's cruise/standby v3 fits at 80 rev/s (stacked `k = 2` 36/29 dB, tail 3–8 dB stacked from k ≈ 15 to 75–100); bench slope of the table profiles |
| Odd orders | even law − (N(5, 2) + N(10, 4)·log10 k); `k = 1` = `k2` − N(5, 2) | gallery: odd lines sink with k |
| Motor family | multiples of 3p, p ∈ {6, 7, 7, 8, 11}: + N(9, 3) dB × 0.8 per multiple | k = 42, 63, 84, 126 lines (21 = 3 × 7) |
| Rotor scatter | N(0, 2.5) per line + N(0, 1.5) level | between-run 3.7 dB minus the per-mic part |
| Mic scatter | 3.4 dB per (mic, line), redrawn per clip | `mic_variance.py` (raw 3.64 dB) |
| Shaft | OU: σ_ν ~ LogN(log 0.4 rad/s, 0.4), λ ~ LogN(log 12, 0.4); `gamma_hz` ≡ 0 | quick OU fits, 16/20 recordings |
| Pedestal (AM + fast wobble) | per order, shared by mics: σ² ~ LogN(log 0.05, 1) (k ≤ 8), LogN(log 0.12, 0.7) (9–30), min(0.07 + k²σ_ψ², 0.6) above (σ_ψ ~ LogN(log 0.012, 0.7) rad); rate 0.4 → 1 Hz, 25 Hz where the wobble dominates; same-parity AR(1) ρ 0.25 | direct AM measurement; six-rig telemetry S_max |
| Floor | mean N(−36, 4) dB, hump/tilt template 50/50, σ_B U(4, 5); speed law LogN(log 5, 0.3), static share LogN(log 2.5e−3, 1) | DREGON/Michael v3 floors |
| Wind | SC generator (its own Weibull gust profile, `gustiness` 3, LPC following the speed) on some capsules: p 0.7, mean level up to 33 dB over the 20–100 Hz floor minus Exp(8 dB) shielding | DREGON free flight (`free-flight_nosource_room2`, mics 0/1/4): fitted wind 16/15/10/16 dB over the rotor floor on the windy capsules (a lower bound: the quietest mic defines zero); 20–100 Hz envelope 10–90 % 11.5–12.3 dB, 1.0–1.4 bursts/s; burst−gap excess 14–19 dB at 50–400 Hz, 2–12 dB at 800 Hz, 0–6 dB at 1.6 kHz, gone by 2.4 kHz. SC reproduces this (prior windy capsule: 10–14.5 dB, 13–19 / −1…8 / −6…8 dB); the renderer must NOT feed it a constant speed profile, which is what made the first draws' wind steady and band-limited to a shelf |

The comb is written as LINE OVER FLOOR because that is what the fits
measure and what a recording shows; `profile_db` is obtained by reading
the per-order conversion (window gain, OU width at order k, floor shape)
off the model's own expected periodogram of a flat profile
(`line_contrast_db`, one rotor, 2048-point Hann at 60 rev/s). Two
earlier forms were wrong: a profile-space level `k2_over_floor` N(−8, 8)
put k = 2 anywhere from +40 to −10 dB over the floor (half the draws had
no visible lines, yet passed the 1 dB gate: the gate is statistical, not a
visibility test), and a two-slope comb cut at K = 115 left the last
orders 20–30 dB over a tilted floor (a cliff no recording has). The gate
now reads the pedestal through the lag law (`identifiability.line_shapes`
× exp(σ² e^{−2πγ|τ|})) and the lab gates at 3 dB / ≥ 2 orders / 80 % of
frames (12 draws for 10 rigs at seed 0). At the same colour scale a prior
draw and the DREGON free-flight / hovering recordings look alike
(`results/prior_rigs/prior_v1_seed0/real_vs_prior.png`); Michael's FLY125
hover shows a steadier comb because its speeds barely move.
