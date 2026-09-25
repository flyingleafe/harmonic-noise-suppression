# Noise model v3 — spectral and parity checks on the single-seed timing fits

Fits: `results/noise_v3/fits_gpu/{dregon_room2_floor,michaels_fly125_cruise,michaels_fly125_standby}__flight_v3.json`
(schema `noise-v3-fit/1`, one seed, 3 rounds, T4 timing jobs of `f65179a7`; scalar-wander round).
Code at git `448312f8`. Everything below ran on the laptop (scoring 168 s DREGON, 439 s Michael's; spectral probe 385 s).

## Protocol

- **Windows** = the R5 score windows, each render driven by that window's OWN raw scoring label
  (`GT.RAW_RPS_KEY`: DREGON `motors_command`, Michael's `rps`): the five frozen DREGON room-2 4 s
  cruise windows and the FLY124 standby @8/@16 and cruise @40/@56 8 s windows (plus the ramp @27.68,
  scored by the parity gate only). Seeds 2001–2004, 8 mics.
- **Parity** = `scripts/noise_v2_round_score.py` exactly as R5 invoked it, with `--fit`:

  ```
  PYTHONPATH=src python scripts/noise_v2_round_score.py --round 5 --fits results/noise_v3/fits_gpu --rigs dregon \
    --fit dregon=results/noise_v3/fits_gpu/dregon_room2_floor__flight_v3.json \
    --arm-out results/noise_v3/checks_timing_fits/parity/dregon/arm_dregon_v3_timing.json --dump-audio <audio>
  PYTHONPATH=src python scripts/noise_v2_round_score.py --round 5 --fits results/noise_v3/fits_gpu --rigs michaels \
    --fit michaels=results/noise_v3/fits_gpu/michaels_fly125_cruise__flight_v3.json \
    --fit michaels_standby=results/noise_v3/fits_gpu/michaels_fly125_standby__flight_v3.json \
    --arm-out results/noise_v3/checks_timing_fits/parity/michaels/arm_michaels_v3_timing.json --dump-audio <audio>
  ```

  Frozen probe `hppnet_l2_r2_s0/best` (sha256 verified), real-arm reproduction 1.218708 rev/s against
  the frozen 1.218708 (relative 1.8e-08). The v3 fit is scored AS FITTED (no calibration pin).
- **Spectral** = (i) the R5 proxy `ltas_abs_db` (`revised_eval.ltas_deviation_db(real, render, mic=0)`
  `mean_abs_db`, seed 2001, absolute level; the gate reads the DREGON 5-window mean and the Michael's
  cruise 2-window mean) and (ii) the R4 prominence ladder k = 1..8, `peak_over_base_db` and −3 dB width
  of `noise_v2_legacy_truth.order_profiles`/`profile_stats` (= `noise_v2_widen_dregon.line_width_db3`,
  8192/1024, order-tracked on the label, mic- and rotor-averaged), seed-mean over 2001–2004. v3 audio
  is the audio the parity pass scored (dumped; its `real` channel matches the reloaded clip bit for
  bit); v2 is re-rendered through `noise_v2_round_score`'s own arm route on the same labels and seeds.
  Protocol check: the v2 re-renders reproduce R5/R3's recorded per-window `ltas_abs_db` to 4 decimals
  (DREGON pin 2.4631/1.8790/3.6168/1.2399/0.9503; Michael's 0.5012/0.3418/0.4727/2.0906).
  Record `spectral/spectral.json` (written by a throwaway laptop script calling exactly these
  functions; not committed).
- v2 comparanda: DREGON = R5 `flight_profile` + the +3.75 dB pin (the scored R5 arm) and R5
  `flight_profile` as fitted; Michael's = R3 per-regime pair (the R5 incumbent).

## Spectral: `ltas_abs_db`, mic 0 (dB, lower is better)

| rig / group | bar | v3 timing, seed 2001 (gate read) | v3, seed mean | v2 scored (R5) | v2 seed mean | v2 R5 as fitted (DREGON) |
|---|---:|---:|---:|---:|---:|---:|
| DREGON cruise (5) | 1.9786 | **2.6866** FAIL | 2.9089 | 2.0298 FAIL | 2.0168 | 3.3914 |
| Michael's cruise (2) | 1.2197 | **1.8422** FAIL | 2.6193 | 1.2816 FAIL | 1.2462 | — |
| Michael's standby (2, report only) | — | 2.4369 | 2.2659 | 0.4215 | 0.4491 | — |

Per window, v3 seed 2001..2004 (v2 pin in brackets): free-flight 3.23/4.88/1.94/3.41 (2.46),
hovering 2.70/4.31/1.76/2.89 (1.88), updown 4.57/6.01/3.12/4.53 (3.62), rectangle 1.46/2.34/1.86/1.35
(1.24), spinning 1.47/3.15/1.49/1.70 (0.95); FLY124 @40 1.59/2.37/2.78/2.15 (0.47), @56
2.10/3.50/3.96/2.51 (2.09). v3's seed-to-seed spread is 0.5–2.9 dB against v2's ≤ 0.2 dB: the v3
render draws a fresh level wander per clip, so a one-seed absolute-level proxy is a noisy read of it.
Restoring the fitted mic-0 channel gain (v3 renders every mic at unit gain; DREGON mic 0 is −0.434 dB,
Michael's +0.028 dB) does not help: DREGON 3.0464, Michael's cruise 1.8542.

## Spectral: R4 prominence ladder, k = 1..8 (dB over the 0.45–0.70 f̄ annulus; seed mean, window mean)

`(n/N)` = only n of N windows show a peak (> 0 dB excess) at that order; `—` = none do.
mean |Δ| = mean absolute prominence difference to the real window over the (window, k) cells where
both have a peak.

**DREGON room-2 cruise, 5 windows**

| arm | k=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | mean \|Δ\| vs real | \|Δ\| k=1,2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real | 6.71 | 2.65 (4/5) | 0.81 (3/5) | 1.15 (3/5) | 0.90 (2/5) | 0.57 (4/5) | 0.11 (2/5) | 0.38 (3/5) | — | — |
| v3 timing | 7.27 | 5.40 | 0.65 (4/5) | 0.64 (4/5) | 1.20 (4/5) | 1.08 (4/5) | 0.14 (4/5) | 0.68 | **1.06** | **2.24** |
| v2 R5 + pin (scored) | 9.42 | 8.49 | 0.85 (3/5) | 2.03 | 0.81 | 1.37 | 0.24 | 0.87 | 2.04 | 4.32 |
| v2 R5 as fitted | 6.73 | 5.48 | 0.83 (3/5) | 1.09 | 0.64 | 0.87 | 0.22 | 0.54 | 1.16 | 2.44 |

−3 dB width k=1/2 Hz: real 13.5/29.0, v3 11.3/20.6, v2 pin 11.2/13.1.

**Michael's FLY124 cruise @40/@56**

| arm | k=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | mean \|Δ\| vs real | \|Δ\| k=1,2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real | 6.54 (1/2) | 22.52 | — | 9.93 | 8.28 | 8.18 | 2.35 | 6.58 | — | — |
| v3 timing | 2.39 | 21.69 | — | 7.92 | 4.60 | 6.71 | 1.64 | 6.17 | **2.20** | **1.82** |
| v2 R3 regimes | — | 22.51 | — | 8.22 | 5.15 | 6.85 | 1.05 | 5.87 | 1.67 | 0.44 |

**Michael's FLY124 standby @8/@16**

| arm | k=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | mean \|Δ\| vs real | \|Δ\| k=1,2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real | — | 11.56 | — | 3.69 | 0.38 | 6.19 | 5.49 | 5.52 | — | — |
| v3 timing | 21.42 | 9.17 | 5.34 | 3.93 | 4.01 | 8.22 | 8.87 | 8.17 | **2.59** | **3.17** |
| v2 R3 regimes | 13.72 | 14.38 | — | 4.51 | 3.52 | 7.83 | 4.83 | 4.74 | 1.84 | 3.11 |

The standby v3 render carries a 21.4 dB k = 1 line (and a 5.3 dB k = 3 line) where the real FLY124
standby windows show none; v2's k = 1 was already spurious (13.7 dB), v3 is 7.7 dB louder there.

## Parity: HPPNet PIT MAE (rev/s), as R5 scored it

| rig | quantity | v3 timing | v2 (R5) | PARITY bar | verdict |
|---|---|---:|---:|---:|---|
| DREGON | 5-recording cruise mean | **1.784241** | 1.820065 | 2.187786 | **PASS** (+0.403544) |
| DREGON | one-sided 95 % upper | **2.100938** | 2.158331 | 2.187786 | **PASS** (+0.086848) |
| DREGON | mean vs STRETCH 1.897063 | 1.784241 | 1.820065 | 1.897063 | PASS (+0.112821) |
| DREGON | upper vs STRETCH 1.897063 | 2.100938 | 2.158331 | 1.897063 | FAIL (−0.203875), as v2 |
| Michael's | equal-regime mean (ratio to 3.026661) | **1.657333 (0.5476)** | 1.621670 (0.5358) | 3.177994 (ratio ≤ 1.05) | **PASS** (+1.520662) |

Per support (real / v3 / v2): DREGON free-flight 0.638 / 1.364 / 1.423, hovering 0.888 / 1.636 /
1.649, updown 1.695 / 1.947 / 1.931, rectangle 1.685 / 2.246 / 2.365, spinning 1.187 / 1.729 /
1.732. Michael's per regime (v3 / v2 / frozen legacy): standby 0.794 / 0.785 / 0.317, ramp 3.243 /
3.267 / 8.007, cruise 0.936 / 0.813 / 0.756. v3's per-support seed spread is larger (DREGON
0.24–0.41 against v2's 0.05–0.16 rev/s; FLY124 standby @16 1.60).

**Familiarity caveat.** Every scored window — the five DREGON room-2 recordings and FLY124 — is in
HPPNet's training pool (83.3 % real / 16.7 % zero-labelled silence / 0 % rig renders; R5
`tracker_probe/findings.md:150-176`), so every PIT number here is in-domain. The DREGON v3 pool is
the five 8 s floor segments adjacent to (disjoint from) the score windows; the Michael's fits are
FLY125, scored on the held-out FLY124.

Side reading (not asked, not a parity gate): the same pass's Michael's comb-band likelihood is
model −348 031 nats/s against oracle −371 917 (margin +23 885, FAIL; v2 R3 −1 750, PASS). [INFERENCE]
Much of this is expected to be the channel gains: the v3 expected periodogram is unit-gain on every
mic (the fit saw channel-normalised data), while the gate reads the un-normalised FLY124 channels,
whose fitted gains span −6.5 … +5.0 dB. Not verified here.

## Verdict

1. **Spectrum: farther than v2 on the proxy, mixed on the ladder.** `ltas_abs_db` worsens on both rigs
   (DREGON 2.69 vs 2.03, Michael's cruise 1.84 vs 1.28; both still FAIL, now by 0.71 / 0.62 dB), with a
   0.5–2.9 dB seed spread v2 does not have. The DREGON ladder is CLOSER to real than the scored v2
   (mean |Δ| 1.06 vs 2.04 dB; k = 2 5.4 vs 8.5 against 2.7 real), and v3 as fitted beats v2 as fitted on the proxy
   (2.69 vs 3.39). Michael's ladder is FARTHER (cruise 2.20 vs 1.67: k = 1 too weak, k = 5 −3.7 dB;
   standby 2.59 vs 1.84: a spurious 21 dB k = 1 line).
2. **Parity: PASS on both rigs.** DREGON 1.784 (95 % upper 2.101) ≤ 2.188, better than v2 without a
   pin; Michael's 1.657, ratio 0.548 ≤ 1.05, 0.036 rev/s worse than v2 (cruise 0.936 vs 0.813).
3. **Campaign:** the scalar-wander round costs no tracker parity but buys no spectral closure. The
   restarts should be ranked on the seed-mean proxy and the Michael's k = 1 lines (standby overshoot,
   cruise deficit), not on HPPNet, which saturates here. A one-seed `ltas_abs_db` cannot rank v3 fits
   whose render level wanders by ±1.5 dB per seed.
