# Noise model v2 — the legacy model's Whittle risk on a SHORT front end

Author: `ShortWhittle`. Every number is produced by `scripts/noise_v2_short_whittle.py` (→ `short_whittle.json`) in this directory. One measurement, not a study.

The objective is the FROZEN composite spectral risk, taken from the evaluator itself (`revised_eval.predicted_m` → `marginal_frame_nll` → `FrameScore` → `composite_score`): positive exposure-weighted `Σ a_i (I_i/M_i + log M_i)` on 30–7900 Hz, per unique observed second, with the `hop/n_fft` exposure factor and the duplicate-frame rule. **Only `(NFFT, hop)` changes between rows**, so the column is comparable down its own length. Lower is better; the numbers are ABSOLUTE-level risks in nats per unique second, so they are large and signed — read the `model − oracle` difference, not the magnitude.

## The table

| rig | NFFT | hop | legacy model (nats/s) | oracle (nats/s) | model − oracle | model wall (s) | setting wall (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| dregon | 2048 | 512 | 3,575,832.5 | -396,275.2 | 3,972,107.7 | 1.8 | 2.1 |
| dregon | 1024 | 256 | 852,996.1 | -398,038.6 | 1,251,034.7 | 2.7 | 3.1 |
| dregon | 4096 | 1024 | 5,997,418.7 | -393,366.4 | 6,390,785.1 | 3.6 | 4.0 |
| dregon (legacy ref) | 16384 | 1024 | 8,430,103.9 | -382,137.5 | 8,812,241.5 | 11.0 | 12.1 |
| michaels | 2048 | 512 | -393,486.7 | -459,805.1 | 66,318.4 | 6.1 | 6.8 |
| michaels | 1024 | 256 | -392,654.2 | -459,503.9 | 66,849.6 | 5.5 | 6.2 |
| michaels | 4096 | 1024 | -393,340.7 | -459,338.6 | 65,997.9 | 7.1 | 7.9 |
| michaels (legacy ref) | 16384 | 1024 | -390,782.6 | -456,757.4 | 65,974.8 | 27.5 | 30.7 |

Oracle = the study's own non-parametric floor: `M` is the Welch mean periodogram of a speed-matched legal disjoint real segment of the SAME recording at the SAME NFFT (`results/noise_v2/criteria/findings.md` § 3.1). The oracle carries no model and no carrier, and it is the same material at every setting.

Why the four rows are comparable at all: the exposure weights sum to `sr/n_fft` per second and a frame carries `mics × band_bins` cells, so the CELL BUDGET per unique second is `mics × band_bins × sr/n_fft` = [62961, 62969, 63000] — the same to 0.1 % at every setting. A short front end changes the resolution, not how much material is scored.

## What it says

* **dregon** — gap to the floor in nats/s: 2048/512 3,972,108, 1024/256 1,251,035, 4096/1024 6,390,785, 16384/1024 8,812,241. The best setting is 1024/256 at 1,251,035, a factor 7.04 of the legacy 16384/1024 reference (8,812,241).
  Median in-band level offset (real over model): 1024/256 +3.92 dB, 2048/512 +4.44 dB, 4096/1024 +5.08 dB, 16384/1024 +5.97 dB. Share of `Σ I/M` carried by the worst 0.1 % of in-band cells (median over supports): 1024/256 34.5%, 2048/512 31.9%, 4096/1024 25.4%, 16384/1024 20.5%.
* **michaels** — gap to the floor in nats/s: 2048/512 66,318, 1024/256 66,850, 4096/1024 65,998, 16384/1024 65,975. The best setting is 16384/1024 at 65,975, a factor 1.00 of the legacy 16384/1024 reference (65,975).
  Median in-band level offset (real over model): 1024/256 -1.17 dB, 2048/512 -1.23 dB, 4096/1024 -1.31 dB, 16384/1024 -1.46 dB. Share of `Σ I/M` carried by the worst 0.1 % of in-band cells (median over supports): 1024/256 3.1%, 2048/512 4.0%, 4096/1024 4.4%, 16384/1024 4.9%.

The per-regime split, because one export is used on every support of its rig and the frames of the five supports are disjoint (so the numerator and the unique seconds simply add):

| rig | regime | 2048/512 | 1024/256 | 4096/1024 | 16384/1024 |
|---|---|---:|---:|---:|---:|
| dregon | cruise | 3,972,108 | 1,251,035 | 6,390,785 | 8,812,241 |
| michaels | cruise | -107 | 470 | -409 | -240 |
| michaels | ramp | 12,853 | 14,100 | 12,659 | 17,267 |
| michaels | standby | 159,477 | 159,604 | 159,074 | 156,544 |

(model − oracle gap in nats/s, pooled over that regime's supports.)

## Exports used, and what they can and cannot supply

* **dregon** — `results/S1/bayes_rig.json`, bench-fitted (S1 rig fit, 20 single-motor bench cells); route `aggregate` over 20 fitted clips (`aggregate_nuisance`, linear-power mean, `power_scale` folded per clip, clip-local latents dropped: ['h_db', 'floor_level_db', 'floor_tilt_gp', 'rps_offset', 'carrier']).
  The bench rig fit carries 1 rotor, so its tied profile is tiled onto the flight's 4 rotors (one shape at one level per rotor, as in `scripts/_dregon_transfer.py`); the bench is single-channel, so no per-microphone pattern is transferred and the absolute level and the room floor are the bench's.
  The scored recording is named by no clip of this export: this is the manifest's declared extrapolation, not an identity match.
* **michaels** — `results/S2/cruise_8clip_refined.json`, flight-fitted (S2 FLY125 cruise, 8 x 16 s, rps_refined); route `aggregate` over 8 fitted clips (`aggregate_nuisance`, linear-power mean, `power_scale` folded per clip, clip-local latents dropped: ['h_db', 'floor_level_db', 'floor_tilt_gp', 'rps_offset', 'carrier']).
  The scored recording is named by no clip of this export: this is the manifest's declared extrapolation, not an identity match.

## Run time

| rig | NFFT | hop | audio (s) | `predicted_m` (s) | s per audio-s | wall (s) |
|---|---:|---:|---:|---:|---:|---:|
| dregon | 1024 | 256 | 20 | 2.7 | 0.14 | 3.1 |
| dregon | 2048 | 512 | 20 | 1.8 | 0.09 | 2.1 |
| dregon | 4096 | 1024 | 20 | 3.6 | 0.18 | 4.0 |
| dregon | 16384 | 1024 | 20 | 11.0 | 0.55 | 12.1 |
| michaels | 1024 | 256 | 40 | 5.5 | 0.14 | 6.2 |
| michaels | 2048 | 512 | 40 | 6.1 | 0.15 | 6.8 |
| michaels | 4096 | 1024 | 40 | 7.1 | 0.18 | 7.9 |
| michaels | 16384 | 1024 | 40 | 27.5 | 0.69 | 30.7 |

The criteria study measured ~20 s of `predict_spectrum` (the CANDIDATE path) per audio-second at NFFT 16384. The legacy `predicted_m` path measured here is two to three orders of magnitude cheaper, so the whole grid runs on the laptop in minutes and no cluster submission is needed — which is just as well, since both exports are gitignored and `omnirun` ships the git revision only.

## Per-support detail

| rig | support | regime | NFFT | model (nats/s) | oracle (nats/s) | model − oracle | model level offset (dB) | speed mismatch (rev/s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| dregon | free-flight_nosource_room2@1512727397.205045+4.000000 | cruise | 1024 | 871,983.7 | -394,798.2 | 1,266,782.0 | +5.11 | 0.397 |
| dregon | free-flight_nosource_room2@1512727397.205045+4.000000 | cruise | 2048 | 3,720,283.5 | -394,797.0 | 4,115,080.5 | +5.40 | 0.397 |
| dregon | free-flight_nosource_room2@1512727397.205045+4.000000 | cruise | 4096 | 6,525,885.9 | -394,081.6 | 6,919,967.5 | +5.86 | 0.397 |
| dregon | free-flight_nosource_room2@1512727397.205045+4.000000 | cruise | 16384 | 9,773,698.3 | -388,258.8 | 10,161,957.2 | +7.31 | 0.397 |
| dregon | hovering_nosource_room2@1511903905.394490+4.000000 | cruise | 1024 | 749,472.1 | -396,851.5 | 1,146,323.7 | +3.92 | 1.725 |
| dregon | hovering_nosource_room2@1511903905.394490+4.000000 | cruise | 2048 | 3,393,617.0 | -394,964.8 | 3,788,581.8 | +4.44 | 1.725 |
| dregon | hovering_nosource_room2@1511903905.394490+4.000000 | cruise | 4096 | 5,488,430.5 | -393,063.5 | 5,881,494.0 | +5.08 | 1.725 |
| dregon | hovering_nosource_room2@1511903905.394490+4.000000 | cruise | 16384 | 7,869,638.1 | -383,364.5 | 8,253,002.6 | +6.57 | 1.725 |
| dregon | rectangle_nosource_room2@1511905725.952559+4.000000 | cruise | 1024 | 702,130.7 | -411,083.1 | 1,113,213.8 | +2.18 | 0.984 |
| dregon | rectangle_nosource_room2@1511905725.952559+4.000000 | cruise | 2048 | 2,727,616.4 | -408,569.4 | 3,136,185.8 | +2.47 | 0.984 |
| dregon | rectangle_nosource_room2@1511905725.952559+4.000000 | cruise | 4096 | 4,327,974.9 | -403,766.7 | 4,731,741.5 | +2.88 | 0.984 |
| dregon | rectangle_nosource_room2@1511905725.952559+4.000000 | cruise | 16384 | 4,814,333.9 | -407,835.3 | 5,222,169.3 | +2.62 | 0.984 |
| dregon | spinning_nosource_room2@1511905200.978012+4.000000 | cruise | 1024 | 706,169.5 | -418,984.0 | 1,125,153.5 | +2.81 | 1.846 |
| dregon | spinning_nosource_room2@1511905200.978012+4.000000 | cruise | 2048 | 2,846,755.3 | -419,327.6 | 3,266,082.9 | +3.06 | 1.846 |
| dregon | spinning_nosource_room2@1511905200.978012+4.000000 | cruise | 4096 | 4,607,803.0 | -418,950.7 | 5,026,753.7 | +3.69 | 1.846 |
| dregon | spinning_nosource_room2@1511905200.978012+4.000000 | cruise | 16384 | 5,998,981.9 | -414,107.2 | 6,413,089.1 | +4.97 | 1.846 |
| dregon | updown_nosource_room2@1511903578.348311+4.000000 | cruise | 1024 | 1,235,224.5 | -368,475.9 | 1,603,700.3 | +4.68 | 1.488 |
| dregon | updown_nosource_room2@1511903578.348311+4.000000 | cruise | 2048 | 5,190,890.4 | -363,717.0 | 5,554,607.4 | +4.95 | 1.488 |
| dregon | updown_nosource_room2@1511903578.348311+4.000000 | cruise | 4096 | 9,036,999.4 | -356,969.5 | 9,393,969.0 | +5.29 | 1.488 |
| dregon | updown_nosource_room2@1511903578.348311+4.000000 | cruise | 16384 | 13,693,867.4 | -317,121.8 | 14,010,989.2 | +5.97 | 1.488 |
| michaels | FLY124@16.000000+8.000000 | standby | 1024 | -424,186.5 | -585,901.3 | 161,714.8 | -14.42 | 0.047 |
| michaels | FLY124@16.000000+8.000000 | standby | 2048 | -424,415.6 | -586,571.2 | 162,155.6 | -14.48 | 0.047 |
| michaels | FLY124@16.000000+8.000000 | standby | 4096 | -424,296.6 | -586,404.2 | 162,107.6 | -14.50 | 0.047 |
| michaels | FLY124@16.000000+8.000000 | standby | 16384 | -424,192.6 | -584,392.0 | 160,199.4 | -14.59 | 0.047 |
| michaels | FLY124@27.680000+8.000000 | ramp | 1024 | -357,526.4 | -371,626.6 | 14,100.2 | -0.67 | 8.286 |
| michaels | FLY124@27.680000+8.000000 | ramp | 2048 | -358,321.7 | -371,174.5 | 12,852.7 | -0.72 | 8.286 |
| michaels | FLY124@27.680000+8.000000 | ramp | 4096 | -357,593.3 | -370,252.2 | 12,658.9 | -0.89 | 8.286 |
| michaels | FLY124@27.680000+8.000000 | ramp | 16384 | -350,711.3 | -367,978.3 | 17,267.0 | -1.03 | 8.286 |
| michaels | FLY124@40.000000+8.000000 | cruise | 1024 | -365,573.1 | -365,160.4 | -412.7 | +0.33 | 0.877 |
| michaels | FLY124@40.000000+8.000000 | cruise | 2048 | -366,137.4 | -364,483.6 | -1,653.7 | +0.27 | 0.877 |
| michaels | FLY124@40.000000+8.000000 | cruise | 4096 | -365,892.3 | -363,349.1 | -2,543.3 | +0.24 | 0.877 |
| michaels | FLY124@40.000000+8.000000 | cruise | 16384 | -361,427.5 | -359,359.7 | -2,067.8 | +0.10 | 0.877 |
| michaels | FLY124@56.000000+8.000000 | cruise | 1024 | -392,962.9 | -394,315.5 | 1,352.7 | -1.17 | 0.389 |
| michaels | FLY124@56.000000+8.000000 | cruise | 2048 | -395,336.7 | -396,775.9 | 1,439.2 | -1.23 | 0.389 |
| michaels | FLY124@56.000000+8.000000 | cruise | 4096 | -395,843.5 | -397,568.9 | 1,725.4 | -1.31 | 0.389 |
| michaels | FLY124@56.000000+8.000000 | cruise | 16384 | -394,594.3 | -396,181.5 | 1,587.2 | -1.46 | 0.389 |
| michaels | FLY124@8.000000+8.000000 | standby | 1024 | -423,022.3 | -580,515.4 | 157,493.1 | -14.28 | 0.020 |
| michaels | FLY124@8.000000+8.000000 | standby | 2048 | -423,222.1 | -580,020.3 | 156,798.2 | -14.30 | 0.020 |
| michaels | FLY124@8.000000+8.000000 | standby | 4096 | -423,077.8 | -579,118.5 | 156,040.8 | -14.32 | 0.020 |
| michaels | FLY124@8.000000+8.000000 | standby | 16384 | -422,987.1 | -575,875.6 | 152,888.5 | -14.41 | 0.020 |

## Provenance

* git HEAD `ae47fd4d0c63e06bcebe2a0724d6829f67f1a868`, 2026-09-16T23:28:08+0100, wall 78.0 s
* band 30.0–7900.0 Hz, 16 kHz periodic Hann
* historical-forward baseline diagnostic: the LEGACY descriptive model's own forward prediction (model.CombSpectrum — frame-mean carrier plus a known chirp-width covariate), NOT the exact moving-window kernel the candidate path uses. Its ramp prediction in particular is an approximation the contract rejects for a candidate.

* pooled `dregon`/`legacy_model` at 1024/256: 852,996.0996 nats/s, 211.5566 nats/band-cell, 19.760 unique s over 5 supports, 1235 frames
* pooled `dregon`/`legacy_model` at 2048/512: 3,575,832.5047 nats/s, 443.4316 nats/band-cell, 19.520 unique s over 5 supports, 610 frames
* pooled `dregon`/`legacy_model` at 4096/1024: 5,997,418.7386 nats/s, 372.0483 nats/band-cell, 18.880 unique s over 5 supports, 295 frames
* pooled `dregon`/`legacy_model` at 16384/1024: 8,430,103.9436 nats/s, 130.7560 nats/band-cell, 15.040 unique s over 5 supports, 235 frames
* pooled `dregon`/`oracle_np` at 1024/256: -398,038.5561 nats/s, -98.7199 nats/band-cell, 19.760 unique s over 5 supports, 1235 frames
* pooled `dregon`/`oracle_np` at 2048/512: -396,275.1609 nats/s, -49.1413 nats/band-cell, 19.520 unique s over 5 supports, 610 frames
* pooled `dregon`/`oracle_np` at 4096/1024: -393,366.4042 nats/s, -24.4024 nats/band-cell, 18.880 unique s over 5 supports, 295 frames
* pooled `dregon`/`oracle_np` at 16384/1024: -382,137.5402 nats/s, -5.9272 nats/band-cell, 15.040 unique s over 5 supports, 235 frames
* pooled `michaels`/`legacy_model` at 1024/256: -392,654.2472 nats/s, -97.3845 nats/band-cell, 39.760 unique s over 5 supports, 2485 frames
* pooled `michaels`/`legacy_model` at 2048/512: -393,486.7009 nats/s, -48.7955 nats/band-cell, 39.520 unique s over 5 supports, 1235 frames
* pooled `michaels`/`legacy_model` at 4096/1024: -393,340.6817 nats/s, -24.4008 nats/band-cell, 39.040 unique s over 5 supports, 610 frames
* pooled `michaels`/`legacy_model` at 16384/1024: -390,782.5605 nats/s, -6.0613 nats/band-cell, 35.200 unique s over 5 supports, 550 frames
* pooled `michaels`/`oracle_np` at 1024/256: -459,503.8722 nats/s, -113.9643 nats/band-cell, 39.760 unique s over 5 supports, 2485 frames
* pooled `michaels`/`oracle_np` at 2048/512: -459,805.1097 nats/s, -57.0195 nats/band-cell, 39.520 unique s over 5 supports, 1235 frames
* pooled `michaels`/`oracle_np` at 4096/1024: -459,338.5705 nats/s, -28.4949 nats/band-cell, 39.040 unique s over 5 supports, 610 frames
* pooled `michaels`/`oracle_np` at 16384/1024: -456,757.4049 nats/s, -7.0846 nats/band-cell, 35.200 unique s over 5 supports, 550 frames
