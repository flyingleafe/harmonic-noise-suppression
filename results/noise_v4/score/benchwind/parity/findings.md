# Noise model v2 — round 6 gate score (FAIL)

Arm: v2 — dregon: v4 bench rig fit (Motor i -> flight rotor i, per-rotor floor, pair EQ)
Render seeds [2001, 2002, 2003, 2004], 8 mics, git `0a74352bdf30724f71a9050e752aafc275619d62`.

## Gate verdicts

| gate | verdict | number | threshold |
|---|---|---|---|
| HPPNet DREGON cruise | FAIL | 2.627299 rev/s (95 % upper 2.876260) | <= 1.897063 |
| HPPNet Michael's ratio | NOT RUN | no FLY124 support measured (5 missing) | <= 1.05 (3.177994 rev/s) |
| proxy `ltas_abs_db` dregon_cruise | FAIL | 3.9388 dB (spread 2.6278) | <= 1.9786 dB |
| proxy `ltas_abs_db` michaels_cruise | FAIL | nan dB (spread nan) | <= 1.2197 dB |
| likelihood | FAIL | no Michael's cruise likelihood cell was built | — |

## HPPNet PIT MAE per support (rev/s)

| rig | support | regime | real | candidate | seed spread |
|---|---|---|---:|---:|---:|
| dregon | `free-flight_nosource_room2@1512727397.205045+4.000000` | cruise | 0.638129 | 2.369305 | 0.163723 |
| dregon | `hovering_nosource_room2@1511903905.394490+4.000000` | cruise | 0.888408 | 2.455699 | 0.296873 |
| dregon | `updown_nosource_room2@1511903578.348311+4.000000` | cruise | 1.694930 | 2.959203 | 0.408530 |
| dregon | `rectangle_nosource_room2@1511905725.952559+4.000000` | cruise | 1.685142 | 2.851958 | 0.214885 |
| dregon | `spinning_nosource_room2@1511905200.978012+4.000000` | cruise | 1.186931 | 2.500331 | 0.811641 |

DREGON cruise: candidate mean 2.627299 rev/s over 5 recording-level clusters, one-sided 95 % interval [2.378338, 2.876260] (t and 20000-draw cluster bootstrap, conservative), against the frozen target 1.897063 = 1.218708 + 0.7 x (2.187786 - 1.218708). Margin -0.730237 rev/s.

Real arm reproduction: measured 1.218708 rev/s against the frozen 1.218708 (relative 1.188e-09).

Michael's FLY124: NOT RUN — no FLY124 support was measured in this round (5 of 5 missing).

## Proxy `ltas_abs_db` per support (dB, mic 0, absolute level)

| group | support | ltas_abs_db | seed spread | mr_ltas | level offset |
|---|---|---:|---:|---:|---:|
| dregon_cruise | `free-flight_nosource_room2@1512727397.205045+4.000000` | 4.4369 | 0.5396 | 5.7455 | -4.4188 |
| dregon_cruise | `hovering_nosource_room2@1511903905.394490+4.000000` | 3.9474 | 0.6458 | 3.9541 | -3.9474 |
| dregon_cruise | `updown_nosource_room2@1511903578.348311+4.000000` | 5.4205 | 0.6689 | 5.8418 | -5.4205 |
| dregon_cruise | `rectangle_nosource_room2@1511905725.952559+4.000000` | 2.7926 | 0.6296 | 2.3255 | -2.3799 |
| dregon_cruise | `spinning_nosource_room2@1511905200.978012+4.000000` | 3.0967 | 0.5966 | 2.9096 | -2.8681 |

dregon_cruise: mean 3.9388 dB (spread 2.6278 over 5 supports) against the closure-0.7 gate 1.9786 dB = 2.4308 - 0.7 x (2.4308 - 1.7848); margin -1.9602 dB. Secondary `mr_ltas` 4.1553 dB (reported, never decisive).

michaels_cruise: mean nan dB (spread nan over 0 supports) against the closure-0.7 gate 1.2197 dB = 1.5333 - 0.7 x (1.5333 - 1.0852); margin — dB. Secondary `mr_ltas` — dB (reported, never decisive).

## Likelihood: composite risk against the speed-matched stationary oracle

Unavailable: no Michael's cruise likelihood cell was built

## Protocol provenance

Reproduced from the previous campaign (sources in `src/experiments/noise_model/gates.py`): the frozen scorer `hppnet_l2_r2_s0/best` with its SHA-256 verified before anything is scored; `revised_eval.pit_mae` over all eight microphones on the pre-registered raw-telemetry regime support; the five DREGON room-2 4 s cruise windows and the five FLY124 windows the frozen evaluator resolved; the four frozen render seeds `[2001, 2002, 2003, 2004]`; `per_window`/`per_recording` aggregation; the DREGON target as `real + 0.7 x (baseline - real)` on the frozen v2 scalars; the recording-level one-sided 95 % interval of `revised_eval.cluster_interval` (t bound and 20000-draw cluster bootstrap at seed 0, conservative side); Michael's ratio of equally weighted per-regime MAEs; `ltas_abs_db` as `revised_eval.ltas_deviation_db` mean_abs_db on mic 0 with no normalisation; the composite risk of `marginal_frame_nll` + `FrameScore` + `composite_score`; and the `oracle_np` definition and reference segments of `results/noise_v2/short_whittle/short_whittle.json`.

Re-derived, because the source is not in the repository or does not exist yet: (a) the DREGON paired `baseline - candidate` improvement interval needs the per-recording baseline MAEs of the gitignored `results/revised_phase/baseline_v2/calibration.json`, so the one-sided 95 % recording-level instrument is applied to the candidate's own cluster mean against the frozen scalar target instead; (b) the per-band (300 Hz) oracle risk, because `short_whittle.json` records the oracle over the full 30-7900 Hz band only — it is recomputed here from the same oracle periodogram and the full-band value is cross-checked against the recorded number; (c) v2 has no fitted standby/ramp model, so the FLY125 cruise fit drives those two regimes on their own real carriers through the renderer's trajectory sampler (approved decision 3), where the old campaign used the manifest `baseline_map` cross-recording extrapolation of the FLY125 exports.
