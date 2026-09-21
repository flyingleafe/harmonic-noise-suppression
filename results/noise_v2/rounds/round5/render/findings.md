# Noise model v2 — round 5 gate score (FAIL)

Arm: v2 — dregon: v2 DREGON room-2 floor fit with the bench comb (mean of Motor1-4 @70)
Render seeds [2001, 2002, 2003, 2004], 8 mics, git `c4e623186b5b895f0f83b420c4c5c830f36f00e9`.

## Gate verdicts

| gate | verdict | number | threshold |
|---|---|---|---|
| HPPNet DREGON cruise | FAIL | 1.820065 rev/s (95 % upper 2.158331) | <= 1.897063 |
| HPPNet Michael's ratio | NOT RUN | no FLY124 support measured (5 missing) | <= 1.05 (3.177994 rev/s) |
| proxy `ltas_abs_db` dregon_cruise | FAIL | 2.0298 dB (spread 2.6665) | <= 1.9786 dB |
| proxy `ltas_abs_db` michaels_cruise | FAIL | nan dB (spread nan) | <= 1.2197 dB |
| likelihood | FAIL | no Michael's cruise likelihood cell was built | — |

## HPPNet PIT MAE per support (rev/s)

| rig | support | regime | real | candidate | seed spread |
|---|---|---|---:|---:|---:|
| dregon | `free-flight_nosource_room2@1512727397.205045+4.000000` | cruise | 0.638129 | 1.423033 | 0.160554 |
| dregon | `hovering_nosource_room2@1511903905.394490+4.000000` | cruise | 0.888408 | 1.648841 | 0.102950 |
| dregon | `updown_nosource_room2@1511903578.348311+4.000000` | cruise | 1.694930 | 1.931465 | 0.135731 |
| dregon | `rectangle_nosource_room2@1511905725.952559+4.000000` | cruise | 1.685142 | 2.364579 | 0.131817 |
| dregon | `spinning_nosource_room2@1511905200.978012+4.000000` | cruise | 1.186931 | 1.732407 | 0.047572 |

DREGON cruise: candidate mean 1.820065 rev/s over 5 recording-level clusters, one-sided 95 % interval [1.481799, 2.158331] (t and 20000-draw cluster bootstrap, conservative), against the frozen target 1.897063 = 1.218708 + 0.7 x (2.187786 - 1.218708). Margin 0.076998 rev/s.

Real arm reproduction: measured 1.218708 rev/s against the frozen 1.218708 (relative 1.926e-08).

Michael's FLY124: NOT RUN — no FLY124 support was measured in this round (5 of 5 missing).

## Proxy `ltas_abs_db` per support (dB, mic 0, absolute level)

| group | support | ltas_abs_db | seed spread | mr_ltas | level offset |
|---|---|---:|---:|---:|---:|
| dregon_cruise | `free-flight_nosource_room2@1512727397.205045+4.000000` | 2.4631 | 0.0462 | 2.4374 | -2.4631 |
| dregon_cruise | `hovering_nosource_room2@1511903905.394490+4.000000` | 1.8790 | 0.0471 | 1.2956 | -1.8790 |
| dregon_cruise | `updown_nosource_room2@1511903578.348311+4.000000` | 3.6168 | 0.1067 | 2.4552 | -3.6168 |
| dregon_cruise | `rectangle_nosource_room2@1511905725.952559+4.000000` | 1.2399 | 0.0628 | 2.5239 | -0.1815 |
| dregon_cruise | `spinning_nosource_room2@1511905200.978012+4.000000` | 0.9503 | 0.0578 | 1.8075 | -0.7381 |

dregon_cruise: mean 2.0298 dB (spread 2.6665 over 5 supports) against the closure-0.7 gate 1.9786 dB = 2.4308 - 0.7 x (2.4308 - 1.7848); margin -0.0512 dB. Secondary `mr_ltas` 2.1039 dB (reported, never decisive).

michaels_cruise: mean nan dB (spread nan over 0 supports) against the closure-0.7 gate 1.2197 dB = 1.5333 - 0.7 x (1.5333 - 1.0852); margin — dB. Secondary `mr_ltas` — dB (reported, never decisive).

## Likelihood: composite risk against the speed-matched stationary oracle

Unavailable: no Michael's cruise likelihood cell was built

## Protocol provenance

Reproduced from the previous campaign (sources in `src/experiments/noise_model/gates.py`): the frozen scorer `hppnet_l2_r2_s0/best` with its SHA-256 verified before anything is scored; `revised_eval.pit_mae` over all eight microphones on the pre-registered raw-telemetry regime support; the five DREGON room-2 4 s cruise windows and the five FLY124 windows the frozen evaluator resolved; the four frozen render seeds `[2001, 2002, 2003, 2004]`; `per_window`/`per_recording` aggregation; the DREGON target as `real + 0.7 x (baseline - real)` on the frozen v2 scalars; the recording-level one-sided 95 % interval of `revised_eval.cluster_interval` (t bound and 20000-draw cluster bootstrap at seed 0, conservative side); Michael's ratio of equally weighted per-regime MAEs; `ltas_abs_db` as `revised_eval.ltas_deviation_db` mean_abs_db on mic 0 with no normalisation; the composite risk of `marginal_frame_nll` + `FrameScore` + `composite_score`; and the `oracle_np` definition and reference segments of `results/noise_v2/short_whittle/short_whittle.json`.

Re-derived, because the source is not in the repository or does not exist yet: (a) the DREGON paired `baseline - candidate` improvement interval needs the per-recording baseline MAEs of the gitignored `results/revised_phase/baseline_v2/calibration.json`, so the one-sided 95 % recording-level instrument is applied to the candidate's own cluster mean against the frozen scalar target instead; (b) the per-band (300 Hz) oracle risk, because `short_whittle.json` records the oracle over the full 30-7900 Hz band only — it is recomputed here from the same oracle periodogram and the full-band value is cross-checked against the recorded number; (c) v2 has no fitted standby/ramp model, so the FLY125 cruise fit drives those two regimes on their own real carriers through the renderer's trajectory sampler (approved decision 3), where the old campaign used the manifest `baseline_map` cross-recording extrapolation of the FLY125 exports.
