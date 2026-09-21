# Noise model v2 — round 5 gate score: PARITY PASS / STRETCH PASS

Top-level pass (legacy parity on both HPPNet gates): **true**. Stretch (frozen 0.70-gap DREGON target): **true**. All three frozen gates: **false**. Record `results/noise_v2/rounds/round5.json`, git `06897cd49301`.

## Candidates

| candidate | rig(s) | fit JSON | converged | fit wall (s) | render+probe job |
|---|---|---|---|---:|---|
| `dregon_v2_profile_pin` | dregon | `results/noise_v2/rounds/round5/calibration/dregon_room2_floor__flight_profile_pin.json` | **NO** (none, |grad| 407) | 4928 | `—` |
| `michaels_v2_r3_regimes` | michaels | `results/noise_v2/rounds/round3/fits/michaels_fly125_standby__flight.json + results/noise_v2/rounds/round3/fits/michaels_fly125_cruise__flight.json` | **NO** ({'standby': 'none', 'cruise': 'none'}, |grad| 1174) | 14781 | `nv2-r3-score-michaels-re-ac8bc5` |

* `dregon_v2_profile_pin` — the R5 flight_profile fit (bench-frozen dynamics, free per-order comb) with the +3.75 dB calibration offset of round5/calibration/pin.json applied to profile_db; scored on this laptop, no job id
* `michaels_v2_r3_regimes` — R3's per-regime Michael's incumbent, UNCHANGED and not re-scored: the arm record is round3/render/arm_michaels_v2_regimes.json verbatim

## The two bars

| rig | gate quantity | number | PARITY bar | parity | STRETCH bar | stretch |
|---|---|---:|---:|---|---:|---|
| dregon | five-recording cruise PIT MAE mean, one-sided 95 % upper bound beside it | 1.820065 | 2.187786 | **PASS** (margin +0.367721) | 1.897063 | **PASS** (margin +0.076998) |
| michaels | equal-regime mean PIT MAE over standby/ramp/cruise (ratio of means) | 1.621670 | 3.177994 | **PASS** (margin +1.556325) | — | — (margin —) |

PARITY is the legacy previous best: DREGON synthetic cruise PIT MAE 2.187786 rev/s (real arm 1.218708), Michael's 1.05 x 3.026661 = 3.177994 rev/s. STRETCH exists only on DREGON: 1.897063 = real + 0.7 x (baseline - real); Michael's frozen gate IS its parity bar.

## Round 3 → round 5

| quantity | unit | round 3 | round 5 | Δ (R5 − R3) | direction | moved |
|---|---|---:|---:|---:|---|---|
| dregon cruise PIT MAE (5-recording mean) | rev/s | 71.866599 | 1.820065 | -70.046534 | lower is better | **better** |
| dregon cruise PIT MAE (95 % upper) | rev/s | 75.029755 | 2.158331 | -72.871424 | lower is better | **better** |
| michaels equal-regime PIT MAE | rev/s | 1.621670 | 1.621670 | +0.000000 | lower is better | — |
| michaels ratio vs the legacy regime mean | x | 0.5358 | 0.5358 | +0.0000 | lower is better | — |
| proxy ltas_abs_db, dregon cruise | dB | 3.7480 | 2.0298 | -1.7182 | lower is better | **better** |
| proxy ltas_abs_db, michaels cruise | dB | 1.2816 | 1.2816 | +0.0000 | lower is better | — |
| mr_ltas, dregon cruise (report-only) | — | 2.6592 | 2.1039 | -0.5553 | lower is better | **better** |
| mr_ltas, michaels cruise (report-only) | — | 1.4615 | 1.4615 | +0.0000 | lower is better | — |
| likelihood comb-band margin vs oracle | nats/s | -1749.799951 | -1749.799951 | +0.000000 | higher is better | — |

Both sides are read verbatim out of `results/noise_v2/rounds/round3.json` (git `3fcae519cddb`, candidates `michaels_v2_r3_regimes`, `michaels_v2_r3_all`, `dregon_v2_r3_floor_lowk`) and this record; a dash is a quantity one of the two rounds did not measure, never a substituted number.

## Per-candidate numbers

| candidate | HPPNet | proxy `ltas_abs_db` (gate) | `mr_ltas` | likelihood comb margin |
|---|---:|---:|---:|---:|
| `dregon_v2_profile_pin` | 1.820065 rev/s (95 % upper 2.158331) | 2.0298 (1.9786) | 2.1039 | not run |
| `michaels_v2_r3_regimes` | 1.621670 rev/s (ratio 0.5358) | 1.2816 (1.2197) | 1.4615 | -1,749.8000 nats/s |


## Gate verdicts

| gate | verdict | number | threshold |
|---|---|---|---|
| HPPNet DREGON cruise | FAIL | 1.820065 rev/s (95 % upper 2.158331) | <= 1.897063 |
| HPPNet Michael's ratio | PASS | 0.535795 (1.621670 rev/s) | <= 1.05 (3.177994 rev/s) |
| proxy `ltas_abs_db` dregon_cruise | FAIL | 2.0298 dB (spread 2.6665) | <= 1.9786 dB |
| proxy `ltas_abs_db` michaels_cruise | FAIL | 1.2816 dB (spread 1.6179) | <= 1.2197 dB |
| likelihood comb (decisive) | PASS | model -373,666.3040 nats/s | oracle -371,916.5041, margin -1,749.8000 |
| likelihood floor | report | model -8,345.2279 nats/s | oracle -8,713.2663, margin +368.0384 |
| likelihood full | report | model -382,011.5319 nats/s | oracle -380,629.7704, margin -1,381.7615 |

## HPPNet PIT MAE per support (rev/s)

| rig | support | regime | real | candidate | seed spread |
|---|---|---|---:|---:|---:|
| dregon | `free-flight_nosource_room2@1512727397.205045+4.000000` | cruise | 0.638129 | 1.423033 | 0.160554 |
| dregon | `hovering_nosource_room2@1511903905.394490+4.000000` | cruise | 0.888408 | 1.648841 | 0.102950 |
| dregon | `updown_nosource_room2@1511903578.348311+4.000000` | cruise | 1.694930 | 1.931465 | 0.135731 |
| dregon | `rectangle_nosource_room2@1511905725.952559+4.000000` | cruise | 1.685142 | 2.364579 | 0.131817 |
| dregon | `spinning_nosource_room2@1511905200.978012+4.000000` | cruise | 1.186931 | 1.732407 | 0.047572 |
| michaels | `FLY124@8.000000+8.000000` | standby | 0.398474 | 0.599740 | 0.465573 |
| michaels | `FLY124@16.000000+8.000000` | standby | 0.289978 | 0.971028 | 1.790409 |
| michaels | `FLY124@27.680000+8.000000` | ramp | 3.171148 | 3.267119 | 0.325035 |
| michaels | `FLY124@40.000000+8.000000` | cruise | 0.884854 | 1.076823 | 0.200746 |
| michaels | `FLY124@56.000000+8.000000` | cruise | 0.290248 | 0.548191 | 0.143747 |

DREGON cruise: candidate mean 1.820065 rev/s over 5 recording-level clusters, one-sided 95 % interval [1.481799, 2.158331] (t and 20000-draw cluster bootstrap, conservative), against the frozen target 1.897063 = 1.218708 + 0.7 x (2.187786 - 1.218708). Margin 0.076998 rev/s.

Real arm reproduction: measured 1.218708 rev/s against the frozen 1.218708 (relative 1.926e-08).

Michael's FLY124 per regime (rev/s):

| regime | blocks | candidate | frozen baseline | ratio |
|---|---:|---:|---:|---:|
| standby | 2 | 0.785384 | 0.317103 | 2.4768 |
| ramp | 1 | 3.267119 | 8.007098 | 0.4080 |
| cruise | 2 | 0.812507 | 0.755783 | 1.0751 |

Equal-regime mean 1.621670 rev/s against the frozen baseline 3.026661; ratio 0.535795 against the 1.05 bound (ratio of equally weighted per-regime MAEs (NOT a mean of ratios)).

## Proxy `ltas_abs_db` per support (dB, mic 0, absolute level)

| group | support | ltas_abs_db | seed spread | mr_ltas | level offset |
|---|---|---:|---:|---:|---:|
| dregon_cruise | `free-flight_nosource_room2@1512727397.205045+4.000000` | 2.4631 | 0.0462 | 2.4374 | -2.4631 |
| dregon_cruise | `hovering_nosource_room2@1511903905.394490+4.000000` | 1.8790 | 0.0471 | 1.2956 | -1.8790 |
| dregon_cruise | `updown_nosource_room2@1511903578.348311+4.000000` | 3.6168 | 0.1067 | 2.4552 | -3.6168 |
| dregon_cruise | `rectangle_nosource_room2@1511905725.952559+4.000000` | 1.2399 | 0.0628 | 2.5239 | -0.1815 |
| dregon_cruise | `spinning_nosource_room2@1511905200.978012+4.000000` | 0.9503 | 0.0578 | 1.8075 | -0.7381 |
| michaels_cruise | `FLY124@40.000000+8.000000` | 0.4727 | 0.0720 | 0.9711 | -0.2345 |
| michaels_cruise | `FLY124@56.000000+8.000000` | 2.0906 | 0.2212 | 1.9519 | +2.0906 |

dregon_cruise: mean 2.0298 dB (spread 2.6665 over 5 supports) against the closure-0.7 gate 1.9786 dB = 2.4308 - 0.7 x (2.4308 - 1.7848); margin -0.0512 dB. Secondary `mr_ltas` 2.1039 dB (reported, never decisive).

michaels_cruise: mean 1.2816 dB (spread 1.6179 over 2 supports) against the closure-0.7 gate 1.2197 dB = 1.5333 - 0.7 x (1.5333 - 1.0852); margin -0.0620 dB. Secondary `mr_ltas` 1.4615 dB (reported, never decisive).

## Likelihood: composite risk against the speed-matched stationary oracle

NFFT 2048 / hop 512, pooled over 2 Michael's FLY124 cruise supports, band split at 300 Hz.

| band | Hz | model (nats/s) | oracle (nats/s) | margin | below oracle |
|---|---|---:|---:|---:|---|
| comb | 300-7900 | -373,666.3040 | -371,916.5041 | -1,749.8000 | True |
| floor | 30-300 | -8,345.2279 | -8,713.2663 | +368.0384 | False |
| full | 30-7900 | -382,011.5319 | -380,629.7704 | -1,381.7615 | True |

Frozen comb-band margin: -1,749.8000 nats/s (R1 freezes the comb-band model-minus-oracle margin; later rounds are held to it).

Oracle reproduction: recomputed full-band pooled oracle -380,629.7704 nats/s against the recorded -380,629.7704 nats/s of `results/noise_v2/short_whittle/short_whittle.json` (relative 0.000e+00).

## Protocol provenance

Reproduced from the previous campaign (sources in `src/experiments/noise_model/gates.py`): the frozen scorer `hppnet_l2_r2_s0/best` with its SHA-256 verified before anything is scored; `revised_eval.pit_mae` over all eight microphones on the pre-registered raw-telemetry regime support; the five DREGON room-2 4 s cruise windows and the five FLY124 windows the frozen evaluator resolved; the four frozen render seeds `[2001, 2002, 2003, 2004]`; `per_window`/`per_recording` aggregation; the DREGON target as `real + 0.7 x (baseline - real)` on the frozen v2 scalars; the recording-level one-sided 95 % interval of `revised_eval.cluster_interval` (t bound and 20000-draw cluster bootstrap at seed 0, conservative side); Michael's ratio of equally weighted per-regime MAEs; `ltas_abs_db` as `revised_eval.ltas_deviation_db` mean_abs_db on mic 0 with no normalisation; the composite risk of `marginal_frame_nll` + `FrameScore` + `composite_score`; and the `oracle_np` definition and reference segments of `results/noise_v2/short_whittle/short_whittle.json`.

Re-derived, because the source is not in the repository or does not exist yet: (a) the DREGON paired `baseline - candidate` improvement interval needs the per-recording baseline MAEs of the gitignored `results/revised_phase/baseline_v2/calibration.json`, so the one-sided 95 % recording-level instrument is applied to the candidate's own cluster mean against the frozen scalar target instead; (b) the per-band (300 Hz) oracle risk, because `short_whittle.json` records the oracle over the full 30-7900 Hz band only — it is recomputed here from the same oracle periodogram and the full-band value is cross-checked against the recorded number; (c) v2 has no fitted standby/ramp model, so the FLY125 cruise fit drives those two regimes on their own real carriers through the renderer's trajectory sampler (approved decision 3), where the old campaign used the manifest `baseline_map` cross-recording extrapolation of the FLY125 exports.

## Legacy smoke comparability (recorded earlier, NOT re-run)

| legacy replay (`noise-v2-r1-legacy-smoke-453b44`) | observed | frozen scalar | relative delta |
|---|---:|---:|---:|
| DREGON cruise synthetic PIT MAE | 2.187786 | 2.187786 | 4.79e-08 |
| Michael's equal-regime synthetic PIT MAE | 3.199486 | 3.026661 | 5.71e-02 |

Status `not_comparable`: The gitignored baseline calibration that selects the old export family and per-recording route was unavailable. The DREGON number happens to reproduce its frozen scalar, but the Michael's replay does not; because the inputs are not proven identical, the 2% legacy-smoke criterion is intentionally not applied.

## Exact blocker

none: every gate of this round was evaluated on rendered candidate audio and the frozen probe.

## Provenance

The frozen support identities, render seeds, one-sided interval, regime-mean ratio, proxy thresholds, probe checkpoint digest and likelihood/oracle definition are all read from `src/experiments/noise_model/gates.py`; this record adds no scalar of its own. Probe `hppnet_l2_r2_s0/best`, sha256 `6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1`, verified in-job: **true** (the scoring pass dies on a digest mismatch, so a scored arm cannot exist without the match). Render seeds [2001, 2002, 2003, 2004], 8 microphones. Record git `06897cd49301`.

| candidate | rig(s) | fit | fit git | converged | job | audio |
|---|---|---|---|---|---|---|
| `dregon_v2_profile_pin` | dregon | `results/noise_v2/rounds/round5/calibration/dregon_room2_floor__flight_profile_pin.json` | `c4e623186b5b` | **NO** | `local` | `—` |
| `michaels_v2_r3_regimes` | michaels | `results/noise_v2/rounds/round3/fits/michaels_fly125_standby__flight.json + results/noise_v2/rounds/round3/fits/michaels_fly125_cruise__flight.json` | `d9bd19401caf` | **NO** | `nv2-r3-score-michaels-re-ac8bc5` | `—` |

* `dregon_v2_profile_pin` — arm record `results/noise_v2/rounds/round5/render/arm_dregon_v2_profile_pin.json`, render wall 85 s, 1 rig(s) dregon, primary for dregon.
* `michaels_v2_r3_regimes` — arm record `results/noise_v2/rounds/round3/render/arm_michaels_v2_regimes.json`, render wall 433 s, 1 rig(s) michaels, primary for michaels.

Support availability from `results/noise_v2/rounds/round2/supports/index.json` (updated 2026-09-18T11:41:51+0100): dregon-bench 21/21, bench-points 135/135, michaels-all 10/10, dregon-floor 10/10.
