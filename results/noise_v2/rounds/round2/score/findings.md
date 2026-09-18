# Noise model v2 — round 2 gate score: PARITY FAIL / STRETCH FAIL

Top-level pass (legacy parity on both HPPNet gates): **false**. Stretch (frozen 0.70-gap DREGON target): **false**. All three frozen gates: **false**. Record `results/noise_v2/rounds/round2.json`, git `12546c11e8a2`.

## Candidates

| candidate | rig(s) | fit JSON | converged | fit wall (s) | render+probe job |
|---|---|---|---|---:|---|
| `dregon_v2_r2_floor_benchcomb` | dregon | `results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only.json` | **NO** (none, |grad| 780) | 2480 | `nv2-r2-score-dregon-3f8397` |

* `dregon_v2_r2_floor_benchcomb` — the fit it renders from is NOT converged (which_converged `none`, |grad| 780.1, L-BFGS restart gain 6.94e-03 nats/cell against the 1e-4 tolerance, 2480 s wall), and its comb is frozen from four R2 bench fits that are themselves 0/4 converged (|grad| 359.5 / 1539.2 / 1898.9 / 3792.9 on Motor3/4/2/1 @70). The comb-mean check reproduces the declared log-mean to 0.0 relative, so this row is a faithful score OF THAT FIT — not a converged-model parity result.

## The two bars

| rig | gate quantity | number | PARITY bar | parity | STRETCH bar | stretch |
|---|---|---:|---:|---|---:|---|
| dregon | five-recording cruise PIT MAE mean, one-sided 95 % upper bound beside it | 75.093968 | 2.187786 | **FAIL** (margin -72.906182) | 1.897063 | **FAIL** (margin -73.196905) |
| michaels | equal-regime mean PIT MAE over standby/ramp/cruise (ratio of means) | NOT RUN (fit in flight: nv2-r2-michaels-flight-cb36cd) | 3.177994 | — (margin —) | — | — (margin —) |

PARITY is the legacy previous best: DREGON synthetic cruise PIT MAE 2.187786 rev/s (real arm 1.218708), Michael's 1.05 x 3.026661 = 3.177994 rev/s. STRETCH exists only on DREGON: 1.897063 = real + 0.7 x (baseline - real); Michael's frozen gate IS its parity bar.

`michaels`: NOT RUN (fit in flight: nv2-r2-michaels-flight-cb36cd). Every michaels cell of this record is a dash.

## Round 1 → round 2

| quantity | unit | round 1 | round 2 | Δ (R2 − R1) | direction | moved |
|---|---|---:|---:|---:|---|---|
| dregon cruise PIT MAE (5-recording mean) | rev/s | 78.803342 | 75.093968 | -3.709374 | lower is better | **better** |
| dregon cruise PIT MAE (95 % upper) | rev/s | 79.482310 | 75.551936 | -3.930374 | lower is better | **better** |
| michaels equal-regime PIT MAE | rev/s | 16.799053 | — | — | lower is better | — |
| michaels ratio vs the legacy regime mean | x | 5.5504 | — | — | lower is better | — |
| proxy ltas_abs_db, dregon cruise | dB | 3.7518 | 3.6328 | -0.1190 | lower is better | **better** |
| proxy ltas_abs_db, michaels cruise | dB | 1.4156 | — | — | lower is better | — |
| mr_ltas, dregon cruise (report-only) | — | 2.6359 | 2.6285 | -0.0074 | lower is better | **better** |
| mr_ltas, michaels cruise (report-only) | — | 1.6456 | — | — | lower is better | — |
| likelihood comb-band margin vs oracle | nats/s | -1497.614580 | — | — | higher is better | — |

Both sides are read verbatim out of `results/noise_v2/rounds/round1.json` (git `066651e8eb31`, candidates `dregon_v2_floor_benchcomb`, `michaels_v2_fly125cruise`, `michaels_v2_retry_pinned_lam`) and this record; a dash is a quantity one of the two rounds did not measure, never a substituted number.

## Per-candidate numbers

| candidate | HPPNet | proxy `ltas_abs_db` (gate) | `mr_ltas` | likelihood comb margin |
|---|---:|---:|---:|---:|
| `dregon_v2_r2_floor_benchcomb` | 75.093968 rev/s (95 % upper 75.551936) | 3.6328 (1.9786) | 2.6285 | not run |

Rendered audio of `dregon_v2_r2_floor_benchcomb` (uncommitted): `s3://omnirun-artifacts/nv2-r2-score-dregon-3f8397/outputs/results/noise_v2/rounds/round2/render/audio/dregon_v2` (job `nv2-r2-score-dregon-3f8397`, written to `results/noise_v2/rounds/round2/render/audio/dregon_v2` in the job's worktree).

## Gate verdicts

| gate | verdict | number | threshold |
|---|---|---|---|
| HPPNet DREGON cruise | FAIL | 75.093968 rev/s (95 % upper 75.551936) | <= 1.897063 |
| HPPNet Michael's ratio | NOT RUN | no FLY124 support measured (5 missing) | <= 1.05 (3.177994 rev/s) |
| proxy `ltas_abs_db` dregon_cruise | FAIL | 3.6328 dB (spread 3.2145) | <= 1.9786 dB |
| likelihood | FAIL | no Michael's arm was scored, so no likelihood cell exists | — |

## HPPNet PIT MAE per support (rev/s)

| rig | support | regime | real | candidate | seed spread |
|---|---|---|---:|---:|---:|
| dregon | `free-flight_nosource_room2@1512727397.205045+4.000000` | cruise | 0.638129 | 74.576455 | 11.065307 |
| dregon | `hovering_nosource_room2@1511903905.394490+4.000000` | cruise | 0.888408 | 74.693775 | 14.227503 |
| dregon | `updown_nosource_room2@1511903578.348311+4.000000` | cruise | 1.694930 | 75.331026 | 11.625884 |
| dregon | `rectangle_nosource_room2@1511905725.952559+4.000000` | cruise | 1.685142 | 75.112452 | 21.271691 |
| dregon | `spinning_nosource_room2@1511905200.978012+4.000000` | cruise | 1.186931 | 75.756130 | 15.566315 |

DREGON cruise: candidate mean 75.093968 rev/s over 5 recording-level clusters, one-sided 95 % interval [74.636000, 75.551936] (t and 20000-draw cluster bootstrap, conservative), against the frozen target 1.897063 = 1.218708 + 0.7 x (2.187786 - 1.218708). Margin -73.196905 rev/s.

Real arm reproduction: measured 1.218708 rev/s against the frozen 1.218708 (relative 2.688e-09).

Michael's FLY124: NOT RUN — no FLY124 support was measured in this round (5 of 5 missing).

## Proxy `ltas_abs_db` per support (dB, mic 0, absolute level)

| group | support | ltas_abs_db | seed spread | mr_ltas | level offset |
|---|---|---:|---:|---:|---:|
| dregon_cruise | `free-flight_nosource_room2@1512727397.205045+4.000000` | 4.2061 | 0.0275 | 3.8344 | -4.2061 |
| dregon_cruise | `hovering_nosource_room2@1511903905.394490+4.000000` | 3.6546 | 0.0322 | 1.9959 | -3.6546 |
| dregon_cruise | `updown_nosource_room2@1511903578.348311+4.000000` | 5.4759 | 0.0134 | 4.0814 | -5.4759 |
| dregon_cruise | `rectangle_nosource_room2@1511905725.952559+4.000000` | 2.2614 | 0.0627 | 1.7502 | -2.0004 |
| dregon_cruise | `spinning_nosource_room2@1511905200.978012+4.000000` | 2.5661 | 0.0676 | 1.4805 | -2.5016 |

dregon_cruise: mean 3.6328 dB (spread 3.2145 over 5 supports) against the closure-0.7 gate 1.9786 dB = 2.4308 - 0.7 x (2.4308 - 1.7848); margin -1.6542 dB. Secondary `mr_ltas` 2.6285 dB (reported, never decisive).

## Likelihood: composite risk against the speed-matched stationary oracle

Unavailable: no Michael's arm was scored, so no likelihood cell exists

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

Michael's arm: NOT RUN (fit in flight: nv2-r2-michaels-flight-cb36cd on uni-cpu, submitted 2026-09-18T07:51:24Z, still in Adam at step 925/1500 when this record was composed), so the Michael's HPPNet row, the michaels_cruise proxy group and the entire likelihood gate are not_run here and the round cannot be a joint verdict. The DREGON side has no blocker: all five frozen room-2 cruise supports were rendered from the R2 floor-only flight fit over all four frozen seeds and probed with the frozen HPPNet checkpoint (sha256 verified in-job) — it FAILS both bars by ~34x, it is not unmeasured. What the DREGON number is NOT: a converged-model result (see the candidate note).

## Provenance

The frozen support identities, render seeds, one-sided interval, regime-mean ratio, proxy thresholds, probe checkpoint digest and likelihood/oracle definition are all read from `src/experiments/noise_model/gates.py`; this record adds no scalar of its own. Probe `hppnet_l2_r2_s0/best`, sha256 `6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1`, verified in-job: **true** (the scoring pass dies on a digest mismatch, so a scored arm cannot exist without the match). Render seeds [2001, 2002, 2003, 2004], 8 microphones. Record git `12546c11e8a2`.

| candidate | rig(s) | fit | fit git | converged | job | audio |
|---|---|---|---|---|---|---|
| `dregon_v2_r2_floor_benchcomb` | dregon | `results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only.json` | `fc18285faea7` | **NO** | `nv2-r2-score-dregon-3f8397` | `s3://omnirun-artifacts/nv2-r2-score-dregon-3f8397/outputs/results/noise_v2/rounds/round2/render/audio/dregon_v2` |

* `dregon_v2_r2_floor_benchcomb` — arm record `results/noise_v2/rounds/round2/render/arm_dregon_v2.json`, render wall 127 s, 1 rig(s) dregon, primary for dregon.

Support availability from `results/noise_v2/rounds/round2/supports/index.json` (updated 2026-09-18T02:25:27+0100): dregon-bench 21/21, bench-points 135/135, michaels-all 10/10.

### Addendum — this scoring pass

The `Provenance` section above is kept verbatim from the earlier, unscored record; the gates it calls `not_run` are the ones measured here, and whatever is still not run is named in `Exact blocker`. This pass:

* `dregon_v2_r2_floor_benchcomb` — fit git `fc18285faea7`, render/probe job `nv2-r2-score-dregon-3f8397`, arm record `results/noise_v2/rounds/round2/render/arm_dregon_v2.json`, render wall 127 s.
