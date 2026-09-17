# Noise model v2 — round 1 gate score: PARITY FAIL / STRETCH FAIL

Top-level pass (legacy parity on both HPPNet gates): **false**. Stretch (frozen 0.70-gap DREGON target): **false**. All three frozen gates: **false**. Record `results/noise_v2/rounds/round1.json`, git `463fd10b993b`.

## Candidates

| candidate | rig(s) | fit JSON | converged | fit wall (s) | render+probe job |
|---|---|---|---|---:|---|
| `michaels_v2_fly125cruise` | michaels | `results/noise_v2/rounds/round1/fits/michaels_fly125_cruise__flight.json` | **NO** (none, |grad| 8555) | 6470 | `nv2-r1-score-michaels2-c3bbbd` |

## The two bars

| rig | gate quantity | number | PARITY bar | parity | STRETCH bar | stretch |
|---|---|---:|---:|---|---:|---|
| dregon | five-recording cruise PIT MAE mean, one-sided 95 % upper bound beside it | not run | 2.187786 | — (margin —) | 1.897063 | — (margin —) |
| michaels | equal-regime mean PIT MAE over standby/ramp/cruise (ratio of means) | 16.799053 | 3.177994 | **FAIL** (margin -13.621059) | — | — (margin —) |

PARITY is the legacy previous best: DREGON synthetic cruise PIT MAE 2.187786 rev/s (real arm 1.218708), Michael's 1.05 x 3.026661 = 3.177994 rev/s. STRETCH exists only on DREGON: 1.897063 = real + 0.7 x (baseline - real); Michael's frozen gate IS its parity bar.

## Per-candidate numbers

| candidate | HPPNet | proxy `ltas_abs_db` (gate) | `mr_ltas` | likelihood comb margin |
|---|---:|---:|---:|---:|
| `michaels_v2_fly125cruise` | 16.799053 rev/s (ratio 5.5504) | 1.4156 (1.2197) | 1.6456 | -1,497.6146 nats/s |

Rendered audio of `michaels_v2_fly125cruise` (uncommitted): `s3://omnirun-artifacts/nv2-r1-score-michaels2-c3bbbd/outputs/results/noise_v2/rounds/round1/render/audio/michaels_v2` (job `nv2-r1-score-michaels2-c3bbbd`, written to `results/noise_v2/rounds/round1/render/audio/michaels_v2` in the job's worktree).

## Gate verdicts

| gate | verdict | number | threshold |
|---|---|---|---|
| HPPNet DREGON cruise | NOT RUN | no DREGON support measured (5 missing) | <= 1.897063 |
| HPPNet Michael's ratio | FAIL | 5.550358 (16.799053 rev/s) | <= 1.05 (3.177994 rev/s) |
| proxy `ltas_abs_db` michaels_cruise | FAIL | 1.4156 dB (spread 1.6608) | <= 1.2197 dB |
| likelihood comb (decisive) | PASS | model -373,414.1187 nats/s | oracle -371,916.5041, margin -1,497.6146 |
| likelihood floor | report | model -8,487.7119 nats/s | oracle -8,713.2663, margin +225.5544 |
| likelihood full | report | model -381,901.8305 nats/s | oracle -380,629.7704, margin -1,272.0601 |

## HPPNet PIT MAE per support (rev/s)

| rig | support | regime | real | candidate | seed spread |
|---|---|---|---:|---:|---:|
| michaels | `FLY124@8.000000+8.000000` | standby | 0.398474 | 32.640554 | 1.259826 |
| michaels | `FLY124@16.000000+8.000000` | standby | 0.289978 | 32.623536 | 1.072229 |
| michaels | `FLY124@27.680000+8.000000` | ramp | 3.171148 | 16.955751 | 1.930254 |
| michaels | `FLY124@40.000000+8.000000` | cruise | 0.884854 | 1.108014 | 0.068582 |
| michaels | `FLY124@56.000000+8.000000` | cruise | 0.290248 | 0.510712 | 0.018152 |

DREGON cruise: NOT RUN — no DREGON support was measured in this round (5 of 5 missing), so neither the cluster mean nor its one-sided 95 % bound exists to put against the frozen target 1.897063 rev/s.

Michael's FLY124 per regime (rev/s):

| regime | blocks | candidate | frozen baseline | ratio |
|---|---:|---:|---:|---:|
| standby | 2 | 32.632045 | 0.317103 | 102.9069 |
| ramp | 1 | 16.955751 | 8.007098 | 2.1176 |
| cruise | 2 | 0.809363 | 0.755783 | 1.0709 |

Equal-regime mean 16.799053 rev/s against the frozen baseline 3.026661; ratio 5.550358 against the 1.05 bound (ratio of equally weighted per-regime MAEs (NOT a mean of ratios)).

## Proxy `ltas_abs_db` per support (dB, mic 0, absolute level)

| group | support | ltas_abs_db | seed spread | mr_ltas | level offset |
|---|---|---:|---:|---:|---:|
| michaels_cruise | `FLY124@40.000000+8.000000` | 0.5852 | 0.0658 | 1.2045 | +0.1672 |
| michaels_cruise | `FLY124@56.000000+8.000000` | 2.2460 | 0.0873 | 2.0867 | +2.2460 |

michaels_cruise: mean 1.4156 dB (spread 1.6608 over 2 supports) against the closure-0.7 gate 1.2197 dB = 1.5333 - 0.7 x (1.5333 - 1.0852); margin -0.1959 dB. Secondary `mr_ltas` 1.6456 dB (reported, never decisive).

## Likelihood: composite risk against the speed-matched stationary oracle

NFFT 2048 / hop 512, pooled over 2 Michael's FLY124 cruise supports, band split at 300 Hz.

| band | Hz | model (nats/s) | oracle (nats/s) | margin | below oracle |
|---|---|---:|---:|---:|---|
| comb | 300-7900 | -373,414.1187 | -371,916.5041 | -1,497.6146 | True |
| floor | 30-300 | -8,487.7119 | -8,713.2663 | +225.5544 | False |
| full | 30-7900 | -381,901.8305 | -380,629.7704 | -1,272.0601 | True |

Frozen comb-band margin: -1,497.6146 nats/s (R1 freezes the comb-band model-minus-oracle margin; later rounds are held to it).

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

The DREGON arm is NOT scored: the floor-only flight fit results/noise_v2/rounds/round1/fits/dregon_room2_floor__flight_floor_only.json does not exist yet (owner R1Fit, uni-cpu job nv2-r1-dregon-floor-fffda7, submitted 2026-09-17T19:41Z from ebaecd66, still in its optimiser phase). Without it there is no DREGON candidate audio, so the HPPNet DREGON cruise number, the proxy dregon_cruise group and therefore the joint frozen HPPNet gate are not_run; the Michael's arm is fully scored. The likelihood gate needs only Michael's cruise cells and is evaluated.

## Provenance

The frozen support identities, one-sided interval, regime-mean ratio, proxy thresholds, HPPNet checkpoint hash, and likelihood/oracle definition are reproduced in `src/experiments/noise_model/gates.py`. The support availability evidence is committed in `b5d369de`. This score record intentionally reports `not_run` rather than substituting legacy-model measurements for absent R1 fits.

### Addendum — this scoring pass

The `Provenance` section above is kept verbatim from the earlier, unscored record; the gates it calls `not_run` are the ones measured here, and whatever is still not run is named in `Exact blocker`. This pass:

* `michaels_v2_fly125cruise` — fit git `8093285f352cd33e31e56b424bb64c5dbe35f06d`, render/probe job `nv2-r1-score-michaels2-c3bbbd`, arm record `results/noise_v2/rounds/round1/render/arm_michaels_v2.json`, render wall 253 s.

### Addendum — this scoring pass

The `Provenance` section above is kept verbatim from the earlier, unscored record; the gates it calls `not_run` are the ones measured here, and whatever is still not run is named in `Exact blocker`. This pass:

* `michaels_v2_fly125cruise` — fit git `8093285f352c`, render/probe job `nv2-r1-score-michaels2-c3bbbd`, arm record `results/noise_v2/rounds/round1/render/arm_michaels_v2.json`, render wall 253 s.
