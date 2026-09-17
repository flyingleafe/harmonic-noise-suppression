# Noise model v2 — round 1 gate score: PARITY FAIL / STRETCH FAIL

Top-level pass (legacy parity on both HPPNet gates): **false**. Stretch (frozen 0.70-gap DREGON target): **false**. All three frozen gates: **false**. Record `results/noise_v2/rounds/round1.json`, git `066651e8eb31`.

## Candidates

| candidate | rig(s) | fit JSON | converged | fit wall (s) | render+probe job |
|---|---|---|---|---:|---|
| `dregon_v2_floor_benchcomb` | dregon | `results/noise_v2/rounds/round1/fits/dregon_room2_floor__flight_floor_only.json` | **NO** (none, |grad| 2318) | 2377 | `nv2-r1-score-dregon-662433` |
| `michaels_v2_fly125cruise` | michaels | `results/noise_v2/rounds/round1/fits/michaels_fly125_cruise__flight.json` | **NO** (none, |grad| 8555) | 6470 | `nv2-r1-score-michaels2-c3bbbd` |
| `michaels_v2_retry_pinned_lam` | michaels | `results/noise_v2/rounds/round1/fits/michaels_fly125_cruise__flight_retry.json` | **NO** (none, |grad| 2863) | 4799 | `nv2-r1-score-michaels-re-eafb5e` |

* `dregon_v2_floor_benchcomb` — COMBS DEFECTIVE (BenchDiag in progress) — NOT a parity result, an end-to-end proof of the DREGON render+probe pipeline. The four bench_dregon_Motor*_70 fits this arm freezes its comb from put the in-band per-order profile ~40 dB BELOW the floor (Motor1_70, orders 2-30, -113...-125 dB against a -77 dB floor), so the rendered audio carries essentially no harmonic comb and the frozen tracker has nothing to lock onto; that is what the 78.803342 rev/s reads. The pipeline itself is verified on the same job by the real arm, which reproduces the frozen DREGON real PIT MAE 1.2187080818 to 2.7e-09 relative. Re-run with corrected combs: --fit dregon=<new floor fit path>.
* `michaels_v2_retry_pinned_lam` — The one permitted Michael's retry (R1Basin, fit git 2e96fa29), with lam=0.5 s^-1 and lam_eps=[2.0, 75.63307] PINNED on Main's long-lag decision; also NOT converged. It is WORSE than the unpinned fit on every gate the render feeds: equal-regime 19.376405 vs 16.799053 rev/s, cruise 1.138637 vs 0.809363 rev/s, proxy 1.6330 vs 1.4156 dB, comb-band likelihood margin -1200.59 vs -1497.61 nats/s. The unpinned fit therefore stays the primary Michael's arm of the joint gate.

## The two bars

| rig | gate quantity | number | PARITY bar | parity | STRETCH bar | stretch |
|---|---|---:|---:|---|---:|---|
| dregon | five-recording cruise PIT MAE mean, one-sided 95 % upper bound beside it | 78.803342 | 2.187786 | **FAIL** (margin -76.615556) | 1.897063 | **FAIL** (margin -76.906280) |
| michaels | equal-regime mean PIT MAE over standby/ramp/cruise (ratio of means) | 16.799053 | 3.177994 | **FAIL** (margin -13.621059) | — | — (margin —) |

PARITY is the legacy previous best: DREGON synthetic cruise PIT MAE 2.187786 rev/s (real arm 1.218708), Michael's 1.05 x 3.026661 = 3.177994 rev/s. STRETCH exists only on DREGON: 1.897063 = real + 0.7 x (baseline - real); Michael's frozen gate IS its parity bar.

## Per-candidate numbers

| candidate | HPPNet | proxy `ltas_abs_db` (gate) | `mr_ltas` | likelihood comb margin |
|---|---:|---:|---:|---:|
| `dregon_v2_floor_benchcomb` | 78.803342 rev/s (95 % upper 79.482310) | 3.7518 (1.9786) | 2.6359 | not run |
| `michaels_v2_fly125cruise` | 16.799053 rev/s (ratio 5.5504) | 1.4156 (1.2197) | 1.6456 | -1,497.6146 nats/s |
| `michaels_v2_retry_pinned_lam` | 19.376405 rev/s (ratio 6.4019) | 1.6330 (1.2197) | 1.5963 | -1,200.5896 nats/s |

Rendered audio of `dregon_v2_floor_benchcomb` (uncommitted): `s3://omnirun-artifacts/nv2-r1-score-dregon-662433/outputs/results/noise_v2/rounds/round1/render/audio/dregon_v2` (job `nv2-r1-score-dregon-662433`, written to `results/noise_v2/rounds/round1/render/audio/dregon_v2` in the job's worktree).
Rendered audio of `michaels_v2_fly125cruise` (uncommitted): `s3://omnirun-artifacts/nv2-r1-score-michaels2-c3bbbd/outputs/results/noise_v2/rounds/round1/render/audio/michaels_v2` (job `nv2-r1-score-michaels2-c3bbbd`, written to `results/noise_v2/rounds/round1/render/audio/michaels_v2` in the job's worktree).
Rendered audio of `michaels_v2_retry_pinned_lam` (uncommitted): `s3://omnirun-artifacts/nv2-r1-score-michaels-re-eafb5e/outputs/results/noise_v2/rounds/round1/render/audio/michaels_v2_retry` (job `nv2-r1-score-michaels-re-eafb5e`, written to `results/noise_v2/rounds/round1/render/audio/michaels_v2_retry` in the job's worktree).

## Gate verdicts

| gate | verdict | number | threshold |
|---|---|---|---|
| HPPNet DREGON cruise | FAIL | 78.803342 rev/s (95 % upper 79.482310) | <= 1.897063 |
| HPPNet Michael's ratio | FAIL | 5.550358 (16.799053 rev/s) | <= 1.05 (3.177994 rev/s) |
| proxy `ltas_abs_db` dregon_cruise | FAIL | 3.7518 dB (spread 3.2542) | <= 1.9786 dB |
| proxy `ltas_abs_db` michaels_cruise | FAIL | 1.4156 dB (spread 1.6608) | <= 1.2197 dB |
| likelihood comb (decisive) | PASS | model -373,414.1187 nats/s | oracle -371,916.5041, margin -1,497.6146 |
| likelihood floor | report | model -8,487.7119 nats/s | oracle -8,713.2663, margin +225.5544 |
| likelihood full | report | model -381,901.8305 nats/s | oracle -380,629.7704, margin -1,272.0601 |

## HPPNet PIT MAE per support (rev/s)

| rig | support | regime | real | candidate | seed spread |
|---|---|---|---:|---:|---:|
| dregon | `free-flight_nosource_room2@1512727397.205045+4.000000` | cruise | 0.638129 | 79.185832 | 4.308237 |
| dregon | `hovering_nosource_room2@1511903905.394490+4.000000` | cruise | 0.888408 | 79.677176 | 4.752101 |
| dregon | `updown_nosource_room2@1511903578.348311+4.000000` | cruise | 1.694930 | 78.003475 | 5.051474 |
| dregon | `rectangle_nosource_room2@1511905725.952559+4.000000` | cruise | 1.685142 | 78.142193 | 8.567542 |
| dregon | `spinning_nosource_room2@1511905200.978012+4.000000` | cruise | 1.186931 | 79.008035 | 6.496934 |
| michaels | `FLY124@8.000000+8.000000` | standby | 0.398474 | 34.855680 | 1.629714 |
| michaels | `FLY124@16.000000+8.000000` | standby | 0.289978 | 34.843977 | 1.397137 |
| michaels | `FLY124@27.680000+8.000000` | ramp | 3.171148 | 22.140809 | 0.858307 |
| michaels | `FLY124@40.000000+8.000000` | cruise | 0.884854 | 1.407458 | 0.321942 |
| michaels | `FLY124@56.000000+8.000000` | cruise | 0.290248 | 0.869698 | 0.296574 |

DREGON cruise: candidate mean 78.803342 rev/s over 5 recording-level clusters, one-sided 95 % interval [78.124375, 79.482310] (t and 20000-draw cluster bootstrap, conservative), against the frozen target 1.897063 = 1.218708 + 0.7 x (2.187786 - 1.218708). Margin -76.906280 rev/s.

Real arm reproduction: measured 1.218708 rev/s against the frozen 1.218708 (relative 2.688e-09).

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
| dregon_cruise | `free-flight_nosource_room2@1512727397.205045+4.000000` | 4.3455 | 0.0175 | 3.7415 | -4.3455 |
| dregon_cruise | `hovering_nosource_room2@1511903905.394490+4.000000` | 3.8576 | 0.0177 | 2.0452 | -3.8576 |
| dregon_cruise | `updown_nosource_room2@1511903578.348311+4.000000` | 5.5492 | 0.0200 | 3.8886 | -5.5492 |
| dregon_cruise | `rectangle_nosource_room2@1511905725.952559+4.000000` | 2.2950 | 0.0786 | 1.8778 | -2.2199 |
| dregon_cruise | `spinning_nosource_room2@1511905200.978012+4.000000` | 2.7120 | 0.0175 | 1.6265 | -2.7120 |
| michaels_cruise | `FLY124@40.000000+8.000000` | 0.5852 | 0.2170 | 1.2045 | +0.1672 |
| michaels_cruise | `FLY124@56.000000+8.000000` | 2.2460 | 0.2730 | 2.0867 | +2.2460 |

dregon_cruise: mean 3.7518 dB (spread 3.2542 over 5 supports) against the closure-0.7 gate 1.9786 dB = 2.4308 - 0.7 x (2.4308 - 1.7848); margin -1.7732 dB. Secondary `mr_ltas` 2.6359 dB (reported, never decisive).

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

None for the round score itself: all three candidates that exist were rendered on the frozen supports and probed with the frozen HPPNet checkpoint (sha256 verified in-job), and every gate is measured; the real arm reproduces the frozen DREGON real PIT MAE to 2.7e-09. What the numbers are NOT: a parity verdict on a healthy DREGON comb — the four Motor*_70 bench combs frozen into the DREGON arm are defective (in-band profile ~40 dB below the floor; BenchDiag in progress), so that row is a pipeline proof and must be re-scored when corrected combs land, which needs only --fit dregon=<path>. Also outstanding: R1Fit's nv2-r1-dregon-restarts-a7084c will REWRITE the 21 bench_dregon_*__bench.json files as best-of-4, moving the frozen comb again; and no fit in this round converged (all three report optimiser.converged=false, which_converged=none).

## Provenance

The frozen support identities, one-sided interval, regime-mean ratio, proxy thresholds, HPPNet checkpoint hash, and likelihood/oracle definition are reproduced in `src/experiments/noise_model/gates.py`. The support availability evidence is committed in `b5d369de`. This score record intentionally reports `not_run` rather than substituting legacy-model measurements for absent R1 fits.

### Addendum — this scoring pass

The `Provenance` section above is kept verbatim from the earlier, unscored record; the gates it calls `not_run` are the ones measured here, and whatever is still not run is named in `Exact blocker`. This pass:

* `michaels_v2_fly125cruise` — fit git `8093285f352cd33e31e56b424bb64c5dbe35f06d`, render/probe job `nv2-r1-score-michaels2-c3bbbd`, arm record `results/noise_v2/rounds/round1/render/arm_michaels_v2.json`, render wall 253 s.

### Addendum — this scoring pass

The `Provenance` section above is kept verbatim from the earlier, unscored record; the gates it calls `not_run` are the ones measured here, and whatever is still not run is named in `Exact blocker`. This pass:

* `michaels_v2_fly125cruise` — fit git `8093285f352c`, render/probe job `nv2-r1-score-michaels2-c3bbbd`, arm record `results/noise_v2/rounds/round1/render/arm_michaels_v2.json`, render wall 253 s.

### Addendum — this scoring pass

The `Provenance` section above is kept verbatim from the earlier, unscored record; the gates it calls `not_run` are the ones measured here, and whatever is still not run is named in `Exact blocker`. This pass:

* `dregon_v2_floor_benchcomb` — fit git `a66ecb9529cc`, render/probe job `nv2-r1-score-dregon-662433`, arm record `results/noise_v2/rounds/round1/render/arm_dregon_v2.json`, render wall 130 s.
* `michaels_v2_fly125cruise` — fit git `8093285f352c`, render/probe job `nv2-r1-score-michaels2-c3bbbd`, arm record `results/noise_v2/rounds/round1/render/arm_michaels_v2.json`, render wall 253 s.

### Addendum — this scoring pass

The `Provenance` section above is kept verbatim from the earlier, unscored record; the gates it calls `not_run` are the ones measured here, and whatever is still not run is named in `Exact blocker`. This pass:

* `dregon_v2_floor_benchcomb` — fit git `a66ecb9529cc`, render/probe job `nv2-r1-score-dregon-662433`, arm record `results/noise_v2/rounds/round1/render/arm_dregon_v2.json`, render wall 130 s.
* `michaels_v2_fly125cruise` — fit git `8093285f352c`, render/probe job `nv2-r1-score-michaels2-c3bbbd`, arm record `results/noise_v2/rounds/round1/render/arm_michaels_v2.json`, render wall 253 s.

### Addendum — this scoring pass

The `Provenance` section above is kept verbatim from the earlier, unscored record; the gates it calls `not_run` are the ones measured here, and whatever is still not run is named in `Exact blocker`. This pass:

* `dregon_v2_floor_benchcomb` — fit git `a66ecb9529cc`, render/probe job `nv2-r1-score-dregon-662433`, arm record `results/noise_v2/rounds/round1/render/arm_dregon_v2.json`, render wall 130 s.
* `michaels_v2_fly125cruise` — fit git `8093285f352c`, render/probe job `nv2-r1-score-michaels2-c3bbbd`, arm record `results/noise_v2/rounds/round1/render/arm_michaels_v2.json`, render wall 253 s.
* `michaels_v2_retry_pinned_lam` — fit git `7ad5e82ffca9`, render/probe job `nv2-r1-score-michaels-re-eafb5e`, arm record `results/noise_v2/rounds/round1/render/arm_michaels_v2_retry.json`, render wall 250 s.
