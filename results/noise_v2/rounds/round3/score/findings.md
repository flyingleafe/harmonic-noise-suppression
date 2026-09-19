# Noise model v2 — round 3 gate score: PARITY FAIL / STRETCH FAIL

Top-level pass (legacy parity on both HPPNet gates): **false**. Stretch (frozen 0.70-gap DREGON target): **false**. All three frozen gates: **false**. Record `results/noise_v2/rounds/round3.json`, git `3fcae519cddb`.

## Candidates

| candidate | rig(s) | fit JSON | converged | fit wall (s) | render+probe job |
|---|---|---|---|---:|---|
| `michaels_v2_r3_regimes` | michaels | `results/noise_v2/rounds/round3/fits/michaels_fly125_standby__flight.json + results/noise_v2/rounds/round3/fits/michaels_fly125_cruise__flight.json` | **NO** ({'standby': 'none', 'cruise': 'none'}, |grad| 1174) | 14781 | `nv2-r3-score-michaels-re-ac8bc5` |
| `michaels_v2_r3_all` | michaels | `results/noise_v2/rounds/round3/fits/michaels_fly125_all__flight.json` | **NO** (none, |grad| 5430) | 5621 | `nv2-r3-score-michaels-ed8576` |
| `dregon_v2_r3_floor_lowk` | dregon | `results/noise_v2/rounds/round3/fits/dregon_room2_floor__flight_floor_lowk.json` | **NO** (none, |grad| 3217) | 4744 | `nv2-r3-score-dregon-31ff78` |

## The two bars

| rig | gate quantity | number | PARITY bar | parity | STRETCH bar | stretch |
|---|---|---:|---:|---|---:|---|
| dregon | five-recording cruise PIT MAE mean, one-sided 95 % upper bound beside it | 71.866599 | 2.187786 | **FAIL** (margin -69.678813) | 1.897063 | **FAIL** (margin -69.969536) |
| michaels | equal-regime mean PIT MAE over standby/ramp/cruise (ratio of means) | 1.621670 | 3.177994 | **PASS** (margin +1.556325) | — | — (margin —) |

PARITY is the legacy previous best: DREGON synthetic cruise PIT MAE 2.187786 rev/s (real arm 1.218708), Michael's 1.05 x 3.026661 = 3.177994 rev/s. STRETCH exists only on DREGON: 1.897063 = real + 0.7 x (baseline - real); Michael's frozen gate IS its parity bar.

## Round 2 → round 3

| quantity | unit | round 2 | round 3 | Δ (R3 − R2) | direction | moved |
|---|---|---:|---:|---:|---|---|
| dregon cruise PIT MAE (5-recording mean) | rev/s | 72.340606 | 71.866599 | -0.474007 | lower is better | **better** |
| dregon cruise PIT MAE (95 % upper) | rev/s | 74.830651 | 75.029755 | +0.199104 | lower is better | **worse** |
| michaels equal-regime PIT MAE | rev/s | 2.456399 | 1.621670 | -0.834729 | lower is better | **better** |
| michaels ratio vs the legacy regime mean | x | 0.8116 | 0.5358 | -0.2758 | lower is better | **better** |
| proxy ltas_abs_db, dregon cruise | dB | 3.5024 | 3.7480 | +0.2456 | lower is better | **worse** |
| proxy ltas_abs_db, michaels cruise | dB | 2.1491 | 1.2816 | -0.8675 | lower is better | **better** |
| mr_ltas, dregon cruise (report-only) | — | 2.5833 | 2.6592 | +0.0760 | lower is better | **worse** |
| mr_ltas, michaels cruise (report-only) | — | 2.2111 | 1.4615 | -0.7496 | lower is better | **better** |
| likelihood comb-band margin vs oracle | nats/s | -970.154760 | -1749.799951 | -779.645191 | higher is better | **worse** |

Both sides are read verbatim out of `results/noise_v2/rounds/round2.json` (git `9579ca94592c`, candidates `michaels_v2_r2_all`, `dregon_v2_r2_floor_combgain`, `dregon_v2_r2_floor_benchcomb`) and this record; a dash is a quantity one of the two rounds did not measure, never a substituted number.

## Per-candidate numbers

| candidate | HPPNet | proxy `ltas_abs_db` (gate) | `mr_ltas` | likelihood comb margin |
|---|---:|---:|---:|---:|
| `michaels_v2_r3_regimes` | 1.621670 rev/s (ratio 0.5358) | 1.2816 (1.2197) | 1.4615 | -1,749.8000 nats/s |
| `michaels_v2_r3_all` | 2.334258 rev/s (ratio 0.7712) | 1.8863 (1.2197) | 2.0460 | -862.8349 nats/s |
| `dregon_v2_r3_floor_lowk` | 71.866599 rev/s (95 % upper 75.029755) | 3.7480 (1.9786) | 2.6592 | not run |

Rendered audio of `dregon_v2_r3_floor_lowk` (uncommitted): `s3://omnirun-artifacts/nv2-r3-score-dregon-31ff78/outputs/results/noise_v2/rounds/round3/render/audio/dregon_v2` (job `nv2-r3-score-dregon-31ff78`, written to `results/noise_v2/rounds/round3/render/audio/dregon_v2` in the job's worktree).

## Gate verdicts

| gate | verdict | number | threshold |
|---|---|---|---|
| HPPNet DREGON cruise | FAIL | 71.866599 rev/s (95 % upper 75.029755) | <= 1.897063 |
| HPPNet Michael's ratio | PASS | 0.535795 (1.621670 rev/s) | <= 1.05 (3.177994 rev/s) |
| proxy `ltas_abs_db` michaels_cruise | FAIL | 1.2816 dB (spread 1.6179) | <= 1.2197 dB |
| proxy `ltas_abs_db` dregon_cruise | FAIL | 3.7480 dB (spread 3.2995) | <= 1.9786 dB |
| likelihood comb (decisive) | PASS | model -373,666.3040 nats/s | oracle -371,916.5041, margin -1,749.8000 |
| likelihood floor | report | model -8,345.2279 nats/s | oracle -8,713.2663, margin +368.0384 |
| likelihood full | report | model -382,011.5319 nats/s | oracle -380,629.7704, margin -1,381.7615 |

## HPPNet PIT MAE per support (rev/s)

| rig | support | regime | real | candidate | seed spread |
|---|---|---|---:|---:|---:|
| michaels | `FLY124@8.000000+8.000000` | standby | 0.398474 | 3.031747 | 6.864292 |
| michaels | `FLY124@16.000000+8.000000` | standby | 0.289978 | 3.295666 | 4.850301 |
| michaels | `FLY124@27.680000+8.000000` | ramp | 3.171148 | 3.145259 | 0.230119 |
| michaels | `FLY124@40.000000+8.000000` | cruise | 0.884854 | 0.978594 | 0.049200 |
| michaels | `FLY124@56.000000+8.000000` | cruise | 0.290248 | 0.409024 | 0.033139 |
| dregon | `free-flight_nosource_room2@1512727397.205045+4.000000` | cruise | 0.638129 | 67.553539 | 23.120403 |
| dregon | `hovering_nosource_room2@1511903905.394490+4.000000` | cruise | 0.888408 | 73.202644 | 10.536427 |
| dregon | `updown_nosource_room2@1511903578.348311+4.000000` | cruise | 1.694930 | 74.312218 | 12.437738 |
| dregon | `rectangle_nosource_room2@1511905725.952559+4.000000` | cruise | 1.685142 | 69.172455 | 18.119620 |
| dregon | `spinning_nosource_room2@1511905200.978012+4.000000` | cruise | 1.186931 | 75.092137 | 13.856118 |

DREGON cruise: candidate mean 71.866599 rev/s over 5 recording-level clusters, one-sided 95 % interval [68.703443, 75.029755] (t and 20000-draw cluster bootstrap, conservative), against the frozen target 1.897063 = 1.218708 + 0.7 x (2.187786 - 1.218708). Margin -69.969536 rev/s.

Real arm reproduction: measured 1.218708 rev/s against the frozen 1.218708 (relative 1.078e-09).

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
| michaels_cruise | `FLY124@40.000000+8.000000` | 0.4727 | 0.0383 | 0.9711 | -0.2345 |
| michaels_cruise | `FLY124@56.000000+8.000000` | 2.0906 | 0.0803 | 1.9519 | +2.0906 |
| dregon_cruise | `free-flight_nosource_room2@1512727397.205045+4.000000` | 4.3844 | 0.0229 | 3.8362 | -4.3844 |
| dregon_cruise | `hovering_nosource_room2@1511903905.394490+4.000000` | 3.8596 | 0.0148 | 2.0652 | -3.8596 |
| dregon_cruise | `updown_nosource_room2@1511903578.348311+4.000000` | 5.5377 | 0.0108 | 3.9103 | -5.5377 |
| dregon_cruise | `rectangle_nosource_room2@1511905725.952559+4.000000` | 2.2383 | 0.1031 | 1.8739 | -2.0919 |
| dregon_cruise | `spinning_nosource_room2@1511905200.978012+4.000000` | 2.7200 | 0.0191 | 1.6105 | -2.7200 |

michaels_cruise: mean 1.2816 dB (spread 1.6179 over 2 supports) against the closure-0.7 gate 1.2197 dB = 1.5333 - 0.7 x (1.5333 - 1.0852); margin -0.0620 dB. Secondary `mr_ltas` 1.4615 dB (reported, never decisive).

dregon_cruise: mean 3.7480 dB (spread 3.2995 over 5 supports) against the closure-0.7 gate 1.9786 dB = 2.4308 - 0.7 x (2.4308 - 1.7848); margin -1.7694 dB. Secondary `mr_ltas` 2.6592 dB (reported, never decisive).

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

The frozen support identities, render seeds, one-sided interval, regime-mean ratio, proxy thresholds, probe checkpoint digest and likelihood/oracle definition are all read from `src/experiments/noise_model/gates.py`; this record adds no scalar of its own. Probe `hppnet_l2_r2_s0/best`, sha256 `6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1`, verified in-job: **true** (the scoring pass dies on a digest mismatch, so a scored arm cannot exist without the match). Render seeds [2001, 2002, 2003, 2004], 8 microphones. Record git `0843f3ab4b24`.

| candidate | rig(s) | fit | fit git | converged | job | audio |
|---|---|---|---|---|---|---|
| `dregon_v2_r3_floor_lowk` | dregon | `results/noise_v2/rounds/round3/fits/dregon_room2_floor__flight_floor_lowk.json` | `4a0ab14f1b0c` | **NO** | `nv2-r3-score-dregon-31ff78` | `s3://omnirun-artifacts/nv2-r3-score-dregon-31ff78/outputs/results/noise_v2/rounds/round3/render/audio/dregon_v2` |
| `michaels_v2_r3_all` | michaels | `results/noise_v2/rounds/round3/fits/michaels_fly125_all__flight.json` | `62e45d96743b` | **NO** | `nv2-r3-score-michaels-ed8576` | `—` |

* `dregon_v2_r3_floor_lowk` — arm record `results/noise_v2/rounds/round3/render/arm_dregon_v2.json`, render wall 167 s, 1 rig(s) dregon, primary for dregon.
* `michaels_v2_r3_all` — arm record `results/noise_v2/rounds/round3/render/arm_michaels_v2.json`, render wall 323 s, 1 rig(s) michaels, primary for michaels.

Support availability from `results/noise_v2/rounds/round2/supports/index.json` (updated 2026-09-18T11:41:51+0100): dregon-bench 21/21, bench-points 135/135, michaels-all 10/10, dregon-floor 10/10.

## R1 → R2 → R3 (hand-added, copied from the three round records)

This section is HAND-ADDED (by `R3Record`) and is NOT produced by
`noise_v2_round_score.py --compose`; a recompose of round 3 rewrites everything
above it, so this section has to be re-added by hand after every recompose.
Every cell is read verbatim out of
`results/noise_v2/rounds/round1.json` (git `066651e8eb31`),
`round2.json` (git `9579ca94592c`) and `round3.json` (git `3fcae519cddb`) —
primary candidate per rig in each round, which for round 3's michaels is the
PER-REGIME arm `michaels_v2_r3_regimes`
(`round3/render/arm_michaels_v2_regimes.json`). **No fit in any of the three
rounds is converged**, so every number below carries that label.

| quantity | unit | R1 | R2 | R3 | bar |
|---|---|---:|---:|---:|---:|
| DREGON cruise PIT MAE (5-recording mean) | rev/s | 78.803342 | 72.340606 | **71.866599** | ≤ 2.187786 parity / 1.897063 stretch |
| DREGON cruise PIT MAE, one-sided 95 % upper | rev/s | 79.482310 | 74.830651 | **75.029755** | ≤ 2.187786 / 1.897063 |
| Michael's equal-regime mean PIT MAE | rev/s | 16.799053 | 2.456399 | **1.621670** | ≤ 3.177994 |
| Michael's aggregate ratio vs legacy 3.026661 | × | 5.550358 | 0.811587 | **0.535795** | ≤ 1.05 |
| · per regime, standby (legacy 0.317103) | rev/s | 32.632045 | 3.006774 | **0.785384** | — |
| · per regime, ramp (legacy 8.007098) | rev/s | 16.955751 | 3.393256 | **3.267119** | — |
| · per regime, cruise (legacy 0.755783) | rev/s | 0.809363 | 0.969167 | **0.812507** | — |
| proxy `ltas_abs_db`, DREGON cruise | dB | 3.751843 | 3.502432 | **3.748014** | ≤ 1.978609 |
| proxy `ltas_abs_db`, Michael's cruise | dB | 1.415590 | 2.149092 | **1.281622** | ≤ 1.219668 |
| likelihood comb-band margin vs oracle | nats/s | −1497.614580 | −970.154760 | **−1749.799951** | higher is better |

Round 3's SECOND Michael's arm, the pooled `michaels_v2_r3_all`
(`round3/render/arm_michaels_v2.json`, one FLY125 all-regime fit driving every
regime), is in the same `round3.json` and is the one the R3 column would carry
if the per-regime arm had not landed: equal-regime 2.334258 rev/s (ratio
0.771232), standby 3.163707 / ramp 3.145259 / cruise 0.693809, proxy 1.886305
dB, comb margin −862.834939 nats/s.

Reading, per row family:

* **DREGON** is still two orders of magnitude off parity in every round. R1's
  arm carried the DEFECTIVE bench combs (its own record says so); R2 added the
  single `comb_gain_db` re-levelling scalar (−3.2323 dB) and bought 6.46 rev/s;
  R3 added the eight per-order `low_order_gain_db` freedoms (k = 1..8) on top of
  a deeper re-level (`comb_gain_db` −6.918144 dB) and bought a further
  0.474007 rev/s on the mean while the 95 % upper bound went the WRONG way
  (+0.199104, because the between-recording spread grew). The DREGON proxy is
  back at its R1 value (+0.245582 dB vs R2): the low-order freedoms trade LTAS
  level fidelity for likelihood, and the PIT metric barely notices. Two rounds
  of added comb-gain freedom have moved the DREGON gate by 0.6 % of the gap
  that has to close.
* **Michael's** is the opposite picture: inside its parity bar since R2 and
  improving monotonically on the equal-regime mean (16.799053 → 2.456399 →
  1.621670), now at 53.6 % of the legacy regime mean. The per-regime
  composition is what bought R3's step: the dedicated standby fit collapses the
  standby regime from 3.006774 (R2) / 3.163707 (R3 pooled) to **0.785384**
  rev/s, i.e. from ratio 9.4820 / 9.9769 to 2.4768 — the single largest
  per-regime move of the campaign. Ramp stays far below legacy (0.4080) and
  cruise pays a small price for the blend (0.812507 vs the pooled arm's
  0.693809, ratio 1.0751 vs 0.9180).
* **The Michael's proxy gate is now within 0.062 dB of passing**
  (1.281622 vs 1.219668) — from −0.9294 dB of margin in R2 to −0.0620 dB in R3.
  It is the only frozen gate in the campaign that is close.
* **The likelihood comb margin moved the wrong way** (−970.154760 → −1749.799951
  nats/s, worse than R1's −1497.614580): the regime blend buys PIT MAE and LTAS
  level on the two cruise supports while costing Whittle risk in the comb band.
  The gate still reads PASS because it is scored against the frozen R1 margin
  convention recorded in the round JSON, not against R2.

### Addendum — this scoring pass

The `Provenance` section above is kept verbatim from the earlier, unscored record; the gates it calls `not_run` are the ones measured here, and whatever is still not run is named in `Exact blocker`. This pass:

* `michaels_v2_r3_regimes` — fit git `d9bd19401caf`, render/probe job `nv2-r3-score-michaels-re-ac8bc5`, arm record `results/noise_v2/rounds/round3/render/arm_michaels_v2_regimes.json`, render wall 433 s.
* `michaels_v2_r3_all` — fit git `62e45d96743b`, render/probe job `nv2-r3-score-michaels-ed8576`, arm record `results/noise_v2/rounds/round3/render/arm_michaels_v2.json`, render wall 323 s.
* `dregon_v2_r3_floor_lowk` — fit git `4a0ab14f1b0c`, render/probe job `nv2-r3-score-dregon-31ff78`, arm record `results/noise_v2/rounds/round3/render/arm_dregon_v2.json`, render wall 167 s.
