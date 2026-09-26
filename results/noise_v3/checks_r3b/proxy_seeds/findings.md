# Noise model v2 — round 6 gate score (FAIL)

Arm: v2 — michaels: v2 FLY125 PER-REGIME pair: the standby-only fit below 45 rev/s, the cruise-only fit above 65, their powers interpolated across the ramp band (render_noise_regimes)
Render seeds [2001, 2002, 2003, 2004], 8 mics, git `ffb61244a0921458a6800135f527eb2fe6c1d7b7`.

## Gate verdicts

| gate | verdict | number | threshold |
|---|---|---|---|
| proxy `ltas_abs_db` dregon_cruise | FAIL | nan dB (spread nan) | <= 1.9786 dB |
| proxy `ltas_abs_db` michaels_cruise | FAIL | 2.4837 dB (spread 0.1190) | <= 1.2197 dB |
| likelihood comb (decisive) | FAIL | model -334,118.4638 nats/s | oracle -371,916.5041, margin +37,798.0403 |
| likelihood floor | report | model -6,804.1953 nats/s | oracle -8,713.2663, margin +1,909.0710 |
| likelihood full | report | model -340,922.6590 nats/s | oracle -380,629.7704, margin +39,707.1113 |

## HPPNet PIT MAE per support (rev/s)

The HPPNet probe did not run in this pass (no probe run), so no PIT MAE is reported and the round cannot pass.

| rig | support | regime | real | candidate | seed spread |
|---|---|---|---:|---:|---:|
| michaels | `FLY124@8.000000+8.000000` | standby | — | — | — |
| michaels | `FLY124@16.000000+8.000000` | standby | — | — | — |
| michaels | `FLY124@27.680000+8.000000` | ramp | — | — | — |
| michaels | `FLY124@40.000000+8.000000` | cruise | — | — | — |
| michaels | `FLY124@56.000000+8.000000` | cruise | — | — | — |


## Proxy `ltas_abs_db` per support (dB, mic 0, absolute level)

| group | support | ltas_abs_db | seed spread | mr_ltas | level offset |
|---|---|---:|---:|---:|---:|
| michaels_cruise | `FLY124@40.000000+8.000000` | 2.5432 | 0.9203 | 2.7991 | +0.1177 |
| michaels_cruise | `FLY124@56.000000+8.000000` | 2.4242 | 0.9587 | 2.9214 | +2.0842 |

dregon_cruise: mean nan dB (spread nan over 0 supports) against the closure-0.7 gate 1.9786 dB = 2.4308 - 0.7 x (2.4308 - 1.7848); margin — dB. Secondary `mr_ltas` — dB (reported, never decisive).

michaels_cruise: mean 2.4837 dB (spread 0.1190 over 2 supports) against the closure-0.7 gate 1.2197 dB = 1.5333 - 0.7 x (1.5333 - 1.0852); margin -1.2640 dB. Secondary `mr_ltas` 2.8602 dB (reported, never decisive).

## Likelihood: composite risk against the speed-matched stationary oracle

NFFT 2048 / hop 512, pooled over 2 Michael's FLY124 cruise supports, band split at 300 Hz.

| band | Hz | model (nats/s) | oracle (nats/s) | margin | below oracle |
|---|---|---:|---:|---:|---|
| comb | 300-7900 | -334,118.4638 | -371,916.5041 | +37,798.0403 | False |
| floor | 30-300 | -6,804.1953 | -8,713.2663 | +1,909.0710 | False |
| full | 30-7900 | -340,922.6590 | -380,629.7704 | +39,707.1113 | False |

Frozen comb-band margin: +37,798.0403 nats/s (R1 freezes the comb-band model-minus-oracle margin; later rounds are held to it).

Oracle reproduction: recomputed full-band pooled oracle -380,629.7704 nats/s against the recorded -380,629.7704 nats/s of `results/noise_v2/short_whittle/short_whittle.json` (relative 0.000e+00).

## Protocol provenance

Reproduced from the previous campaign (sources in `src/experiments/noise_model/gates.py`): the frozen scorer `hppnet_l2_r2_s0/best` with its SHA-256 verified before anything is scored; `revised_eval.pit_mae` over all eight microphones on the pre-registered raw-telemetry regime support; the five DREGON room-2 4 s cruise windows and the five FLY124 windows the frozen evaluator resolved; the four frozen render seeds `[2001, 2002, 2003, 2004]`; `per_window`/`per_recording` aggregation; the DREGON target as `real + 0.7 x (baseline - real)` on the frozen v2 scalars; the recording-level one-sided 95 % interval of `revised_eval.cluster_interval` (t bound and 20000-draw cluster bootstrap at seed 0, conservative side); Michael's ratio of equally weighted per-regime MAEs; `ltas_abs_db` as `revised_eval.ltas_deviation_db` mean_abs_db on mic 0 with no normalisation; the composite risk of `marginal_frame_nll` + `FrameScore` + `composite_score`; and the `oracle_np` definition and reference segments of `results/noise_v2/short_whittle/short_whittle.json`.

Re-derived, because the source is not in the repository or does not exist yet: (a) the DREGON paired `baseline - candidate` improvement interval needs the per-recording baseline MAEs of the gitignored `results/revised_phase/baseline_v2/calibration.json`, so the one-sided 95 % recording-level instrument is applied to the candidate's own cluster mean against the frozen scalar target instead; (b) the per-band (300 Hz) oracle risk, because `short_whittle.json` records the oracle over the full 30-7900 Hz band only — it is recomputed here from the same oracle periodogram and the full-band value is cross-checked against the recorded number; (c) v2 has no fitted standby/ramp model, so the FLY125 cruise fit drives those two regimes on their own real carriers through the renderer's trajectory sampler (approved decision 3), where the old campaign used the manifest `baseline_map` cross-recording extrapolation of the FLY125 exports.
