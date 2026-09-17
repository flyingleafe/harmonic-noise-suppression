# Noise model v2 — round 1 gate score: FAIL / NOT RUN

| result | value |
|---|---:|
| top-level PASS | **false** |
| materialized supports | **0 / 179** |
| DREGON bench | 0 / 21 |
| bench survey points | 0 / 135 |
| DREGON floor set (five frozen score + five disjoint floor fits) | 0 / 10 |
| Michael's cruise set (eight FLY125 fit + five frozen FLY124 score supports) | 0 / 13 |
| HPPNet gate | **not_run** |
| proxy gate | **not_run** |
| likelihood gate | **not_run** |

## Gate verdicts

| gate | verdict | number | frozen protocol / why it was not run |
|---|---|---:|---|
| HPPNet DREGON cruise | **NOT RUN** | — | Candidate audio does not exist. The frozen probe would use `hppnet_l2_r2_s0/best`, SHA-256 `6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1`, 8 mics, and render seeds 2001–2004. |
| HPPNet Michael's FLY124 ratio | **NOT RUN** | — | Candidate audio does not exist for the frozen standby/ramp/cruise FLY124 supports. |
| proxy `ltas_abs_db` / `mr_ltas` | **NOT RUN** | — | No candidate audio exists for mic-0 absolute LTAS or report-only multi-resolution LTAS. |
| likelihood, comb band | **NOT RUN** | — | No candidate expected periodogram exists for either frozen FLY124 cruise support, so no model-versus-oracle composite risk or margin is fabricated. |
| likelihood, floor/full bands | **NOT RUN** | — | Report-only bands are likewise absent because the candidate does not exist. |

## Exact blocker

R1 requires a candidate fit before any gate can be evaluated. The committed support availability record, `results/noise_v2/rounds/round1/supports/index.json`, reports **0 materialized supports**:

* `dregon-floor`: **0 / 10** (the five frozen 4-second DREGON score windows and five disjoint 8-second floor-fit windows);
* `michaels-cruise`: **0 / 13** (eight FLY125 cruise-fit and five frozen FLY124 score supports);
* `dregon-bench`: **0 / 21** and `bench-points`: **0 / 135**.

The initial `uni-cpu` support jobs failed before execution because `PYTHONPATH=src` was absent. Their one permitted retries reached dload but failed to access its manifests with `botocore.exceptions.NoCredentialsError: Unable to locate credentials`. These are availability failures, not valid cache/materialized inputs. `R1CoreRecovery` consequently confirmed that no scientifically valid R1 fit JSON exists. The final HPPNet job was deliberately not submitted: scoring without a candidate would fabricate evidence.

## Frozen support and scoring protocol

The missing run would have rendered DREGON from the floor-only flight fit with its four single-rotor bench-comb fits frozen, and Michael's from the FLY125 cruise fit on the real FLY124 carriers (including sampled standby/ramp trajectories). The HPPNet quantity is PIT MAE on all eight mics over frozen render seeds `[2001, 2002, 2003, 2004]`; DREGON uses the five-recording one-sided 95% interval against 1.897063 rev/s, Michael's the equal-regime mean ratio against 1.05 × 3.026661. The proxy is mic-0 absolute `ltas_abs_db` against 1.978609 dB (DREGON) and 1.219668 dB (Michael's); `mr_ltas` is report-only. The likelihood would pool the two Michael's cruise supports with periodic-Hann NFFT 2048 / hop 512, split at 300 Hz, and decide solely on a comb-band composite-risk margin against the speed-matched oracle.

## Legacy smoke comparability

| legacy replay (`noise-v2-r1-legacy-smoke-453b44`) | observed | frozen scalar | relative delta |
|---|---:|---:|---:|
| DREGON cruise synthetic PIT MAE | 2.187786 | 2.187786 | 4.79e-08 |
| Michael's equal-regime synthetic PIT MAE | 3.199486 | 3.026661 | 5.71% |

This is **not a comparable legacy acceptance result**. The gitignored baseline calibration that fixes the old export-family selection and per-recording route was unavailable. Although the DREGON scalar happens to match, Michael's does not; because identical inputs cannot be established, no 2% reproduction pass/fail is asserted.

## Provenance

The frozen support identities, one-sided interval, regime-mean ratio, proxy thresholds, HPPNet checkpoint hash, and likelihood/oracle definition are reproduced in `src/experiments/noise_model/gates.py`. The support availability evidence is committed in `b5d369de`. This score record intentionally reports `not_run` rather than substituting legacy-model measurements for absent R1 fits.
