# src/metrics — consolidated evaluation metrics

Every metric lives here, once (extracted from the deleted root
`metrics.py`, `valid.py`, `final_valid.py` and inline copies in the old
trainers). Same adapter pattern as `src/losses`: pure functions + Frame
adapters declaring `requires_pred` / `requires_target` FrameSpecs.

| Module | Contents |
|---|---|
| `separation.py` | `sdr`, `si_sdr`, `l1_freq`, `neg_log_wmse`, `aura_stft`, `aura_mrstft`, `bleedless`/`fullness`, `pesq`, `stoi`, `estoi` |
| `rps.py` | PIT-aware RPS metrics plus the GPU-vectorized training-panel pair: `batched_pit_mae` (24-permutation MAE-optimal assignment per clip) and `batched_pit_frame_errors` (`(B, T)` rotor-mean error under that SAME clip assignment, so its time mean is the clip score); target resampled `align_corners=False` like `experiments.rps_bench`. |
| `regimes.py` | `frame_regimes(target (B,R,T), frame_rate) -> (B,T) int8` — the four-regime rule (ramp ≥ 20 rev/s² over ±0.25 s after a 0.288 s boxcar, else zero max < 1, else standby mean < 45, else cruise), codes index `REGIMES`; shared by `training.validation` and `scripts/_regime_decomp.py` |
| `salience.py` | `SalienceBCEMetric` — the validation twin of `losses.SalienceRPSBCELoss` (shared map) |
| `salience_layers.py` | `LayerPITSalienceBCEMetric` (the twin of `losses.LayerPITSalienceBCELoss`) and `LayerPeakRPSMetric` — the same per-rotor layers read in **rev/s**: each layer's peak refined by the three-point log-parabolic fit (exact on a Gaussian layer), PIT-aligned. `rps_mae` is the number to compare arms on. It is deliberately NOT the deployed decoder (`models.harmonic_ports.layer_readout`'s CRF best path), which costs ~15 s per clip on CPU, where metrics run |
| `perf.py` | RTF, FLOPs (thop, lazy import), peak GPU memory |
| `suite.py` | `MetricSuite` — named collection, per-sample evaluation, mean/median aggregation, group-by on a meta key (e.g. `input_snr` → per-SNR tables in eval.py) |
| `_common.py` | `get_array`, `Metric` protocol, shared specs |

Selection is config-driven: `conf/metrics/*.yaml` names the suite;
`eval.py` writes `metrics.json`, `per_sample.csv`, `per_snr.csv` into
`results/<experiment>/eval/`. Training never selects checkpoints on a suite
metric; the panel (`training.validation`) scores RPS through `rps.py`.
