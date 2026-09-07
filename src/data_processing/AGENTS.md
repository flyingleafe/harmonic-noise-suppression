# data_processing/ — datasets, online mixing, RPS processing

## Purpose

Everything between raw recordings and a training loop's `td.Frame`s: raw-source
registry, frozen derived-dataset specs, the single copy of the mixing math, the
online-mix compiler, torch `Dataset` adapters. Every dataset is declared once
and materialized only through `scripts/derive.py`. Design (the *why*):
`docs/refactor-data-pipelines.md`. VK/refinement tracking: `src/tracking`.

## Map

**Layer 1 — raw sources (`sources/`)**, torch-free.
- `sources/__init__.py` — `REGISTRY`: one `SourceDataset` per raw dataset
  (pinned `DownloadSpec` / `fetcher` / `raw_dataset` + `builder(raw_dir) ->
  Iterator[(key, td.Frame)]`, `tdframe-v1`). DREGON is one entry like MIMII.
  Helpers `get`/`raw_root`/`iter_frames`/`geometry`.
- `sources/_common.py` — builder helpers, `LAYOUT = "tdframe-v1"`.
- `sources/dregon.py` — DREGON builder + raw loaders (`load_timeframe`,
  `load_dregon_timeframes`, `get_geometry`, `clean_command_spikes`).
- `sources/michaels.py` — Michael's DJI M100 rig: `MICHAELS_FILES` /
  `MICHAELS_TEST_FILES` (relative to the raw tree; `resolve_raw_root()`),
  `MICHAELS_RPS_SCALE`, `load_raw_aligned` (`anchor=True` for FLY103/108),
  `build` / `build_test`, `load_michaels_timeframes`, `get_geometry`.
- `sources/{aerosonicdb,avq,droneaudio,hornbase,hustmotor,kaist,mimii,spcup19}.py`
  — one builder per external dataset. `downloaders.py` — idempotent fetchers.

**Layer 2 — derived datasets**, torch-free.
- `derivations.py` — `SPECS`: every derived dataset as a frozen dload pipeline
  spec + generator (`dregon_lm`, `dregon_lm_subset`, `dn_lm`, `source_frames`,
  `frame_subset`, `raw_subset`, `pcm16_mono`, `beatvk_valid`, `avq_vkrps`,
  `se_valid`, `decomp_frames`); `PARENTS` (pinned parent URIs, drift-guarded by
  tests), `HISTORICAL_PINS`, `SE_HELDOUT_SPEAKERS`, `fingerprint`.
- `mixing.py` — the single copy of the per-sample mixing math (disk-free):
  offline `mix_at_snr`, online `scale_source_to_snr`/`mix_at_source_to_noise_snr`,
  `resolve_motor_tracks`/`find_inflight_window`, `load_noise_source_frames`,
  `render_multichannel_sample`, `iter_real_valid_clips`, `mix_dn_lm`.
- `canonical.py` — `CANONICAL_ENTRIES`/`ENTRY_ALIASES` + `coerce_frame`
  (source-specific entry names → the canonical vocabulary).
- `frames.py` — `td.Frame` conventions: `audio_series`/`rps_series`,
  `get_meta`/`with_meta`, `adapt_recording_frame` (rich recording → minimal
  audio+rps Frame), `PUBLISHED_RPS_KEYS`.

**Layer 3 — consumption.**
- `streams.py` — dload ↔ tdseries bridge: `DloadFrameDataset`, the `tdframe-v1`
  codec (`frame_to_sample`/`sample_to_frame`), combinators (`to_frames`/
  `frame_windows`/`mix_frames`), `ensure_local`/`resolve_source` (`dload:`
  URIs), `iter_published_frames`, `open_repository`.
- `online_mixing.py` — the online-mix compiler: policy YAML → one infinite
  `dload.Pipeline` of `td.Frame`s (`build_online_mix_pipeline` =
  `build_noise_stream` + `build_speech_stream`); `make_rng(seed, sample_id)`.
- `frame_datasets.py` — torch adapters: `DregonLMFrameDataset`,
  `DNLMFrameDataset`, `SEValidFrameDataset`, `OnlineMixFrameDataset`,
  `NoiseGenFrameDataset`, `StaticCombGenDataset`, `DecompFrameDataset`
  (`dataset` is a **list**; each record keeps its `drone` rig id), and the
  validation compositors (`FixedSynthFrameDataset`, `ConcatFrameDataset`,
  `MixtureMatchedValidDataset`, …).
- `noise_rps_dataset.py` — `NoiseRPSDataset`: chunkable noise+RPS over DREGON
  `in_flight_noise` + Michael's (`frames:NAME[@VER]`, `dload:` URIs, paths).
- Noise engines (`sources.noise[].kind`): `generated_noise.py`
  (`GeneratedNoisePool`, `generated`: spawn CUDA producer + shared-memory
  seqlock ring), `gp_noise.py` (`GPRotorNoisePool`, `gp`; core
  `egonoise_gp.py`), `rotor_spectral_model.py` (`StaticCombNoisePool`,
  `static_comb`), `stochastic_rotor_noise.py` (`StochasticNoisePool`,
  `stochastic` — the generative direction of `tracking.joint_decompose`),
  `silence_noise.py` (`SilenceNoisePool`, `silence`: zero-RPS floors).
- Augmentation: `noise_augmentations.py` (`policy.noise_augmentations`),
  `time_warp.py` (`policy.noise_time_warp`), `rps_corruption.py` (clean-RPS
  corruption → extra `rps_cond` entry; telemetry label-noise
  `tachometer_corrupt` / `presmooth_track`).
- `harmonicity.py` (analysis-stage `measure_harmonicity`), `rps_synthesis.py`
  / `collate.py` (mixer constants from `tracking.rotors`), `comb_bench*.py`.

## Contracts / invariants

- **Fixes are baked in at derivation time.** `DREGON-frames` `motors_command`
  is already `clean_command_spikes`-cleaned; `michaels-frames` `rps` is
  aligned and rev/s-calibrated (`MICHAELS_RPS_SCALE` inside `load_raw_aligned`).
  Consumers re-apply nothing (`adapt_recording_frame` → `rps`;
  `resolve_motor_tracks` no-cleaning path).
- Published `*-frames` carry ONE `rps` track chosen by `PUBLISHED_RPS_KEYS`
  (`motors_measured` preferred; only the five DREGON `free-flight_*_room1`
  recordings log it) — detection and labelling use the same track, so
  command-only recordings keep their trailing logging freeze.
- Telemetry is time-last `(…, M)` on `StampIndex`-backed Series.
- Two SNR conventions: offline/LibriMix (`mix_at_snr`, speech is the
  reference) vs online (`scale_source_to_snr`, noise is the reference).
- Manifest layouts, dispatched on `meta["layout"]`: `sample-dir-v1`
  (`sample_NNNNN/` dirs + `_meta` sample; `dregon_lm`, `dn_lm`), `tdframe-v1`
  (one Frame per recording/clip; all frame generators), `raw-files` /
  `pcm16-mono-v1`.
- `adopt_only` specs are historical uploads whose bytes predate the spec;
  re-deriving would push a near-duplicate (mixing RNG is not byte-stable).
  `michaels-test-frames` (FLY103/FLY108) is a **TEST set**: no training
  derivation may root on it.
- Online-mix determinism: content is deterministic per `(base_seed, epoch,
  worker, position)`; curriculum/augmentation decisions are pure functions of
  the global sample id. `kind: generated` with `refresh: true` is not.

## Entry points

- **Publish**: `python scripts/derive.py list -v | derive <NAME> | adopt
  <NAME> --commit`, then `dload pin NAME && git add dload.lock`. Raw trees:
  `dload commit NAME --from data/NAME` + a `REGISTRY` `raw_dataset`. Bump
  `recipe_version` on any behavioural change — the fingerprint keys on the spec.
- **Train on a published split**: `DregonLMFrameDataset` returns a `td.Frame`
  per sample — `mixture` `(time,)` mono / `(mic, time)` multichannel, `rps`
  `(rotor, time)` on the STFT frame grid. `channel=<int>` selects one mic;
  `flatten_channels=True` expands each sample into `n_channels` mono-view
  Frames (`meta.channel` tagged; `conf/data/dregon_lm_v4_8ch_flat.yaml`).
  DCUNet/DCCRN stay 1-channel. Streamed: `DloadFrameDataset`
  (`conf/data/dregon_lm_v4_stream.yaml`).
- **Train on the online stream**: `OnlineMixFrameDataset.from_yaml(policy)`
  via a `conf/data/*.yaml` wrapper; infinite, so the experiment sets
  `samples_per_validation`; validation stays a fixed map-style set.
  `flatten_channels` = a `flat_map` stage, `rps_corruption` = a `map` stage.
- **Noise pool spec** (shared by derivations and `kind: frames`,
  `load_noise_source_frames`): `dataset` (`NAME[@version]`), `splits`/`split`,
  `recording_ids`, `exclude_recording_ids`, `take`, `min_motor_rps`,
  `channels`, `weight` (explicit = own sub-pool; unweighted reals merge
  duration-weighted).
- **Check a policy before training**: `python scripts/check_stream.py
  --experiment <name>` — real chunk→frame expansion, stage boundaries,
  empirical augmentation fire rates; nonzero exit on FAIL. Mandatory after any
  data-path refactor.
- **RPS helpers**: `sources.dregon.load_timeframe` / `load_dregon_timeframes`;
  `mixing.resolve_motor_tracks(tf) -> (detect_key, rps_key, needs_cleaning)`;
  `mixing.find_inflight_window(tf, key, min_motor_rps, clean=…) -> (t0, t1)`
  (pass `min_motor_rps=30.0`; the default 0 is backward-compat).

## Conventions

- Torch stays out of layers 1–2 so fingerprinting and `adopt` run anywhere.
- Public interface is config-in/stream-out: no source-cache prep scripts or
  cache flags; the memoized speech decode is the `librispeech-pcm16` derivation.
- Publish a derived subset (`frame_subset`) once a fixed small selection
  becomes an experiment's durable noise pool; `audio_pool` `include_keys` is
  for ad-hoc restriction only (it downloads every shard to find matches).

## Gotchas

- Datasets are gitignored — stream (`dload:` URIs) or `dload pull <name>`.
- `michaels_dir` in `conf/data/noise_rps_dregon_michaels*.yaml` is stale
  (`data/new-drone-noises`, effectively ignored); don't copy it.
- `new-drone-noises` holds exactly FLY103/FLY108 (mono, 48 kHz, audio is a
  sub-window of a ~2× longer log) — the held-out TEST set.
- `load_timeframe(target_sr=…)`: `librosa.resample` on the `(n_ch, N)` array
  with `axis=-1`, `res_type="soxr_hq"`; the wrong axis hangs.
- `real_valid` splits have no `vocals.wav` — RPS evaluation only.
- V4-Michaels leakage split: original LibriSpeech pin (never generated
  `vocals.wav`), exclude DREGON `free-flight_nosource_room1`, train `FLY125`,
  validate `FLY124`.
- Michael's telemetry published before 2026-07-31 is uncalibrated
  (`rps_scale` 1.0 in frame meta).
- Keep the 50k unaugmented warmup stage (G5: removing it regresses).

## Further reading

- `docs/refactor-data-pipelines.md` — architecture; § "Online-mix policy
  reference" (every `kind`, `generated`/`interp`, `gp`, `audio_pool` holdouts,
  `noise_augmentations`, SE task mode, `check_stream`).
- `docs/data-catalog.md` — DREGON recording inventory and telemetry, Michael's
  calibration constants and history, dataset variants, every `dload.lock` pin.
- `docs/data-and-artifacts.md` — dload/R2 workflow, cache env vars, omnirun.
- `docs/derived-datasets-plan.md`, `docs/external-datasets-plan.md`.
