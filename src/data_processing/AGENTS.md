# data_processing/ — datasets, online mixing, RPS processing

## Purpose

Everything between raw recordings and a training loop's `td.Frame`s. Every
dataset is declared once and materialized only through `scripts/derive.py`.
Design: `docs/refactor-data-pipelines.md`. VK/refinement: `src/tracking`.

## Map

**Layer 1 — raw sources (`sources/`)**, torch-free.
- `sources/__init__.py` — `REGISTRY`: one `SourceDataset` per raw dataset
  (pinned `DownloadSpec` / `fetcher` / `raw_dataset` + `builder(raw_dir) ->
  Iterator[(key, td.Frame)]`, `tdframe-v1`); helpers `get`/`raw_root`/
  `iter_frames`/`geometry`.
- `sources/_common.py` — builder helpers, `LAYOUT = "tdframe-v1"`.
- `sources/dregon.py` — DREGON builder + raw loaders (`load_timeframe`,
  `load_dregon_timeframes`, `get_geometry`, `clean_command_spikes`);
  `SPLIT_RECORDINGS`, `TEST_RECORDING_IDS`.
- `sources/michaels.py` — Michael's DJI M100 rig: `MICHAELS_FILES`/
  `MICHAELS_TEST_FILES` (relative to the raw tree; `resolve_raw_root()`),
  `MICHAELS_RPS_SCALE`, `load_raw_aligned` (`anchor=True` for FLY103/108),
  `build`/`build_test`, `load_michaels_timeframes`, `get_geometry`.
- `sources/{aerosonicdb,avq,droneaudio,hornbase,hustmotor,kaist,mimii,spcup19}.py`
  — one builder each; `downloaders.py` — idempotent fetchers.

**Layer 2 — derived datasets**, torch-free.
- `derivations.py` — `SPECS`: every derived dataset as a frozen dload
  pipeline spec + generator (`dregon_lm`, `dregon_lm_subset`, `dn_lm`,
  `real_valid_panel`, `source_frames`, `frame_subset`, `raw_subset`,
  `pcm16_mono`, `beatvk_valid`, `avq_vkrps`, `se_valid`, `decomp_frames`);
  `PARENTS` (pinned, drift-guarded), `HISTORICAL_PINS`, `fingerprint`.
- `mixing.py` — the per-sample mixing math, one copy (disk-free): offline
  `mix_at_snr`, online `scale_source_to_snr`/`mix_at_source_to_noise_snr`,
  `resolve_motor_tracks`/`find_inflight_window`, `load_noise_source_frames`,
  `render_multichannel_sample`, `iter_real_valid_clips`, `mix_dn_lm`.
- `canonical.py` — `CANONICAL_ENTRIES`/`ENTRY_ALIASES` + `coerce_frame`
  (source entry names → the canonical vocabulary).
- `frames.py` — `td.Frame` conventions: `audio_series`/`rps_series`,
  `get_meta`/`with_meta`, `adapt_recording_frame` (rich → minimal audio+rps
  Frame), `PUBLISHED_RPS_KEYS`.

**Layer 3 — consumption.**
- `streams.py` — dload ↔ tdseries bridge: `DloadFrameDataset`, the
  `tdframe-v1` codec (`frame_to_sample`/`sample_to_frame`), combinators
  (`to_frames`/`frame_windows`/`mix_frames`), `ensure_local`/`resolve_source`
  (`dload:` URIs), `iter_published_frames`.
- `online_mixing.py` — the online-mix compiler: policy YAML → one infinite
  `dload.Pipeline` of `td.Frame`s (`build_online_mix_pipeline` =
  `build_noise_stream` + `build_speech_stream`); `make_rng(seed, sample_id)`.
- `frame_datasets.py` — torch adapters: `DregonLMFrameDataset`,
  `DNLMFrameDataset`, `SEValidFrameDataset`, `OnlineMixFrameDataset`,
  `NoiseGenFrameDataset`, `StaticCombGenDataset`, `DecompFrameDataset`
  (`dataset` is a **list**; each record keeps its `drone` rig id) + the
  validation compositors (`FixedSynthFrameDataset`, `ConcatFrameDataset`, …).
- `noise_rps_dataset.py` — `NoiseRPSDataset`: chunkable noise+RPS over
  DREGON `in_flight_noise` + Michael's (`frames:NAME[@VER]`, `dload:`, paths).
- Noise engines (`sources.noise[].kind` → class): `generated` →
  `generated_noise.GeneratedNoisePool` (CUDA + shm ring); `gp` →
  `gp_noise.GPRotorNoisePool`; `static_comb` →
  `rotor_spectral_model.StaticCombNoisePool`; `stochastic` →
  `stochastic_rotor_noise.StochasticNoisePool`; `noise_v2` →
  `noise_v2_pool.NoiseV2Pool` (FITTED v2 rigs; `fits:`/`preset_bank:`);
  `silence` → `silence_noise.SilenceNoisePool`.
- `noise_model/` — v2/v3 renderer (numba line kernel; prior-rig blocks
  `am`/`mic_dev_sd_db`/`wind_sc`), re-exported by `experiments.noise_model`.
- `trajectory_model/` — the FITTED rps trajectory model: `rps.kind:
  fitted_traj` + `FlightCache`/`window_flight`, fed by `rps-traj-fits`
  (fitted by `experiments.rps_traj`).
- Augmentation: `noise_augmentations.py` (`policy.noise_augmentations`),
  `rate_warp.py` (`policy.noise_rate_warp`), `rps_corruption.py` (clean-RPS
  corruption → extra `rps_cond`; label-noise `tachometer_corrupt`/
  `presmooth_track`).
- `rps_gating.py` + `refined_label_track.py` — refined rps labels
  (`refined_labels/`, from `scripts/refine_dregon_rps.py`).
- `harmonicity.py` (`measure_harmonicity`), `rps_synthesis.py`/`collate.py`
  (mixer constants from `tracking.rotors`), `comb_bench*.py`.

## Contracts / invariants

- **Fixes are baked in at derivation time.** `DREGON-frames`
  `motors_command` is `clean_command_spikes`-cleaned; `michaels-frames` `rps`
  is aligned and rev/s-calibrated (`MICHAELS_RPS_SCALE` in
  `load_raw_aligned`). Consumers re-apply nothing (`adapt_recording_frame` →
  `rps`; `resolve_motor_tracks` no-cleaning path).
- Published `*-frames` carry TWO rps tracks: the raw one
  (`PUBLISHED_RPS_KEYS`; `*_room2` log only `motors_command`) and
  `rps_refined` (regime-gated, standby never refined; on the AUDIO `t_start`).
- Telemetry is time-last `(…, M)` on `StampIndex`-backed Series.
- Two SNR conventions: offline/LibriMix (`mix_at_snr`, speech is the ref) vs
  online (`scale_source_to_snr`, noise is the ref).
- Manifest layouts, on `meta["layout"]`: `sample-dir-v1` (`sample_NNNNN/`
  dirs + `_meta` sample; `dregon_lm`, `dn_lm`, `real_valid_panel`),
  `tdframe-v1` (one Frame per recording/clip; all frame generators),
  `raw-files`, `pcm16-mono-v1`.
- `adopt_only` specs are historical uploads whose bytes predate the spec;
  re-deriving would push a near-duplicate (mixing RNG is not byte-stable).
- **TEST data**: `michaels-test-frames` (FLY103/FLY108) and the DREGON
  `test` split (`TEST_RECORDING_IDS`: speech-high, whitenoise-high,
  silent-flight). No training or validation derivation/pool may root on them;
  `load_noise_source_frames` and `real_valid_panel` raise. Published
  `DREGON-frames` `meta.split` predates the split (still `in_flight_source`/
  `noise_free`): select by `recording_id`, never by `meta.split`.
- `rp-real-valid-v1` (`real_valid_panel`) = the `rp_*` panel: DREGON room1
  22 clips, FLY124 15 raw + 15 dev-clean mixes, a `meta.json` each
  (`REAL_VALID_PANEL_META_KEYS`); per-clip key `clip_index`.
- Online-mix determinism: content is deterministic per `(base_seed, epoch,
  worker, position)`; curriculum/augmentation decisions are pure functions of
  the global sample id. `kind: generated` + `refresh: true` is not.

## Entry points

- **Publish**: `python scripts/derive.py list -v | derive <NAME> | adopt
  <NAME> --commit`, then `dload pin NAME && git add dload.lock`. Raw trees:
  `dload commit NAME --from data/NAME` + a `REGISTRY` `raw_dataset`. Bump
  `recipe_version` on any behavioural change — the fingerprint keys on it.
- **Train on a published split**: `DregonLMFrameDataset` → one `td.Frame` per
  sample: `mixture` `(time,)` / `(mic, time)`, `rps` `(rotor, time)` on the
  STFT grid, `meta` = every key of the sample's `meta.json` (else
  `recording_id` = dir name). `channel=<int>` picks one mic;
  `flatten_channels=True` expands to mono Frames (`meta.channel`).
  DCUNet/DCCRN stay 1-channel. Streamed: `DloadFrameDataset`.
- **Train on the online stream**: `OnlineMixFrameDataset.from_yaml(policy)`
  via a `conf/data/*.yaml` wrapper; infinite, so the experiment sets
  `samples_per_validation`; validation stays a fixed map-style set.
  `flatten_channels` = `flat_map`, `rps_corruption` = `map`.
- **Noise pool spec** (derivations + `kind: frames`,
  `load_noise_source_frames`): `dataset` (`NAME[@version]`), `splits`/`split`,
  `recording_ids`, `exclude_recording_ids`, `take`, `min_motor_rps`,
  `channels`, `weight` (explicit = own sub-pool; unweighted reals merge
  duration-weighted).
- **Check a policy**: `python scripts/check_stream.py --experiment <name>` —
  chunk→frame expansion, stage boundaries, empirical fire rates; nonzero exit
  on FAIL. Mandatory after a data refactor.
- **RPS helpers**: `mixing.resolve_motor_tracks(tf)`,
  `mixing.find_inflight_window(tf, key, min_motor_rps, clean=…)` (pass 30.0).

## Conventions

- Torch stays out of layers 1–2 so fingerprinting and `adopt` run anywhere.
- Public interface is config-in/stream-out: no source-cache prep scripts or
  cache flags; the memoized speech decode is `librispeech-pcm16`.
- Publish a derived subset (`frame_subset`) for a durable noise pool;
  `audio_pool` `include_keys` is ad-hoc only (it pulls every shard to match).

## Gotchas

- Datasets are gitignored — stream (`dload:`) or `dload pull <name>`.
- `michaels_dir` in `conf/data/noise_rps_dregon_michaels*.yaml` is stale
  (`data/new-drone-noises` = FLY103/FLY108, the TEST set); don't copy it.
- `load_timeframe(target_sr=…)`: resample with `axis=-1` (`soxr_hq`); the
  wrong axis hangs.
- `real_valid` splits have no `vocals.wav` — RPS evaluation only.
- V4-Michaels leakage split: train = DREGON without
  `free-flight_nosource_room1`, plus `FLY125`; validate = room1, `FLY124`.
- Michael's telemetry published before 2026-07-31 is uncalibrated
  (`rps_scale` 1.0 in frame meta).
- `SPCUP19-egonoise` = historical pin; `SPCUP19-frames` is built from
  `sources/spcup19_meta/<Team>.yaml` (report tags + geometry).
- Pre-rp online-mix policies are archival (removed augs raise at build);
  use `rp_*`. Ramps + native-rate `noise_rate_warp`: refactor doc
  § "Augmentations".
- `rps.kind: fitted_traj` draws ONE drone per flight (`rigs:` weights;
  reserved name `posterior` = the global fit) and shifts its mean; the ESC
  clamp and the idle level move with it. Bad policies fail at pool build.

## Further reading

- `docs/refactor-data-pipelines.md` — architecture, § "Online-mix policy
  reference" (every `kind`, holdouts, augmentations, SE mode, `check_stream`).
- `docs/data-catalog.md` — DREGON inventory + `test` split, Michael's
  calibration, dataset variants (incl. `rp-real-valid-v1`), every pin.
- `docs/data-and-artifacts.md` — dload/R2, caches, omnirun;
  `docs/{derived-datasets,external-datasets}-plan.md`.
