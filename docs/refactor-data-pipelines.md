# Refactor: one dload-pipeline data layer

Branch `refactor/data-pipelines`. Status: complete (review pass applied — see
"Review follow-ups" at the end).

## Problem

The data layer grew three parallel generations of machinery that duplicate each
other:

1. **Bespoke per-dataset loaders** — `dregon.py` (656 LOC) and `michaels.py`
   (232 LOC) get privileged loader modules with their own downloaders, geometry
   getters, and telemetry fixups, while the 10 other external datasets share the
   uniform `external_datasets.py` registry (download spec + builder →
   `tdframe-v1`). `external_recordings.py` (223 LOC) is dead code.
2. **Bespoke publisher scripts** — `publish_frame_datasets.py` (DREGON/michaels),
   `publish_external_datasets.py` (10 externals), `publish_avq_{raw,egonoise,
   vkrps}.py`, `publish_beatvk_valid.py`, `build_se_valid.py` — six drivers all
   doing "iterate samples → `repo.commit` with a layout meta".
3. **Bespoke mixed-dataset CLIs** — `create_dregon_librimix.py` (1777 LOC, 30
   flags), `create_dregon_librimix_v3.py`, `create_dataset.py`, `dataset.py`
   (ZFTurbo legacy) — duplicating the same audio helpers (`load_audio`,
   `adjust_length`, `calculate_snr`, `mix_at_snr`) in three files, while
   `derivations.py` already re-declares the same recipes as dload pipelines and
   imports the CLI cores through a `sys.path` shim.

Consumption is already uniform (`streams.DloadFrameDataset`, `dload:` URIs,
`frames:` specs, online-mix `kind: frames`); creation is not.

## Target architecture

Every dataset is defined **once**, in exactly one of two registries, and
materialized **exclusively** through dload:

```
external origin (zenodo / hf / mendeley / gdrive / http / local-only)
   │  sources registry entry: fetch spec + builder(raw_dir) → td.Frame stream
   ▼
raw dload dataset            (only for project-local raws: DREGON,
                              new-drone-noises, librispeech, …)
   │  derivations SPECS: generate_source_frames / generate_*_lm_split /
   │  generate_frame_subset / generate_se_valid / … (dload pipelines,
   ▼  frozen JSON specs, recipe_version, parent pins in the fingerprint)
derived dload datasets       (*-frames, DREGON-LM-*, DN-LM-*, SE-valid-*,
                              AVQ-egonoise*, …)
   │  streams.py (DloadFrameDataset / dload: URIs / frames: specs)
   ▼
training & eval
```

- `src/data_processing/sources/` — **the** external-dataset registry. One
  module per dataset (DREGON and michaels included — no preferential
  treatment), each exporting a uniform `SourceDataset` entry: `DownloadSpec`
  (or `None` for local-only raws already committed to dload) + `builder`.
- `src/data_processing/derivations.py` — **every** derived dataset as a frozen
  pipeline spec. Historical pins are adopted in place; fresh work derives.
- `src/data_processing/mixing.py` — the pure per-sample mixing cores (hoisted
  out of the deleted CLIs; torch-free), shared by the derivation generators.
- `scripts/derive.py` — the single driver: `list` / `derive` / `adopt`.

## Deleted

`scripts/{create_dregon_librimix,create_dregon_librimix_v3,create_dataset,
dataset,build_se_valid,publish_frame_datasets,publish_external_datasets,
publish_avq_raw,publish_avq_egonoise,publish_avq_vkrps,publish_beatvk_valid}.py`;
`src/data_processing/{dregon,michaels,external_datasets,
external_recordings}.py`; all `.ipynb_checkpoints/`.

Dataset *recreation* after this refactor: add/edit a spec, then
`python scripts/derive.py derive <NAME>`. Historical recipes remain in git
history and in the adopted specs' `note` fields.

## Consumer migrations

- online-mix noise sources `kind: dregon` / `kind: michaels` (local-dir
  loaders) are removed; every real-recording pool is `kind: frames` over a
  published frames dataset (fixes are baked in at derivation time).
  `conf/online_mix/*.yaml` migrated accordingly.
- `noise_rps_dataset`, `frame_datasets`, `generated_noise`, the GP experiments
  and the notebook libs: geometry/loaders come from the sources registry or
  published frames — never from bespoke loader modules.
- `src/localization/` (near-field SRP-PHAT) deleted: it had no callers left and
  its finding (separation, not localization, is the wall) is recorded; the code
  remains in git history.

## Online mixing: pipelines, not pool classes

`online_mixing.py` re-implemented sampling imperatively (TimeFrameNoisePool,
MixedNoisePool, AudioFileSourcePool, DloadAudioPool — weighted record choice,
shard LRU, packed caches, redraw loops). All of that is dload combinator
composition:

- duration-weighted record choice → `dload.choice(per-record window streams,
  weights=durations, seed=...)`;
- random window cut → `random_stream(seed).map(cut_window)` (one uniform per
  chunk);
- speech files → `ds.samples().shuffle(buf, seed).repeat().map(decode+cut)`
  with the cut drawn from `dload.seeded(key, "window")` (the canonical
  per-key randomness idiom); C independent channels = `.window(C)` + stack;
- the bespoke `packed_int16` speech cache → a **derived dataset**
  (`librispeech-pcm16`, `Repository.derive` — memoized preprocessing is what
  derive IS); decode dispatch (flac vs pcm16) by manifest layout, one code path;
- `audio_pool` holdouts → shard-subset manifests (`dload.Dataset(repo,
  manifest_subset)`), key filters → `.filter(...)`, non-audio redraw →
  `.filter(...)`;
- augmentations firing → per-sample-id RNG (`make_rng`) kept exactly (the
  check_stream control-stream methodology depends on draw-count stability),
  driven by an explicit id stream (`from_iterable(count, shard=True)` —
  IterableSourceNode striping reproduces the old `worker_id + k*num_workers`
  global-id assignment exactly);
- generated/GP/static-comb noise → `random_stream(seed).map(synthesize)`;
  the neural generator's CUDA producer/ring-buffer is the one genuinely
  stateful resource and stays behind one factory function.

The train stream is one infinite `Pipeline` consumed through
`dload.torch.as_iterable_dataset`; `OnlineMixFrameDataset.from_yaml` keeps its
Hydra-facing signature (configs unchanged in shape) but is a thin wrapper over
`build_online_mix_pipeline(cfg)`. `flatten_channels` becomes a `flat_map`,
`rps_corruption` a `map` reading the sample id from the item (no lockstep
id recomputation).

## Online-mix policy reference

The per-source detail formerly in `src/data_processing/AGENTS.md`, aligned
with the compiler above. A policy YAML (`conf/online_mix/*.yaml`) is consumed
by `OnlineMixFrameDataset.from_yaml` (Hydra-facing; `conf/data/*.yaml` wraps
it, e.g. `conf/data/online_mix_v4_michaels.yaml`) over
`build_online_mix_pipeline(cfg)`. The stream is infinite, so the experiment
must set `samples_per_validation` (top-level Hydra field, `conf/config.yaml`)
— it defines the validation cadence/"epoch" size. Validation stays a fixed
map-style set (`valid:` entry of the data config, e.g. `DregonLMFrameDataset`
over a published valid split); never early-stop on an advancing online stream.

Top-level keys read by the compiler: `sample_rate` (16000), `duration_s`
(1.0), `n_fft`/`hop_length` (2048/512 — the model's own STFT grid, used by
RPS-target interpolation and the STFT-domain noise augmentations),
`base_seed`, `start_sample_id`, `task` (`rps_prediction` | `speech_enhancement`),
`snr_ref_floor_rms` (reference-power floor on the speech scaling in
`mixing.scale_source_to_snr`, so zero-RPS chunks are not globally quiet —
pairs with `kind: silence`), `sources.noise` (one spec or a list),
`sources.speech` (`dataset` default `librispeech`, `subpath` default
`LibriSpeech/train-clean-100`, `include`/`exclude` speaker-id path-token
filters, `shuffle_buffer`, `version`), `policy` (curriculum stages,
`augmentations`, `noise_augmentations`, `noise_time_warp`, speech-lane
options such as `speech_per_channel: independent`).

**Pool weighting** (`build_noise_stream`). Unweighted real (`kind: frames`)
sources merge into one duration-weighted `dload.choice` over per-record window
streams; the merged pool's weight is the number of merged items. A real source
with an explicit `weight`, and every engine/`audio_pool` source, becomes its
own sub-stream at that pool-level weight (default 1.0). That is the knob that
lets a long auxiliary corpus enter at a modest share — e.g. `AVQ-egonoise-vkrps`
(~617 s) at `weight: 0.5` vs the merged DREGON+michaels 2.0 in
`conf/online_mix/beatvk_avq_dload.yaml` (= 20 % of noise chunks; being mono,
~3 % of training frames under `flatten_channels`). The **channel ceiling**
(max kept channels over real records; 8 for engines; `n_channels` for
`kind: silence`) sizes the speech-lane count under `speech_per_channel:
independent` — a mono real pool draws ONE speech lane and
`flatten_channels=True` yields ONE training frame per chunk.

### `kind: frames` — published rich-frame recordings

The fixed rich-frame datasets published by the `source_frames` derivation
(`DREGON-frames`, `michaels-frames`, `michaels-test-frames`; `tdframe-v1`,
decoded by `data_processing.streams`) feed the pool directly through
`mixing.load_noise_source_frames` — the **same spec schema** a derivation
uses, so a derivation and a training policy select noise identically:

| Key | Meaning |
|-----|---------|
| `dataset` | published frames dataset name, optionally `NAME@version` (required; default version = `dload.lock` pin) |
| `splits` / `split` | filter on frame `meta.split` (e.g. `in_flight_noise`) |
| `recording_ids` / `exclude_recording_ids` | keep / drop recording ids (bare published ids, e.g. `FLY125`) |
| `take` | cap the number of recordings |
| `min_motor_rps` | in-flight window threshold (30.0) |
| `channels` | int or list — keep only those audio channels of every chunk, applied before augmentation and mixing; the mic count is a per-source property (the reality ladder's 1-mic vs 8-mic rungs, `conf/online_mix/real_r1_dload.yaml` vs `real_r2`) |
| `weight` | explicit pool-level weight (see above) |

```yaml
sources:
  noise:
    - kind: frames
      dataset: DREGON-frames
      splits: [in_flight_noise]
      exclude_recording_ids: [free-flight_nosource_room1]
      min_motor_rps: 30.0
    - kind: frames
      dataset: michaels-frames
      recording_ids: [FLY125]
      channels: [0]
```

**Fixes are baked in at publish time** — DREGON `motors_command` is already
`clean_command_spikes`-cleaned, michaels `rps` already aligned and rev/s
calibrated (`docs/data-catalog.md`) — so the loader re-applies nothing: it
renames the rotor track to the generic `rps` entry (the no-cleaning path of
`mixing.resolve_motor_tracks`), keeps only `audio` + `rps` + `meta` per
recording (IMU/GPS/raw telemetry dropped, one frame decoded at a time), and
soxr-resamples audio to the pool `sample_rate`. After adaptation only ONE
rotor track survives, so detection and labelling use the same one —
`motors_measured` where it exists, else the cleaned `motors_command`, whose
trailing logging freeze is then not trimmed. `noise_rps_dataset.build_noise_rps_datasets`
likewise accepts `dregon_dir="frames:DREGON-frames[@VERSION]"` /
`michaels_dir="frames:michaels-frames"` in place of local folders.

### `kind: audio_pool` — telemetry-free audio datasets

A dload-backed audio dataset as a noise pool **without** any rotor telemetry:
random recording (shard weighted by sample count), random channel, resample
to the pool rate, loop/pad to the chunk. Streams lazily at *shard* granularity
(never materializes the dataset — works on 258 GiB MIMII / 88 GiB
DroneAudioSet). Handles both `tdframe-v1` (audio under `audio`) and raw-audio
datasets; skips non-audio samples (e.g. `new-drone-noises` csv flight logs);
zip-blob datasets (`zenodo_drone_noises`) unsupported. Usable only for
`task: speech_enhancement` (no rotor track).

```yaml
sources:
  noise:
    - kind: audio_pool
      dataset: MIMII                 # any dload dataset name
      channel: random               # or an int
      holdout: {split: train, valid_shards: 2}   # leak-free train/valid split
      include_keys: [S1_seq1]       # optional: restrict to named recordings
      exclude_keys: []              # optional: drop named recordings
      max_shards: null              # optional cap
      weight: 1.0
```

`include_keys` / `exclude_keys`: a sample key is kept when it equals or
*contains* a listed entry, `exclude_keys` applied last. Keys are shard-local
(the manifest has no key list), so filtering stays shard-lazy: a shard is
dropped (draw weight zeroed) the first time it is opened and found to hold no
match. **Cost**: every shard must therefore be downloaded at least once before
the pool knows it holds nothing wanted — filtering a many-shard dataset down
to a few recordings pays the full download for a tiny slice (the F2
replication paid ~4 GiB across 11 AVQ shards for ~12 minutes of audio, and hit
the multipart-ETag rejection on Kaggle). Rule of thumb: `include_keys` for
ad-hoc restriction; publish a derived subset (`frame_subset` generator —
`AVQ-egonoise` is the template) once a fixed small selection becomes an
experiment's durable noise pool.

`holdout` reserves the last `valid_shards` whole shards (= whole recording
groups) as the *valid* partition and the rest as *train*; single-shard
datasets fall back to a per-shard sample-index split at `fraction` (0.1).
`derivations.generate_se_valid` uses `split: valid` with the same
`valid_shards` to build the fixed SE valid sets, complementary to the training
pools' `split: train`; `sources.speech.exclude` holds the `SE_HELDOUT_SPEAKERS`
out of training speech and the derivation `include`s exactly those.

### `kind: generated` — trained neural noise generator

A trained `PositionalHarmonicNoiseGen` listed as a `sources.noise` entry
exactly like a real recording: unlimited rotating-noise variety with an exact
RPS label (`generated_noise.GeneratedNoisePool`; example
`conf/online_mix/online_mix_generated_augment_example.yaml`).

Process/buffer design: the mixer runs in **forked** DataLoader workers, and
CUDA cannot init in a forked child. So one **spawn** producer owns the single
generation CUDA context and renders batches into a **shared-memory ring
buffer** (torch shared tensors); fork workers only read finished chunks. Reads
are lock-free via a per-slot **seqlock** (`version` odd = writing; reader
retries if odd or changed across its copy). Generation rate is decoupled from
consumption — workers sample-with-replacement from filled slots, so a slow GPU
just means more chunk reuse. The producer starts once in the main process
(never in a worker); `close()`/`atexit` tears it down.

Config (defaults): `checkpoint` (bundle path or `r2://` URI, auto-downloaded
via `training.artifacts.resolve_checkpoint_uri`; required); `drone` (codebook
key + geometry source, `michaels`/`dregon`; set `dregon_dir: dload:DREGON` on
cloud so the producer can load DREGON geometry); `n_harmonics` (**must match
the checkpoint**); `device` (`cuda:0`); `gen_batch` (32); `random_phase`
(true — per-chunk harmonic phases, model stays in eval); `refresh` (true;
**false** = fill the buffer once for a reproducible fixed bank); `rps.kind`
(`synthetic_intermittent` = cruise-only, or `full_flight` =
ground→warm-up→takeoff→cruise→landing→ground windows, which drive the
generator into silence at zero RPS — the static-comb source takes the same two
kinds; `conf/online_mix/e10_full_flight_dload.yaml`), `rps.flight_fs` (200.0),
`rps.aggressiveness` (1.0); `buffer.slots` (512 ≈ 384 MB), `buffer.warmup`
(16); `weight`. Determinism caveat: a live (`refresh: true`) stream is **not**
seed-reproducible (buffer contents depend on timing) — keep validation on
real/fixed sources or use `refresh: false`.

**Vicinal `interp` mode** (E7, `conf/online_mix/rps_generated_only_interp.yaml`):
an `interp:` sub-block makes each producer batch sample a *novel* drone along
the DREGON↔Michael's embedding segment. Per batch: `α ~ U(alpha.low,
alpha.high)`; `z = (1−α)·z0 + α·z1 + N(0, embedding_noise·‖z1−z0‖)`;
`rotor_interp` linearly interpolates rotor positions at α; `jitter_sigma:
interp` blends the learned per-drone OU σ at α (or a float, or `off`; forced ON
at eval); `mic_sampling` picks a rig (`rigs`, `prob`) independently of α and
jitters each mic by `N(0, jitter_std m)`; `endpoints` are codebook names; α
also feeds the `rps_synthesis` `drone_profile` blend. Two per-batch knobs widen
the *rotor* axis (both default 0; `conf/online_mix/m3cur_s1_dload.yaml`):
`perrotor_noise` replaces the checkpoint's per-rotor code deltas with
`δz_r + N(0, perrotor_noise·RMS_r‖δz_r‖)`, and `rotor_jitter_std` adds
`N(0, σ m)` per rotor per coordinate to the interpolated rotor positions.
Requires a **flat conditioned checkpoint** (`_CodebookConditionedNoiseGen`
state_dict with `codebook.codes.*` + optional `log_jitter_sigma.*` + optional
`rotor_deltas`; the modern `training.loop` format) — `_load_generator` rebuilds
the composite via `models.registry.build_noise_gen_model` (a `rotor_deltas
[R, d]` key switches `per_rotor_deltas` on, and the producer conditions each
rotor's emitter on `z_r = z_drone + δz_r`); the reduced `save_bundle` (no σ) is
single-drone only.

Benchmarks: generator inference (236k params, mostly FFTs) is ~128 ms per 1 s
8-mic chunk on CPU batched; GPU far faster, hence the GPU producer. On the
V4-Michaels setup (`batch_size=16`, `num_workers=4`, `speech_per_channel=
independent`) direct FLAC decode ran ~2.9 batch/s; the historical
`packed_int16` speech cache reached ~13.5 batch/s (1728 clip/s) — its
replacement is the `librispeech-pcm16` derived dataset; a fixed precomputed
loader runs ~21 batch/s (5394 clip/s). Optimize only behind the same public
config-in/stream-out interface. Early smoke runs used generated
`DREGON-LM-V4-michaels/train/**/vocals.wav` as speech — obsolete; use the
original LibriSpeech pin.

### `kind: gp` — per-drone egonoise GP (G3)

`gp_noise.GPRotorNoisePool`: the per-drone egonoise GP checkpoints (trained by
`src/experiments/gp_rotor_noise/train_egonoise_gp.py`, inference core
`src/data_processing/egonoise_gp.py`; `r2://ml-data/artifacts/gp_egonoise/{dregon,matrice100}/best.pt`)
with exact synthetic RPS labels. Mirrors the static comb, not the neural
producer: the GP posterior is batch-queried **once at pool init** on a dense
rps grid at the rig mic positions and reduced to a `(G, M, 2H+1)` coefficient
table (~1.6 MB, picklable; the gpytorch model is dropped), so per-chunk
synthesis is pure numpy in the fork workers (~200 ms per 1 s 8-mic chunk):
rps-interpolate coefficients at the chunk-mean rps, FM-synthesize the comb at
`render_fs` 24 kHz (anti-aliased), add the checkpoint's σ_b(rps) colored
broadband, decimate to 16 kHz, normalize global RMS. Config: `checkpoint`
(r2:// ok), `drone` (geometry + `rps_synthesis` profile), `mic_mode: shell`
(default — rig mics projected radially onto the GP's training shell; native
rig positions are ~3 lengthscales out-of-support and mean-revert),
`rotor_mode: per_rotor` (default — Σ_r S(mic, rps_r)/R, non-degenerate
per-rotor labels; `mean` = four_way_lib convention, degenerate labels),
`broadband`, `rps.kind: synthetic_intermittent` **only** (GP support is rps
40–85; no full-flight excitation). Policy `conf/online_mix/g3_gp_aug_dload.yaml`;
`docs/experiments/g3-gp-curriculum.md`.

### `kind: static_comb`, `kind: stochastic`, `kind: silence`

Analytic engines, documented in their modules: `rotor_spectral_model.py`
(`StaticCombNoisePool`; `FixedCombSpec`/`render_fixed_comb` for the
frozen-profile generator probe; `rps_scale_range`, `normalize_rms`),
`stochastic_rotor_noise.py` (`StochasticNoisePool` — colored floor + one
Lorentzian line per harmonic per rotor with GP-drifting amplitudes drawn
independently of the rotor-speed trajectory; the generative direction of
`tracking.joint_decompose`; knobs `rps_scale_range`, `normalize_rms`,
`aggressiveness` range, `rps.phases`; `docs/experiments/stochastic-transfer.md`),
`silence_noise.py` (`SilenceNoisePool` — per chunk one of `room_tone` /
`colored` / `lf_rumble` floors, independent per channel, `rps` all zeros;
`docs/experiments/honest-base-frontends.md`).

### `policy.noise_augmentations` — strong noise-chunk augmentations (G6)

`noise_augmentations.py`: six strong transforms of the **noise chunk** (audio +
RPS pair, before speech mixing — unlike `policy.augmentations`, which is
post-mix on the mixture and provably weak for RPS prediction: polarity is an
exact no-op for mag/IF front-ends, gain a log-offset). Same `probability` +
`choices` fire/choice schema; on a hit the chunk Frame is rebuilt in
`time_warp.apply_time_warp`'s output convention (audio exactly `target_len` +
a clean uniform 100 Hz `rps` track). The six: `freq_scale` (resample by
α∈U(0.75,1.3) at natural scaled length — the sourcing pipeline oversamples the
noise window by α_high so the crop never pads; labels ×α — the one that
manufactures new (audio, RPS) pairs), `spectral_recolor` (smooth random EQ
±8 dB, 10 log-spaced anchors, per channel), `random_reverb` (deterministic
200-RIR synthetic bank: RT60 U(0.1,0.8) s, DRR U(3,15) dB; RMS renormalized),
`tooth_dropout` (zero ±2 STFT bins around k·rps_r(t), 1–4 random teeth k≤25 —
label-aware), `spec_mask` (SpecAugment bands/time masks), `floor_inject`
(1/f^tilt floor at U(−20,0) dB rel RMS). STFT ops run on the model's 2048/512
grid. Keep the 50k unaugmented warmup stage — G5 measured a real regression
without it. Policy `conf/online_mix/g6_strongaug_dload.yaml`;
`docs/experiments/g1-vk-parity.md` § "Phase G6"; tests
`tests/test_noise_augmentations.py`.

### `task: speech_enhancement` — SE target mode (F1 baselines)

With `task: speech_enhancement` the stream yields `(mixture, clean_speech)`
instead of `(audio, rps_target)`: the clean target is the gain-scaled speech
exactly as mixed (SNR of the pair == the drawn SNR), post-mix augmentation
(`random_gain`/`random_polarity`) is applied identically to mixture and
target, and RPS interpolation is skipped (telemetry-free noise sources work).
The stream is **mono** — a random mic channel is picked from multichannel
noise. `OnlineMixFrameDataset` packs each pair into a `{mixture, target, meta}`
Frame (the DN-LM layout the SE task / `losses.MaskedLoss` consume). Speech is
always drawn. See `conf/online_mix/se_{drone_only,all_harmonic}.yaml`,
`docs/experiments/f1-se-blind-baselines.md`. The map-style `SEValidFrameDataset`
streams a published SE valid set as the same frames for `eval.py`;
`local_root=<dir>` reads an unpublished set from a local dload repository
(`streams.local_repository`).

### Stream sanity-checking — `scripts/check_stream.py`

**Run it against any new or edited policy before submitting training jobs,
and re-run it whenever the data path is refactored** (loader, wrapper,
flattening, policy resolution). It exists because of the CKLA staging bug
(`docs/experiments/ckla.md` § "THE STAGING BUG"): `flatten_channels=True`
turns each generated chunk into C=8 training frames, so `until:` stage
boundaries (in chunk units) sat at effective epoch ~80 instead of 10 and the
staged augmentations silently never fired for ~3 weeks — config inspection
and the provenance print both looked correct; only the stream showed the truth.

```bash
python scripts/check_stream.py --policy conf/online_mix/<name>.yaml \
    --flatten --samples-per-epoch 5000 --epochs 0 5 10 12 [--probes 48]
python scripts/check_stream.py --experiment <name>   # exactly what train.py trains on
```

It measures, on the actually-generated stream: (1) the chunk→frame expansion
ratio C; (2) each stage boundary in chunks / frames / effective epochs (WARN
beyond `--warn-boundary-epoch`); (3) **empirical** per-key fire rates of
`augmentations` / `noise_augmentations` / `noise_time_warp` at the probed
epochs vs the configured probabilities (exact binomial test), plus the
label-diff rate (`freq_scale` must change labels on every nonzero-RPS fire —
all-zero-RPS full-flight chunks are exempt; post-mix augs must never touch
labels); (4) per-id determinism. Nonzero exit on any FAIL — usable as a
submission gate. Fire detection compares each sample against a control stream
whose aug block has `probability: 1e-9` (never removes the key — a
present-but-missed block still consumes the fire-decision RNG draw, so
stripping it would shift the whole downstream RNG stream and fake a ~1.0
rate). Known cost: it rebuilds the whole pipeline per probed epoch and per
control stream; a `kind: generated` policy spawns a CUDA producer per rebuild.

## Verification gates

- `pytest tests/data_processing -q` green (baseline: 169 passed).
- `pytest tests -q` green (630 passed).
- `python scripts/check_experiment_configs.py` — all experiments compose
  (150 OK / 0 FAIL, unchanged).
- `ruff check src scripts tests` / `ruff format --check` at the pre-refactor
  baseline (8 pre-existing findings, all in `src/models/multif0`).
- `python scripts/check_stream.py --experiment <one online-mix experiment>`
  unchanged stream behaviour for a migrated policy.
- Line-count delta measured per area (reported in the PR).

## Review follow-ups (applied)

Gaps found reviewing the refactor, and how they were closed:

1. **`generate_se_valid` was a stub that raised**, while
   `scripts/build_se_valid.py` (its only predecessor) was deleted — the SE
   validation sets could not be rebuilt, and the F1 sets are known to need a
   rebuild after the silent-draw fix. Implemented on the new stream builders:
   `iter_se_valid_category` composes `build_noise_stream` + `build_speech_stream`
   with the complementary filters (valid-side holdouts, `include`-ing exactly
   `SE_HELDOUT_SPEAKERS`) and the SNR grid driven off the sample index.
   `build_speech_stream` gained the `include` filter to make that expressible
   in the same schema training policies use.
2. **The `drone_seen` probe category was dropped** and
   `notebooks/generalization_lib.py` still imported `build_se_valid` (broken).
   Category restored to `SE_CATEGORY_NOISE`; the notebook lib now calls
   `iter_se_valid_category` (public for exactly this reason).
3. **`librispeech-pcm16` was referenced but never defined** —
   `_decode_speech_chunk` dispatches on the `pcm16-mono-v1` layout, but no spec
   produced it, so the deleted `packed_int16` speech cache (~4.6× decode
   throughput on the 8-lane RPS streams) had no replacement. Added the
   `pcm16_mono` generator + a derivable `librispeech-pcm16` spec.
4. **`sources.michaels.MICHAELS_FILES` changed root convention** (now relative
   to the `recording_with_motor_speed` tree) without updating every caller —
   `tests/test_geom_calibration.py` failed. Added
   `michaels.resolve_raw_root()`, which accepts the raw tree, a `dload:` URI,
   an enclosing `data/` dir, or `None` (the dload pin), mirroring
   `dregon._dregon_dir`; all callers go through it.
5. **`generate_beatvk_valid` called `td.Frame.get`**, which does not exist —
   it would have crashed if ever run. Fixed to `frames.get_meta`, and the
   thrice-duplicated `dload:NAME@sha` parsing folded into `_split_dload_uri`.
6. **`_apply_one_augmentation` / `_apply_one_augmentation_pair`** were ~90 lines
   of near-duplicate; merged into one function over a tuple of signals (the RPS
   stream passes the mixture, the SE stream mixture + target). The fire/choice
   draw is factored into `_augmentation_draw`, so the draw order the
   `check_stream` control-stream methodology depends on is unchanged.
7. **`mixing.mix_at_snr` / `mix_audio` / `generate_white_noise`** each carried
   their own copy of the noise-scaling math; all three now call
   `scale_noise_to_snr` (the offline mirror of `scale_source_to_snr`).
8. **`audio_pool` `include_keys` were silently ignored** whenever the holdout
   degenerated to the per-shard index window; the filters now compose.
9. Dead back-compat aliases (`_resolve_motor_tracks`, `_scale_source_to_snr`,
   …) removed and their few callers repointed at `data_processing.mixing`;
   `_holdout_shards`' `(0, 1)` sentinel replaced by a bool; the `.env` load
   path in `online_mixing` fixed (it pointed at `src/.env`).
10. Docs realigned with the code: root `AGENTS.md`, `src/AGENTS.md`,
    `src/data_processing/AGENTS.md` (Files / Dataset Variants / RPS Processing /
    Publishing sections rewritten), `docs/data-and-artifacts.md`, and the
    `create-dregon-dataset` skill, which still documented the deleted CLIs.

Known remaining gaps (not addressed here):

- `notebooks/explore_data.ipynb` imports the deleted `create_dataset` module.
- `README.md` / `REPLICATION.md` still cite the deleted creation CLIs as build
  commands. They are historical replication records, so the citations are
  provenance rather than instructions — but they read as instructions.
- `check_stream.py` rebuilds the whole pipeline per probed epoch and per
  control stream; for a `kind: generated` policy that spawns a CUDA producer
  process per rebuild.
