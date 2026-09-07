# Data catalog — recordings, telemetry, derived datasets, pinned names

Reference companion to `src/data_processing/AGENTS.md` (the structural map) and
`docs/refactor-data-pipelines.md` (the architecture). This page holds the
*inventory*: what the raw recordings contain, how their telemetry was
calibrated, which derived-dataset recipes exist, and what every `dload.lock`
pin is. Publishing mechanics: `docs/data-and-artifacts.md`.

## DREGON recording inventory

All recordings are **8-channel, 44.1 kHz**; motor telemetry rate ≈ 929 Hz.

| Split | Recording ID | Duration | `motors_measured` | `motors_command` | Notes |
|-------|-------------|----------|-------------------|------------------|-------|
| `in_flight_noise` | `free-flight_nosource_room1` | 71 s | ✓ | ✓ | pure drone noise |
| `in_flight_noise` | `free-flight_nosource_room2` | 82 s | — | ✓ | pure drone noise |
| `in_flight_noise` | `hovering_nosource_room2` | 45 s | — | ✓ | hovering |
| `in_flight_noise` | `updown_nosource_room2` | 54 s | — | ✓ | up-down manoeuvre |
| `in_flight_noise` | `rectangle_nosource_room2` | 44 s | — | ✓ | rectangle path |
| `in_flight_noise` | `spinning_nosource_room2` | 41 s | — | ✓ | spinning |
| `in_flight_source` | `free-flight_speech-low_room1` | 63 s | ✓ | ✓ | drone + co-recorded speech (low level) |
| `in_flight_source` | `free-flight_speech-high_room1` | 53 s | ✓ | ✓ | drone + co-recorded speech (high level) |
| `in_flight_source` | `free-flight_whitenoise-low_room1` | 65 s | ✓ | ✓ | drone + co-recorded whitenoise (low) |
| `in_flight_source` | `free-flight_whitenoise-high_room1` | 61 s | ✓ | ✓ | drone + co-recorded whitenoise (high) |
| `noise_free` | `silent-flight_whitenoise-low_room1` | 78 s | — | — | no drone motors |
| `clean_source` | `clean_speech_*`, `clean_whitenoise_*`, `clean_chirps_*` | 1–10 s | — | — | isolated sources, no drone |
| `motor` | `motor_Motor{1-4}_{50,60,70,80,90}`, `motor_allMotors_70` | 25–45 s | — | — | individual/combined motor runs |

### Telemetry tracks: `motors_command` vs `motors_measured`

Both are time-last `(4, M)` Series (4 rotors), carried as `tdseries`
`StampIndex`-backed Series (values `.data`, stamps `.tindex.abs_stamps`).

- **`motors_command`** — commanded RPS from the flight controller. Present in
  every recording with motor data. Two logging artefacts:
  - *Leading freeze*: the first N samples are stuck at a constant high value
    before the real command sequence begins. `sources.dregon.clean_command_spikes`
    zeros this (and median-filters along time). Already applied in the
    published `DREGON-frames`.
  - *Trailing freeze*: the last 45–1577 samples are stuck at a constant value —
    the logger stopped before the motors spun down. Landing is **not** visible
    in command values; never use raw command tail samples as ground truth.
- **`motors_measured`** — actual measured rotor speeds. Only in the five
  `free-flight_*_room1` recordings. Shows the real spindown during landing
  (drops to ~55–78 RPS in the last seconds before the trailing freeze). Used
  for in-flight window detection whenever present.

Published `*-frames` datasets are reduced to a single `rps` entry by
`frames.adapt_recording_frame`, which prefers `motors_measured`
(`frames.PUBLISHED_RPS_KEYS`, since 2026-08-05). On the five room1 recordings
that log it, both the label track and the in-flight detection are therefore the
measured one; every other recording falls back to the cleaned command, whose
trailing freeze is then not trimmed (end trim effectively 0 s). The raw
`motors_command_raw` block is kept alongside as the uncleaned counterpart.

### In-flight window (takeoff / landing trim)

All recordings begin with ~5–13 s of pre-takeoff/ramp-up. Computed with
`min_motor_rps: 30.0` (the recommended value; the loader default is 0 for
backward compatibility — always pass 30.0 for new datasets):

| Recording | trim_start | trim_end | inflight | detect key |
|-----------|-----------|---------|---------|-----------|
| free-flight_nosource_room1 | 9.4 s | 0.2 s | 59.8 s | measured |
| free-flight_nosource_room2 | 8.1 s | 0.0 s | 70.7 s | command |
| hovering_nosource_room2 | 7.2 s | 0.0 s | 32.8 s | command |
| updown_nosource_room2 | 5.7 s | 0.0 s | 41.6 s | command |
| rectangle_nosource_room2 | 7.6 s | 0.0 s | 33.4 s | command |
| spinning_nosource_room2 | 8.7 s | 0.0 s | 28.0 s | command |
| free-flight_speech-low_room1 | 9.8 s | 0.4 s | 50.7 s | measured |
| free-flight_speech-high_room1 | 8.6 s | 0.4 s | 42.9 s | measured |
| free-flight_whitenoise-low_room1 | 12.2 s | 0.4 s | 49.9 s | measured |
| free-flight_whitenoise-high_room1 | 9.8 s | 0.4 s | 47.5 s | measured |

Implementation: `mixing.find_inflight_window(tf, motor_key, min_motor_rps,
clean=…)`; `mixing.resolve_motor_tracks(tf)` → `(detect_key, rps_key,
needs_cleaning)` picks `motors_measured` for detection when present, else
`motors_command`. In the `sample-dir-v1` mixes the saved `rps.npy` comes from
`motors_command` (cleaner signal at the time those were cut).

## Michael's DJI Matrice 100 rig

Four recordings of the same airframe (DatCon `ACType|M100`), in two raw trees.
Telemetry (`Motor:Speed:*`, RPM) logs at ~29 Hz vs DREGON's ~929 Hz, so
`rps.npy` is shorter in the mixes; handled downstream via the per-sample
`motor_sample_rate`.

| | FLY124 / FLY125 | FLY103 / FLY108 |
|---|---|---|
| raw dataset | `recording_with_motor_speed` | `new-drone-noises` (`103_2.wav`, `108_2.wav` + DatCon CSVs) |
| channels | 8 (mic ring) | **1 (mono)**, no `mic_pos` |
| native rate | 44.1 kHz | 48 kHz |
| duration | 122 / 178 s | 106.5 / 99.4 s |
| log vs audio | log starts ~0.15 s early | log is ~2× longer than the audio |
| alignment path | legacy (crop the audio head) | **anchored** (`load_raw_aligned(..., anchor=True)`) |
| constants | `MICHAELS_FILES` (measured, WP13/WP14) | `MICHAELS_TEST_FILES` (measured 2026-08-24) |
| role | training (FLY125) / validation (FLY124) | **held-out TEST set** — no training derivation may root on it |
| published as | `michaels-frames` (adopt-only) | `michaels-test-frames` (derivable, pinned) |

Current shipped constants (`src/data_processing/sources/michaels.py`):

| recording | `time_offset` | `time_dilation` | `MICHAELS_RPS_SCALE` |
|---|---|---|---|
| FLY124 | −20.753813 | 1.001654644 | ×1.00698 (g = 0.698 % ± 0.069) |
| FLY125 | −26.337849 | 1.005178509 | ×1.00706 (g = 0.706 % ± 0.034) |
| FLY103 | −0.898263 | 1.011846126 | ×1.005251 (g = 0.525 % ± 0.028, 5 windows) |
| FLY108 | −0.394115 | 1.011957365 | ×1.005696 (g = 0.570 % ± 0.012, 4 windows) |

**Where the calibration is applied.** Inside `sources/michaels.py:load_raw_aligned`
(keyed by CSV stem via `rps_scale_for()`), so every consumer — the registry
builders that publish `michaels-frames`/`michaels-test-frames`,
`load_michaels_timeframes`, the `derivations.py` specs rooted on those frames,
the online-mix `kind: frames` source — is calibrated with no call-site change.
Recordings with no measured constant fall back to 1.0. **Not calibrated**: the
raw `Motor:Speed:*` columns published as the `motor_speed` block (the raw
counterpart of the canonical `rps`, like DREGON's `motors_command_raw`), and
anything that parses the CSV itself. Frame `meta` records `time_offset` /
`time_dilation` / `rps_scale` + a `provenance.calibration` note; frames
published before 2026-07-31 carry `rps_scale` 1.0.

### FLY124 / FLY125 calibration history (WP13 / WP14, 2026-07-31)

All three constants are measured against the label-free VK reconstruction
residual (referee + stages in `scripts/michaels_calib/calib.py`; write-up
`docs/experiments/rps-refine-precision.md` §§ WP13 (timing + model form) and
WP14 (rev/s magnitudes, refit)). They replaced the hand-tuned pair from
`notebooks/michael_data_analysis.ipynb` (offsets −20.84 / −26.51, dilations
1.001 / 1.0048, scales ×1.00839 / ×1.00690).

- **Timing was a dilation error on both**: the audio-optimal telemetry lag
  drifts linearly with time (+0.654 ms/s FLY124, +0.377 ms/s FLY125; R² 0.94 /
  0.92; residual RMS 2.9 / 4.5 ms vs 12.0 / 16.2 ms for a constant lag).
  Folded in via `time_dilation/(1−b)`, `time_offset − a/(1−b)` — the
  `fit_lag` algebra, which now lives in `scripts/michaels_calib/fit_new.py`
  (the original `fit.py` and the `run_sweep.py` driver were removed in the
  2026-08 script cleanup; git history keeps them).
- **Values were ~0.6 rev/s low at cruise.** Additive vs multiplicative is a
  statistical tie (per-rotor discriminator has no power: free-line R² ≈ 0.01,
  per-rotor spread 6–17× what either model predicts). Shipped as
  **multiplicative** because the frames cover warm-up/idle, where a scale
  vanishes as rps → 0 but an additive +0.6 rev/s would invent motion. WP14
  adds a second reason: the global gain is **degenerate with a sample-clock
  error** (`dilation − 1 ≈ εa − εt`, `g ≈ −εa − εr`), which is exactly
  multiplicative — a *label-for-this-audio* correction, not proof the ESC is
  miscalibrated. The same degeneracy backs the **global-only** model form: a
  clock error is common to all four rotors. Per-rotor constants are
  unidentifiable (between-rotor spread 0.099/0.051 < within-rotor scatter
  0.155/0.109) and a per-rotor lag is refuted three ways (WP14 §§ A–D).
- **Scale supersession (WP14).** The first shipped scales (×1.00839 / ×1.00690)
  came from a 2–4 window per-rotor mean that included the near-equal-speed
  twin rotors (LFront/RBack), which the audio cannot resolve; they disagreed by
  0.15 pp and FLY124's was 0.14 pp too high (+0.12 rev/s over-correction at
  cruise). Both are now one global fit per recording over all 13 cruise
  windows, non-twin rotors only, agreeing to 0.008 pp. Timing unchanged.
- **Validation** (`post` stage on the shipped constants, 13 cruise windows, no
  edge hits; job `python-b7e225`, old scales `python-0445f6`):

  | recording | n | resid lag mean / RMS / max | resid value offset mean | (old scales) |
  |---|---|---|---|---|
  | FLY124 | 4 | −2.92 / 3.40 / 4.29 ms | **−0.054 rev/s** | −0.178 rev/s |
  | FLY125 | 9 | +1.21 / 3.22 / 8.04 ms | **−0.009 rev/s** | +0.004 rev/s |

  Timing stays closed (residual lag RMS at the fit's own level, no drift); the
  value error is ≈ 0 on both (FLY124's residual fell 3.3×, ~1/10 of the
  0.56 rev/s the calibration removes). The −0.054 leftover exceeds the +0.004
  the refit projected because the estimators differ (refit: per-rotor offsets
  on non-twin rotors; this scan: one global offset over all four). Both are
  far inside the 0.175–0.213 rev/s per-window scatter.
- **Anything built from Michael's telemetry before 2026-07-31 is stale** —
  `beatvk-valid-raw`, `results/beatvk_vk_arms`, `DREGON-LM-V4-michaels*`, and
  any published FLY124 label-accuracy number.

### FLY103 / FLY108 — the held-out TEST recordings (2026-08-24)

Registry entry `michaels-test` (builder `michaels.build_test`), derivation
`michaels-test-frames` (derivable, not adopt-only — the builder *is* the
recipe; `recipe_version` 1 = the 2026-08 calibration). Pinned in `dload.lock`.

**Anchored alignment.** The legacy path uses `time_offset` only to decide how
much audio head to crop and then treats the log stamps as audio time — right
only because FLY124/FLY125's logs start ~0.15 s before their audio. FLY103/108's
logs start 50–78 s early, so they use the explicit model
`t_log = time_offset + t_audio / time_dilation`: the audio is never cropped, the
stamps become `(t_log − time_offset) · time_dilation`, and the row cut is
`wav_duration / time_dilation` of log time. Folding a measured
`lag(t) = a + b·t` gives `time_dilation / (1 − b)` and
`time_offset − a / time_dilation` (**not** the legacy `− a / (1 − b)`).

**Calibration, two stages (both done).**

1. `scripts/michaels_calib/coarse_align.py`: one STFT per recording, scored at
   the telemetry-predicted comb bins (`k · 2 · rps`, k = 1..10, cruise frames
   only), scanned over every feasible log offset; the global peak anchors five
   per-segment peaks and the line through them is the (offset, dilation) pair.
   Seeds: FLY103 −0.8915 s / 1.0119078, FLY108 −0.3956 s / 1.0119673;
   per-segment residual RMS 1.9 / 3.4 ms; whole-recording score +5.6 % / +6.1 %
   over the best single offset at dilation 1. Both agree on a **1.19 % clock
   dilation to five decimals** — 7× FLY125's and 20× FLY124's, a property of
   that recording session.
2. `scripts/michaels_calib/fit_new.py` (job `michaels-fit-new-6743bd`): the
   WP13/WP14 VK procedure on the coarse seed — `lag` per cruise window → the
   refined (offset, dilation); `val` at each window's best lag → the global
   multiplicative rev/s scale. Residual lag RMS 2.53 ms (5 windows) / 1.11 ms
   (4 windows); fine constants agree with the coarse pass to 7 ms offset and
   1e-4 dilation. The per-rotor `prot` stage is deliberately off: with one
   channel and rotor pairs 2.0 / 2.5 rev/s apart, an offset of the expected
   size walks one rotor into its neighbour and the VK solve stalls; WP14
   already settled what that stage decides.

**Content.** Over the audio span the telemetry calls FLY103 1.4 % ground /
12.7 % warm-up / 86.0 % cruise (89.8 s of cruise, mean 69.6 rev/s) and FLY108
6.8 % / 9.6 % / 83.6 % (81.6 s, mean 80.6 rev/s). FLY108's first 16 s window is
the takeoff ramp (per-rotor std 26.6 rev/s) and `fit_new.py` drops it.

## Derived dataset variants

Every variant is a frozen spec in `derivations.SPECS`; there are no creation
CLIs. `python scripts/derive.py list -v` prints them with notes and
fingerprints; `derive <NAME>` materializes one; `adopt <NAME> --commit` points a
derivation ref at an existing historical pin.

### DN-LM (DroneNoise-LibriMix) — Paper 1

Specs `DN-LM-train` / `DN-LM-valid` (generator `dn_lm`, derivable).

- Sources: LibriSpeech `train-clean-100` + `drone_audio/Binary_Drone_Audio/yes_drone`
  (the label-1 recordings; the raw tree's `*/unknown/` mixes ESC-50/WN/silence
  negatives, which the paper's DN-LM excluded).
- 1 s samples, 16 kHz mono, SNR −30…0 dB; inverse-distance attenuation.
- Split: 6480 train / 720 valid. No `rps` field.

### DREGON-LM V4 — current (multichannel)

Specs `DREGON-LM-V4-{train,valid}`, `DREGON-LM-V4-michaels-{train,valid}`,
`DREGON-LM-V4-michaels-valid-full` (generator `dregon_lm`, **adopt-only** — the
published bytes are the historical uploads; the generators reproduce the recipe
but the mixing RNG is not byte-stable across machines).

- `mode: "synthesized"` (train): noise chunks from the published frames
  datasets + LibriSpeech, mixed per channel. Per sample: `mixture.wav (T, 8)`,
  `vocals.wav`, `noise.wav`, `rps.npy (4, M)`, `meta.json`.
- `mode: "real_valid"` (valid): raw 8-channel clips from `in_flight_source`
  recordings, **no mixing**. `mixture.wav` = the raw recording, `rps.npy` =
  telemetry; no `vocals`/`noise` (no clean reference exists), so `eval.py`'s
  speech-enhancement metrics do not apply — RPS evaluation only.
  `params.max_non_overlapping: true` emits every non-overlapping clip of every
  in-flight window instead of `num_samples` random draws. With 8 s clips only
  ~15 non-overlapping clips exist across the 2 default recordings; larger
  `num_samples` overlaps (fine for RPS eval).
- Shared V4 knobs (`_V4_PARAMS`): 1 s train / 8 s valid, 16 kHz, SNR U[−30, 0],
  `speech_per_channel: independent`, `source_white_noise_prob: 0.3` (replaces
  speech with white noise as the *target source*; distinct from
  `white_noise_prob`, which adds WN on top of speech — use 0.2–0.4 for diversity
  training), `min_motor_rps: 30.0`.
- Noise pool of the michaels variants: DREGON `in_flight_noise` excluding
  `free-flight_nosource_room1` + Michael's `FLY125` for training; room1 +
  `FLY124` for validation (leakage split — keep it in online-mix policies too).

#### The noise-only twin (`DREGON-LM-V4-michaels-valid-full-nospeech`)

`DREGON-LM-V4-michaels-valid-full` is 37 real 8 s clips, 14 of which are not
rotor noise only: `free-flight_speech-low_room1` (7) and
`free-flight_whitenoise-low_room1` (7) play a loudspeaker into the room during
the flight. `real_valid` mode mixes nothing — `mixture` *is* the raw 8-mic
recording (`snr_range` / `source_white_noise_prob` are inert) — so that source
is acoustic and cannot be removed. The twin therefore **drops** those 14 clips:
it is the 23 `source_type: nosource` clips (`free-flight_nosource_room1` 8 +
FLY124 15), copied **byte for byte** by the `dregon_lm_subset` generator
(measured: max abs difference 0.0 on audio and `rps`, per-clip RMS ratio
1.000000; same clip order as the parent's samples 00014–00036; each row keeps
its parent id under `source_id`). Read it against the full set to price what
the speech costs — parts `real_nospeech` vs `real` in
`experiments.rps_bench.PARTS`, configs `conf/data/m3cur_s2_nospeech.yaml` vs
`conf/data/m3cur_s2.yaml`.

The subset is cut from the published bytes, **not** re-derived, because the
parent is a stale artifact: re-running its generator on today's pinned parents
moves the labels (DREGON `rps` by up to 16 rev/s, since `adapt_recording_frame`
prefers `motors_measured` while the parent was cut from `motors_command`) and
moves the FLY124 audio (the 2026-07-31 telemetry calibration changed the
alignment). A twin whose labels had moved would not be comparable.

### Superseded variants

The V1/V2/V3/test recipes and the one-off `rps_*` probe sets are consumed as
plain pinned uploads and are **not** re-derived — `derivations.HISTORICAL_PINS`
(V1 mono, Paper 2 baseline; V2 motor combos; V3 per-channel mono; `test`
smoke recipe). Recipes are in git history (the deleted creation CLIs).

## Pinned catalog (`dload.lock` — 49 datasets)

Tests assert every `dload.lock` name is a `SPECS` entry, a `sources.REGISTRY`
entry, or listed in `HISTORICAL_PINS`.

- **Raw sources** (7, CLI convention, from `data/`): `DREGON`, `librispeech`,
  `drone_audio`, `music`, `new-drone-noises`, `recording_with_motor_speed`,
  `zenodo_drone_noises`.
- **Derived DREGON-LM** (17, `sample-dir-v1`, per split):
  `DREGON-LM-{train,valid}`, `DREGON-LM-V2-{train,valid}`,
  `DREGON-LM-V3-{train,valid}`, `DREGON-LM-V4-{train,valid}`,
  `DREGON-LM-V4-michaels-{train,valid}`, `DREGON-LM-V4-michaels-valid-full`,
  `DREGON-LM-V4-michaels-valid-full-nospeech` (its noise-only twin),
  `DREGON-LM-test-{train,valid}`,
  `DREGON-LM-rps_{eval_long,eval_specific,train_specific}_samples`.
- **DN-LM** (2, `sample-dir-v1`): `DN-LM-{train,valid}`.
- **Rich frames** (3, `tdframe-v1`): `DREGON-frames`, `michaels-frames`
  (adopt-only, published by the deleted `scripts/publish_frame_datasets.py`;
  the `sources` builders reproduce them), `michaels-test-frames` (FLY103/FLY108,
  the held-out TEST set; derivable).
- **External harmonic-noise datasets** (10, `tdframe-v1`; registry
  `src/data_processing/sources/`, driver `scripts/derive.py`, plan
  `docs/external-datasets-plan.md`):
  `MIMII` (54057; industrial fan/pump/slider/valve, 8-ch 16 kHz 10 s, 3 SNR
  tiers), `MIMII-DG` (17999; fan/gearbox/bearing/slider/valve mono, domain-shift
  sections), `drone-detection-samples` (180320; mono 16 kHz binary
  drone/no-drone), `DroneAudioSet` (2313; 2 quads × 2 throttles × 3 rooms, 8-ch,
  drone-only/source-only/mixed subsets), `AeroSonicDB` (1895; aircraft flyover +
  rich aircraft/engine/prop meta), `SPCUP19-egonoise` (278; 10 heterogeneous
  drone-team ego-noise rigs, 1–16 ch, mic geometry in meta where exposed),
  `HornBase` (1080; horn/not-horn — tonal, not rotating-source), `HUSTmotor`
  (24; 6 health states × 4 speeds, acoustic + X/Y/Z vibration),
  `KAIST-rotating-acoustic` (5; sound-pressure at 3010 RPM), `AVQ` (12;
  audio-visual quadrotor — onboard 8-ch array, 44.1 kHz, rotor ego-noise + a
  moving speech source; labeled seqs carry `angle_vad` DOA/VAD + `mic_pos`;
  builder `build_avq`, http+extract). Every recording Frame carries
  `system`/`observation`/`operating`/`label` meta (make/model, how observed —
  onboard vs flyover — SNR, condition). Harmonicity is measured separately
  (`harmonicity.py`, analysis stage).
- **Byte-exact raw companions** (1, `raw-files`): `AVQ-raw` (26; the AVQ
  videos + `cameraParams.mat` + `.docx` docs + raw mic_pos/angle_vad/
  av_calibration mats — everything except the per-channel `MONO-*.wav`, which
  is the audio in `AVQ`). Generator `raw_subset`.
- **Purpose-built derived subsets** (2, `tdframe-v1`):
  - `AVQ-egonoise` (5; the pure rotor ego-noise sequences of `AVQ` —
    `S1_seq1`/`S1_seq2`/`S1_seq3`/`S2_seq1`/`S2_seq2`, the only AVQ recordings
    *without* an `angle_vad` entry and therefore without the speech source —
    **channel 0 only, 16 kHz mono**, 705 s in one 43 MiB shard, AVQ's
    per-recording `meta` + provenance kept). Generator `frame_subset`. It exists
    so the F2 pools (`conf/online_mix/se_avq_survey.yaml`,
    `se_survey_alldrone.yaml`, `se_survey_allharmonic.yaml`,
    `derivations.SE_CATEGORY_NOISE`'s `avq_ego` category) can be a plain
    `audio_pool` with no `include_keys`/`channel`: manifests carry no key list,
    so key filtering over `AVQ` had to download all 11 shards (~4 GiB) to find
    those 5 recordings, and one AVQ shard is a 352 MB **multipart** object that
    s3transfer's ETag validation rejects on some boto3 builds (Kaggle:
    `S3DownloadFailedError ... did not match expected ETag`) — a false alarm
    (the sha256 matches), but fatal there. `AVQ`/`AVQ-raw` are unchanged.
  - `AVQ-egonoise-vkrps` (7; `AVQ-egonoise` joined with blind-VK RPS
    **pseudo-labels** from the annotator `scripts/vk_pseudolabel.py` @ fa5053fc
    — the script was removed in the 2026-08 cleanup; the commit hash in the
    spec's `gen.annotator` is the provenance). One Frame per contiguous accepted
    segment, recordings split at NaN/refused spans (a frame is accepted iff all
    4 rotor labels are finite), segments ≥ 10 s kept: ~617 s total, one 38 MiB
    shard. Each Frame: mono 16 kHz `audio` + `rps` `(rotor, time)` StampIndex
    Series on the 0.032 s grid (michaels-frames events convention) + provenance
    meta (annotator commit, `refuse_conf`, per-segment mean VK confidence).
    Spec adopt-only (a GPU annotator sits in the loop); consumed as
    `kind: frames` in `conf/online_mix/beatvk_avq_dload.yaml` (beat-VK R2 arm).
    Labels are cruise-only (66–117 rev/s), not telemetry — pseudo-ground-truth.
- **VK decompositions** (2 materialized, `tdframe-v1`; consumer
  `DecompFrameDataset`, `docs/experiments/amplitude-target-training.md`):
  `decomp-frames-v1` — the coupled Vold-Kalman decomposition of the three
  decomposable real recordings (DREGON `free-flight_nosource_room1` on the
  refined labels, Michael's FLY124/FLY125) as per-`(mic, rotor, k)` amplitude
  envelopes at 100 Hz + validity mask + broadband residual + the exact
  audio-rate carrier the solve used, on one re-anchored time origin.
  Adopt-only (a multi-hour cluster solve sits in the loop;
  `scripts/vk_decompose.py`, artifacts under `r2://ml-data/artifacts/vk-decompose/`).
  v1 used a **flat 1 Hz** per-track bandwidth, so its mid/high-k amplitudes are
  underestimates and its residual carries the leaked comb energy.
  `decomp-frames-v2` — the same join over the linewidth-matched re-solve
  (`artifacts/vk-decompose-v2/`), pinned. `decomp-frames-v3-{dregon,michaels}`
  (per-rig, the combined-rig arms' source) exist only as placeholder names in
  `conf/data/decomp_frames_v3_combined.yaml` — no spec, no pin yet.
- **Fixed SE validation sets** (4, `tdframe-v1`, `{mixture,target,meta}` per
  clip; generator `se_valid`): `SE-valid-drone`, `SE-valid-harmonic` (F1;
  `recipe_version` 2 = rebuilt with the silent-draw filter,
  `mixing.MIN_DRAW_POWER`), `SE-valid-avq-survey` (F2, 250 clips),
  `SE-valid-avq-split` (session-split memorisation probe, S1 seen / S2 unseen).
- **beat-VK frozen validation** (1, `tdframe-v1`): `beatvk-valid-raw` — 4
  recordings (DREGON room1 nosource/speech-low/whitenoise-low on
  `motors_measured` + FLY124), native-rate audio + raw telemetry + the 16 s
  window manifest. Adopt-only.
- **Declared, not materialized**: `librispeech-pcm16` (memoized mono int16
  decode of the librispeech pin at 16 kHz — the derived-dataset replacement for
  the deleted `packed_int16` speech cache; point a policy's
  `sources.speech[].dataset` at it to skip per-encounter FLAC decode).

Consumption paths: `DloadFrameDataset` / `dload:NAME[@VER][/subpath]` URIs /
`frames:NAME` specs — `streams.py`'s module docstring and
`docs/data-and-artifacts.md` (end-to-end flow, cache env vars, measured
streaming numbers).
