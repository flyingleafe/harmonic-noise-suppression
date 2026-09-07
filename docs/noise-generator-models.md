# Noise generators (`src/models/generative/`) — reference

Long-form reference for the RPS → drone-noise generative models and the
noise-generation task contract. Moved out of `src/models/AGENTS.md` and
`src/tasks/noise-generation/AGENTS.md` on 2026-09-07; those files keep the map
and the contracts, this file keeps the reasoning and the detail. Experiment
history: `experiments/noise-generation-augmentation.md` (E2–E4),
`experiments/amplitude-target-training.md` (amplitude objective, MicEQ
C-series), `experiments/wind-channel-likelihood.md` (wind channel,
likelihood objective), `experiments/residual-attribution.md`,
`experiments/generator-label-sensitivity.md`, `experiments/noise-gen-linewidth.md`.

## Package origin

Full sync of `drone_audition.models` (no `env.settings`; sample rate is an
explicit constructor arg on every module, default 16 kHz). The
`harmonic_gen_new` predictors run natively at 16 kHz; older 44.1 kHz
checkpoints from `drone_audition` are not weight-compatible.
`DroneNoisePlusFilterGen` (`filtered_noise.py`) is port-only: it was used by
the deleted `train_noise_gen.py` and has no unified-framework `conf/model`
entry (REPLICATION.md § E1 records it as a dead end). The VP-transform family
(`VP_transform`, `lstsq_VP_transform`, `inverse_VP_transform`,
`harmonic_VP_transform`, `HarmonicTransformModule`) now lives in
`utils/dsp.py` and is re-exported from `models.generative`; its users are
`tracking.rps_refinement` and `experiments/kalman_harmonic/phase0.py`.
`good_lstsq` picks `gelsd` on CPU / `gels` on CUDA; `iterative_lstsq_minimize`
needs the optional `torchmin` package (imported lazily).

## Class map

| Class | File | Purpose |
|-------|------|---------|
| `DroneNoiseGen` | `harmonic_noise_gen.py` | Per-rotor harmonic oscillator bank |
| `PropellerNoiseGen` / `PolynomialRegression` / `PolyWithExpLog` | `harmonic_noise_gen.py` | Single-rotor bank + scalar gain regressors |
| `DroneNoisePlusFilterGen` / `FilteredNoiseSynth` / `RPSFilterNet` | `filtered_noise.py` | DroneNoiseGen + RPS-conditioned filtered-noise residual (port-only) |
| `HarmonicNoiseGenNew` | `harmonic_gen_new.py` | End-to-end RPS→audio: NN predicts harmonic amps + noise mags → oscillator bank + filtered noise |
| `JointAmplitudePredictor` / `ConstantAmplitudePredictor` / `DirectionalOutputHead` / `SpeedsPostprocessingWrapper` / `LearnableTimeShift` | `harmonic_gen_new.py` | Amplitude predictors + helpers for `HarmonicNoiseGenNew` |
| `SimpleHarmonicNoiseGen` / `PropellerAmplitudePredictor` | `harmonic_gen_new.py` | DEPRECATED random-phase synthesiser + per-prop predictor |
| `PositionalHarmonicNoiseGen` + `propagate` / `fractional_delay` | `positional_harmonic_gen.py` | Position-aware generator (below) |
| `MicEQ` | `propagation.py` | Frequency-dependent per-mic gain curve (below) |
| `WindWakeChannel` + `wake_flow_speed` / `QuadDynamics` / `WindTransduction`; `PositionalHarmonicPlusWindGen` | `wind_wake_gen.py` | Additive incoherent wind-noise channel (below); the registered combination is `positional_harmonic_wind_gen` |
| `DroneCodebook`, `geometry_to_rel_pos` | `codebook.py` | Name-keyed per-drone code table; mic/rotor geometry → `rel_pos` (imported by `tasks.codecs`; `tasks.noise_generation` holds only the `NoiseGenerator` protocol and `load_input_set`) |
| `MultiScaleSTFT`, `smoothness_penalty` | `losses.py` | DDSP-style multi-scale spectral loss; squared-2nd-difference control-curve regulariser |
| `CausalConv1d` / `CausalConv1dBlock` / `ResNet` / `RnnSandwich` / … | `nn.py` | Shared building blocks (used by the predictors) |

## `PositionalHarmonicNoiseGen`

Single-rotor `HarmonicNoiseGenNew` (rotor folded into batch) **emits** per-rotor
sources, then **propagates** to observation point(s) with 1/r attenuation +
fractional delay (`r/c`, c=343). Native multi-observer — rotors summed in the
rfft domain, so M mics cost R forward + M inverse transforms. Differentiable
w.r.t. position. Isotropic point source (distance-only).

- **Per-drone conditioning is external**: `cond_dim=d` FiLM-conditions the
  emitter on a code `z (B,d)` passed to `forward`; the `name→z` table is a
  separate `DroneCodebook`, so model params never resize with drone count and
  an unseen drone is few-shot-adaptable by freezing the model and fitting just
  its code (see "Per-drone conditioning" below).
- **`amp_stats(rps, rel_pos, z)`** returns the per-`(mic, rotor, harmonic)`
  amplitude ENVELOPES (1/r gains only — no delay, no rotor sum, no synthesis,
  jitter-free by construction) plus the power-summed per-mic broadband
  envelope, and `freq` (needed by `MicEQ`): the training path of the
  Vold-Kalman amplitude objective, ~100x cheaper than a render.
- **`build_noise_gen_model(amp_calibration=True, n_mics=…, noise_floor_bands=…)`**
  adds the per-drone absolute-level gains that objective needs (global, per-mic,
  a separate power-domain constant and a static per-mic per-band floor for the
  broadband branch — per-rotor attribution of the residual is refuted, see
  `experiments/residual-attribution.md`); they apply to EVERY prediction path,
  so a calibrated model also renders at the recording's level.
- **`forward(..., return_dict=True)`** also exposes the emitter's per-rotor
  control curves (`harm_amps` `[B,R,O,H,t]`, `noise_amps` `[B,R,F,t]`) — the
  inputs to the Stage-2 smoothness regularisers (`losses.SmoothnessPenalty`,
  squared 2nd difference; harmonic amps over time, noise shape over
  time+freq). Historically wired via the deleted `train_noise_generation.py`
  (`--harm_smooth_weight`/`--noise_smooth_weight`, default 0); now
  `conf/experiment/e3_noise_gen_swapped_smoothness.yaml` on the entries
  `tasks.codecs.NoiseGenerationCodec(return_dict=True)` exposes.
- **`spectral_stats()`** supplies `coherent`/`noise_psd` for
  `losses.SpectralLikelihoodLoss` (`task_params.distributional=true` widens the
  task's `output_spec`) — see `src/losses/AGENTS.md`.
- **Initial harmonic phases** (`HarmonicNoiseGenNew.forward`): random
  per-harmonic in **train** mode (phase augmentation), **zero** in **eval**
  mode (deterministic) — always `model.eval()` for inference/rendering.
  Overridable via `forward(..., initial_phases=[B,R,H])` for reproducible
  synthesis (distinct from the heavier `use_random_phases` spectral-phase
  randomiser, off by default).
- `models.registry.build_noise_gen_model` / `build_noise_gen_loss` are the
  model/loss factories (`conf/model/positional_harmonic_gen*.yaml`'s
  `_target_`); figure scripts call them directly to reload a checkpoint outside
  the training loop.

## `MicEQ` (`propagation.py`)

The frequency-dependent half of the amplitude-only propagation head:
`A_obs[r,k,c](t) = A_src[r,k](t) * (1/dist_{r,c}) * EQ_c(f_k(t))`. `EQ_c` is a
learnable smooth magnitude response — `n_knots` (<=16) control points
log-spaced in frequency, holding the log gain, linear between knots, held (not
extrapolated) outside the span — one curve per **(rig, microphone)**, **shared
across rotors** (room + capsule belong to the receiver). Zero init = unity, so
an untrained head is the plain 1/r law. Built by
`build_noise_gen_model(mic_eq_knots=…, eq_f_min=…, eq_f_max=…)`, which then
does NOT build the frequency-flat `log_mic_gain` (a flat EQ *is* that scalar).
Applied on every prediction path: on `amp_stats`'s cells at their own
`f = k * rps_r(t)`, and on rendered audio / `spectral_stats`' `coherent` as a
zero-phase rfft-domain multiply, so a checkpoint renders with the response it
was fitted with. The knot curve rides out as the `mic_eq` prediction entry, so
its curvature penalty is an ordinary `losses.SmoothnessPenalty` composite term.
Rationale + arms: `experiments/amplitude-target-training.md` § C-series.

## `WindWakeChannel` (`wind_wake_gen.py`)

**Additive, incoherent wind-noise channel** — the flow-noise (pseudo-sound)
that the coherent `PositionalHarmonicNoiseGen` propagation path structurally
cannot make. Same `rps+geometry → [B,M,T]` contract; its output is **summed** at
each mic. Physics places the air, only the mic response is learned: **A**
`QuadDynamics` (grey-box quad, RPS→`V_rel`, hover-anchored; skipped at
`V_rel=0`), **B** `wake_flow_speed` (closed-form bent-wake-column gate →
per-mic flow speed `U_m`; aero constants `k,α,β` only), **C**
`WindTransduction` (learned `U→½ρU² level ·` low-pass `H(f/f_c)`, OU gust
envelope, independent filtered noise per mic ⇒ incoherent by construction).
`rps=0 → silence`; differentiable through positions/params. Tests:
`tests/test_wind_wake_gen.py`. The pre-training de-risk (Spearman 0.92 vs a
0.74 `1/r` control on DREGON's per-mic low-band floor; Michael's array out of
the wake at max exposure 0.006 m/s) and the eventual refutation of the channel
are in `experiments/wind-channel-likelihood.md`; the driver script
`scripts/wind_wake_validation.py` was deleted, the document is the record.

## The noise-generation task

Generate the drone noise observed at each microphone from rotor speeds and the
array geometry — the inverse of RPS prediction (audio → rotor speeds).

### Geometry as input

Microphone/rotor positions are non-temporal array metadata, carried as Frame
entries `mic_pos (M,3)`/`rotor_pos (R,3)` (dims `("mic", None)`/`("rotor", None)`).
`geometry_to_rel_pos` gives `rel_pos[m, r] = mic[m] - rotor[r]` and has two
dispatch paths: unbatched numpy `(M,3),(R,3) -> (M,R,3)` (figure scripts,
`data_processing.generated_noise`) and batched torch
`(B,M,3),(B,R,3) -> (B,M,R,3)` (differentiable, on-device), used by
`tasks.codecs.NoiseGenerationCodec` to build `rel_pos` from a training batch's
`mic_pos`/`rotor_pos` before calling the model (the fix for the codec/model
signature mismatch REPLICATION.md § E2/E3 documents). Geometry is fixed per
array: `data_processing.sources.dregon.get_geometry(dregon_dir)` (DREGON 8-mic)
and `data_processing.sources.michaels.get_geometry()` (Michael's circular 8-mic
ring on a DJI Matrice 100 — derived from the rig photos in
`data/recording_with_motor_speed/`; rotor rows ordered RFront, LFront, LBack,
RBack to match the telemetry). The deleted `train_noise_generation.py` selected
geometry with a `--geometry {dregon,michaels}` flag; the unified framework
resolves it per chunk via the data source (`NoiseGenFrameDataset`).

### Mixed-geometry datasets and channel policy

`data_processing.frame_datasets.NoiseGenFrameDataset` wraps
`data_processing.noise_rps_dataset.NoiseRPSDataset` (whose chunks carry a
per-draw `origin`, `"dregon"`/`"michaels"`) and attaches that origin's geometry
+ a `meta.drone` name per sample, so DREGON and Michael's chunks stream together
in one dataset, each with its own geometry. `channel_policy="first"` (default)
keeps the single-microphone behaviour the Hydra migration introduced; `"all"`
renders every microphone jointly, restoring the historical online trainer's
native multi-observer rendering. `"random"` is rejected — the inner dataset
does not report which index it drew, so the geometry could not be matched to
the audio. **The choice is load-bearing for any channel model**: a component
distinguished from the coherent field only by its *spatial* law (the wind-wake
channel above all) is **unidentifiable at M=1**, where it degenerates into
another broadband shape competing with a far more flexible learned filter. Use
`"all"` whenever such a component is in play
(`conf/data/noise_rps_dregon_michaels_swapped_stream_multimic.yaml`).

### Per-drone conditioning — external codebook

DREGON and Michael's are different drones (different harmonics/broadband/
dynamics), so the source is conditioned on a learned per-drone code, external
to the model by the same logic that keeps geometry external:

- The generator takes the code `z (B, d)` as an **input**, not a `drone_id`:
  `PositionalHarmonicNoiseGen(cond_dim=d)` builds the emitter with
  `JointAmplitudePredictor(film=True)` so `z → (γ,β)` modulate the CNN features
  (per-drone spectral envelope *and* RPS→sound dynamics). `cond_dim == 0`
  disables conditioning. FiLM starts near-identity (γ≈1, β≈0) with a small
  non-zero weight so the code gets gradient from step 1.
- The `name → z` table is `DroneCodebook(d, names)`, a **name-keyed**
  `nn.ParameterDict`. The model owns `d` (architectural — it sizes the FiLM
  generator) but **not** `K` (the number of drones, a data property). Adding a
  drone never resizes model weights; codes load by name with `strict=False`, so
  no index drift between datasets.
- **Few-shot adaptation to an unseen drone** = freeze the generator, warm-start
  from a trained checkpoint, and optimise just the new drone's `d`-vector. This
  is the payoff of the external/`z`-input design and is not clean with an
  in-model table.
- In the unified framework `build_noise_gen_model(..., cond_dim=d,
  drone_names=[...])` returns a composite `_CodebookConditionedNoiseGen(generator,
  codebook)` — the codebook is a genuine submodule, so its params train and
  checkpoint through the normal single-model path.
  `NoiseGenerationCodec(conditioned=True)` resolves each sample's `drone_names`
  from `meta.drone` and calls `model(rps, rel_pos, drone_names)`; the model
  resolves `z` from its own codebook. Multi-drone training in one run works
  out of the box. History of this migration: `experiments/noise-generation-augmentation.md`
  § "Framework migration notes".

### Loss guidance

Multi-scale STFT (`losses.MultiScaleSTFTLoss`, `pred_key=target_key="audio"`),
the mic axis folded into the batch by `losses.spectral._flatten_to_2d`. For a
model with a **stochastic** branch (broadband residual, wind) prefer
`losses.SpectralLikelihoodLoss` with `task_params.distributional=true`: the
magnitude loss compares one gust realization to another and fits any stochastic
component 1.6 dB low. Magnitude loss is blind to a *common* delay but sees
inter-rotor delay differences (the geometric signal). E3's Stage-2 smoothness
regularisers: `losses.SmoothnessPenalty(entry="harm_amps"|"noise_amps",
series_dims=..., series_time=None)` on the extra pred entries
`NoiseGenerationCodec(return_dict=True)` exposes.
