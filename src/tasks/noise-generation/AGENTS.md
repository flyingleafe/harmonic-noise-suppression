# Task: Noise Generation

## Purpose

Generate the drone noise observed at each microphone from rotor speeds (RPS)
and the array geometry — the **inverse of RPS prediction** (audio → rotor
speeds); here *rotor speeds + geometry → multichannel noise*.

## Map

| Where | What |
|-------|------|
| `src/tasks/noise_generation.py` | `NoiseGenerator` protocol, `load_input_set` TimeFrame loader |
| `src/tasks/codecs.py::NoiseGenerationCodec` | batch → `(rps, rel_pos[, drone_names])`; `conditioned=True`, `return_dict=True` |
| `src/models/generative/` | models; `codebook.py` holds `DroneCodebook` + `geometry_to_rel_pos` |
| `src/models/registry.py` | `NOISE_GEN_MODEL_REGISTRY`, `build_noise_gen_model(...)`, `build_noise_gen_loss(...)` |
| `src/data_processing/frame_datasets.py::NoiseGenFrameDataset` | wraps `noise_rps_dataset.NoiseRPSDataset`; attaches geometry + `meta.drone` |
| `conf/model/positional_harmonic_{gen,wind_gen}*.yaml` | model configs (`_conditioned`, `_amp_*`, `_perdrone_*`, `_cond_jitter*`, …) |
| `conf/data/noise_rps_*.yaml`, `conf/loss/{multiscale_stft*,spectral_likelihood,amplitude_target*}.yaml`, `conf/metrics/noise_gen_spectral.yaml` | data / loss / metrics |
| `conf/experiment/e2_noise_gen_dregon_michaels.yaml`, `e3_noise_gen_swapped_smoothness.yaml`, `e4_*.yaml` | canonical experiments (REPLICATION.md § E1–E4) |

## Model interface

```python
class NoiseGenerator(nn.Module):
    def forward(self, rps: Tensor, rel_pos: Tensor, z: Tensor | None = None) -> Tensor:
        """
        rps:     (B, R, T)      per-rotor speed (Hz) at audio rate
        rel_pos: (B, M, R, 3)   vector rotor_r -> mic_m (metres)
        z:       (B, d)         external per-drone conditioning code (or None)
        returns: (B, M, T)      noise at each of the M microphones
        """
```

## Contracts and invariants

- **Input rate** 16 kHz; `rps` is upsampled to the audio grid, not the STFT grid.
- **Geometry is an input**, carried as Frame entries `mic_pos (M,3)` /
  `rotor_pos (R,3)` (dims `("mic", None)` / `("rotor", None)`).
  `geometry_to_rel_pos` gives `rel_pos[m, r] = mic[m] - rotor[r]` with two
  dispatch paths: unbatched numpy `(M,3),(R,3) -> (M,R,3)` and batched torch
  `(B,M,3),(B,R,3) -> (B,M,R,3)` (differentiable; the codec uses this one).
  Fixed per array: `data_processing.sources.dregon.get_geometry(dir)` and
  `data_processing.sources.michaels.get_geometry()`.
- **Multichannel is rendered jointly** (native multi-observer), not flattened
  into the batch like RPS prediction; the reference model sums rotors in the
  rfft domain, so M mics cost R forward + M inverse transforms.
- **Channel policy** (`NoiseGenFrameDataset`): `"first"` (default, one mic) or
  `"all"` (every mic); `"random"` is rejected. Any component defined only by
  its *spatial* law (wind channel) is unidentifiable at M=1 — use `"all"`
  (`conf/data/noise_rps_dregon_michaels_swapped_stream_multimic.yaml`).
- **Per-drone conditioning is external to the model**: the generator takes
  `z (B, d)` (FiLM on the emitter, `cond_dim == 0` disables); the `name → z`
  table is `DroneCodebook(d, names)`, a name-keyed `nn.ParameterDict` — the
  model owns `d`, never `K`, so adding a drone never resizes weights and an
  unseen drone is few-shot-adaptable by fitting only its code. In the unified
  framework `build_noise_gen_model(cond_dim=d, drone_names=[...])` returns
  `_CodebookConditionedNoiseGen(generator, codebook)` (codebook is a trainable
  submodule) and `NoiseGenerationCodec(conditioned=True)` passes `drone_names`
  from `meta.drone`; the model resolves `z`. Mixed DREGON + Michael's training
  in one run works out of the box.
- **Output** is clean drone noise `(B, M, T)` — no speech.

## Training integration

- **Script**: unified `train.py` + `conf` (`python train.py experiment=e2_noise_gen_dregon_michaels`).
  The dedicated `train_noise_generation.py` is deleted; E1 is a dead end (its
  model class is unregistered).
- **Registry**: `build_noise_gen_model(name, sample_rate, n_harmonics,
  use_diff_noise, cond_dim, drone_names, amp_calibration=…, mic_eq_knots=…, …)`.
- **Dataset**: `NoiseGenFrameDataset` emits Frames with `rps (rotor,time)` at
  audio rate, `audio (mic,time)` (clean target; M=1 or the full array per
  `channel_policy`), `mic_pos`/`rotor_pos`, `meta.drone`.
- **Loss**: `losses.MultiScaleSTFTLoss` (`pred_key=target_key="audio"`, mic axis
  folded into batch). With a **stochastic** branch (broadband residual, wind)
  prefer `losses.SpectralLikelihoodLoss` + `task_params.distributional=true`
  (the model supplies `coherent`/`noise_psd` via `spectral_stats()`; see
  `src/losses/AGENTS.md`). Smoothness: `losses.SmoothnessPenalty(entry=
  "harm_amps"|"noise_amps", …)` on the extra entries `return_dict=True` exposes.

## Existing implementations

| Model | Key | Approach |
|-------|-----|----------|
| `PositionalHarmonicNoiseGen` | `positional_harmonic_gen` | Per-rotor harmonic + filtered-noise emitter (`HarmonicNoiseGenNew`), propagated (1/r + fractional delay) to every mic and summed |
| `PositionalHarmonicPlusWindGen` | `positional_harmonic_wind_gen` | The above plus the additive incoherent `WindWakeChannel` (refuted as a channel model — docs/experiments/wind-channel-likelihood.md) |

## Checklist for a new noise-generation model

1. [ ] Implement in `src/models/generative/`.
2. [ ] Satisfy `forward(rps (B,R,T), rel_pos (B,M,R,3)[, z]) -> (B,M,T)`.
3. [ ] Register in `NOISE_GEN_MODEL_REGISTRY`; add a `conf/model` YAML.
4. [ ] Smoke test: `tests/tasks/test_noise_generation.py` covers the codec/task
   layer generically; mirror `tests/models/test_positional_harmonic_gen.py` for
   the model.
5. [ ] One-epoch run to verify gradient flow and loss decrease.

## Gotchas

- Always `model.eval()` before rendering: train mode randomises initial
  harmonic phases.
- The magnitude loss fits any stochastic component ~1.6 dB low and is blind to
  a common delay (sees inter-rotor delay differences only).
- `"random"` channel policy cannot work: the inner dataset does not report the
  drawn mic index, so geometry could not be matched.

## Further reading

- `docs/noise-generator-models.md` — full generative reference: `amp_stats`,
  calibration gains, `MicEQ`, wind channel, conditioning/geometry rationale.
- `docs/experiments/noise-generation-augmentation.md` — E2–E4 history and
  § "Framework migration notes" (codebook bundling, codec fix, channel policy).
- `docs/experiments/amplitude-target-training.md`, `wind-channel-likelihood.md`,
  `residual-attribution.md`, `generator-label-sensitivity.md`.
- REPLICATION.md § E1–E4 — historical vs unified commands and caveats.
