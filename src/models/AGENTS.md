# src/models/ — Model Implementations

## Purpose

All neural network models. Editable-installed package (`models` in
`pyproject.toml`): `from models.X import Y` works anywhere. Every model is
built through `models.registry`.

## Map

```
src/models/
  registry.py           RPS_MODEL_REGISTRY / LEGACY_MODEL_BUILDERS / NOISE_GEN_MODEL_REGISTRY /
                        DIRECT_FACTORY_TYPES; build_model(), build_noise_gen_model(), model_types()
  frontends/            Spectral front-ends (build_frontend, register_frontend)
  rps_predictor.py      SimpleConv* family (encoders, pools, temporal heads, GatedProjection)
  salience_rps.py       Salience-map RPS baselines (multif0_salience, basic_pitch_salience)
  salience_crf.py       Per-rotor salience layers + CRF readout (lossless encode/decode pair)
  multif0/              Multi-F0 HCQT+CNN (Cuesta 2020): LateDeep etc., hcqt.py, nnaudio_cqt.py
  basic_pitch/          Basic Pitch PyTorch port (Bittner 2022) — own AGENTS.md
  harmonic_ports/       HarmoF0/HPPNet/hFT on a linear rate grid + *_orig controls + layer_readout.py
  comb_salience.py      CombGather / CombScoreHead — hypothesis scoring at k*r
  comb_crf.py           Linear-chain CRF over the rate grid (the deployed decoder)
  comb_slots.py         SlotCombNet — R slots that ALLOCATE bins (candidate C1)
  comb_slots_emission_v2.py / comb_slots_prior.py   v2 emission groups / pairwise rate prior
  ckla.py / ckla_triton.py   Complex Kalman Linear Attention head (+ optional fused Triton scan)
  hg_ckla.py            Harmonic-gather CKLA refiner (docs/pikalman-ckla-design.md)
  fkla/                 Vendored plain-KLA flat layer (kla-loglinear@11e5a39)
  dcunet.py / dccrn.py  DCUNet / DCCRN, encoder-side RPS fusion
  dcunet_refactored.py  DCUNetRefactored / DCCRNRefactored, decoder-side RPS fusion
  demucs4ht.py / htdemucs_ft.py / tfgridnet.py / mpsenet.py / sgmse/ / diffusion_buffer.py / dptnet/
                        SE baselines (HTDemucs + official-checkpoint fine-tune, TF-GridNet, MP-SENet,
                        SGMSE+, DiffusionBuffer, DPTNet)
  edge_bs_rof/          BSRoformer, MelBandRoformer; rps.py = Edge-BS-RoFormer trunk for RPS
  generative/           RPS → noise generators — docs/noise-generator-models.md
```

## Registries (`registry.py`)

`model_types()` merges all four; `kind` ∈ `rps` / `legacy` / `noise_gen` / `factory`.

**Legacy SE models** (`LEGACY_MODEL_BUILDERS`, config-dict builders): `edge_bs_rof`,
`dcunet`, `dccrn` (encoder-side RPS: `use_rps`, `dcunet_rps_fusion`),
`dcunet_refactored`, `dccrn_refactored` (decoder-side RPS only: `rps_fusion`),
`rps_predictor`, `dptnet`, `htdemucs`, `diffusion_buffer`, `sgmse`.
**Direct factories** (`DIRECT_FACTORY_TYPES`): `tfgridnet`, `mpsenet`, `htdemucs_ft`.
**Noise generators** (`NOISE_GEN_MODEL_REGISTRY`): `positional_harmonic_gen`,
`positional_harmonic_wind_gen`.

**RPS models** (`RPS_MODEL_REGISTRY`, `build_model(name, **params)`) — the dict
is the authoritative key list. By family:

| Family | Keys | Notes |
|--------|------|-------|
| SimpleConv baselines | `simple_conv`, `_wide`, `_tcn`, `_multiscale`, `_bigru`, `_bigru_v2`, `_magphase_bigru`, `_attn_pool`, `_se_next` | encoder variants + Conv1d/TCN/BiGRU heads |
| `simple_conv_v2` (residual + SE + attn pool + BiGRU) | `_tcn`, `_causal_tcn`, `_smol_tcn`, `_smol_causal_tcn`, `_smol_bigru`, `_uni_gru*`, `_causal_gru{,96}`, `_transformer`, `_local_attn`, `_multires`, `_dwt`, `_magphase`, `_dual_pool`, `_freqpos`, `_freqcat`, `_freqhires`, `_gru96` | `_uni_gru*` head-only causal; `_causal_gru*` fully time-causal |
| SMoLnet backbone | `smolnet_rps_{tcn,simple_head,causal_tcn}` | compressed re/im STFT body |
| Transformer × front-end | `simple_conv_v2_transformer_{hcqt,if,learned,comb,pyramid}` | VK-parity arms (G2a/G2b/G4/G8) |
| CKLA heads | `simple_conv_v2_ckla{,_mag,_norot,_mag_norot,_phasediff,_phaseonly,_phaseunit}`, `_ckla_phaseonly_cond` (refiner: `forward(audio, cond)`, non-PIT MSE), `_fkla`, `hg_ckla_refiner` | docs/ckla-design.md, docs/pikalman-ckla-design.md |
| Complex-encoder RPS | `dcunet_enc_rps`, `dccrn_enc_rps`, `dccrn_lite_rps`, `edge_bs_rof_rps` (lazy) | encoder + `RPSPredictionHead` |
| Salience (multi-pitch) | `multif0_rps`, `multif0_salience`, `basic_pitch_salience` | `salience_rps.py` |
| Harmonic ports | `harmof0_rps`, `hppnet_rps`, `hft_rps`; controls `harmof0_orig`, `hppnet_orig` | `harmonic_ports/` |

## Spectral front-ends (`frontends/`)

```python
class SpectralFrontEnd(nn.Module):
    key: str; out_channels: int
    def forward(self, audio: Tensor) -> Tensor:   # (B, N) → (B, C, F, T)
    def num_frames(self, n_samples: int) -> int:
```

| Key | Ch | Content |
|-----|----|---------|
| `stft_mag` | 1 | log₁₊ magnitude (default; weight-identical to pre-front-end checkpoints) |
| `stft_magphase` | 3 | log mag + cos θ + sin θ |
| `stft_mag_if` | 2 | log₁₊ mag + IF deviation (fractional bins) |
| `stft_ssq` | 1 | log₁₊ synchrosqueezed magnitude (power scattered to the rounded IF bin); same grid as `stft_mag` |
| `hcqt` | H / 2H | Harmonic CQT (librosa); `phase=True` → 2H; `stacked=True` = one CQT + shifts (lossy at h≥3) |
| `pyramid_if` | 2 / 8 | 4-band multi-window STFT pyramid, log1p-mag + IF on a 340-row log-f axis (`collapse_bands`) |
| `comb_if`, `comb_if_ramp` | 4 / 3 | whitened comb matched-filter + IF consensus + occupancy + coord row on a 361-row candidate-f0 grid (`coord_channel`); `_ramp` widened for ramps |
| `learned_conv` | C | free time-domain filterbank on the raw waveform (phase kept) |

Adding one: subclass `SpectralFrontEnd`, set `key`/`out_channels`, decorate
`@register_frontend`, import it from `frontends/__init__.py::_ensure_imported`.
SimpleConv* models take `frontend=` as an instance **or a registry key string**
(built with the model's `n_fft`/`hop_length`); the first encoder block adapts.

## Contracts and invariants

- **RPS regressors**: `forward(audio (B, N)) -> (B, num_rotors, T)` on the
  hop-512 STFT grid; PIT-MSE. `voicing_gate=True` swaps the head's final
  `Linear` for `GatedProjection` (`speed * sigmoid(gate)`; keys move to
  `head.proj.linear.*`, not weight-compatible with the default).
- **RPS conditioning of SE models**: `RotorEncoder` (two Conv1d, 64 ch) encodes
  `rps (B, R, T_rps)`; `dcunet.py`/`dccrn.py` fuse encoder-side
  (`dcunet_rps_fusion` ∈ `bottleneck`/`gru`/`hierarchical`),
  `dcunet_refactored.py` decoder-side only (`rps_fusion` ∈
  `bottleneck`/`hierarchical`/None) — docs/dcunet-refactored.md.
- **Salience models** (`outputs_salience = True`): `forward(audio) -> (B, G, T)`
  logits; BCE via a salience `conf/loss`; `predict_rps()` decodes to the STFT
  grid so PIT metrics apply unchanged. Harmonic ports declare their linear rate
  grid as `SalienceRPSPredictor.out_freqs`. `n_maps=4` (`_l4` configs) stacks
  per-rotor layers along the codec's freq axis (width `n_maps * G`), needs a
  linear output grid and `conf/{loss,metrics}/salience_layers_r150{,_h256}.yaml`;
  `n_maps=1` is the old model exactly. Zero convention: target dark only for
  `rps <= 0.1`; a frame with no peak decodes to 0 rev/s for every rotor (never
  hold-over, never NaN), track identity survives.
- **Slot-comb CRF**: `SlotCombNet(emission="classical"|"partial"|"v2", parts=…,
  off_state, learned_transition, mask_below_grid)`; every option holds the
  zero-parameter corner at init. OFF index is `G` in every `comb_crf` path.
- **Noise generators**: `forward(rps (B,R,T), rel_pos (B,M,R,3), z|drone_names) -> (B,M,T)`;
  `cond_dim>0` returns `_CodebookConditionedNoiseGen` (codebook is a submodule).
- **Checkpoint compatibility**: pre-0.13 SimpleConv* stored the Hann window as
  `window`, now `frontend.window`; every class's `load_state_dict` remaps, so
  `strict=True` loads of old files work. `LateDeep(fused_branches=True)`
  converts layouts via a pre-hook likewise.

## Conventions

- New RPS model: implement, add to `RPS_MODEL_REGISTRY`, add a `conf/model`
  YAML; salience-type models need a matching `conf/loss` + `conf/metrics`.
- Heavy optional stacks (roformer, triton) are imported lazily inside builders.
- `harmonic_ports/__init__` exports its models lazily (PEP 562) because
  `salience_rps` imports `LayerCRFReadout` back out of it — keep that order.

## Gotchas

- Swapping `BiGRUHead` for a unidirectional GRU is unstable; fully time-causal
  STFT variants underfit — an alignment problem, not a head problem
  (docs/experiments/simpleconv-rps-architecture-search.md § causal follow-up).
- A rate grid that reaches 0 needs `f_min` (near-DC gather), constant-`k_max`
  normalisation and `r_min` (sub-bin candidates): `harmof0_rps.py`, `hft_rps.py`.
- `harmof0_orig`/`hppnet_orig` take an L2 output adapter (`superres_out=True`,
  `n_maps=4`, `out_fmin/out_fmax/out_bins`, `head_hidden/head_kernel`; attached as
  `superres_head`, since HPPNet's `head` is its `FreqGroupLSTM`): `FreqSuperResHead`
  over the native 352-bin log grid → four 300-bin maps on 0–150 rev/s; clamps below
  27.5 Hz. `n_maps>1` REQUIRES it. Configs `conf/model/{harmof0,hppnet}_l2.yaml`.
- Slot-comb v2: a knob that starts "off" starts with ~0 gradient (`gap_mu_init`,
  `read_sigma_init`); weights are `softplus`, never `1 + MLP` (went non-finite).
  `mask_k_max` caps a 121 s mask-bank build at `r_lo=10`.
- `hcqt` `stacked=True` and non-stacked checkpoints are not interchangeable.
- Generators: always `model.eval()` before rendering (train mode randomises
  initial phases); 44.1 kHz `drone_audition` checkpoints do not load.
- `DroneNoisePlusFilterGen` is port-only (no `conf/model`, REPLICATION.md § E1).

## Further reading

- `docs/noise-generator-models.md` — generative package, conditioning, geometry, losses.
- `docs/harmonic-ports-design.md` + `docs/experiments/harmonic-multipitch-ports.md` — linear rate grid, per-rotor layers, controls.
- `docs/experiments/comb-slot-crf.md` § "Implementation reference" — v2 emission parts, chain flags, costs.
- `docs/experiments/salience-map-rps-tracking.md` § "Implementation notes" — BCE path, `_l4`, `fused_branches`, `stacked_hcqt`.
- `docs/experiments/honest-base-frontends.md` § "Code" — voicing gate, string front-ends, `hb_*` configs.
- `docs/ckla-design.md`, `docs/pikalman-ckla-design.md`, `docs/slot-comb-v2-design.md` (archived), `docs/dcunet-refactored.md`.
