# Handoff: fitting the rotor-noise model to DREGON and Michael's, and refining the labels

**Branch** `stochastic-fit`, 112 commits, merged into `main`. Everything below
is on `main` now. Read this first, then the two explainer pages, then the
experiment records.

## 1. What exists, in one paragraph

A generative model of rotor noise — one harmonic comb per rotor plus a smoothly
coloured broadband floor — is fitted to real recordings by **Whittle MAP on its
expected periodogram**, and the fitted vector maps one-to-one onto the
renderer's `StochasticParams`, so what is fitted is what is rendered. Three
regimes are fitted: the DREGON single-motor bench, Michael's FLY125 cruise and
FLY125 standby. Synthetic cruise and standby clips are tracked by a **frozen**
rotor-speed model at close to real-data accuracy; a blind bench→flight transfer
is not. Separately, every recording with telemetry now has a **refined
rotor-speed label** published as an extra series in the frames datasets.

## 2. The model and where it lives

* `src/experiments/stochastic_fit/model.py` — the model as its expected
  periodogram, `CombSpectrum`. `M = W * [floor + Σ_r g_mr L_r]`, lines
  `Σ_k P_rk · D(f − k·c_r; γ_rk)`, `P_rk = 10^((A_rk+h_rk)/10)·(s_r/80)^q`,
  floor = 14-knot log-frequency shape + tilt + level with its own exponent.
  `whittle()` is the likelihood, `prior()` the penalties, `export()` the
  renderer coordinates.
* **The line is two components** (the main modelling result). Each order carries
  one power split by a coherent fraction `w_k = exp(−(k/k_half)²)`: share `w_k`
  is a coherent tone carrying the analysis window's own power response
  `|W(f−f0)|²` (`needle()`, `hann_power_response`, `Spec.needle_window_shape`),
  share `1−w_k` is a Rayleigh pedestal of the fitted width. On Motor1_80:
  one component −1.1404, Lorentzian needle −1.1427, window needle **−1.1462**
  nats/cell — about 1030 nats for one parameter.
* `src/experiments/stochastic_fit/rig.py` — joint MAP over several clips with
  rig-level ties (`fit_rig`, `RigSpec`, `ClipInRig`). **Careful**: a rig fit over
  BENCH clips has `R = 1`, so its tied profile is one shape plus a scalar level
  per clip. `bayes_rig.json` is *not* four learned motor shapes.
* `src/data_processing/stochastic_rotor_noise.py` — the renderer.
  `coherence_k_half` splits each order between the tone bank and narrowband
  noise; `gamma_min_bins` must be 0.01 for these fits (0.6 bins is a 12.9 Hz
  censor at 44.1 kHz).
* `src/experiments/stochastic_fit/native.py` — native 44.1 kHz loading
  (`load_native_clip`, `decimate`, `bench_clip`). **Native audio is the only
  source**; the published 16 kHz sets have an 88–90 dB brick wall at 7.9 kHz
  that is a learnable domain cue. Raw trees resolve through
  `streams.ensure_local`, so the same code runs on a cluster node.

## 3. Stage fits, results, artifacts

| stage | driver | result file | fitted `k_half` | `γ0` |
|---|---|---|---:|---:|
| S1 bench, Motor 1 | `stage1_bayes.fit` | `results/S1/bayes_motor1_needle.json` | 9.7–54.6 | 0.30 Hz |
| S1 bench rig, 20 cells | `scripts/_stage1b_fit.py` | `results/S1/bayes_rig.json` | 19.4 median | 0.15 Hz |
| S2 FLY125 cruise, 8×16 s | `scripts/_stage2_fit.py` | `results/S2/cruise_8clip.json` | 1.6–2.9 | 0.00 Hz |
| S2 cruise, drifting floor | same, `--floor-dynamics` | `results/S2/cruise_8clip_dyn.json` | ~2.3 | 0.00 Hz |
| S3 FLY125 standby, 1×12.5 s | same, `--regime standby` | `results/S2/standby.json` | 5.58 | 2.62 Hz |
| DREGON free-flight | same, `--recording free-flight_nosource_room2` | `results/S2/dregon_flight.json` | 1.24 | 12.37 Hz |

`k_half` orders itself across regimes with nothing forcing it — 19 clamped,
5.6 standby, 1.6–2.9 flying, 1.2 DREGON flight — which is the direction a
steadier shaft implies.

**Modules**: `stage1_bayes.py` (bench, `BENCH_VARIANT` pins all dynamics off),
`stage2.py` (flight; `REGIMES`, `cruise_windows`, `params_from_export`,
`render_from_export`, `FLOOR_DYNAMICS`), `accept_stats.py` (LTAS bands, order
bands, `paired_delta`).

## 4. The acceptance gate that works

`scripts/_stage2_probe.py` — 12 independent synthetic draws, each with its own
seed and a real rotor trajectory, scored by the **frozen** `hppnet_l2_r2_s0`
through `metrics.salience_layers.LayerPeakRPSMetric`, all 8 channels, PIT.

| arm | median 8-mic MAE (rev/s) | spread | refusals |
|---|---:|---:|---:|
| FLY125 cruise, synthetic | **0.756** | 0.226 | 0/12 |
| FLY125 cruise, real (same pipeline) | 0.347 | 0.097 | 0/12 |
| FLY125 standby, synthetic | **0.084** | 0.034 | 0/12 |
| FLY125 standby, real | 0.054 | 0.014 | 0/12 |
| DREGON flight, real | 1.069 | 0.271 | — |
| DREGON flight, flight-fitted | 2.077 | 0.773 | — |
| DREGON flight, bench-blind transfer | 2.488 | 0.256 | — |
| diverse control policy (counterexample) | 19.600 | up to 66 | 7/24 |

The counterexample row is what makes the gate meaningful: it *can* fail.
Floor dynamics improve the likelihood on every window (−0.519 vs −0.507
nats/cell) and leave the tracker unchanged (0.767 vs 0.756) — a **negative
result**, so gusts are not the explanation for the residual gap to real.

DREGON is the open puzzle: fitting its own flight gets the envelope nearly
right (LTAS 1.54 dB) and still tracks at 2.08, because its lines are fitted
wide and almost fully incoherent (`γ0 = 12.4` Hz, `k_half = 1.2`), so the comb
is smeared. `scripts/_dregon_transfer.py` runs the four-arm comparison.

## 5. Pages to look at (and listen to)

* `docs/explainers/bench-fit-derivations.qmd` → `.html` — the derivations in
  Simplified Technical English: the Whittle likelihood from first principles,
  the coherent fraction of a random-walk tone in 10 steps, the needle/pedestal
  mixture, the bench rig (20 cells), the DREGON bench→flight transfer with
  HPPNet-vs-truth figures, **30 audio players**, and a complete defect list.
* `docs/explainers/stage2-cruise.qmd` → `.html` — Michael's cruise and standby:
  spectrograms, LTAS, per-microphone pattern, the gate tables, 15 players.
* Assets: `docs/explainers/stage1-bench/` (20 panels, 60 WAVs),
  `docs/explainers/stage2-cruise/`, `docs/explainers/dregon-transfer/`.
* Serve with `python -m http.server --directory docs/explainers`; verify with
  Playwright chromium (`PLAYWRIGHT_BROWSERS_PATH` from the flake).

## 6. Label refinement (see `docs/experiments/refined-rps-labels.md`)

* **Standby is never refined.** `src/data_processing/rps_gating.py`: telemetry
  exactly below 45 rev/s, refinement exactly in settled cruise (≥65), the
  correction blended across the ramp; regime from the slowest rotor. Evidence:
  standby corrections made the refiner's own comb fitness *worse* (−0.0021
  FLY125, −0.0788 DREGON room1).
* **Coverage is complete**: 12 sidecars in `src/data_processing/refined_labels/`
  (`<id>.npz` + `.report.json`) — Michael's FLY124/FLY125 and all 10 DREGON
  recordings with telemetry. The five `*_room2` flights publish only
  `motors_command` and were silently skipped until `resolve_rps_key` resolved
  the reference track per frame.
* **Published as an extra series** `rps_refined`, attached by the
  `source_frames` derivation (`refined_label_track.py`), never by the builders:
  `DREGON-frames@261b09971c8a` (recipe_version 2),
  `michaels-frames@8e9d149560dd` (recipe_version 3). The raw track is untouched
  beside it. Each spec's `gen["refined_labels"]` names the sidecar bytes
  (16-hex SHA-256 prefixes), checked by `_verify_refined_labels` at generation.
* Producers: `scripts/refine_dregon_rps.py` (gated inside), re-gate an existing
  sidecar with `scripts/fix_refined_labels.py`.

## 7. Instruments worth reusing before writing a new one

* `scripts/_cell_coherence.py` — per-order cell CV, band CV, equivalent width,
  centre-bin share. This is what identified the two-component line.
* `scripts/_two_component_fit.py` — nested line-model comparison on one clip.
* `scripts/_coherence_profile.py` — where the coherence information lives.
* `scripts/_stage2_panels.py`, `scripts/_stage2_probe.py`,
  `scripts/_dregon_transfer.py`, `docs/explainers/stage1-bench/build_bayes.py`.
* `src/experiments/stochastic_fit/identify.py` — marginal Fisher information
  per line (which orders are identifiable at all).

## 8. Traps, all paid for once

1. **A width floor above what you intend to represent is a censor.** Bit twice:
   `gamma_min_bins = 0.6` in the fit and the same constant in the renderer.
2. **A band narrower than the line censors the measurement.** A ±6 Hz band
   cannot report a width above ~6 Hz; that produced a fake "saturation" and a
   fake −6.25 dB trend at order 64.
3. **Coherence cannot be scored bin-by-bin.** A coherent tone's shape is
   `|W|²` with nulls; spreading coherent power over a Lorentzian charged the
   truth +3268 nats. It lives in the MEAN spectrum instead, via the mixture.
4. **Non-overlapping analysis frames.** The Whittle likelihood treats cells as
   independent; 50 % overlap double-counts the data. `HOP = N_FFT`.
5. **Memory.** The two-component forward pass keeps several
   `(rotors, orders, frames, bins)` intermediates; order chunks are recomputed
   in the backward pass (`torch.utils.checkpoint`) or a 5-clip fit is killed
   with no traceback. The refiner needs ~12 GB per worker: `--jobs 1`, ≥64 GB.
6. **A change that changes no number is a bug**, not a null result — a failed
   string patch left a parameter inert with an identical NLL in every digit.
7. **Cache keys must name the data, not the loop index.** `probe_00` made the
   standby probe read cruise clips and report a fake failure at 2.78 rev/s.
8. **`load()` in an explainer is rooted per page**; a doubled path returns
   `None` and the page renders an empty table silently.

## 9. What is genuinely open

1. The **DREGON flight gap**: envelope right, comb too smeared to track
   (2.08 vs 1.069 real). Next suspects, in order: no per-microphone inter-rotor
   phase structure; the width law; telemetry error the carrier cannot absorb.
2. **Per-motor bench shapes.** `bayes_rig.json` is one shape + clip gains;
   four independent single-motor fits (`scripts/_stage1b_fit.py --motors m`)
   were submitted for this and should be re-run and folded into the transfer.
3. **S1 absolute gates.** Real bench clips fail the preregistered 1.5/3.0 dB
   LTAS thresholds against *other real clips* (2.00 / 4.49 dB), because floor
   shape varies by motor position. The gate needs restating distributionally
   or conditionally — **user decision pending**
   (`docs/experiments/stage1-bench-fit.md`).
4. **Dynamics.** Line drift stays off because it absorbed +4.97 dB of static
   per-order level; a capacity-matched reparameterisation is the fix.
5. `accept_stats.paired_delta` sizes its integration band from `γ` alone and
   must integrate needle and pedestal separately.
6. The **GPU refinement path** (`TRACKING_DEVICE=cuda`) is unbenchmarked;
   Kaggle buffers logs until completion and collected no outputs.
7. `decomp-frames-v1/v2` still pin `rps_override_dir` for room1 only.

## 10. Records

* `docs/experiments/refined-rps-labels.md` — label refinement, gating, publication.
* `docs/experiments/stage1-bench-fit.md` — the bench campaign and the gate blocker.
* `docs/explainers/bench-fit-derivations.qmd`, `docs/explainers/stage2-cruise.qmd`.
* `src/data_processing/AGENTS.md`, `docs/data-catalog.md` — dataset contracts and pins.
