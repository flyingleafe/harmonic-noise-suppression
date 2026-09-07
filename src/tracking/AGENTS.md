# AGENTS.md — `src/tracking`

## Purpose

Rotor-speed tracking: given drone audio, each rotor's speed over time. Vold–Kalman order
tracking, phase-increment refinement, blind seeding, beam/DP search, trajectory goodness-of-fit,
and the windowed comb decomposition. Pure array code (Purity rule below). Extracted from
`data_processing` in the 2026-08 refactor (`docs/refactor-2026-08-plan.md` §4).

## Map

Four layers; each knows only the layer below. `protocols.py` sits beside all four.

| Layer | Module | What it is |
|---|---|---|
| Front door | `top.py` | frame plumbing, EVERY stage (+ config dataclass), EVERY shipped variant as a named composition |
| Ladder cores + frozen configs | `pipelines.py` | blind-annotation ladder + `CAPTURE_CFG`/`REFINE_CFG`/`TRACK_CFG`/`SEED_CFG` (changing them invalidates published annotations), the `blind_fullrange` coarse pass, the flagship peel (`ARMS`, `make_peels`), ONE tooth sampler `comb_teeth`. Computes, never a Stage |
| Algorithm cores | one algorithm each (below) | arrays in, arrays out |
| Primitives | `dsp.py` | `zoom_bands`, `demod` (the one demodulation driver), `boxcar`; `dsp_config`/`resolve` (device, pad), `thread_pool`. Torch only, numpy at every seam; imports nothing but numpy + torch |

Algorithm cores:

- **Tracking** — `vk_tracking.py` (`VKConfig`, `vk_track`, `vk_envelopes`, `ls_project_envelopes` = the LS peel); `phase_increment_tracker.py` (`pi_kalman_refine(peel_audio=, pair_audio=)` — the PEEL SEAM); `joint_phase_kalman.py` (joint 4-rotor variant, probe only); `warp_refinement.py` (`iter_warp_refine`); `rps_refinement.py` (`refine_coherent`).
- **Seeding** — `vk_blind_seeding.py` (`SeedConfig`, `blind_seed`, `stage_guard`, `logmag_spectrogram` = THE whitened spectrogram); `rotor_dp.py` (`viterbi_path`); `joint_beam_tracker.py` (`joint_beam_track`); `comb_seed.py` / `comb_fit.py` (research cores, not yet stages).
- **Judging** — `fitness.py` (`score_cells -> FitnessScore`, `Holdout`, `apply_control`, `residual_decompose`, `measure_tdoa`); `fitness_vk.py` (F_VK: `fvk_score`, `fvk_loss`, `optimize_trajectory`); `telemetry_refit.py` (`refit_window`, `presmooth`, `scale_summary`); `comb_displacement.py` (`demod_comb_bank`); `order_domain.py` (`comb_scan(half=)`); `phase_noise.py` (WP18 `arm_covariance`, `brickwall` = THE whole-window FFT filter, linewidth readings).
- **Decomposition** — `decompose.py` (v2: `solve_config`, `k_cap`, `group_plan`, `solve_window`, `windowed`/`stitch_windows`, `order_cell_profile`) and `joint_decompose.py` (v3/v4: `JointState`, the three blocks, `stochastic_split`, `map_objective`, `fit_floor_powers`). Meant to be REBUILT from primitives (Further reading).
- **Constants / data** — `rotors.py` (`MIXER`, `modes_from_rps`; re-exported by `data_processing.rps_synthesis`); `protocols.py` (Contracts).

Tests: `tests/tracking/`. Frozen guard clip: `results/tracking_ref/`.

## Contracts / Invariants

- **No module outside `top.py` defines a stage; no transform lives outside `dsp.py`.**
- **Stage API.** `Stage = Callable[[td.Frame], td.Frame]`. `"audio"`: `(mic, time)` float32 on a `GridIndex` (`tracking_frame` accepts `(T,)`; `dtype=np.float64` keeps float64 exact). `"rps"`: `(rotor, time)` float64 on a `StampIndex` — the current candidate. `"rps_meas"`: optional reference, never touched. A stage replaces `rps` via `with_rps` and appends one `{"stage": name, ...}` dict to `meta["tracking"]` (append-only; frames never mutated). Cores take `(T,)` or `(C, T)` audio; frame times re-base to the audio's `t_start`.
- **Seams.** A stage that leaves the trajectory alone puts its product in `meta` and logs nothing; the consumer logs what it ate: `peel_stage` → `meta["peel_seam"]` (eaten by `pi_kalman_stage`); `decompose_stage` → `meta["decompose"]`; `meta["joint"]` = the v3 `JointState`, read AND rewritten by the block stages. Hence `pipeline(peel_stage, pi_kalman_stage)` equals `pi_kalman_arm_stage` bit for bit (`tests/tracking/test_top.py`).
- **Purity rule.** Imports only `numpy`, `scipy`, `torch`, `tdseries`, `utils`. NEVER `data_processing`, `models`, `training`. Direction: `data_processing → tracking`.
- **Protocols as data.** `protocols.py`: `ProtocolSpec`/`WindowSpec`/`PoolSpec`, registries `BEATVK`, `VK37`, `PROTOCOLS`; `iter_windows`, `to_frame`, `FROZEN_FLY124_ALIGNMENT`; operations that exist exactly once — `slice_window`, `pit_align` (THE Hungarian assignment; `losses.pit.align_rps_to_gt` delegates), `pool_means`, `load_prep_window`. Audio loaders are injected by scripts.
- **Fixed degrees of freedom in every judge.** Band, grid, gate and harmonic cap are pinned to the window's REFERENCE trajectory, so every candidate is scored on identical cells.

## Entry points / API

Stages (`top.py`): `blind_seed_stage`, `coarse_init_stage`, `vit2dsp_stage`, `vk_stage`, `peel_stage`, `pi_kalman_stage`, `warp_stage`, `refine_coherent_stage`, `presmooth_stage`/`scale_stage`/`shift_stage`, `fitness_stage`, `fvk_stage`/`fvk_refine_stage`/`decompose_stage`, `joint_init_stage`, `vk_solve_stage`/`phase_split_stage`/`floor_stage`/`stochastic_stage`, `refit_stage`. Combinators: `pipeline`, `iterate(stage, n)`, `windowed(inner, window_s=, hop_s=)`, `guarded(inner)`.

Recipes: `vit2dsp`, `blind_fullrange`, `flagship(n_apps)` = n × (peel → pi_kalman), `peel_alternation` (every frame kept), `judge(candidate)`, `joint_solve_window` (floor → (iters−1) × (solve → split → floor) → solve; `stochastic=True` adds regime 3; `JointConfig.v4` swaps the BLOCKS, not the alternation, and refuses `stochastic`).

```python
import tracking as trk
frame = trk.tracking_frame(audio, 16000, meta={"recording_id": rid})
out = trk.pipeline(trk.blind_seed_stage(4), trk.guarded(trk.vk_stage(trk.VKConfig())))(frame)
r, ft = trk.get_rps(out)   # (4, N) rev/s + frame times; log: out["meta"]["tracking"]
```

Live drivers (each injects data and calls a recipe): `scripts/tracking_ref.py` (the guard), `scripts/beatvk_*.py`, `scripts/rps_eval.py`, `scripts/vk_validation.py`, `scripts/vk_phase_validation.py`, `scripts/telemetry_*.py`, `scripts/displacement/*.py`, `scripts/refine_dregon_rps.py` (→ `src/data_processing/refined_labels/`), `scripts/vk_decompose.py`, `scripts/joint_rescore.py`, `scripts/phase_coherence_probe.py`, `scripts/rps_refine_lab.py` (blind-seed arm ladder, not yet promoted), `scripts/{jb,sr_dp,joint_kalman}_probe.py`.

## Conventions

- A script never assembles a ladder of its own: a variant worth running is a named recipe in `top.py` with its array core in `pipelines.py`.
- Performance knobs, all bit-identical except `pad`: `TRACKING_FFT_WORKERS`/`thread_pool(n)` (default 1 — opt in off Slurm), `TRACKING_DEMOD_BUDGET_MB` (a CACHE knob on CPU), `TRACKING_DEVICE=cuda`, `vk_tracking.LS_TILE_BYTES`, `TRACKING_PAD=fast` (opt-in). No backend knob — torch is the only transform (`docs/tracking-performance.md`).
- Retired drivers' RESULTS stay on disk and scorable; only the producing code goes.

## Gotchas

- `optimize_trajectory(smooth_lambda=1.0)` is cruise-calibrated (a takeoff ramp cannot move); whole-recording drivers want `"auto"`. Basin knob: `bw_rps`, not `k_max`. In `fvk_loss` the `(stride/2) rho^2` prior weight and `edge_taper` are load-bearing (`docs/trajectory-fitness-design.md` §6).
- `residual_decompose.scale_pct` is unidentified on a cruise window — read `design_cond`, use `telemetry_refit.scale_summary`; `presmooth` detrends before `brickwall` (`docs/experiments/telemetry-fitness.md` § "Two defects").
- `phase_noise` line widths need their OWN 500 Hz envelope grid; a censored width is `nan` (`gamma_bound_hz` = upper bound); fit the law on per-`k` MEDIANS by Theil-Sen; quote `rho_k` with its `smooth_hz` (`docs/experiments/rps-refine-precision.md` § "WP18 follow-up").
- Decomposition: the ladder starts at `k` 3 (envelope band, not phase unwrap); whitening is bandwidth-neutral by default; the floor mask is ~3 linewidths, capped; read `group_plan` before sizing a job. v4: `J_v4` scores the ORIGINAL signal; an unfactorizable group FAILS (never `splu`); twin pairs hit the `v4_band_law=False` retry.
- `scripts/tracking_ref.py --exact` no longer passes and is not expected to; tolerance mode is the bar.
- `pad="fast"` helps only when the bad factor is in `n_env`, never in `stride`; pick an `fs_env` whose stride factorizes.
- `comb_displacement` and `order_domain` are two INDEPENDENT estimators built to fail differently; trust neither without the half-integer null (`half=True`).
- Search/refinement ablations: `Vit2dspConfig(stop_after="vit2dsp")` returns the spatial two-pair DP trajectory BEFORE the midband and refine VK stages (`"viterbi_c"` stops at the pair-mean search; neither computes the discarded stages). `scripts/blind_valid_row.py annotate --arm vit2dsp_dp` persists it; `--init-traj-dir <search>/traj --phase-iterations 1|3` consumes that exact trajectory (hash recorded). Journal 37-clip driver, not `vk37`/`beatvk`.

## Further reading

- `docs/vk-order-tracking-design.md` — coupled VK design; §7.5 closes the blind-annotation campaign (its deleted drivers vk_blind_annotation / vk_blind_sweep became `pipelines.py` + `protocols.pit_align`); §8 2026-07 CPU fast paths.
- `docs/vk-decompose-v3-design.md` — v3 model, blocks, knobs, stitch; §10 the primitive inventory (state, blocks, regime 3, window layer, readings, marginal + H-aware objectives, 2026-08 renames). v1/v2: `docs/experiments/vk-decomposition.md`.
- `docs/v4-unified-model-design.md` — the unified model and its as-built inventory (F1/F2, `J_v4`, conditioning floor, band-law fallback, constants, tests).
- `docs/trajectory-fitness-design.md` — F_VK's basis; §6 implementation notes.
- `docs/tracking-performance.md` — knobs, kernel paths, CPU/GPU tables, the peel, the profile, the guard.
- `docs/tracking-hands-on-tutorial.md` — notebook walkthrough.
- `docs/experiments/telemetry-fitness.md` — issue 17 phases 6a–6e (judge, fitter, campaign, ridge, time shift).
- `docs/experiments/dregon-comb-displacement.md` — the displacement campaign and what it withdrew.
- `docs/experiments/rps-refine-precision.md` — WP1–WP20 refinement ladder, WP18 covariance (its window-builder script died with the campaign), line-shape follow-up.
- `docs/experiments/beat-vk.md` — beat-VK protocol; the deleted neural_reanchor driver (negative result) and the never-run `cd_iter` chain are recorded there.
