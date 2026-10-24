# Synthetic training on the rp protocol

**Status:** in progress · started 2026-10-22

## Motivation

The rp protocol (`paper-real-protocol.md`) trains on real noise recordings
only. This campaign trains the same models on synthetic noise, synthesized
online, under the same protocol: the same model, loss, LR schedule, batch,
speech mixing and real validation panel. Each synthetic family is a "simple
variant" that keeps one property of real rotor noise and drops the rest. The
first variant is the **diverse static comb**:

- each rotor gets a harmonic comb whose profile stays constant within a window;
- profiles come from a wide population: even/odd power laws, Student-t line
  departures, drop-outs and rotor levels;
- each microphone gets a white floor that buries a drawn share of the lines;
- the rotors fly the fitted trajectory model under the reduced hyperprior
  (`rps-trajectory-model.md` § "Reduced hyperprior").

The stream is `conf/online_mix/rp_combdiv.yaml` and the draws are shown in
`docs/explainers/diverse-comb.qmd`.

## Setup

**Stream.** `rp_combdiv` is `rp_all` with the real noise replaced by
`kind: diverse_comb` (weight 2.0), next to the silence arm (0.4). Everything
else matches `rp_all`: 2 s clips, LibriSpeech train-clean-100 per channel at
U(−30, +5) dB, random_gain and spectral_recolor at p 0.5. Each comb render
is 4 s long and is reused for 40 draws, each draw cutting its own 2 s window
(`render_reuse: 40`, `render_duration_s: 4.0`), as in the `prior` stream.
The lines go through the noise model's line kernel (`noise_model.render`).

Measured on one thread, 8 mics, 2026-10-22:

| stream | cost per 2 s draw |
|---|---|
| comb | 0.006–0.012 s |
| `noise_prior_v1_5050` | 0.024–0.027 s |

**Validation** (`conf/validation/rp_real_combdiv.yaml`). Two parts:

- the rp real panel, unchanged: all 22 scores, with `real_macro` among them;
- `static_comb`: 512 mono items (64 draws × 8 mics) of the `rp_combdiv`
  stream, drawn once at the start of each run (`FixedSynthFrameDataset`).
  - Seed 771001; training uses 20261022.
  - 8 s clips, like the real panel.
  - One drone, flight and render per clip.
  - Speech and SNR as in the stream; augmentations stripped.
  - The silence arm is included at its stream weight.

**Stop.** `validation.control: [real_macro, static_comb]`. The run stops at
saturation (`min_rounds` 50, `final_patience` 30), and only once neither
control's 5-round median has improved for 30 rounds. A run that still
improves on the synthetic panel therefore keeps training while its real
scores get worse. Every primary score keeps both best checkpoints, `median`
and `raw`, including `best_static_comb_{median,raw}.ckpt`.

## Runs

| run | model | stream | job | real_macro | static_comb |
|---|---|---|---|---|---|
| `rp_pyrseplstm_k64_combdiv` | `hppnet_pyrseplstm_k64` (as `rp_pyrseplstm_k64_all`) | `rp_combdiv` | `rp-pyrseplstm-k64-combdi-a48dbc` (vast A100-80, `b44b5d29`; W&B `al9s52jz`); finished 2026-10-23 17:48 UTC at the 100 000-step cap (200 rounds, 17.4 h) | 16.20 at the synthetic-selected step 97 000; 15.39 at the real-selected step 100 000 (5-round median 15.80) | 0.387 at step 97 000 |

At the synthetic-selected step 97 000, the real scores are: DREGON 20.08,
MD2 12.32; zero 0.73, standby 27.60, ramp 31.77, cruise 13.13 rev/s
(`results/rp_pyrseplstm_k64_combdiv/validation_history.jsonl`, identical to
W&B `al9s52jz`). `omnirun pull` timed out three times (ReadTimeout after about
60 s, 486 files / 469 MB), so the outputs were fetched straight from the
daemon's artifact bucket (`omnirun-artifacts/<job>/outputs/`).

The run was submitted twice by mistake. Both jobs write to the same R2 prefix,
so the R2 best-checkpoint aliases are whichever job last improved. The
duplicate `cf3649` lost its vast instance at round 137 (W&B `envtdy23`). omnirun
restarted it from step 0 (`resume: false`), and it was cancelled at 12:43 UTC on
2026-10-23, before that restart uploaded anything (newest R2 object 12:28).

## Results

**Interim panel re-score (2026-10-23, `scripts/rp_panel_eval.py` on uni-gpushort, job
`rp-panel-synth-262144`, `results/rp_panel_eval/synthetic.json`).** The rows are:

- three synthetic-only HPPNet-L2 arms of the earlier campaigns, at their
  `best_real_overall` checkpoints (frozen build `hppnet_l2_r2legacy`);
- the combdiv run at R2's `best_real_macro_median`, which is `a48dbc`'s step
  48 500 (raw `real_macro` 15.69 in its W&B history, the same number here).

All four checkpoints were selected on real recordings: the old arms on room 1 /
FLY124 under the pre-calibration labels, combdiv on this panel. Rev/s PIT MAE:

| run | real_macro | dregon | md2 | zero | standby | ramp | cruise |
|---|---:|---:|---:|---:|---:|---:|---:|
| `prior_v1_hppnet_l2` | 6.77 | 8.09 | 5.46 | 12.11 | 14.98 | 17.77 | 2.97 |
| `nv3r4_hard_legacytraj_hppnet_l2` | 4.25 | 5.22 | 3.27 | 1.33 | 6.31 | 15.17 | 3.24 |
| `nv3r4_hard_fullhyper_hppnet_l2` | 4.67 | 6.07 | 3.28 | 1.90 | 6.19 | 12.21 | 3.88 |
| `rp_pyrseplstm_k64_combdiv` | 15.69 | 18.99 | 12.38 | 1.30 | 29.07 | 30.12 | 12.11 |

Per dataset, zero / standby / ramp / cruise:

| run | dregon | md2 |
|---|---|---|
| `prior_v1_hppnet_l2` | 28.99 / 23.24 / 17.64 / 3.91 | 2.00 / 12.83 / 17.89 / 2.08 |
| `nv3r4_hard_legacytraj_hppnet_l2` | 0.21 / 15.60 / 20.12 / 3.68 | 2.00 / 3.89 / 10.53 / 2.82 |
| `nv3r4_hard_fullhyper_hppnet_l2` | 0.01 / 15.31 / 13.17 / 5.15 | 3.04 / 3.81 / 11.31 / 2.67 |
| `rp_pyrseplstm_k64_combdiv` | 0.11 / 14.48 / 33.80 / 20.51 | 2.02 / 32.87 / 26.67 / 4.13 |

On its own synthetic panel the combdiv run reaches `static_comb` ≈ 0.42
(5-round median). On real audio its cruise error is 12.1 against 3.0–3.9 for the
fitted-noise arms, and 20.5 on DREGON alone.

**Failure analysis and a panel defect** (explainer
`docs/explainers/synthetic-transfer-failures.qmd`; per-frame predictions from job
`rp-panel-dump-1878a4`).

- prior_v1 makes few octave errors (×2 ≤ 6 %, ×½ ≤ 9 % of rotor-frames). It
  fails in three ways:
  - it invents rotors under DREGON's white-noise loudspeaker;
  - it predicts "off" on MD2's audible 31–41 rev/s rotors;
  - it misses DREGON idle (15.3 rev/s), and so does nv3r4.
- **The DREGON labels in each recording's first clip are time-warped.** The
  telemetry starts 5.5–6.5 s after the audio. `mixing._valid_span` takes the
  Series' container bound instead of the first stamp, and `rps.npy` is
  untimed, so the loader's end-to-end stretch spreads the clip's last
  1.5–2.5 s of telemetry over all 8 s.
  - Idle shows 2.5–3 s early, over silent audio.
  - All 3160 DREGON "zero" frames have no telemetry.
  - Later clips are within ±0.09 s.
  - With labels at their recorded stamps, DREGON MAE becomes 5.83 for
    prior_v1 (was 8.09) and 5.17 for nv3r4 legacytraj (was 5.22).
  - The panel has not been re-derived.
- **The training streams are not stretched, but their DREGON idle is
  missing.** `online_mixing` interpolates the telemetry by time. The rp
  streams' DREGON sources are the five room-2 flights, which publish only
  `motors_command`.
  - The command has no idle. Room 1, which has both tracks, shows the
    tachometer at 15.3 rev/s while the command is 0 and then ramps up from 0.
  - In room 2, the motors are audible 1.8–2.1 s before the command reaches
    15 rev/s, or 1.2–1.5 s after removing the 0.6 s start-up seen in room 1.
    The rotors idle there, labelled 0 or 1–15 rev/s: ≈ 7 s of 244 s.
  - The 2.5–6.2 s of silence before each first stamp is sampled and labelled
    0, which is correct.
  - **Decision (Dmitrii, 2026-10-23): tolerated.** The streams stay as they
    are and no run is retrained: retraining every run on these streams costs
    too much for ≈ 3 % of the room-2 frames.

## The prior_v1 rigs in noise model v4 (2026-10-24)

The hard setup draws drones from the wide prior. Instead of resampling it, the
16 384 gated rigs of the old wide prior that `prior_v1_scv2` /
`prior_v1_hppnet_l2` trained on (`prior-v1-banks`, `prior_v1_n16384.npz`,
`static-rig-profiles.md` § "Generic drone prior") are rewritten as
`noise-v4-fit/3` payloads (Dmitrii, 2026-10-24; `noise-v4` merged into main
for this at `a7968caf`). Code: `experiments.noise_model.prior_v4.convert_prior_v1`,
bank `data_processing.noise_model.prior_bank_v4` (`noise-prior-bank-v4/1`,
lazy `PriorBankV4`, read by `kind: noise_v2` via `preset_bank`), builder
`scripts/prior_bank_v4_build.py` (77 s on 13 workers), report
`results/prior_rigs/prior_v4_n16384.report.json`.

Rules (`RULES_VERSION = prior_v4/1`), per rig:

| v4 part | from the v1 rig | decision |
|---|---|---|
| aero + motor lines | power-preserving split at `k = 3pn`: the motor line takes the boost's share of the coherent core, `C (1 - 10^{-b/10})`, without pedestal; a boost ≤ 0 leaves no motor line | the coherent core is exact; the boosted line loses its pedestal skirt |
| `alpha_aero` | `amp_exp` | — |
| `alpha_motor` | **redrawn** from N(0, 2) truncated to ±4 | Dmitrii: v4 physics (bench −2.1…+1.9) |
| pedestal | rotor mean of v1's σ² per order; `σ²_A = min(σ², c_g)`, the rest to the phase at the same rate | Dmitrii: same expected spectrum, no flicker above the v4 caps |
| shaft | `eps = σ_ν / (2π·80)`, `λ_ref = λ`, flat law | exact |
| floor | one floor on the mean speed, `b0 = floor_exp`, `b1 = 0` | v4 flight rule (v1 summed `s_r^b` per rotor) |
| mics | unit response; v1's 3.4 dB per-(mic, line) scatter becomes the pool option `line_scatter_db: 3.4`, drawn per clip outside the model | Dmitrii |
| wind | SC `sc` block drawn once per rig from v1's law (p 0.7; per mic `33u − Exp(8)` dB over the rig's 20–100 Hz floor at s̄ = 1; gust law offset 0, scale 2 m/s, 0.75 Hz) | Dmitrii: fixed windy capsules per rig |

Result: 89.5 % of rigs have live motor lines (3–6 orders); `alpha_motor`
sd 1.77; 70.3 % windy, 65.5 % of all capsules live; 72.8 % of (rig, order)
cells exceed the amplitude cap (96 % at k ≥ 41, mean phase σ² 0.40 there).
Checks: the coherent core is identical to v1's; with a rotor-shared pedestal
the expected spectrum at 80 rev/s matches v1 to < 0.05 dB away from motor
orders. On the real rigs the rotor-mean pedestal moves it by up to 0.58 dB
(median worst 0.23 dB), and within ±2 orders of a boosted motor line by up to
3 dB (the removed skirt). Two cells keep a phase σ² above v4's cap of 3 (3.25,
4.41, both k ≤ 8). The renderer is no slower than v1's (8 rigs × 4 s × 8
mics: 15.8 s against 17.5 s).

## Paper grid (2026-10-23): pending approval

Nothing below is submitted until Dmitrii approves the grid. These runs fill
the `\wip` cells of the pretraining table and the simplistic-synthetic
ablation in the rewritten Section VI of the paper. Recipe (Dmitrii,
2026-10-23):

- **Synthetic stream.** The real stream with the recordings replaced by
  renders. No label augmentation.
- **Stop and selection.** The stop works as in combdiv: both controls. Two
  checkpoints are reported, synthetic-selected and real-selected.
- **Fine-tuning.** Starts from the synthetic-selected checkpoint and uses
  `rp_all` without label augmentation. A fresh AdamW, max LR 1e-4, a short
  warm-up, and a cosine over 20 000 steps.
- **Reference.** The better of the model's two real-only runs.

Models: HRSPNet (`hppnet_pyrseplstm_k64`), HarmoF0-L2, Conv+BiGRU
(`hb_scv2_mag_nogate`), HPPNet-L2.

| paper cell | runs | stream | blocked on |
|---|---|---|---|
| pretraining table, easy | 4 (one per model) | drones drawn around the DREGON / MD2 fits, each flown on its own drone's trajectory law | (a), (b), (d) |
| pretraining table, hard | 4 | drones from the revised wide prior, flown on the constrained hyperprior (`hover_range` [35, 95], `rotor_sep` [2, 15], shrink 0.57) | (c), (d) |
| pretraining table, tuned | 8 | `rp_all` without label augmentation | the 8 runs above |
| ablation table, static comb, tuned | 1 (HRSPNet) | `rp_all` without label augmentation, from combdiv's synthetic-selected checkpoint (step 97 000) | nothing; can start on approval |
| ablation table, stochastic comb | 2 (HRSPNet: alone, tuned) | a stochastic-comb stream under this protocol | (e) |

Open decisions:

- **(a)** The perturbation of the easy setup. The η-tempered Laplace draw was
  explained; Dmitrii answered "not yet".
- **(b)** The DREGON flight stage of noise v4 fails the parity gate (best
  converged stage 15.8 rev/s against the bar 2.188;
  `noise-model-v4-bench.md`, round 6). Without it, the DREGON half of the
  easy setup has no fitted flight drone.
- **(c)** Resolved 2026-10-24: the hard setup's wide prior is the prior_v1
  bank converted to v4 (§ "The prior_v1 rigs in noise model v4"); the
  paper's wide-prior table still needs rewriting from those laws.
- **(d)** The synthetic validation set: "sectors" of the wide prior, among
  them the fit neighbourhoods and one sector close to the static comb. The
  split is undecided ("we need to think about it").
- **(e)** The stochastic comb under the rp protocol is not defined.

Cost, from combdiv: a synthetic-only run hit the 100 000-step cap after
17.4 h on a vast A100-80. Per draw, fitted-model renders cost more than combs
(v2: 1.97 s per 2 s × 8 mics; v4 not measured). A fine-tuning run is 20 000
steps, 1–3 GPU-h at the real runs' rate (HarmoF0-L2 1.4 h per 25 000 steps,
HRSPNet 4.1 h per 32 000). In total: 19 training runs, 8 of them
synthetic-only at up to ~17 h each.

The real-data runs which the rewrite needs are in `paper-real-protocol.md`,
§ "Paper grid after the experiments rewrite".

## Conclusion

Pending.
