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
| pedestal | rotor mean of v1's σ² per order; `σ²_A = min(σ², c_g)`, the rest to the phase at the same rate; `c_g` the bank's own caps `prior_v4.AMP_CAPS` (rules `prior_v4/2`, below) | Dmitrii: same expected spectrum, no flicker above the caps |
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

**Phase pedestal proposal (2026-10-25, under review).** The converted lines
are needles: v1 measured only the slow amplitude flicker (0.25 s frames,
0.2–1 Hz) and gave the phase to one quiet shaft, while the bench measures a
per-line phase pedestal once the shaft is removed (step A0). Skirt over floor
at the midpoints between lines, 80 rev/s: the DREGON flight fit stays under
the floor up to k = 60 (−21 to −3.6 dB) and goes above it only where its
bench data say nothing (k > 60, +1.9 dB median, pedestals at their caps);
the converted median rig sits 11–26 dB under. Proposed
(`prior_v4.with_phase_pedestal`, not in the published bank): σ²_φ and W
log-normal around the A0 blade-pass medians per order group (σ² 0.04 / 0.20 /
0.96 / 1.70 / 1.67, W 0.8 / 0.8 / 1.6 / 4.9 / 8.4 Hz for k 1–3 / 4–8 / 9–20 /
21–40 / 41+), caps σ² ≤ 3 and W ≤ 24 Hz, and a floor guard: all pedestal
skirts 3 dB under the wind-free floor at every midpoint at 80 rev/s (σ²_φ
shrinks; the analytic guard reads 1.6 dB under the exact periodogram, so it
is that lenient). On rig 13504 the guard shrinks 43 orders. Listening
comparison (v1 / converted / proposed on one trajectory):
`notebooks/prior_rigs_explainer.ipynb`, last section.

**Without the guard (2026-10-25, Dmitrii).** The guard cut the total pedestal
(amplitude + phase σ²) below the converted rig's on 13–72 of 115 orders on
the six comparison rigs (2–5 of them because the draw replaces the
converted phase part, the rest the guard). Without it, the total drops on
2–5 orders per rig, and the skirts over the wind-free floor at the midpoints
(80 rev/s, exact periodogram, median per k 1–8 / 9–20 / 21–40 / 41–60 /
61–98) are −10.1 / −12.2 / −6.6 / −2.5 / −2.3 dB on rig 13504 but reach
+6 / +14 / +18 dB above k = 20 on rig 2624 (whose converted skirts are
already +3 / +7 dB over the floor above k = 40). The notebook comparison now
shows the unguarded draw (`prior_rigs_lab.PED_GUARD`).

**The amplitude cap** (`v4_priors.AMP_GROUPS`, from the 0.25 s-frame
`amplitude_jitter.json`) sits under A0's demodulated amplitude measurement
(non-motor orders, unfiltered; p90 0.05 / 0.61 / 0.23 / 0.21 / 0.23 against
caps 0.075 / 0.29 / 0.24 / 0.20 / 0.12 for k 1–3 / 4–8 / 9–20 / 21–40 / 41+;
7–22 % of orders above the cap). The MD2 flight fit holds all 90 orders at
k ≥ 41 at the cap; the DREGON bench fit 8 of 108 there and 5 of 12 at
k 9–20. In the v1 bank the cap moves v1's σ² into the slow phase part on
95 % of orders at k ≥ 41 (that σ² is v1's k² σ_ψ² wobble term, meant at
25 Hz, but `drone_prior.PriorSpec.wobble_gamma_hz` is never used: v1 renders
it at its 1 Hz amplitude rate).

**Bank amplitude caps raised (2026-10-25, Dmitrii; rules `prior_v4/2`).**
`prior_v4.AMP_CAPS` = the larger of the fits' cap and the A0 p90: 0.075 /
0.61 / 0.24 / 0.21 / 0.23 for k 1–3 / 4–8 / 9–20 / 21–40 / 41+. The fits'
priors (`v4_priors.AMP_GROUPS`, used by the bench and flight fits and the
paper's amplitude table) are unchanged. The converted expected spectrum does
not move (the total σ² per order equals the published bank's to the float16
storage, ≤ 5e-4 relative); only the split does: v1's σ² exceeds the cap on
1 / 17 / 39 / 84 % of orders at k 4–8 / 9–20 / 21–40 / 41+ (was 4 / 17 / 41
/ 95 %), and the amplitude part at k > 40 goes from 0.12 to 0.23 (median of
five of the six comparison rigs). With the phase pedestal on top (no guard),
the total falls below the converted one on 1–5 of 115 orders per rig. The
published `prior-v4-banks@fd1e8ae9` is still `prior_v4/1`; it is rebuilt
once the phase-pedestal rule is fixed. The notebook comparison converts on
the fly under the current rules.

**Identifiability gate for v4 rigs (2026-10-25, Dmitrii).** The gate of
`experiments.noise_model.identifiability` now reads `noise-v4-fit/3` payloads
(same rule, frame pool and verdict; `L` and `B` from the v4 lag law and
`v4.broadband_periodogram`). Decisions: a rig is gated AFTER its phase
pedestal is drawn, and the gate IGNORES the rig's wind (`B` = floor + static
share + every other line). A motor line counts as part of its order's
harmonic. Checked against the exact wind-free `v4.expected_periodogram`
split into one harmonic and the rest: 0.004 dB worst over 195 lines of rig
13504 (`test_prior_v4.py` pins 0.05 dB). On 49 bank rigs (every 341st; the
v1 bank's own pool, seed 20261002), with the phase pedestal: 98.0 % pass the
bank's rule (3 dB, 2 orders, 80 %), 95.9 % the v3 rule (2 dB / 3 / 90 %);
±2–3 % at n = 49. Without the pedestal, 100 % pass. For reference, counting
the wind in `B` would drop the pass rate to 79.6 % / 73.5 %. Cost: ~26
core-s per rig (49 rigs in 105 s on 12 workers with a test run alongside),
so a full regate is ~120 core-h on `uni-cpu`.

**Wind per clip, not per rig (2026-10-25, Dmitrii; rules `prior_v4/3`, bank
format `noise-prior-bank-v4/2`).** Converted rigs carry no wind
(`params.wind` None). As in v1, wind is a render-time augmentation: the
`render_noise` / `kind: noise_v2` option `wind_sc` (v1's law, keys
`p_wind`, `max_over_floor_db`, `shield_mean_db`, `gustiness`; v1's bank uses
0.7 / 33 / 8 / 3) draws per clip none with probability `1 - p_wind`, else
one intensity `u` and per capsule `max_over_floor_db u - Exp(shield)` dB
over that capsule's rendered 20-100 Hz floor (v1's `_sc_wind`, reused). It is
refused for a payload with its own wind and for non-v4 payloads. Every clip
draws its own wind, so the audio is more diverse, and the gate question
above no longer depends on wind. The published `prior-v4-banks@fd1e8ae9`
(`/1`, per-rig wind) no longer loads; the rebuild with the phase pedestal
and the gate replaces it.

**The bank is redrawn from the prior, not converted (2026-10-25, Dmitrii;
rules `prior_v4/4`).** So that the paper can say every training rig is a draw
from the described prior, the bank no longer maps the 16 384 v1-bank rigs
(which had passed v1's own gate). `prior_v4.sample_prior_v4(seed, worker,
attempt)` draws a drone from the v1 laws (`drone_prior.sample_prior`,
`PriorSpec` defaults), maps it by the rules above (`v4_from_v1`), and draws its
phase pedestal (`PHASE_GROUPS`, no floor guard; it replaces the map's phase
part). `scripts/prior_bank_v4_build.py` gates every draw on one trajectory
pool (8 hyperprior windows of 8 s, 80 frames, stream `[20261024, 7919]`)
under the agreed rule `GateConfig()` (2 dB, 3 orders, 90 % of eligible
frames, at least 10 eligible; the rule the fits pass). A failing draw is
rejected and redrawn. Each bank rig records `(draw_worker, draw_attempt)`,
which gives it back exactly (`test_prior_v4.py`). The report gives every v1
law's scalar over all draws and over the kept ones.

Measured before the build (120 draws): 49 % pass the agreed rule, 70 % the v1
bank's rule (3 dB, 2, 80 %), and 64 % pass the agreed rule without the phase
pedestal. The failures are rotors short of 90 % of their eligible frames;
the pool's eligible counts are 46 / 37 / 48 / 27. A 12-rig smoke kept 12 of
28 draws. Over these few draws, the kept ones have a higher tail (−2.6 vs −4.5 dB),
a larger odd penalty at k = 3 (7.6 vs 4.7 dB) and a larger motor boost (7.7 vs 5.6 dB). The
caps are part of the laws: 20 % of the k 9–20 amplitude pedestals sit at
`AMP_CAPS`, and the phase σ² reaches its cap of 3 at the p95. Cost: ~2.5 s of
gate per draw on one thread, ~1.6 GB RSS per worker.

**Built and published (2026-10-25).** omnirun `prior-v4-bank-a14f5a`
(`uni-cpu`, 48 workers, 2794 s, code at `055f746c`) kept 16 384 of 31 257
draws (52.4 %). By first failing rotor, the rejections were rotor 1: 3877,
rotor 2: 9512, rotor 3: 195 and rotor 4: 1289, matching the pool's eligible
counts of 46 / 37 / 48 / 27 frames. The gate selects louder lines: the kept
draws have motor boost 6.25 vs 4.98 dB (mean over all draws), k = 2 over the
floor 22.9 vs 22.0 dB, tail −3.2 vs −4.0 dB, slope 24.3 vs 24.9 dB/decade.
The float16 round trip is within 0.031 dB on the tables and 5e-4 relative on
the pedestal. Published as `prior-v4-banks@2ee5131e`
(`prior_v4_n16384.npz`, 23.3 MiB, pinned); report
`results/prior_rigs/prior_v4_n16384.report.json`.

## Validation sectors (2026-10-26)

The priorv4 runs are scored on six synthetic sectors next to the real panel
and `static_comb` (`conf/validation/rp_real_priorv4.yaml`). Dmitrii decided
their content on 2026-10-25/26:

- the distance to a fit is perceptual;
- the axis sectors are weak comb, strong motor family and wide phase pedestal;
- the static-comb panel is reused;
- DREGON is the bench fit c7;
- MD2 is the per-regime pair r6c1 / r6c3;
- the neighbourhoods are drawn by prior-shaped perturbation.

Code: `experiments.noise_model.sectors_v4` and
`scripts/validation_sectors_v4_build.py`.

**Distance.** The distance between two rigs is computed as follows:

- take the mic-mean `v4.expected_periodogram` without wind (2048-point Hann
  at 16 kHz);
- evaluate it at 40, 60 and 80 Hz, with the four rotors at speed −3, −1, +1
  and +3 Hz;
- pass it through a 64-band mel filterbank (30 Hz–7.9 kHz) and take dB;
- remove the mean, so level does not count, because the stream normalises
  level;
- the distance is the RMS of the dB difference over bands and speeds.

A standby/cruise pair is power-blended by `regime_blend_weight`. The DREGON
and MD2 fits are 4.83 dB apart. Fresh prior draws give no neighbourhood:
the nearest of 1200 is 3.0 dB away and the 32nd nearest 4.0–4.1 dB, so the
neighbourhoods are perturbations of the fits.

**Prior-shaped perturbation** (`sectors_v4.perturb_fit`). The fit's own
parameters move by the wide prior's spreads times one strength *s*. Each rig
is one draw, shared by the two regimes of a pair. What moves:

- the comb against the floor: k2 (all orders), the decay slope, the
  blade-pass boost, the odd-order penalty and the motor boost;
- a level per rotor and a scatter per line;
- `alpha_aero`, `alpha_motor` (kept within ±4), the floor exponent `b0`, the
  static share and the 14 floor-shape weights;
- the amplitude σ² per order, and the phase σ² and half-width per order,
  within their caps.

The shaft, the mic response, the motor orders and the floor level stay the
fit's. The fit's wind is dropped; the stream adds `wind_sc` per clip, as for
the bank.

**Calibration.** For each fit, *s* is set so that the median distance to the
fit is 1/3 of the 4.83 dB gap (1.61 dB). With 32 draws per rung:

| strength | 0.1 | 0.2 | 0.35 | 0.5 | 0.75 | 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| DREGON median (dB) | 0.42 | 0.85 | 1.49 | 2.13 | 3.21 | 4.39 |
| MD2 median (dB) | 0.43 | 0.85 | 1.49 | 2.14 | 3.34 | 4.48 |

This gives *s* = 0.375 (DREGON) and 0.374 (MD2).

**Gate.** The identifiability gate now handles a shaft law that depends on
speed: each rotor's shaft is evaluated at its own frame speed, with one kernel
table per rotor and frame. It also handles a standby/cruise pair of v4
payloads, blended per frame. `test_prior_v4.py` checks both against the exact
expected periodogram (within 0.05 dB). On the bank's trajectory pool, both
fits pass the gate.

**Build.** omnirun `val-sectors-v4-36a4a4` (`uni-cpu`, 32 workers, 420 s,
code at `3dbdc5c0`):

- rig seed 20261026 (the training bank's is 20261024);
- the bank's gate rule and trajectory pool;
- 32 rigs per sector;
- the 1024-rig axis pool kept 1024 of 2049 gated prior draws;
- the DREGON neighbourhood kept 32 of 33 draws, the MD2 neighbourhood 32 of
  113 (rejected on rotors 1 and 3: 35 and 40 draws).

Published as `validation-sectors-v4@9dfde9ea` (pinned). Its `report.json`
holds the per-sector statistics. Medians, with p10–p90 in brackets:

| sector | rule | k2 over floor (dB) | motor boost (dB) | pedestal width (Hz) | to DREGON (dB) | to MD2 (dB) |
|---|---|---|---|---|---|---|
| axis pool (1024) | gated prior draws | 22.4 [14.0, 32.0] | 6.2 [1.7, 11.2] | 0.37 [0.04, 4.4] | 7.0 [4.7, 10.5] | 7.5 [4.9, 13.2] |
| `uniform` | 32 at random | 24.7 [17.7, 32.4] | 8.3 [3.7, 12.8] | 0.15 [0.03, 1.3] | 8.0 [5.0, 11.0] | 10.1 [5.9, 14.4] |
| `weak_comb` | lowest k2 | 6.3 [3.6, 8.6] | 5.4 [0.8, 11.3] | 3.2 [0.17, 7.3] | 7.8 [5.4, 9.3] | 7.3 [5.8, 10.7] |
| `strong_motor` | highest motor boost | 23.4 [16.4, 29.5] | 13.6 [13.1, 15.4] | 0.16 [0.03, 2.5] | 8.1 [5.1, 10.4] | 10.5 [5.7, 13.1] |
| `wide_pedestal` | widest lines | 14.6 [6.1, 25.9] | 5.8 [2.0, 8.5] | 7.3 [6.6, 9.9] | 9.8 [8.0, 12.1] | 7.8 [6.0, 9.6] |
| `near_dregon` | perturbed bench fit | – | – | – | 1.65 [1.04, 2.38] | 5.0 [4.7, 5.3] |
| `near_md2` | perturbed MD2 pair | – | – | – | 5.0 [4.4, 5.5] | 1.64 [1.16, 2.22] |

The pedestal width is the power-weighted (1 − e^{−σ²})·W of the aerodynamic
orders at 80 Hz. The phase pedestal is drawn independently per order, so no
single law coordinate measures it. Every neighbourhood rig is nearer its own
fit than the other fit: the largest own-fit distance is 3.6 / 3.0 dB, and the
smallest other-fit distance is 4.5 / 3.3 dB.

**Panels.** Each sector is rendered through
`conf/online_mix/rp_priorv4_sectors.yaml`:

- the training stream without its silence arm;
- `entry_order: cycle`, so 32 clips × 8 mics = 256 items render each rig once;
- 8 s clips, augmentations off, every clip its own flight;
- panel seeds 771101–771106;
- a neighbourhood rig flies its own drone's fitted trajectory law
  (`traj_rig` dregon / michaels).

The stop waits on `real_macro` and `uniform`. `uniform` alone selects the
synthetic checkpoint (`best_uniform_median`), which seeds the fine-tuning.

## Prior coverage map (2026-10-26)

**Question (Dmitrii).** Does the wide-prior bank cover the neighbourhood of the
MD2 fit at all? The `near_md2` scores and the synthetic-only models' real MD2
against real DREGON cruise suggest it does not.

**Distance** (`experiments.noise_model.rig_distance`, decided with Dmitrii).
The old sector distance (`sectors_v4.log_mel`) fixes which rotor gets which
speed offset, so relabelling a rig's rotors moved it by 0.85–2.0 dB (median
over 24 labellings; max 3.0 dB). That is the size of the neighbourhoods
themselves. The new distance is invariant by construction:

- every rotor at each of 15 reference speeds, 10–150 rev/s;
- the expected periodogram on a 4096-point Hann window at 16 kHz (3.9 Hz bins),
  mic-mean, 30 Hz – 7.9 kHz, no wind, split into each rotor's lines `L_r`
  (`v4.rotor_line_periodograms`) and the one broadband part `B`;
- harmonic feature per rotor: `10 log10(1 + L_r / B)` per bin and speed;
- broadband feature: `10 log10 B`, the rig's mean removed;
- harmonic distance: RMS over (speed, bin) of two rotors' feature difference,
  averaged over the best of the 24 rotor matchings (one matching for all
  speeds); broadband distance: RMS difference of the broadband features;
  total = their sum, in dB;
- a standby/cruise pair is power-blended per speed with the renderer's weight.

A relabelled rig is at exactly 0. The map is metric MDS (SMACOF from the
classical solution) of each distance, with Kruskal's stress-1.

**Run.** `scripts/rig_distance_map.py` (`37822beb`) on `uni-cpu`
(`rig-distance-map-159b00`): 2048 random rigs of the 16 384-rig bank (seed 0),
both fits and all 192 sector rigs. Features 79 s on 32 cores; the map is
`results/rig_distance_v4/map.png` (`--plot`), the numbers
`results/rig_distance_v4/summary.json`. Stress-1: total 0.16, harmonic 0.17,
broadband 0.09.

**Results** (dB, full distance, not the map; nearest / median over the 2048
bank rigs):

| | DREGON fit | MD2 fit |
|---|---|---|
| total | 35.5 / 42.2 | 16.2 / 21.8 |
| harmonic | 17.2 / 22.0 | 7.2 / 12.1 |
| broadband | 15.0 / 20.2 | 6.1 / 10.0 |
| own neighbourhood sector, total median | 4.5 | 3.7 |

Fit to fit: total 36.6. Two random bank rigs: median about 10.7 total.

- Both neighbourhoods lie outside the bank's cloud in all three maps; DREGON's
  about as far from the bank as from MD2, MD2's next to it.
- Restricted to 40–100 rev/s (the speeds the fits saw; 160 bank rigs), MD2's
  harmonic distance to the bank drops to 3.4 / 3.9 and its broadband to
  4.5 / 6.9: MD2's noise is covered where it flies. Its full-range distance
  comes from 110–150 rev/s (harmonic median 19.2), where its fit extrapolates.
  DREGON stays 15–22 dB away in every speed range.
- The axis sectors (weak comb, strong motor, wide pedestal) overlap `uniform`
  and each other: tails of the prior's coordinates are not distinct spectral
  regions under this distance.
- Why DREGON is far: its bench fit's lines stand 36–38 dB over its floor
  (rotor mean of `10 log10(1 + L/B)` over 30 Hz – 2 kHz at 80 rev/s), against
  1.3–3.9 for MD2 cruise and 0.6–6.9 for six bank rigs. `near_dregon` is
  therefore bench-clean audio, much easier than DREGON flight audio.
- The DREGON and MD2 trajectory laws are alike in an 8 s window: median
  closest-pair separation 1.8 vs 2.1 rev/s, share of time a pair is within
  1 rev/s 0.48 vs 0.31. MD2's law spends more windows at low speed (p10 of a
  window's mean speed 8 vs 16 rev/s; bank 27).

Open: why `near_md2` scores far worse than `uniform` when MD2's spectra are
typical of the bank. Diagnostic `sector-diag-5f321c` (`uni-gpushort`) scores
the four pretrained checkpoints (`best_uniform_median`) on the whole
`rp_real_priorv4` panel with per-frame dumps (`scripts/rp_panel_eval.py
--dump`).

## Paper grid (2026-10-23): hard setup approved 2026-10-26

Dmitrii approved the hard setup for the four models on 2026-10-26. The rest of
the grid waits for approval. These runs fill
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
| pretraining table, easy | 4 (one per model) | drones drawn around the DREGON / MD2 fits, each flown on its own drone's trajectory law | (a), (b) |
| pretraining table, hard | 4: `rp_{pyrseplstm_k64,harmof0_l2,scv2,hppnet_l2}_priorv4` | `rp_priorv4`: the wide prior in v4 (`prior-v4-banks@2ee5131e`), flown on the constrained hyperprior (`hover_range` [35, 95], `rotor_sep` [2, 15], shrink 0.57); validation `rp_real_priorv4` | approved 2026-10-26; submitted (§ "Hard setup runs") |
| pretraining table, tuned | 8: the hard four are `rp_*_priorv4_ft` (`checkpoint: best:uniform@rp_*_priorv4`) | `rp_all` without label augmentation | the hard four: queued after their pretraining (`--after`) |
| ablation table, static comb, tuned | 1 (HRSPNet) | `rp_all` without label augmentation, from combdiv's synthetic-selected checkpoint (step 97 000) | nothing; can start on approval |
| ablation table, stochastic comb | 2 (HRSPNet: alone, tuned) | a stochastic-comb stream under this protocol | (e) |

Open decisions:

- **(a)** The perturbation of the easy setup. The η-tempered Laplace draw was
  explained; Dmitrii answered "not yet".
- **(b)** The DREGON flight stage of noise v4 fails the parity gate (best
  converged stage 15.8 rev/s against the bar 2.188;
  `noise-model-v4-bench.md`, round 6). Without it, the DREGON half of the
  easy setup has no fitted flight drone.
- **(c)** Resolved 2026-10-25: the hard setup's wide prior is drawn fresh
  from the prior_v1 laws in v4 (rules `prior_v4/4`, `prior-v4-banks@2ee5131e`);
  the paper's wide-prior table still needs rewriting from those laws.
- **(d)** Resolved 2026-10-26: § "Validation sectors".
- **(e)** The stochastic comb under the rp protocol is not defined.

Cost, from combdiv: a synthetic-only run hit the 100 000-step cap after
17.4 h on a vast A100-80. Per draw, fitted-model renders cost more than combs
(v2: 1.97 s per 2 s × 8 mics; v4 prior rigs 0.026 s per 2 s draw at 40 reuses, prior_v1's 0.027, one thread, 2026-10-26). A fine-tuning run is 20 000
steps, 1–3 GPU-h at the real runs' rate (HarmoF0-L2 1.4 h per 25 000 steps,
HRSPNet 4.1 h per 32 000). In total: 19 training runs, 8 of them
synthetic-only at up to ~17 h each.

The real-data runs which the rewrite needs are in `paper-real-protocol.md`,
§ "Paper grid after the experiments rewrite".

### Hard setup runs (submitted 2026-10-26)

All eight run on `uni` at `0d839b3b`. The pretraining runs use 16 CPUs (12
data workers), 64 GB and a 48 h limit. HRSPNet and HPPNet-L2 run on an H100,
because `sae` rejects the A100-80 request ("Requested node configuration is
not available"). Each fine-tune runs with `--after` its pretraining, on the
same GPU type, with 8 h.

Smoke on `uni-gpushort` (`smoke-priorv4-8c23e2`, `smoke-priorv4-ft-cfa194`;
HarmoF0-L2, 120 + 60 steps, rounds of 20):

- the six sectors build at start, about 1 min each;
- one validation pass over all panels takes 11–13 s;
- every sector is scored each round;
- the pretraining stage uploads `best_uniform_median.ckpt` once the 5-round
  median window fills;
- `checkpoint: best:uniform@…` loads it with 0 missing and 0 unexpected keys.

| model | pretraining job | fine-tuning job, max LR 1e-4 | fine-tuning job, max LR 5e-4 |
|---|---|---|---|
| HRSPNet | `rp-pyrseplstm-k64-priorv-952a47` (H100) | `rp-pyrseplstm-k64-priorv-6d5279` | `rp-pyrseplstm-k64-priorv-a462b2` |
| HPPNet-L2 | `rp-hppnet-l2-priorv4-87756c` (H100) | `rp-hppnet-l2-priorv4-ft-0dc27e` | `rp-hppnet-l2-priorv4-ftl-d77f38` |
| HarmoF0-L2 | `rp-harmof0-l2-priorv4-e2322c` | `rp-harmof0-l2-priorv4-ft-969c0d` | `rp-harmof0-l2-priorv4-ft-bd45da` |
| Conv+BiGRU | `rp-scv2-priorv4-5a5364` | `rp-scv2-priorv4-ft-cd0055` | `rp-scv2-priorv4-ftlr5e4-d4da7a` |

The 5e-4 fine-tunes (`rp_*_priorv4_ft_lr5e4`, `ef74a51e`; Dmitrii,
2026-10-26) set the max LR to half the pretraining max (1e-3) instead of a
tenth. Everything else is the same as the 1e-4 fine-tunes. The HarmoF0-L2 and
Conv+BiGRU jobs were submitted directly; the HRSPNet and HPPNet-L2 jobs wait
`--after` their pretraining jobs, next to the 1e-4 ones.

**Results so far (2026-10-26, W&B).** Each score is the raw value at the
round with the lowest 5-round median of the selection score, as in
`paper-real-protocol.md`. All values are PIT-MAE in Hz on `val/real_macro`
unless the column says `uniform`.

| model | state | synth-selected step: `uniform` / `real_macro` | real-selected step: `real_macro` | hard, tuned (step of 20 000) | real-only reference |
|---|---|---|---|---|---|
| HRSPNet | running, step 60 000 | 50 500: 1.25 / 5.81 | 36 000: 5.57 | held | 2.52 |
| HPPNet-L2 | running, step 55 500 | 52 500: 2.61 / 7.98 | 40 000: 6.44 | held | 3.39 |
| HarmoF0-L2 | saturation stop at 68 500 | 53 500: 4.73 / 12.28 | 36 500: 9.84 | 2.66 (19 000) | 2.62 |
| Conv+BiGRU | saturation stop at 73 500 | 71 500: 3.28 / 9.89 | 45 000: 8.62 | 3.18 (3000) | 2.65 |

- The fine-tuning started from the loop's `best_uniform_median.ckpt`. For
  Conv+BiGRU, that checkpoint is from step 58 500, not 71 500: the median at
  71 500 is lower by only 0.001, which the loop does not count as an
  improvement.
- Neither finished fine-tune beats its real-only reference. HarmoF0-L2 ends
  level with it (2.66 against 2.62). Conv+BiGRU is worse (3.18 against 2.65);
  its best round comes at step 3000 and no later round improves on it.
- The fine-tunes are validated on the real split only, so they have no
  `uniform` score yet.
- The same numbers are in the paper's Table IX. The two running rows are
  marked `\prov` there.

## Conclusion

Pending.
