# Sim-to-real transfer on the FITTED noise model (noise-model-v2)

**Status:** in progress — configs and streams landed 2026-09-22, nothing
submitted. Ten arms: `nv2_{easy,hard,mixed}_{scv2,hppnet_l2}` and
`nv2_{easy,hard}_ft_{scv2,hppnet_l2}`. Per-experiment docs sit beside each
config in `conf/experiment/`.

## Motivation

The rig-sampler pair (`docs/experiments/rig-sampler-transfer-pair.md`) asked
whether a WIDER synthetic distribution transfers, and answered yes with a
caveat: widening helps (hard 5.37, easy 5.72 best real MAE against 9.97 for
two point presets), but both arms stayed 1.8x away from the real-trained
reference (`real_r4_scv2_unified`, `val/real_r3` **2.99**), and the whole
residual gap lived in the TRANSITIONS, where both synthetic arms collapsed the
four rotors onto nearly one speed (output spread 0.28 / 0.19 against 4.71 for
the real-trained model).

That pair sampled the STOCHASTIC family: a hand-written rig generator whose
parameters were fitted per rig. This batch replaces the family itself. The
noise-model-v2 campaign fitted the two rigs to a generative model with an
ABSOLUTE level — a shared integrated-OU shaft per rotor, one free Lorentzian
half-width per rotor and order, a coloured floor with its own speed envelope —
and the campaign also produced a fitted GENERATIVE MODEL OF TRAJECTORIES
(`docs/experiments/rps-trajectory-model.md`), so for the first time both the
comb and the flight it is flown on are measured objects rather than invented
ones.

The question this batch asks is therefore the same question with a better
family, plus one the previous pair could not ask at all:

1. **Does the fitted family transfer better than the hand-written one?**
   `nv2_easy_*` and `nv2_hard_*` against the legacy 5.72 / 5.37 and against
   the real references (`real_r4_scv2_unified` 2.99 for the regressor,
   `hppnet_l2_r2_s0` **2.27** for the salience trunk).
2. **Does pre-training on it help a real-data run?** The curriculum arms
   `nv2_{easy,hard}_ft_*` warm-start the family's real reference arm from the
   corresponding synthetic arm's best `real_overall` checkpoint. If the
   fitted family carries anything the real pool does not, the curriculum arm
   beats the real reference; if it does not, this is the cleanest possible
   negative result.
3. **Does it ADD to real data when mixed in?** `nv2_mixed_*` puts the hard
   family in the joint stream at exactly the ratio the stochastic family had
   in `hb_stochmixed_dload.yaml` (real 2.0 : silence 0.4 : synthetic 2.0).

Both model families run every arm, so the batch also separates "the noise
family transfers" from "this trunk transfers": the direct regressor (SCv2) and
the salience trunk (HPPNet-L2) read the same stream.

## Setup

### The streams

Three new policies, all of them a `kind: noise_v2` source over a preset bank of
fitted rigs:

| policy | bank | trajectory | `rps_scale_range` | structure |
|---|---|---|---|---|
| `conf/online_mix/noise_v2_easy_5050.yaml` | `noise_v2_easy_n2048.json` | per-entry `traj_rig` (dregon / michaels) | [0.45, 1.2] | v2 source 0.8 + silence 0.2, else `rig_easy_5050.yaml`'s |
| `conf/online_mix/noise_v2_hard_5050.yaml` | `noise_v2_hard_n2048.json` | `rigs: {posterior: 1}` (all entries `traj_rig: null`) | [1.0, 1.0] | as easy |
| `conf/online_mix/noise_v2_mixed_dload.yaml` | `noise_v2_hard_n2048.json` | `rigs: {posterior: 1}` | [1.0, 1.0] | real 2.0 : silence 0.4 : v2 2.0, else `hb_stochmixed_dload.yaml`'s |

Outside the noise source every policy is its legacy parent verbatim: sample
rate, clip length, base seed, `snr_ref_floor_rms: 0.02`, the silence arm, the
LibriSpeech source and the one-stage mixing policy (speech at −30..0 dB,
independent per-channel speech, 50 % gain/polarity; the mixed policy also
keeps the `freq_scale` noise augmentation and the noise time warp).

Five decisions are written into those files and are recorded here because they
are the arms' content, not their plumbing.

**One source, not two.** The legacy arms carried the same bank in TWO
stochastic sources purely for two independent render pools. A v2 render costs
1973 ms per 2 s x 8 mics against the legacy 416 ms
(`results/noise_v2/stream_bench/`), so a second pool buys nothing but render
cost. One source at weight 0.8 carries the combined weight and the
noise : silence ratio is unchanged at 0.8 : 0.2.

**`render_reuse: 192`, not 48.** At 48 the v2 loader is RENDER-BOUND: 12
workers deliver a batch of 128 about every 1.2 s, so the stream would be
starved rather than varied. 192 = 4 x 48 is the value at which these arms
produce FRESH RENDERS AT THE SAME WALL-CLOCK RATE the legacy arms had at 48,
since one v2 render costs about four legacy ones. This is a throughput
decision and it does change the distribution a model sees per unit of compute:
each rendered window is reused four times as often as in the legacy pair.

**The speed law is PINNED, not fitted.** Every bank entry carries the v2
short-span contract — `amp_exp` 2.0, `floor_exp` 2.0, `floor_static_rel`
2.5e-3, the model's prior medians, with `span_pin_record` set. The R5 fitted
exponents (−0.87 / 15.3) are UNIDENTIFIED on a cruise-only fit pool and must
not be flown: at `floor_exp` 15.3 a 32 rev/s window sits ~60 dB under the same
rig at cruise, which is not a drone. The cruise profile itself is untouched,
at its 80 rev/s reference.

**The level draw differs from the legacy pair, deliberately.**
`rig_easy_5050.yaml` names no level key and therefore ran the stochastic
pool's defaults, `normalize_rms: 0.1` with `level_mode: window` — every window
leaves at the same root-mean-square, so level carries no speed cue. All three
v2 policies use `normalize_rms_range: [0.02, 0.25]` with `level_mode: flight`
(the values `hb_stochmixed_dload.yaml` uses, the campaign's only measured
level-draw range): the drawn level is the level AT THE REFERENCE SPEED and the
window's own speed envelope is kept on top of it. `render_noise` is absolute
and its speed-level law is a fitted quantity; normalising it away per window
would discard the one thing the v2 fit measures that the stochastic family
guessed. The range brackets both rigs' fitted absolute levels (DREGON
~0.03–0.05, Michael's cruise ~0.1 at 8 mics).

**`rps_scale_range` is [0.45, 1.2] on easy and [1.0, 1.0] on hard and mixed.**
On the easy arm the multiplier is legacy AUGMENTATION around two known real
rigs and is kept for parity with the legacy pair's trajectory treatment. On
the hard and mixed arms the trajectories come from the rig hyperprior, which
already carries its own fitted hover level and ESC clamp; a multiplicative
0.45–1.2 on top of that would smear a rig-balanced speed distribution and
distort the very hyperprior the arm exists to fly. There is no hover
rescaling anywhere: the hyperprior trajectories keep their fitted RPS labels,
and `mean_shift: [-5, 5]` (as `traj_fitted_5050.yaml`) is the only level move.

### The trajectory cap `rps_max: 150`, and what it truncates

HPPNet-L2's salience grid is 0–150 rev/s (`conf/model/hppnet_l2.yaml`, loss
`salience_layers_r150`) and the rig posterior draws flights far above it. The
model is NOT changed; the stream is capped, by a new additive `rps_max` key on
the `rps` block (`src/data_processing/trajectory_model/source.py`). A flight in
which any rotor exceeds the cap anywhere is REJECTED AND REDRAWN whole — a
fresh drone for `posterior`, a fresh hover shift and realisation for a stored
rig — up to `MAX_RPS_REDRAWS` = 64 attempts, after which the source raises
rather than returning an out-of-range flight. Nothing is clipped: a clipped
trajectory is a flat-topped flight no rig flies. The source counts accepted
and rejected flights in `stats`, and the accepted drone's `RigDraw.redraws`
records how many attempts it cost. Both model families fly the same capped
population, so the arms stay comparable across trunks.

`rps_max: 150` is written in all three policies for uniformity. **On the easy
arm it never fires**: over 128 flights per rig at `mean_shift: [-5, 5]` the
peak rotor speed is median 92.3 / max 97.4 rev/s (dregon) and median 93.9 /
max 104.8 (michaels), 0 rejections.

**On the hard and mixed arms it is a real truncation of the hyperprior, and
this batch says so plainly.** Measured on `dload:rps-traj-fits` with
`mean_shift: [-5, 5]`:

* **acceptance 0.296** — 512 accepted flights cost 1217 rejections (an
  independent 256-flight run gave 0.309). Uncapped, the per-flight peak rotor
  speed is median 190, p90 396, max 615 rev/s, and 66 % of flights exceed 150.
  The arms therefore train on the hyperprior CONDITIONED on max rotor speed
  ≤ 150, not on the hyperprior.
* **The cap truncates hover level, not rig diversity.** The posterior is a
  diagonal Gaussian in 32 scale-free coordinates of which coordinate 0 is the
  log hover scale. Comparing the 512 accepted draws against all 1729 draws,
  coordinate 0 moves hard — z = −33.2, mean 4.971 → 4.406, std 0.497 → 0.275 —
  while **all 31 shape and dynamics coordinates are statistically unmoved**:
  max |z| = 1.71 (coordinate 14), nothing above |z| = 2 let alone 3, and the
  accepted/all standard-deviation ratio is 0.951–1.031 with median 0.991. The
  rig balance the hyperprior exists to provide survives the cap.
* **Surviving hover** (drone mean `mu`): min 35.1, median 86.1, p90 111.9, max
  182.0 rev/s, against all-draws min 35.1, median 144.2, p90 274.5, max 613.6.
  Drones above 150 survive because the per-flight level offset can sit a
  flight well below its drone's hover: acceptance is per FLIGHT, not per
  drone.

The redraw budget is 64 and not 32 for one measured reason: at 0.296
acceptance, 33 attempts fail once per 2.1e5 flights — about one job in seven
over a 31k-flight run — while 65 attempts fail once per 4.7e9.

### The banks

`data/rig_banks/noise_v2_easy_n2048.json` and `noise_v2_hard_n2048.json`,
2048 entries each, format `noise-v2-bank/1`, built by
`python scripts/noise_v2_build_bank.py --preset easy|hard` (seed 20260921,
bit-reproducible, idempotent — a rebuild whose provenance digest matches is
skipped). They are gitignored BUILD PRODUCTS and `omnirun` ships a clean
pushed checkout, so every job rebuilds its bank before `train.py`;
`scripts/noise_v2_submit_arms.sh` does exactly that.

* **Canonical regime pair.** Michael's = {standby, cruise} as fitted; DREGON =
  {standby: null, cruise: R5 `dregon_room2_floor__flight_profile.json`}.
* **Easy**: neighbourhood draws around each canonical rig, each keeping its own
  K (88 DREGON, 81 Michael) and its own `traj_rig`, so a DREGON comb is never
  flown on Michael's envelope.
* **Hard**: the path interpolates CRUISE ↔ CRUISE only, on the common order
  range K = 1..81 (DREGON's orders 82–88 are dropped for the path bank and the
  drop is recorded in the provenance). Regime policy is CARRIED, not
  interpolated: at mixing coordinate *t* the standby slot is Michael's standby
  fit with probability *t* and null with probability 1 − *t*, so a path rig is
  always a coherent regime pair or a coherent single-regime rig.

### The arms

Every arm keeps `override /validation: rps_unified` VERBATIM and is selected on
**`real_overall`**, not `overall_macro`: on the fitted-preset run the synthetic
half of the macro improved monotonically while every real view degraded, so the
macro neither stopped the run nor reduced the LR.

| arm | parent config | the one thing it changes | stream |
|---|---|---|---|
| `nv2_easy_scv2` | `rig_easy_scv2_unified` | train stream | `noise_v2_easy_5050` |
| `nv2_hard_scv2` | `rig_easy_scv2_unified` | train stream | `noise_v2_hard_5050` |
| `nv2_mixed_scv2` | `rig_easy_scv2_unified` | train stream | `noise_v2_mixed_dload` |
| `nv2_easy_ft_scv2` | `real_r4_scv2_unified` | warm start + patience/lr | `hb_m3s2_dload` (real) |
| `nv2_hard_ft_scv2` | `real_r4_scv2_unified` | warm start + patience/lr | `hb_m3s2_dload` (real) |
| `nv2_easy_hppnet_l2` | `rig_easy_hppnet_l2_unified` | train stream | `noise_v2_easy_5050` |
| `nv2_hard_hppnet_l2` | `rig_easy_hppnet_l2_unified` | train stream | `noise_v2_hard_5050` |
| `nv2_mixed_hppnet_l2` | `rig_easy_hppnet_l2_unified` | train stream | `noise_v2_mixed_dload` |
| `nv2_easy_ft_hppnet_l2` | `hppnet_l2_r2_s0` | `rps_unified` + unified batch/clip + warm start | `hb_silence_dload` (real) |
| `nv2_hard_ft_hppnet_l2` | `hppnet_l2_r2_s0` | `rps_unified` + unified batch/clip + warm start | `hb_silence_dload` (real) |

The SCv2 arms therefore inherit `real_r1_scv2` + `rps_unified`, bfloat16, batch
128, 12 workers, 2 s clips, `samples_per_validation: null`; the HPPNet arms
inherit the L2 trunk with `salience_layers_r150`, batch 128, 6 workers, 2 s
clips, validation batch 64, lr 1e-3, patience 20. The curriculum arms take
`checkpoint: best:real_overall@<synthetic arm>`, which `training.loop`
resolves to
`r2://<bucket>/<prefix>/<arm>/checkpoints/best_real_overall.ckpt` and loads
`strict=False` with fresh optimizer / scheduler / early-stopping state; both
carry `patience: 20` and `lr: 1e-3`.

Reference rows, unchanged by this batch:

| row | role |
|---|---|
| `real_r4_scv2_unified` | regressor on real noise, `val/real_r3` **2.99** |
| `hppnet_l2_r2_s0` | salience trunk on real noise, **2.27** |
| `rig_easy_scv2_unified` / `rig_hard_scv2_unified` | the legacy stochastic pair, **5.72** / **5.37** |

### Submission

`scripts/noise_v2_submit_arms.sh <arm|all-synth|all-ft|mixed>` prints (with
`--dry-run`) or runs one `omnirun` submission per arm on backend `uni`, each in
a detached worktree at a pinned SHA, each rebuilding its bank in-job before
`train.py`. The curriculum arms must not be submitted before their stage-1 arm
has finished and uploaded its `best_real_overall` checkpoint.

## What will be measured

* Best **`real_overall`** and the per-view `real_r1` / `real_r2` / `real_r3`
  rev/s MAE at that epoch for every arm, with the epoch it occurred at,
  against the family's real reference (2.99 SCv2, 2.27 HPPNet-L2) and against
  the legacy pair (5.72 / 5.37).
* The **four-regime decomposition** of each arm's best checkpoint,
  `python scripts/_regime_decomp.py --exp <arm> --ckpt best` (ramp / zero /
  standby / cruise, thresholds stated not fitted). This is the measurement the
  legacy pair's conclusion turns on: both legacy synthetic arms were close to
  real at cruise and collapsed the four rotors onto one speed in the
  transitions (output spread 0.28 / 0.19 against 4.71). Whether the fitted
  family, with fitted trajectories under a stated cap, moves the RAMP cell is
  the question this batch exists to answer.
* Whether the post-best real degradation of the point-preset run recurs, and
  the `r1`/`r2` ratio against that run's 1.93.
* For the curriculum arms: the delta against the real reference trained from
  scratch, which is the only number that says whether the fitted family is
  worth carrying.

## Results

**PENDING** — nothing submitted as of 2026-09-22. Configs, streams, the
`rps_max` cap and the submit script have landed on `main`; the banks build
in-job.

## Conclusion

**PENDING.**
