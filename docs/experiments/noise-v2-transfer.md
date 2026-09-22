# Sim-to-real transfer on the FITTED noise model (noise-model-v2)

**Status:** DONE — all ten arms ran to their own early stop on backend `uni`
between 2026-09-22 02:00Z and 17:35Z; none failed and none was killed. Ten
arms: `nv2_{easy,hard,mixed}_{scv2,hppnet_l2}` and
`nv2_{easy,hard}_ft_{scv2,hppnet_l2}`. Per-experiment docs sit beside each
config in `conf/experiment/`. Headline: the fitted family does NOT transfer
better than the hand-written one it replaces (7.94 / 7.06 against 5.72 / 5.37)
but it DOES fix the rotor collapse it was built to fix, and used as a warm
start or a mixing component it beats the real-data references (2.23 against
2.99; 2.11 against 2.27).

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
stochastic sources, which gives two independent render pools over ONE
distribution. That is a cache-topology choice and not a throughput one: each
source refreshes once per `render_reuse` of its own draws, so two sources at
half the weight render no more often in steady state than one at the full
weight. What two pools cost is two warm-ups before the stream saturates and
two resident renders, which at 1973 ms per 2 s x 8 mics against the legacy
416 ms (`results/noise_v2/stream_bench/`) is worth avoiding; what they buy is
two windows to alternate between rather than one. One source at weight 0.8
carries the combined weight and the noise : silence ratio is unchanged at
0.8 : 0.2. The mixed policy's two replaced sources were two DIFFERENT
families (a stochastic rig and a static comb), which one v2 bank subsumes.

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
This knob IS a multiplicative rescaling of the drawn flight's speeds, applied
per window. On the easy arm it is the LEGACY EASY AUGMENTATION, kept verbatim
so the trajectory treatment is at parity with the arm this one is read
against, and defensible there because the rigs it augments around are the two
real ones. On the hard and mixed arms the trajectories come from the rig
hyperprior, which already carries its own fitted hover level and ESC clamp; a
multiplicative 0.45–1.2 on top would smear a rig-balanced speed distribution
and distort the hyperprior the arm exists to fly, so those arms drop it to
the identity. `mean_shift: [-5, 5]` (as `traj_fitted_5050.yaml`) is then the
only level move the hard and mixed streams make.

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

### The second cap: `freq_scale.rps_max`, on the LABEL

Capping the trajectory is not enough. The mixed policy's
`noise_augmentations` block fires `freq_scale` with probability 1.0, and that
transform MULTIPLIES the rotor-speed labels by its own `alpha`
(`src/data_processing/noise_augmentations.py`): a flight capped at 150 rev/s
comes back out at 195 at `alpha_high` 1.3, straight off the grid the first cap
exists to respect. The source-level cap cannot see this, because it runs
before the augmentation.

So `freq_scale` takes an additive `rps_max` key of its own (default `None` =
the historical behaviour, every other policy unchanged). When set, the
upper bound of the alpha draw becomes
`min(alpha_high, rps_max / max(rps in this frame))`; the lower bound is
untouched, and a frame that cannot take even `alpha_low` is passed through at
`alpha = 1` rather than silently rescaled. The draw is taken in both branches,
so the random stream does not depend on the frame's labels. The mixed policy
sets `rps_max: 150`; the easy and hard policies inherit the legacy rig pair's
policy block, which has NO `noise_augmentations` at all, so there is nothing
to cap there.

**Verified on YIELDED frames, not on the source** (256 chunk-level samples per
stream, `render_reuse` and `flight_reuse` lowered to 2 for the measurement so
the sample covers many distinct flights rather than one):

| stream | bank | max label | p99 | median | frames > 150 |
|---|---|---:|---:|---:|---:|
| `noise_v2_hard_5050` (no augmentation block) | production 2048 | 144.17 | 143.96 | 87.07 | 0 |
| `noise_v2_mixed_dload` (freq_scale p = 1.0, capped) | production 2048 | 149.20 | 142.45 | 78.64 | 0 |
| `noise_v2_hard_5050` | 2-entry stub | 142.27 | 141.86 | 80.01 | 0 |
| `noise_v2_mixed_dload` | 2-entry stub | 147.50 | 145.86 | 86.39 | 0 |
| `noise_v2_mixed_dload`, CONTROL with the key removed | 2-entry stub | **175.29** | 165.42 | — | **17** |

The control is the point: with the source cap alone, 17 of 256 mixed chunks
carry labels above the salience grid and the worst is 175.29 rev/s. With the
label cap, none do. Real-source frames in the mixed stream are untouched by
the bound — their own peaks sit near 90 rev/s, so
`min(1.3, 150 / peak)` = 1.3 and the draw is the unmodified one.

### The banks

`noise_v2_easy_n2048.json` (29.8 MB, sha256 `cea6c5ad56b1337d…`, content
digest `16ad7d6797aec704…`) and `noise_v2_hard_n2048.json` (29.3 MB, sha256
`b5ff345ecd2b8289…`, content digest `b43eacd34b993d70…`), 2048 entries
each, format `noise-v2-bank/1`, seed 20260921, built by
`python scripts/noise_v2_build_bank.py --preset easy|hard` (bit-reproducible;
idempotent — a rebuild whose content digest matches is skipped). The hard
bank's realised mixing coordinate is uniform (mean *t* 0.501, KS 0.0166
against a 0.0300 threshold, deciles 233/194/186/214/196/197/191/213/192/232)
and 50.2 % of its entries carry the standby slot.

**Sampler strength 3.0**, chosen by the coverage ladder and not by taste: at
strength 2.0 — the legacy pair's setting — the hard cloud brackets only 89.2 %
of DREGON's 1/3-octave bands above 300 Hz against a 90 % target, while at 3.0
it reaches 93.1 % (easy 91.5 / 94.2 %) on the 64-draw probe; at full size the
realised coverage is easy DREGON 100.0 % / Michael's 99.0 % and hard DREGON
98.5 % / Michael's 100.0 % of bands above 300 Hz. The widths ladder, the guard rejection
statistics and the coverage table are in
`results/noise_v2/rig_sampler/findings.md`, and each bank ships a sidecar
build report (`build_easy.json`, `build_hard.json`: acceptance, guards fired,
coverage, self-check, wall time, digest) plus `structure.json` (the measured
widths and the real-window band levels).

**Transport.** A 2048-entry build takes about 45 minutes, so the banks are
NOT rebuilt per job: they are published once as the dload dataset
`noise-v2-banks` and each policy names the file inside it WITH ITS VERSION —
`preset_bank: dload:noise-v2-banks@7f6a3242b353…/noise_v2_easy_n2048.json`. The
same pin sits in the committed `dload.lock`, so the arm resolves one exact
bank from either side and a later republish cannot silently move it. The
banks were in fact republished once (version `7f6a3242b353…`, `dload.lock`
line 55, commit `de02fe10`) so that the published digest matches what the
COMMITTED builder source computes; the draws are byte-for-byte the previous
build's (identical acceptance 2.017 / 1.778 attempts per entry, identical
guard counts, identical realised *t* and KS), only the provenance source
hashes moved, so every number in this section still describes the shipped
bytes.
`load_preset_bank` routes its path through
`data_processing.streams.resolve_source`, the same convention `rps.fits` uses,
so a pinned dataset resolves to a local file and a plain path still works
unchanged; the pin sits in the policy AND in the lock, so a job cannot
silently train on a different bank. `scripts/noise_v2_submit_arms.sh` runs
`dload pull noise-v2-banks` once before `train.py`, so the fetch happens up
front and never inside a DataLoader worker. Verified on this machine:
`load_preset_bank('dload:noise-v2-banks@7f6a3242b353…/noise_v2_hard_n2048.json')`
returns 2048 entries (first `path_s3_00000`, `traj_rig` null, standby present).

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

### Submission and preflight

`scripts/noise_v2_submit_arms.sh <arm|all-synth|all-ft|mixed>` prints (with
`--dry-run`) or runs one `omnirun` submission per arm on backend `uni`, each
in a detached worktree at a pinned SHA, each pulling the pinned bank dataset
before `train.py`. The curriculum arms must not be submitted before their
stage-1 arm has finished and uploaded its `best_real_overall` checkpoint;
`--after <job id>` lets the scheduler enforce that.

**Preflight** (`conf/AGENTS.md`: `python train.py experiment=<name>
validate_only=true` before any GPU job), run on the laptop under
`systemd-run --user --scope -p MemoryMax=10G` against the PINNED 2048-entry
banks the arms actually name (an earlier pass against 64-entry stub banks gave
the same result):

| arm | `validate_only` | stage-1 job |
|---|---|---|
| `nv2_easy_scv2` | PASS | `nv2-easy-scv2-15cff2` |
| `nv2_hard_scv2` | PASS | `nv2-hard-scv2-d926d0` |
| `nv2_mixed_scv2` | PASS | `nv2-mixed-scv2-83d2cd` |
| `nv2_easy_ft_scv2` | PASS | `nv2-easy-ft-scv2-9ff3c2` (chained `--after` `nv2-easy-scv2-15cff2`) |
| `nv2_hard_ft_scv2` | PASS | `nv2-hard-ft-scv2-56dbff` (chained `--after` `nv2-hard-scv2-d926d0`) |
| `nv2_easy_hppnet_l2` | PASS | `nv2-easy-hppnet-l2-9a2ecf` |
| `nv2_hard_hppnet_l2` | PASS | `nv2-hard-hppnet-l2-f71237` |
| `nv2_mixed_hppnet_l2` | PASS | `nv2-mixed-hppnet-l2-08c04f` |
| `nv2_easy_ft_hppnet_l2` | PASS | `nv2-easy-ft-hppnet-l2-1a711c` (chained `--after` `nv2-easy-hppnet-l2-9a2ecf`) |
| `nv2_hard_ft_hppnet_l2` | PASS | `nv2-hard-ft-hppnet-l2-6fe90f` (chained `--after` `nv2-hard-hppnet-l2-f71237`) |

The six stage-1 jobs were submitted on **2026-09-22 02:00Z** (`all-synth`) and
**02:01Z** (`mixed`), each from its own detached worktree pinned to
**`d2586b67`** on backend `uni`, 1 GPU, 24 h. The four `_ft_` arms are NOT
submitted: they resolve `best:real_overall@<stage 1>` and cannot run until
their stage-1 arm has uploaded that checkpoint.

One caveat worth stating: `validate_only` checks the spec and runs one CPU
batch; it does NOT resolve the curriculum arms' `best:real_overall@<stage 1>`
warm start, so a green row there does not prove the checkpoint exists. It
will not until stage 1 has run.

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

All ten arms finished. The six stage-1 jobs were submitted 2026-09-22
02:00Z/02:01Z; each curriculum arm was submitted with `--after` its stage-1 job
once that job's `checkpoints/best_real_overall.ckpt` was confirmed present on
R2.

### How these numbers are read

`real_overall` is the aggregate `{real_r3: 1.0}` (`conf/validation/rps_unified.yaml`),
so it IS `val/real_r3` and is directly comparable to the reference rows' 2.99 /
2.27 and to the legacy pair's 5.72 / 5.37.

Selection is on the SMOOTHED (5-round median) `real_overall`:
`best_real_overall.ckpt` — the object the curriculum arms warm-start from — is
written at the round where that median last improved, which is NOT in general
the round with the lowest raw value. Every row therefore quotes the selected
round, its smoothed value, its raw value and the per-view `real_r1`/`r2`/`r3`
AT THAT ROUND, plus the best raw value anywhere in the run.

The live source is R2: `training.loop` uploads
`artifacts/<exp>/checkpoints/validation_history.jsonl` and
`best_checkpoints.json` after every validation round, so the monitoring rows
below are the job's own records and not a stdout scrape.

### Monitoring log

| time (UTC) | job / arm | status | round | sel round | smoothed | raw @ sel | best raw | lr |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 09-22 03:52 | `nv2-easy-scv2-15cff2` | running | 59 | 53 | 8.05 | 7.69 | 7.18 | 1e-03 |
| 09-22 03:52 | `nv2-hard-scv2-d926d0` | running | 63 | 15 | 9.17 | 7.06 | 7.06 | 3e-04 |
| 09-22 03:52 | `nv2-easy-hppnet-l2-9a2ecf` | running | 26 | 20 | 7.23 | 6.46 | 5.52 | 1e-03 |
| 09-22 03:52 | `nv2-hard-hppnet-l2-f71237` | running | 26 | 24 | 12.03 | 12.03 | 9.89 | 1e-03 |
| 09-22 03:52 | `nv2-mixed-scv2-83d2cd` | queued | — | — | — | — | — | — |
| 09-22 03:52 | `nv2-mixed-hppnet-l2-08c04f` | queued | — | — | — | — | — | — |
| 09-22 03:55 | `nv2-easy-scv2-15cff2` | running | 62 | 53 | 8.05 | 7.69 | 7.18 | 5e-04 |
| 09-22 03:55 | `nv2-hard-scv2-d926d0` | running | 66 | 15 | 9.17 | 7.06 | 7.06 | 3e-04 |
| 09-22 03:55 | `nv2-easy-hppnet-l2-9a2ecf` | running | 27 | 20 | 7.23 | 6.46 | 5.52 | 1e-03 |
| 09-22 03:55 | `nv2-hard-hppnet-l2-f71237` | running | 27 | 24 | 12.03 | 12.03 | 9.89 | 1e-03 |
| 09-22 03:55 | `nv2-mixed-scv2-83d2cd` | queued | — | — | — | — | — | — |
| 09-22 03:55 | `nv2-mixed-hppnet-l2-08c04f` | queued | — | — | — | — | — | — |
| 09-22 04:20 | `nv2-easy-scv2-15cff2` | running | 76 | 53 | 8.05 | 7.69 | 7.18 | 3e-04 |
| 09-22 04:20 | `nv2-hard-scv2-d926d0` | running | 81 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 04:20 | `nv2-easy-hppnet-l2-9a2ecf` | running | 33 | 20 | 7.23 | 6.46 | 5.52 | 5e-04 |
| 09-22 04:20 | `nv2-hard-hppnet-l2-f71237` | running | 33 | 24 | 12.03 | 12.03 | 9.89 | 1e-03 |
| 09-22 04:20 | `nv2-mixed-scv2-83d2cd` | queued | — | — | — | — | — | — |
| 09-22 04:20 | `nv2-mixed-hppnet-l2-08c04f` | queued | — | — | — | — | — | — |
| 09-22 04:45 | `nv2-easy-scv2-15cff2` | running | 90 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 04:45 | `nv2-hard-scv2-d926d0` | running | 96 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 04:45 | `nv2-easy-hppnet-l2-9a2ecf` | running | 39 | 20 | 7.23 | 6.46 | 5.52 | 5e-04 |
| 09-22 04:45 | `nv2-hard-hppnet-l2-f71237` | running | 39 | 39 | 10.86 | 10.14 | 9.89 | 1e-03 |
| 09-22 04:45 | `nv2-mixed-scv2-83d2cd` | queued | — | — | — | — | — | — |
| 09-22 04:45 | `nv2-mixed-hppnet-l2-08c04f` | queued | — | — | — | — | — | — |
| 09-22 05:10 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 05:10 | `nv2-hard-scv2-d926d0` | running | 111 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 05:10 | `nv2-easy-hppnet-l2-9a2ecf` | running | 46 | 20 | 7.23 | 6.46 | 5.52 | 3e-04 |
| 09-22 05:10 | `nv2-hard-hppnet-l2-f71237` | running | 46 | 42 | 10.14 | 9.95 | 9.20 | 1e-03 |
| 09-22 05:10 | `nv2-mixed-scv2-83d2cd` | running | 1 | — | — | — | 4.49 | 1e-03 |
| 09-22 05:10 | `nv2-mixed-hppnet-l2-08c04f` | queued | — | — | — | — | — | — |
| 09-22 05:16 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 05:16 | `nv2-hard-scv2-d926d0` | running | 114 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 05:16 | `nv2-easy-hppnet-l2-9a2ecf` | running | 47 | 20 | 7.23 | 6.46 | 5.52 | 3e-04 |
| 09-22 05:16 | `nv2-hard-hppnet-l2-f71237` | running | 47 | 42 | 10.14 | 9.95 | 9.20 | 1e-03 |
| 09-22 05:16 | `nv2-mixed-scv2-83d2cd` | running | 4 | 4 | 5.68 | 5.67 | 4.49 | 1e-03 |
| 09-22 05:16 | `nv2-mixed-hppnet-l2-08c04f` | queued | — | — | — | — | — | — |
| 09-22 05:16 | `nv2-easy-ft-scv2-9ff3c2` | queued | — | — | — | — | — | — |
| 09-22 05:42 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 05:42 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 05:42 | `nv2-easy-hppnet-l2-9a2ecf` | running | 54 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 05:42 | `nv2-hard-hppnet-l2-f71237` | running | 53 | 42 | 10.14 | 9.95 | 9.20 | 1e-03 |
| 09-22 05:42 | `nv2-mixed-scv2-83d2cd` | running | 20 | 20 | 3.59 | 3.59 | 2.55 | 1e-03 |
| 09-22 05:42 | `nv2-mixed-hppnet-l2-08c04f` | running | 6 | 5 | 4.02 | 3.91 | 3.61 | 1e-03 |
| 09-22 05:42 | `nv2-easy-ft-scv2-9ff3c2` | queued | — | — | — | — | — | — |
| 09-22 05:43 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 05:43 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 05:43 | `nv2-easy-hppnet-l2-9a2ecf` | running | 54 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 05:43 | `nv2-hard-hppnet-l2-f71237` | running | 54 | 42 | 10.14 | 9.95 | 9.20 | 1e-03 |
| 09-22 05:43 | `nv2-mixed-scv2-83d2cd` | running | 21 | 20 | 3.59 | 3.59 | 2.55 | 1e-03 |
| 09-22 05:43 | `nv2-mixed-hppnet-l2-08c04f` | running | 6 | 5 | 4.02 | 3.91 | 3.61 | 1e-03 |
| 09-22 05:43 | `nv2-easy-ft-scv2-9ff3c2` | queued | — | — | — | — | — | — |
| 09-22 05:43 | `nv2-hard-ft-scv2-56dbff` | queued | — | — | — | — | — | — |
| 09-22 06:09 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 06:09 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 06:09 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 06:09 | `nv2-hard-hppnet-l2-f71237` | running | 60 | 55 | 9.77 | 9.77 | 8.17 | 1e-03 |
| 09-22 06:09 | `nv2-mixed-scv2-83d2cd` | running | 37 | 36 | 2.99 | 2.98 | 2.51 | 1e-03 |
| 09-22 06:09 | `nv2-mixed-hppnet-l2-08c04f` | running | 14 | 12 | 2.63 | 2.58 | 2.58 | 1e-03 |
| 09-22 06:09 | `nv2-easy-ft-scv2-9ff3c2` | running | — | — | — | — | — | — |
| 09-22 06:09 | `nv2-hard-ft-scv2-56dbff` | queued | — | — | — | — | — | — |
| 09-22 06:10 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 06:10 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 06:10 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 06:10 | `nv2-hard-hppnet-l2-f71237` | running | 60 | 55 | 9.77 | 9.77 | 8.17 | 1e-03 |
| 09-22 06:10 | `nv2-mixed-scv2-83d2cd` | running | 37 | 36 | 2.99 | 2.98 | 2.51 | 1e-03 |
| 09-22 06:10 | `nv2-mixed-hppnet-l2-08c04f` | running | 14 | 12 | 2.63 | 2.58 | 2.58 | 1e-03 |
| 09-22 06:10 | `nv2-easy-ft-scv2-9ff3c2` | running | 0 | — | — | — | 3.32 | — |
| 09-22 06:10 | `nv2-hard-ft-scv2-56dbff` | queued | — | — | — | — | — | — |
| 09-22 06:10 | `nv2-easy-ft-hppnet-l2-1a711c` | queued | — | — | — | — | — | — |
| 09-22 06:35 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 06:35 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 06:35 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 06:35 | `nv2-hard-hppnet-l2-f71237` | running | 67 | 61 | 8.66 | 8.66 | 8.17 | 1e-03 |
| 09-22 06:35 | `nv2-mixed-scv2-83d2cd` | running | 53 | 38 | 2.95 | 2.81 | 2.51 | 5e-04 |
| 09-22 06:35 | `nv2-mixed-hppnet-l2-08c04f` | running | 22 | 18 | 2.52 | 2.41 | 2.41 | 1e-03 |
| 09-22 06:35 | `nv2-easy-ft-scv2-9ff3c2` | running | 16 | 10 | 2.79 | 2.73 | 2.64 | 1e-03 |
| 09-22 06:35 | `nv2-hard-ft-scv2-56dbff` | queued | — | — | — | — | — | — |
| 09-22 06:35 | `nv2-easy-ft-hppnet-l2-1a711c` | queued | — | — | — | — | — | — |
| 09-22 07:01 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 07:01 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 07:01 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 07:01 | `nv2-hard-hppnet-l2-f71237` | running | 73 | 72 | 8.21 | 8.21 | 7.34 | 1e-03 |
| 09-22 07:01 | `nv2-mixed-scv2-83d2cd` | running | 69 | 57 | 2.80 | 2.71 | 2.51 | 3e-04 |
| 09-22 07:01 | `nv2-mixed-hppnet-l2-08c04f` | running | 29 | 18 | 2.52 | 2.41 | 2.41 | 1e-03 |
| 09-22 07:01 | `nv2-easy-ft-scv2-9ff3c2` | running | 33 | 10 | 2.79 | 2.73 | 2.56 | 3e-04 |
| 09-22 07:01 | `nv2-hard-ft-scv2-56dbff` | queued | — | — | — | — | — | — |
| 09-22 07:01 | `nv2-easy-ft-hppnet-l2-1a711c` | queued | — | — | — | — | — | — |
| 09-22 07:26 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 07:26 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 07:26 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 07:26 | `nv2-hard-hppnet-l2-f71237` | running | 80 | 79 | 7.15 | 6.80 | 6.72 | 1e-03 |
| 09-22 07:26 | `nv2-mixed-scv2-83d2cd` | running | 84 | 79 | 2.58 | 2.58 | 2.47 | 3e-04 |
| 09-22 07:26 | `nv2-mixed-hppnet-l2-08c04f` | running | 37 | 18 | 2.52 | 2.41 | 2.26 | 1e-03 |
| 09-22 07:26 | `nv2-easy-ft-scv2-9ff3c2` | running | 49 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 07:26 | `nv2-hard-ft-scv2-56dbff` | queued | — | — | — | — | — | — |
| 09-22 07:26 | `nv2-easy-ft-hppnet-l2-1a711c` | queued | — | — | — | — | — | — |
| 09-22 07:51 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 07:51 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 07:51 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 07:51 | `nv2-hard-hppnet-l2-f71237` | running | 86 | 79 | 7.15 | 6.80 | 6.72 | 1e-03 |
| 09-22 07:51 | `nv2-mixed-scv2-83d2cd` | running | 101 | 91 | 2.53 | 2.53 | 2.47 | 1e-04 |
| 09-22 07:51 | `nv2-mixed-hppnet-l2-08c04f` | running | 45 | 40 | 2.48 | 2.48 | 2.26 | 1e-03 |
| 09-22 07:51 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 07:51 | `nv2-hard-ft-scv2-56dbff` | running | 11 | 6 | 2.83 | 2.65 | 2.62 | 1e-03 |
| 09-22 07:51 | `nv2-easy-ft-hppnet-l2-1a711c` | queued | — | — | — | — | — | — |
| 09-22 07:53 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 07:53 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 07:53 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 07:53 | `nv2-hard-hppnet-l2-f71237` | running | 86 | 79 | 7.15 | 6.80 | 6.72 | 1e-03 |
| 09-22 07:53 | `nv2-mixed-scv2-83d2cd` | running | 102 | 91 | 2.53 | 2.53 | 2.47 | 1e-04 |
| 09-22 07:53 | `nv2-mixed-hppnet-l2-08c04f` | running | 46 | 40 | 2.48 | 2.48 | 2.26 | 1e-03 |
| 09-22 07:53 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 07:53 | `nv2-hard-ft-scv2-56dbff` | running | 12 | 6 | 2.83 | 2.65 | 2.62 | 1e-03 |
| 09-22 07:53 | `nv2-easy-ft-hppnet-l2-1a711c` | queued | — | — | — | — | — | — |
| 09-22 08:19 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 08:19 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 08:19 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 08:19 | `nv2-hard-hppnet-l2-f71237` | running | 93 | 79 | 7.15 | 6.80 | 6.72 | 1e-03 |
| 09-22 08:19 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 08:19 | `nv2-mixed-hppnet-l2-08c04f` | running | 54 | 40 | 2.48 | 2.48 | 2.26 | 1e-03 |
| 09-22 08:19 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 08:19 | `nv2-hard-ft-scv2-56dbff` | running | 31 | 24 | 2.47 | 2.23 | 2.23 | 1e-03 |
| 09-22 08:19 | `nv2-easy-ft-hppnet-l2-1a711c` | queued | — | — | — | — | — | — |
| 09-22 08:20 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 08:20 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 08:20 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 08:20 | `nv2-hard-hppnet-l2-f71237` | running | 93 | 79 | 7.15 | 6.80 | 6.72 | 1e-03 |
| 09-22 08:20 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 08:20 | `nv2-mixed-hppnet-l2-08c04f` | running | 54 | 40 | 2.48 | 2.48 | 2.26 | 1e-03 |
| 09-22 08:20 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 08:20 | `nv2-hard-ft-scv2-56dbff` | running | 33 | 24 | 2.47 | 2.23 | 2.23 | 5e-04 |
| 09-22 08:20 | `nv2-easy-ft-hppnet-l2-1a711c` | queued | — | — | — | — | — | — |
| 09-22 08:23 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 08:23 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 08:23 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 08:23 | `nv2-hard-hppnet-l2-f71237` | running | 94 | 79 | 7.15 | 6.80 | 6.72 | 1e-03 |
| 09-22 08:23 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 08:23 | `nv2-mixed-hppnet-l2-08c04f` | running | 55 | 40 | 2.48 | 2.48 | 2.26 | 1e-03 |
| 09-22 08:23 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 08:23 | `nv2-hard-ft-scv2-56dbff` | running | 35 | 24 | 2.47 | 2.23 | 2.23 | 5e-04 |
| 09-22 08:23 | `nv2-easy-ft-hppnet-l2-1a711c` | running | — | — | — | — | — | — |
| 09-22 08:56 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 08:56 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 08:56 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 08:56 | `nv2-hard-hppnet-l2-f71237` | running | 102 | 79 | 7.15 | 6.80 | 6.72 | 1e-03 |
| 09-22 08:56 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 08:56 | `nv2-mixed-hppnet-l2-08c04f` | running | 65 | 61 | 2.46 | 2.36 | 2.26 | 1e-03 |
| 09-22 08:56 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 08:56 | `nv2-hard-ft-scv2-56dbff` | running | 60 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 08:56 | `nv2-easy-ft-hppnet-l2-1a711c` | running | 10 | 5 | 2.46 | 2.11 | 2.11 | 1e-03 |
| 09-22 09:22 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 09:22 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 09:22 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 09:22 | `nv2-hard-hppnet-l2-f71237` | running | 109 | 79 | 7.15 | 6.80 | 6.72 | 5e-04 |
| 09-22 09:22 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 09:22 | `nv2-mixed-hppnet-l2-08c04f` | running | 72 | 72 | 2.38 | 2.25 | 2.25 | 5e-04 |
| 09-22 09:22 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 09:22 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 09:22 | `nv2-easy-ft-hppnet-l2-1a711c` | running | 18 | 5 | 2.46 | 2.11 | 2.11 | 1e-03 |
| 09-22 09:48 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 09:48 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 09:48 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 09:48 | `nv2-hard-hppnet-l2-f71237` | running | 115 | 79 | 7.15 | 6.80 | 6.72 | 5e-04 |
| 09-22 09:48 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 09:48 | `nv2-mixed-hppnet-l2-08c04f` | running | 80 | 72 | 2.38 | 2.25 | 2.25 | 5e-04 |
| 09-22 09:48 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 09:48 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 09:48 | `nv2-easy-ft-hppnet-l2-1a711c` | running | 26 | 5 | 2.46 | 2.11 | 2.11 | 5e-04 |
| 09-22 10:13 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 10:13 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 10:13 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 10:13 | `nv2-hard-hppnet-l2-f71237` | running | 122 | 122 | 7.04 | 6.33 | 6.33 | 5e-04 |
| 09-22 10:13 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 10:13 | `nv2-mixed-hppnet-l2-08c04f` | running | 87 | 72 | 2.38 | 2.25 | 2.25 | 3e-04 |
| 09-22 10:13 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 10:13 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 10:13 | `nv2-easy-ft-hppnet-l2-1a711c` | running | 35 | 5 | 2.46 | 2.11 | 2.11 | 5e-04 |
| 09-22 10:39 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 10:39 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 10:39 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 10:39 | `nv2-hard-hppnet-l2-f71237` | running | 128 | 124 | 6.47 | 6.47 | 6.33 | 5e-04 |
| 09-22 10:39 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 10:39 | `nv2-mixed-hppnet-l2-08c04f` | running | 95 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 10:39 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 10:39 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 10:39 | `nv2-easy-ft-hppnet-l2-1a711c` | running | 44 | 5 | 2.46 | 2.11 | 2.11 | 5e-04 |
| 09-22 11:04 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 11:04 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 11:04 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 11:04 | `nv2-hard-hppnet-l2-f71237` | running | 135 | 124 | 6.47 | 6.47 | 6.33 | 3e-04 |
| 09-22 11:04 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 11:04 | `nv2-mixed-hppnet-l2-08c04f` | running | 103 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 11:04 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 11:04 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 11:04 | `nv2-easy-ft-hppnet-l2-1a711c` | running | 53 | 5 | 2.46 | 2.11 | 2.11 | 3e-04 |
| 09-22 11:30 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 11:30 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 11:30 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 11:30 | `nv2-hard-hppnet-l2-f71237` | running | 141 | 124 | 6.47 | 6.47 | 6.33 | 1e-04 |
| 09-22 11:30 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 11:30 | `nv2-mixed-hppnet-l2-08c04f` | running | 110 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 11:30 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 11:30 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 11:30 | `nv2-easy-ft-hppnet-l2-1a711c` | running | 62 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 11:56 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 11:56 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 11:56 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 11:56 | `nv2-hard-hppnet-l2-f71237` | running | 147 | 124 | 6.47 | 6.47 | 6.25 | 1e-04 |
| 09-22 11:56 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 11:56 | `nv2-mixed-hppnet-l2-08c04f` | succeeded | 112 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 11:56 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 11:56 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 11:56 | `nv2-easy-ft-hppnet-l2-1a711c` | running | 71 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 12:21 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 12:21 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 12:21 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 12:21 | `nv2-hard-hppnet-l2-f71237` | running | 154 | 124 | 6.47 | 6.47 | 6.25 | 1e-04 |
| 09-22 12:21 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 12:21 | `nv2-mixed-hppnet-l2-08c04f` | succeeded | 112 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 12:21 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 12:21 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 12:21 | `nv2-easy-ft-hppnet-l2-1a711c` | succeeded | 79 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 12:47 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 12:47 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 12:47 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 12:47 | `nv2-hard-hppnet-l2-f71237` | running | 161 | 124 | 6.47 | 6.47 | 6.25 | 1e-04 |
| 09-22 12:47 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 12:47 | `nv2-mixed-hppnet-l2-08c04f` | succeeded | 112 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 12:47 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 12:47 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 12:47 | `nv2-easy-ft-hppnet-l2-1a711c` | succeeded | 79 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 13:13 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 13:13 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 13:13 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 13:13 | `nv2-hard-hppnet-l2-f71237` | succeeded | 162 | 124 | 6.47 | 6.47 | 6.25 | 1e-04 |
| 09-22 13:13 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 13:13 | `nv2-mixed-hppnet-l2-08c04f` | succeeded | 112 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 13:13 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 13:13 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 13:13 | `nv2-easy-ft-hppnet-l2-1a711c` | succeeded | 79 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 13:40 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 13:40 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 13:40 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 13:40 | `nv2-hard-hppnet-l2-f71237` | succeeded | 162 | 124 | 6.47 | 6.47 | 6.25 | 1e-04 |
| 09-22 13:40 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 13:40 | `nv2-mixed-hppnet-l2-08c04f` | succeeded | 112 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 13:40 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 13:40 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 13:40 | `nv2-easy-ft-hppnet-l2-1a711c` | succeeded | 79 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 13:40 | `nv2-hard-ft-hppnet-l2-6fe90f` | running | 5 | 4 | 1.96 | 4.38 | 1.81 | 1e-03 |
| 09-22 14:06 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 14:06 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 14:06 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 14:06 | `nv2-hard-hppnet-l2-f71237` | succeeded | 162 | 124 | 6.47 | 6.47 | 6.25 | 1e-04 |
| 09-22 14:06 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 14:06 | `nv2-mixed-hppnet-l2-08c04f` | succeeded | 112 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 14:06 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 14:06 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 14:06 | `nv2-easy-ft-hppnet-l2-1a711c` | succeeded | 79 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 14:06 | `nv2-hard-ft-hppnet-l2-6fe90f` | running | 11 | 4 | 1.96 | 4.38 | 1.81 | 1e-03 |
| 09-22 14:32 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 14:32 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 14:32 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 14:32 | `nv2-hard-hppnet-l2-f71237` | succeeded | 162 | 124 | 6.47 | 6.47 | 6.25 | 1e-04 |
| 09-22 14:32 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 14:32 | `nv2-mixed-hppnet-l2-08c04f` | succeeded | 112 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 14:32 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 14:32 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 14:32 | `nv2-easy-ft-hppnet-l2-1a711c` | succeeded | 79 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 14:32 | `nv2-hard-ft-hppnet-l2-6fe90f` | running | 18 | 4 | 1.96 | 4.38 | 1.81 | 1e-03 |
| 09-22 14:58 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 14:58 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 14:58 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 14:58 | `nv2-hard-hppnet-l2-f71237` | succeeded | 162 | 124 | 6.47 | 6.47 | 6.25 | 1e-04 |
| 09-22 14:58 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 14:58 | `nv2-mixed-hppnet-l2-08c04f` | succeeded | 112 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 14:58 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 14:58 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 14:58 | `nv2-easy-ft-hppnet-l2-1a711c` | succeeded | 79 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 14:58 | `nv2-hard-ft-hppnet-l2-6fe90f` | running | 24 | 4 | 1.96 | 4.38 | 1.81 | 1e-03 |
| 09-22 15:24 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 15:24 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 15:24 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 15:24 | `nv2-hard-hppnet-l2-f71237` | succeeded | 162 | 124 | 6.47 | 6.47 | 6.25 | 1e-04 |
| 09-22 15:24 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 15:24 | `nv2-mixed-hppnet-l2-08c04f` | succeeded | 112 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 15:24 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 15:24 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 15:24 | `nv2-easy-ft-hppnet-l2-1a711c` | succeeded | 79 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 15:24 | `nv2-hard-ft-hppnet-l2-6fe90f` | running | 31 | 4 | 1.96 | 4.38 | 1.81 | 5e-04 |
| 09-22 15:49 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 15:49 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 15:49 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 15:49 | `nv2-hard-hppnet-l2-f71237` | succeeded | 162 | 124 | 6.47 | 6.47 | 6.25 | 1e-04 |
| 09-22 15:49 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 15:49 | `nv2-mixed-hppnet-l2-08c04f` | succeeded | 112 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 15:49 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 15:49 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 15:49 | `nv2-easy-ft-hppnet-l2-1a711c` | succeeded | 79 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 15:49 | `nv2-hard-ft-hppnet-l2-6fe90f` | running | 37 | 4 | 1.96 | 4.38 | 1.81 | 5e-04 |
| 09-22 16:21 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 16:21 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 16:21 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 16:21 | `nv2-hard-hppnet-l2-f71237` | succeeded | 162 | 124 | 6.47 | 6.47 | 6.25 | 1e-04 |
| 09-22 16:21 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 16:21 | `nv2-mixed-hppnet-l2-08c04f` | succeeded | 112 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 16:21 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 16:21 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 16:21 | `nv2-easy-ft-hppnet-l2-1a711c` | succeeded | 79 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 16:21 | `nv2-hard-ft-hppnet-l2-6fe90f` | running | 45 | 4 | 1.96 | 4.38 | 1.81 | 3e-04 |
| 09-22 16:46 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 16:46 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 16:46 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 16:46 | `nv2-hard-hppnet-l2-f71237` | succeeded | 162 | 124 | 6.47 | 6.47 | 6.25 | 1e-04 |
| 09-22 16:46 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 16:46 | `nv2-mixed-hppnet-l2-08c04f` | succeeded | 112 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 16:46 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 16:46 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 16:46 | `nv2-easy-ft-hppnet-l2-1a711c` | succeeded | 79 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 16:46 | `nv2-hard-ft-hppnet-l2-6fe90f` | running | 51 | 4 | 1.96 | 4.38 | 1.81 | 1e-04 |
| 09-22 17:12 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 17:12 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 17:12 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 17:12 | `nv2-hard-hppnet-l2-f71237` | succeeded | 162 | 124 | 6.47 | 6.47 | 6.25 | 1e-04 |
| 09-22 17:12 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 17:12 | `nv2-mixed-hppnet-l2-08c04f` | succeeded | 112 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 17:12 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 17:12 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 17:12 | `nv2-easy-ft-hppnet-l2-1a711c` | succeeded | 79 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 17:12 | `nv2-hard-ft-hppnet-l2-6fe90f` | running | 58 | 4 | 1.96 | 4.38 | 1.81 | 1e-04 |
| 09-22 17:37 | `nv2-easy-scv2-15cff2` | succeeded | 101 | 81 | 7.94 | 7.94 | 7.18 | 1e-04 |
| 09-22 17:37 | `nv2-hard-scv2-d926d0` | succeeded | 115 | 15 | 9.17 | 7.06 | 7.06 | 1e-04 |
| 09-22 17:37 | `nv2-easy-hppnet-l2-9a2ecf` | succeeded | 60 | 20 | 7.23 | 6.46 | 5.52 | 1e-04 |
| 09-22 17:37 | `nv2-hard-hppnet-l2-f71237` | succeeded | 162 | 124 | 6.47 | 6.47 | 6.25 | 1e-04 |
| 09-22 17:37 | `nv2-mixed-scv2-83d2cd` | succeeded | 115 | 91 | 2.53 | 2.53 | 2.43 | 1e-04 |
| 09-22 17:37 | `nv2-mixed-hppnet-l2-08c04f` | succeeded | 112 | 72 | 2.38 | 2.25 | 2.25 | 1e-04 |
| 09-22 17:37 | `nv2-easy-ft-scv2-9ff3c2` | succeeded | 53 | 10 | 2.79 | 2.73 | 2.56 | 1e-04 |
| 09-22 17:37 | `nv2-hard-ft-scv2-56dbff` | succeeded | 70 | 24 | 2.47 | 2.23 | 2.23 | 1e-04 |
| 09-22 17:37 | `nv2-easy-ft-hppnet-l2-1a711c` | succeeded | 79 | 5 | 2.46 | 2.11 | 2.11 | 1e-04 |
| 09-22 17:37 | `nv2-hard-ft-hppnet-l2-6fe90f` | succeeded | 60 | 4 | 1.96 | 4.38 | 1.81 | 1e-04 |

### Stage-1 and curriculum results

`real_overall` = `val/real_r3`, rev/s PIT MAE, lower is better. **sel** is the
round `best_real_overall.ckpt` was written at (best SMOOTHED); **raw @ sel** is
that round's unsmoothed value, is the number the regime rows below belong to,
and is the number to quote. **best raw** is the lowest raw value anywhere in
the run, which no checkpoint corresponds to; it is here only so the gap between
the two is visible. `r1`/`r2`/`r3` are the per-view values at **sel**. All ten
arms ran to their own early stop; none was killed, none failed.

| arm | job | rounds | sel | smoothed | **raw @ sel** | best raw | r1 | r2 | r3 | r1/r2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| *`real_r4_scv2_unified`* — real reference, regressor | — | 70 | 24 | — | 3.11 | **2.99** | — | — | 2.99 | — |
| *`hppnet_l2_r2_s0`* — real reference, salience | — | — | — | — | 2.22 | **2.27** | — | — | 2.27 | — |
| *`rig_easy_scv2_unified`* — legacy stochastic easy | — | 94 | 57 | — | 6.09 | **5.72** | — | — | — | — |
| *`rig_hard_scv2_unified`* — legacy stochastic hard | — | 137 | 124 | — | 5.41 | **5.37** | — | — | — | — |
| `nv2_easy_scv2` | `nv2-easy-scv2-15cff2` | 102 | 81 | 7.94 | **7.94** | 7.18 | 10.89 | 9.67 | 7.94 | 1.13 |
| `nv2_hard_scv2` | `nv2-hard-scv2-d926d0` | 116 | 15 | 9.17 | **7.06** | 7.06 | 25.11 | 9.38 | 7.06 | 2.68 |
| `nv2_mixed_scv2` | `nv2-mixed-scv2-83d2cd` | 116 | 91 | 2.53 | **2.53** | 2.43 | 2.59 | 2.93 | 2.53 | 0.88 |
| `nv2_easy_ft_scv2` | `nv2-easy-ft-scv2-9ff3c2` | 54 | 10 | 2.79 | **2.73** | 2.56 | 3.34 | 3.18 | 2.73 | 1.05 |
| `nv2_hard_ft_scv2` | `nv2-hard-ft-scv2-56dbff` | 71 | 24 | 2.47 | **2.23** | 2.23 | 2.80 | 2.67 | 2.23 | 1.05 |
| `nv2_easy_hppnet_l2` | `nv2-easy-hppnet-l2-9a2ecf` | 61 | 20 | 7.23 | **6.46** | 5.52 | 10.59 | 8.55 | 6.46 | 1.24 |
| `nv2_hard_hppnet_l2` | `nv2-hard-hppnet-l2-f71237` | 163 | 124 | 6.47 | **6.47** | 6.25 | 11.50 | 8.35 | 6.47 | 1.38 |
| `nv2_mixed_hppnet_l2` | `nv2-mixed-hppnet-l2-08c04f` | 113 | 72 | 2.38 | **2.25** | 2.25 | 2.66 | 2.75 | 2.25 | 0.97 |
| `nv2_easy_ft_hppnet_l2` | `nv2-easy-ft-hppnet-l2-1a711c` | 80 | 5 | 2.46 | **2.11** | 2.11 | 2.38 | 2.48 | 2.11 | 0.96 |
| `nv2_hard_ft_hppnet_l2` | `nv2-hard-ft-hppnet-l2-6fe90f` | 61 | 4 | 1.96 | **4.38** | 1.81 | 6.26 | 6.19 | 4.38 | 1.01 |

Three things in that table are artefacts of SELECTION, not of training, and are
marked here so they are not read as results:

* **`nv2_hard_ft_hppnet_l2` is the one arm the smoother cost.** Its raw score
  hit **1.81** at round 1 — the best number in the whole batch, 20 % under the
  2.27 real salience reference — and the 5-round median bottomed at round 4,
  whose own raw score is 4.38. The file on R2 is round 4. The arm's real result
  is therefore unmeasured: the checkpoint that scored 1.81 was never saved.
* **`nv2_hard_scv2`'s `r1`/`r2` ratio of 2.68** (25.11 single-mic against 9.38
  eight-mic) is the same effect: its selected round is 15 of 116, before the
  single-mic view had converged at all.
* **`nv2_easy_ft_hppnet_l2`** also selected early (round 5 of 80) but its raw
  score there, 2.11, happens to be its best, so nothing is lost.

The post-best real degradation of the point-preset run **recurs on every arm,
but its size splits the batch cleanly in two.** Final round against best raw:
synthetic-only 8.70/7.18 (+21 %), 12.18/7.06 (+73 %), 15.28/5.52 (+177 %),
7.09/6.25 (+13 %); with real audio in the stream 2.55/2.43 (+5 %), 2.85/2.56
(+11 %), 2.68/2.23 (+20 %), 2.73/2.25 (+21 %), 2.64/2.11 (+25 %). The one arm
that degrades like a synthetic-only run despite training on real audio is
`nv2_hard_ft_hppnet_l2` (3.80/1.81, +110 %), and its +110 % is measured against
a round-1 score, i.e. against a number the run held for one validation round.
Real data in the stream bounds
the drift to a fifth of the score; the fitted family on its own does not bound
it at all.

### Four-regime decomposition

`python scripts/_regime_decomp.py --exp <arm> --ckpt best_real_overall`, 8 mics,
the frozen real split, thresholds stated not fitted (ramp
|d mean speed/dt| >= 20 rev/s^2 taken FIRST, then zero / standby / cruise on
level). Cells are per-frame PIT MAE in rev/s; **overall** is training's own
clip-level PIT metric recomputed on the SELECTED checkpoint, so it can be
checked against `raw @ sel` above and in every row it agrees to ~1 %. Frame
shares: zero 12.7 %, standby 11.6 %, ramp 3.8 %, cruise 72.0 %.

**Output spread** is the mean over frames of the rotor peak-to-peak
(`max - min`) of the PREDICTED speeds — the quantity that reads directly as
"how far apart does the model put the outer two rotors", and the one the legacy
pair's collapse claim was about. It is measured here for every row including
the references, and `scripts/_regime_decomp.py` now records
`spread.{regime}.{pred,true}` in its JSON so it is reproducible.

**The legacy 0.28 / 0.19 / 4.71 are NOT reproduced by this definition and are
not used below.** Re-measuring the same three checkpoints peak-to-peak gives
ramp spreads 2.87 (legacy easy) / 4.14 (legacy hard) / 6.58 (real): the
collapse the legacy conclusion described is a factor of 2.3 and 1.6, not of 17
and 25. Whatever quantity produced 0.28 was never committed and cannot be
checked. Every spread number in this batch is the peak-to-peak one, measured
the same way for arms and references alike. The DIRECTION of the legacy finding
survives the re-measurement; its magnitude does not.

| arm | overall | zero | standby | ramp | cruise | spread standby | spread ramp | spread cruise |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *`real_r4_scv2_unified`* — real reference, regressor | 3.11 | 2.44 | 3.20 | 8.17 | 2.95 | 9.74 | 6.58 | 10.94 |
| *`hppnet_l2_r2_s0`* — real reference, salience (`best.ckpt`) | 2.22 | 1.05 | 3.11 | 12.51 | 1.74 | 8.81 | 6.76 | 13.53 |
| *`rig_easy_scv2_unified`* — legacy stochastic easy | 6.08 | 2.37 | 13.70 | 18.82 | 4.85 | 1.65 | 2.87 | 10.36 |
| *`rig_hard_scv2_unified`* — legacy stochastic hard | 5.40 | 10.61 | 8.92 | 13.86 | 3.48 | 1.55 | 4.14 | 11.50 |
| `nv2_easy_scv2` | 7.99 | 10.64 | 11.05 | 11.12 | 6.87 | 5.71 | 6.42 | 16.76 |
| `nv2_hard_scv2` | 7.05 | 1.25 | 6.16 | 15.81 | 7.76 | 6.27 | 8.14 | 18.56 |
| `nv2_mixed_scv2` | 2.54 | 1.78 | 2.25 | 7.00 | 2.49 | 9.76 | 7.82 | 12.77 |
| `nv2_easy_ft_scv2` | 2.73 | 2.23 | 2.36 | 8.16 | 2.60 | 10.16 | 7.37 | 12.75 |
| `nv2_hard_ft_scv2` | 2.23 | 0.93 | 1.44 | 6.71 | 2.35 | 9.22 | 6.63 | 11.23 |
| `nv2_easy_hppnet_l2` | 6.45 | 3.58 | 5.04 | 26.50 | 6.14 | 7.54 | 9.95 | 15.46 |
| `nv2_hard_hppnet_l2` | 6.47 | 1.63 | 4.25 | 30.07 | 6.44 | 5.37 | 6.11 | 17.25 |
| `nv2_mixed_hppnet_l2` | 2.25 | 0.29 | 2.28 | 9.09 | 2.23 | 8.31 | 7.83 | 14.44 |
| `nv2_easy_ft_hppnet_l2` | 2.11 | 1.50 | 2.84 | 8.25 | 1.77 | 9.08 | 7.29 | 13.06 |
| `nv2_hard_ft_hppnet_l2` | 4.42 | 20.91 | 2.83 | 9.50 | 1.51 | 10.57 | 7.42 | 13.84 |
| **the TARGET labels of the same frames** | — | 0.01 | 10.63 | 7.85 | 13.79 | 10.63 | 7.85 | 13.79 |

`nv2_hard_ft_hppnet_l2`'s row is the round-4 checkpoint discussed above and not
a result: its 20.91 at zero — 8x every other row — is what a model looks like
one round after a warm start, before it has learned that all rotors stopped
means predict nothing. Its cruise cell (1.51, the batch's best) shows what the
arm was on its way to.

## Conclusion

**Question 1 — does the fitted family transfer better than the hand-written
one? No.** On the regressor the fitted arms score 7.94 (easy) and 7.06 (hard)
against the legacy stochastic pair's 5.72 and 5.37; on the salience trunk they
score 6.46 and 6.47, which has no legacy counterpart but sits 2.8x from that
family's 2.27 real reference. Every synthetic-only v2 arm is WORSE than the
hand-written family it replaces. This is a clean negative and it does not
depend on the selection caveats: the best raw scores anywhere in those four
runs are 7.18 / 7.06 / 5.52 / 6.25, all still above 5.37.

**And the decomposition says the family did exactly what it was built to do.**
The failure the whole noise-model-v2 campaign was aimed at is the rotor
collapse, and the fitted family fixes it. At standby the legacy arms put the
four rotors 1.65 and 1.55 rev/s apart where the labels are 10.63 apart; the v2
arms put them 5.71 / 6.27 / 7.54 / 5.37 apart. At ramp the legacy arms give
2.87 and 4.14 against a 7.85 target and a real-trained 6.58; the v2 arms give
6.42 / 8.14 / 9.95 / 6.11 — i.e. the fitted trajectories under a stated cap,
flown at an absolute fitted level, produce a model that separates the rotors
about as much as the real data does. The ramp CELL follows on the easy
regressor arm — 11.12 against the legacy easy arm's 18.82 — but not elsewhere:
the hard regressor arm is 15.81 against 13.86, and both salience arms are far
worse there (26.50 and 30.07). Rotor separation is restored; ramp ACCURACY is
restored on one arm of four.

**The deficit moved to cruise, and cruise is 72 % of the frames.** The v2
synthetic-only arms score 6.87 / 7.76 / 6.14 / 6.44 at cruise against the
legacy 4.85 / 3.48 and the real 2.95, and they OVER-separate there: cruise
spread 16.76 / 18.56 / 15.46 / 17.25 against a 13.79 target and a real-trained
10.94; the legacy arms, by contrast, were tight there (10.36 / 11.50) and
accurate (4.85 / 3.48). The fitted absolute-level family buys transition
behaviour and pays for
it in steady-flight precision, and because cruise carries 62–79 % of each arm's
total error, the trade is a net loss on the headline metric. That is the
batch's substantive finding, and it is a sharper statement than "synthetic
transfer is 1.8x away": the remaining gap is no longer the transitions.

**Question 2 — does pre-training on it help a real-data run? Yes, and this is
the batch's positive result.** `nv2_hard_ft_scv2` reaches **2.23** against the
real reference's 2.99 — 25 % better than training the same arm on real audio
from scratch — and `nv2_easy_ft_scv2` reaches 2.73, 9 % better.
`nv2_easy_ft_hppnet_l2` reaches **2.11** against 2.27, 7 % better.
`nv2_hard_ft_hppnet_l2` touched 1.81 but its saved checkpoint is the round-4
one at 4.38, so it is reported as unmeasured rather than as its best number.
All three MEASURED curriculum arms beat their real reference; none is worse,
and the fourth is unmeasured rather than negative. The fitted family therefore does carry something the real pool does
not, even though a model trained on it alone is poor.

**Question 3 — does it add to real data when mixed in? Yes on the regressor,
neutrally on the salience trunk.** `nv2_mixed_scv2` gets **2.53** against 2.99
(15 % better); `nv2_mixed_hppnet_l2` gets **2.25** against 2.27, which is a
tie. Mixing is cheaper than the curriculum — one stage, not two — and on the
regressor it recovers about three fifths of the curriculum's gain (0.46 rev/s
of 0.76).

**Ranking the two ways of using the family:** curriculum 2.23 < mixed 2.53 on
the regressor, curriculum 2.11 < mixed 2.25 on the salience trunk. Warm-starting
beats mixing on both families, by 0.30 and 0.14 rev/s.

**What the best arms fixed.** `nv2_hard_ft_scv2` at 2.23 is better than the
real reference in every single cell — zero 0.93 vs 2.44, standby 1.44 vs 3.20,
ramp 6.71 vs 8.17, cruise 2.35 vs 2.95 — while keeping the real-trained rotor
separation (standby spread 9.22, ramp 6.63 against targets 10.63 and 7.85). The
gain is not one regime; the warm start moves the whole surface.

**Practical reading.** The fitted noise model is not a substitute for real
audio and is a worse substitute than the hand-written sampler it replaced. It
IS a better initialiser and a useful mixing component, and it is the first
synthetic family in this project that does not collapse the rotors in the
transitions. The next thing to fix is its cruise over-separation, which is
where all of its remaining error lives.

**One process finding worth carrying out of this batch:** selecting
`best_<score>.ckpt` on the 5-round median cost one arm its result outright
(1.81 raw, 4.38 saved, round 4 of 61) and put two others on the checkpoints of
round 5 of 80 and round 15 of 116. On fast-improving warm-started runs the
median is far enough behind the raw score to save the wrong file. Saving the
best RAW checkpoint alongside the smoothed one would have cost one extra file
per run and would have kept the best number in this batch.
