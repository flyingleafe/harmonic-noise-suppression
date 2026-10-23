# Paper real-data protocol (v2): streams, schedule, panel, checkpoints

**Status:** design complete (2026-10-14) · no run started · supersedes the "unified
regime" of `pyramid-harmonic.md` / `paper-regime-matrix.md` for every real-data
number the wrap-up paper will report. Synthetic pretraining is a separate,
later protocol and is out of scope here.

Experiment prefix: `rp_*` (`conf/experiment/rp_<model>_<stream>[_<ablation>].yaml`).

## Motivation

The paper (`writing/papers/2026-08_wrapup/`, narrative of record in
`docs/wrapup-paper-extended-abstract.md`) is being rewritten around HRSPNet
and the real-data training ladder. The evidence behind its claims comes from a
large but incoherent set of runs (two regimes, single seeds, cancelled arms,
early-epoch aliases). This protocol is the single regime every reported
real-data row is rerun under. It was redesigned from the unified regime
(`real_r4_*_unified`: `conf/online_mix/hb_m3s2_dload.yaml`,
`conf/validation/rps_unified.yaml`) after four defects were found while
mapping it (2026-10-14):

1. **The LR depended on validation.** `MultiMetricController` halves the LR on
   8 stale rounds of the median-smoothed `overall_macro`
   (`src/training/validation.py:296-303`); the nominal `ReduceLROnPlateau`
   is never stepped in that branch (`src/training/loop.py:893-895`). The
   selection signal therefore leaked into the optimisation, and the
   realised LR trajectory differed per arm.
2. **The control signal was half synthetic.** `overall_macro` = ½ real + ½
   the four `salv2_*` synthetic sets, so real-only arms were selected and
   LR-scheduled on synthetic audio.
3. **MD2 validation labels were pre-calibration.** The frozen split
   `DREGON-LM-V4-michaels-valid-full` (pin `9604f3ff…`, published 2026-07-12
   in `8f0216d9`, `adopt_only`) predates the FLY124/FLY125 calibration of
   2026-07-31 (`src/data_processing/sources/michaels.py:42-130`:
   shift + dilation + ×1.00698/1.00706). The training source
   `michaels-frames` was republished calibrated (`8856b0bb`, `efda3bf8`,
   repinned `d817f5cb`), the validation split never was. Measured on frozen
   clip 30 (FLY124 at 64.09 s, cruise frames): frozen/calibrated label ratio
   0.99320, calibrated − frozen = +0.570 rev/s, and the frozen track sits
   ≥ 32 ms early. Every MD2 validation number of the campaign carries that
   ≈ 0.57 rev/s label term. The calibration is three global constants
   (value scale, time offset, time dilation) applied to the raw DatCon
   table; trajectory shape is untouched. It is **not** `rps_refined`.
4. **The panel never scored the training condition.** `real_valid` mixes
   nothing (`src/data_processing/derivations.py:1680`): "source-present"
   means the 14 DREGON clips with an in-room loudspeaker; MD2 was scored on
   pure ego-noise only, while training mixes LibriSpeech at −30…0 dB.

Two further findings fix the design: `freq_scale` resamples at 16 kHz
(`src/data_processing/noise_augmentations.py:230`), so α < 1 leaves
[8α, 8] kHz silent and the time warp does the same locally; and the DREGON
training labels are `motors_command` (room 2 publishes nothing else), 0.5–3 %
above the acoustic rate per rotor (`paper-regime-matrix.md`, pyramid
level-3 analysis in `pyramid-harmonic.md`).

Decisions taken in the design discussion (2026-10-14, Dmitrii):
no `rps_refined` anywhere (proven still wrong on DREGON); best raw labels
only (`motors_measured` where present, else `motors_command`; MD2 calibrated
telemetry); train/validation clip-length mismatch (2 s / 8 s) is intentional;
one seed per arm, replication case by case; LibriSpeech stays the speech
corpus; which checkpoint the paper reports is decided after the runs.

## Data

### Splits (DREGON room 1 and MD2 are the only labelled recordings not in training)

| role | recordings | labels |
|---|---|---|
| train | DREGON `in_flight_noise` minus `free-flight_nosource_room1` (room 2: `free-flight`, `hovering`, `updown`, `rectangle`, `spinning` `_nosource_room2`); MD2 FLY125 | DREGON `motors_command` (cleaned, `clean_command_spikes`); FLY125 calibrated |
| validation | DREGON `free-flight_nosource_room1`, `free-flight_speech-low_room1`, `free-flight_whitenoise-low_room1`; MD2 FLY124 | DREGON `motors_measured`; FLY124 calibrated |
| **test** (scored once, at the end) | DREGON `free-flight_speech-high_room1`, `free-flight_whitenoise-high_room1`; MD2 FLY103, FLY108 (mono, same airframe, calibrated); optionally `silent-flight_whitenoise-low_room1` (rotors off) | as above |

The two high-level source flights have never been in a training stream or a
validation panel of the current models (checked 2026-10-14; their only uses
are label-side analyses and one V2-era regressor eval,
`dregon-lm-v2-v3-baseline.md`). To be made explicit as a `test` split in
`src/data_processing/sources/dregon.py` and in `src/data_processing/AGENTS.md`
so no derivation can root on them.

### Training streams (four arms; everything else identical)

| stream | noise sources | label-altering augmentation |
|---|---|---|
| `dregon` | DREGON room 2 (as above) | none |
| `md2` | FLY125 | none |
| `all` | both, duration-weighted as today | none |
| `all-labelaug` | both | `rate_warp` (below) |

Common to all: `min_motor_rps 0.0` (whole envelope), silence arm at weight
0.4 against the merged real weight (16.7 % of noise chunks, zero labels,
`snr_ref_floor_rms 0.02`), LibriSpeech `train-clean-100` mixed per channel
independently, 8 mics flattened to mono training frames, 2 s chunks, 16 kHz,
`base_seed` fixed. Each arm is scored on both drones, so the cross-drone
transfer table (`dregon → md2`, `md2 → dregon`) needs no extra runs.
`dregon-labelaug` / `md2-labelaug` are a possible later ablation, not part of
the base matrix.

### Validation panel (new derived dataset, versioned, NOT adopt-only)

Replaces `DREGON-LM-V4-michaels-valid-full`. Same clip recipe (8 s,
`max_non_overlapping`, `min_motor_rps 0`, seed 43) through the current
loaders, so the DREGON half is byte-identical in audio and the MD2 half gets
calibrated labels. Every clip carries `recording_id`, `t_start`,
`source_type`, `snr_db`, `dataset` in meta (the frozen split had none; the
calibration check above had to locate clip 30 by cross-correlation).

| group | clips | content |
|---|---|---|
| `dregon_nosource` | 8 | `free-flight_nosource_room1` |
| `dregon_source` | 7 + 7 | `free-flight_speech-low_room1`, `free-flight_whitenoise-low_room1` (acoustic loudspeaker, low level) |
| `md2_nosource` | 15 | FLY124 raw |
| `md2_source` | 8 + 7 | the same 15 FLY124 clips, each mixed once with LibriSpeech **dev-clean** (unseen in training): 8 clips at −5 dB, 7 at −20 dB source-to-noise SNR |

52 clips × 8 channels × 8 s ≈ 55 min; synthetic validation sets are dropped
from this protocol. MD2 mixing: one deterministic dev-clean utterance per
(clip, channel), independent per channel as in training, SNR over the 8 s
clip with the training mixer's reference floor, no augmentation, baked into
the dataset bytes. The two "source" cells answer different questions
(acoustic low-level source vs digital mix at a stated SNR) and are not
cross-drone comparable; the paper says so.

**Metric.** Per-frame permutation-matched MAE over the four rotors, rev/s,
from the model's own decoder on its linear output grid (as today,
`batched_pit_mae`). Per view: frame mean, clip-level mean ± clip-to-clip std,
clip count. No `_log` variants.

**Frame regimes** (the committed rule of `scripts/_regime_decomp.py`, applied
first-match): `ramp` |d(rotor-mean)/dt| ≥ 20 rev/s² (central difference
±0.25 s); else `zero` max rotor < 1 rev/s; else `standby` mean < 45 rev/s;
else `cruise`. Computed over all clips of a drone.

**Tracked scores (15).**

| | scores |
|---|---|
| totals | `dregon` (22 clips), `md2` (30), `real_macro` = ½ `dregon` + ½ `md2` — the **control** |
| source | `dregon_nosource`, `dregon_source`, `md2_nosource`, `md2_source` |
| regime | `{dregon,md2}_{zero,standby,ramp,cruise}` |

Per-SNR, per-regime×source and per-clip readings are post-hoc from the saved
per-clip predictions. Small cells to state next to every number: DREGON
zero/standby come from the ground segments of three flights; MD2 ramp from one
take-off/landing; ramp is ~4 % of frames.

Cadence: every 500 optimizer steps, 200 rounds (= the step budget).

## Training

### Budget and LR

Budget in optimizer steps: **up to 100 000** (today's "200 epochs" is
exactly 200 validation rounds × 500 steps at the step cap). AdamW, max LR
per family (below), weight decay 1e-4, **batch 128** mono frames (16
chunks × 8 mics), **micro-batch 128, no gradient accumulation, for every
arm** (Dmitrii, 2026-10-15). Target hardware is the A100-80; a smaller
card only where the model fits (the range tests measured: HPPNet's
`FreqGroupLSTM` over 352 bins OOMs at 128 on 16 and 32 GB — 10.9 GiB more
on top of 24.9 GiB at 32 GB — and LateDeep OOMs at 128 on 16 GB; HarmoF0,
Basic Pitch and the regressors run at 128 on 16 GB). If the `uni` queue
for the large cards is too slow, the arm goes to `vast` (A100-80). The
64 × 2 variant the first HPPNet pilot used is retired: the historical
64 × 2 came from `97b25a89`, "to fit any backend". AMP, grad-clip 5 —
unchanged. Schedule:
linear warm-up 0 → max over the first 5 % (5 000 steps), then cosine to
1 % of max at 100 000 steps, a pure function of the step count. **The LR
no longer depends on
validation**: the `MultiMetricController` loses its LR branch
(`lr_patience`, `lr_factor`, `min_lr_reductions`) and the plateau
scheduler goes; it keeps median smoothing for checkpoint selection and the
**saturation early stop** (`min_rounds`, `final_patience` on the control
score `real_macro`), which stays. Scheduler state goes into `train_state.pt` so
a resumed run continues the cosine.

Stop parameters (Dmitrii, 2026-10-15): **`min_rounds` 50** (25 000 steps,
a quarter of the budget) and **`final_patience` 30** (15 000 steps on the
control's 5-round median). The two are parallel conditions
(`rounds_completed ≥ min_rounds` AND `stale_rounds ≥ final_patience`, in
`training/stopping.py`), not additive: stale rounds count from round 5,
so a run whose control never improves after its first window stops at
round 50 = 25k steps, and one that improves at round r stops no earlier
than round max(50, r + 30). (The first HPPNet pilot stopped at round 40
under the earlier `min_rounds` 40 — the earliest the rule allows, not
"20k + 15k = 35k" as the 2026-10-14 text wrongly claimed.) An arm that
stops early never reaches the cosine tail, so arms differ in realised
budget; every table reports the stopping step next to the score.
`validation.control` is a list (2026-10-22): real arms use `[real_macro]`;
synthetic arms (`rp-synthetic.md`) use `[real_macro, static_comb]`, and
`stale_rounds` then counts rounds since EITHER median last improved.

**Max LR: 1e-3 for every model** (decided 2026-10-15, Dmitrii) — the LR
each architecture we build on was published with, so the choice is
citable rather than ours: HarmoF0 "Adam with default hyperparameters and
a learning rate of 10⁻³ for 50 epochs" (Wei et al. 2022, § 3); Cuesta et
al. 2020 (the Late/Deep HCQT CNN) "Adam optimizer with a learning rate of
0.001 … batch size of 16"; SimpleConv "AdamW (learning rate 10⁻³, weight
decay 10⁻⁴), batch size 16, mixed-precision, gradient clipping at 5.0"
(Mukhutdinov & Wang, Quiet Drones 2026, § 2.4) — the regime every earlier
run of this project used; HPPNet "Adam optimizer with mini-batch size 4,
and a learning rate of 0.0006" (Wei et al. 2022, § 3.2), rounded up to
1e-3 at our 32× larger batch; Basic Pitch's paper states no optimizer
(its public code trains with Adam at 1e-3). Context, not adopted: PESTO
Adam 1e-4 at batch 256; Liu et al. 2025 AdamW 5e-4; FCNF0++ a fixed 250k
steps with no early stopping. All library full texts, read 2026-10-15.
`min_lr` stays 1 % of the max.

**LR range test (appendix check, not the selection rule).** One sweep per
trunk family (Smith 2017; fastai `lr_find`): the real data/loss/optimizer
on `all` with the step-0 augmentation set, no validation, LR growing
geometrically 1e-5 → 1e-1 over 3 000 steps, logging (step, LR, training
loss); the sweep stops on a non-finite loss. Driver
`scripts/lr_range_test.py`; raw curves committed under
`results/rp_lr_range/`; `--summary` reads them with the rule below. The
curves so far all put 1e-3 inside the stable descent: HarmoF0 flat from
1e-4 to 1e-2, drifting up from 1.6e-2; Basic Pitch never rises (floor at
4.8e-3); SimpleConv's PIT-MSE falls 4 800 → 95 between 1e-4 and 6e-3 and
rises from 7.8e-3; SCv2 still falling steeply at 3.4e-3 when the fp16
forward went non-finite at 3.5e-3. HPPNet (64 × 2) and LateDeep pending
(32 GB cards; 16 GB OOMs). The automatic reading the driver reports —
EMA β = 0.99; hard divergence → divergence / 3; else rise = first LR after
the floor staying ≥ 5 % above it for 100 steps → rise / 10, at most the
floor's LR; snapped to half-decades — gave 1e-3, 3e-3, 1e-3, 1e-3 for the
four; it is ours (the ÷10 is fastai's `minimum` heuristic, the ÷3 Smith
folklore) and is reported, not used. A "steepest sustained descent"
reading was tried first and dropped: the loss falls at any LR during the
first few hundred steps (HarmoF0's steepest slope landed at 1.8e-5).
Families: HPPNet L0, HPPNet L2 (inherited by L1), +k84, +ConvLSTM, HRSPNet,
HarmoF0 L2, Basic Pitch L2, LateDeep L2, SimpleConv, best regressor — 10
sweeps × 3k steps, each a `uni-gpushort` job.

Precedent for the schedule: PESTO trains 50 epochs under cosine annealing
(Adam 1e-4); FCNF0++ omits early stopping and trains a fixed 250k steps
(library full texts, 2026-10-14).

### Checkpoints

Every round writes `last.ckpt`. For each of the 22 scores (the 14
dataset × {source, regime} views, the macro `real_macro` (the control, ½ dregon + ½ md2), and the seven pooled frame-weighted `real`, `real_{nosource,source}`, `real_{zero,standby,ramp,cruise}` — frame-
weighted over both datasets; added 2026-10-15, Dmitrii), two bests:
`best_<score>_median.ckpt` (5-round median improves, ≥ 1 % relative — what
`stable_improvements` does today) and `best_<score>_raw.ckpt` (raw round
value improves). All are hard links to `last.ckpt`
(`src/training/loop.py:311-316`), so coincident bests share one inode;
`best_checkpoints.json` records score → {step, round, raw, median};
R2 uploads use `alias_checkpoint` (verify it copies server-side). The
legacy single `best.ckpt` and `optim.monitor` go away. Which checkpoint the
paper reports is decided after the runs; `last.ckpt` is also reported
(selection-free).

### Augmentations (no schedule)

Selection rule (Dmitrii, 2026-10-14): a label-preserving augmentation is
in the recipe only if it is in common use for pitch tracking AND for other
audio tasks (separation, ASR); the label-altering one is compared against
the existing pitch-tracking alternative. Dropped on that rule:
`random_polarity` (only magnitude front ends are considered now; a no-op),
`tooth_dropout` and `spec_mask` (masking is an ASR/SED convention — no
pitch-tracking paper in the library uses it), `floor_inject` (duplicates
the drone's own floor and the speech mix).

**Revision 2026-10-15 (Dmitrii): everything at full strength from step 0,
and `random_reverb` is out.** The first HPPNet-L2 pilot (`rp_hppnet_l2_all`
at `8c3053b7`…`b90f7bb5`, 64 × 2, LR 1e-3) trained under the ramps below
memorised: `real_macro` 7.9 rev/s at step 1.5k, then rising monotonically to ~59
by step 7.5k and flat there — zeros everywhere except on the memorised
clips — until the saturation stop at round 40 (20k steps). Every earlier
unified HPPNet-L2 run (`hppnet_l2_r2_s0`, `real_r4_hppnet_l2_unified`)
saw the full −30…0 dB mixture distribution from sample 1 and fell steadily
through its first rounds; the ramped stream instead handed the model 10k
steps of near-noise-only input (speech ≤ −30 dB) on a 7-minute pool, in
which the per-clip label prior is the winning strategy. Hence one stage,
no ramps: `rp_all` = speech SNR U(−30, +5) dB from step 0, `random_gain`
and `spectral_recolor` at full strength from step 0. `random_reverb` is
dropped altogether — the earlier runs never used it and no pitch paper
does (table below); it had been the one transform with precedent only
outside pitch work. The ramped `rp_*` streams are history (`b90f7bb5`).

**Post-mortem, same day (the collapse was the off head, not the regime).**
The rerun on the no-schedule stream degraded identically, ~1k steps later
(`real_macro` 5.8 at step 2.5k → 20.3 at 4.5k → 36.8 at 5k) while
`hppnet_l2_r2_s0`'s own W&B history shows it training at a flat 1e-3 from
step 0 with no upturn, which ruled out the LR too. Probing the collapsed
checkpoint on the panel (`/tmp/off_probe.py`, 12 clips): the salience map
was correct throughout — cruise peak p 0.81 at ≈ 80 rev/s — but the OFF
logit sat at p 0.94–0.97 on every regime, cruise included, so the CRF
decoded "off" on 98 % of frames; forcing the off bin to −8 at decode time
took the same checkpoint from 64.8 to 10.7 rev/s MAE. Cause: the layer
BCE's `focus` weight (`1 + focus·target`, focus ≈ 65 = G / Gaussian mass)
also multiplied bin 0, whose positive is a class present on every stopped
frame (~20 % of training), tilting the off bin's loss ~65:1 towards "off";
the cheapest `Linear` off head on pooled energy then says off everywhere.
The earlier runs had no off bin (a stopped rotor was bin 0 of a plain
0–150 grid). First fix (`a2407adc`): the off bin excluded from the focus
weight; its rerun held through the collapse window (`real_macro` 5.8 →
5.2 → 4.7 over steps 2–4.5k, `real_cruise` 1.3–1.5). Second fix (Dmitrii,
same day, item 4 of the ladder): the linear grid goes back to the uniform
convention — no off class, bin 0 is the 0 rev/s rate bin and a stopped
rotor a Gaussian peak there, focus-weighted like any peak. **Pilot 4 on
that convention reproduced the collapse** (`rp_hppnet_l2_all` W&B
`pxdmmjuk`: 6.7 → 4.65 at 2.5k → 7.8 → 13.7 → 20.8 at 4.5k;
`rp_hppnet_l2_alllab` `t8o60jqx`: 4.1 at 2.5k → 18.4 at 4k, `real_zero`
1.6 and `real_cruise` 22 — "0 everywhere" again). Reading: on a log trunk
re-gridded to linear, bin 0 is still a pooled scalar (the L1 off logit is
the only path into it; bins 1–27 are unreachable), its target is still a
class on ~20 % of frames, and the ~65× focus on that class made "0" the
cheapest answer — the off state was never the issue, the weight was. The
published `hppnet_l2_r2_s0` survived because its conv stack over all bins
had no pooled scalar feeding bin 0. **Final fix (Dmitrii, 2026-10-15):**
keep the uniform linear convention and the pooled logit into bin 0 (an L2
model reads 0 or ≥ `fmin`, never in between), and leave bin 0 at weight 1
on every grid — the only configuration that empirically held (regression
test `test_bin_0_is_not_focus_weighted_on_any_grid`, both grids). The
no-schedule stream stays — it is what every earlier run and every pitch
paper did — but the ramp was not what broke HPPNet.

| transform | where | parameters | labels | class |
|---|---|---|---|---|
| `random_gain` | post-mix, p 0.5 | U(−6, +6) dB | kept | info-preserving |
| `spectral_recolor` | pre-mix, p 0.5 | ±8 dB smooth EQ, 10 log-spaced anchors | kept | mild |
| speech mixing | mix, p 1 | source-to-noise SNR U(−30, +5) dB, independent utterance per channel, `snr_ref_floor_rms` 0.02 | kept | strongly destroying at high SNR |
| `rate_warp` (`*_labelaug` only) | pre-mix, native rate, p 1 | α(t) = α₀·(c + a sin(2πft+φ)), α₀ ~ U(0.5, 1.5), c ~ U(1 ± 0.08), a ~ U(0, 0.04), f ~ U(0.1, 1) Hz | r̃(t) = α(t)·r(τ(t)) | info-preserving, label-altering |
| silence arm | source | weight 0.4 (0.2 on the single-drone streams) against the merged real 2.0 | zero | a class, not an augmentation |

Note on the plain streams (`rp_all`, `rp_dregon`, `rp_md2`): they carry **no
label-altering augmentation**, unlike `hppnet_l2_r2_s0` and every
`real_r*_unified` run, which trained with `freq_scale` α U(0.7, 1.3) at
p 1 (after a 50k-chunk unaugmented warm-up) plus `noise_time_warp` p 0.5.
The `*_labelaug` arms are therefore the comparison to those numbers; the
plain arms ask whether any label-altering augmentation is needed at all
(decision: Dmitrii, 2026-10-15).

Precedent for the label-preserving set (library, full text unless noted):

| ours | pitch / F0 tracking | separation / other audio |
|---|---|---|
| `random_gain` | PESTO U(−6, 3) dB; SwiftF0 U(−6, 6) dB; LPCNet-pitch U(−60, 10) dB | Demucs / Open-Unmix / MMDenseLSTM random scaling 0.25–1.25 per source (Uhlich 2017 recipe) |
| speech mixing (additive interference) | PESTO white noise + background mix; SwiftF0 CHiME/Gaussian at SNR U(10, 30) dB; LPCNet-pitch DEMAND; Airaksinen 2019 (PiENet) NOISEX at SNR U(−5, 15) dB | Demucs in-batch remix; WHAM!/WHAMR; MUSAN |
| `spectral_recolor` (random EQ) | LPCNet-pitch random 2nd-order IIR; Airaksinen 2019 random 17-tap FIR "channel augmentation" (reported to greatly increase noise robustness) | VTLP / channel simulation in ASR |
| `random_reverb` (dropped 2026-10-15) | — (none of the pitch papers) | RIR augmentation in ASR (Ko 2017), speaker verification (MUSAN + RIR), WHAMR |

Airaksinen 2019 also reports that its label-altering "diversity
augmentation" (vocoder resynthesis at a new mean F0 U(100, 350) Hz) did
nothing with 43k utterances but helped with 4k — the small-data regime is
ours (the real pool is ~7 min).

SNR semantics (`src/data_processing/mixing.py:196`): `snr_db` is the
*speech* level relative to the rotor noise; −30 dB is the easy end for RPS,
0…+5 dB the hard end. The upper bound +5 dB (vs 0 dB in every earlier
stream) extends the audible-speech end; the validation panel's MD2 mixes
sit at −5 and −20 dB, inside the range either way.

`rate_warp` replaces `freq_scale` + `noise_time_warp`: the warp is already a
local frequency scale with the same label transform
(`src/data_processing/time_warp.py:1-30`), so one time-varying resampling
covers both. It runs on the **native 44.1 kHz** recording and decimates to
16 kHz last; 44.1 kHz covers any local α ≥ 16/44.1 = 0.36, i.e. the
U(0.5, 1.5) range with the ±12 % warp on top (worst local 0.44), so no
silent band appears for α < 1. (Storing the α = 0.5 version at 16 kHz is
byte-identical to storing 32 kHz audio and has no headroom for the warp;
native rate is simpler.) Pool memory grows ~2.8× (~0.6 GB); the per-sample
soxr call on a 2 s × 8-ch window is milliseconds. The stream-window
oversampling (`freq_scale_source_factor`, warp margin) collapses to one
factor, 1/α_min·(1 + dev) + margin.

**`rate_warp` against the existing label-altering augmentations.** What
pitch work does to move the label: (1) integer bin shift of a
log-frequency input (SPICE, PESTO, Cwitkowitz self-supervised MPE) —
exact, but needs a log axis and is equivariance training rather than data
augmentation; not applicable to the pyramid's linear grid. (2)
Phase-vocoder pitch shift with duration preserved (Cuesta et al. 2020
multi-f0 vocals: −2…+2 semitones per voice then remixed; Salamon & Bello
2017; Demucs: 20 % of batches, pitch {−2…+2} semitones and tempo
U(0.88, 1.12) via Soundstretch). (3) Vocoder resynthesis at a scaled F0
(Airaksinen 2019) — speech only. (4) Resampling / speed perturbation (Ko
et al. 2015, factors 0.9/1.0/1.1; torchaudio `SpeedPerturbation`) — pitch
and tempo scale together; standard in ASR, not used in pitch papers.
`rate_warp` is (4) with a time-varying factor and a wide range: at constant
α it IS speed perturbation. For a rotating source it is the physically
exact transform — comb, floor, line widths and modulation sidebands scale
as a real rotor at α·r would produce — except for the room, whose RIR does
not scale; a phase-vocoder shift is the converse (room and time structure
kept, source approximated: phasiness, line widths and AM not scaled). The
time-varying part has no precedent in pitch work (nearest: SpecAugment's
feature-domain time warp, which changes no label).

Decision (Dmitrii, 2026-10-14): no ablation against the phase-vocoder or
narrow-range variants; `rate_warp` is used as is. Motivation stated in the
paper: it is simple (one resampling), rich (a continuum of rates plus slow
wander, not a few fixed semitone steps), and it destroys no structure of
this signal — the duration-preserving (vocoder) form exists for music
because tempo must not change, a constraint rotor noise does not have.
Scope of the "not used in pitch tracking" claims above: the neural pitch
estimators we build on and have read in full, plus the melody-extraction
review (arXiv 2202.01078) — not an exhaustive survey; the paper words it as
"not among the augmentations used by the estimators we build on".

Literature on scheduling (library, full text): no pitch-estimator paper
schedules augmentations — PESTO (noise σ U(0.1, 2) + gain U(−6, 3) dB at
p 0.7), SwiftF0, LPCNet-pitch, Airaksinen 2019, FCNF0++ (none). Range-bound
schedules of an online augmentation exist in neighbouring tasks: ACCAN
(Braun, Neil & Liu, EUSIPCO 2017, arXiv 1606.06864) stages the SNR range of
per-epoch noise mixing (start at 0 dB, widen to 50 dB) and reports up to
−31.4 % WER against multi-condition training (the fixed range from step 0)
— note the direction is hard → easy; PS-SapAug (arXiv 2412.00415) ramps the
SpecAugment probability over epochs plus per-sample loss-adaptive strength,
up to −8.1 % relative WER over fixed SpecAugment; SNR-decaying curriculum
for speaker verification (arXiv 2510.18533), easy → hard. In this project,
G5 (`noise_augmentations.py` docstring; `data_processing/AGENTS.md`) found
augmentations from sample 0 regressed a regressor's validation 63.7 → 117.9
(MSE) against a 50k unaugmented warm-up — the easy → hard argument the
first draft of this protocol took, with cosine ramps of the speech SNR
upper bound (−30 → +5 dB over 10–35 %) and the reverb DRR. The HPPNet-L2
pilot showed the opposite failure on the salience side (above): the easy
regime is where a salience net memorises the 7-minute pool. Multi-condition
training from step 0 is what every earlier run of this project and every
pitch paper did, and is what the protocol does; no schedule ablation.
Whether recolor earns its place is a second ablation (`{gain, speech}`
only) if we want to claim it.

## Models, output grid and the adaptation ladder

**Output grid (decided 2026-10-14): 0–160 rev/s, 164 bins at the pyramid's
native 0.977 rev/s step**, for every per-rotor model including HRSPNet.
Measured maxima of the real data (published frames, rotors > 5 rev/s):
FLY125 100.8 (p99.9 97.1), FLY124 99.5, DREGON room 1 measured ≤ 92.3,
room 2 commanded ≤ 90.0 rev/s; at `rate_warp` α = 1.5 the tail reaches
151, so the old L2 grid (0–150) would clip it and HRSPNet's 0–300 (a
by-product of `level0_fmax` 1200 Hz ÷ 4 harmonics, not a data choice) is
twice what is needed. 164 states halve the CRF against today's 307. The
Gaussian target width (rev/s), the CRF readout with its off state and the
BCE loss are shared by every per-rotor model, so rows differ only in the
trunk.

**Ladder for the four multi-pitch baselines** (HPPNet, HarmoF0, Basic
Pitch, LateDeep), replacing Block S of `paper-regime-matrix.md`
(L1-as-resolution-only was never run; L3 is retired):

| level | content | grid | readout |
|---|---|---|---|
| L0 | the published architecture, trained on our data: own log-frequency map (352 bins @ 48/oct from 27.5), own harmonic device, seam deviations as documented (`freq_pool: 1`, logits, hop-512 frames) | log | threshold + Hungarian |
| L1 | **four per-rotor maps on the model's own log grid** — the last layer emits 4 maps instead of 1; nothing else changes | log | CRF (log-domain sub-bin interpolation, off state for a stopped rotor — the log grid has no zero bin) |
| L2 | L1 + a **minimal translation head** to the linear 0–160 grid: the fixed interpolation matrix W (log → linear, 0 parameters) followed by ONE depthwise linear convolution along frequency per map (kernel 9, no normalisation, no activation, no cross-map mixing; 4 × 10 parameters) | linear 0–160 | CRF |

L0 → L1 isolates the per-rotor encoding principle (the large effect: the
encode–decode floor of a shared map is a crossing floor, 8.2 rev/s on
DREGON); L1 → L2 isolates the target grid. The head cannot add resolution
the trunk's grid lacks — that is what HRSPNet's linear front end is for —
which is why it is kept minimal; today's `FreqSuperResHead` (2 × Conv(5×1)
→ 32 ch → BN → ReLU, first layer mixing the maps, ~6k parameters) is
replaced.

What the old head actually does, measured (`scripts/_probe_l2_head.py` at
commit eb62e8d8, removed with `FreqSuperResHead` on 2026-10-14;
`hppnet_l2_r2_s0` `best.ckpt`, the frozen real split at mic 0,
rotors-on frames): full model PIT-MAE 1.92 rev/s; with the head's conv
stack removed (W interpolation only, same CRF) **27.5**; the trunk's own
maps read by argmax **203** — they are saturated (sigmoid median 0.94,
half-max width 253 of 352 bins), i.e. not a salience map but a code only
this net decodes. The net's output peak is 8 rev/s wide at half max;
flattening the other three maps moves a rotor's output by 3.5 rev/s
(identity is partly decided in the head); and an ideal one-bin input at
65/80/95 rev/s decodes to 0.0 rev/s — the net is tuned to the trunk's
code, not to peaks. So the published L2 number belonged to trunk + decoder
jointly and could not separate "per-rotor encoding" from "this 6k-parameter
decoder"; the minimal head below forces the trunk to express the map.

Translation head, definition:
1. **Logit domain.** Input and output are logits (BCE-with-logits, as every
   per-rotor model); the map is non-negative after the sigmoid. No ReLU:
   a ReLU on logits would floor p at 0.5 everywhere.
2. **Resampling matrix W** (fixed): per output bin, the mean of the log
   bins inside ±0.49 rev/s where the log grid is finer (below ≈ 67 rev/s,
   where a log bin = 1.45 % of the rate < 0.977), linear interpolation
   where it is coarser. Rows sum to 1 (logit scale preserved); no log bin
   is skipped (today's point-sampling `_freq_interp_matrix` skips bins
   below 67 rev/s).
3. **Below the trunk's grid.** Output bins whose ±half-bin window holds no
   log bin (those in (0, ~27.3) rev/s for CQT `fmin` 27.5) have zero W rows
   AND a fixed, non-learned logit of −10 after the conv (a zero logit would
   be p = 0.5 there, an unlearnable BCE term the CRF could prefer on a
   weak frame); the mask is the head's `unreachable` buffer. A rotor
   passing 1–27 rev/s is unrepresentable by L0/L1/L2 — a stated limitation
   of the published grids.
4. **Zero on the linear grid is a rate; on a log grid it is a state; bin 0
   is never focus-weighted** (Dmitrii, 2026-10-15, replacing the 2026-10-14
   "one off state everywhere"). The linear 0–160 grid keeps the uniform
   convention of every earlier linear-grid run — bin 0 is the 0 rev/s rate
   bin, a stopped rotor is a Gaussian peak at 0, the CRF and readout treat
   it like any rate — because that is what keeps the linear grid simple;
   it applies to L2, HRSPNet, the pyramid and the plain ports. A log grid
   has no zero rate, so L1 emits ``G_log + 1`` bins with an OFF state at
   bin 0 (hard target 1 on stopped frames, free transitions, read as
   exactly 0.0), fed by one per-map off logit (a linear read of the trunk's
   frequency-pooled last features). L2's translation head routes that
   logit into linear bin 0, where it is trained as the 0 rev/s rate logit —
   a log trunk has no bins below its `fmin`, so the pooled logit is the only
   thing that can say "0 rev/s" there, and an L2 model reads **either 0 or a
   rate ≥ `fmin`** (~27.5 rev/s), never in between (test
   `test_l2_reads_zero_or_at_least_the_trunk_s_lowest_rate`). The flag is
   the grid kind (`grid_from_spec` → `has_off`; `linear` False, `log_off`
   True), carried by the model as `out_has_off`. On EVERY grid the loss
   leaves bin 0 at weight 1 (`layer_pit_bce`): its positive is a class
   present on ~20 % of training frames, not a sparse peak, and the ~65×
   focus there is what collapsed the HPPNet-L2 pilots (post-mortem below).
5. **Depthwise linear conv**, kernel 9 (±4 rev/s), identity-initialised
   (centre tap 1, bias 0): L2 starts as L1 re-gridded; the only thing it can
   learn is a 9-tap sharpening kernel per rotor.

Attainable precision is a fraction of the TRUNK's bin (1.45 % of the rate:
1.16 rev/s at 80, 2.3 at 160), via mass split across adjacent bins and the
CRF's sub-bin readout; the head neither adds nor removes any. At cruise a
log bin is comparable to the 0.977 linear bin, so L1 → L2 is expected to be
small — the precision HRSPNet buys comes from its front end (linear-STFT
levels 4.5–9× finer than the CQT at high k), not from the grid.

**HPPNet chain to HRSPNet**, one addition per step, all on the 0–160 grid.
Parameter counts from the registry builds (2026-10-14; trunk blocks
`block_4…8` are identical everywhere, 49 k + 49 k + 3 × 82 k):

| step | adds | params |
|---|---|---|
| HPPNet‑L2 | per-rotor maps + translation head on the published CQT trunk, k ≤ 8, FreqGroupLSTM (100 k) | ≈ 0.52 M |
| +k84 | HDC harmonic range k ≤ 84 on the CQT (one `Conv2d(16→128, 1×3)` per harmonic: 6.1 k each, 520 k for 84) | ≈ 0.99 M |
| +ConvLSTM | **separable** rate ConvLSTM (hidden 64, kernel 33, bidirectional; per-channel 33-tap displacement kernel + 1 × 1 mix ≈ 37 k) in place of FreqGroupLSTM | ≈ 0.93 M |
| +pyramid = **HRSPNet** | multi-resolution STFT pyramid (16384/8192/4096/2048) in place of the CQT; HPDC k ≤ 84 + sub-harmonics 1/2…1/8 as ONE gather with a 1-tap mix per harmonic (`(91, 128, 16)` = 186 k; HDC's 3-tap kernel is 3× that), **no frequency ceiling on the taps**; no translation head (the output grid is level 0's own) | ≈ 0.59 M |

Decisions (2026-10-14): (a) **HRSPNet is k ≤ 84 with no tap ceiling.** The
paper draft's "taps read below 4.8 kHz" was an eval-time patch for a data
artefact (the k ≈ 70 motor line belongs to a different rotor in room 2 than
in room 1, and room 2's commanded labels put it 10+ bins off on the 7.8 Hz
top level; the CQT suffers less only because the same error is 1.4 of its
bins) and is dropped: masking input frequencies is not architecture. The
tap-zeroing stays in the paper as the eval-time DIAGNOSTIC that explains a
DREGON deficit; the remedies are on the data side (ConvLSTM, labelaug,
later synthetic pretraining), which the grid exercises. One extra run,
HRSPNet at k ≤ 64 on `all`, makes the harmonic-range axis a curve
(8 → 84) instead of a single point. The 2.17 rev/s run
(`real_r4_hppnet_pyrlstm_unified`) used `k_max 32` / `f_max 7500` and the
dense ConvLSTM, so the protocol's HRSPNet is a new configuration, not that
checkpoint. (b) The chain uses the **separable** ConvLSTM: the dense form
(`Conv1d(64 → 256, k = 33)` per direction = 1.08 M, 72 % of the old model
and 11× the FreqGroupLSTM it replaces) made "+ConvLSTM" also a ×3
parameter step; the separable form keeps the mechanism (state moves along
the rate axis) at the baseline's size. One dense-ConvLSTM HRSPNet run on
`all` is the control; if it wins clearly, the paper reports both and says
why. (c) Every model table in the paper carries the parameter count.

**Zero-path pair (Dmitrii, 2026-10-15; `rp_pyrsep_{in,bn}_all`).** The
first seven baseline cells put a number on the HPPNet family's zero
problem: at L1 HPPNet's `real_cruise` is the best of any row (1.56; L2 1.32)
and its `real_zero` the worst (9.5; L2 16.6) against HarmoF0's 0.5. Cause,
read off the ports: every HPPNet block ends in a non-affine
`InstanceNorm2d`, so from `block_1` on every feature is per-clip and
gain-invariant — an all-stopped clip is normalised to the same scale as a
cruise clip, the frequency-pooled off logit (L1/L2) sees contrast only, and
the pyramid's 0 rev/s bin (whose harmonic taps all read DC and are masked)
can only be a "nothing else fired" default. HarmoF0's blocks end in
`BatchNorm2d` (dataset statistics, affine): absolute level survives to the
off path, and "silent" is a linear threshold. The published `hppnet_l2_r2_s0`
only escaped because its deleted `FreqSuperResHead` clamped the bins below
27.5 rev/s to copies of the trunk's lowest bin and ran a BatchNorm'd conv
stack over them — a level read smuggled in on the head's own path. The pair
tests the mechanism on the HRSPNet line without a zero head: the pyramid
with `block_4`/`block_5` dropped (the harmonic gather already reads `r/k`)
and a **separable rate kernel** head (`head: sepconv`: the ConvLSTM's 33-tap
per-channel kernel + 1 × 1 mix on the CURRENT frame, no recurrence, ≈ 21 k
params; ≈ 0.48 M total), `norm: instance` vs `norm: batch`, everything else
HRSPNet's. Expected: batch norm much better on `real_zero` at equal cruise.
If it is, the norm becomes a chain step for the whole HPPNet family (L0–L2
rows rerun with it), and the ConvLSTM row is then the recurrence-only
ablation against `sepconv`.

**Regressors**: the published SimpleConv (Quiet Drones 2026) and its best
modification (the `all` results name it; Conv+BiGRU and SCv2 are the
candidates from `paper-regime-matrix.md`), unchanged, on the same frame
grid.

**Panel re-scores of earlier checkpoints (2026-10-16 → 22;
`scripts/rp_panel_eval.py`, JSONs in `results/rp_panel_eval/`).** Every
trained checkpoint, old or new, can be scored on the rp_real panel through
`zoo.load` + the loop's own `validate_rps`; the old pyramid/ConvLSTM builds
are frozen as `hppnet_pyramid{,_k84}_r4legacy` and the retired
`FreqSuperResHead` as `hppnet_orig_superres_legacy` / `hppnet_l2_r2legacy`
(`real_r4_hppnet_l2_unified`'s checkpoints were deleted in the R2 cleanup and
cannot be scored). Correction (2026-10-18): the old `real_r4_*_unified` and
`hppnet_l2_r2_s0` runs trained on the same recording selection as `rp_all`
(`in_flight_noise` minus room 1, FLY125) and were validated on room 1 /
FLY124 — **held out at recording level like the new runs**; their caveats
are label-altering augmentation (`freq_scale` + `noise_time_warp`),
selection on the pre-calibration MD2 labels, and the retired regime's LR /
selection-signal defects — not leakage. Findings so far (best-median
checkpoints, `real_macro` on rp_real; per-view rows in the JSONs):

| row | real_macro | zero | standby | ramp | cruise | note |
|---|---:|---:|---:|---:|---:|---|
| old conv-only pyrk84 (labelaug) | 2.86 | 2.54 | 3.92 | 8.11 | 2.03 | |
| ↳ taps k ≥ 61 zeroed at eval | 2.47 | 2.11 | 4.09 | 7.85 | 1.57 | all of it DREGON cruise 3.48 → 2.52 |
| old dense ConvLSTM pyrlstm (labelaug, 1.6 M) | 2.57 | 1.24 | 4.12 | 5.71 | 1.83 | the old 2.17 was the pre-calibration panel |
| old HPPNet L2 + FreqSuperResHead (`hppnet_l2_r2_s0`) | 2.62 | 0.35 | 5.29 | 12.20 | 1.40 | the frozen-split 2.27 was a panel effect of ≈ 0.35 |
| `rp_pyrsep_in_all` (sepconv, instance norm) | 2.59 | 1.83 | 4.25 | 9.93 | 1.57 | |
| `rp_pyrsep_bn_all` (batch norm) | 3.17 | 3.14 | 8.28 | 10.78 | 1.25 | batch norm refuted: standby collapses |
| `rp_pyrseplstm_all` (separable ConvLSTM) | 3.01 | 1.31 | 4.98 | 9.89 | 1.91 | 8 s vs 2 s windows identical (3.01 / 3.03): no length mismatch |
| ↳ taps k ≥ 61 zeroed | 37.8 | — | — | — | — | plain-stream models USE the top taps |
| `rp_pyrseplstm_alllab` | 2.74 | 2.27 | 3.95 | 6.34 | 1.96 | ramps 5.0 on DREGON — recurrence + moving labels |
| ↳ taps k ≥ 61 zeroed | 2.55 | 3.27 | 4.93 | 6.60 | 1.41 | labelaug models gain from the knock-out, like the old ones |
| `rp_harmof0_l2cqt_all` (HarmoF0 trunk, CQT front end) | 2.56 | 1.50 | 4.79 | 9.52 | 1.47 | front end alone: STFT 2.62 → CQT 2.56, all md2 cruise |
| old CQT HPPNet L2 + dense ConvLSTM (`real_r4_hppnet_l2convlstm_unified`, labelaug, k ≤ 9) | 2.59 | 0.17 | 4.73 | 7.95 | 1.71 | frozen build `hppnet_l2_convlstm_r4legacy`; dregon 3.90, md2 1.29, DREGON cruise 2.80 (2026-10-22) |
| `rp_pyrseplstm_k64_alllab` (separable ConvLSTM, k ≤ 64 at training) | 2.52 | 2.08 | 3.65 | 6.12 | 1.76 | dregon 3.89, md2 1.14: both drones gain over k ≤ 84 (2.74), unlike the knock-out |
| `rp_pyrseplstm_k64_all` (k ≤ 64, plain stream) | 2.81 | 4.56 | 5.45 | 10.78 | 1.30 | dregon 3.50, md2 2.12; best median at step 5k, saturation stop at 25k (raw low 2.57); k ≤ 84 twin 3.01 (2026-10-22) |

Readings: (1) held-out plain-stream rows cluster at 2.56–2.65 across four
families; no architecture is ahead by more than the ±0.3 round scatter.
(2) The eval-time tap knock-out helps every labelaug-trained model and
destroys every plain-stream one: with `rate_warp` the high harmonics are
unreliable as a comb and the model leans on k < 61; on the plain stream it
builds on them. The knock-out is therefore not a transferable diagnostic;
the k ≤ 64 arm (trained without the taps) is the honest test, on both
streams. (3) Recurrence owns DREGON ramps (5.0–5.1 vs 8–12 for every
stateless head) only when the stream has moving rotors; on steady DREGON
cruise it costs ≈ 0.8 rev/s against the same-frame kernel. (4) The HPPNet
trunk's zero problem was the pooled off logit, not instance norm: bin 0 as a
map default with same-frame neighbours (sepconv) or the old clamp-copy +
BatchNorm head both solve it; batch norm in the trunk does not.

## Experiment grid (32 runs, one seed, up to 100k steps each)

`rp_<model>_<stream>[_<ablation>]`; streams `all`, `alllab` (= `all-labelaug`),
`dregon`, `md2`, `dregonlab`/`md2lab`. Not runs: the blind two-stage tracker
and the encode–decode floor (training-free).

| claim | arms | runs |
|---|---|---|
| 1. unadapted salience models fail | L0 of HPPNet, HarmoF0, Basic Pitch, LateDeep on `all` | 4 |
| 2. HPPNet best up to L2; L0→L1 vs L1→L2 attribution | L1 and L2 of all four on `all` | 8 |
| 3. HRSPNet > HPPNet‑L2, with each step attributed | `+k84`, `+k84+ConvLSTM`, HRSPNet on `all`; dense-ConvLSTM HRSPNet control; HRSPNet k ≤ 64 (harmonic-range curve) | 5 |
| regressor row | SimpleConv, best modification on `all` | 2 |
| label-altering augmentation | `alllab` for HRSPNet, HPPNet‑L2, best regressor | 3 |
| 4. schedule (on HRSPNet only) | `alllab`: everything at full strength from step 0; `{gain, speech}` only | 2 |
| 5. no cross-drone transfer | HRSPNet, HPPNet‑L2, best regressor on `dregon` and on `md2` | 6 |
| 6. labelaug does not buy transfer | `<direction>lab` for HRSPNet and HPPNet‑L2; the direction with the nominally better transfer in claim 5 | 2 |

Dependencies: claims 1–4 and the HRSPNet transfer pair start at once; the
HPPNet‑L2 / regressor transfer runs need only the regressor's name; claim 6
waits for claim 5. Not in the grid (possible later ablations): the other
two baselines at L2 on `alllab`, `dregonlab` + `md2lab` for the regressor,
a second seed anywhere — case by case, if a one-seed gap is inconclusive.
The schedule verdict rests on one model and one seed; the paper says "on
HRSPNet", not "overall", unless the gaps are large.

Each `conf/experiment/rp_*.yaml` gets its sibling `.md` with
`batch: paper-real-protocol.md`.

### Paper grid after the experiments rewrite (2026-10-23): pending approval

Nothing below is submitted until Dmitrii approves the grid. Section VI of
the paper (submodule `writing/papers/2026-08_wrapup`) was rewritten on
2026-10-23. HRSPNet in the paper is now `hppnet_pyrseplstm_k64` (k ≤ 64);
k ≤ 84 (`rp_pyrseplstm_{all,alllab}`) is an ablation row. Finished runs
supply every number in the output-level, real-data and chain tables
(`scripts/rp_history.py --best`). The remaining `\wip` cells on the real
streams need these runs:

| paper cell | run | model config | stream | state |
|---|---|---|---|---|
| chain table, `+ harmonics k ≤ 64` | `rp_hppnet_l2k64_{all,alllab}` | `hppnet_l2k64` (new: `hppnet_l2k84` with `harmonic_k_max: 64`, 385 CQT bins, 0.87M) | `all` / `alllab` | config to write |
| chain table, `+ ConvLSTM` | `rp_hppnet_l2k64_convlstm_{all,alllab}` | `hppnet_l2k64_convlstm` (new: `hppnet_l2k84_convlstm` with `harmonic_k_max: 64`, 0.87M) | `all` / `alllab` | config to write |
| transfer table, HRSPNet | `rp_pyrseplstm_k64_{dregonlab,md2lab}` | `hppnet_pyrseplstm_k64` | `dregonlab` / `md2lab` | submitted to `uni` 2026-10-23 (Dmitrii) |
| transfer table, Conv+BiGRU | `rp_scv2_{dregon,md2}` | `hb_scv2_mag_nogate` | `dregon` / `md2` | submitted to `uni` 2026-10-23 (Dmitrii) |

The four transfer runs stop on the training drone's score alone
(`validation.control: [dregon]` or `[md2]`), so the held-out drone never
steers the run. The transfer cell is the held-out drone's score (all, cruise)
at the round with the best five-round median of the training drone.

That is 8 training runs on `uni`. The finished siblings took 1.4–4.7 GPU-h
each: `rp_pyrseplstm_k64_alllab` 4.1 h to step 32 000, `rp_hppnet_l2_all`
2.8 h, `rp_scv2_all` 3.2 h.

Evaluation only, on saved checkpoints (no training):

- **Outputs figure.** One DREGON room-1 clip with a take-off and one FLY124
  clip with close rotors at cruise. Models: HRSPNet `alllab`, HarmoF0-L2
  `all`, Conv+BiGRU `all`. Tool: `scripts/rp_panel_eval.py` with `--dump`.
- **Frequency-scaling probe figure.** `freq_probe_panels.pdf` regenerated
  with the checkpoints of the real-data and pretraining tables.
- **Discussion.** The line-zeroing probe (taps above 4.8 kHz zeroed, DREGON
  validation) on `rp_pyrseplstm_alllab` (k ≤ 84).
- **Synthetic-data section, parity check.** The best HRSPNet scored on the
  renders at the validation labels and on the recordings, per drone.

Second seed (the `\wip` in the limits paragraph of the Discussion): HRSPNet
`all` / `alllab`, and the fine-tuned HRSPNet. The project rule is one seed
unless asked, so these are listed, not proposed.

Configs which the rewrite leaves out of the paper are kept but not run:

- `rp_hppnet_l2_{dregon,md2,dregonlab,md2lab}`: the transfer table has
  HRSPNet and Conv+BiGRU only.
- Claim 4 (schedule) and the dense-ConvLSTM control.

The synthetic pretraining grid is in `rp-synthetic.md`, § "Paper grid".

## Implementation notes (for the cutover)

- `conf/validation/`: new panel config (datasets = the new derived split;
  views by meta predicates, not index ranges; 22 scores; `control: real_macro`);
  the controller loses its LR branch; validation emits no `_log` scores and
  no per-clip `_clips`/`_clip_mean`/`_clip_std` (dropped 2026-10-15; the
  SCv2 pilot still logs them — reads in `val_samples/` instead).
- `src/training/loop.py`: step-based warm-up + cosine; remove
  `ReduceLROnPlateau` and the epoch-level `patience` counter (the
  controller's saturation stop stays); dual best checkpoints per score;
  drop `best.ckpt`/`optim.monitor`; scheduler state in `train_state.pt`.
- `src/models/salience_rps.py`: replace `FreqSuperResHead` by the minimal
  translation head (W + depthwise linear conv); `n_maps: 4` on the native
  log grid for L1 (`out_freqs` unset); CRF readout on a log grid with an
  off state; every per-rotor config on the 0–160 / 164-bin grid (pyramid:
  truncate `out_bins`).
- `src/data_processing/`: `rate_warp` (native-rate pool, one resampling,
  label transform), severity ramps per augmentation (today only
  *probability* ramps exist, `online_mixing.py:131-152`), `test` split in
  `sources/dregon.py`, new validation derivation with meta and the MD2
  dev-clean mixing.
- `scripts/`: regime decomposition and per-SNR readout from saved per-clip
  predictions.

## Results

Not run.

## Conclusion

Pending.
