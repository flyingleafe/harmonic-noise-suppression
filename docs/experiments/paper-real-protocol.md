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
| totals | `dregon` (22 clips), `md2` (30), `real` = ½ `dregon` + ½ `md2` — the **control** |
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
per family (below), weight decay 1e-4, **effective batch 128** mono frames
(16 chunks × 8 mics), micro-batch 128 for every arm unless a one-batch
forward/backward memory probe says otherwise — then 64 × grad-accum 2, a
memory knob, not a protocol parameter. Probe outcome (the range tests,
2026-10-15): **HPPNet** (L0 and L2; the `FreqGroupLSTM` over 352 bins
asks for a further 10.9 GiB on top of 10.6 GiB at 16 GB and 24.9 GiB at
32 GB) runs at **64 × 2** for every rp_hppnet_* arm; HarmoF0 L2 ran at
128 on the same cards. The pyramid arms are probed when they are
configured (the historical 64 × 2 came from `97b25a89`, "to fit any
backend", with the 307-bin grid and the dense ConvLSTM; the 164-bin grid
and the separable ConvLSTM halve the post-gather activations). The two
BatchNorm models (Basic Pitch, LateDeep) always run at 128. AMP,
grad-clip 5 — unchanged. Schedule:
linear warm-up 0 → max over the first 5 % (5 000 steps), then cosine to
1 % of max at 100 000 steps, a pure function of the step count. **The LR
no longer depends on
validation**: the `MultiMetricController` loses its LR branch
(`lr_patience`, `lr_factor`, `min_lr_reductions`) and the plateau
scheduler goes; it keeps median smoothing for checkpoint selection and the
**saturation early stop** (`min_rounds`, `final_patience` on the control
score `real`), which stays. Scheduler state goes into `train_state.pt` so
a resumed run continues the cosine.

Stop parameters (Dmitrii, 2026-10-14): **`min_rounds` 40** (20 000 steps)
and **`final_patience` 30** (15 000 steps) — larger than today's 30 / 12
because the LR-patience rounds no longer accumulate before a stop. The
earliest possible stop is therefore 20k + 15k = 35k steps, exactly where
the last augmentation ramp ends, so the saturation test never runs on a
moving training distribution (schedule plot: `scripts/_sched_plot.py`).
An arm that stops early never reaches the cosine tail, so arms differ in
realised budget; every table reports the stopping step next to the score.

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

Every round writes `last.ckpt`. For each of the 15 scores, two bests:
`best_<score>_median.ckpt` (5-round median improves, ≥ 1 % relative — what
`stable_improvements` does today) and `best_<score>_raw.ckpt` (raw round
value improves). All are hard links to `last.ckpt`
(`src/training/loop.py:311-316`), so coincident bests share one inode;
`best_checkpoints.json` records score → {step, round, raw, median};
R2 uploads use `alias_checkpoint` (verify it copies server-side). The
legacy single `best.ckpt` and `optim.monitor` go away. Which checkpoint the
paper reports is decided after the runs; `last.ckpt` is also reported
(selection-free).

### Augmentations and their schedule

Selection rule (Dmitrii, 2026-10-14): a label-preserving augmentation is
in the recipe only if it is in common use for pitch tracking AND for other
audio tasks (separation, ASR); the label-altering one is compared against
the existing pitch-tracking alternative. Dropped on that rule:
`random_polarity` (only magnitude front ends are considered now; a no-op),
`tooth_dropout` and `spec_mask` (masking is an ASR/SED convention — no
pitch-tracking paper in the library uses it), `floor_inject` (duplicates
the drone's own floor and the speech mix). "Full strength" = the
parameters listed.

| transform | where | parameters | labels | class | schedule |
|---|---|---|---|---|---|
| `random_gain` | post-mix, p 0.5 | U(−6, +6) dB | kept | info-preserving | full from step 0 |
| `spectral_recolor` | pre-mix | ±8 dB smooth EQ, 10 log-spaced anchors | kept | mild | full from step 0 |
| `random_reverb` | pre-mix | RT60 U(0.1, 0.8) s, DRR U(3, 15) dB, per-channel diffuse tails | kept | mildly destroying (smears transients, not stationary lines) | DRR lower bound ramps 15 → 3 dB over 5–20 % |
| speech mixing | mix, p 1 | source-to-noise SNR | kept | strongly destroying at high SNR | lower bound fixed at −30 dB; **upper bound ramps −30 → +5 dB** over 10–35 % |
| `rate_warp` (`all-labelaug` only) | pre-mix, native rate | α(t) = α₀·(c + a sin(2πft+φ)), α₀ ~ U(0.5, 1.5), c ~ U(1 ± 0.08), a ~ U(0, 0.04), f ~ U(0.1, 1) Hz | r̃(t) = α(t)·r(τ(t)) | info-preserving, label-altering | full from step 0 |
| silence arm | source | weight 0.4 | zero | a class, not an augmentation | constant |

Precedent for the label-preserving set (library, full text unless noted):

| ours | pitch / F0 tracking | separation / other audio |
|---|---|---|
| `random_gain` | PESTO U(−6, 3) dB; SwiftF0 U(−6, 6) dB; LPCNet-pitch U(−60, 10) dB | Demucs / Open-Unmix / MMDenseLSTM random scaling 0.25–1.25 per source (Uhlich 2017 recipe) |
| speech mixing (additive interference) | PESTO white noise + background mix; SwiftF0 CHiME/Gaussian at SNR U(10, 30) dB; LPCNet-pitch DEMAND; Airaksinen 2019 (PiENet) NOISEX at SNR U(−5, 15) dB | Demucs in-batch remix; WHAM!/WHAMR; MUSAN |
| `spectral_recolor` (random EQ) | LPCNet-pitch random 2nd-order IIR; Airaksinen 2019 random 17-tap FIR "channel augmentation" (reported to greatly increase noise robustness) | VTLP / channel simulation in ASR |
| `random_reverb` | — (none of the pitch papers) | RIR augmentation in ASR (Ko 2017), speaker verification (MUSAN + RIR), WHAMR |

Reverb is the one with precedent only outside pitch work; the paper says
so. Airaksinen 2019 also reports that its label-altering "diversity
augmentation" (vocoder resynthesis at a new mean F0 U(100, 350) Hz) did
nothing with 43k utterances but helped with 4k — the small-data regime is
ours (the real pool is ~7 min).

SNR semantics (`src/data_processing/mixing.py:196`): `snr_db` is the
*speech* level relative to the rotor noise; −30 dB is the easy end for RPS,
0…+5 dB the hard end. Ramps are cosine in the severity parameter, not in
probability; done by 35 % of training; the full distribution stays active
through the cosine LR tail.

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
(MSE) against a 50k unaugmented warm-up; that is the easy → hard argument
kept here: the hard regime (strong speech) is where prior-regression wins.

The schedule is justified by one ablation: `all-labelaug` with every
transform at full strength from step 0 (= multi-condition training), same
budget. Whether recolor and reverb earn their place is a second ablation
(`{gain, speech}` only) if we want to claim them.

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
4. **Zero bin = off state.** L1 needs a per-map off logit for its CRF (a
   linear read of the trunk's frequency-pooled last features, C → 1 per
   map); L2 routes it into bin 0 (identity). "Off" then means the same on
   every model and the CRF is one implementation.
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

**Regressors**: the published SimpleConv (Quiet Drones 2026) and its best
modification (the `all` results name it; Conv+BiGRU and SCv2 are the
candidates from `paper-regime-matrix.md`), unchanged, on the same frame
grid.

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

## Implementation notes (for the cutover)

- `conf/validation/`: new panel config (datasets = the new derived split;
  views by meta predicates, not index ranges; 15 scores; `control: real`);
  the controller loses its LR branch; validation emits no `_log` scores.
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
