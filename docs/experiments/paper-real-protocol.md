# Paper real-data protocol (v2): streams, schedule, panel, checkpoints

**Status:** design (2026-10-14 –) · no run started · supersedes the "unified
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
   is never stepped in that branch (`src/training/loop.py:893-895`). Stopping
   was also validation-driven ("saturation", `final_patience 12`). Runs were
   therefore not comparable in budget and the selection signal leaked into
   the optimisation.
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

Budget in optimizer steps: **100 000** (today's "200 epochs" is exactly 200
validation rounds × 500 steps at the step cap). AdamW, lr 1e-3, weight decay
1e-4, batch 64 × grad-accum 2 (128 mono frames = 16 chunks × 8 mics), AMP,
grad-clip 5 — unchanged. Schedule: linear warm-up 0 → 1e-3 over the first 5 %
(5 000 steps), then cosine to 1e-5 at 100 %. **No early stopping, no plateau
logic, no validation-driven LR**: the `MultiMetricController` keeps only its
median smoothing for checkpoint selection; `lr_patience`, `lr_factor`,
`min_lr_reductions`, `final_patience`, `min_rounds` and the plateau
scheduler are removed. Scheduler state goes into `train_state.pt` so a
resumed run continues the cosine.

Precedent: PESTO trains 50 epochs under cosine annealing (Adam 1e-4);
FCNF0++ omits early stopping and trains a fixed 250k steps (library full
texts, 2026-10-14).

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

Ablation arms on the `all` stream, same budget: **A** `rate_warp` as above
(ours); **B** phase-vocoder pitch shift (pyrubberband/librosa), same α
range, duration preserved, labels ×α — the existing method at our range;
**C** (only if A ≈ B) A at the literature's range, ±2 semitones
(α 0.89–1.12), to separate range from mechanism.

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

## Experiment matrix — TBD

Models and per-model arms are decided after the augmentation section is
revised. Each experiment config gets its sibling `.md` with `batch:
paper-real-protocol.md`.

## Implementation notes (for the cutover)

- `conf/validation/`: new panel config (datasets = the new derived split;
  views by meta predicates, not index ranges; 15 scores; `control: real`);
  the controller loses its LR branch; validation emits no `_log` scores.
- `src/training/loop.py`: step-based warm-up + cosine; remove
  `ReduceLROnPlateau`, `patience`/early stop; dual best checkpoints per score;
  drop `best.ckpt`/`optim.monitor`; scheduler state in `train_state.pt`.
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
