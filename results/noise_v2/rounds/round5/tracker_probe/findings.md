# R5 probe (b): what the RPS trackers respond to on DREGON-like data

Runner: `scripts/noise_v2_tracker_probe.py` (`tracks`, `cqt`). Renders are R4's
own route at seed 2001, 8 microphones, on the three frozen cruise score
windows; scoring is the frozen path (`_synthetic_probe.score` through
`revised_eval.pit_mae`). Every HPPNet number below reproduces
`round4/legacy_truth/score_legacy_fit.json` to the digit and every SCv2 number
for `real`/`legacy`/`v2_real` reproduces `round3/dregon_humps/widen_scv2.json`
to the digit, which is the bit-identity check on the renders.

Data: `tracks.json`, `cqt.json`. Figures: `tracks_hppnet.png`,
`tracks_scv2.png`, `cqt_contrast.png`.

## 1. Both trackers, five arms: the failure is an ON/OFF verdict, not mistracking

3-window mean PIT MAE (rev/s), 8 mics. The two SCv2 columns marked **new** are
this round's; the rest are re-derivations.

| arm | HPPNet | SCv2 |
|---|---:|---:|
| `real` — the DREGON room-2 clip | 1.074 | 1.614 |
| `legacy` — legacy stage-2 render | 1.963 | 1.594 |
| `v2_real` — v2 fitted to the REAL clip (R3, `flight_floor_lowk`) | 69.555 | 11.297 |
| `v2_legacy_lowk` — v2 fitted to the LEGACY render (`flight_floor_lowk`) | 5.560 | **1.960** |
| `v2_legacy_free` — v2 fitted to the LEGACY render (`flight`) | 1.746 | **1.798** |

SCv2 per window (free-flight / hovering / updown): `v2_legacy_lowk` 1.817 /
1.890 / 2.172; `v2_legacy_free` 1.470 / 1.691 / 2.234. **SCv2 ratifies R4's
verdict on a second, architecturally unrelated tracker**: v2 fitted to the
legacy render passes on both trackers (1.96 / 1.80 against the legacy render's
own 1.59 and the real clip's 1.61), and the mode that frees the profile is the
better of the two on both. The R3 `flight_floor_lowk` gap that HPPNet reads as
5.56 SCv2 reads as 1.96 — SCv2 is the less brittle instrument, but it agrees
on the ordering.

### The decomposition

Per frame the four tracks are sorted; the CENTRE is their mean and the three
adjacent differences are the comb's GAPS. 3-window, 8-mic means; the label's
own row is the same statistic on the four label tracks.

| tracker / arm | mean gap | gap drift over t | gap CV in-frame | centre err (signed) | centre err abs | spread err abs | pred < 5 rev/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| **label** | 2.552 | 0.592 | 0.359 | — | — | — | — |
| HPPNet `real` | 2.680 | 0.478 | 0.392 | -0.207 | 0.783 | 0.717 | 0.000 |
| HPPNet `legacy` | 3.136 | 0.598 | 0.593 | +0.657 | 1.294 | 1.338 | 0.000 |
| HPPNet `v2_real` | 1.275 | 1.501 | 0.772 | **-69.536** | 69.536 | 3.398 | **0.863** |
| HPPNet `v2_legacy_lowk` | 4.262 | 2.675 | 0.586 | -3.234 | 4.835 | 2.737 | 0.044 |
| HPPNet `v2_legacy_free` | 3.067 | 0.548 | 0.581 | -0.015 | 1.134 | 1.260 | 0.000 |
| SCv2 `real` | 2.336 | 0.230 | 0.272 | -1.061 | 1.334 | 0.739 | 0.000 |
| SCv2 `legacy` | 2.480 | 0.104 | 0.217 | -0.812 | 1.393 | 0.729 | 0.000 |
| SCv2 `v2_real` | 2.254 | 0.120 | 0.277 | **-9.104** | 11.126 | 0.937 | **0.124** |
| SCv2 `v2_legacy_lowk` | 2.490 | 0.104 | 0.213 | -1.589 | 1.843 | 0.746 | 0.000 |
| SCv2 `v2_legacy_free` | 2.443 | 0.126 | 0.213 | -1.533 | 1.664 | 0.726 | 0.000 |

**(a) The 69.6 rev/s is the trackers saying the rotors are OFF.** On `v2_real`
HPPNet puts **86.3 %** of its predicted values under 5 rev/s and **84.6 %** of
frames have all four rotors under 5 rev/s — its four tracks sit pinned at 0.0
rev/s for the whole window on 6-7 microphones of 8 (`tracks_hppnet.png`, row
3; per-mic MAE on free-flight 80.37 / 10.23 / 80.37 / 4.37 / 80.37 / 80.37 /
80.37 / 80.38 against a mean carrier of 80.39). This is not a comb read at the
wrong spacing; it is the zero-RPS class of the training stream (§2). No other
arm triggers it: `real`, `legacy` and `v2_legacy_free` are at 0.000 and
`v2_legacy_lowk` at 0.044.

**(b) SCv2 does the same thing on exactly one microphone.** Its `v2_real`
per-mic MAE is 79.82 / 1.69 / 0.77 / 0.78 / 2.15 / 0.89 / 3.05 / 0.99 on
free-flight: mic 0 collapses to zero and the other **seven track the render to
0.8-3.0 rev/s**. The 11.3 rev/s mean is one dead channel out of eight
(0.124 = 1/8 of values under 5 rev/s, in all three windows). So the R3 v2
candidate is NOT globally untrackable — it is on the HPPNet side of an on/off
boundary that SCv2 crosses on 7 mics of 8. Broadband level is not the
explanation: `v2_real`'s per-mic dB-rms on free-flight is -26.3 / -22.1 /
-27.7 / -24.3 / -25.9 / -24.6 / -26.5 / -25.2 against the real clip's -22.6 /
-18.8 / -24.5 / -26.9 / -19.4 / -25.6 / -22.1 / -26.7 (`level_dbrms_per_mic`
in `tracks.json`), i.e. 3-7 dB down, and the mic that dies for SCv2 (0) is not
the quietest and the mic that survives for HPPNet (1) is the loudest.

**(c) Does SCv2 emit a roughly equally spaced comb around the trajectory mean?
YES on real — and identically on every render, including the one it gets
80 rev/s wrong.** Its mean gap is 2.336 on `real`, 2.480 on `legacy`, 2.490 /
2.443 on the two fit-to-legacy arms, 2.254 on `v2_real`, against the label's
2.552; its in-frame gap CV is 0.21-0.28 everywhere (the label's is 0.359) and
its comb drifts by only 0.10-0.23 rev/s over the window where the label's own
comb drifts 0.592. **The comb is a prior, not a measurement**: the spread error
is 0.73-0.94 rev/s on all five arms — it does not move when the material moves
— while the centre error carries 83-98 % of the total (`real` 1.334 of 1.614;
`v2_real` 11.126 of 11.297). SCv2 is therefore a usable instrument for the
MEAN carrier and not for per-rotor spread.

**(d) HPPNet's comb does respond to the material, and that is what costs it.**
Its mean gap moves 2.680 (`real`) -> 3.136 (`legacy`) -> 3.067
(`v2_legacy_free`) -> 4.262 (`v2_legacy_lowk`), and its comb drift over time
goes 0.478 -> 0.598 -> 0.548 -> 2.675. The whole of `v2_legacy_lowk`'s 5.56
against `v2_legacy_free`'s 1.75 is comb instability plus a 3.2 rev/s centre
bias: spread error 2.737 against 1.260, centre error 4.835 against 1.134. On
the arms it accepts, HPPNet's centre is the better of the two trackers
(-0.207 on `real`, -0.015 on `v2_legacy_free`, against SCv2's -1.061 /
-1.533); SCv2 has a systematic 0.8-1.6 rev/s low bias on every arm.

## 2. H3 (provenance): neither tracker has ever seen a legacy rig render

Config reading only, no code run.

**`hppnet_l2_r2_s0`.** `conf/experiment/hppnet_l2_r2_s0.yaml:6` takes
`data: e12_real_fullflight` and `:22` overrides its training stream to
`conf/online_mix/hb_silence_dload.yaml`. That policy's `sources.noise` block
(`hb_silence_dload.yaml:29-46`) is exactly three entries:

| source | spec | weight |
|---|---|---:|
| real DREGON | `kind: frames`, `dataset: DREGON-frames`, `splits: [in_flight_noise]`, `exclude_recording_ids: [free-flight_nosource_room1]`, `min_motor_rps: 0.0` | 1.0 (default) |
| real Michael's | `kind: frames`, `dataset: michaels-frames`, `recording_ids: [FLY125]`, `min_motor_rps: 0.0` | 1.0 (default) |
| synthetic silence | `kind: silence`, `n_channels: 8` | 0.4 |

Speech is LibriSpeech `train-clean-100`. The default weight is 1.0
(`src/data_processing/frame_datasets.py:1511`), so the merged real weight is
2.0 and the split is **83.3 % real recordings / 16.7 % zero-labelled silence /
0.0 % rig renders**. The silence arm is synthetic but is not a rotor render:
`hb_silence_dload.yaml:9-13` describes it as quiet room tone, colored noise up
to flight level, or a 30-60 Hz rumble, each with an exactly zero rotor-speed
label. Validation is
`dload:DREGON-LM-V4-michaels-valid-full@9604f3ff…` (`:25`), built from real
recordings only (`src/data_processing/derivations.py:1550-1560`:
`free-flight_nosource_room1`, `free-flight_speech-low_room1`,
`free-flight_whitenoise-low_room1`, michaels `FLY124`).

**`real_r4_scv2_unified`.** `conf/experiment/real_r4_scv2_unified.yaml:3`
inherits `real_r4_scv2`, whose `:16` takes the same
`data: e12_real_fullflight` and whose `:29` overrides the training stream to
`conf/online_mix/hb_m3s2_dload.yaml`. That policy's `sources` block
(`hb_m3s2_dload.yaml:32-49`) is `hb_silence_dload.yaml`'s **verbatim** — its
own header says so at `:5-11` — so the same 83.3 / 16.7 / 0.0 split holds. No
warm start: neither file sets `checkpoint:`, and the root default is
`checkpoint: null` (`conf/config.yaml:42`); `real_r4_scv2.yaml:3-4` states
"NO warm-up stage and NO warm start" explicitly.

`real_r4_scv2_unified.yaml:4` does add `validation: rps_unified`, whose panel
carries four SYNTHETIC views (`static_nomix`, `static_mix`,
`stochastic_nomix`, `stochastic_mix`, from `conf/online_mix/salv2_comb.yaml`
and `salv2_stoch.yaml`, i.e. `kind: static_comb` and `kind: stochastic`).
These never enter the loss — they are monitored and they steer the LR
schedule. And the checkpoint this round scores,
**`best_real_r2.ckpt`, is by construction the one selected on the REAL view**:
`conf/validation/rps_unified.yaml:91-94` defines `real_r2` as the real dataset,
samples 0-176, all channels, and `src/training/loop.py:965-968` names each
checkpoint `best_<score_name>.ckpt`.

**Legacy rig renders exist as a training corpus — in other experiments.**
`conf/online_mix/rig_easy_5050.yaml:75`, `rig_hard_5050.yaml:79` and
`traj_fitted_5050.yaml:80` set `preset_bank: data/rig_banks/rig_{easy,hard}_n2048.json`,
banks drawn offline by `experiments.stochastic_fit.rig_sampler` and anchored on
`results/S2/cruise_8clip.json:fly125_cruise_00` (the header at
`rig_easy_5050.yaml:35-40`). Neither of the two trackers' streams references
any of them, and neither references the legacy DREGON export this round scores
against (`results/S2/dregon_room2_cruise_refined.json`).

**Verdict on H3: REFUTED.** Zero percent of either tracker's training clips
are legacy rig renders, of any rig. The trackers do not prefer the legacy
render because they were trained on its family.

**But the provenance does carry a different, load-bearing fact.** The
`in_flight_noise` split is
`[free-flight_nosource_room1, free-flight_nosource_room2,
hovering_nosource_room2, updown_nosource_room2, rectangle_nosource_room2,
spinning_nosource_room2]` (`src/data_processing/sources/dregon.py:88-95`) and
only `free-flight_nosource_room1` is excluded — so **all three recordings this
round scores are in both trackers' TRAINING pool**, at whole-envelope
`min_motor_rps: 0.0`, while the validation pool is room 1 + FLY124. The real
arm's 1.07 / 1.61 is an in-domain number on seen recordings, and the 16.7 %
zero-labelled silence arm is exactly the class HPPNet assigns to the R3 v2
render (§1a). The provenance therefore explains the SHAPE of the failure (a
trained OFF class exists, and the real clips are memorisable) without
explaining the render's membership in it — that is §3.
