# R5 probe (b): what the RPS trackers respond to on DREGON-like data

## Synthesis

**The verdict is BINARY, and on renders it is bought with CQT carrier contrast.**
HPPNet does not mistrack the R3 v2-fit-to-real render: it emits 0.0 rev/s on
**86.3 %** of scored values (6-7 mics of 8) — the zero class of its own
training stream — so 69.6 rev/s is just the mean carrier; SCv2 does it on 1 mic
of 8 and tracks the other seven to 0.8-3.0 rev/s. In the model's OWN front end
(`CQT2010v2`, 48 bins/oct, Q = 68.75, 855 ms window at k=1) the render boundary
is **+1.6 to +5.6 dB of carrier gain** `S₈(1) − med_{α≠1} S₈(α)`: `v2_real`
+1.63 OFF; `v2_legacy_free` +5.57, `legacy` +5.69, `v2_legacy_lowk` +5.98 ON.

**H1 — a finer magnitude statistic separates what the 2048-frame cells could
not: PARTLY, render side only.** The CQT splits the legacy family from
`v2_real` by ≈ 4 dB of carrier gain (c₁ +3.41 against +7.38…+9.56) but NOT
`real` from `v2_real`. **H2 — drift against the γ ladder and σ_ν = 0.74:
mechanism confirmed, explanation refuted.** The 855 ms / 1.17 Hz window costs
real 4.3 dB of its R4 prominence (+5.77 → +1.50) and `v2_real` 4.0 dB
(+7.42 → +3.41); placement is flat too (|peak offset| 1.33 / 1.22 bins, both at
the 1.2-bin no-signal expectation). **H3 — trained on legacy rig renders:
REFUTED, 0 %** — 83.3 % real / 16.7 % silence, no rig bank; but all three
scored recordings are IN both training pools.

**The real clip is the anomaly and no magnitude statistic explains it.** On c₁,
S₈, frac c₁>3 dB and carrier gain it is the WEAKEST arm (α curve peaks at 0.98)
and the best tracked: **its score is not a target a generative fit can reach.**

**SCv2 as an instrument: the MEAN carrier only.** Its comb is a prior — mean
gap 2.25-2.49 rev/s and spread error 0.73-0.94 on all five arms, including the
one it misses by 80 rev/s, against a label gap of 2.55 drifting 0.59 where
SCv2's drifts 0.10; 83-98 % of its error is the centre. It ratifies R4 at
1.96 / 1.80.

**R5: a CALIBRATION PIN, not an objective term.** What moves the tracker is a
THRESHOLD on one scalar that saturates above ~5.6 dB and lives on the tracker's
front end, not in the Whittle likelihood; an objective term would trade
likelihood continuously against a step. Fit the flight pool by Whittle in mode
`flight` (free profile — 1.75 HPPNet / 1.80 SCv2, within 0.4 dB of the legacy
render in every cell class, α = 0.5 alias 4.6 dB down against
`flight_floor_lowk`'s 1.1 dB), then pin `comb_gain_db`, `low_order_gain_db` and
the floor block to reach ≥ 5.6 dB, reporting the pin's cost in nats/cell.

Runner: `scripts/noise_v2_tracker_probe.py` (`tracks`, `cqt`). Renders are R4's
own route at seed 2001, 8 microphones, on the three frozen cruise score
windows; scoring is the frozen path (`_synthetic_probe.score` through
`revised_eval.pit_mae`). Every HPPNet number below reproduces
`round4/legacy_truth/score_legacy_fit.json` at the reported precision and
every SCv2 number for `real`/`legacy`/`v2_real` reproduces
`round3/dregon_humps/widen_scv2.json` at the reported precision, which is the
identity check on the renders. SCv2's pass repeats bit for bit; HPPNet's CPU
forward repeats only to ~1e-8 relative (re-running `tracks` on `updown` gives
1.694930334 against the committed 1.694930323), so nothing here is read
beyond three decimals.

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

## 3. H1/H2: HPPNet's own CQT statistic separates the legacy family — and puts the REAL clip last

Front end read off the registry and run as the model's own module
(`models.harmonic_ports.hppnet_orig.CQTLogSpecgram`, built at
`conf/model/hppnet_l2.yaml:13-21`): nnAudio `CQT2010v2`, **sr 16000, hop 512,
fmin 27.5 Hz, 352 bins, 48 bins/octave** (25 cents per bin, top bin 4368 Hz),
then `torchaudio.AmplitudeToDB(top_db=80)` — an ABSOLUTE log-magnitude, one
floor per clip; no arm hits the clamp (`frac_at_top_db_floor` 0.0 everywhere).
`filter_scale` 1, so **Q = 1/(2^(1/48)-1) = 68.75**. At the free-flight
window's mean carrier 80.39 rev/s that is:

| k | 1 | 2 | 4 | 8 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| analysis window (ms) | **855** | 428 | 214 | 107 | 71 | 53 |
| -3 dB bandwidth = bin spacing (Hz) | 1.17 | 2.34 | 4.68 | 9.35 | 14.03 | 18.71 |

Statistic: per frame, rotor and order, the bin nearest `k*f_r(t)` minus the
INTER-TOOTH floor (median of the six bins nearest `(k±1/2)*f_r`). The brief's
literal ±1/2-semitone annulus is reported as `c_k_local` and is only usable to
k≈6: at 48 bins/octave the four rotors span **1.7 bins** at every k while
order k's neighbours sit `48*log2(1+1/k)` bins away, so the annulus has 6.0
free bins at k=1, 2.0 at k=6, 0.5 at k=8 and **0.0 from k≈12** — the
four-rotor comb stops being resolvable on the model's own grid above k≈11.
That is itself a front-end fact: HPPNet cannot see four separate rotors above
about the 11th order.

3-window, 8-mic, 4-rotor medians (`cqt.json` `summary`; figure
`cqt_contrast.png`):

| arm | c₁ | c₂ | c₄ | c₈ | c₁₆ | S₄ | S₈ | S₁₆ | carrier gain S₈ | frac c₁>3 dB | HPPNet |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `real` | **+1.50** | +1.34 | +0.33 | +0.06 | -0.05 | 5.92 | 10.32 | 18.46 | **+0.98** | 0.308 | **1.07** |
| `legacy` | +7.47 | +3.75 | +1.38 | +0.29 | +0.18 | 14.40 | 18.96 | 27.34 | +5.69 | 0.877 | 1.96 |
| `v2_real` | **+3.41** | +1.29 | +0.13 | -0.08 | +0.12 | 7.11 | 11.21 | 19.35 | **+1.63** | 0.536 | **69.56** |
| `v2_legacy_lowk` | +9.56 | +4.32 | +0.19 | +0.39 | +0.16 | 15.93 | 20.18 | 28.77 | +5.98 | 0.969 | 5.56 |
| `v2_legacy_free` | +7.38 | +3.82 | +1.57 | +0.22 | +0.19 | 14.75 | 19.69 | 28.17 | +5.57 | 0.861 | 1.75 |

`carrier gain` is the decisive column: `S₈` evaluated on the comb of
`α·f_r(t)` for α over 0.90…1.10 in 2 % steps (plus the octave confusions
α = 0.5, 2.0), minus its median over the mis-tuned α — the dB of evidence the
LABEL's carrier has over a wrong one in this front end, with the spectral tilt
cancelled because each hypothesis carries its own floor reference.

**Answer to the key question: NO for the pair that matters, YES for the
legacy family.** The CQT contrast does **not** separate `real` from
`v2_real` where the 2048-frame cells did not — it separates them the WRONG
WAY. On every statistic the R3 v2-fit-to-real render is the STRONGER comb:
c₁ +3.41 against real's +1.50 (+1.9 dB), frac c₁>3 dB 0.536 against 0.308,
S₈ 11.21 against 10.32, carrier gain +1.63 against **+0.98, the lowest of all
five arms**. And the real clip's own α curve peaks at **α = 0.98** (S₈ 11.24)
rather than at 1.00 (10.32), i.e. the magnitude CQT weakly prefers a carrier
2 % BELOW `motors_command`. The tracker nevertheless reads `real` at 1.07 and
`v2_real` at 69.56.

What the CQT contrast DOES explain is the legacy family: `legacy`,
`v2_legacy_lowk` and `v2_legacy_free` carry +4.3…+6.8 dB of carrier gain and
+7.4…+9.6 dB of c₁, against +1.6…+2.1 and +3.4 for `v2_real`, and all three
are tracked. On the RENDER side the ON/OFF boundary of §1a therefore sits
between **1.6 and 5.6 dB of carrier gain** (equivalently between +3.4 and
+7.4 dB of c₁): every render above it is tracked, the one below it is called
silence. The real clip sits below that boundary on both statistics and is
tracked anyway — which, with §2, points at in-domain familiarity (all three
scored recordings are in both trackers' training pool) rather than at evidence
the front end exposes.

**H1 — the CQT resolves what the periodogram cells hid: PARTLY, and not for
the pair in question.** It cleanly separates the legacy family from `v2_real`
(6 dB of carrier gain, 4-6 dB of c₁) where R3's cell-class residuals put the
fit-to-real within +2..+4 dB of the real periodogram everywhere. It does not
separate `real` from `v2_real`.

**H2 — drifting narrow real line against the transplanted γ ladder plus
σ_ν = 0.74 rev/s: the MECHANISM is confirmed, the EXPLANATION is refuted.**
The mechanism is real and is a front-end fact: at k=1 the constant-Q window is
**855 ms** with a 1.17 Hz bandwidth, so a carrier that wanders by ±1 rev/s
inside one analysis window smears its line over ~2 bins. That is exactly why
the real clip reads +1.50 dB here while R4's order-tracked 8192-point STFT
estimator — which re-aligns to the instantaneous carrier every 512 ms frame —
read +5.77 dB on the same audio: 4.3 dB is lost to integrating 855 ms of a
moving line. But the same attenuation hits the real arm HARDER than the v2
arm (real loses 4.3 dB of its R4 prominence, `v2_real` loses 4.0 dB from
+7.42 to +3.41), so shaft wander cannot be why one is tracked and the other
is not. Line PLACEMENT is not the lever either: the strongest bin within
±2 bins sits a mean |1.33| bins from the label for `real` and |1.22| for
`v2_real`, both at the 1.2-bin no-signal expectation of a 5-bin argmax, and
allowing the ±2-bin search buys +2.58 dB on `real`, +1.94 dB on `v2_real` and
+2.62 dB on `legacy` at k=1 (`c_k_peak2_median_db`) — it does not reorder
them.

**Plainly, then: the feature that makes the REAL DREGON clip trackable is not
a magnitude-spectrum comb statistic at any resolution this campaign has
tried** — not R3's 2048/512 periodogram cells, not R4's order-tracked
8192-point prominence ladder, and not the model's own 48-bin/octave CQT, which
is the only one of the three the model actually sees. On the SYNTHETIC side
the same statistic is strongly predictive, which is what makes it usable as a
fitting target.

One render-side caveat the α sweep exposes: `v2_legacy_lowk` scores S₈ 19.12
at the OCTAVE-BELOW hypothesis α = 0.5 against 20.18 at α = 1.0 — a 1.1 dB
margin, because its comb is concentrated in k ≤ 2 (c₁ +9.56, c₄ +0.19). The
free-profile fit keeps a ladder above k=2 (c₄ +1.57) and its α = 0.5 alias is
4.6 dB down. That is the same defect §1d measures as an over-spread, unstable
HPPNet comb on `v2_legacy_lowk` (mean gap 4.26, drift 2.675) and is a second
reason to prefer the free-profile mode.
