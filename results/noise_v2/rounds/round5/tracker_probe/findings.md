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
