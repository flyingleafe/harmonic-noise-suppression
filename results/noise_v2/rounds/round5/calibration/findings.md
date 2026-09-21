# R5 DREGON calibration: one scalar, one threshold, and what it cost

Everything here is read back from the committed JSONs of this directory —
`pin.json`, `check.json`, `score.json` — and from the probe's own
`round5/tracker_probe/{tracks,cqt}.json` for the quoted arms. The rig being
calibrated is `results/noise_v2/rounds/round5/fits/dregon_room2_floor__flight_profile.json`,
mode `flight_profile`, **NOT converged** (`converged=false`,
`which_converged="none"`); the calibrated rig is written as a fit
JSON at `results/noise_v2/rounds/round5/calibration/dregon_room2_floor__flight_profile_pin.json`.

## The pin: +3.75 dB

The search is the smallest offset on a 0.25 dB grid over
0–24 dB whose WEAKEST window clears S8 carrier gain
≥ 5.6 dB, the probe's ON boundary. It is **+3.75 dB**; +3.5 dB is the last
grid point that fails (9 offsets evaluated, bisection plus a 3 dB audit sweep).

| window | S8 as fitted (dB) | S8 at +3.75 dB (dB) | Δ | clears 5.6 |
|---|---:|---:|---:|---|
| free-flight | 5.049 | 7.461 | +2.412 | yes |
| hovering | 3.908 | 5.697 | +1.789 | yes |
| updown | 5.087 | 7.543 | +2.455 | yes |
| rectangle (cohort check) | 4.975 | 7.140 | +2.166 | yes |
| spinning (cohort check) | 3.978 | 5.774 | +1.796 | yes |

**hovering binds**: it is the weakest window at every offset, and the other two
clear the threshold ~1.8 dB earlier. The two windows the pin was NOT bisected
on — rectangle and spinning, the rest of the frozen cruise cohort — clear it
at the same offset, so the pin was not re-bisected (`check.json`).

The bisection assumes the pass set is upward-closed; the audit sweep
(`audit.offsets_db` = [0.0, 3.0, 3.25, 3.5, 3.75, 4.5, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]) confirms the carrier gain is
monotone in the offset on all three windows.

### The prominence ladder, k = 1..8

| k | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| as fitted (dB) | 4.19 | 4.48 | 0.08 | 1.08 | 0.43 | 0.55 | 0.07 | 0.39 |
| at +3.75 dB (dB) | 5.38 | 6.01 | 0.13 | 1.74 | 0.65 | 0.88 | 0.15 | 0.52 |
| Δ | +1.19 | +1.53 | +0.04 | +0.66 | +0.22 | +0.33 | +0.09 | +0.14 |

`c_k` is the read bin over the inter-tooth floor on HPPNet's own CQT, mean
over 8 microphones and the three windows. The pin does NOT move the ladder by
3.75 dB: it moves k = 1 by +1.19 and k = 2 by +1.53 dB and the rest by under
0.7 dB, because raising the comb also raises the inter-tooth floor the
contrast is read against. Only k = 1, 2 and 4 carry a resolvable tooth at all;
k = 3, 5, 6, 7, 8 sit within 1 dB of their own floor both before and after.

## The price: the pin does not cost likelihood, it BUYS it

Priced on the **three 4 s score windows' own cells** (93 frames after the
fit's own stride, 749,952 cells), with the Whittle risk evaluated in
8-frame chunks and summed — exact, because the risk is a weighted sum
over frames:

| | as fitted | at +3.75 dB | delta |
|---|---:|---:|---:|
| nats/cell (all cells) | -5.499965 | -5.980339 | **-0.480374** |
| comb band (723,912 cells) | -5.757089 | -6.241131 | -0.484042 |
| floor band (26,040 cells) | 1.648086 | 1.269676 | -0.378410 |

**The `nats_per_cell_cost` is negative.** The pin is not a concession the
likelihood pays for: on the windows the trackers are scored on, the comb
raised by 3.75 dB fits the data BETTER by 0.480374 nats/cell, and it does so
in both bands. That is a fact about where the fit's pool is: the R5 fit was
run on the five 8 s FLOOR segments, which sit beside the scored cruise
windows and are quieter, so the level the pooled fit chose is too low for the
cruise windows themselves. The threshold crossing and the likelihood want the
same direction here, which is why the offset is 3.75 dB and not 20.

What is NOT computed: the same price on the fit's OWN five 8 s segments.
`likelihood.fit_pool_cost.status` is `not_computed` — the flight forward model on the fit's own 255-frame pool is the step that OOM-killed this terminal three times; the laptop rule for R5 is that it is never built here.
The command that computes it on a 64 GB node is recorded verbatim in
`pin.json` (`likelihood.fit_pool_cost.command`): it is
`scripts/noise_v2_calibrate_dregon.py pin --pool fit --allow-fit-pool` inside
an `omnirun --backend uni-cpu --cpus 16 --mem 64` job that rebuilds the five
8 s supports first. For context only, the fit's own recorded objective on
that pool is -8.712373 nats/cell — a different pool and a different
cell count, not a comparandum of the two numbers above.

## The verdict: both trackers, three windows, eight arms

PIT MAE in rev/s, mean over the three cruise score windows, seed 2001, 8
microphones, the widen runner's own render and PIT path. The frozen HPPNet is
`hppnet_l2_r2_s0/best`
(sha256 `6e50e025ba40…`); SCv2 is
`real_r4_scv2_unified/best_real_r2.ckpt`.

| arm | HPPNet | SCv2 | source |
|---|---:|---:|---|
| the real DREGON room-2 clip | 1.0738 | 1.6143 | round5/tracker_probe/tracks.json |
| legacy stage-2 render | 1.9632 | 1.5942 | round5/tracker_probe/tracks.json |
| R3 v2 fitted to the real clip (flight_floor_lowk) | 69.5551 | 11.2967 | round5/tracker_probe/tracks.json |
| R4 v2 fitted to the LEGACY render (flight_floor_lowk) | 5.5597 | 1.9598 | round5/tracker_probe/tracks.json |
| R4 v2 fitted to the LEGACY render (flight, free profile) | 1.7464 | 1.7983 | round5/tracker_probe/tracks.json |
| **R5 v2 flight_profile, as fitted** | **1.6185** | **1.1007** | this pass |
| **R5 v2 flight_profile + the calibration pin** | **1.6563** | **1.0491** | this pass |
| **R5 v2 flight_profile + the pin + the stability margin** | **1.6783** | **1.0420** | this pass |

Per window (free-flight / hovering / updown):

| arm | HPPNet | SCv2 |
|---|---|---|
| `real` | 0.638 / 0.888 / 1.695 | 0.983 / 1.314 / 2.546 |
| `legacy` | 1.496 / 1.895 / 2.499 | 1.273 / 1.547 / 1.963 |
| `v2_r5_asis` | 1.345 / 1.632 / 1.878 | 0.772 / 1.052 / 1.479 |
| `v2_r5_pin` | 1.453 / 1.648 / 1.868 | 0.816 / 1.057 / 1.274 |
| `v2_r5_pin_plus` | 1.488 / 1.683 / 1.863 | 0.842 / 1.062 / 1.222 |

The `real` and `legacy` rows are the identity check: this pass re-renders and
re-scores them and reproduces the probe's committed numbers exactly
(HPPNet 1.073823 / 1.963204, SCv2 1.614303 / 1.594223), as it must, since the
audio is bit-identical.

What the table says:

1. **The R5 rig tracks, with or without the pin.** As fitted it is 1.6185
   (HPPNet) and 1.1007 (SCv2); pinned, 1.6563 and 1.0491. Both are under the
   DREGON parity bar of 2.187786 and under the legacy render's own 1.9632
   (HPPNet). On SCv2 the R5 rig beats the REAL clip (1.6143) and the legacy
   render (1.5942).
2. **The pin is insurance, not the mechanism.** It costs 0.0378 rev/s on
   HPPNet and buys 0.0516 rev/s on SCv2 — both inside the between-window
   spread — and it does not change the verdict on any window.
3. **Stable at +3 dB.** The stability arm (+6.75 dB total) reads 1.6783 and
   1.0420: the verdict is not balanced on the threshold crossing.
4. **The R5 rig beats R4's fit-to-legacy `flight` rig on both trackers**
   (1.7464 / 1.7983), and it is a fit to the REAL recording, not to another
   model's render.
5. **R3's fit-to-real rig remains the OFF verdict** (69.5551 / 11.2967).

### The caveat the threshold itself carries

The REAL DREGON clip's own S8 carrier gain is **0.98 dB** — far BELOW the 5.6 dB
'ON threshold' — and HPPNet tracks it at 1.0738 rev/s. The R5 rig as fitted
reads 3.91–5.09 dB, also below it, and is tracked at 1.6185. So S8 ≥ 5.6 dB is
NOT a necessary condition for the ON verdict: it is the sufficient side of a
boundary the probe drew across five arms, and the one arm that is OFF
(R3 fit-to-real, 1.63 dB) fails for reasons the single statistic does not
separate. The pin therefore buys margin on a statistic that is not the whole
story — which is exactly why it is priced, scored and reported as a stability
arm rather than trusted.

### Line width, for the record

The −3 dB width of the order-tracked k = 1 line on the 8192-point grid, per
window, is 11.6, 12.6, 12.0 Hz as fitted and 10.8, 11.3, 11.2 Hz pinned. The frozen bench
`gamma_hz` is under 0.01 Hz at k = 1; this width is the label's own drift
inside the 512 ms analysis frame plus the four-rotor spread — the same
quantity R4 read 13 Hz on — and the pin narrows it slightly by lifting the
line out of the floor.

## Figures

* `carrier_gain_vs_offset.png` — the dose-response of S8 carrier gain in the
  offset on all three windows, the threshold, the pin, and the probe's own
  committed gains for the real clip, the legacy render and R3's fit-to-real.
* `prominence_ladder.png` — `c_k`, k = 1..8, as fitted and pinned.
* `score_arms.png` — both trackers on all eight arms against the parity bar.

