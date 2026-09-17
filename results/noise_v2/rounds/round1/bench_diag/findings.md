# R1 bench diagnosis: why the DREGON bench fits carry no comb

`scripts/noise_v2_bench_diag.py` (`census`, `diag`, `objective`, `verify-patch1`,
`figures`). Every number below is in the JSONs in this directory; no number is quoted from
memory, and the proposed patch 1 is run offline on all 21 recordings before being proposed.

## Verdict in one paragraph

The comb is absent from the round-1 DREGON bench fits because **the supports are absent
of motor**. The bench stationarity rule selects a window by frequency-residual
stationarity alone, and silence satisfies that test perfectly: `line_present` is measured
on the WHOLE recording and never inside the chosen window. **12 of the 21 `dregon-bench`
supports sit >= 10 dB (10 of them >= 20 dB) below the loudest same-length window of their
own recording** (`census.json`). `bench_dregon_Motor1_70`'s frozen window [35.707, 43.0] s
is 24.28 dB down — the motor stops at ~36 s — and `bench_dregon_Motor2_60`'s [15.9, 23.0] s
is 21.37 dB down. Both were fitted anyway; `Motor1_70` even carries
`stationary_pass: false, line_present: false, line_margin_db: 2.70`. Registration is a real
but second-order defect, and it is the FROZEN INDEX carriers that are wrong, not only the
fit's refinement. The reported "comb 40 dB below the floor" is a unit misreading:
`profile_db` is a per-order line VARIANCE, and the model's own unit-line response puts the
line's peak bin +52.1 dB (k = 1) to +37.9 dB (k = 80) above it on this support
(`objective_bench_dregon_Motor1_70.json:c_oracle.index.peak_scale_db`).

## 0. Units: `profile_db` is not a periodogram level

`peak_scale_db` = `10 log10 max_j lines_unit_j`, the model's own response to a unit-power
order (`fit.initial_values`' `unit - quiet` probe), measured at each fit's own parameters:

| support | k=1 | k=2 | k=5 | k=10 | k=20 | k=40 | k=80 |
|---|---|---|---|---|---|---|---|
| Motor1_70 (fitted dynamics) | +52.05 | +50.96 | +49.53 | +46.88 | +43.93 | +40.93 | +37.93 |
| Motor1_80 (fitted dynamics) | +49.85 | +47.37 | +41.99 | +36.45 | +31.76 | +27.61 | +23.75 |
| Motor1_70 motor-on (prior-centre dynamics) | +46.13 | +44.63 | +41.94 | +37.70 | +33.89 | +30.54 | +27.41 |

So `Motor1_70`'s fitted `profile_db[0] = -88.4` is a line peak at **-36.4 dB** against a
`-76.1 dB` floor, i.e. 40 dB ABOVE the floor, not 12 dB below it; and `Motor1_80`'s
`-47.6 dB` at k = 1 is a peak near **0 dB** against a `-48.7 dB` floor — confirmed against
the data by the initialiser check of section 4 (`predicted_line_peak_db` +1.08 dB at k = 1
against `data_line_peak_db` -0.12 dB). The premise that Motor1_80 is "a comb at the per-bin
floor" does not survive the conversion: that fit has a real, correctly scaled comb.

## Verdicts

| hypothesis | verdict | the numbers |
|---|---|---|
| **H1** in-fit refinement misregisters the comb, so the optimiser drives the profile to zero | **supported as a defect, refuted as the cause** | On windows that contain a comb the in-fit refinement is right: Motor1_80 refined 78.2223 against a registration argmax of 78.2176, raising orders above 6 dB from 21/90 (index carrier) to 41/90. On the two silent windows it wanders (+0.256 and **-0.762** rev/s) and the MAP carrier then walks **3.791 prior sigmas** to 70.1955, where k = 14/28/42/56 land on a fixed 89.27 Hz interferer comb at 22–26 dB. Cost of misregistration where a comb exists: 0.17-0.28 nats/cell at the oracle (0.87 before the oracle's scalar-offset search), against 2.3 for the empty window. It is also the FROZEN INDEX carriers that are wrong — biased low by +0.09 to +0.20 rev/s on every motor-on window. |
| **H2** a scale/units inconsistency between support builds | **refuted as a units/loader/length bug, confirmed as a real level difference** | Same audio, same loader, 21 s -> 7 s moves the median in-band level by **+1.13 dB** for a -4.77 dB length ratio, i.e. the periodogram units are length-invariant. The 27.1 dB (-76.08 vs -48.99 dB) is the motor being off: Motor1_70's window is **24.28 dB** below the loudest same-length window of its own recording. |
| **H3** an initialiser bug (profile init reads the wrong bins) | **refuted** | `fit.initial_values`' profile predicts the data's line peak to **0.4–1.5 dB at every order of every support**, motor on or off (e.g. motor-on control k = 1..4 predicted -10.8 / -2.5 / -14.6 / -10.4 dB against data -11.8 / -3.5 / -15.6 / -11.4 dB). Its only weakness is the +-3-bin read window, which fails only when registration already has. |
| **root cause** the bench rule selects post-spin-down silence | **confirmed, and the fix is verified** | 12 of 21 supports >= 10 dB down, 10 of them >= 20 dB. Within one window the comb is worth **0.14 nats/cell** on the frozen Motor1_70 support and **2.42** on a motor-on window of the same length. With patch 1 every one of the 21 windows lands on the motor (level deficit 0.48–2.14 dB) and every in-window line margin clears the 3 dB rule (3.71–16.29 dB). |

## 1. What the three supports actually contain

`<support>.json:scale`, `0.5 s` mic-mean RMS envelope; `<support>__motor_on` is the
CONTROL — same recording, same loader, same duration, the loudest window on a 0.5 s grid.

| support | window (s) | median in-band | segment RMS | window env | loudest same-len | deficit |
|---|---|---|---|---|---|---|
| Motor1_70 (committed) | 35.707–43.0 | **-76.08 dB** | -57.94 | -71.95 | -47.68 @17.5 s | **24.28 dB** |
| Motor1_70 motor-on | 17.5–24.79 | **-49.70 dB** | -43.27 | -47.66 | -47.68 | -0.02 dB |
| Motor2_60 (committed) | 15.9–23.0 | **-78.67 dB** | -62.28 | -72.11 | -50.74 | **21.37 dB** |
| Motor2_60 motor-on | 4.0–11.10 | **-52.88 dB** | — | — | — | 0.00 dB |
| Motor1_80 (committed) | 2.0–23.0 | -48.99 dB | -41.86 | -56.41 | -55.86 | 0.55 dB |
| Motor1_80 motor-on | 0.5–21.5 | -48.53 dB | — | — | — | 0.00 dB |

Motor1_70's envelope (`scale.envelope_db`): `-76 -76 -62 -64 -70 -72 -71 -72` then `-48`
from 4.0 s to 35.5 s, then `-61 -71 -73 -74 -75 -76 …` to the end. The frozen window starts
at 35.707 s, i.e. 0.3 s before the motor stops.

Per-order line SNR (peak over local median floor within +-50 bins), at the INDEX carrier,
orders in band:

| support | orders | > 6 dB | sum SNR | median SNR |
|---|---|---|---|---|
| Motor1_70 committed | 103 | **17** | 427.0 dB | 3.93 dB |
| Motor1_70 motor-on | 103 | **28** | 642.1 dB | 4.30 dB |
| Motor2_60 committed | 120 | **4** | 374.2 dB | 2.92 dB |
| Motor2_60 motor-on | 120 | **25** | 588.1 dB | 3.52 dB |
| Motor1_80 committed | 90 | 21 | 484.6 dB | 4.58 dB |
| Motor1_80 motor-on | 90 | 25 | 487.8 dB | 4.74 dB |

A pure-noise order already scores ~5 dB under this statistic (max of 5 Gamma(8)/8 bins over
the 101-bin median), so `Motor2_60`'s 4 of 120 is *no comb at all*. On the motor-on windows
the low orders are unmistakable (`initialiser.data_line_snr_db_k1_10`, at that window's own
refined carrier): Motor1_70 control k = 1..4 line SNR **27.8 / 41.4 / 36.6 / 33.8 dB** at
peaks -11.8 / -3.5 / -15.6 / -11.4 dB, Motor2_60 control **28.8 / 37.6 / 25.1 / 28.2 dB**,
against 6.2 / 0.9 / 3.6 / 8.6 dB (Motor1_70) and 4.4 / 0.1 / 3.0 / 3.4 dB (Motor2_60) on
the committed windows (`fig_overlay_Motor1_70_motor_on.png`).

## 2. Registration (H1)

Carriers, and the registration score `S(f) = sum_k [min(peak_dB - local_floor_dB, 10 dB)]`
on the 0.002 rev/s grid over +-1.5 rev/s of the index carrier (`registration.scores`;
the assignment's literal `2 bins + 0.3 k residual_std` window is also tabulated as
`spec__*`, and the bins-only one as `tight__*`):

| support | index | survey | fit refined | fitted | prior sigmas of fitted | S argmax |
|---|---|---|---|---|---|---|
| Motor1_70 committed | 68.3174 | 68.30 | 68.5731 | **70.1955** | **3.791** | 68.5614 |
| Motor1_70 motor-on | 68.3174 | 68.30 | (68.5042) | — | — | **68.5154** |
| Motor2_60 committed | 58.0979 | 58.07 | 57.3363 | 57.3452 | -1.505 | 59.0764 |
| Motor2_60 motor-on | 58.0979 | 58.07 | (58.1974) | — | — | **58.1924** |
| Motor1_80 committed | 78.0596 | 78.06 | 78.2223 | 78.2268 | 0.334 | **78.2176** |
| Motor1_80 motor-on | 78.0596 | 78.06 | (78.2226) | — | — | **78.2321** |

`S` at the named carriers, Motor1_70 committed / Motor1_80 committed:
`index 426.31 / 453.64`, `survey 419.09 / 451.10`, `fit_refined 493.47 / 561.54`,
`fitted 449.73 / 555.08`, `argmax 521 / 564`.

Three findings, in order of size:

1. **The frozen index carriers are biased low.** On every motor-on window the score argmax
   is +0.09 to +0.20 rev/s above the index carrier (68.5154 vs 68.3174; 58.1924 vs 58.0979;
   78.2321 vs 78.0596). At the index order (62, 63, 67) that is 12.3, 5.5 and 11.9 Hz —
   90, 39 and 250 bins. `index.json`'s `carrier_shift_rev_s` is +0.017, +0.028, -0.0004:
   `supports.refine_carrier` demodulates the WHOLE recording, two thirds of which is
   silence, so its residual-frequency mean is diluted to ~zero and the index carrier is
   effectively the survey value.
2. **The in-fit refinement is right when there is a comb and arbitrary when there is not.**
   `Motor1_80`: refined 78.2223 vs argmax 78.2176 (0.005 rev/s) and it raises the count of
   orders above 6 dB from 21/90 (index) to **41/90**. `Motor1_70` control: 68.5042 vs
   argmax 68.5154. On the two silent supports it wanders: +0.256 rev/s (Motor1_70) and
   **-0.762 rev/s** (Motor2_60), the latter 1.5 prior sigmas the wrong way.
3. **The MAP carrier is not held by its prior and locks onto a fixed-frequency
   interferer.** Motor1_70's fitted carrier is 70.1955 = **3.791 sigma** from the survey
   mean. The support's loudest narrow lines are a FIXED comb, `89.268 / 178.672 / 267.940 /
   357.344 / 446.612 Hz` at 14.4 / 26.4 / 28.0 / 27.2 / 23.9 dB over the local floor, plus
   `982.354 / 1965.12 / 2947.47 / 3930.24 Hz` at 22.4 / 21.5 / 26.4 / 24.2 dB
   (`registration.interferer`). Those last four are k = **14, 28, 42, 56** of 70.179
   (`interferer.aliased_orders.fitted`), and the same family appears at 88.6 Hz in
   Motor2_60 — the same absolute levels in the motor-on window (-34.4 / -44.3 / -41.8 /
   -38.7 dB) as in the silent one (-48.5 / -41.6 / -38.5 / -37.6 dB), i.e. a rig/room
   source independent of rotor speed. With the rotor's own orders at 2–9 dB, four
   interferer lines at 22–28 dB decide any uncapped harmonic sum: the assignment's
   `spec__raw` score (max over `2 bins + 0.3 k residual_std`, which reads the per-ORDER
   residual and so over-widens the window by a factor of ~62) ranks the spurious 70.179
   carrier BEST of all (`-7097.75` vs `-7209.73` at the index carrier). That is why the
   capped, floor-normalised score above is the one to use.

**H1: supported as a defect, refuted as the cause.** Misregistration costs ~0.17-0.28
nats/cell (section 5); the empty window costs 2.3.

## 3. Scale (H2)

**Refuted as a units/loader/length inconsistency.** The length control (same audio, same
segment start, same loader and periodogram, shorter window) leaves the level where it is:

| support | window | median in-band | control | change | length ratio |
|---|---|---|---|---|---|
| Motor1_80 | 21.0 s | -48.99 dB | 7.0 s | **+1.13 dB** | -4.77 dB |
| Motor1_70 | 7.29 s | -76.08 dB | 7.0 s | -0.005 dB | -0.18 dB |

A 4.77 dB change of window length moves the level by 1.13 dB (the residual is the extra
8 s of silence in the 21 s window, not a normalisation): `data.periodogram`'s
`|rfft(w x)|^2 / sum w^2` is length-invariant, as the contract requires.

**Confirmed as a real level difference in the data**, caused by section 1: raw mic-mean
segment RMS is -57.94 dB (Motor1_70), -62.28 dB (Motor2_60), -41.86 dB (Motor1_80), while
the whole-recording RMS values are -44.89 / -49.11 / -42.62 dB. The 70 %/80 % "29 dB floor
difference" is 27.1 dB of *median in-band level* (-76.08 vs -48.99) and it is the motor
being off, not a scale bug. The fitted floors track the data to ~1 dB:
`floor_level_db -76.10 / -76.43 / -48.72` against data medians `-76.08 / -78.67 / -48.99`.

Two by-products worth fixing while the fits are rebuilt:

* `floor_mean_db` ends **bit-identical** to `init_floor_mean_db` in all three fits
  (`-76.78462896811223`, `-77.43843722052347`, `-48.05273060209423`;
  `scale.fit_floor_mean_moved_from_init: false`). The identified quantity is
  `floor_mean_db + mean(mic_floor_db)` and the split is an exact ridge, so this is
  cosmetic — but it means `init_floor_mean_db` is not independent evidence about the fit.
* **Odd-length supports run the model on the wrong grid.** `bench_batch` recovers
  `n = 2 (F - 1)` from the periodogram width, which is `n - 1` for an odd segment:
  Motor1_70 `n_fft 116683` -> model grid `116682`, `bin_hz 0.13712483502` against the
  data's `0.13712365983`, a relative error of `8.57e-6` = 0.062 Hz (0.45 bins) at 7.2 kHz,
  systematically stretching the model comb against the data one. Motor2_60 is odd too
  (113573); Motor1_80 (336000) is exact.

## 4. Initialiser (H3)

**Refuted.** `fit.initial_values` run on each support (`<support>.json:initialiser`), with
its profile converted to the line peak it predicts (`predicted_line_peak_db`) and the data
read at the SAME (init) carrier with the same window (`data_line_peak_db`), so there is no
carrier mismatch in the comparison:

| support | init carrier | init floor | init profile k=1..4 | predicted peak k=1..4 | DATA peak k=1..4 |
|---|---|---|---|---|---|
| Motor1_70 committed | 68.5731 | -77.07 | -94.4 -110.5 -107.1 -98.5 | -48.4 -65.6 -63.3 -55.9 | **-49.0 -65.8 -63.7 -56.5** |
| Motor1_70 motor-on | 68.5042 | -49.56 | -56.3 -47.9 -58.7 -53.3 | -10.8 -2.5 -14.6 -10.4 | **-11.8 -3.5 -15.6 -11.4** |
| Motor2_60 committed | 57.3363 | -78.33 | -108.1 -109.6 -108.8 -108.0 | -62.2 -64.6 -64.9 -65.3 | **-62.7 -65.0 -66.3 -65.6** |
| Motor2_60 motor-on | 58.1974 | -53.78 | -56.3 -49.4 -64.3 -62.5 | -9.7 -3.8 -19.9 -19.4 | **-11.1 -5.2 -21.3 -20.8** |
| Motor1_80 committed | 78.2223 | -49.21 | -49.1 -43.9 -56.8 -55.9 | +1.1 +4.5 -10.3 -11.6 | **-0.1 +3.3 -11.5 -12.7** |
| Motor1_80 motor-on | 78.2226 | -48.78 | -48.4 -43.1 -56.0 -55.0 | +1.8 +5.3 -9.5 -10.6 | **+0.6 +4.1 -10.7 -11.7** |

The initialiser reproduces the data's line peak to **0.4–1.5 dB at every order of every
support**, motor on or off. It starts "below the floor" only where the data is below the
floor, which is the correct answer for a window with no motor. Its one genuine weakness is
the read window: +-3 bins around `k * carrier_init`, while the order's own jitter is
`k * 0.00246 rev/s` for Motor1_70 (1.8 bins at k = 100) and a 0.2 rev/s carrier error is 90
bins at k = 62 — so it degrades exactly when registration does, and only then.

## 5. Objective decomposition (nats/cell, lower is better)

Whittle risk per observed cell against the CURRENT committed fits (`fa5019e3`, the
best-of-four multi-starts: selected seeds 1, 2, 0), reproduced to 1e-9 relative of each
fit's own `objective.whittle_nats` (`-6869282.342`, `-6894010.510`, `-12203270.388`; the
reproduction is recorded as `a_fitted.whittle_nats` beside `fit_json_whittle_nats`). Levels
are not comparable BETWEEN windows — the `log M` term moves with the data level — so
compare within a row.

| support | (a) as fitted | (b) profile = -300 dB | (c) oracle comb, best carrier | (d) index carrier, NO in-fit refine |
|---|---|---|---|---|
| **Motor1_70 committed** | **-16.4217** | **-15.6823** | -15.9914 (@70.1955, the interferer) | **-15.8262** |
| Motor2_60 committed | -16.9319 | -16.6486 | -16.9310 (@57.3372, the interferer) | **-16.6510** |
| Motor1_80 committed | -10.1308 | +40.0478 | **-10.0079** (@78.2223) | **-9.8432** |
| Motor1_70 motor-on control | -10.0384 (init params) | **-7.7037** | **-10.1250** (@68.5042) | -10.0366 |

* comb-band-only (>= 300 Hz): Motor1_70 `-16.5508 / -15.7825 / -16.1035 / -15.9319`;
  Motor2_60 `-17.0560 / -16.7615 / -17.0551 / -16.7640`; Motor1_80 `-10.1635 / 39.2369 /
  -10.0395 / -9.8811`; control `-10.0810 / -9.1490 / -10.1710 / -10.0826`.
* **(d)** is Main's arm: the batch built with `refine_carrier=False` so both the carrier and
  the order geometry come from the frozen index value, the comb read off the data at that
  carrier, everything else as fitted. It lands on exactly the same objective as (c) at the
  index carrier to four decimals on all four supports (`d_index_no_refine` against
  `c_oracle.index`), even though the no-refine batch's `k_max` differs (117 against 116 on
  Motor1_70): **the batch geometry contributes nothing, the carrier is everything.** (d) is
  0.14 nats/cell better than floor-only on Motor1_70 and 0.60 WORSE than the fitted
  misregistered solution; on Motor1_80 it is **0.165 worse** than the same construction at
  the refined carrier (-9.8432 against -10.0079) and 0.29 worse than the fit; on Motor2_60
  it is **0.28 worse** (-16.6510 against -16.9310). Freezing at the current index carrier
  costs real objective on every support, whether the structure at the other carrier is the
  rotor (Motor1_80) or the interferer (Motor1_70, Motor2_60).
* Oracle variants on Motor1_70: peak-matched (the initialiser's own estimator) -15.8245;
  degenerate dynamics (`sigma_nu = 0.959 rad/s` = the index residual std in rad/s,
  `lam = 0.5`) **-16.2272** — a broadband pedestal, not a comb, and it recovers most of the
  fitted model's advantage.
* Oracle carrier scan (+-0.30 rev/s of the index carrier, 0.02 steps): Motor1_70 committed
  is FLAT (best -15.8247 at 68.5574 against -15.8021 at the index carrier); Motor1_80 runs
  from -9.13 (index) to **-10.0093** at 78.2196.

**(a) beats (c) and (d) on the committed supports, and (a) - (b) is not lines.** Splitting
the fitted comb block by order (`comb_block_split`): on Motor1_70 the orders whose lines are
IN band (k <= 102) give -15.9339 while the orders whose lines are OUT of band (k = 105..116,
at -41.3..-54.9 dB, near the -45 dB prior mean) give **-16.2518** on their own. 0.57 of the
fitted model's 0.74 nats/cell advantage over the floor is therefore a broadband PEDESTAL:
with `sigma_nu = 0.189 rad/s, lam = 0.091 /s`, `k^2 V_theta(tau)/2` at k = 110 decoheres in
under a millisecond, so those orders contribute a ~400 Hz-wide floor. The fit used the comb
block as a second floor component. On Motor1_80, by contrast, in-band orders alone give
-10.1299 against the full -10.1308: that fit's comb is genuine lines.

**The model can represent a real comb, and the pipeline can find it.** On the motor-on
control the comb is worth **2.42 nats/cell** against floor-only (-7.7037 -> -10.1250),
versus **0.14** on the committed window (-15.6823 -> -15.8262 at the index carrier). Same
recording, same loader, same duration, same code; only the window moved.

### 5b. Is the in-fit refinement's mode a genuine higher score or a bug?

**Genuine.** `refine_score` in each diag JSON recomputes `refine_bench_carrier`'s own
statistic — `sum_{k >= 4} log P(k f)` at the nearest bin — at every candidate, in the
function's own form (highest order `floor(f_top / f)`, so the summand count moves with `f`)
and on a common order set:

| support | index | refinement's argmax | score @index | score @argmax | delta (common orders) |
|---|---|---|---|---|---|
| Motor1_70 | 68.31742 | 68.57250 | -1727.62 | -1695.04 | **+32.58** |
| Motor2_60 | 58.09792 | 57.33581 | -2185.38 | -2143.39 | **+41.99** |
| Motor1_80 | 78.05962 | 78.23284 | -926.91 | -837.74 | **+89.17** |

The second refinement is therefore finding a real, much higher mode of its own objective,
not misfiring: the statistic is the problem, not the search. On Motor1_80 that mode IS the
rotor comb (78.2328 against a registration argmax of 78.2176–78.2316). On the two silence
windows it is noise and the fixed interferer, and the per-order line table at the fitted
carrier shows exactly which orders pay for it: at 70.1955 the four strongest orders are
**k = 42 (26.1 dB), 56 (24.2), 14 (22.4), 28 (21.5)** — the 982.4 / 1965.1 / 2947.5 /
3930.2 Hz interferer lines, 2.8 to 5.4 bins from `k f`, against a median order SNR of
4.2 dB.

Seeding the SAME function from the index carrier with a +-0.35 rev/s window instead of the
survey value with +-1.0 rev/s (`refine_score.refined_from_index_tight`) already repairs
Motor2_60 on its own: 58.1834 (+0.0855 from the index) against the wide search's 57.3358
(-0.762), and 58.1834 is within 0.009 rev/s of the motor-on control's registration argmax
(58.1919). On Motor1_80 it gives 78.2224, and on Motor1_70 68.5745.

## 6. Proposed fixes

Ordered by measured effect. Patch 1 is the one that matters; 2 and 3 are cheap and
independent. `supports.py` is not owned by any live agent; `spectrum.py`/`model.py` are
R1Fit's.

### Patch 1 — the bench rule must require the motor to be ON inside the window (24 dB)

`stationary_segment` scores stationarity only. Silence is perfectly stationary, and
`line_present` is measured on the whole recording, so the rule walks the window onto the
post-spin-down tail: 12 of 21 supports are >= 10 dB down, and against the loudest 7 s window
**16 of 21** are >= 7.9 dB down (Motor1_80's own 21 s window is 12.06 dB below the best 7 s
window it could have had). The deficits separate cleanly — 17.8..32.8 dB for the twelve bad
ones, 0.2..8.0 dB for the rest — so a 6 dB in-window level floor is unambiguous.

```diff
--- a/src/experiments/noise_model/supports.py
+++ b/src/experiments/noise_model/supports.py
@@
 #: The approved tolerance.
 BENCH_RESIDUAL_TOL_HZ = 1.0
+#: How far below the recording's LOUDEST window of the same length a candidate
+#: window may sit. The residual tests are satisfied perfectly by silence -- a
+#: window with no line has a stationary filtered-noise residual -- and
+#: ``line_present`` is measured on the whole recording, so without this the
+#: rule walks the window onto the post-spin-down tail: 12 of the 21 round-1
+#: DREGON bench supports are >= 10 dB below the loudest same-length window of
+#: their own recording (results/noise_v2/rounds/round1/bench_diag/census.json).
+#: Measured deficits split 17.8-32.8 dB (motor off) against 0.2-8.0 dB (motor
+#: on), so 6 dB separates them with room on both sides.
+BENCH_LEVEL_TOL_DB = 6.0
@@ def stationary_segment
     worst = np.max(np.abs(np.stack(narrow)), axis=0)
     worst_wide = np.max(np.abs(np.stack(wide)), axis=0)
     line_present = min(line_margin_db) >= BENCH_LINE_MARGIN_DB
-    inside = (worst <= BENCH_RESIDUAL_TOL_HZ) & (worst_wide <= BENCH_WIDE_TOL_HZ)
+    # the motor must be RUNNING in the window, on the same grid and with the
+    # same smoothing as the residual tests
+    level = _moving_mean(np.mean(x**2, axis=0), smooth)[inner]
+    level_db = 10.0 * np.log10(np.maximum(level, 1e-30))
+    motor_on = level_db >= level_db.max() - BENCH_LEVEL_TOL_DB
+    inside = (
+        (worst <= BENCH_RESIDUAL_TOL_HZ)
+        & (worst_wide <= BENCH_WIDE_TOL_HZ)
+        & motor_on
+    )
     a, b = _longest_run(inside)
@@
-        a = int(offsets[int(np.argmin([worst[i : i + m].max() for i in offsets]))])
+        # the fallback must not walk into silence either: rank by the residual
+        # among the windows that carry the motor, and by level if none do
+        ok = np.array([bool(motor_on[i : i + m].all()) for i in offsets])
+        pool = offsets[ok] if ok.any() else offsets
+        key = (
+            [worst[i : i + m].max() for i in pool]
+            if ok.any()
+            else [-level_db[i : i + m].mean() for i in pool]
+        )
+        a = int(pool[int(np.argmin(key))])
         b = a + m
@@
         residual_std_hz=[float(np.std(r[keep])) for r in narrow],
+        level_db=float(level_db[keep].mean()),
+        level_deficit_db=float(level_db.max() - level_db[keep].mean()),
+        level_tol_db=float(BENCH_LEVEL_TOL_DB),
```

and refuse to fit a support the rule rejected: `scripts/noise_v2_fit.py:worker` should
raise on `support.meta["stationary_pass"] is False` unless an explicit
`--allow-failed-stationarity` is passed. Three of the 21 carry `stationary_pass: false`:
`Motor1_70` and `Motor4_90` have `longest_inside_s == duration_s`, so by the rule's own
formula it is `line_present` that failed (`Motor1_70`'s recorded margin is 2.70 dB against
the 3.0 dB tolerance), and `Motor2_80`'s longest stationary run was 3.63 s against the 4 s
minimum. All three were fitted.

`line_present` must move inside the window too — the rule measures it with
`select_order(x, ...)` on the WHOLE recording, which is why `Motor1_70` reads 2.70 dB:

```diff
-        k, margin_db = select_order(x, sr, float(f_survey))
+        k, _ = select_order(x, sr, float(f_survey))
 ...
-    line_present = min(line_margin_db) >= BENCH_LINE_MARGIN_DB
+    # measured on the WINDOW the rule selected, not on the recording
+    line_margin_db = [select_order(x[:, i0:i1], sr, float(f))[1] for f in survey_rev_s]
+    line_present = min(line_margin_db) >= BENCH_LINE_MARGIN_DB
```

**Verified, not proposed blind** (`patch1_check.json`, `verify-patch1`, the patched rule run
offline on all 21 recordings so the frozen index keeps building bit-identically):

* every window moves onto the motor: level deficit **0.48–2.14 dB**, against 0.2–32.8 dB now;
* the in-window line margin is **3.71–16.29 dB and 0 of 21 fall below the 3 dB rule**
  (Motor1_70: 2.70 dB on the recording -> **6.04 dB** in its new window [23.659, 29.614];
  Motor4_90: 2.67 -> 3.81);
* but the residual tests then certify less than the 4 s minimum on **10 of 21** recordings
  (`new_longest_inside_s` 0.49–3.58 s) and fall back to a 4 s window, because it is the
  WIDE-band residual that fails while the motor runs (`stationarity.frac_wide_ok` is 0.66,
  0.47 and 1.00 on the three supports diagnosed here, and on the first two the runs that
  pass it are the silent ones) — silence is what used to satisfy it. **Flagging for Main:
  `BENCH_WIDE_TOL_HZ = 3.0` (or `BENCH_WIDE_BAND_FRAC`) has to be re-approved along with
  this patch, or the level + in-window-margin tests must become the gate and the 4 s
  fallback declared acceptable.** 4 s at 16 kHz is a 0.25 Hz bin, which still resolves the
  comb (the line's own jitter is ~0.25 Hz at k = 100), and 11 of the 21 keep 5.2–32.8 s.

### Patch 2 — refine the carrier from the index value, tightly, floor-normalised and capped (0.17-0.28 nats/cell)

`refine_bench_carrier` starts at the SURVEY speed, searches +-1 rev/s (2 prior sigmas) and
sums raw `log P` at the nearest bin of every order from k = 4 up. Unnormalised, it is
decided by whatever is loudest — four interferer lines at 22–28 dB against rotor orders at
2–9 dB — which is how Motor2_60's refinement moved -0.762 rev/s.

```diff
--- a/src/experiments/noise_model/spectrum.py
+++ b/src/experiments/noise_model/spectrum.py
@@ def refine_bench_carrier(
     f0_rev_s: float,
     *,
     sr: int,
-    half_width_rev_s: float = 1.0,
-    n_grid: int = 4001,
+    half_width_rev_s: float = 0.35,
+    n_grid: int = 1401,
     k_min: int = 4,
+    cap_db: float = 10.0,
+    floor_bins: int = 50,
     exclude: np.ndarray | None = None,
 ) -> tuple[float, dict[str, Any]]:
@@
     p = np.asarray(power, dtype=np.float64)
     p = p.reshape(-1, p.shape[-1]).mean(axis=0) if p.ndim > 1 else p
     f = np.asarray(freqs_hz, dtype=np.float64)
-    lp = np.log(np.maximum(p, 1e-24))
+    # Each order contributes its EXCESS over the local floor, capped. A raw
+    # log-power sum is decided by the loudest lines in the band, and on the
+    # DREGON bench those are a fixed ~89 Hz rig comb at 22-28 dB over the
+    # floor while the rotor's own orders carry 2-9 dB: the uncapped score
+    # ranks 70.179 rev/s (k = 14/28/42/56 of the interferer) above the true
+    # carrier on bench_dregon_Motor1_70.
+    from scipy.ndimage import median_filter
+
+    med = median_filter(p, size=2 * int(floor_bins) + 1, mode="nearest")
+    lp = np.minimum(
+        np.log(np.maximum(p, 1e-24)) - np.log(np.maximum(med, 1e-24)),
+        float(cap_db) * math.log(10.0) / 10.0,
+    )
```

and seed it from the index carrier rather than the survey speed, since the index carrier is
a demodulation measurement and the survey value is a Welch harmonic-sum estimate:

```diff
--- a/scripts/noise_v2_fit.py
+++ b/scripts/noise_v2_fit.py
@@ def worker
         batch = MD.bench_batch(
             name=support.name,
             power=np.asarray(support.power, dtype=np.float64),
             sr=int(support.sr),
             carrier_mean=np.asarray(support.carrier_rev_s, dtype=np.float64).mean(axis=1),
+            n_samples=int(support.n_fft),
             k_cap=K_CAP,
         )
```

with `bench_batch(..., n_samples=None)` using `n_samples` when given and asserting
`n // 2 + 1 == power.shape[-1]`, which removes the odd-length grid error of section 3.

### Patch 2b — Main's proposal: drop the in-fit re-refinement and FREEZE the carrier

**The evidence supports dropping the in-fit re-refinement, and supports freezing the
carrier ONLY IF the index carrier is re-refined inside the rebuilt window first. Freezing
at the CURRENT index values is refuted by three independent measurements.**

Dropping the re-refinement — supported:

* its docstring's premise is a survey value good to ~1 rev/s ("3300 bins of a 30 s
  periodogram at order 110"); the index carrier is a demodulation measurement whose
  residual after 2 s smoothing is 0.15–0.20 Hz at k ~ 62–67, i.e. 0.0025–0.0030 rev/s, so
  the +-1 rev/s search it performs is 300x wider than the uncertainty it is correcting;
* it is what lets the MAP walk: Motor1_70's fitted carrier is **3.791 prior sigmas** from
  the survey mean, on a support with no rotor comb, locking k = 14/28/42/56 onto a fixed
  89.27 Hz interferer (26.1 / 24.2 / 22.4 / 21.5 dB line SNR against a 4.2 dB median);
* the batch geometry it also sets is irrelevant: arm (d) shows `refine_carrier=False`
  reproduces the index-carrier objective to four decimals despite a different `k_max`.

Freezing at the current index value — refuted, with numbers:

| support | index | registration argmax on its MOTOR-ON window | index error | at k = index order |
|---|---|---|---|---|
| Motor1_70 | 68.31742 | 68.5154 | +0.198 rev/s | 12.3 Hz = 90 bins at k = 62 |
| Motor2_60 | 58.09792 | 58.1919 | +0.094 rev/s | 5.9 Hz = 42 bins at k = 63 |
| Motor1_80 | 78.05962 | 78.2316 | +0.172 rev/s | 11.5 Hz = 242 bins at k = 67 |

and the objective agrees: arm (d) is 0.165 nats/cell worse than the oracle at the refined
carrier on Motor1_80 and 0.28 worse on Motor2_60, and `refine_bench_carrier`'s own score is
32.6 / 42.0 / 89.2 nats HIGHER at its argmax than at the index carrier (section 5b) — the
index carrier is not a mode of anything. The cause is in `supports.refine_carrier`, which
demodulates the WHOLE recording (two thirds of which is silence on these recordings), so
its residual mean is diluted to ~zero and the index carrier is effectively the survey value
(`index.json` `carrier_shift_rev_s` +0.017, +0.028, -0.0004).

So the fix is an ordering, not a deletion:

1. rebuild the window (patch 1);
2. re-refine the index carrier by demodulation INSIDE that window — same
   `supports.refine_carrier`, restricted to the selected segment, which is a one-line change
   at its call site (`refine_carrier(x[0, i0:i1], ...)`) and is what makes the index carrier
   worth freezing at;
3. freeze the bench carrier block at that value (the fixed-carrier decision): drop the
   `refine_bench_carrier` call in `bench_batch` and take `carrier` out of
   `free_blocks("bench")`, leaving the model to read `batch.carrier_mean`.

With step 2 in place the residual risk of freezing is bounded by the demodulation residual
(0.0025–0.0030 rev/s = 0.19 Hz at k = 67 = 4 bins at 0.0476 Hz), against the 90–242 bins the
current index carriers are off by. **If step 2 is not done, do not freeze:** on Motor1_80
freezing at 78.0596 would throw away 0.165 nats/cell and 20 of 41 orders above 6 dB.
Keeping patch 2's floor-normalised, capped, +-0.35 rev/s refinement as a CHECK rather than
as the estimate is cheap insurance: seeded from the index carrier it already returns
58.1834 on Motor2_60's silence window, 0.009 rev/s from that recording's motor-on argmax.

If the carrier block is frozen, the `N(survey, 0.5^2)` prior disappears with it, which
removes the approved-prior question raised above.

### Patch 3 — read the profile init over the order's own line width

`initial_values` reads `+-3` bins; the line's own jitter is `k * residual_std / order`
rev/s (1.8 bins at k = 100 on Motor1_70) and the decoherence pedestal is wider still.

```diff
--- a/src/experiments/noise_model/fit.py
+++ b/src/experiments/noise_model/fit.py
@@ def initial_values
-                j0 = max(0, int(math.floor(f_lo / df)) - 3)
-                j1 = min(excess.size, int(math.ceil(f_hi / df)) + 4)
+                # +- the window main lobe PLUS the order's own smear: the line
+                # at order k wanders by k * (speed residual), which is 1.8 bins
+                # at k = 100 on a 7.3 s DREGON bench support
+                pad = 3 + int(math.ceil(3.0 * k * jitter_rev_s / df))
+                j0 = max(0, int(math.floor(f_lo / df)) - pad)
+                j1 = min(excess.size, int(math.ceil(f_hi / df)) + pad + 1)
```

with `jitter_rev_s` taken from the support's `meta["stationarity"]["residual_std_hz"] /
orders` (0.00246 rev/s for Motor1_70) and defaulting to 0 when absent.

## 7. Expected gain and what has to be re-run

* Rebuilding the windows: within one window the comb is worth **0.14 nats/cell** on the
  frozen Motor1_70 support and **2.42 nats/cell** on a motor-on window of the same length
  — the comb becomes a first-order term of the objective instead of a rounding error. 12
  of 21 supports change materially; `index.json` and all 20 bench fits must be rebuilt.
* Registration: **0.165 nats/cell** on Motor1_80 at the oracle (index carrier -9.8432 ->
  refined -10.0079; 0.87 before the oracle's offset search) and **0.28** on Motor2_60
  (-16.6510 -> -16.9310); plus 41/90 orders above 6 dB instead
  of 21/90.
* The grid off-by-one and the +-3-bin init window are sub-0.1 nats/cell each but are free.

Not addressed here, worth noting for R1Fit: at the prior-centre dynamics the model's line
is much wider than the data's (`bench_dregon_Motor1_70__motor_on.json:initialiser.
line_fwhm_hz`, at `sigma_nu = 0.45 rad/s, lam = 5.5 /s`): model half-power width
0.274 / 0.411 / 1.234 / 3.291 / 7.405 Hz at k = 2 / 5 / 10 / 20 / 40 against a data line of
0.274 / 0.137 / 0.137 / 0.411 / 0.686 Hz — a factor of 9 at k = 10 and 11 at k = 40. A
rebuilt fit will want `sigma_nu` well below the 0.45 rad/s prior centre, which is also what
the 0.15-0.20 Hz measured residual at order ~62 implies (0.0025 rev/s = 0.015 rad/s).

## 8. Files

| file | content |
|---|---|
| `census.json` | all 21 `dregon-bench` supports: window level vs the loudest same-length and 7 s window |
| `bench_dregon_Motor1_70.json`, `_Motor2_60`, `_Motor1_80` | geometry, stationarity, scale, registration (scores + per-order line tables + interferer), initialiser |
| `bench_dregon_*__motor_on.json` | the same on the loudest same-length window of the same recording |
| `objective_bench_dregon_Motor1_70.json`, `_Motor1_80`, `_Motor1_70__motor_on` | (a)/(b)/(c), oracle variants, carrier scan, comb-block order split |
| `patch1_check.json` | patch 1 run offline on all 21 recordings: old vs new window, level deficit, in-window line margin |
| `fig_overlay_Motor1_70.png` | data vs fitted vs oracle at k = 1,2,4,10,20,40 on the committed support |
| `fig_overlay_Motor1_70_motor_on.png` | the same orders on the motor-on control — the comb the fit never saw |
| `fig_registration_Motor1_70.png` | S(f) with the three carriers marked, per-order line SNR, objective bars |
| `fig_census.png` | per-support level deficit, 21 supports |

Reproduce: `PYTHONPATH=src python scripts/noise_v2_bench_diag.py all` (~9 min on the
laptop; DREGON-frames is local, no submission). Subcommands: `census`, `diag`,
`objective`, `verify-patch1`, `figures`.
