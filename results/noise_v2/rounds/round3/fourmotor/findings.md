# The four-motor transfer, stages 2 and 3

Instrument `scripts/noise_v2_fourmotor.py`, payloads `estimators.json` (stage 2,
schema `noise-v2-fourmotor/1`), `raw_power.npz` (the estimators' raw line
powers, so a re-score never re-measures), `profile_init_e2multi.npz` (the
stage-3 profile init) and `fig_profiles.png`. Every number in section 1–3 is
read out of `estimators.json`.

**The question.** Can the four-motor rig fit of `bench_dregon_allMotors_70`
reproduce the four single-motor R3 fits `bench_dregon_Motor{1..4}_70__bench.json`?
The synthetic study (`results/noise_v2/multirotor_synth/findings.md`) answered
"yes, to 0.71 dB by `estimate_e2(multi=True)`" at the real 0.83 Hz spacing. This
is that estimator on the real recording.

**The caveat, up front.** The four single-motor fits are a DIFFERENT recording:
one motor at a time, different mounting and airflow, no neighbours, and each
one refined to its own carrier (68.31 / 67.55 / 68.63 / 69.33 rev/s against the
four-motor 64.64 / 67.66 / 68.74 / 69.57). They are the stage-3 TARGET, not
ground truth. A gap can be the rig fit's failure or a real difference between
the two recordings, and nothing below separates those two.

## 1. The support, and what it actually contains

`bench_dregon_allMotors_70`, R2 rev-2 window `[3.85, 38.25] s` = **34.40 s**, 8
mics, 16 kHz, 550400 samples. The audio the estimators run on is rebuilt by
`supports.bench_support`'s own route and reproduces the cached support
periodogram to a **max relative error 6.0e-08**, so it is the support's own
material and not a re-cut of it. Carriers 64.63628 / 67.65963 / 68.73641 /
69.56524 rev/s (gaps 3.023 / 1.077 / 0.829), `k_max = 114`, in-band (≤ 7200 Hz)
orders 111 / 106 / 104 / 103.

**The line SNR is the headline.** Peak line power over the local block-median
floor, per in-band line:

| rotor | median line SNR | ≥ 6 dB | ≥ 20 dB | max |
|---|--:|--:|--:|--:|
| 1 (64.636) | 2.6 dB | 9 % | 0 % | 16.0 dB |
| 2 (67.660) | 3.5 dB | 23 % | 3 % | 22.4 dB |
| 3 (68.737) | 3.4 dB | 27 % | 3 % | 28.0 dB |
| 4 (69.565) | 3.3 dB | 20 % | 3 % | 28.5 dB |

By band the medians are 4.0 / 7.8 / 10.1 / 9.1 dB for k ≤ 16 and 2.3–4.2 dB for
17 ≤ k ≤ 48. **The synthetic sweep quotes every headline at a ≥ 20 dB line-SNR
gate; on the real four-motor support 0–3 % of the in-band lines reach it.** The
sweep's own section 5 flagged exactly this hole — its `delta_0p83_real_snr` case
was confounded and "what is missing is one re-run with the comb-to-floor ratio
defined on in-band orders only". That re-run was never done, and the measurement
below is what the untested regime looks like: **the real four-motor support sits
entirely outside the regime in which any of the three estimators was validated.**

The resolvability criterion is not the binding constraint. With the R3 per-rotor
dynamics on the four-motor carriers the within-record worst-neighbour score is
4.68 / 2.22 / 1.87 / 1.87 at k = 1, 2.27 / 0.89 / 0.45 / 0.45 at k = 8 and
0.76 / 0.32 / 0.24 / 0.24 at k = 48 — around and above the threshold 1 at low
orders, below it in the middle, and `offset_spread_ratio = 0.163`, well under 1,
so the rotor LABELS are safe.

## 2. The carrier refinement

`multirotor.refine_offsets` at its committed default `bound_rev_s = 0.5`
returns **−0.1020 / +0.0010 / −0.2889 / +0.0108 rev/s**, and two of those are
wrong. The default search room is half the nearest carrier spacing (0.50 / 0.50 /
0.414 / 0.414 rev/s), which is wide enough to reach a WEAKER comb that is really
in the recording: a harmonic log-power scan of the mic-mean periodogram over
63.5–70.5 rev/s finds a family at ≈ 68.45 rev/s (its k = 1..3 lines at 273.81,
547.62, 821.42 Hz are −7.4 / −17.1 / −21.5 dB) besides the four labelled
carriers, and rotor 3's −0.2889 is a lock onto it. The support index's own
half-window drift at the chosen order is 0.313 / 0.291 / 0.016 / 2.174 Hz at
orders 67/64/63/63, i.e. **0.005 / 0.005 / 0.0003 / 0.035 rev/s** of speed
wander — two orders of magnitude under the wrong locks.

Tightening the room to `--offset-bound 0.05` rev/s (1.4× the worst measured
wander) gives a stable answer, and it is the one used for the estimators:

| rotor | offset (rev/s) | in bins (1/T = 0.02907 Hz) | range over 4 order ladders |
|---|--:|--:|---|
| 1 | +0.00545 | 0.19 | −0.030 … +0.008 |
| 2 | +0.00090 | 0.03 | −0.006 … +0.025 |
| 3 | +0.01725 | **0.59** | −0.017 … +0.032 |
| 4 | +0.01090 | 0.37 | −0.005 … +0.011 |

Independent confirmation: the same harmonic scan puts the best carriers at
68.751 (rotor 3, +0.015) and 69.573 (rotor 4, +0.008). One rotor clears the
0.5-bin threshold, so arm F2 below is run — but the spread over the four order
ladders (0.007–0.018 rev/s) is as large as the estimates themselves, so F2 is a
test of the offsets, not an application of a measurement.

## 3. Stage 2 — the three estimators against the four per-rotor fits

Median `|profile_db(estimator) − profile_db(Motor r)|` in dB over in-band
orders, **collapsed cells excluded and counted separately** (a collapsed cell is
one whose LS power went to zero after its own noise variance was subtracted;
scoring a floored −200 dB against a −60 dB target would report a 140 dB
"error"). Units: the estimator's mic-mean line power converted to `profile_db`
by removing the render transfer and the rig fit's mean-pinned mic gains — a
0.2–0.9 dB correction in band.

| estimator | band | median \|Δ\| per rotor (dB) | collapsed fraction |
|---|---|---|---|
| **E1** | k ≤ 16 | **4.43 / 2.74 / 2.64 / 4.33** | 0.25 / 0.19 / 0.00 / 0.06 |
| | k 17–48 | 9.58 / 3.43 / 5.17 / 7.30 | 0.50 / 0.31 / 0.25 / 0.38 |
| | k 49–103 | 9.33 / 3.71 / 6.77 / 9.12 | 0.30 / 0.10 / 0.20 / 0.15 |
| **E2-single** | k ≤ 16 | 8.04 / 3.66 / 3.56 / 3.89 | 0.12 / 0.06 / 0.00 / 0.00 |
| | k 17–48 | 10.44 / 5.76 / 6.61 / 11.14 | 0.31 / 0.19 / 0.00 / 0.22 |
| | k 49–103 | 10.83 / 5.92 / 8.46 / 8.02 | 0.13 / 0.02 / 0.02 / 0.04 |
| **E2-multi** | k ≤ 16 | 5.41 / 8.39 / 3.98 / 5.54 | **0.56** / 0.19 / 0.00 / 0.12 |
| | k 17–48 | 14.57 / 10.48 / 12.56 / 13.08 | **0.62** / 0.34 / 0.06 / 0.19 |
| | k 49–103 | 15.69 / 8.58 / 9.37 / 16.31 | **0.54** / 0.12 / 0.02 / 0.27 |
| *R3 rig fit (F0)* | k ≤ 16 | 13.85 / 4.90 / 4.35 / 5.50 | — |
| *(for reference)* | k 17–48 | 10.38 / 3.74 / 3.62 / 7.29 | — |
| | k 49–103 | 9.76 / 5.99 / 6.62 / 5.20 | — |

**The synthetic ranking is reversed.** On synthetic data at δ = 0.83 Hz and a
≥ 20 dB line SNR the order was E2-multi (0.71 dB) ≪ E2-single (2.20) < E1
(3.58). On the real support it is **E1 < E2-single < E2-multi** in every band.
Three mechanisms, all of them the low line SNR:

* **The LS noise-variance subtraction collapses.** E2 subtracts each
  coefficient's own variance from `|c|²`; at a 2.6–3.5 dB line-to-floor ratio
  that subtraction takes the estimate to zero. E2-multi loses 56–62 % of
  rotor 1's cells, E2-single 12–31 %, and E1 (which subtracts a smoothed floor
  instead) 19–50 %.
* **E2-multi's steering is estimated from k = 1..16**, where the criterion first
  clears its threshold — but those lines are themselves at 4–10 dB, so the
  steering vector is noise-dominated and the rank-one constraint it imposes is
  wrong. The synthetic study already measured this failure at small δ (its
  estimated-steering arm loses to the oracle by a factor 4); here it is caused
  by SNR instead of by spacing, and the oracle arm does not exist on real data.
* **The carrier refinement is worth little.** The no-offset E2-multi control
  differs from the refined one by ≲ 2 dB in most cells (k ≤ 16: 7.26 / 7.44 /
  3.94 / 5.70 against 5.41 / 8.39 / 3.98 / 5.54), consistent with offsets of
  0.2–0.6 bin.

Restricted to the lines that are actually visible (≥ 6 dB, 10 / 24 / 28 / 21
in-band lines per rotor) the picture is much better and E1 is good:

| estimator | k ≤ 16 (n = 3/10/13/10) | k 17–48 (n = 5/9/9/7) |
|---|---|---|
| E1 | **2.6 / 2.8 / 1.2 / 1.1** | **3.9 / 1.4 / 0.8 / 14.9** |
| E2-single | 7.4 / 2.7 / 2.8 / 3.9 | 6.7 / 2.5 / 4.5 / 11.1 |
| E2-multi | 11.7 / 6.7 / 3.5 / 7.0 | 5.0 / 10.5 / 8.3 / 13.3 |
| R3 rig fit (F0) | 2.6 / 3.2 / 1.8 / 5.2 | 3.6 / 2.2 / 3.4 / 9.5 |

i.e. **where a line is measurable at all, the existing four-motor rig fit is
already within 1.8–3.6 dB of the per-rotor fits on rotors 1–3** and the gap is
rotor 4's (5.2 / 9.5 dB). What the full-band table measures is mostly the
invisible 73–91 % of the comb, where the rig fit's profile is prior-driven and
the per-rotor fits' is prior-driven too — two priors compared to each other.

`fig_profiles.png` shows all of it per rotor, with the ≥ 6 dB lines marked.

## 4. Stage 3 — the rig fit against the combination of the four per-rotor fits

Three arms, all on `bench_dregon_allMotors_70` with the R2/R3 window and the
same 1973192 cells. `F0` is the committed R3 fit, re-used unchanged; `F1` and
`F2` are job `nv2-r3-fourmotor-3d587f` (uni-cpu, 4 starts each, code `d27f58be`,
recipe and support-rebuild check in `submit_note.md`). Payload
`stage3_arms.json`, fits under `fits/F1` and `fits/F2`.

| arm | what changed | nats/cell | conv | grad norm | best−worst start | wall s |
|---|---|--:|:-:|--:|--:|--:|
| F0 | — (the committed R3 fit) | −8.159631 | N (`none`) | 321.1 | 4.08e-4 | 3576 |
| F1 | `profile_db` init + prior from E2-multi (1 dB k ≤ 16, 3 dB 17–48, model default above; 139 lines given a centre and a width, 314 given a centre) | **−8.160531** | N (`none`) | 726.0 | 2.80e-4 | 3595 |
| F2 | F1 + carrier offsets `+0.00545 / +0.00090 / +0.01725 / +0.01090` rev/s | −8.160092 | N (`none`) | 235.9 | 3.37e-4 | 4521 |

No arm converges (as R3's own rig fit did not), and no arm's four starts agree
to the 1e-4 nats/cell tolerance. F1 beats F0 by **9.0e-4 nats/cell** and F2 by
4.6e-4 — real improvements on that scale, so the estimator's profile does buy
likelihood; F2 is worse than F1, i.e. **the carrier offsets do not pay for
themselves** and the 0.59-bin rotor-3 offset is not confirmed by the fit.

### Dynamics

| arm | `sigma_nu` | `lam` | γ(1) | γ(2) | γ(4) | γ(8) | γ(16) | γ log-mean | low-k check |
|---|--:|--:|--:|--:|--:|--:|--:|--:|:-:|
| F0 | 0.5484 | 44.95 | 12.77 | 0.2287 | 0.5722 | 2.213 | 5.428 | 1.861 | `fail_above_floor` |
| F1 | 0.5315 | 44.65 | 22.18 | 0.2346 | 0.5951 | 2.272 | 6.047 | 2.278 | `fail_above_floor` |
| F2 | 0.5301 | 44.64 | 13.68 | 0.2477 | 0.6313 | 2.278 | 6.288 | 2.662 | `fail_above_floor` |
| per-rotor log-mean | 0.7423 | 0.0608 | 0.00137 | 0.00368 | 0.02368 | 0.685 | 1.188 | 3.507 | — |

Ratios to the per-rotor log-mean: `sigma_nu` 0.74 / 0.72 / 0.71, `lam` **739 /
734 / 734**, γ log-mean 0.53 / 0.65 / 0.76. **The dynamics are not restored by
any arm.** All three keep γ(k = 1) at 12.8–22.2 Hz against a per-rotor log-mean
of 0.0014 Hz and a window resolution floor of 0.0145 Hz — the `fail_above_floor`
verdict — and all three sit at `lam ≈ 45` against a per-rotor 0.06. The
profile prior does not touch that; the k = 1 width is still absorbing the
mismatch between four frozen label carriers and four real shaft speeds, and F2
shows that shifting those carriers by the estimator's offsets does not remove it
(γ(1) 12.77 → 13.68).

### The criterion

Per rotor, median `|profile_db(fit) − profile_db(Motor r)|` over **in-band
orders k ≤ 48** (n = 48 per rotor) and the fraction inside 3 dB. The threshold
used here for "harmonic profiles largely restored" is **median ≤ 3 dB AND
≥ 70 % inside 3 dB, for every rotor**.

| arm | median \|Δ\| (rotor 1/2/3/4) | fraction within 3 dB | verdict |
|---|---|---|:-:|
| F0 | 10.96 / 4.30 / 3.88 / 6.56 | 0.17 / 0.42 / 0.40 / 0.21 | **no** |
| F1 | 9.60 / 3.71 / 3.54 / 6.97 | 0.21 / 0.48 / 0.38 / 0.21 | **no** |
| F2 | **7.98 / 3.26 / 3.52 / 6.76** | 0.23 / 0.48 / 0.42 / 0.15 | **no** |

Restricted to the k ≤ 48 lines that clear 6 dB over the local floor (n = 8 / 19 /
22 / 18):

| arm | median \|Δ\| | fraction within 3 dB | verdict |
|---|---|---|:-:|
| F0 | 3.10 / 2.25 / 3.17 / 6.56 | 0.50 / 0.63 / 0.45 / 0.17 | no |
| F1 | 3.68 / 2.13 / 3.40 / 6.77 | 0.50 / 0.68 / 0.41 / 0.17 | no |
| F2 | 3.74 / 2.19 / 3.20 / 7.07 | 0.50 / 0.68 / 0.41 / 0.17 | no |

**Plainly: "harmonic profiles largely restored" does NOT hold, for F0, F1 or
F2.** Every arm fails the median on rotors 1 and 4 and fails the 70 % fraction
on all four. The estimator prior moves the medians in the right direction
(F0 → F2 takes rotor 1 from 10.96 to 7.98 dB and rotor 2 from 4.30 to 3.26) and
buys objective, but it moves nothing across the threshold, and it leaves the
fraction inside 3 dB at 0.15–0.48 against the 0.70 required.

## 5. What this means, and what it does not

1. **The stage-2 remedy is not the binding constraint.** The synthetic study's
   recommendation (E2-multi as a profile init plus per-order prior widths) was
   derived at a ≥ 20 dB line SNR. The real support delivers a median 2.6–3.5 dB,
   E2-multi is the WORST of the three estimators there, and entering it as a
   1 dB / 3 dB prior improves the objective by 9e-4 nats/cell while leaving the
   criterion failed. The prior widths are also unsupported at this SNR: the
   study's 1 dB budget for k ≤ 16 is against a measured real disagreement of
   4–8 dB.
2. **The comparison is bounded below by the target's own uncertainty.** 73–91 %
   of in-band lines are invisible in the four-motor recording, and the four
   single-motor fits' profiles at those orders are prior-driven too. On the
   visible lines the gap is already 2.1–3.7 dB on rotors 1–3 for every arm,
   which is the size of the difference between two recordings of the same motor
   at different mounting — not obviously a rig-fit failure at all. **Rotor 4 is
   the exception: 6.6–7.1 dB on visible lines in every arm**, and it is also the
   rotor whose single-motor fit has the largest half-window drift (2.17 Hz at
   order 63).
3. **The live defect is the dynamics, not the profile.** `lam` 734× the
   per-rotor value and γ(k = 1) at 12.8–22.2 Hz, four orders of magnitude over
   the per-rotor widths and 880–1500× the window resolution, survive every arm.
   That is the frozen-label-carrier absorption named in the R3 findings, and
   this workstream now shows it is **not** fixed by refining the carriers within
   ±0.05 rev/s (F2) nor by pinning the profile (F1). The next thing to test is
   freeing the four bench carriers as sites rather than offsetting them by a
   pre-measured constant.
