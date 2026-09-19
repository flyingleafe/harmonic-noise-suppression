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
