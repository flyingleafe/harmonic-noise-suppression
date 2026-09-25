**Status:** stopped — 2026-09-24 → 2026-09-25. CPU round 1 (σ_v rig-wide) is
superseded: the real round runs on a GPU port with σ_v(k). The CPU fits below
only exercise the check tooling.

# Noise model v3: fits on the free-flight pools and the §3.5 checks

Spec: [`docs/explainers/noise-model-v3-wander.qmd`](../explainers/noise-model-v3-wander.qmd)
(model Part 2, fit Part 3, checks §3.5). Code: `scripts/noise_v2_fit.py flight
--mode flight_v3` (commits `4e8cbc0d`, `fec66ac8`, `c85a40a5`), the measured
wander hyperparameters `results/noise_v3/wander/{dregon,michaels}.json`
(commit `8506a2c8`), the checks `scripts/noise_v3_checks.py`.

## Motivation

Noise model v2 renders every line at a constant level: its lines and floor do
not wander. The v2 Whittle likelihood cannot see that, because it compares
one frame at a time with the time-average of the model. v3 adds per-block
Ornstein–Uhlenbeck latents in dB: a common term per rotor (`d`), a term per line
(`v`), a floor level term (`u`) and a floor colour term (`u_j`). Their `σ` and
`τ` are measured on the real windows beforehand and held fixed during the fit.
v3 also changes the priors. Line widths and shaft wander get half-normal
priors on a linear scale. Every order gets the same profile prior. The floor
is the spline alone, scaled by the measured `σ_B`. The channels are
normalised in the data, and DREGON gets a static wind term. This campaign does
two things:

1. It fits v3 on the free-flight training pools of both rigs.
2. It runs the explainer's §3.5 checks.

A third question sits inside check (b). DREGON's lines appear and disappear.
Does a Gaussian OU wander reproduce that?

## Setup

### Pools and options

| fit | pool (`--set`) | windows | same as |
|---|---|---|---|
| `dregon_room2_floor` | `dregon-floor` (the 8 s members) | five disjoint 8 s room-2 floor segments, `motors_command` label | R5 `flight_profile` (`round5/fits/dregon_room2_floor__flight_profile.json` `supports`) |
| `michaels_fly125_cruise` | `michaels-cruise` (FLY125 members) | eight 8 s FLY125 cruise windows, `rps_refined` | R3 `michaels_fly125_cruise__flight.json` |
| `michaels_fly125_standby` | `michaels-standby` | three 4 s FLY125 standby windows, `rps_refined` | R3 `michaels_fly125_standby__flight.json` |

Every fit uses this command:

```
python scripts/noise_v2_fit.py flight --mode flight_v3 --set <set> --name <name> \
  --wander results/noise_v3/wander/<rig>.json \
  --channel-gains results/noise_v2/mic_gains/mic_gains.json --channel-gains-rig <rig> \
  [--wind  # DREGON only] \
  --rounds 3 --max-frames 0 --seed N --restart-tag sN \
  --jobs 1 --threads 16 --progress 100 --out results/noise_v3/fits --grid-dir results/noise_v3/fits/grid/<name>_sN
```

The fit frees the dynamics, the profile, the floor, the wind and the latents.
Nothing is frozen from the bench. The command sets no optimiser option, so the
driver defaults apply: 1500 Adam steps at lr 0.02 on batches of 8 frames, then
200 + 100 L-BFGS iterations on 64 frames. Each window's latents get 100 L-BFGS
iterations, and each rig refit gets L-BFGS only. Frame stride is 4 and every
frame is kept. There are four restarts per pool, one cluster job each.
Restart `s0` starts from the measured initialisation. Restarts `s1`–`s3` add
log-normal jitter (sd 0.8) to the initial dynamics, as `--seeds` would. Commit
`4890fc4e` made a `--restart-tag` seed behave this way; before it, all four
single-seed jobs started from the same point and differed only in the Adam
batch order. The fit also records the prior centres of the profile
(`diagnostics.measured.profile_centre_db`, commit `e1d0e7f0`). Check (a) draws
from these centres.

Each job rebuilds its pool's support caches, then copies them into the shared
cache without overwriting anything. The copy is atomic, because sibling jobs
share one worktree.

### Jobs

The jobs were submitted on 2026-09-24 through `omnirun --daemon localhost:18787`
from the detached worktree `.worktrees/submit-v3`. The backend is `uni-cpu`
(`--gpus 0 --cpus 16`), with `--time 24h`.

| job | pool | seed | code | mem | state |
|---|---|---|---|---|---|
| `nv3-smoke-dregon-697f8d` | DREGON (smoke: 3 Adam steps, 2 L-BFGS iterations, 1 round, 2 latent iterations) | 0 | `4890fc4e` | 64 GB | done, 567 s, exit 0 |
| `nv3-smoke-michaels-eadc6e` | Michael's standby (the same smoke) | 0 | `4890fc4e` | 64 GB | done, 507 s, exit 0 |
| `nv3-fit-dregon-s0-28cc14` | DREGON | 0 | `e1d0e7f0` | 64 GB | succeeded, 37 507 s |
| `nv3-fit-dregon-s1-0d3805` | DREGON | 1 | `e1d0e7f0` | 64 GB | succeeded, 36 719 s |
| `nv3-fit-dregon-s2-0c2b38` | DREGON | 2 | `e1d0e7f0` | 64 GB | succeeded, 27 014 s |
| `nv3-fit-dregon-s3-64f951` | DREGON | 3 | `e1d0e7f0` | 64 GB | succeeded, 29 705 s |
| `nv3-fit-mstandby-s0-3fcf29` | Michael's standby | 0 | `e1d0e7f0` | 64 GB | **cancelled** 2026-09-25 08:40 by the main agent after round 2 (CPU fits too slow; not resubmitted) |
| `nv3-fit-mstandby-s1-6c5be5` | Michael's standby | 1 | `e1d0e7f0` | 64 GB | succeeded, 28 160 s |
| `nv3-fit-mstandby-s2-98610f` | Michael's standby | 2 | `e1d0e7f0` | 64 GB | succeeded, 27 456 s |
| `nv3-fit-mstandby-s3-32f230` | Michael's standby | 3 | `e1d0e7f0` | 64 GB | succeeded, 27 382 s |
| `nv3-fit-mcruise-s{0..3}-{8cf99e,3a0859,7cba5a,f7ba2c}` | Michael's cruise | 0–3 | `e1d0e7f0` | 64 GB | **OOM-killed** 11 s into the fit (8 windows × 62 frames at full frames; DREGON's 5 × 62 fits in 64 GB) |
| `nv3-fit-mcruise-s0-1fb6f3` | Michael's cruise | 0 | `e1d0e7f0` | 160 GB | **cancelled** 2026-09-25 08:40 by the main agent after round 0 (CPU fits too slow; not resubmitted) |
| `nv3-fit-mcruise-s1-91bf7f` | Michael's cruise | 1 | `e1d0e7f0` | 160 GB | **cancelled**, the same |
| `nv3-fit-mcruise-s2-de2c13` | Michael's cruise | 2 | `e1d0e7f0` | 160 GB | **cancelled**, the same |
| `nv3-fit-mcruise-s3-2955b8` | Michael's cruise | 3 | `e1d0e7f0` | 160 GB | **cancelled**, the same |

The smoke jobs ran the full-frame pools with the iterations cut. They cost
about 10 s per L-BFGS evaluation on DREGON (88 orders, 8 mics, 64 frames), the
same as R5's v2 fit (9.4 s). A latent evaluation takes about 11 s per 8 s
window. A full DREGON job does one Adam + L-BFGS rig fit, three L-BFGS refits
and three latent rounds over five windows, so it was budgeted at 10–13 h. That
is why the jobs ask for 24 h instead of the planned 6 h. The wall time each
job actually took goes in "Fits" below.

Harvest:

```
omnirun --daemon localhost:18787 pull <job>        # or aws s3 sync s3://omnirun-artifacts/<job>/outputs/results/ results/
PYTHONPATH=src python scripts/noise_v2_fit.py reduce --out results/noise_v3/fits --mode flight_v3
```

### Checks (explainer §3.5)

`scripts/noise_v3_checks.py` only renders and measures, so it is safe to run
on the laptop. Every render is driven by the real label of the window it is
compared with. It uses 8 mics and the frozen render seeds 2001–2004.

- **(a) prior predictive** (`prior`). 8 draws of every rig site from each
  fit's recorded v3 prior, rendered on a fixed 4 s held-out trajectory. The
  outputs are the maximum `γ/(0.01k)`, the share of lines wider than 5 `γ0 k`,
  the R4 widths of the render, and the floor-shape swing of the draws and of
  the render against the measured `σ_B`.
- **(b) held-out spread** (`heldout`). The windows are not in any pool. For
  DREGON they are the five frozen room-2 score windows (4 s). For Michael's
  they are FLY124 standby @8/16 s and cruise @40/56 s (8 s, `rps_refined`
  label). The FLY124 ramp window lies outside both regime fits and is not
  rendered. The line-power spread uses the wander estimator of
  `scripts/noise_v3_measure_wander.py`, rig-centred with τ at lag 1, on the
  real windows and on v3 and v2 renders of the same labels. The outputs are
  `σ_total`, `σ_d` and `σ_v` with window-bootstrap intervals. For every rotor
  and every order 8–24 the check also histograms the per-block prominence
  (3-bin line cells over the local q25 floor). It reports the **disappearance
  rate**, meaning the share of blocks where a line's power is below its local
  floor (prominence < 10 log10 2 = 3.01 dB). This rate is computed over all
  lines and over the lines present in their window (window prominence ≥ 6 dB).
  It also reports the share of lines that both appear (≥ 6 dB) and disappear
  (< 3.01 dB) inside one window.
- **(c) tonality**. On the expectation this is
  `scripts/noise_v2_tonality_audit.py --fit` (new `--fit RIG[_standby]=PATH`
  mode), with the same per-pattern rows as the pinned v2 anchors beside them.
  On rendered audio (`rendered`) there are 8 clips per fit: 2 pattern windows ×
  4 seeds, on the real labels. The measures are R4's `line_width_db3` at
  k = 1..8 and the window-prominence ladder, with the real windows next to
  them.
- **(d) parity**. `scripts/noise_v2_round_score.py --fit` as R5 used it.
- **(e) latents** (`latents`). The fitted `d`/`v`/`u`/`u_j` spread and the
  fitted spread plus the measured block noise, against the measured `σ`. The
  fitted lag-1 is compared with the measured `ρ = e^{-T_b/τ}`.
- **Parameter view**. `scripts/noise_v3_param_view.py` draws the v3 fit and the
  v2 fit it replaces the same way.

### Monitoring log

`omnirun --daemon localhost:18787 ps`, polled every 30 min (UTC). One row per
poll; a job named in a row changed state at that poll.

| time (UTC) | job | status |
|---|---|---|
| 2026-09-24 22:20 | `nv3-fit-dregon-s{0..3}`, `nv3-fit-mstandby-s{0..3}` | running (DREGON s0 at Adam step 1300 of 1500) |
| 2026-09-24 22:20 | `nv3-fit-mcruise-s{0..3}` (160 GB) | queued (no free slot) |
| 2026-09-24 22:50 | all 8 running fits | running, Adam step 1300–1400 of 1500; `nv3-fit-mcruise-s{0..3}` queued |
| 2026-09-24 23:16 | 8 running, 4 queued | no change; stage by seed — dregon: round 0 done, Adam done / first L-BFGS, round 0 done, round 0 done; mstandby: Adam done / first L-BFGS, Adam done / first L-BFGS, Adam done / first L-BFGS, Adam done / first L-BFGS |
| 2026-09-24 23:46 | 8 running, 4 queued | no change; stage by seed — dregon: round 0 done, round 0 done, round 0 done, round 0 done; mstandby: Adam done / first L-BFGS, round 0 done, round 0 done, round 0 done |
| 2026-09-25 00:09 | 8 running, 4 queued | no change; stage by seed — dregon: round 0 done, round 0 done, round 0 done, round 0 done; mstandby: Adam done / first L-BFGS, round 0 done, round 0 done, round 0 done |
| 2026-09-25 00:39 | 8 running, 4 queued | no change; stage by seed — dregon: round 0 done, round 0 done, round 0 done, round 0 done; mstandby: round 0 done, round 0 done, round 0 done, round 0 done |
| 2026-09-25 01:10 | 8 running, 4 queued | no change; stage by seed — dregon: round 0 done, round 0 done, round 0 done, round 0 done; mstandby: round 0 done, round 0 done, round 0 done, round 0 done |
| 2026-09-25 01:40 | 8 running, 4 queued | no change; stage by seed — dregon: round 0 done, round 0 done, round 1 done, round 1 done; mstandby: round 0 done, round 0 done, round 1 done, round 0 done |
| 2026-09-25 02:10 | 8 running, 4 queued | no change; stage by seed — dregon: round 1 done, round 0 done, round 1 done, round 1 done; mstandby: round 0 done, round 1 done, round 1 done, round 1 done |
| 2026-09-25 02:40 | 8 running, 4 queued | no change; stage by seed — dregon: round 1 done, round 1 done, round 1 done, round 1 done; mstandby: round 0 done, round 1 done, round 1 done, round 1 done |
| 2026-09-25 03:11 | 8 running, 4 queued | no change; stage by seed — dregon: round 1 done, round 1 done, round 1 done, round 1 done; mstandby: round 0 done, round 1 done, round 1 done, round 1 done |
| 2026-09-25 03:41 | 8 running, 4 queued | no change; stage by seed — dregon: round 1 done, round 1 done, round 2 done, round 2 done; mstandby: round 0 done, round 2 done, round 2 done, round 2 done |
| 2026-09-25 04:11 | 8 running, 4 queued | no change; stage by seed — dregon: round 1 done, round 1 done, round 2 done, round 2 done; mstandby: round 1 done, round 2 done, round 2 done, round 2 done |
| 2026-09-25 04:41 | 8 running, 4 queued | no change; stage by seed — dregon: round 1 done, round 1 done, round 2 done, round 2 done; mstandby: round 1 done, round 2 done, round 2 done, round 2 done |
| 2026-09-25 05:11 | 8 running, 4 queued | no change; stage by seed — dregon: round 2 done, round 1 done, round 2 done, round 2 done; mstandby: round 1 done, round 2 done, round 2 done, round 2 done |
| 2026-09-25 05:41 | `nv3-fit-dregon-s2-0c2b38` | succeeded |
| 2026-09-25 05:41 | `nv3-fit-mstandby-s2-98610f` | succeeded |
| 2026-09-25 05:41 | `nv3-fit-mstandby-s3-32f230` | succeeded |
| 2026-09-25 05:41 | `nv3-fit-mcruise-s0-1fb6f3` | running (Adam 800) |
| 2026-09-25 05:41 | `nv3-fit-mcruise-s1-91bf7f` | running (Adam 300) |
| 2026-09-25 05:41 | `nv3-fit-mcruise-s2-de2c13` | running (Adam 400) |
| 2026-09-25 06:11 | `nv3-fit-mstandby-s1-6c5be5` | succeeded |
| 2026-09-25 06:11 | `nv3-fit-dregon-s3-64f951` | succeeded |
| 2026-09-25 06:11 | `nv3-fit-mcruise-s3-2955b8` | running (Adam done / first L-BFGS) |
| 2026-09-25 06:41 | 7 running, 5 succeeded | no change; stage by seed — dregon: round 2 done, round 2 done; mstandby: round 1 done; mcruise: round 0 done, round 0 done, round 0 done, Adam done / first L-BFGS |
| 2026-09-25 07:11 | 7 running, 5 succeeded | no change; stage by seed — dregon: round 2 done, round 2 done; mstandby: round 1 done; mcruise: round 0 done, round 0 done, round 0 done, round 0 done |
| 2026-09-25 07:41 | 7 running, 5 succeeded | no change; stage by seed — dregon: round 2 done, round 2 done; mstandby: round 2 done; mcruise: round 0 done, round 0 done, round 0 done, round 0 done |
| 2026-09-25 08:11 | `nv3-fit-dregon-s1-0d3805` | succeeded |
| 2026-09-25 08:40 | `nv3-fit-dregon-s0-28cc14` | succeeded |
| 2026-09-25 08:40 | `nv3-fit-mstandby-s0-3fcf29` | cancelled |
| 2026-09-25 08:40 | `nv3-fit-mcruise-s0-1fb6f3` | cancelled |
| 2026-09-25 08:40 | `nv3-fit-mcruise-s1-91bf7f` | cancelled |
| 2026-09-25 08:40 | `nv3-fit-mcruise-s2-de2c13` | cancelled |
| 2026-09-25 08:40 | `nv3-fit-mcruise-s3-2955b8` | cancelled |

At 08:40 the main agent cancelled `nv3-fit-mstandby-s0` and all four 160 GB
cruise jobs, because the CPU fits are too slow. The real round runs on a GPU
port with a per-order σ_v(k). No job was resubmitted.

## Fits

### CPU round 1, σ_v rig-wide (superseded)

**These fits are superseded.** They are kept for two uses only: to run the
check tooling end to end on a real fit, and to record what a full CPU fit
costs and how far it gets. No Michael's cruise fit exists. For standby,
three of the four restarts finished (s1–s3). Harvested with
`omnirun pull`, reduced with `noise_v2_fit.py reduce --mode flight_v3`, and
committed under `results/noise_v3/fits/`: the reduced fits
`{dregon_room2_floor,michaels_fly125_standby}__flight_v3.json` and the
restarts in `restarts/`.

Per restart. "Wall" is the job's `V3FIT_DONE wall_s`, which includes a
20–34 s support build. "Move/cell" is the Whittle move per observed cell in
alternation rounds 1–3; the tolerance is 1e-4. The widths are
`params.gamma_hz`. The latents are the rms of the fitted block tracks
(`latents.summary.*.fitted_sd_db`).

| job | seed | wall s (h) | `which_converged` | move/cell, rounds 1–3 | total objective (nats) | σ_ν rad/s | γ median Hz | γ max per rotor Hz | γ > 50 Hz | max γ/(0.01k) | lines > 5 γ0 k | latent sd d / v / u / u_j dB |
|---|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|---|
| `nv3-fit-dregon-s0-28cc14` | 0 | 37 507 (10.42) | none | 0.1069, 0.0022, 0.0008 | −18 038 593.6 | 5.252 | 0.722 | 7.36, 13.88, 4.97, 9.25 | 0 | 81.6 | 56/352 | 2.77 / 1.89 / 2.10 / 4.00 |
| `nv3-fit-dregon-s1-0d3805` | 1 | 36 719 (10.20) | none | 0.1061, 0.0031, 0.0003 | −18 037 678.5 | 5.307 | 0.609 | 7.17, 11.80, 4.96, 6.54 | 0 | 69.4 | 52/352 | 2.77 / 1.90 / 2.17 / 4.40 |
| `nv3-fit-dregon-s2-0c2b38` | 2 | 27 014 (7.50) | none | 0.1064, 0.0023, 0.0006 | −18 037 105.8 | 5.321 | 0.658 | 7.81, 13.52, 5.25, 7.54 | 0 | 79.5 | 55/352 | 2.76 / 1.90 / 2.15 / 4.23 |
| `nv3-fit-dregon-s3-64f951` | 3 | 29 705 (8.25) | none | 0.1058, 0.0032, 0.0001 | −18 037 490.1 | 5.452 | 0.474 | 7.92, 11.64, 4.40, 6.48 | 0 | 68.4 | 50/352 | 2.74 / 1.89 / 2.18 / 4.19 |
| `nv3-fit-mstandby-s1-6c5be5` | 1 | 28 160 (7.82) | none | 0.0391, 0.0030, 0.0010 | −6 948 693.2 | 0.290 | 0.885 | 6.74, 8.08, 10.14, 7.09 | 0 | 20.0 | 70/520 | 1.24 / 1.87 / 1.39 / 1.85 |
| `nv3-fit-mstandby-s2-98610f` | 2 | 27 456 (7.63) | none | 0.0388, 0.0033, 0.0011 | −6 948 789.1 | 0.289 | 0.890 | 6.87, 7.92, 9.39, 11.94 | 0 | 22.6 | 72/520 | 1.22 / 1.85 / 1.40 / 1.82 |
| `nv3-fit-mstandby-s3-32f230` | 3 | 27 382 (7.61) | none | 0.0392, 0.0030, 0.0011 | −6 948 661.8 | 0.284 | 0.895 | 7.28, 8.43, 10.54, 12.02 | 0 | 19.0 | 75/520 | 1.25 / 1.88 / 1.38 / 1.83 |

**DREGON** (`dregon_room2_floor__flight_v3.json`; 4 restarts, s0 selected):

- **Convergence:** none of the four restarts converged
  (`optimiser.converged` false, `which_converged` "none"; neither the
  alternation nor the rig refit met 1e-4 nats/cell). In round 3 the
  alternation still moved 1–8 × the tolerance. The last rig refit left a
  gradient norm of 2043 and a restart gain of 0.0014 nats/cell.
- **Restart spread:** best − median = 4.0e-4 nats/cell and best − worst =
  6.0e-4 nats/cell, 4–6 × the tolerance. σ_ν runs 5.25–5.45 rad/s (max/min
  1.04). The log-mean γ runs 0.387–0.599 Hz (max/min 1.55). Across restarts
  the rotor-max γ is 0.024–0.026 Hz at k = 1, 0.66–0.79 Hz at k = 8 and
  2.08–2.39 Hz at k = 16.
- **Objective** (selected): total −18 038 593.6 nats over 2 499 840 cells.
  That is Whittle −18 092 279.2, plus the rig −log prior 2 814.9, plus the OU
  −log prior 50 870.8. With the latents set to zero the Whittle term is
  −17 817 683.5, so the latents gain 274 596 nats (0.110 nats/cell).
- **Span pins:** none. The speed span is 1.744, above the 1.5 threshold. The
  low-order γ check passes: at k ≤ 4 the largest γ is 0.24 × the resolution.
- **γ:** median 0.722 Hz. The rotor maxima are 7.36 / 13.88 / 4.97 / 9.25 Hz
  at k = 21 / 17 / 14 / 21. No line is wider than 50 Hz. The largest
  γ/(0.01k) is **81.6** (rotor 2, k = 17, γ = 13.9 Hz), and 56 of the 352
  lines exceed 5 γ0 k. At k ≤ 8 the largest γ is 1.06 Hz, which is
  23.7 × γ0 k.
- **σ_ν:** 5.25 rad/s, which is 8.8 × the scale of its HalfNormal prior
  (0.6 rad/s).
- **Latents** (selected; range over restarts in brackets). Fitted sd: d 2.77
  [2.74–2.77], v 1.89 [1.89–1.90], u 2.10 [2.10–2.18] and u_j 4.00
  [4.00–4.40] dB. The measured σ are 0.94 / 1.15 / 2.83 / 1.52 dB. Fitted
  lag-1: 0.63 / 0.83 / 0.61 / 0.91 against the measured ρ of
  0.19 / 0.58 / 0.90 / 0.70.

**Michael's standby** (`michaels_fly125_standby__flight_v3.json`; 3 restarts,
s2 selected):

- **Convergence:** none of the three restarts converged ("none"). In round 3
  the alternation still moved 10–11 × the tolerance. The last rig refit left a
  gradient norm of 212 and a restart gain of 2.2e-4 nats/cell.
- **Restart spread:** best − median = 1.3e-4 nats/cell and best − worst =
  1.7e-4 nats/cell. σ_ν runs 0.284–0.290 rad/s (max/min 1.02). The log-mean γ
  runs 0.538–0.603 Hz (max/min 1.12).
- **Objective:** total −6 948 789.1 nats over 749 952 cells. That is Whittle
  −6 972 250.8, plus the rig −log prior 2 514.7, plus the OU −log prior
  20 947.1. The latents gain 32 394 nats (0.043 nats/cell).
- **Span pins:** none (speed span 1.928). The low-order γ check passes.
- **γ:** median 0.890 Hz. The rotor maxima are 6.87 / 7.92 / 9.39 / 11.94 Hz
  at k = 83 / 108 / 121 / 65. None is wider than 50 Hz. The largest γ/(0.01k)
  is 22.6 (rotor 3, k = 27), and 72 of the 520 lines exceed 5 γ0 k.
- **σ_ν:** 0.289 rad/s.
- **Latents:** fitted sd d 1.22 [1.22–1.25], v 1.85 [1.85–1.88], u 1.40
  [1.38–1.40] and u_j 1.82 [1.82–1.85] dB, against the measured
  0.52 / 1.91 / 1.73 / 1.63 dB. Fitted lag-1: 0.45 / 0.81 / 0.75 / 0.60
  against the measured ρ of 0.42 / 0.67 / 0.78 / 0.67.

### GPU round

The v3 fit ported to the GPU: the rig objective on the device, chunked over
frames (`f4c5218a`); the warm start from the v2 fit of the pool
(`804884e1`); σ_v(k) (`e6f14cf3`); the L-BFGS relative tolerance
(`9329ccf0`); the unit-atom line kernel (`ac390340`, `cb87eeba`). Jobs go to
kaggle through `omnirun` from the detached worktree `.worktrees/submit-v3gpu`.
Each job builds its supports into a fresh directory, copies the `.npz` into
the cache, and `aws s3 sync`s `results/noise_v3/fits_gpu` to
`s3://omnirun-artifacts/<job>/outputs/results/noise_v3/fits_gpu` every
10 min and at the end. On a failure the job prints every `grid/raw/*.err`
and `*.log` before any summary step.

**GPU.** Every job asks for `--gpu-type P100`, but kaggle placed each one on
a **Tesla T4** (15 GB). The T4's fp64 rate is 1/32 of its fp32 rate, and the
rig step runs in fp64, so the T4 timings below are an upper bound for a
P100.

#### Smoke (2 FLY125 windows, K 24, mics 0 1, 2 rounds)

The command is the CPU smoke's (`--support
flight_michaels:FLY125:{16,32}.0:4.0:rps_refined --mics 0,1 --k-cap 24
--init-from round3 cruise --rounds 2 --adam-steps 30 --lbfgs-iters 20
--latent-iters 20 --max-frames 0`), with `--device cuda`.

- **First try (`nv3gpu4-smoke-451582`, `cb87eeba`):** the CUDA unit crashed
  after 20 Adam steps. `fit_support` built its diagnostics with
  `np.asarray(init["gamma_hz"])`, and `init` holds cuda tensors:
  `TypeError: can't convert cuda:0 device type tensor to numpy`. The wrapper
  did not echo the gridrun `.err`, so the log showed no traceback. Fix in
  `d04ae2a4`: `.detach().cpu()` first. A CPU audit patched
  `Tensor.numpy`/`__array__` to flag every conversion that skips `.cpu()`,
  then ran the real `flight_v3` CLI with and without `--wind`/`--init-from`.
  It flags exactly that line and no other. A new test runs a `flight_v3` fit
  on a CUDA device end to end and writes the record; it is skipped without
  CUDA and passed on the kaggle T4. The same job's `--device cpu` smoke on
  the kaggle CPU (4 cores) finished and is the reference below.
- **Rerun (`nv3gpu4-smoke2-bbc101`, `d04ae2a4`, T4): the CUDA fit matches
  the CPU fit.**

| run | code | total objective (nats) | Whittle (nats) | wall s | rig s/eval, rounds 0 / 1 / 2 | latent s/eval | peak GPU MB |
|---|---|---:|---:|---:|---|---:|---:|
| CUDA, kaggle T4 | `d04ae2a4` | −826 321.8109116681 | −830 312.5208480533 | 20.3 | 0.267 / 0.356 / 0.371 | 0.0175 | 2 317 |
| CPU, kaggle (4 cores) | `cb87eeba` | −826 321.8109116696 | −830 312.5208480536 | 379.8 | 5.46 / 7.69 / 8.02 | 0.053–0.069 | — |
| CPU, laptop (12 threads) | `ac390340` | −826 321.8100351072 | −830 312.519943153 | 190.6 | 2.85 / 3.57 / 3.92 | 0.033–0.148 | — |

CUDA and the kaggle CPU differ by 1.5e-9 nats (2e-15 relative). Both follow
the same path: round-0 L-BFGS 20 + 1 iterations (28 evaluations), then 2 + 1
and 1 + 1 in the refits, and `which_converged` "lbfgs" (the alternation
moved 0.0083 nats/cell in round 2). The laptop run is 8.8e-4 nats away
(1e-9 relative): it used the older commit and 12 threads. On the T4, the
latent cache takes 0.30 s, round 0's Adam 0.075 s per step, and the whole
fit 20.3 s, 19× faster than the kaggle CPU.

## Results

_Fits running; results follow the harvest._

## Conclusion

_Pending._
