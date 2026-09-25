**Status:** done — 2026-09-24 → 2026-09-25. CPU round 1 (σ_v rig-wide) is
superseded and only exercised the check tooling. The GPU campaign (4 restarts
× 3 pools, σ_v(k) from wander schema 2) is fitted and reduced; the §3.5
checks run on its fits.

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
  (< 3.01 dB) inside one window. Since `448312f8` the same statistics are also
  computed on every measured order, broken down by the wander's order groups
  (k 1–2, 3–8, 9–24, 25–60, 61+, the groups `sigma_v_db_by_order` is measured
  on; own bootstrap stream). A further column is the paper's disappearance
  rate: over the lines **visible** in their window (≥ 6 dB in at least 20 %
  of its blocks), the share of their blocks under 6 dB.
- **(c) tonality**. On the expectation this is
  `scripts/noise_v2_tonality_audit.py --fit` (new `--fit RIG[_standby]=PATH`
  mode), with the same per-pattern rows as the pinned v2 anchors beside them.
  On rendered audio (`rendered`) there are 8 clips per fit: 2 pattern windows ×
  4 seeds, on the real labels. The measures are R4's `line_width_db3` at
  k = 1..8 and the window-prominence ladder, with the real windows next to
  them. Since `448312f8` each clip, real or rendered, also gets the audit's
  own estimator (`tonality.prominence_db` with the pattern's bin geometry) on
  its time-mean periodogram, so the visible-order counts at 3, 6 and 10 dB
  are one estimator on expectation, renders and real audio; the ladder
  counts are given at all three bars as well.
- **(d) parity**. `scripts/noise_v2_round_score.py --fit` as R5 used it.
- **(e) latents** (`latents`). The fitted `d`/`v`/`u`/`u_j` spread and the
  fitted spread plus the measured block noise, against the measured `σ`. The
  fitted lag-1 is compared with the measured `ρ = e^{-T_b/τ}`. Since
  `448312f8`: `v` also per order group against its own `σ_v(k)`, `τ_v(k)`;
  and the explainer §3.3a toy with the OU prior (a Kalman smoother through
  white block noise `s²`, the measured median line or floor block noise)
  gives each family's posterior variance, so that the check reads as the
  paper states it: fitted mean square + posterior variance against `σ²`,
  and fitted lag-1 product + posterior lag-1 covariance against `ρσ²`.
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
restarts in `restarts/`. The GPU campaign took over `results/noise_v3/fits/`,
so these files now live in `results/noise_v3/fits_cpu_r1/` (same layout).

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

#### Work rate of the flight kernel (approved change 1)

The flight kernel evaluated every line and floor atom on a 64 kHz work grid
(`n_fft_work` 8192 for the 2048-point frame). The likelihood lives on the
16 kHz analysis bins. The 64 kHz grid has no recorded physical reason:
`revised_phase.py:276` gives only "4x the analysis rate, an integer
multiple". It also keeps near-Nyquist line skirts from aliasing, and 32 kHz
does that just as well. `flight_grid(sr_work=)` takes any integer multiple of
the analysis rate, the analysis rate included, and `--work-rate` sets it.

**Kernel check** (`/tmp/gpufit4/workrate_check.py`, laptop CPU under the
cap). The smoke windows (FLY125 @16 s and @32 s, 4 s each) use their
stride-4 frames and the full K the carrier allows (81), with no transfer and
no latents. The expected periodogram at 32 kHz and 16 kHz is compared with the
one at 64 kHz. The table gives the max |ΔdB| over every mic, frame and bin in
each band. "Top orders" keeps the floor and only the lines with
k·f_max > 6 kHz; "shaft orders" keeps the floor and k ≤ 4. The floor alone
deviates by 0.0000 dB at both rates.

| params | variant | rate | 30–500 | 500–2000 | 2000–6000 | 6000–7900 |
|---|---|---|---:|---:|---:|---:|
| DREGON v3 CPU fit (γ ≤ 13.9 Hz, σ_ν 5.25) | full | 32k | 0.0008 | 0.0012 | 0.0019 | 0.0034 |
| | full | 16k | 0.0025 | 0.0037 | 0.0113 | **0.0684** |
| | top orders | 16k | 0.0000 | 0.0072 | 0.0172 | **0.0572** |
| | shaft orders | 16k | 0.0025 | 0.0312 | **0.0649** | 0.0017 |
| R3 v2 cruise fit (v2 widths) | full | 32k | 0.0145 | 0.0094 | 0.0175 | 0.0207 |
| | full | 16k | **0.0706** | 0.0464 | **0.1023** | **0.1782** |
| | top orders | 32k | 0.626 | 0.621 | 0.098 | 0.001 |
| | top orders | 16k | 3.24 | 3.23 | 0.713 | 0.044 |
| v3 CPU smoke fit (K 24) | full | 32k | 0.0024 | 0.0023 | 0.0007 | 0.0001 |
| | full | 16k | 0.0073 | 0.0069 | 0.0020 | 0.0012 |

**16 kHz fails the < 0.05 dB test and 32 kHz passes it.** On realistic v3
parameters (DREGON), 16 kHz misses by 0.068 dB at 6–7.9 kHz. Two
things fold back past the 8 kHz Nyquist there: the skirts of the top orders,
and the tails of the shaft-phase kernel of the low orders (0.065 dB at
2–6 kHz from k ≤ 4 alone). With the v2 cruise widths the error grows to
0.18 dB. Those lines are broadband pedestals, and their Lorentzian tails
alias at every frequency; the top-order-only variant moves by 3.2 dB below
2 kHz at 16k and by 0.6 dB at 32k. That variant is an isolation test:
in the full model those pedestals sit under the rest, and the full model at
32 kHz stays within 0.021 dB in every band. The v3 default work rate is
**32 kHz**.

**Speed-up** (one rig evaluation, forward + gradient, chunked, unit-atom
kernel, 62 frames × 2 mics, laptop CPU with 4 threads, best of 2):

| K | 64 kHz | 32 kHz | 16 kHz |
|---:|---:|---:|---:|
| 24 | 2.43 s | 1.20 s (2.0×) | 0.54 s (4.5×) |
| 81 | 8.32 s | 3.98 s (2.1×) | 2.09 s (4.0×) |

**Render check** (same windows, R3 cruise v2 fit, all 8 mics, 8 seeds per
rate, `render_noise(sr_work=)`). Each render is read as a Hann 2048/512
periodogram. The table compares band-mean dB against the 64 kHz render and
against the model: the 64 kHz kernel, with the 64 kHz chain transfer T64
(the decimator's roll-off: −0.85 dB at 7–7.5 kHz, −3.12 dB at
7.5–7.9 kHz) and without it. "Seed split" is the difference between two
disjoint halves of the seeds at the same rate, i.e. the noise of the
comparison.

| render rate | 1/3 octaves 30–7900 Hz vs 64 kHz render, max \|Δ\| | 6–6.5 / 6.5–7 / 7–7.5 / 7.5–7.9 kHz vs 64 kHz render | same bands vs model × T64 | same bands vs model, no transfer | seed split, same bands |
|---|---:|---|---|---|---|
| 64 kHz | — | — | +0.009 / −0.002 / +0.007 / −0.043 | +0.015 / −0.056 / −0.854 / −3.178 | −0.017 / −0.047 / −0.015 / −0.075 |
| 32 kHz | 0.054 (seed split there 0.074) | −0.034 / +0.002 / +0.025 / +0.059 | −0.025 / 0.000 / +0.032 / +0.015 | −0.019 / −0.054 / −0.830 / −3.119 | +0.025 / +0.038 / −0.021 / +0.004 |
| 16 kHz | **2.09** (7.5 kHz band) | −0.010 / +0.074 / **+0.869** / **+3.220** | −0.001 / +0.072 / +0.876 / +3.177 | +0.005 / +0.018 / +0.014 / +0.042 | +0.012 / +0.054 / +0.012 / −0.055 |

Below 6 kHz every rate matches the model within the seed noise (up to 0.4 dB
in the 125 Hz band, which holds only a few bins). A 32 kHz render cannot be
told from a 64 kHz one: every difference is inside its seed split, and both
carry T64 (T32 and T64 agree to 0.002 dB over 30–7900 Hz, because the
decimator's roll-off is set by the output band). A 16 kHz render has no
decimator. Its chain is flat to 7.9 kHz, so it reproduces the model WITHOUT
the transfer and sits up to 3.2 dB above the 64 kHz render at the top of the
band.

**Decisions.**

- **Fit.** 16 kHz fails the < 0.05 dB kernel test, so the no-transfer 16 kHz
  grid of the addendum does not apply. The `flight_v3` fit evaluates its
  kernel at **32 kHz** (`fit.V3_WORK_RATE`; CLI `--work-rate`, v2 modes
  unchanged at 64 kHz) and keeps the render-chain transfer at that rate,
  which is T64 to 0.002 dB. That buys 2.0–2.1× per rig evaluation.
- **Render.** The lowest rate that passes the render comparison is **32 kHz**.
  `render_noise` and `expected_periodogram` now default to the fit's own
  `front_end.sr_work`: 32 kHz for the GPU v3 fits, 64 kHz for every v2 or
  CPU-round fit, and for payloads that record none. So a fit renders through
  the chain its likelihood carried. `noise_v2_pool` passes its rate
  explicitly and is unchanged.

#### Rig L-BFGS on a frame subset (approved change 2)

Each rig step of the alternation (round 0 and every refit) runs its L-BFGS
on a 64-frame subset, as v2 does. The subset is stratified over the windows
(`model.stratified_frames`): each window gets frames in proportion to its
length, spread evenly across it. Adam still draws minibatches from the whole
pool. After the last round, ONE all-frames L-BFGS polish of the rig runs at
the final latents, warm-started from the subset optimum. The latent step
always uses every frame. `optimiser.polish` records the full-pool
objective (Whittle − rig log prior) at the subset optimum and after the
polish. The CLI defaults are `--lbfgs-frames 64` for `flight_v3`, with
`lbfgs_stratified`; the v2 modes keep their even stride.

Laptop check on the smoke pool (32 kHz, `--lbfgs-frames 24` out of 62
frames, so that the subset is a real one; otherwise the smoke schedule): the
rig evaluation on the subset costs 0.55–0.84 s against 2.0 s on all frames.
The polish ran 1 + 1 iterations and gained 6.56 nats (5.2e-5 nats/cell,
under the 1e-4 tolerance). The total objective is −826 300.30. That is
21.5 nats (1.7e-4 nats/cell) worse than the all-frames 64 kHz smoke.

#### Timing fits (one seed, 3 rounds, `f65179a7`)

One kaggle job per pool, submitted 2026-09-25 12:48 UTC from
`.worktrees/submit-v3gpu`. The command is the v3 pool fit with both approved
changes in effect: 32 kHz kernel, 64-frame stratified rig L-BFGS, one
all-frames polish. It runs `--rounds 3 --max-frames 0 --seed 0 --device
cuda`, with `--channel-gains results/noise_v2/mic_gains/mic_gains.json
--channel-gains-rig <rig> --wander results/noise_v3/wander/<rig>.json`.
DREGON adds `--wind`. Each job warm-starts from its pool's v2 fit
(`--init-from`). All frames means the pool's stride-4 frames, the same
pool and cell count as the CPU round. Harvested with `aws s3 sync` into
`results/noise_v3/fits_gpu/`: `<pool>__flight_v3.json` and
`<pool>_job.log`. Kaggle ran one job at a time ("provider at capacity"), so
the jobs ran back to back. All three ran on a **Tesla T4** (15 GB; driver
580.159.04, torch 2.7.0+cu126), although each job requested a P100. So did
the fp32 bench below: all 6 of the round's kaggle GPU jobs landed on a T4.

"Fit" is `optimiser.wall_s`. "Job" is the wrapper's `total_wall_s`: the
support build, the fit and the CLI start-up. The kaggle session adds about
95 s of environment setup before the job starts (`preparing` → `running`).
Rig s/eval is one forward + gradient evaluation. It is measured on the
64-frame subset in rounds 0–3 and on all frames in the polish. L-BFGS
iterations are first pass + restart. Every rig L-BFGS ended far below the
200-iteration cap. The record keeps no stop reason; the 1e-5 relative
tolerance is the stop the schedule sets for these early exits. Peak memory is
`torch.cuda.max_memory_allocated` in MiB. The latent step sets it on DREGON
and cruise, the rig step on standby.

| pool | windows / frames / mics / K / cells | fit s | job s (build) | rig s/eval, rounds 0 / 1 / 2 / 3 · polish | L-BFGS iterations, rounds 0 / 1 / 2 / 3 · polish | Adam s/step (300 steps) | latent step: evals × s/eval, cache s | peak GPU MB (rig step) | `which_converged` | total objective (nats) | σ_ν rad/s |
|---|---|---:|---:|---|---|---:|---|---:|---|---:|---:|
| DREGON (`nv3gpu4-dregon-298747`) | 5 / 310 / 8 / 88 / 2 499 840 | 159.8 | 223 (55) | 0.808 / 0.992 / 0.976 / 0.983 · 4.89 | 14+1 / 1+1 / 2+1 / 2+1 · 1+1 | 0.117 | 105 × 0.043, 2.5–2.6 | 2 215 (1 394–1 455) | lbfgs | −18 027 353.4 | 5.432 |
| Michael's cruise (`nv3gpu4-cruise-61d712`) | 8 / 496 / 8 / 81 / 3 999 744 | 154.2 | 189 (27) | 0.503 / 0.530 / 0.538 / 0.542 · 4.11 | 2+2 / 4+1 / 4+1 / 4+1 · 4+1 | 0.065 | 103–109 × 0.061, 3.8–3.9 | 2 756 (1 131–1 468) | lbfgs | −24 806 693.7 | 4.280 |
| Michael's standby (`nv3gpu4-standby-6eb132`) | 3 / 93 / 8 / 130 / 749 952 | 131.0 | 161 (24) | 1.258 / 1.432 / 1.441 / 1.446 · 2.10 | 2+4 / 1+1 / 1+1 / 1+1 · 1+1 | 0.161 | 104–105 × 0.023, 1.1 | 1 677 (1 617–1 677) | lbfgs | −6 947 035.5 | 0.290 |

Whittle term per round (round 0 is the rig with the latents at zero) and the
alternation's move per cell. The tolerance is 1e-4 nats/cell:

| pool | round 0 | round 1 (move) | round 2 (move) | round 3 (move) | polish (gain/cell) |
|---|---:|---:|---:|---:|---:|
| DREGON | −17 800 116.2 | −18 074 433.0 (0.1097) | −18 078 465.6 (0.0016) | −18 079 299.2 (3.3e-4) | −18 079 516.3 (8.8e-5) |
| cruise | −24 427 766.7 | −24 878 314.1 (0.1126) | −24 885 767.2 (0.0019) | −24 888 418.9 (6.6e-4) | −24 891 425.7 (7.6e-4) |
| standby | −6 928 454.0 | −6 971 762.9 (0.0577) | −6 973 350.5 (0.0021) | −6 973 865.5 (6.9e-4) | −6 973 866.3 (1.5e-6) |

Widths (`params.gamma_hz`; rotors numbered from 1; γ0 k = 0.01k Hz):

| pool | γ median Hz | γ max per rotor Hz (k) | γ > 50 Hz | max γ/(0.01k) | lines > 5 γ0 k | max γ at k ≤ 8 Hz |
|---|---:|---|---:|---|---:|---:|
| DREGON | 1.668 | 19.34 (69), 40.94 (26), 10.40 (43), 24.37 (70) | 0 | 157.4 (rotor 2, k = 26, 40.9 Hz) | 141/352 | 4.57 |
| cruise | 0.893 | 28.11 (49), 12.25 (17), 2.81 (15), 3.20 (15) | 0 | 72.0 (rotor 2, k = 17, 12.2 Hz) | 100/324 | 2.71 |
| standby | 2.660 | 17.06 (130), 11.71 (81), 95.58 (128), 13.08 (85) | 3 (rotor 3, k = 126 / 127 / 128: 59.6 / 71.5 / 95.6 Hz) | 74.7 (rotor 3, k = 128, 95.6 Hz) | 244/520 | 2.92 |

- **Convergence.** `which_converged` is "lbfgs" in all three pools:
  the all-frames polish met the 1e-4 restart-gain test (restart gain
  2.1e-5 / 1.4e-5 / 4.3e-7 nats/cell). The alternation did not converge in
  any pool: round 3 still moved 3.3 / 6.6 / 6.9 × the tolerance. The subset
  rig steps failed the restart-gain test in DREGON round 1, in cruise
  rounds 0–3 (restart gain 9.0e-4 nats/cell in round 0), and in standby
  round 0 (6.6e-4).
- **Subset vs polish.** The polish gains 221.2 nats on DREGON
  (8.8e-5/cell), 3 043.7 on cruise (7.6e-4/cell, 7.6 × the tolerance) and
  1.1 on standby. On cruise, a 64-frame subset of 496 frames leaves its rig
  optimum measurably off the full-pool one.
- **Objective.** DREGON: Whittle −18 079 516.3 + rig −log prior 5 741.5 + OU
  −log prior 46 421.4. Cruise: −24 891 425.7 + 4 572.0 + 80 160.1. Standby:
  −6 973 866.3 + 4 840.8 + 21 990.0. These totals cannot be ranked against
  the CPU round: that round used σ_v rig-wide, a 64 kHz kernel and an
  all-frames rig L-BFGS. The CPU round's best DREGON restart is lower by
  11 240 nats (4.5e-3/cell). Its best standby restart is lower by 1 754
  nats (2.3e-3/cell).
- **Span pins.** Cruise pins `amp_exp`, `floor_exp` and `floor_static_rel`
  (speed span 1.436 < 1.5). DREGON (1.744) and standby (1.928) pin nothing.
- **`init_from` clip record.** Each v2 site was clipped to the 0.001–0.999
  prior box.
  - **DREGON**, from R5 `flight_profile` (`1c6ac337`). Sites `amp_exp`,
    `floor_exp`, `floor_static_rel`, `gamma_hz`, `profile_db`, `sigma_nu`.
    Clipped: γ 246 of 352, `profile_db` 2, `floor_exp` 1,
    `floor_static_rel` 1; σ_ν, `amp_exp` and `floor_shape_z` 0. The floor
    starts from the measurement: objective −17 592 338.7, against
    −15 542 065.2 at the v2 floor. v2 λ 0.0608 is not carried (λ is pinned).
  - **Cruise**, from R3 `flight` (`a996e5e5`). Sites `gamma_hz`,
    `profile_db`, `sigma_nu`. Clipped: γ 164 of 324, `profile_db` 3, σ_ν 1,
    `floor_shape_z` 6. The floor starts from the measurement (−24 192 501.9,
    against −23 651 922.6 at the v2 floor).
  - **Standby**, from R3 `flight` (`58e895fb`). Sites `amp_exp`,
    `floor_exp`, `floor_shape_z`, `floor_static_rel`, `gamma_hz`,
    `profile_db`, `sigma_nu`. Clipped: γ 180 of 520, `floor_static_rel` 1,
    all others 0. The floor starts from the v2 floor re-expressed
    (−6 325 922.7, against −5 037 953.2 measured).
- **Time split** (DREGON; the other pools are alike). Round 0 takes 72.5 s:
  35 s of Adam and 17 s of subset L-BFGS. Rounds 1–3 take 16–17 s each, of
  which 7 s is the latent step. The all-frames polish takes 37 s.

#### fp32 unit-atom kernel (`nv3gpu4-fp32bench-a7aed3`, T4)

`/tmp/gpufit4/fp32_bench.py` measures without implementing anything. It
uses the smoke windows at all 8 mics, the full K 81, 62 frames, 32 kHz and
the seed start point. It times one full rig evaluation in fp64 (as shipped),
then replays the kernel's no-grad atom block per chunk in complex128 and in
complex64: unit atoms, the 2n FFT, |·|², the inverse, and the order sum
with the lag law.

| rig s/eval, fp64 | atom block c128 | atom block c64 | block share of eval | projected eval, fp32 block | projected speed-up | block rel. error (max-abs) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.352 | 0.297 s | 0.086 s (3.45×) | 84.5 % | 0.141 s | **2.50×** | 7.8e-6 |

**Objective difference** (`/tmp/gpumon/fp32_obj.py`, laptop CPU under the
cap, one no-grad evaluation each). The same pool and point were used, with
the atom block in complex64 and every accumulation from the lag sum on in
fp64. The fp64 objective is −2 714 463.8343 nats and the fp32-block
objective −2 714 463.8236: a difference of **+0.0107 nats**, 3.9e-9
relative, 2.1e-8 nats/cell over 499 968 cells. That is 5 000 × under the
1e-4 nats/cell tolerance. It was measured at one point, the seed start, not
at an optimum.

#### Does 4 restarts × 3 rounds fit in 1 h per pool?

**Yes, on the T4 kaggle allocates, with a margin of 4.6–5.6×.** Restarts
run one after another in one job, and the support build and the kaggle
setup (≈ 95 s) are paid once:

| pool | 4 × fit s | + build s | + kaggle setup s | total s (min) | share of 1 h |
|---|---:|---:|---:|---:|---:|
| DREGON | 639 | 55 | 95 | 789 (13.2) | 22 % |
| cruise | 617 | 27 | 95 | 739 (12.3) | 21 % |
| standby | 524 | 24 | 95 | 643 (10.7) | 18 % |

All three pools × 4 restarts would fit in one 1 h job as well: 1 981 s,
33 min. No job got a P100, so no P100 number was measured. The P100's fp64
rate is about 19× the T4's (4.7 against 0.25 TFLOPS). The fp32 bench shows
the rig evaluation is dominated by the fp64 atom block (84.5 %), so the T4
times are an upper bound for a P100 [inference]. The margin covers the
current stopping rules only. The rtol stop ends each rig L-BFGS after
1–14 iterations, and the alternation is still 3–7 × off its tolerance after
round 3. One extra round costs 15–19 s per restart. Within 1 h a restart may
take ≈ 860 s ((3 600 − 150) / 4), 5.4–6.6 × the measured single-seed wall.

**DREGON widths are physical.** No line is wider than 50 Hz. The widest,
40.9 Hz at rotor 2, k = 26, is γ/(0.01k) = 157. The v2 R5 fit it
warm-starts from has 7 223 Hz at k = 65 on rotor 1: γ/(0.01k) = 11 113. Its
own maximum is 16 059 Hz at rotor 4, k = 83 (γ/(0.01k) = 19 348), and it has
19 lines over 50 Hz. The v3 widths are 70–120 × narrower in that ratio.
The warm start clipped 246 of the 352 v2 widths into the prior box, and the
v3 fit did not escape to broadband pedestals again. The widths are still
wider than the CPU v3 round's (max γ/(0.01k) 81.6, γ median 0.72 Hz),
and 141 of 352 lines exceed 5 γ0 k.

### Campaign (4 restarts × 3 pools, schema-2 wander)

One kaggle job per pool, submitted 2026-09-25 14:11 UTC from
`.worktrees/submit-v3gpu` detached at `a7a6156c` (the model code of
`f65179a7`; the wander files of `336fcf21`, `noise-v3-wander/2` with
`sigma_v_db_by_order`). The wrapper is the timing runs' (support build once,
R2 sync every 10 min and before any summary, every `grid/raw/*.err`/`*.log`
echoed on a failure), with the out dir `results/noise_v3/fits` and a loop
over the four seeds. Each seed runs

```
python scripts/noise_v2_fit.py flight --mode flight_v3 --set <set> --name <name> \
  --wander results/noise_v3/wander/<rig>.json \
  --channel-gains results/noise_v2/mic_gains/mic_gains.json --channel-gains-rig <rig> [--wind] \
  --init-from <v2 fit of the pool> --rounds 5 --lbfgs-frames 64 --lbfgs-iters 500 \
  --max-frames 0 --seed N --restart-tag sN --jobs 1 --threads 4 --device cuda \
  --progress 50 --out results/noise_v3/fits --grid-dir results/noise_v3/fits/grid/<name>_sN
```

Against the timing fits this changes three things: 5 alternation rounds
instead of 3; four seeds (`s0` from the warm start, `s1`–`s3` with the
log-normal jitter, sd 0.8, of `σ_ν` and `γ` that `4890fc4e` gives a
`--restart-tag` seed); and the rig L-BFGS cap raised from 200 to 500
iterations (restart pass 250), so that no rig step, the all-frames polish
included, can stop on the cap before its 1e-5 relative tolerance. The v2
warm starts are the timing runs': R5 `flight_profile` for DREGON, R3
`flight` for cruise and standby. Priors are the committed defaults
(`gamma_c` 3, `sigma_nu_scale` 0.6).

| job | pool | submitted (UTC) | state |
|---|---|---|---|
| `nv3c-dregon-26c719` | DREGON (`dregon-floor`, `--wind`) | 14:11 | succeeded 14:28, Tesla T4, 871 s to the last fit (build 53 s) |
| `nv3c-cruise-0a9885` | Michael's cruise (`michaels-cruise`) | 14:11 | started 14:28 (kaggle runs one job at a time), succeeded 14:44, Tesla T4, 805 s to the last fit (build 26 s) |
| `nv3c-standby-cf4505` | Michael's standby (`michaels-standby`) | 14:11 | started 14:44, succeeded 14:56, Tesla T4, 711 s to the last fit (build 22 s) |

**Wall time and GPU.** All three jobs ran on a **Tesla T4** (15 GB, driver
580.159.04, torch 2.7.0+cu126), although each requested a P100. Kaggle ran
them back to back; the whole campaign took 45 min of wall clock
(14:11–14:56 UTC). "Fit s" is `optimiser.wall_s` per seed; "to the last
fit" is the wrapper's `total_wall_s` after seed 3 (support build included,
the ≈ 95 s kaggle session setup not).

| pool | job | GPU | build s | fit s, s0 / s1 / s2 / s3 | job s to the last fit | peak GPU MiB |
|---|---|---|---:|---|---:|---:|
| DREGON | `nv3c-dregon-26c719` | Tesla T4 | 53 | 207 / 187 / 189 / 200 | 871 | 2 215 |
| Michael's cruise | `nv3c-cruise-0a9885` | Tesla T4 | 26 | 192 / 183 / 181 / 180 | 805 | 2 756 |
| Michael's standby | `nv3c-standby-cf4505` | Tesla T4 | 22 | 164 / 159 / 158 / 163 | 711 | 1 677 |

**Per restart** (`results/noise_v3/fits/restarts/<pool>__flight_v3__s<N>.json`).
The move is the Whittle move per observed cell between alternation rounds
(tolerance 1e-4). L-BFGS iterations are first pass + restart pass, per rig
step (round 0 is the first rig fit, rounds 1–5 the subset refits) and for
the all-frames polish. The polish gain is the full-pool objective it gains
over the subset optimum. Widths are `params.gamma_hz`, rotors numbered
from 1, γ0 k = 0.01k Hz.

| pool | seed | fit s | `which_converged` | move/cell, rounds 1–5 | rig L-BFGS iterations, rounds 0–5 · polish | polish gain nats (/cell) | total objective (nats) | σ_ν rad/s | γ median Hz | γ max per rotor Hz (k) | γ > 50 Hz | max γ/(0.01k) (rotor, k, Hz) | lines > 5 γ0 k |
|---|---:|---:|---|---|---|---|---:|---:|---:|---|---:|---|---|
| DREGON | 0 | 207 | lbfgs | 0.12, 0.0059, 0.0028, 0.0017, 0.001 | 14+1 / 1+1 / 2+1 / 2+1 / 2+1 / 2+1 · 5+1 | 1301.5 (0.00052) | −18 038 244.8 | 4.726 | 1.668 | 19.28 (69), 38.48 (26), 10.36 (43), 24.09 (70) | 0 | 148.0 (2, 26, 38.48) | 141/352 (40.1 %) |
| DREGON | 1 | 187 | lbfgs | 0.12, 0.0057, 0.0029, 0.0016, 0.001 | 3+1 / 2+1 / 2+1 / 2+1 / 2+1 / 2+1 · 2+1 | 593.8 (0.00024) | −18 041 276.8 | 4.561 | 1.567 | 21.56 (69), 43.54 (26), 13.08 (55), 23.95 (70) | 0 | 167.5 (2, 26, 43.54) | 147/352 (41.8 %) |
| DREGON | 2 | 189 | lbfgs | 0.12, 0.0057, 0.003, 0.0017, 0.00092 | 4+1 / 2+1 / 2+1 / 2+1 / 2+1 / 2+1 · 2+1 | 580.4 (0.00023) | −18 041 310.6 | 4.573 | 1.615 | 22.82 (69), 44.67 (26), 17.6 (55), 24.27 (70) | 0 | 171.8 (2, 26, 44.67) | 145/352 (41.2 %) |
| DREGON | 3 | 200 | lbfgs | 0.12, 0.0055, 0.0027, 0.0017, 0.0011 | 2+2 / 2+1 / 2+1 / 2+1 / 2+1 / 2+1 · 5+1 | 1192.1 (0.00048) | −18 040 823.7 | 4.751 | 1.273 | 18.15 (74), 43.63 (26), 7.2 (71), 14.15 (70) | 0 | 167.8 (2, 26, 43.63) | 118/352 (33.5 %) |
| cruise | 0 | 192 | lbfgs | 0.12, 0.0037, 0.0017, 0.00088, 0.00059 | 2+2 / 3+1 / 3+1 / 3+1 / 3+1 / 3+1 · 4+1 | 1755.5 (0.00044) | −24 805 561.3 | 3.922 | 0.893 | 27.58 (49), 11.93 (17), 2.82 (15), 3.18 (15) | 0 | 70.2 (2, 17, 11.93) | 100/324 (30.9 %) |
| cruise | 1 | 183 | lbfgs | 0.12, 0.0037, 0.0015, 0.001, 0.00097 | 8+1 / 3+1 / 3+1 / 3+1 / 3+1 / 2+1 · 1+1 | 253.9 (6.3e-05) | −24 805 328.5 | 3.865 | 0.779 | 29.42 (49), 12.78 (17), 4.16 (7), 4.74 (30) | 0 | 75.2 (2, 17, 12.78) | 97/324 (29.9 %) |
| cruise | 2 | 181 | lbfgs | 0.11, 0.0036, 0.0018, 0.00092, 0.00093 | 4+1 / 4+1 / 4+1 / 3+1 / 3+1 / 2+1 · 1+1 | 236.3 (5.9e-05) | −24 805 567.3 | 3.837 | 0.808 | 29.98 (49), 12.98 (17), 3.32 (7), 3.39 (14) | 0 | 76.4 (2, 17, 12.98) | 101/324 (31.2 %) |
| cruise | 3 | 180 | lbfgs | 0.12, 0.0038, 0.0016, 0.0009, 0.001 | 4+1 / 4+1 / 4+1 / 3+1 / 3+1 / 1+1 · 1+1 | 223.0 (5.6e-05) | −24 804 482.5 | 4.021 | 0.633 | 30.38 (49), 14.78 (17), 3.49 (80), 3.65 (49) | 0 | 87.0 (2, 17, 14.78) | 97/324 (29.9 %) |
| standby | 0 | 164 | lbfgs | 0.062, 0.0043, 0.0014, 0.00046, 0.00024 | 2+4 / 1+1 / 1+1 / 1+1 / 1+1 / 1+1 · 1+1 | 1.7 (2.2e-06) | −6 943 253.8 | 0.290 | 2.660 | 17.06 (130), 11.71 (81), 95.51 (128), 13.08 (85) | 3 | 74.6 (3, 128, 95.51) | 244/520 (46.9 %) |
| standby | 1 | 159 | lbfgs | 0.061, 0.0048, 0.0011, 0.00068, 0.00014 | 2+1 / 1+1 / 1+1 / 1+1 / 1+1 / 1+1 · 1+1 | 1.5 (1.9e-06) | −6 943 419.4 | 0.300 | 2.670 | 18.19 (108), 11.22 (81), 89.34 (128), 12.56 (85) | 3 | 69.8 (3, 128, 89.34) | 241/520 (46.3 %) |
| standby | 2 | 158 | lbfgs | 0.062, 0.0044, 0.0016, 0.00042, 0.00028 | 1+1 / 1+1 / 1+1 / 1+1 / 1+1 / 1+1 · 1+1 | 1.5 (2e-06) | −6 943 448.2 | 0.348 | 2.262 | 17.27 (108), 10.57 (51), 86.44 (128), 12.44 (85) | 3 | 67.5 (3, 128, 86.44) | 218/520 (41.9 %) |
| standby | 3 | 163 | lbfgs | 0.062, 0.0045, 0.0012, 0.00058, 0.0002 | 3+1 / 1+1 / 1+1 / 1+1 / 1+1 / 1+1 · 1+1 | 1.2 (1.6e-06) | −6 943 361.7 | 0.320 | 2.474 | 17.24 (126), 11.23 (51), 87.37 (128), 11.93 (85) | 3 | 68.3 (3, 128, 87.37) | 235/520 (45.2 %) |

**Restart spread** (`reduce`; the reported fit is the restart with the lowest
total objective):

| pool | restarts | selected | best − median /cell | best − worst /cell | σ_ν min–max rad/s | log-mean γ min–max Hz | restart gain of the selected fit's polish /cell |
|---|---:|---:|---:|---:|---|---|---:|
| DREGON | 4 | s2 | 1.0e-4 | 1.2e-3 | 4.561–4.751 | 0.901–1.237 | 6.1e-5 |
| Michael's cruise | 4 | s2 | 3.1e-5 | 2.7e-4 | 3.837–4.021 | 0.467–0.656 | 1.8e-5 |
| Michael's standby | 4 | s2 | 7.7e-5 | 2.6e-4 | 0.290–0.348 | 1.420–1.704 | 7.5e-7 |

- **Convergence.** `which_converged` is "lbfgs" in all twelve restarts: the
  all-frames polish passed the restart-gain test (restart gain
  4.4e-5–7.0e-5 /cell on DREGON, 1.1e-5–1.9e-5 on cruise, 5.7e-7–7.5e-7 on
  standby). `optimiser.converged` is false in all twelve, because the
  alternation never converged in 5 rounds. Round 5 still moved
  9.2e-4–1.1e-3 /cell on DREGON (9–11 × the tolerance), 5.9e-4–1.0e-3 on
  cruise (6–10 ×) and 1.4e-4–2.8e-4 on standby (1.4–2.8 ×). On DREGON the
  move falls by a factor 0.53–0.57 per round from round 2 on; at that rate
  it reaches 1e-4 after about four more rounds [inference].
- **Iterations used.** Every rig L-BFGS stopped on its 1e-5 relative
  tolerance; none came near the 500-iteration cap. Round 0 used 1–14
  iterations, the subset refits of rounds 1–5 used 1–4, and the all-frames
  polish used 1–5 (+1 restart). Raising the cap from 200 to 500 therefore
  changed nothing: the polish runs to its rtol in at most 5 iterations. Its
  gain is 580–1 302 nats (2.3e-4–5.2e-4 /cell, 2–5 × the tolerance) on
  DREGON, 223–1 756 nats (5.6e-5–4.4e-4 /cell) on cruise and 1.2–1.7 nats
  (≈ 2e-6 /cell) on standby.
- **Restart spread.** DREGON's restarts disagree by up to 1.2e-3 /cell,
  12 × the tolerance; its seed 0 (the unjittered warm start) is the worst,
  3 066 nats above s2. Cruise and standby disagree by 2.7e-4 and 2.6e-4
  /cell (2.6–2.7 ×). σ_ν agrees within 5 % (DREGON, cruise) and 20 %
  (standby); the log-mean width within 1.2–1.4 ×.
- **Objective (selected, s2 in every pool).** DREGON: total
  −18 041 310.6 = Whittle −18 122 294.7 + rig −log prior 5 847.7 + OU −log
  prior 75 136.4, over 2 499 840 cells. Round 0 (latents at zero) had
  Whittle −17 795 035.0, so rounds 1–5 gain 327 259.7 nats
  (0.131 /cell). Cruise: −24 805 567.3 = −24 919 012.8 + 4 938.1 +
  108 507.4 (3 999 744 cells; rounds 1–5 gain 489 112.6, 0.122 /cell).
  Standby: −6 943 448.2 = −6 979 574.4 + 4 589.6 + 31 536.5 (749 952 cells;
  gain 51 527.5, 0.069 /cell). Against the timing fits the Whittle term is
  lower by 42 778 (DREGON), 27 587 (cruise) and 5 708 nats (standby), and
  the OU prior term is larger (DREGON 46 421 → 75 136). The totals do not
  rank: the timing fits used the schema-1 wander (σ_v 1.15 dB rig-wide).
- **Span pins.** Cruise pins `amp_exp`, `floor_exp` and `floor_static_rel`
  (speed span 1.436); DREGON (1.744) and standby (1.928) pin nothing.
- **Widths (selected).** DREGON: γ median 1.615 Hz; rotor maxima 22.82 /
  44.67 / 17.60 / 24.27 Hz at k = 69 / 26 / 55 / 70; no line over 50 Hz; the
  largest γ/(0.01k) is **171.8** (rotor 2, k = 26, 44.7 Hz; 148–172 over the
  restarts); 145 of 352 lines (41.2 %) exceed 5 γ0 k; the widest line at
  k ≤ 8 is 4.65 Hz. Cruise: median 0.808 Hz; maxima 29.98 / 12.98 / 3.32 /
  3.39 Hz at k = 49 / 17 / 7 / 14; none over 50 Hz; largest γ/(0.01k)
  **76.4** (rotor 2, k = 17, 13.0 Hz; 70–87); 101 of 324 (31.2 %) over
  5 γ0 k; widest at k ≤ 8 3.32 Hz. Standby: median 2.262 Hz; maxima 17.27 /
  10.57 / 86.44 / 12.44 Hz at k = 108 / 51 / 128 / 85; **3 lines over
  50 Hz** (rotor 3, k = 126 / 127 / 128: 61.7 / 67.5 / 86.4 Hz, the same
  three as in the timing fit); largest γ/(0.01k) **67.5** (rotor 3,
  k = 128; 67.5–74.6); 218 of 520 (41.9 %) over 5 γ0 k; widest at k ≤ 8
  3.15 Hz.
- **σ_ν (selected).** DREGON **4.573** rad/s, cruise **3.837**, standby
  0.348: 7.6 ×, 6.4 × and 0.58 × the scale of the HalfNormal prior
  (0.6 rad/s).
- **Latent tracks against the schema-2 σ** (selected fit; rms of the fitted
  block latents / the measured σ, dB; `v` per order group). Check (e) below
  adds the posterior variance.

| pool | d | u | u_j | v k 1–2 | v k 3–8 | v k 9–24 | v k 25–60 | v k ≥ 61 |
|---|---|---|---|---|---|---|---|---|
| DREGON | 1.80 / 0.94 | 2.45 / 2.83 | 3.97 / 1.52 | 0 / 0 | 0 / 0 | 4.08 / 2.98 | 4.87 / 3.91 | 4.58 / 7.18 |
| Michael's cruise | 1.02 / 0.52 | 3.41 / 1.73 | 3.61 / 1.63 | 0.42 / 0.32 | 5.80 / 4.17 | 5.18 / 3.79 | 4.03 / 3.61 | 3.56 / 5.69 |
| Michael's standby | 0.91 / 0.52 | 1.74 / 1.73 | 1.96 / 1.63 | 0.42 / 0.32 | 4.51 / 4.17 | 3.74 / 3.79 | 2.82 / 3.61 | 3.97 / 5.69 |

  The rotor-common `d` spreads 1.75–1.96 × its measured σ in every pool,
  and DREGON's floor colour `u_j` 2.6 ×. The per-line `v` spreads 1.1–1.4 ×
  its σ at k 3–60 on DREGON and cruise, 0.8–1.1 × on standby, and 0.6–0.7 ×
  at k ≥ 61 in every pool, where 94–99 % of the held-out line blocks are
  under the floor (check (b)) and the latents are shrunk toward zero.

## Results

The campaign fits are `results/noise_v3/fits/<pool>__flight_v3.json` (s2 in
every pool). The checks ran on the laptop under the cap, with the tooling of
`448312f8`: `noise_v3_checks.py prior|heldout|rendered|latents|summary`,
`noise_v2_tonality_audit.py --fit`, `noise_v2_round_score.py --round 6 …
--fit` (the R5 parity command), `noise_v3_param_view.py`. Every output is
under `results/noise_v3/checks/`; `findings.md` there has every table in
full, with window-bootstrap intervals (median [5 %, 95 %]). Renders use the
real label of each window, 8 mics and the frozen seeds 2001–2004.

### §3.5 table, DREGON (held-out = the five frozen room-2 score windows, 4 s)

| check | statistic | real | v3 render | v2 render (R5) |
|---|---|---|---|---|
| (a) | prior draws: max γ/(γ0 k); max \|c_j − μ\|/σ_B; max shape sd/σ_B | — | 11.80; 3.01; 1.48 (σ_B 17.60 dB) | — |
| (b) | wander σ_total of the resolvable line tracks, dB | no resolvable track (0) | 3.40 [2.83, 3.64] (16 tracks) | no track |
| (b) | disappearance rate, visible lines k 8–24 | 68.8 % [66.4, 72.5] (10 lines) | 66.2 % [63.3, 68.4] (37) | no visible line |
| (b) | lines that appear and disappear, k 8–24 | 10.9 % [6.2, 15.3] | 7.2 % [6.2, 8.2] | 1.8 % [0.0, 3.7] |
| (b) | block cells under the floor, k 8–24 | 81.8 % | 87.9 % | 92.2 % |
| (b) | floor level σ_u, dB | 3.66 | 0.88 | 0.54 |
| (c) | R4 width k = 1, 2, 7, 8, Hz (rendered clips) | 10.4, 23.4, 30.7, 30.2 | 9.5, 17.6, 17.6, 17.6 | 10.3, 10.2, 6.8, 10.3 |
| (c) | audit orders ≥ 3 / 6 / 10 dB per rotor, clips (narrow; wide) | 7.5/1.0/0.0; 5.5/1.0/0.0 | 3.8/2.0/0.0; 3.2/2.0/0.2 | 3.0/2.0/0.0; 2.6/1.0/0.0 |
| (c) | audit on the expectation (narrow; wide) | — | 7.0/3.2/0.0; 5.2/2.5/0.0 | anchor 23.2/9.5/2.0; 17.2/6.8/0.2 |
| (d) | HPPNet PIT MAE, rev/s (bar 2.187786) | 1.218708 | **1.716079** (upper 2.062063) | 1.820065 (upper 2.158331) |
| (d) | proxy `ltas_abs_db` (gate 1.9786) | — | 2.4302 | 2.0298 |

### §3.5 table, Michael's (held-out = FLY124 standby @8/16 s, cruise @40/56 s, 8 s)

| check | statistic | real | v3 render | v2 render (R3) |
|---|---|---|---|---|
| (a) | prior draws, cruise fit: max γ/(γ0 k); max \|c_j − μ\|/σ_B; max shape sd/σ_B | — | 11.26; 3.24; 1.63 (σ_B 3.95 dB) | — |
| (a) | prior draws, standby fit: same | — | 11.66; 3.07; 1.53 (σ_B 13.01 dB) | — |
| (b) | wander σ_total of the dominant line tracks, dB | 1.46 [0.53, 1.50] (5 tracks) | 3.16 [2.59, 3.47] (22) | 0.38 [0.28, 0.44] (30) |
| (b) | block sd of a present line, k 8–24, dB | 2.76 [2.57, 3.91] | 2.62 [2.52, 2.76] | 1.18 [1.14, 1.21] |
| (b) | disappearance rate, visible lines k 8–24 | 38.2 % [32.9, 51.1] (38) | 42.9 % [38.9, 58.5] (181) | 33.9 % [29.9, 38.6] (170) |
| (b) | lines that appear and disappear, k 8–24 | 22.1 % [3.3, 39.8] | 27.6 % [4.4, 44.1] | 17.0 % [0.0, 32.8] |
| (b) | floor level σ_u, dB | 2.97 | 0.83 | 0.21 |
| (c) | audit orders ≥ 3 / 6 / 10 dB, cruise clips (narrow; wide) | 8.8/3.0/1.2; 11.8/6.8/3.2 | 8.5/3.1/1.4; 11.4/5.9/3.4 | 7.8/1.9/1.2; 10.5/5.5/2.8 |
| (c) | same, standby clips (narrow; wide) | 7.0/3.5/2.8; 4.5/1.2/0.0 | 18.6/5.2/2.2; 8.0/2.6/1.4 | 8.9/3.9/2.0; 4.4/2.0/1.2 |
| (c) | audit on the expectation, cruise; standby (narrow / wide) | — | 18.0/9.8/4.2, 18.8/11.0/3.8; 8.8/4.0/2.0, 13.0/3.8/1.2 | anchors 15.0/8.0/4.2, 17.0/8.2/3.8; 8.5/4.2/2.8, 11.5/3.8/1.5 |
| (d) | HPPNet equal-regime PIT MAE, rev/s (bar 3.177994 = 1.05 × 3.026661) | 1.3676 | **1.686297** (ratio 0.557) | 1.621670 (ratio 0.536) |
| (d) | per regime standby / ramp / cruise | 0.344 / 3.171 / 0.588 | 0.885 / 3.283 / 0.891 | 0.785 / 3.267 / 0.813 |
| (d) | proxy `ltas_abs_db` (gate 1.2197) | — | 2.3656 | 1.2816 |

### Disappearance rate per order group (check (b), the headline)

Share of blocks under 6 dB among the lines visible in their window (≥ 6 dB
in ≥ 20 % of its blocks); number of visible lines in brackets (renders:
4 seeds per window); "a+d" = share of all lines that both reach 6 dB and fall
under 3.01 dB inside one window.

| rig | orders (σ_v dB) | real | v3 render | v2 render | a+d real / v3 / v2 |
|---|---|---|---|---|---|
| DREGON | 1–2 (0) | 71.9 % [68.8, 75.0] (8) | 75.0 % (3) | 75.0 % (3) | 50.0 / 1.2 / 8.8 % |
| DREGON | 3–8 (0) | no visible line | none | none | 0 / 0 / 0 % |
| DREGON | 9–24 (2.98) | 68.8 % [66.2, 71.9] (10) | 66.4 % [63.3, 68.5] (37) | none | 11.6 / 7.7 / 2.0 % |
| DREGON | 25–60 (3.91) | 72.7 % [68.8, 75.0] (10) | 44.4 % [41.1, 47.1] (94) | none | 6.8 / 6.0 / 0.0 % |
| DREGON | 61+ (7.18) | 75.0 % (2) | 75.0 % (4) | none | 1.6 / 2.1 / 0.0 % |
| DREGON | all orders (point) | 71.2 % (30) | 51.7 % (138) | 75.0 % (3) | |
| Michael's | 1–2 (0.32) | 5.6 % [0.0, 10.3] (9) | 1.0 % [0.0, 1.9] (33) | 0.0 % (32) | 10.0 / 5.0 / 0.0 % |
| Michael's | 3–8 (4.17) | 23.7 % [18.8, 29.5] (24) | 27.5 % [23.6, 35.2] (145) | 17.6 % [13.3, 21.3] (120) | 25.0 / 36.2 / 16.4 % |
| Michael's | 9–24 (3.79) | 40.4 % [34.6, 54.4] (34) | 44.9 % [41.2, 68.8] (153) | 37.2 % [33.8, 41.5] (146) | 21.5 / 26.9 / 17.1 % |
| Michael's | 25–60 (3.61) | 65.6 % (2) | 68.6 % [68.2, 69.2] (36) | none | 3.5 / 10.5 / 0.7 % |
| Michael's | 61+ (5.69) | 25.8 % [25.6, 26.0] (23) | 48.6 % (223) | 30.4 % [29.6, 31.2] (149) | 1.6 / 6.9 / 2.2 % |
| Michael's | all orders (point) | 29.6 % (92) | 41.0 % (590) | 27.0 % (447) | |

### Latent consistency (check (e), selected fits)

Fitted mean square + OU-smoother posterior variance (the lag-0 sum) against
the measured σ², and fitted lag-1 product + posterior lag-1 covariance
against ρσ², in dB², as ratio sum / measured:

| family | DREGON lag 0 | DREGON lag 1 | cruise lag 0 | cruise lag 1 | standby lag 0 | standby lag 1 |
|---|---:|---:|---:|---:|---:|---:|
| d | 3.75/0.89 (4.2) | 1.71/0.17 (10) | 1.26/0.28 (4.5) | 0.89/0.12 (7.4) | 1.04/0.28 (3.7) | 0.50/0.12 (4.2) |
| v, all orders | 20.5/24.2 (0.84) | 12.6/12.9 (0.98) | 19.4/18.3 (1.06) | 14.0/11.5 (1.22) | 14.5/23.6 (0.61) | 9.6/15.0 (0.64) |
| v k 3–8 | σ = 0 | σ = 0 | 34.7/17.4 (2.0) | 27.0/11.3 (2.4) | 21.5/17.4 (1.23) | 15.9/11.3 (1.41) |
| v k 9–24 | 17.7/8.9 (2.0) | 11.3/4.7 (2.4) | 28.0/14.4 (1.9) | 19.7/8.1 (2.4) | 15.1/14.4 (1.05) | 8.8/8.1 (1.09) |
| v k 25–60 | 24.8/15.3 (1.6) | 16.1/6.8 (2.4) | 17.3/13.0 (1.3) | 12.7/8.1 (1.6) | 9.0/13.0 (0.69) | 5.2/8.1 (0.64) |
| v k ≥ 61 | 22.3/51.5 (0.43) | 12.5/29.1 (0.43) | 13.8/32.3 (0.43) | 9.4/20.9 (0.45) | 16.9/32.3 (0.52) | 11.7/20.9 (0.56) |
| u | 6.1/8.0 (0.76) | 4.4/7.2 (0.61) | 11.7/3.0 (3.9) | 10.3/2.3 (4.4) | 3.1/3.0 (1.04) | 2.6/2.3 (1.12) |
| u_j | 15.9/2.3 (6.9) | 14.6/1.6 (9.1) | 13.2/2.7 (4.9) | 11.3/1.8 (6.4) | 4.0/2.7 (1.48) | 3.1/1.8 (1.76) |

The posterior variance uses the measured median block noise of a line
(1.30 / 1.29 dB²) or of the floor (0.12 dB²); for `d`, which every line of a
rotor measures, that overstates its block noise, and for lines under the
floor it understates it. The v k 1–2 rows (σ 0 on DREGON, 0.32 dB on
Michael's) are in `findings.md`.

### Figures

- `results/noise_v3/checks/heldout/prominence_hist.png` (block prominence, k 8–24, real / v3 / v2 per rig)
- `results/noise_v3/checks/tonality/fits_ladder.png` (expectation prominence ladder, v3 against the v2 anchors)
- `results/noise_v3/checks/param_view/{dregon,michaels_cruise,michaels_standby}_{params,comb}.png` (v3 beside the v2 fit it replaces)

## Conclusion

| check | DREGON | Michael's cruise | Michael's standby |
|---|---|---|---|
| (a) prior predictive: widths | PASS (max 11.8 γ0 k, 9.3 % of drawn lines > 5 γ0 k; v2 fits reached 10⁴) | PASS (11.3, 9.1 %) | PASS (11.7, 9.4 %) |
| (a) prior predictive: floor | at the σ_B scale, not within it (draw shape sd ≤ 1.48 σ_B, extreme control point 3.0 σ_B); PASS against the v2 pathology, FAIL on a literal "no swing beyond σ_B" | same (1.63, 3.2) | same (1.53, 3.1); renders' median floor shape sd 9.3 dB against 3.6 dB real |
| (b) line-power spread | not testable: no resolvable line on the held-out windows | FAIL: v3 3.16 dB against real 1.46 on the dominant tracks (v2 0.38 fails the other way); PASS on the block sd of present lines (2.62 against 2.76; v2 1.18) | pooled with cruise (the rig's held-out set) |
| (b) disappearance | PASS at k 9–24 (66.4 % against 68.8 %; v2 shows no visible line), FAIL at k 25–60 (44 % against 73 %, 2.4 × as many visible lines per window) and at k 1–2 (a+d 1.2 % against 50 %: σ_v is 0 there) | PASS at k 3–24 (intervals overlap), FAIL at k ≥ 61 (48.6 % against 25.8 %) | pooled with cruise |
| (c) tonality | FAIL at 3 dB (v3 clips 3.2–3.8 orders per rotor against real 5.5–7.5), 1 order over at 6 dB; closer than v2 on the expectation (v2 anchor 17–23 at 3 dB) | PASS (clip counts within 1 order at every bar, closer than v2) | FAIL (clips 8.0–18.6 at 3 dB against 4.5–7.0; the expectation's k = 1 line is 14–22 dB) |
| (d) parity | PASS: 1.716079, upper 2.062063 ≤ 2.187786 (margin +0.472); stretch 1.897063 passed on the mean, not on the upper bound | PASS: ratio 0.557 ≤ 1.05 (margin +1.492 rev/s), equal-regime with standby | (inside Michael's) |
| (e) latents | FAIL: `d` 4.2 × and `u_j` 6.9 × σ²; v 2.0 / 1.6 / 0.43 × by group | FAIL: `d` 4.5 ×, `u` 3.9 ×, `u_j` 4.9 ×, v 2.0 / 1.9 / 1.3 / 0.43 × | mixed: `u` 1.04, v k 9–24 1.05, v k 3–8 1.23; `d` 3.7 ×, v k ≥ 25 0.52–0.69 × |

**The headline question, (b).** The Gaussian OU with σ_v(k) does reproduce
DREGON's intermittent mid-order lines: at k 9–24 the renders show about as
many visible lines per window (1.85 against 2.0) and they drop under 6 dB in
66 % of their blocks against 69 % in the real audio, where v2 renders no
visible line at all. It overshoots at k 25–60, where σ_v = 3.9 dB lifts
2.4 × as many lines over 6 dB as the real audio shows and they stay up
(44 % against 73 %), and it cannot produce the shaft-order dropouts: at
k 1–2 half the real lines appear and disappear within a window, the renders
1 %, because the measured σ_v there is 0 and σ_d is 0.94 dB. On Michael's
the disappearance rate matches at k 3–24 and is too high at k ≥ 61, and the
line-power spread of the dominant tracks is 2.2 × too large. The parity
gate still passes on both rigs, with DREGON better than v2 (1.716 against
1.820) and Michael's slightly worse (1.686 against 1.622); the spectral
proxy is worse than v2 on both (2.43 against 2.03 dB, 2.37 against 1.28).
Parity measures familiarity, not transfer: the transfer test is a retrained
arm.

**The DREGON width verdict.** The DREGON widths are physical: no fitted line
is wider than 50 Hz (widest 44.7 Hz at rotor 2, k = 26), against the R5 v2
fit's 16 kHz lines. On rendered audio the R4 −3 dB widths at k = 1, 2, 7, 8
are 9.5, 17.6, 17.6, 17.6 Hz against 10.4, 23.4, 30.7, 30.2 Hz real
(0.91 / 0.75 / 0.57 / 0.58 ×) and 10.3, 10.2, 6.8, 10.3 Hz for v2: v3 closes
most of the gap at k = 2, 7, 8 and stays narrower than the real lines.

**Open question for the next round: the prior scales.** The priors stayed at
the committed defaults (`gamma_c` 3, `sigma_nu_scale` 0.6), and the fits sit
far outside them. The timing fits already had σ_ν 5.4 (DREGON) and 4.3 rad/s
(cruise), 9.1 × and 7.1 × the 0.6 rad/s HalfNormal scale, and max
γ/(0.01k) of 157 / 72 / 75 (DREGON / cruise / standby) against a prior
scale of 3 (52 / 24 / 25 ×). The campaign fits keep that: σ_ν 4.57 / 3.84 /
0.35 rad/s (7.6 / 6.4 / 0.58 ×) and max γ/(0.01k) 172 / 76 / 68, with 31–42 %
of the lines over 5 γ0 k, while the prior draws of check (a) never pass
12 γ0 k. Either the bench-law scales are wrong for flight (label scatter,
real flight decoherence), or the likelihood buys width that the wander
should carry. Which one is not decided here. The alternation also never
converged in 5 rounds (DREGON still moves 10 × the tolerance), and every
latent family but a few fails the (e) sums, `d` by 3.7–4.5 × in every pool,
which by §3.3a sends the wander measurement (block length, line set) back
for revision, not the fit.
