**Status:** in progress — 2026-09-24 → (fits running on `uni-cpu`)

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
| `nv3-fit-dregon-s0-28cc14` | DREGON | 0 | `e1d0e7f0` | 64 GB | running |
| `nv3-fit-dregon-s1-0d3805` | DREGON | 1 | `e1d0e7f0` | 64 GB | running |
| `nv3-fit-dregon-s2-0c2b38` | DREGON | 2 | `e1d0e7f0` | 64 GB | running |
| `nv3-fit-dregon-s3-64f951` | DREGON | 3 | `e1d0e7f0` | 64 GB | running |
| `nv3-fit-mstandby-s0-3fcf29` | Michael's standby | 0 | `e1d0e7f0` | 64 GB | running |
| `nv3-fit-mstandby-s1-6c5be5` | Michael's standby | 1 | `e1d0e7f0` | 64 GB | running |
| `nv3-fit-mstandby-s2-98610f` | Michael's standby | 2 | `e1d0e7f0` | 64 GB | running |
| `nv3-fit-mstandby-s3-32f230` | Michael's standby | 3 | `e1d0e7f0` | 64 GB | running |
| `nv3-fit-mcruise-s{0..3}-{8cf99e,3a0859,7cba5a,f7ba2c}` | Michael's cruise | 0–3 | `e1d0e7f0` | 64 GB | **OOM-killed** 11 s into the fit (8 windows × 62 frames at full frames; DREGON's 5 × 62 fits in 64 GB) |
| `nv3-fit-mcruise-s0-1fb6f3` | Michael's cruise | 0 | `e1d0e7f0` | 160 GB | queued (daemon slot limit) |
| `nv3-fit-mcruise-s1-91bf7f` | Michael's cruise | 1 | `e1d0e7f0` | 160 GB | queued |
| `nv3-fit-mcruise-s2-de2c13` | Michael's cruise | 2 | `e1d0e7f0` | 160 GB | queued |
| `nv3-fit-mcruise-s3-2955b8` | Michael's cruise | 3 | `e1d0e7f0` | 160 GB | queued |

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

## Fits

_Filled in per (rig, pool) family as its four restarts finish._

## Results

_Fits running; results follow the harvest._

## Conclusion

_Pending._
