# R3 PER-REGIME Michael's fits — submitted, HARVESTED

Written by `R3Standby`. Both fits are committed and both are **NOT
converged**; the numbers live in `round3/fits/findings_regimes.md`. Job
summary: standby `nv2-r3-michaels-standby--6c860e` succeeded 05:14Z, exit 0,
wall 11 863.2 s (3.30 h), harvested and committed; cruise
`nv2-r3-michaels-cruise-03bf4e` succeeded 02:20Z, exit 0, wall 2 917.9 s (48.6
min), harvested and committed. The score arm that renders both through
`render.render_noise_regimes` is `nv2-r3-score-michaels-re-ac8bc5`
(`round3/render/submit_note_michaels_regimes.md`). Everything below is the
submission handover as written before the results landed.

## Why two fits instead of one pool

R3's pooled fit (`michaels_fly125_all__flight.json`, NOT converged) scores
HPPNet equal-regime **2.334258** rev/s (bar 3.177994, within), with cruise
0.693809 (legacy 0.755783) and ramp 3.145259 (legacy 8.007098) BETTER than the
legacy arm but standby **3.163707** against the legacy **0.317103** — a factor
of ten worse (`round3/render/submit_note_michaels.md`). One speed law over a
4.7x carrier span cannot serve both ends. The instruction: stop fitting cruise
and standby jointly — fit cruise alone, standby alone, and let the RENDERER
choose/interpolate by carrier, the way the previous-generation model's per-
regime exports were selected.

## The jobs

| field | standby | cruise |
| --- | --- | --- |
| job | `nv2-r3-michaels-standby--6c860e` | `nv2-r3-michaels-cruise-03bf4e` |
| backend | `uni-cpu`, `--gpus 0 --cpus 8 --mem 48 --time 5h` | `uni-cpu`, `--gpus 0 --cpus 8 --mem 48 --time 3h` |
| submitted | 2026-09-19T01:54:39Z, `running` 01:54:51Z | 2026-09-19T01:30:30Z, `running` 01:30:44Z |
| code SHA | `58e895fb` | `a996e5e5` (identical fit path; the delta is the renderer/scorer composition) |
| pool | `michaels-standby`: FLY125 @2.0+4, @6.0+4, @10.0+4 (3 windows, 12.0 s) | `michaels-cruise` filtered to FLY125 by the driver: 8 windows @16/32/48/64/96/112/128/144 + 8 s (64.0 s) |
| output | `round3/fits/michaels_fly125_standby__flight.json` | `round3/fits/michaels_fly125_cruise__flight.json` |

The standby fit was FIRST submitted as `nv2-r3-michaels-standby-28a220` with
`--time 2h` (01:30:12Z) and **cancelled** at 01:57Z, before it could be killed
by its own wall: a standby-only pool renders 130 orders (`k_max` =
`min(K_CAP, floor(8000 / 40.1)) = 130`) against the pooled fit's 81, so its
Adam phase ran at ~16 steps/min (750 steps in 48 min, i.e. ~96 min for 1500)
and the pooled R3 job's own split (Adam 2069 s of a 5621 s wall) puts L-BFGS
at ~1.5 h on top. 2 h could not hold both phases; nothing was lost by
cancelling (the driver checkpoints nothing mid-run) and the recipe is
UNCHANGED in the resubmission — only `--time`.

Both from the detached worktree `.worktrees/submit-R3Standby` via
`omnirun --daemon localhost:18787`; expected wall 3.0–3.5 h (standby) and
1.0–1.5 h (cruise).

## The standby pool, and why its windows are 4 s

`supports.set_michaels_standby` (new at `a996e5e5`) takes every DISJOINT
window the standby band admits at the longest campaign length that admits at
least three. FLY125 holds ONE standby run, **1.775–15.270 s** (13.50 s), read
off `rps_refined` on the 200 Hz telemetry grid `clips.windows` uses, so:

* 8 s windows: the anchored grid admits **0**, and at a 1 s stride the six
  candidates (2.0 … 7.0) all overlap — at most **1** disjoint window. Fewer
  than three.
* 4 s windows: the 1 s-stride candidates are 2.0 … 11.0; packed earliest-first
  without overlap that is **3** windows — 2.0, 6.0, 10.0 — i.e. 12.0 s of
  standby material against the pooled fit's single 8 s window.

All three carry every rotor inside `gates.REGIME_BANDS["standby"]` = 20–45
rev/s for the whole span (asserted on the real label by
`tests/experiments/test_noise_model_supports.py::
TestSpecsAndCache::test_the_standby_pool_is_disjoint_and_inside_the_standby_band`),
and all three sit inside spans already declared as Michael's TRAINING material
(`noise_v2_likelihood_window.TRAINING_SUPPORTS`: FLY125 @2+8 and @10+8 cover
2.0–18.0 s), so no held-out measurement moves and no new training span is
declared. FLY124 carries no row in either pool: the `michaels-cruise` set
lists the five frozen FLY124 score windows so one cache index describes the
round, and `noise_v2_fit._specs` drops them (`scripts/noise_v2_fit.py:787`) —
the frozen cohort is never fitted on.

## What the model pins by itself (no `--pin` on either command)

* **λ = 0.5.** In FLIGHT mode `lam` is a constant of the R3 model, not a site:
  `fit._seed_params` takes `MD.flight_lam(priors, fz)` = `Priors.flight_lam`
  = 0.5, recorded as `priors.flight_lam_pin`. `optimiser.pinned_dynamics` will
  read `null` while `params.lam` reads 0.5 — the harvest verifies both.
* **Speed exponents, standby fit: PINNED by the span rule.** The standby
  pool's carriers span ~1.1x, under `Priors.speed_span_pin` = 1.5, so
  `diagnostics.span_pins.pinned` must come back with `amp_exp`, `floor_exp`
  and `floor_static_rel` at their prior medians (no site, no prior term).
  That is the point: a single-regime fit must not extrapolate a speed law it
  cannot see, and the composition never asks it to (below 45 rev/s it is the
  only fit rendering).
* **Speed exponents, cruise fit: FREE.** The cruise pool spans 68.2–97.9 rev/s
  = 1.44x, under 1.5 — so `span_pins.pinned` is expected NON-empty here too,
  unlike R1's `/1` cruise fit at the same span, which had no span rule and
  took `amp_exp` 10.02 / `floor_exp` -10.19. The harvest reports whichever it
  is verbatim.

Optimiser on both: the driver defaults at this SHA, identical to the pooled R3
recipe — 1500 Adam steps at lr 0.02 on batches of 8 frames, then 200 + 100
L-BFGS iterations on 64 frames, seed 0, `--frame-stride 4 --max-frames 256`,
`--threads 8 --jobs 1`, one start.

## The commands

```
cd .worktrees/submit-R3Standby && COLUMNS=200 omnirun --daemon localhost:18787 submit \
  --backend uni-cpu --gpus 0 --cpus 8 --mem 48 --time 2h \
  --name nv2-r3-michaels-standby \
  --outputs 'results/noise_v2/rounds/round3/**' \
  -- bash -lc "$(cat /tmp/r3standby_submit.sh)"
```

and the same with `--time 3h --name nv2-r3-michaels-cruise` and
`/tmp/r3cruise_submit.sh`, which is the standby body with `michaels-standby`
-> `michaels-cruise` and `michaels_fly125_standby` ->
`michaels_fly125_cruise`. The standby body:

```
set -a
[ -f ./.env ] && . ./.env
JOBENV="${OMNIRUN_OUTPUT%/outputs}/.env"
[ -f "$JOBENV" ] && . "$JOBENV"
set +a
export PYTHONPATH=src
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
F=results/noise_v2/rounds/round3/fits
S=results/noise_v2/rounds/round3/supports_standby
C=results/noise_v2/rounds/round1/supports   # SU.CACHE_DIR
mkdir -p "$F" "$S" "$C"

$PY scripts/noise_v2_supports.py build --set michaels-standby --out "$S"
exit_build=$?
cp -f "$S"/flight_michaels_FLY12*.npz "$C"/ 2>/dev/null   # REQUIRED: load_support reads CACHE_DIR
ls -1 "$C" | wc -l

$PY scripts/noise_v2_fit.py flight --set michaels-standby --name michaels_fly125_standby \
  --out "$F" --jobs 1 --threads 8 --progress 25
exit_fit=$?

<python readback of the fit JSON>
echo "R3STANDBY_DONE exit_build=$exit_build exit_fit=$exit_fit"
```

The `.npz` caches are gitignored, so the supports are rebuilt in-job; the `cp`
into `results/noise_v2/rounds/round1/supports` is REQUIRED because
`supports.load_support` reads `SU.CACHE_DIR` and not `--out`. The two jobs
write their caches to DIFFERENT `--out` dirs (`supports_standby` /
`supports_cruise`) so the two concurrent builds cannot race on one index, and
neither touches `round3/supports/`, which is `R3Bench`'s.

## Harvest

```
omnirun --daemon localhost:18787 status nv2-r3-michaels-standby--6c860e
omnirun --daemon localhost:18787 logs  nv2-r3-michaels-standby--6c860e | tail -40  # R3STANDBY_DONE exit_fit=0
set -a; . ./.env; set +a
aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  s3://omnirun-artifacts/nv2-r3-michaels-standby--6c860e/outputs/results/ /tmp/r3standby_pull/
cp /tmp/r3standby_pull/noise_v2/rounds/round3/fits/michaels_fly125_standby__flight.json \
   results/noise_v2/rounds/round3/fits/
git add -f results/noise_v2/rounds/round3/fits/michaels_fly125_standby__flight.json
```

and the same for `nv2-r3-michaels-cruise-03bf4e` /
`michaels_fly125_cruise__flight.json`. Sync into a SCRATCH dir and copy only
the one fit JSON out: a job's `--outputs` tree carries the whole
`round3/**` of the node's checkout, and `R3Bench` owns
`round3/fits/findings.md`, `round3/fits/bench_*`, `round3/fits/restarts/`,
`round3/fits/dregon_room2_floor__flight_floor_lowk.json` and
`round3/supports/`.

Read out of each committed JSON (schema `noise-v2-fit/2`): `params.sigma_nu`,
`params.lam` (must be 0.5), `params.gamma_hz` at k = 1, 2, 4, 8,
`params.profile.amp_exp`, `params.floor.floor_exp`,
`params.floor.floor_static_rel`, `diagnostics.floor_level_db`,
`diagnostics.span_pins`, `objective.whittle_nats` / `objective.n_cells`,
`optimiser.converged` / `which_converged` / `grad_norm` /
`lbfgs_restart_gain_per_cell` (tolerance 1e-4 nats/cell) / `wall_s`.

## Then: the composition and the score arm

`render.render_noise_regimes` (this agent's next step) renders the two fits on
one carrier track and chooses between them per sample by the `rps_gating`
thresholds — standby fit below 45 rev/s on the slowest rotor, cruise fit at or
above 65, the two blended in between. The score arm is

```
scripts/noise_v2_round_score.py --round 3 --fits results/noise_v2/rounds/round3/fits \
  --rigs michaels \
  --fit michaels=results/noise_v2/rounds/round3/fits/michaels_fly125_cruise__flight.json \
  --fit michaels_standby=results/noise_v2/rounds/round3/fits/michaels_fly125_standby__flight.json \
  --candidate michaels_v2_r3_regimes \
  --arm-out results/noise_v2/rounds/round3/render/arm_michaels_v2_regimes.json
```

on `colab --gpus 1 --time 30m` with the three `--env` R2 variables, exactly as
`round3/render/submit_note_michaels.md` documents. Bars: Michael's PARITY
equal-regime <= 3.177994 rev/s (legacy per-regime 0.317103 standby / 8.007098
ramp / 0.755783 cruise), proxy `ltas_abs_db` <= 1.219668 dB; the pooled R3
candidate to beat is 2.334258 (0.693809 / 3.145259 / 3.163707).
