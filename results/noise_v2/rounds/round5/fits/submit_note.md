# R5 DREGON flight fit, mode `flight_profile` — submitted, awaiting harvest

Written by `R5Dregon`. The four restarts are RUNNING on the cluster; nothing in
this note is a result. Job ids, the exact commands, and what the harvest does.

## Why this mode

R4 (`round4/legacy_truth/findings.md`) showed that v2 in mode `flight` — free
per-order profile AND free dynamics — reproduces the LEGACY render well enough
for HPPNet (1.746 rev/s against the 2.188 bar), while the bench-transplant mode
`flight_floor_lowk` reaches only 5.560. But in DREGON FREE FLIGHT neither the
raw nor the refined rotor-speed labels follow the harmonics' trajectories
closely, and a free dynamics block absorbs that label error as shaft wander and
line width: the R4 free fit landed at a 13 Hz half width at k=1, which is a
property of the label, not of the rotor.

`flight_profile` (`src/experiments/noise_model/model.py:310`, commit
`1c6ac337`) is therefore the MIRROR of `flight_floor_lowk`: free blocks
`("profile", "floor", "mic")`, dynamics FROZEN from the bench. The transplant
is the existing one (`scripts/noise_v2_fit.py:mean_comb`, now with
`per_rotor_gamma`): `sigma_nu` and `lam` at the log-means over the four
`bench_dregon_Motor{1..4}_70` R3 fits (0.742318 / 0.060800 — bit-equal to the
values the R3 `flight_floor_lowk` fit carried), and the per-line `gamma_hz`
ladder ROTOR-MATCHED, rotor `r` taking `Motor{r+1}`'s own widths (the rule the
four-motor validation transplants widths by), falling back to the pooled
log-mean above that fit's own order cap (the fits are 116-118 orders wide, the
pool's cap is 88, so the fallback never fires here). `profile_db`'s prior is
the mode's own `N(measured, 10)` / `N(mu_F - 15, 8)`, as in `flight`.

## The jobs

| field | value |
| --- | --- |
| jobs, 4 restarts | `nv2-r5-flightprofile-s0-2e53c0`, `nv2-r5-flightprofile-s1-6a537a`, `nv2-r5-flightprofile-s2-6dacb1`, `nv2-r5-flightprofile-s3-59fbd5` |
| backend | `uni-cpu`, `--gpus 0 --cpus 16 --mem 64 --time 2h`, `--jobs 1 --threads 16` |
| code SHA | `1c6ac337` (`noise-v2 R5: flight_profile mode`, pushed on `main`) |
| submitted | 2026-09-20, via `omnirun --daemon localhost:18787` |
| submitted from | detached worktree `.worktrees/submit-R5Dregon` at `1c6ac337` |
| pool | `--set dregon-floor`, the five disjoint 8 s room-2 fit segments — the SAME pool and windows as R3's `dregon_room2_floor__flight_floor_lowk` |
| seeds | 4, ONE CLUSTER JOB EACH (`--seed N --restart-tag sN`); R4's `--jobs 2` was OOM-killed, one pooled DREGON flight fit wants most of a 64 GB node |
| output | `results/noise_v2/rounds/round5/fits/restarts/dregon_room2_floor__flight_profile__s<N>.json`, reduced to `..fits/dregon_room2_floor__flight_profile.json` |

Pins, all automatic and recorded in the payload, none passed on the CLI: the
dynamics are frozen wholesale by the mode (no `sigma_nu`/`lam`/`gamma_hz` site
at all — `tests/experiments/test_noise_model_core.py::test_flight_profile_frees_the_comb_and_keeps_no_dynamics_site`
asserts it on the model trace and on `fit.initial_values`), and `amp_exp`,
`floor_exp`, `floor_static_rel` follow the span rule (this pool's carrier span
is 1.744, over the 1.5 threshold, so they are FREE here — verify in
`diagnostics.span_pins` after the harvest).

## The commands

```
cd .worktrees/submit-R5Dregon
for SEED in 0 1 2 3; do
  sed -e "s/SEED/$SEED/g" /tmp/r5fit_unit.sh > /tmp/r5fit_s$SEED.sh
  COLUMNS=200 omnirun --daemon localhost:18787 submit \
    --backend uni-cpu --gpus 0 --cpus 16 --mem 64 --time 2h \
    --name nv2-r5-flightprofile-s$SEED \
    --outputs 'results/noise_v2/rounds/round5/fits/**' \
    -- bash -lc "$(cat /tmp/r5fit_s$SEED.sh)"
done
```

In-job body (`/tmp/r5fit_unit.sh`, with `SEED` substituted):

```
set -a; [ -f ./.env ] && . ./.env
JOBENV="${OMNIRUN_OUTPUT%/outputs}/.env"; [ -f "$JOBENV" ] && . "$JOBENV"; set +a
export PYTHONPATH=src
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
F=results/noise_v2/rounds/round5/fits; S=results/noise_v2/rounds/round5/supports
C=results/noise_v2/rounds/round1/supports          # SU.CACHE_DIR
R3=results/noise_v2/rounds/round3/fits
$PY scripts/noise_v2_supports.py build --set dregon-floor --out "$S"
cp -f "$S"/*.npz "$C"/                             # REQUIRED: load_support reads CACHE_DIR
FM=<the four $R3/bench_dregon_Motor{1..4}_70__bench.json>
$PY scripts/noise_v2_fit.py flight --set dregon-floor --name dregon_room2_floor \
  --frozen-dynamics per-rotor --frozen-mean $FM \
  --seed SEED --restart-tag sSEED \
  --jobs 1 --threads 16 --progress 300 --out "$F" --grid-dir "$F/grid_sSEED"
# then a read-back print of nats/cell, the convergence label, the frozen
# dynamics, amp_exp, the floor, the span pins and the low-order profile
```

Each job prints `R5FIT specs: ...` then `R5FIT_DONE exit_build=.. exit_fit=..`.

`--outputs` names the FITS only, not `round5/supports/**`: the five `.npz`
support caches are 297 MB (cf. `round3/supports`), they are rebuilt from the
raw DREGON dataset by the job itself, and the same caches already exist on the
laptop — syncing them back four times buys nothing.

## Harvest

```
for J in nv2-r5-flightprofile-s0-2e53c0 nv2-r5-flightprofile-s1-6a537a \
         nv2-r5-flightprofile-s2-6dacb1 nv2-r5-flightprofile-s3-59fbd5; do
  omnirun --daemon localhost:18787 logs "$J" | tail -25
  aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
    s3://omnirun-artifacts/"$J"/outputs/results/ results/
done
PYTHONPATH=src python scripts/noise_v2_fit.py reduce \
  --out results/noise_v2/rounds/round5/fits --mode flight_profile
git add -f results/noise_v2/rounds/round5/fits/*.json \
           results/noise_v2/rounds/round5/fits/restarts/*.json
```

Quote next to the convergence label (`optimiser.converged`,
`which_converged`, `lbfgs_restart_gain_per_cell` against the 1e-4 tolerance):
`objective.whittle_nats / objective.n_cells`, the restart spread, the fitted
`profile_db` at k <= 16 per rotor against the four bench profiles (delta dB),
`params.profile.amp_exp`, `diagnostics.floor_level_db`,
`params.floor.{floor_mean_db,floor_tilt_db_oct,floor_exp,floor_static_rel}`
and `diagnostics.span_pins`. The comparandum on the SAME pool is R3's
`flight_floor_lowk` at -17670197.599347502 / 2056320 = -8.59398 nats/cell
(NOT converged); it is the same model and the same cells, so the difference is
a mode difference alone.
