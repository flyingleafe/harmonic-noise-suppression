# R3 DREGON floor refit, mode `flight_floor_lowk` — submitted, awaiting harvest

Written by `R3Bench`. The refit is RUNNING on the cluster; nothing in this note
is a result. Job id, the exact command, and what the harvest step must do.

## Why this mode and not `--floor-only`

Main's instruction: do NOT use plain `--floor-only`. `R3Model` added mode
`flight_floor_lowk` (`5c5b69e1`, `noise-v2 R3 model: flight_floor_lowk, a
per-order gain below k=K_low`): `--floor-only`'s freezing PLUS one gain per
ORDER for `k <= --low-orders` (8 here), shared across rotors, prior
`N(0, 20^2)` dB, on top of the single `comb_gain_db`. Both gains are folded
into `profile_db` inside `sample_params`, so the renderer and `params_to_dict`
are unchanged, and the fitted per-order dB is recorded at
`params.profile.low_order_gain_db` (already applied — never re-apply it).

## The job

| field | value |
| --- | --- |
| job | `nv2-r3-dregon-floor-lowk-c3b27b` |
| backend | `uni-cpu`, `--gpus 0 --cpus 16 --mem 64 --time 3h` |
| code SHA | `67318397087d90fa6c2949d0b3dc04607a4fcf83` (`67318397`, pushed on `main`; contains `5c5b69e1` and the harvested R3 bench fits) |
| submitted | 2026-09-19, via `omnirun --daemon localhost:18787` (ssh tunnel) |
| submitted from | detached worktree `.worktrees/submit-R3Bench` at `67318397` |
| pool | `--set dregon-floor` (the five DISJOINT 8 s room-2 fit segments, none of them a scored window) |
| frozen comb | the four R3 `bench_dregon_Motor{1,2,3,4}_70__bench.json`, driver log-mean rule |
| output | `results/noise_v2/rounds/round3/fits/dregon_room2_floor__flight_floor_lowk.json` |

Pins, all automatic and recorded in the payload, none passed on the CLI:

- `lam` is a CONSTANT in flight, taken from the frozen mapping's bench value —
  the log-mean of the four R3 `Motor*_70` `lam` — so no `--pin lam=` is needed
  (`R3Model`, confirming the R3 spec). `sigma_nu` and the per-line `gamma_hz`
  come frozen from the same mapping (`gamma_hz` by log-mean per order over the
  fits that reach each order).
- `b` and `s` (the speed laws) are pinned at their prior medians by the span
  rule: this pool is cruise-only, so its max/min carrier span is under
  `priors.speed_span_pin`. Verify in `diagnostics.span_pins` after the harvest.

## The command

```
cd .worktrees/submit-R3Bench && COLUMNS=200 omnirun --daemon localhost:18787 submit \
  --backend uni-cpu --gpus 0 --cpus 16 --mem 64 --time 3h \
  --name nv2-r3-dregon-floor-lowk \
  --outputs 'results/noise_v2/rounds/round3/fits/dregon_room2_floor__*.json' \
  --outputs 'results/noise_v2/rounds/round3/supports/*' \
  -- bash -lc "$(cat /tmp/r3floor_submit.sh)"
```

In-job body (`/tmp/r3floor_submit.sh`), the R2 `floor3` recipe with the round-3
paths and the new mode:

```
set -a; [ -f ./.env ] && . ./.env
JOBENV="${OMNIRUN_OUTPUT%/outputs}/.env"; [ -f "$JOBENV" ] && . "$JOBENV"; set +a
export PYTHONPATH=src
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
F=results/noise_v2/rounds/round3/fits
S=results/noise_v2/rounds/round3/supports
C=results/noise_v2/rounds/round1/supports          # SU.CACHE_DIR
$PY scripts/noise_v2_supports.py build --set dregon-floor --out "$S"
cp -f "$S"/*.npz "$C"/                             # REQUIRED: load_support reads CACHE_DIR
FM=<the four $F/bench_dregon_Motor{1..4}_70__bench.json>
$PY scripts/noise_v2_fit.py flight --set dregon-floor --name dregon_room2_floor \
  --floor-low-k --low-orders 8 --frozen-mean $FM \
  --jobs 1 --threads 16 --progress 100 --out "$F"
# then a read-back print of comb_gain_db, low_order_gain_db, the floor, the
# span pins, the objective and convergence
```

The bench fit JSONs are IN the job checkout: `results/` is gitignored but those
21 files are tracked (`git add -f`, commit `1f404c63`), so `$FM` resolves.

The job prints `R3FLOOR ...` then `R3FLOOR_DONE exit_build=.. exit_floor=..`.

## Harvest

```
omnirun --daemon localhost:18787 logs nv2-r3-dregon-floor-lowk-c3b27b | tail -30
set -a; . ./.env; set +a
aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  s3://omnirun-artifacts/nv2-r3-dregon-floor-lowk-c3b27b/outputs/results/ results/
git add -f results/noise_v2/rounds/round3/fits/dregon_room2_floor__flight_floor_lowk.json
```

Quote next to the convergence label (`optimiser.converged`, `which_converged`,
`lbfgs_restart_gain_per_cell` against the 1e-4 tolerance):
`params.profile.comb_gain_db`, `params.profile.low_order_gain_db` (the eight
per-order dB), `diagnostics.floor_level_db`, `params.floor.floor_mean_db`
against `diagnostics.init_floor_mean_db`, `diagnostics.span_pins`, and
`objective.whittle_nats / objective.n_cells`. The R2 comparanda on the same
pool, both NOT converged: `floor2`
(`dregon_room2_floor__flight_floor_only.json`) -17697677.791956328 / 2056320 =
-8.6065 nats/cell, and the R2 `comb_gain_db` refit
(`dregon_room2_floor__flight_floor_only_v2.json`). They are the `/1` law, so
the objective difference is a model difference, not only a mode difference.
