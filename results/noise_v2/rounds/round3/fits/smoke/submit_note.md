# R3 smoke bench fit — submitted, awaiting harvest

Written by `R3Model`. The fit is RUNNING on the cluster; nothing else in this
directory is a result yet. This note is the handover: job id, the exact
command, and what the harvest step must do.

## The job

| field | value |
| --- | --- |
| job | `nv2-r3-smoke-bench-12121c` |
| backend | `uni-cpu`, `--gpus 0 --cpus 8 --mem 32 --time 1h` |
| code SHA | `748e8081` (`noise-v2 R3 model: fit driver on schema /2`, pushed on `main`) |
| submitted | 2026-09-18 (local), via `omnirun --daemon localhost:18787` (wg route down, ssh tunnel) |
| submitted from | detached worktree `.worktrees/submit-R3Model` at `748e8081` |
| support | `bench_dregon_motor:Motor2:60` (R2 window, rebuilt in-job) |
| starts | 4 (`--seeds 4`, jitter 0.8 on starts 1-3), `--jobs 4 --threads 2` |
| output | `results/noise_v2/rounds/round3/fits/smoke/bench_dregon_Motor2_60__bench.json` (+ `restarts/`, `findings.md`) |

## The recipe (in-job body, `/tmp/r3smoke_submit.sh` at submit time)

```
set -a; [ -f ./.env ] && . ./.env
JOBENV="${OMNIRUN_OUTPUT%/outputs}/.env"; [ -f "$JOBENV" ] && . "$JOBENV"; set +a
export PYTHONPATH=src
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
F=results/noise_v2/rounds/round3/fits/smoke
S=results/noise_v2/rounds/round3/supports
C=results/noise_v2/rounds/round1/supports          # SU.CACHE_DIR
$PY scripts/noise_v2_supports.py build --set dregon-bench --out "$S"
cp -f "$S"/bench_dregon_*.npz "$C"/                # REQUIRED: load_support reads CACHE_DIR
$PY scripts/noise_v2_fit.py bench --support bench_dregon_motor:Motor2:60 \
  --out "$F" --jobs 4 --threads 2 --seeds 4 --progress 250
```

The job prints `R3SMOKE sigma_nu/gamma/check/obj/converged/restarts` lines and
ends with `R3SMOKE_DONE exit_build=.. exit_fit=..`.

## Harvest

```
omnirun --daemon localhost:18787 logs nv2-r3-smoke-bench-12121c | tail -40
set -a; . ./.env; set +a
aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  s3://omnirun-artifacts/nv2-r3-smoke-bench-12121c/outputs/results/ results/
git add -f results/noise_v2/rounds/round3/fits/smoke/*.json \
  results/noise_v2/rounds/round3/fits/smoke/restarts/*.json
```

What to read out of the committed JSON (schema `noise-v2-fit/2`):
`params.sigma_nu`, `params.lam`, `params.gamma_hz` at k = 1, 2, 4, 8, 16, 32,
`diagnostics.gamma_low_order_check` (verdict, `max_over_resolution`,
`k4_over_k1`), `objective.whittle_nats / objective.n_cells`,
`optimiser.converged` / `which_converged` / `wall_s`, and the `restarts` block's
`gamma_hz.log_mean` spread across the four starts.
