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

## Harvested (2026-09-18)

`R3SMOKE_DONE exit_build=0 exit_fit=0`, wall 208.6 s for the reported start
(4 starts in parallel, 8 cpus). Fetched from R2 and committed here:
`bench_dregon_Motor2_60__bench.json` + the four `restarts/*.json`.

| quantity | value |
| --- | --- |
| objective | −7 662 814.7 nats over 687 752 cells = **−11.1418 nats/cell** |
| convergence | `converged: true` (`which_converged: lbfgs`), restart gain 1.1e-4 nats/cell over the worst start |
| `sigma_nu` | 0.3970 rad/s (R2 on this support: 2.12, at the prior edge) |
| `lam` | 0.1703 /s (R2: 654.8) |
| `gamma_hz` k = 1, 2, 4, 8, 16, 32 | 0.00025, 0.0085, 0.0145, 0.0458, 1.193, 34.72 Hz |
| low-order check | **pass** — max γ(k ≤ 4) is 0.35 × the 0.0417 Hz resolution floor |
| speed-law pins | none (bench mode has no speed law) |
| restart agreement | γ log-mean over orders spans 1.07 × across the four starts |

The four starts agree on the widths to 7 % in log-mean, and the reported start
is the lowest objective of the four.

NOTE on the verdict: the job at `748e8081` recorded `fail_k2_ramp`, because the
verdict order read the γ₄/γ₁ ratio (58.5) before the floor test even though all
four low-order widths sit UNDER the resolution. Widths under a bin are one
statement and their ratio is noise, so the floor test is now decisive and the
ramp is only read above the floor (`fit.gamma_low_order_check`, with a
regression test). The committed JSONs carry the recomputed verdict and a
`recomputed` field naming the job's original one.
