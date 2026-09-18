# R3 DREGON bench refit (21 supports x 4 starts) — submitted, awaiting harvest

Written by `R3Bench`. The fit is RUNNING on the cluster; nothing else at the
top level of this directory is a result yet (`smoke/` is `R3Model`'s harvested
one-support smoke). This note is the handover: job id, the exact command, and
what the harvest step must do.

## The job

| field | value |
| --- | --- |
| job | `nv2-r3-bench-9ebc3b` |
| backend | `uni-cpu`, `--gpus 0 --cpus 16 --mem 64 --time 3h` |
| code SHA | `01e1464bc0c5b424c3d65abca4e690f6ae6acc82` (`01e1464b`, pushed on `main`) |
| submitted | 2026-09-18, via `omnirun --daemon localhost:18787` (wg route down, ssh tunnel) |
| submitted from | detached worktree `.worktrees/submit-R3Bench` at `01e1464b` |
| supports | the 21 `dregon-bench` supports, R2 windows, rebuilt in-job |
| starts | 4 per support (`--seeds 4`, `--init-jitter` default 0.8 on starts 1-3 — R2's schedule) |
| parallelism | `--jobs 16 --threads 1` (84 units) |
| output | `results/noise_v2/rounds/round3/fits/*.json`, `restarts/*.json`, `findings.md` |

### Why the windows are R2's

The set builder is unchanged since R2, so `--set dregon-bench` rebuilds the R2
rev-2 windows. Checked before submit by diffing
`round2/supports/index.json` against `round3/supports/index.json` (the smoke
job's rebuild) over all 21 supports: same names, same specs, same window rule;
the only differing fields are float ULP wobble in `level_db`,
`level_deficit_db`, `residual_max_abs_hz` and a few `npz_bytes` compression
counts. Same data, so R3 and R2 objectives per cell ARE comparable support by
support.

## The recipe (in-job body, `/tmp/r3bench_submit.sh` at submit time)

```
set -a; [ -f ./.env ] && . ./.env
JOBENV="${OMNIRUN_OUTPUT%/outputs}/.env"; [ -f "$JOBENV" ] && . "$JOBENV"; set +a
export PYTHONPATH=src
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
F=results/noise_v2/rounds/round3/fits
S=results/noise_v2/rounds/round3/supports
C=results/noise_v2/rounds/round1/supports          # SU.CACHE_DIR
mkdir -p "$F" "$S" "$C"
$PY scripts/noise_v2_supports.py build --set dregon-bench --out "$S"
cp -f "$S"/bench_dregon_*.npz "$C"/                # REQUIRED: load_support reads CACHE_DIR
$PY scripts/noise_v2_fit.py bench --set dregon-bench \
  --out "$F" --jobs 16 --threads 1 --seeds 4 --progress 500
# then a per-fit read-back print and `findings --out $F`
```

Submitted as:

```
cd .worktrees/submit-R3Bench && COLUMNS=200 omnirun --daemon localhost:18787 submit \
  --backend uni-cpu --gpus 0 --cpus 16 --mem 64 --time 3h \
  --name nv2-r3-bench \
  --outputs 'results/noise_v2/rounds/round3/fits/**' \
  --outputs 'results/noise_v2/rounds/round3/supports/**' \
  -- bash -lc "$(cat /tmp/r3bench_submit.sh)"
```

The job prints one `R3BENCH <support> sigma_nu/lam/check/obj/conv/bw/wall` line
per fit and ends with `R3BENCH_DONE exit_build=.. exit_fit=..`.

Wall estimate: the smoke (`nv2-r3-smoke-bench-12121c`, `Motor2_60`, 4 starts on
8 cpus) took 209 s per start; 84 units over 16 workers is ~5.3 waves, and the
widest supports (1.8M cells against `Motor2_60`'s 688k) run longer, so ~1.3-2 h
inside the 3 h wall.

## Harvest

```
omnirun --daemon localhost:18787 logs nv2-r3-bench-9ebc3b | tail -60
set -a; . ./.env; set +a
aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  s3://omnirun-artifacts/nv2-r3-bench-9ebc3b/outputs/results/ results/
# re-run reduce_restarts in the main checkout and verify byte-identity with the
# job's own reduction (R2Harvest's check), then
git add -f results/noise_v2/rounds/round3/fits/*.json \
  results/noise_v2/rounds/round3/fits/restarts/*.json
```

Read out of each committed JSON (schema `noise-v2-fit/2`): `params.sigma_nu`,
`params.lam`, `params.gamma_hz.value` at k = 1, 2, 4, 8, 16, 32, 64,
`diagnostics.gamma_low_order_check` (verdict, `max_over_resolution`,
`k4_over_k1`), `objective.whittle_nats / objective.n_cells`,
`optimiser.converged` / `which_converged` / `wall_s`, and the `restarts` block
(`best_minus_worst_per_cell`, per-start `sigma_nu`/`lam`/`gamma_hz.log_mean`).

## Next after the harvest

The DREGON floor-only frozen-comb refit, comb + `gamma` frozen from the four R3
`bench_dregon_Motor*_70__bench.json` by the driver's log-mean rule (`lam` comes
with them: in flight the rate is a constant taken from the frozen mapping), on
`--set dregon-floor`, whose carrier span pins `b`, `s` at their prior medians.
Recipe: `round2/fits/floor_v2_submit_note.md`, with round3 paths.
