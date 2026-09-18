# R3 pooled Michael's flight fit — submitted, awaiting harvest

Written by `R3Michaels`. The fit is RUNNING on the cluster; nothing in this
file is a result. This note is the handover: job id, the exact command, and
what the harvest step must do — plus the TWO code gates that block the
downstream reductions on schema `noise-v2-fit/2` (reported to Main, no code
changed here).

## The job

| field | value |
| --- | --- |
| job | `nv2-r3-michaels-flight-8197c4` |
| backend | `uni-cpu`, `--gpus 0 --cpus 8 --mem 48 --time 3h` |
| code SHA | `01e1464bc0c5b424c3d65abca4e690f6ae6acc82` (`01e1464b`, pushed on `main`) |
| submitted | 2026-09-18T22:51:11Z, via `omnirun --daemon localhost:18787` (wg route down, ssh tunnel) |
| status at submit | `starting` (`omnirun status`, scheduler `running`, started 22:51:24Z) |
| submitted from | detached worktree `.worktrees/submit-R3Michaels` at `01e1464b` |
| pool | `michaels-all` (`supports.set_michaels_all`), 10 FLY125 fit windows of 8 s: 8 cruise + 1 standby (`flight_michaels:FLY125:2.0:8.0:rps_refined`) + 1 ramp (`…:11.5875:8.0:…`), rebuilt in-job |
| output | `results/noise_v2/rounds/round3/fits/michaels_fly125_all__flight.json` |
| expected wall | 1.3–2 h (R2's identical pool/optimiser: 1.33–1.80 h) |

## The R3 model at this SHA, and why the command carries no `--pin`

R2 pinned three rates (`lam=0.5 lam_eps_even=2.0 lam_eps_odd=75.633…`). R3
deleted the four `*_eps` scalars: the dynamics block is `sigma_nu` plus `lam`,
and the line shape is the per-line `gamma_hz` `(R, k_max)` block under the
LN(0.01·k Hz, 1.0) prior (`model.Priors.gamma_hz`). In FLIGHT mode `lam` is a
CONSTANT of the model, not a fitted site: `fit.seeds` allocates no `lam` init
when `flight` is true and `fit._seed_params` takes `MD.flight_lam(priors, fz)`
= `Priors.flight_lam` = **0.5** (`model.py:155`, recorded in the JSON's
`priors.flight_lam_pin`). So the driver pins λ = 0.5 by itself and an explicit
`--pin lam=0.5` would be redundant; `optimiser.pinned_dynamics` will therefore
be `null` while `params.lam` reads 0.5 — the harvest MUST verify both.

Both speed exponents stay FREE: the pool's carrier span is 4.7× (36–98 rev/s),
over `Priors.speed_span_pin` = 1.5, so `diagnostics.span_pins.pinned` must come
back EMPTY (a span under the threshold would pin `amp_exp`, `floor_exp`,
`floor_static_rel` at their prior medians).

Optimiser: the driver defaults at this SHA, identical to R2 — 1500 Adam steps
at lr 0.02 on batches of 8 frames, then 200 + 100 L-BFGS iterations on 64
frames, seed 0, `--frame-stride 4 --max-frames 256`, `--threads 8 --jobs 1`,
`init_jitter` 0 for a single start.

## The command

```
cd .worktrees/submit-R3Michaels && COLUMNS=200 omnirun --daemon localhost:18787 submit \
  --backend uni-cpu --gpus 0 --cpus 8 --mem 48 --time 3h \
  --name nv2-r3-michaels-flight \
  --outputs 'results/noise_v2/rounds/round3/**' \
  -- bash -lc "$(cat /tmp/r3flight_submit.sh)"
```

The in-job body (`/tmp/r3flight_submit.sh` at submit time):

```
set -a
[ -f ./.env ] && . ./.env
JOBENV="${OMNIRUN_OUTPUT%/outputs}/.env"
[ -f "$JOBENV" ] && . "$JOBENV"
set +a
export PYTHONPATH=src
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
F=results/noise_v2/rounds/round3/fits
S=results/noise_v2/rounds/round3/supports
C=results/noise_v2/rounds/round1/supports   # SU.CACHE_DIR
mkdir -p "$F" "$S" "$C"

$PY scripts/noise_v2_supports.py build --set michaels-all --out "$S"
exit_build=$?
cp -f "$S"/flight_michaels_FLY12*.npz "$C"/ 2>/dev/null   # REQUIRED: load_support reads CACHE_DIR
ls -1 "$C" | wc -l

$PY scripts/noise_v2_fit.py flight --set michaels-all --name michaels_fly125_all \
  --out "$F" --jobs 1 --threads 8 --progress 25
exit_fit=$?

<python readback of the fit JSON, then>
echo "R3FLIGHT_DONE exit_build=$exit_build exit_fit=$exit_fit"
```

The `.npz` caches are gitignored, so the supports are rebuilt in-job (~7 min
for the michaels set in R1/R2). The `cp` is REQUIRED for the same reason as in
R2 and in the R3 smoke note: `noise_v2_supports.py build --out` writes where
you ask, but `supports.load_support` reads `SU.CACHE_DIR` =
`results/noise_v2/rounds/round1/supports`.

KNOWN, harmless: the in-job readback block reads `params.gamma_hz` as if it
were the ladder dict of the driver's own findings reduction; in the JSON it is
the raw `(R, k_max)` list (`model.params_to_dict`), so that heredoc raises
after printing schema/`sigma_nu`/`lam`/`pinned`. The fit JSON, the exit codes
and `R3FLIGHT_DONE` are unaffected — every number in the findings comes from
the committed JSON, not from the log.

## Harvest

```
omnirun --daemon localhost:18787 status nv2-r3-michaels-flight-8197c4
omnirun --daemon localhost:18787 logs nv2-r3-michaels-flight-8197c4 | tail -40   # R3FLIGHT_DONE exit_fit=0
set -a; . ./.env; set +a
aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  s3://omnirun-artifacts/nv2-r3-michaels-flight-8197c4/outputs/results/ results/
git add -f results/noise_v2/rounds/round3/fits/michaels_fly125_all__flight.json
```

The sync also brings the job's `round3/supports/index.json` and
`round3/fits/findings.md`; `R3Bench` owns `round3/supports/` and the bench
findings, so check `git diff` and leave those to their owner.

Read out of the committed JSON (schema `noise-v2-fit/2`): `params.sigma_nu`,
`params.lam` (must be 0.5), `params.gamma_hz` at k = 1, 2, 4, 8, 16 (ladder
also in `diagnostics.gamma_hz_at`), `params.profile.amp_exp`,
`params.floor.floor_exp`, `params.floor.floor_static_rel`,
`diagnostics.floor_level_db`, `diagnostics.gamma_low_order_check`,
`diagnostics.span_pins`, `objective.whittle_nats / objective.n_cells`,
`optimiser.converged` / `which_converged` / `lbfgs_restart_gain_per_cell`
(tolerance 1e-4 nats/cell) / `grad_norm` / `wall_s` / `adam_wall_s`.

## The two code gates that block the downstream reductions on `/2`

Both are one-line schema literals; the rendering library itself is already
`/2`-ready (`render.READABLE_SCHEMAS = (FIT_SCHEMA, "noise-v2-fit/1")`, and
`model.gamma_from_params` maps a `/1` payload forward), and the `params`
layout the two scripts read (`params.profile.amp_exp`,
`params.floor.floor_exp`, `params.floor.floor_static_rel`) is unchanged
between `/1` and `/2`.

1. **Render-regime sanity — BLOCKED.** `scripts/noise_v2_render_regime.py:455`
   `if str(fit.get("schema")) != "noise-v2-fit/1": die(...)`. An R3 fit
   carries `noise-v2-fit/2`, so `--fit <r3 json>` dies with
   `not a noise-v2-fit/1 payload` before any rendering. Needs the same
   `READABLE_SCHEMAS`-style acceptance the library has.
2. **Round score — BLOCKED.** `scripts/noise_v2_round_score.py:83`
   `FIT_SCHEMA = "noise-v2-fit/1"`. `read_fits` SKIPS every `/2` file (so
   `--fits results/noise_v2/rounds/round3/fits` dies with
   `no file carries schema 'noise-v2-fit/1'`) and `v2_arm` dies with
   `not a noise-v2-fit/1 fit` on `--fit michaels=<r3 json>`.

`R3Michaels` changes no code (assignment constraint). Reported to Main; the
score arm is submitted only once a SHA that accepts `/2` is on `main`.

## Bars for the score arm

Michael's PARITY equal-regime ≤ 3.177994 rev/s (legacy per-regime 0.317
standby / 8.007 ramp / 0.756 cruise), proxy `ltas_abs_db` ≤ 1.219668 dB.
Render-regime real k = 2 comb/floor to compare against: standby 15.1 dB,
ramp 6.9 dB, cruise 25.3 dB.
