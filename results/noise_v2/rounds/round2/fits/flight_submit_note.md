# R2 pooled Michael's flight fit — submitted, awaiting harvest

Written by `R2FlightSubmit`. The fit is RUNNING on the cluster; nothing in this
directory is a result yet. This note is the handover: job id, the exact command,
and what the harvest step must do.

## The job

| field | value |
| --- | --- |
| job | `nv2-r2-michaels-flight-cb36cd` |
| backend | `uni-cpu`, `--gpus 0 --cpus 8 --mem 48 --time 3h` |
| code SHA | `c514bb507a4fdcd41a4bd658f4b66902848cc191` (`c514bb50`, pushed on `main`) |
| submitted | 2026-09-18T07:51:24Z (08:51 local) |
| status at submit | `starting` (verified with `omnirun status nv2-r2-michaels-flight-cb36cd`) |
| expected wall | 1.5–2 h (R1's 8-window cruise pool: 1.80 h first fit, 1.33 h pinned retry; this pool is 10 windows) |
| output the fit writes | `results/noise_v2/rounds/round2/fits/michaels_fly125_all__flight.json` |
| submitted from | detached worktree `.worktrees/submit-R2FlightSubmit` at `c514bb50` |

## The recipe

Pool: `michaels-all` (`supports.set_michaels_all`), 10 FLY125 windows of 8 s —
8 cruise + 1 standby (`flight_michaels:FLY125:2.0:8.0:rps_refined`) + 1 ramp
context (`flight_michaels:FLY125:11.5875:8.0:rps_refined`), so the pool spans
~36–98 rev/s instead of cruise's 68.2–97.9 and both speed exponents are FITTED
rather than extrapolated. `floor_exp` is sampled in log space at this SHA, so it
cannot go negative.

R1's optimiser settings are unchanged (driver defaults at this SHA): 1500 Adam
steps at lr 0.02 on batches of 8 frames, then 200 + 100 L-BFGS iterations on 64
frames, seed 0, `--frame-stride 4 --max-frames 256`. The three rate pins are
R1Basin's ridge coordinates approved by Main; both exponents stay FREE.

Submit command actually used (job script `/tmp/r2flight_submit.sh`, reproduced
below it):

```
cd .worktrees/submit-R2FlightSubmit && COLUMNS=200 omnirun submit \
  --backend uni-cpu --gpus 0 --cpus 8 --mem 48 --time 3h \
  --name nv2-r2-michaels-flight \
  --outputs 'results/noise_v2/rounds/round2/fits/*' \
  --outputs 'results/noise_v2/rounds/round2/fits/grid/**' \
  --outputs 'results/noise_v2/rounds/round2/supports/*' \
  -- bash -lc "$(cat /tmp/r2flight_submit.sh)"
```

The in-job body (the mandated env wrapper, then build and fit):

```
set -a
[ -f ./.env ] && . ./.env
JOBENV="${OMNIRUN_OUTPUT%/outputs}/.env"
[ -f "$JOBENV" ] && . "$JOBENV"
set +a
export PYTHONPATH=src
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
F=results/noise_v2/rounds/round2/fits
S=results/noise_v2/rounds/round2/supports
C=results/noise_v2/rounds/round1/supports

$PY scripts/noise_v2_supports.py build --set michaels-all --out "$S"
cp -f "$S"/flight_michaels_FLY125@*.npz "$C"/      # SU.CACHE_DIR is round1/supports

$PY scripts/noise_v2_fit.py flight --set michaels-all --name michaels_fly125_all \
  --out "$F" --jobs 1 --threads 8 --progress 25 \
  --pin lam=0.5 lam_eps_even=2.0 lam_eps_odd=75.63307088769216
```

The npz caches are gitignored, so the supports are rebuilt in-job (~7 min for
13 supports in R1). The `cp` is REQUIRED: `scripts/noise_v2_supports.py build
--out` writes where you ask, but `supports.load_support` reads `SU.CACHE_DIR`
= `results/noise_v2/rounds/round1/supports`; without the copy the fit would
rebuild every window itself.

## What the harvest step must do

1. **Fetch.** `omnirun pull` times out; use R2 directly. Endpoint is derived
   from `R2_ACCOUNT_ID` in `./.env` (there is no `R2_ENDPOINT` variable — this
   exact form was verified against R1's job):

   ```
   set -a; . ./.env; set +a
   aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
     s3://omnirun-artifacts/nv2-r2-michaels-flight-cb36cd/outputs/results/ results/
   ```

   Check `omnirun logs nv2-r2-michaels-flight-cb36cd | tail` for `exit_fit=0`
   and the `R2FLIGHT_DONE` marker first; the job script prints the converged
   flag, the six dynamics, both exponents and whittle/cell on the way out.

2. **Commit** `results/noise_v2/rounds/round2/fits/michaels_fly125_all__flight.json`
   with `git add -f` (results/ is gitignored), plus the updated
   `round2/supports/index.json` if the in-job rebuild changed it.

3. **Write `results/noise_v2/rounds/round2/fits/findings_flight.md`** in the R1
   format (`round1/fits/findings.md` is the template), every number quoted from
   the committed JSON:
   - pool composition: regime × window count × carrier range (from
     `round2/supports/index.json`, set `michaels-all`, and the fit JSON's
     `supports` list) — 8 cruise + 1 standby + 1 ramp, and the pooled span;
   - objective per regime as well as overall (`objective.whittle_nats /
     objective.n_cells`, and the per-support breakdown if the JSON carries one);
   - the six dynamics (`sigma_nu`, `lam`, `sigma_eps_even/odd`,
     `lam_eps_even/odd`) with the three PINNED values marked as pinned, the two
     exponents (`amp_exp`, `floor_exp`) and the floor level;
   - convergence: `optimiser.converged`, `which_converged`,
     `lbfgs_restart_gain_per_cell` vs the 1e-4 tolerance, `grad_norm`; the
     convergence status MUST sit next to every fit number quoted anywhere;
   - wall (`optimiser.wall_s`, and the Adam/L-BFGS split the JSON records).

4. **Render sanity BEFORE scoring** — the k = 2 comb-to-floor of the new fit at
   the standby, ramp and cruise carriers:

   ```
   PYTHONPATH=src python scripts/noise_v2_render_regime.py \
     --fit results/noise_v2/rounds/round2/fits/michaels_fly125_all__flight.json \
     --regimes cruise,standby,ramp --figures \
     --out results/noise_v2/rounds/round2/render_regime_flight
   ```

   It reports `comb/floor k=2 dB` per arm (also k = 4, 8). Compare the v2 row
   against the `real` row of the same table: standby 15.1 dB, ramp 6.9 dB
   (round-1 v2, fitted on cruise alone, gave 4.4 dB at standby and −0.7 dB at
   ramp — see `round2/render_regime/findings.md`). Runs on CPU in a few minutes;
   `--probe` (frozen HPPNet PIT) needs `uni-gpushort`, `--gpus 1`, ≤ 1 h.

5. **Then** hand the fit path + SHA to whoever owns the round-2 score:
   `scripts/noise_v2_round_score.py --fit michaels=<path> ...` into
   `results/noise_v2/rounds/round2.json` + `round2/score/findings.md`. Bars:
   PARITY Michael's equal-regime ≤ 3.177994, DREGON ≤ 2.187786; STRETCH DREGON
   ≤ 1.897063.

If the fit fails the convergence test, record it as-is; ONE retry is allowed
only with a named specific cause (e.g. Adam batch 8 vs a 10-window pool).
