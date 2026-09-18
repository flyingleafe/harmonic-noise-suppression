# R2 DREGON floor-only REFIT (`comb_gain_db` free) — submitted, awaiting harvest

Written by `R2FloorFix`. The refit is RUNNING on the cluster; nothing in this
note is a result. It is the handover: job id, the exact command, and what the
harvest step must do.

## Why a refit

`DregonRender`'s measurements (`round2/render_dregon/findings.md`) showed the R2
DREGON arm renders to HPPNet PIT MAE ~75 rev/s because the TRANSPLANTED bench
comb has no level of its own: `render_noise` mean-centres `mic_line_gain_db`
over mics per rotor and `mic_gains_db` over mics, so `profile_db` is the only
absolute comb scale a render has, and the `flight_floor_only` objective froze it
at the BENCH rig's level. On the DREGON cruise supports the frozen comb sits
-18.66 (hovering) / -18.12 (updown) dB relative to the fitted floor, against
+13.06 dB over it on the bench where it was identified — a 31.7 dB swing, and
the arm scored INDISTINGUISHABLY from one with no comb at all.

The patch (`fca85d9c`) gives `flight_floor_only` exactly one more free scalar,
`comb_gain_db ~ N(0, 20^2)` dB, folded into `profile_db` inside `sample_params`,
so `render_noise`, `params_to_dict` and `expected_periodogram` are unchanged and
the comb's SHAPE and dynamics stay frozen. `initial_values` seeds it by
measurement (a third probe pass renders the frozen comb; the seed is the median
per-line dB gap to the observed line excess). Regression test:
`tests/experiments/test_noise_model_core.py::test_floor_only_fit_recovers_the_level_of_a_transplanted_comb`
(comb planted 20 dB off, recovered to 19.71 dB, within the 1 dB bar).

The same commit fixes a recording bug found while testing it: `AutoDelta` adopts
the tensors `init_to_value` is handed and updates them IN PLACE, so
`diagnostics.init_floor_mean_db` was recording the FITTED floor. The R1/R2
floor fits' apparent "floor never left its initialisation (delta 0.00e+00 dB)"
is therefore an ARTEFACT of that aliasing, not evidence about the optimiser;
the guide is now initialised from clones and the diagnostic is a real snapshot.
(H3 itself does not rest on that line — it rests on the rendered comb-to-floor
ratios, which are measured from separated renders.)

## The job

| field | value |
| --- | --- |
| job | `nv2-r2-dregon-floor3-6ee9e2` |
| backend | `uni-cpu`, `--gpus 0 --cpus 8 --mem 48 --time 2h` |
| code SHA | `fca85d9ca298...` (`fca85d9c`, pushed on `main`) |
| submitted | 2026-09-18T09:15:39Z |
| started | 2026-09-18T09:15:55Z |
| status at submit | `starting` / scheduler `running` (`omnirun status nv2-r2-dregon-floor3-6ee9e2`) |
| submitted from | detached worktree `.worktrees/submit-R2FloorFix` at `fca85d9c` |
| expected wall | ~40–45 min (`nv2-r2-dregon-floor2-01c39a`, the same fit without the free scalar: 43 min 18 s, 01:49:20 → 02:32:38Z) |
| output the job writes | `results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only_v2.json` |

### Backup job (the node is ~2.8x slower than `floor2`'s)

`nv2-r2-dregon-floor3-6ee9e2` ran Adam 1500 in ~42 min against `floor2`'s 15
min (900 s), which puts its L-BFGS polish (`floor2`: 1579 s) at ~74 min and the
whole fit right at the `--time 2h` wall (11:15:55Z). A BYTE-IDENTICAL backup was
therefore submitted at 10:42Z with a 5 h wall:
`nv2-r2-dregon-floor3b-f97a58` (same code SHA `fca85d9c`, same worktree, same
in-job script, same `--outputs`). Harvest whichever finishes and CANCEL the
other (`omnirun cancel <job>`): the two are the same computation with the same
seed, so either payload is the refit.

## The command

```
cd .worktrees/submit-R2FloorFix && COLUMNS=200 omnirun submit \
  --backend uni-cpu --gpus 0 --cpus 8 --mem 48 --time 2h \
  --name nv2-r2-dregon-floor3 \
  --outputs 'results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only_v2.json' \
  --outputs 'results/noise_v2/rounds/round2/supports/*' \
  -- bash -lc "$(cat /tmp/r2floorfix_submit.sh)"
```

The in-job body (`/tmp/r2floorfix_submit.sh`) is `nv2-r2-dregon-floor2-01c39a`'s
verbatim, plus the rename and a read-back print:

```
set -a
[ -f ./.env ] && . ./.env
JOBENV="${OMNIRUN_OUTPUT%/outputs}/.env"
[ -f "$JOBENV" ] && . "$JOBENV"
set +a
export PYTHONPATH=src
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
S2=results/noise_v2/rounds/round2/supports
F2=results/noise_v2/rounds/round2/fits
CACHE=results/noise_v2/rounds/round1/supports

$PY scripts/noise_v2_supports.py build --set dregon-floor --out $S2; echo "exit_build=$?"
mkdir -p $CACHE
cp -f $S2/*.npz $CACHE/
FM=""
for m in 1 2 3 4; do
  [ -f "$F2/bench_dregon_Motor${m}_70__bench.json" ] && FM="$FM $F2/bench_dregon_Motor${m}_70__bench.json"
done
$PY scripts/noise_v2_fit.py flight --set dregon-floor --name dregon_room2_floor --floor-only \
  --frozen-mean $FM --jobs 1 --threads 8 --progress 100 --out $F2
echo "exit_floor=$?"
mv -f $F2/dregon_room2_floor__flight_floor_only.json \
      $F2/dregon_room2_floor__flight_floor_only_v2.json
# then a read-back print of comb_gain_db, the two init diagnostics, the floor,
# convergence and the objective
echo R2FLOORFIX_DONE
```

Unchanged from `floor2`: the support set (`dregon-floor`, the five room-2
free-flight windows, telemetry rule untouched), the frozen comb (the four R2
`bench_dregon_Motor{1,2,3,4}_70__bench.json`, log-mean rule), the optimiser
(driver defaults: 1500 Adam at lr 0.02 on batches of 8 frames, 200 + 100 L-BFGS
on 64 frames, seed 0, `--frame-stride 4 --max-frames 256`), and the in-job
supports rebuild + `cp` into `SU.CACHE_DIR` (= `round1/supports`), which is
REQUIRED because the npz caches are gitignored. The bench fit JSONs ARE in the
job checkout: `results/` is gitignored but those files are tracked (`git add
-f`), so `$FM` resolves to four paths.

Changed: only the output name. `--out` is a directory and the driver names the
file `<name>__<mode>.json`, so the rename happens after the fit; the committed
`floor2` payload (`dregon_room2_floor__flight_floor_only.json`) is left intact
and the two are comparable side by side. The `support` field is still
`dregon_room2_floor`, which is what `noise_v2_round_score.py --fit dregon=<path>`
requires.

## What the harvest step must do

1. `omnirun status nv2-r2-dregon-floor3-6ee9e2`; on `succeeded`, check
   `omnirun logs nv2-r2-dregon-floor3-6ee9e2 | tail -30` for `exit_floor=0`,
   the read-back print and `R2FLOORFIX_DONE`, then fetch:

   ```
   set -a; . ./.env; set +a
   aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
     s3://omnirun-artifacts/nv2-r2-dregon-floor3-6ee9e2/outputs/results/ results/
   ```

2. Commit `results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only_v2.json`
   with `git add -f`. Quote, next to the convergence label
   (`optimiser.converged`, `which_converged`, `lbfgs_restart_gain_per_cell` vs
   the 1e-4 tolerance, `grad_norm`): `params.profile.comb_gain_db`,
   `diagnostics.init_comb_gain_db`, `diagnostics.floor_level_db`,
   `params.floor.floor_mean_db` vs `diagnostics.init_floor_mean_db` (now a real
   snapshot), both exponents, and `objective.whittle_nats / objective.n_cells`
   against `floor2`'s -17697677.791956328 / 2056320 = -8.6065 nats/cell (that
   fit was NOT converged: `lbfgs_restart_gain_per_cell` 6.94e-3).

3. Score the DREGON arm — the recipe is `round2/render/submit_note.md` with the
   fit, candidate and arm path moved to the refit:

   ```
   cd <detached worktree at the harvest SHA> && COLUMNS=200 omnirun submit \
     --backend uni-gpushort --gpus 1 --time 45m --name nv2-r2-score-dregon-v2b \
     --outputs 'results/noise_v2/rounds/round2/render/*.json' \
     --outputs 'results/noise_v2/rounds/round2/render/*.md' \
     --outputs 'results/noise_v2/rounds/round2/render/audio/**' \
     -- bash -lc 'set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src:scripts; python scripts/noise_v2_round_score.py --round 2 --fits results/noise_v2/rounds/round2/fits --rigs dregon --fit dregon=results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only_v2.json --candidate dregon_v2_r2_floor_combgain --arm-out results/noise_v2/rounds/round2/render/arm_dregon_v2b.json --dump-audio results/noise_v2/rounds/round2/render/audio/dregon_v2b'
   ```

   The refit JSON must be committed and pushed BEFORE this submit: the score job
   reads it out of its own checkout.

4. Bars: DREGON cruise PIT MAE PARITY ≤ 2.187786 rev/s, STRETCH ≤ 1.897063;
   proxy `ltas_abs_db` ≤ 1.978609 dB. The `v2` arm to beat is 75.093968 rev/s
   (`round2/render/arm_dregon_v2.json`); the indicative counterfactual, two of
   five supports with the comb hand-levelled onto the real band level, is 2.029
   and 4.366 rev/s (`round2/render_dregon/findings.md`) — that is a bound on
   what one free scalar can buy, not a parity claim.
