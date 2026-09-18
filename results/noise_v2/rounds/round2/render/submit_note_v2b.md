# R2 DREGON score arm for the `comb_gain_db` refit — submitted

Written by `R2FloorFix`. The arm is QUEUED/RUNNING on the cluster; nothing here
is a result.

## The fit it scores

`results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only_v2.json`
(committed in `dec46b62`), produced by `nv2-r2-dregon-floor3-6ee9e2` at code
`fca85d9c` — the same floor-only recipe as
`dregon_room2_floor__flight_floor_only.json` with ONE extra free scalar,
`comb_gain_db` (patch `fca85d9c`, note `round2/fits/floor_v2_submit_note.md`).

| quantity | refit (`_v2`) | previous (`floor2`) |
| --- | ---: | ---: |
| `params.profile.comb_gain_db` | **-3.2323 dB** | — (no such freedom) |
| `diagnostics.init_comb_gain_db` (measured seed) | +8.4863 dB | — |
| `params.floor.floor_mean_db` | -39.1143 | -38.0968 |
| `diagnostics.init_floor_mean_db` | -38.5175 (a real snapshot since `fca85d9c`) | -38.0968 (ALIASED — reads the fitted value) |
| `diagnostics.floor_level_db` | -37.8274 | -37.6280 |
| `params.floor.floor_exp` | 38.7747 | 13.0041 |
| `params.floor.floor_static_rel` | 17.1087 | 0.4690 |
| `objective.whittle_nats / n_cells` | -17696690.330698 / 2056320 = -8.60599 | -17697677.791956 / 2056320 = -8.60604 |
| `optimiser.converged` | **False** (`none`, restart gain/cell 3.190e-3 vs tol 1e-4, grad 1230.02) | **False** (`none`, 6.938e-3, grad 780.08) |
| `optimiser.wall_s` | 5905.5 | 2480.2 |

NEITHER fit is converged; every number above must be quoted with that label.

The result that matters for the hypothesis: the flight Whittle likelihood moves
the transplanted comb DOWN 3.23 dB, even though its own measured seed offered
+8.49 dB and the tracker-side counterfactual in
`round2/render_dregon/findings.md` wanted about +21.8 dB. The missing degree of
freedom is now there and identified — it just does not point where the PIT
metric needs it to.

## The job

| field | value |
| --- | --- |
| job | `nv2-r2-score-dregon-v2b-41d97e` |
| backend | `uni-gpushort`, `--gpus 1 --time 45m` |
| code SHA | `dec46b62` (`origin/main`, includes the refit JSON and `R2MichaelsHarvest`'s `MICHAELS_FIT_SUPPORTS` fix) |
| submitted | 2026-09-18T12:15Z |
| status at submit | `queued` — "no slot free right now"; any later `omnirun ps/status/tick` retries placement |
| submitted from | detached worktree `.worktrees/submit-R2FloorFix` at `dec46b62` |
| candidate | `dregon_v2_r2_floor_combgain` |
| arm it writes | `results/noise_v2/rounds/round2/render/arm_dregon_v2b.json` (+ `render/score/findings.md`, `render/audio/dregon_v2b/*.npz`) |

## The command

```
cd .worktrees/submit-R2FloorFix && COLUMNS=200 omnirun submit \
  --backend uni-gpushort --gpus 1 --time 45m --name nv2-r2-score-dregon-v2b \
  --outputs 'results/noise_v2/rounds/round2/render/*.json' \
  --outputs 'results/noise_v2/rounds/round2/render/*.md' \
  --outputs 'results/noise_v2/rounds/round2/render/audio/**' \
  -- bash -lc 'set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src:scripts; python scripts/noise_v2_round_score.py --round 2 --fits results/noise_v2/rounds/round2/fits --rigs dregon --fit dregon=results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only_v2.json --candidate dregon_v2_r2_floor_combgain --arm-out results/noise_v2/rounds/round2/render/arm_dregon_v2b.json --dump-audio results/noise_v2/rounds/round2/render/audio/dregon_v2b'
```

This is `round2/render/submit_note.md`'s invocation with only the fit path, the
candidate name, the arm path and the audio directory moved to the refit, so the
two arms are comparable: `arm_dregon_v2.json` (candidate
`dregon_v2_r2_floor_benchcomb`, PIT MAE 75.093968 rev/s) against
`arm_dregon_v2b.json` (candidate `dregon_v2_r2_floor_combgain`).

## What the harvest step must do

1. `omnirun status nv2-r2-score-dregon-v2b-41d97e`; on `succeeded`:

   ```
   set -a; . ./.env; set +a
   aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
     s3://omnirun-artifacts/nv2-r2-score-dregon-v2b-41d97e/outputs/results/ results/
   ```

2. Commit `round2/render/arm_dregon_v2b.json` with `git add -f` and read off
   `gates.hppnet` (PIT MAE + its 95 % upper bound) and `gates.proxy.ltas_abs_db`.
   `protocol.scorer.sha256` must be
   `6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1` (the run
   dies on a mismatch, so its presence IS the verification).
   Bars: PARITY ≤ 2.187786 rev/s, STRETCH ≤ 1.897063; proxy ≤ 1.978609 dB.

3. Hand it to whoever owns the round record (`R2MichaelsHarvest` at the time of
   writing): the compose is
   `scripts/noise_v2_round_score.py --round 2 --compose <arms...>
   --supports-index results/noise_v2/rounds/round2/supports/index.json
   --arm-job dregon_v2_r2_floor_combgain=nv2-r2-score-dregon-v2b-41d97e`.
   `arm_dregon_v2.json` is then the SUPERSEDED DREGON arm, not a second
   candidate on equal footing: same supports, same seed, same frozen scorer,
   one extra fitted scalar.
