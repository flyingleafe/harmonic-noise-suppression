# R2 Michael's score arm — submitted, awaiting harvest

Written by `R2MichaelsHarvest`. Nothing in `arm_michaels_v2.json` exists yet;
this note is the handover (job id + exact command) in case the harness dies
before the arm lands.

## The job

| field | value |
| --- | --- |
| job | `nv2-r2-score-michaels-5099b4` |
| backend | `uni-gpushort`, `--gpus 1 --time 45m` |
| code SHA | `c089d978b7455940249c4beb12b662a484d8a9e0` (`c089d978`, `origin/main` — the commit that carries the harvested flight fit) |
| submitted | 2026-09-18T09:30:21Z |
| status at submit | `queued` ("no slot free right now"; any later `omnirun ps`/`status`/`tick` retries placement) |
| submitted from | detached worktree `.worktrees/submit-R2MichaelsHarvest` at `c089d978` |
| output the job writes | `results/noise_v2/rounds/round2/render/arm_michaels_v2.json` (plus the runner's own `round2/render/score/findings.md`) |

## The command

```
cd .worktrees/submit-R2MichaelsHarvest && COLUMNS=200 omnirun submit \
  --backend uni-gpushort --gpus 1 --time 45m --name nv2-r2-score-michaels \
  --outputs 'results/noise_v2/rounds/round2/render/**' \
  -- bash -lc 'set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src:scripts; python scripts/noise_v2_round_score.py --round 2 --fits results/noise_v2/rounds/round2/fits --rigs michaels --fit michaels=results/noise_v2/rounds/round2/fits/michaels_fly125_all__flight.json --candidate michaels_v2_r2_all --arm-out results/noise_v2/rounds/round2/render/arm_michaels_v2.json'
```

This is the DREGON arm's recipe (`round2/render/submit_note.md`) with the rig,
the `--fit`, the candidate name and the arm path moved to Michael's. No
`--dump-audio`: R1's first Michael's attempt was cut off mid-upload by the
~100 MB audio dump, and the arm record is what the round needs.

The score path loads each frozen FLY124 scored window with `RE.load_window`,
so no supports `.npz` and no in-job supports build is needed (same reasoning as
the DREGON arm's note).

## What the harvest step must do

1. `omnirun status nv2-r2-score-michaels-5099b4`; on `succeeded`:

   ```
   set -a; . ./.env; set +a
   aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
     s3://omnirun-artifacts/nv2-r2-score-michaels-5099b4/outputs/results/ results/
   ```

   The sync overwrites `round2/fits/findings.md` and `round2/render/score/findings.md`
   with the runner's regenerated versions — check `git diff` and restore any
   hand-written file the job did not mean to replace (`git checkout --`).

2. Verify `round2/render/arm_michaels_v2.json` carries `gates.hppnet`,
   `gates.proxy.ltas_abs_db`, `bars` and
   `protocol.scorer.sha256 == 6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1`,
   then commit it with `git add -f`.

3. Re-compose the round record with BOTH arms (R2FloorFix confirmed at 09:2xZ
   that `arm_dregon_v2b.json` cannot exist before ~10:30Z, so the existing
   `arm_dregon_v2.json` is the DREGON arm; if v2b lands later it becomes the
   PRIMARY dregon arm and the record is re-composed):

   ```
   PYTHONPATH=src:scripts python scripts/noise_v2_round_score.py --round 2 \
     --compose results/noise_v2/rounds/round2/render/arm_michaels_v2.json \
               results/noise_v2/rounds/round2/render/arm_dregon_v2.json \
     --supports-index results/noise_v2/rounds/round2/supports/index.json \
     --arm-job michaels_v2_r2_all=nv2-r2-score-michaels-5099b4 \
     --arm-job dregon_v2_r2_floor_benchcomb=nv2-r2-score-dregon-3f8397 \
     --previous results/noise_v2/rounds/round1.json \
     --compare results/noise_v2/rounds/round1.json
   ```

Bars: Michael's PARITY equal-regime ≤ 3.177994 rev/s (legacy per-regime
0.317 standby / 8.007 ramp / 0.756 cruise), DREGON PARITY ≤ 2.187786, STRETCH
≤ 1.897063. Proxy `ltas_abs_db` ≤ 1.219668 dB Michael's, 1.978609 dB DREGON.
