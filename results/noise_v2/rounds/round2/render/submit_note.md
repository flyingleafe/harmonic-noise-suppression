# R2 DREGON arm — submitted, LANDED and harvested

Written by `R2ScoreDregon`. The job **succeeded** at 2026-09-18T08:08:47Z (exit
code 0, 4 min wall); its outputs were synced and committed, so this note is now
only the record of how the arm was produced. The scored arm is
`round2/render/arm_dregon_v2.json` (committed in `12546c11`) and the round
record it feeds is `results/noise_v2/rounds/round2.json` (`e3c80203`).

## The job

| field | value |
| --- | --- |
| job | `nv2-r2-score-dregon-3f8397` |
| backend | `uni-gpushort`, `--gpus 1 --time 45m` |
| code SHA | `fc18285faea7684eb86a281685d7c548938e0ec1` (`fc18285f`, `origin/main`) |
| submitted | 2026-09-18T08:04:30Z |
| started | 2026-09-18T08:04:44Z |
| status at submit | `starting` / scheduler `running` (`omnirun status nv2-r2-score-dregon-3f8397`) |
| submitted from | detached worktree `.worktrees/submit-R2ScoreDregon` at `fc18285f` |
| output the job writes | `results/noise_v2/rounds/round2/render/arm_dregon_v2.json`, `round2/render/score/findings.md`, `round2/render/audio/dregon_v2/*.npz` |

## The command

```
cd .worktrees/submit-R2ScoreDregon && COLUMNS=200 omnirun submit \
  --backend uni-gpushort --gpus 1 --time 45m --name nv2-r2-score-dregon \
  --outputs 'results/noise_v2/rounds/round2/render/*.json' \
  --outputs 'results/noise_v2/rounds/round2/render/*.md' \
  --outputs 'results/noise_v2/rounds/round2/render/audio/**' \
  -- bash -lc 'set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src:scripts; python scripts/noise_v2_round_score.py --round 2 --fits results/noise_v2/rounds/round2/fits --rigs dregon --fit dregon=results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only.json --candidate dregon_v2_r2_floor_benchcomb --arm-out results/noise_v2/rounds/round2/render/arm_dregon_v2.json --dump-audio results/noise_v2/rounds/round2/render/audio/dregon_v2'
```

This is R1's `nv2-r1-score-dregon` invocation (see `round1/render/`) with the
round, the fits directory, the explicit `--fit`, the candidate name and the arm
path moved to round 2. The three output globs are R1's: their union is
`round2/render/**`, and the JSON/markdown globs are listed FIRST because R1's
first Michael's attempt collected everything and was cut off mid-upload — the
small records must be safe before the ~100 MB audio dump is attempted.

## Why no in-job supports build

The score path never touches a supports `.npz`: `measure_support`
(`scripts/noise_v2_round_score.py:524`) loads each frozen scored window
directly with `RE.load_window`, and the frozen DREGON supports are the five
room-2 free-flight recordings (`GT.DREGON_CRUISE_SUPPORTS`), not bench windows.
The only place the bench enters is `_comb_mean_check`
(`scripts/noise_v2_round_score.py:412`), which re-derives the frozen comb from
the four bench FIT JSONs named in the floor fit's `frozen_from.frozen_comb_paths`
— and those paths are `round2/fits/bench_dregon_Motor{1,2,3,4}_70__bench.json`,
which were fitted on the R2 windows (`front_end.duration_s = 31.69`,
`n_samples = 507040`, the `round2/supports/index.json` `dregon-bench` geometry;
R1's windows were 7.29 s / 116683 samples). `--fits
results/noise_v2/rounds/round2/fits` is therefore what pins the R2 index into
this job, and no `cp` into `SU.CACHE_DIR` is needed here (unlike the fit jobs,
which do load supports).

## What the harvest step must do

1. `omnirun status nv2-r2-score-dregon-3f8397`; on `succeeded`, fetch:

   ```
   set -a; . ./.env; set +a
   aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
     s3://omnirun-artifacts/nv2-r2-score-dregon-3f8397/outputs/results/ results/
   ```

2. Verify `round2/render/arm_dregon_v2.json` carries `gates.hppnet` (PIT MAE and
   the 95 % upper bound), `gates.proxy.ltas_abs_db`, `bars`, and
   `protocol.scorer.sha256 == 6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1`
   (the run dies on a digest mismatch, so its presence IS the verification).
   Commit with `git add -f` (`results/` is gitignored).

3. Compose the round record:

   ```
   PYTHONPATH=src:scripts python scripts/noise_v2_round_score.py --round 2 \
     --compose results/noise_v2/rounds/round2/render/arm_dregon_v2.json \
     --supports-index results/noise_v2/rounds/round2/supports/index.json \
     --arm-job dregon_v2_r2_floor_benchcomb=nv2-r2-score-dregon-3f8397
   ```

Bars: PARITY DREGON cruise PIT MAE ≤ 2.187786 rev/s, STRETCH ≤ 1.897063;
Michael's PARITY equal-regime ≤ 3.177994. Proxy `ltas_abs_db` ≤ 1.978609 dB
DREGON, 1.219668 dB Michael's.
