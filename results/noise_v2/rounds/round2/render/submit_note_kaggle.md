# R2 score arms — daemon outage and the kaggle re-pin (handover)

Written by `R2Close`. Nothing here is a result: both arms are still unscored at
the time of writing. This note exists so the exact state and the exact commands
survive a harness death.

## Timeline (all UTC, 2026-09-18)

| time | event |
| --- | --- |
| 10:59 | `nv2-r2-score-michaels2-78d01c` submitted (`uni-gpushort`, SHA `e6c06769`) |
| 11:13 | `nv2-r2-score-dregon-v2b-41d97e` submitted (`uni-gpushort`, SHA `dec46b62`) |
| 14:42 → 16:33 | `R2Close` polls both every 5 min: `status: queued` on every single call, never `starting` |
| 16:10 | `omnirun explain` shows all three `uni-gpushort` slots "provider at capacity", estimated wait down to 720 s (from 2640 s at 13:17) — contention easing, but no placement |
| ~16:39 | **the omnirun daemon becomes unreachable**: every CLI call returns `error: cannot reach the omnirun daemon at http://10.100.0.1:8787 (timed out)` |
| 18:08 | raw `curl http://10.100.0.1:8787/` also times out (exit 124) — the tunnel/host is down, not the CLI |
| 18:08 | R2 bucket checked directly: `s3://omnirun-artifacts/nv2-r2-score-michaels2-78d01c/` and `.../nv2-r2-score-dregon-v2b-41d97e/` are both **empty** — neither job ever ran or uploaded anything |
| 18:12 | Main relays a user override: re-pin BOTH arms to the `kaggle` backend, retry the submit every 2 min until the daemon answers, cancel the two `uni-gpushort` jobs once the kaggle ones are accepted |
| 18:14 → 19:02+ | kaggle submit retried every ~3 min from `.worktrees/submit-R2Close` @ `643b3058`; every attempt dies at `· resolving repo state…` with the same daemon-unreachable error |

Nothing has been lost. The two `uni-gpushort` jobs are still registered
daemon-side (they cannot be cancelled while the daemon is unreachable either),
and no artifact of either job exists anywhere yet.

## The kaggle commands being retried

Both are the notes' invocations (`submit_note_michaels.md`,
`submit_note_v2b.md`) with only the backend, the wall and the job name changed;
the wrapper, the scorer invocation, the fit paths, the candidate names, the arm
paths and the `--outputs` patterns are untouched, so the arms stay comparable
with `arm_dregon_v2.json` and with R1.

```
cd .worktrees/submit-R2Close && COLUMNS=200 omnirun submit \
  --backend kaggle --gpus 1 --time 30m --name nv2-r2-score-michaels-kg \
  --outputs 'results/noise_v2/rounds/round2/render/**' \
  -- bash -lc 'set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src:scripts; python scripts/noise_v2_round_score.py --round 2 --fits results/noise_v2/rounds/round2/fits --rigs michaels --fit michaels=results/noise_v2/rounds/round2/fits/michaels_fly125_all__flight.json --candidate michaels_v2_r2_all --arm-out results/noise_v2/rounds/round2/render/arm_michaels_v2.json'
```

```
cd .worktrees/submit-R2Close && COLUMNS=200 omnirun submit \
  --backend kaggle --gpus 1 --time 30m --name nv2-r2-score-dregon-v2b-kg \
  --outputs 'results/noise_v2/rounds/round2/render/*.json' \
  --outputs 'results/noise_v2/rounds/round2/render/*.md' \
  --outputs 'results/noise_v2/rounds/round2/render/audio/**' \
  -- bash -lc 'set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src:scripts; python scripts/noise_v2_round_score.py --round 2 --fits results/noise_v2/rounds/round2/fits --rigs dregon --fit dregon=results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only_v2.json --candidate dregon_v2_r2_floor_combgain --arm-out results/noise_v2/rounds/round2/render/arm_dregon_v2b.json --dump-audio results/noise_v2/rounds/round2/render/audio/dregon_v2b'
```

`643b3058` is `origin/main` and is a descendant of BOTH `e6c06769` (the
`MICHAELS_FIT_SUPPORTS` arm-selection fix) and `dec46b62` (the
`comb_gain_db` refit JSON), so one SHA carries everything both arms need.

### Kaggle-specific checks the harvest step must make

The user's instruction: in the FIRST log lines of each kaggle run, verify that
(a) the HPPNet checkpoint is fetched and its sha256 check passes and (b) the
`dload`/R2 credentials resolve on that backend. If kaggle fails on a missing
dependency or credential, the failure log is committed and reported rather than
retried blindly.

## Harvest + compose, unchanged

```
set -a; . ./.env; set +a
aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  s3://omnirun-artifacts/<job>/outputs/results/ results/
```

then `git add -f` the arm JSON and compose:

```
PYTHONPATH=src:scripts python scripts/noise_v2_round_score.py --round 2 \
  --compose results/noise_v2/rounds/round2/render/arm_michaels_v2.json \
            results/noise_v2/rounds/round2/render/arm_dregon_v2b.json \
            results/noise_v2/rounds/round2/render/arm_dregon_v2.json \
  --supports-index results/noise_v2/rounds/round2/supports/index.json \
  --arm-job michaels_v2_r2_all=<michaels job> \
  --arm-job dregon_v2_r2_floor_combgain=<dregon job> \
  --arm-job dregon_v2_r2_floor_benchcomb=nv2-r2-score-dregon-3f8397 \
  --previous results/noise_v2/rounds/round1.json \
  --compare results/noise_v2/rounds/round1.json
```

`arm_dregon_v2b.json` leads the DREGON list (candidate
`dregon_v2_r2_floor_combgain` is the primary; `arm_dregon_v2.json` is the
superseded one). If a rig never lands, add
`--not-run <rig>=<reason>` for it instead of its arm.

Bars: DREGON PARITY ≤ 2.187786 rev/s, STRETCH ≤ 1.897063, proxy ≤ 1.978609 dB.
Michael's PARITY equal-regime ≤ 3.177994 rev/s (legacy per-regime 0.317103
standby / 8.007098 ramp / 0.755783 cruise), proxy ≤ 1.219668 dB.
`protocol.scorer.sha256` must be
`6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1`.
