# R3 PER-REGIME Michael's score arm — submitted on colab, awaiting harvest

Written by `R3Standby`. `arm_michaels_v2_regimes.json` does not exist yet;
this note is the handover (job id + exact commands) in case the harness dies
first. The two fits it scores ARE committed.

## The job

| field | value |
| --- | --- |
| job | `nv2-r3-score-michaels-re-ac8bc5` |
| backend | `colab`, `--gpus 1 --time 30m` |
| code SHA | `d9bd1940` (on `main`) |
| submitted | 2026-09-19T05:20Z, via `omnirun --daemon localhost:18787` (ssh tunnel) |
| submitted from | detached worktree `.worktrees/submit-R3Standby` at `d9bd1940` |
| candidate | `michaels_v2_r3_regimes` |
| fits scored | `round3/fits/michaels_fly125_standby__flight.json` (job `nv2-r3-michaels-standby--6c860e`, **NOT converged**) below 45 rev/s + `round3/fits/michaels_fly125_cruise__flight.json` (job `nv2-r3-michaels-cruise-03bf4e`, **NOT converged**) above 65 |
| output | `results/noise_v2/rounds/round3/render/arm_michaels_v2_regimes.json` |

## The command

Identical to the pooled R3 Michael's arm
(`round3/render/submit_note_michaels.md`) except for the candidate name, the
arm-out path and the two `--fit` flags, so the two arms stay comparable.

```
cd .worktrees/submit-R3Standby && COLUMNS=200 omnirun --daemon localhost:18787 submit \
  --backend colab --gpus 1 --time 30m --name nv2-r3-score-michaels-regimes \
  --env R2_ACCOUNT_ID="$R2_ACCOUNT_ID" --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --outputs 'results/noise_v2/rounds/round3/render/**' \
  -- bash -lc 'set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src:scripts; python scripts/noise_v2_round_score.py --round 3 --fits results/noise_v2/rounds/round3/fits --rigs michaels --fit michaels=results/noise_v2/rounds/round3/fits/michaels_fly125_cruise__flight.json --fit michaels_standby=results/noise_v2/rounds/round3/fits/michaels_fly125_standby__flight.json --candidate michaels_v2_r3_regimes --arm-out results/noise_v2/rounds/round3/render/arm_michaels_v2_regimes.json'
```

The three R2 variables were expanded from the LOCAL `./.env` at call time, so
no secret appears in this note, in the job command or in any log.

`--fit michaels_standby=...` is what makes the arm PER-REGIME
(`noise_v2_round_score.v2_regime_arm`, new at `58e895fb`): both fits are
rendered on each frozen support's own carrier and composed by
`render.render_noise_regimes` — the standby fit below
`rps_gating.STANDBY_MAX_RPS` = 45 rev/s on the slowest rotor, the cruise fit
at or above `CRUISE_MIN_RPS` = 65, their POWERS interpolated by the
`rps_gating` smoothstep in between. The likelihood gate reads the matching
`render.expected_periodogram_regimes`, so the predicted periodogram is the
same blend and not one fit's.

The composition was smoked locally at `58e895fb` before this submission (5
frozen FLY124 supports, 1 seed, 2 mics, `--no-probe`): the arm record carried
`candidate.fits.michaels.per_regime` for both fits and two likelihood cells.

## Harvest

```
omnirun --daemon localhost:18787 status nv2-r3-score-michaels-re-ac8bc5
omnirun --daemon localhost:18787 logs   nv2-r3-score-michaels-re-ac8bc5 | tail -40
set -a; . ./.env; set +a
aws s3 cp --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  s3://omnirun-artifacts/nv2-r3-score-michaels-re-ac8bc5/outputs/results/noise_v2/rounds/round3/render/arm_michaels_v2_regimes.json \
  results/noise_v2/rounds/round3/render/arm_michaels_v2_regimes.json
git add -f results/noise_v2/rounds/round3/render/arm_michaels_v2_regimes.json
```

`aws s3 cp` of the ONE file, not `s3 sync`: a job's `--outputs` tree carries
the node's whole `round3/render/**` (and the sync of the fit jobs' trees ran
at ~0.5 MB/s over 300 MB). The runner's own `round3/render/findings.md` is
NOT committed — it is `R3Bench`'s file for the round.

Verify before committing: `protocol.scorer.sha256 ==
6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1`, and that
`candidate.fits.michaels.per_regime` names BOTH fits with
`converged: false` on each.

Bars: Michael's PARITY equal-regime <= 3.177994 rev/s (legacy per-regime
0.317103 standby / 8.007098 ramp / 0.755783 cruise), proxy `ltas_abs_db` <=
1.219668 dB. The pooled R3 candidate to beat: 2.334258 equal-regime
(0.693809 cruise / 3.145259 ramp / 3.163707 standby), proxy 1.886305,
likelihood comb margin -862.83 nats/s.
