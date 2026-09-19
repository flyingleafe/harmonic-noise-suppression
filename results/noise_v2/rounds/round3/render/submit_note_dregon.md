# R3 DREGON score arm — submitted on colab, awaiting harvest

Written by `R3ScoreDregon`. `arm_dregon_v2.json` does not exist yet; this note
is the handover (job id + exact commands) in case the harness dies first.

## The fit it scores

`results/noise_v2/rounds/round3/fits/dregon_room2_floor__flight_floor_lowk.json`
(schema `noise-v2-fit/2`, committed in `7022c984`, produced by
`nv2-r3-dregon-floor-lowk-c3b27b` at code `67318397`). It is the R3 floor-only
flight refit that adds the per-order low-order gains `low_order_gain_db`
(k = 1..8) on top of R2's single `comb_gain_db` re-levelling scalar; the frozen
comb is the log-mean of the four `Motor{1..4}_70` bench fits
(`frozen_from.profile_orders = 118`).

| quantity | value |
| --- | ---: |
| `params.profile.comb_gain_db` | **−6.918144 dB** |
| `params.profile.low_order_gain_db` | 8 free scalars, k = 1..8 (recorded in the fit) |
| `params.floor.floor_mean_db` | −38.272988 |
| `params.floor.floor_exp` / `floor_static_rel` | 22.816840 / 3.807542 |
| `objective.whittle_nats / n_cells` | −17670197.599348 / 2056320 = −8.59409 |
| `optimiser.converged` | **False** (restart gain/cell 5.9811e-03 vs tol 1e-4, grad 3217.44) |
| `optimiser.wall_s` | 4744.5 |

The fit is **NOT converged**; every number derived from it must be quoted with
that label. See `round3/fits/findings.md` for the fit-side reading.

## The job

| field | value |
| --- | --- |
| job | `nv2-r3-score-dregon-31ff78` |
| backend | `colab`, `--gpus 1 --time 30m` |
| code SHA | `4a0ab14f1b0c5382143398e3a7ca4403a7822b05` (`4a0ab14f`, `origin/main`) |
| submitted | 2026-09-19, via `omnirun --daemon localhost:18787` (ssh tunnel) |
| status at submit | `queued` — "no slot free right now"; any later `omnirun ps/status/tick` retries placement |
| submitted from | detached worktree `.worktrees/submit-R3ScoreDregon` at `4a0ab14f` |
| candidate | `dregon_v2_r3_floor_lowk` |
| arm it writes | `results/noise_v2/rounds/round3/render/arm_dregon_v2.json` (+ audio `round3/render/audio/dregon_v2/*.npz`) |

`4a0ab14f` descends from `58ab33ca` (the scorer accepts `noise-v2-fit/2`) and
from `7022c984` (the fit itself), so it is a SHA that can score this fit.

## The command

This is `round2/render/submit_note_v2b.md`'s invocation with round-3 paths, the
lowk fit, the candidate name, the arm path and the audio directory moved, plus
R3's colab backend and the three `--env` vars (as
`round3/render/submit_note_michaels.md`), so the two R3 arms stay comparable
with each other and with R2's.

```
cd .worktrees/submit-R3ScoreDregon && COLUMNS=200 omnirun --daemon localhost:18787 submit \
  --backend colab --gpus 1 --time 30m --name nv2-r3-score-dregon \
  --env R2_ACCOUNT_ID="$R2_ACCOUNT_ID" --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --outputs 'results/noise_v2/rounds/round3/render/**' \
  -- bash -lc 'set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src:scripts; python scripts/noise_v2_round_score.py --round 3 --fits results/noise_v2/rounds/round3/fits --rigs dregon --fit dregon=results/noise_v2/rounds/round3/fits/dregon_room2_floor__flight_floor_lowk.json --candidate dregon_v2_r3_floor_lowk --arm-out results/noise_v2/rounds/round3/render/arm_dregon_v2.json --dump-audio results/noise_v2/rounds/round3/render/audio/dregon_v2'
```

The three variables were expanded from the LOCAL `./.env` at call time (`set -a;
. ./.env; set +a` in the submitting shell), so no secret appears in this note,
in the job command or in any log. They are required on a managed backend: there
is no `.env` on a colab/kaggle node, and the r2:// checkpoint loader derives
bucket and endpoint from `R2_ACCOUNT_ID`.

## Harvest

```
omnirun --daemon localhost:18787 status nv2-r3-score-dregon-31ff78
omnirun --daemon localhost:18787 logs nv2-r3-score-dregon-31ff78 | tail -40
set -a; . ./.env; set +a
aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  s3://omnirun-artifacts/nv2-r3-score-dregon-31ff78/outputs/results/ /tmp/r3score_dregon_pull/
git add -f results/noise_v2/rounds/round3/render/arm_dregon_v2.json
```

Sync into a SCRATCH dir and copy only `arm_dregon_v2.json` out: the job's
`--outputs` tree carries the node checkout's other `round3/render/` files
(including the runner's own `findings.md`, which belongs to the round record,
not to this arm).

Verify before committing: `gates.hppnet` (PIT MAE + 95 % upper bound),
`gates.proxy.ltas_abs_db`, `bars`, and `protocol.scorer.sha256 ==
6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1` (the run dies
on a mismatch, so its presence IS the verification).

Bars: DREGON PARITY ≤ 2.187786 rev/s, STRETCH ≤ 1.897063; proxy ≤ 1.978609 dB.
R2's DREGON arm to beat: PIT MAE 72.34 rev/s, proxy 3.50 dB
(`round2/render/arm_dregon_v2b.json`).

## Round record

Composed by this agent once the arm lands:

```
python scripts/noise_v2_round_score.py --round 3 \
  --compose results/noise_v2/rounds/round3/render/arm_dregon_v2.json \
            results/noise_v2/rounds/round3/render/arm_michaels_v2.json \
  --supports-index results/noise_v2/rounds/round2/supports/index.json \
  --previous results/noise_v2/rounds/round2.json \
  --compare results/noise_v2/rounds/round2.json \
  --arm-job dregon_v2_r3_floor_lowk=nv2-r3-score-dregon-31ff78 \
  --arm-job michaels_v2_r3_all=nv2-r3-score-michaels-ed8576
```

If colab stalls > 40 min with no log lines, cancel and resubmit once on
`--backend kaggle` with the same three `--env` vars, and note it here.
