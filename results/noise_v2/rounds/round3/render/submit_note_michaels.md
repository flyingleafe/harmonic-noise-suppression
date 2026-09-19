# R3 Michael's score arm — submitted on colab, awaiting harvest

Written by `R3Michaels`. `arm_michaels_v2.json` does not exist yet; this note
is the handover (job id + exact commands) in case the harness dies first.

## The job

| field | value |
| --- | --- |
| job | `nv2-r3-score-michaels-ed8576` |
| backend | `colab`, `--gpus 1 --time 30m` |
| code SHA | `62e45d96743b27eaf9d95f08ce69f39417578b74` (`62e45d96`, on `main`) |
| submitted | 2026-09-19T00:5xZ, via `omnirun --daemon localhost:18787` (ssh tunnel) |
| submitted from | detached worktree `.worktrees/submit-R3Michaels` at `62e45d96` |
| fit scored | `results/noise_v2/rounds/round3/fits/michaels_fly125_all__flight.json` (schema `noise-v2-fit/2`, job `nv2-r3-michaels-flight-8197c4`, **NOT converged** — see `round3/fits/findings_flight.md`) |
| output | `results/noise_v2/rounds/round3/render/arm_michaels_v2.json` (+ the runner's own `round3/render/score/findings.md`) |

`62e45d96` is the first SHA that can score an R3 fit at all: it descends from
`58ab33ca` ("noise-v2 R3 score: accept schema /2", which replaced the two
one-line `noise-v2-fit/1` gates in `scripts/noise_v2_round_score.py` and
`scripts/noise_v2_render_regime.py` by
`experiments.noise_model.READABLE_FIT_SCHEMAS`) and from `e7510b5f`, which
committed the fit itself.

## The command

Backend, wall and the three `--env` vars are the only differences from R2's
Michael's arm (`round2/render/submit_note_michaels.md`); the wrapper, the
scorer invocation and the `--outputs` pattern are the R2 ones with round 3
paths, so the arms stay comparable.

```
cd .worktrees/submit-R3Michaels && COLUMNS=200 omnirun --daemon localhost:18787 submit \
  --backend colab --gpus 1 --time 30m --name nv2-r3-score-michaels \
  --env R2_ACCOUNT_ID="$R2_ACCOUNT_ID" --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --outputs 'results/noise_v2/rounds/round3/render/**' \
  -- bash -lc 'set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src:scripts; python scripts/noise_v2_round_score.py --round 3 --fits results/noise_v2/rounds/round3/fits --rigs michaels --fit michaels=results/noise_v2/rounds/round3/fits/michaels_fly125_all__flight.json --candidate michaels_v2_r3_all --arm-out results/noise_v2/rounds/round3/render/arm_michaels_v2.json'
```

The three variables were expanded from the LOCAL `./.env` at call time (`set
-a; . ./.env; set +a` in the submitting shell), so no secret appears in this
note, in the job command or in any log. They are required and sufficient on a
managed backend: there is no `.env` on a colab/kaggle node (settled
empirically in R2, `round2/render/submit_note_kaggle.md`), and
`utils.checkpoints.load_r2_env` plus
`stochastic_fit_revised_eval._download_s3_uri` derive bucket and endpoint from
`R2_ACCOUNT_ID`. Without them the run dies with `r2:// checkpoint ...
requested but R2 creds missing in .env`.

No `--dump-audio`: the arm record is what the round needs (R2's Michael's arm
made the same call).

## Harvest

```
omnirun --daemon localhost:18787 status nv2-r3-score-michaels-ed8576
omnirun --daemon localhost:18787 logs nv2-r3-score-michaels-ed8576 | tail -40
set -a; . ./.env; set +a
aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  s3://omnirun-artifacts/nv2-r3-score-michaels-ed8576/outputs/results/ /tmp/r3score_pull/
git add -f results/noise_v2/rounds/round3/render/arm_michaels_v2.json
```

Sync into a SCRATCH dir and copy only `arm_michaels_v2.json` out:
`R3Bench` owns `round3/fits/findings.md` and `round3/supports/index.json`, and
a job's `--outputs` tree can carry the other agent's files (the flight job's
tree did, because both jobs share the node's checkout at the same SHA).

Verify before committing: `gates.hppnet`, `gates.proxy.ltas_abs_db`, `bars`,
and `protocol.scorer.sha256 ==
6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1`.

Bars: Michael's PARITY equal-regime ≤ 3.177994 rev/s (legacy per-regime
0.317103 standby / 8.007098 ramp / 0.755783 cruise), proxy `ltas_abs_db` ≤
1.219668 dB. The round record is composed by whoever owns both arms
(`--compose ... --arm-job michaels_v2_r3_all=nv2-r3-score-michaels-ed8576`).

## Harvested (2026-09-19)

`succeeded`, exit 0, ran 00:40:46Z → 01:00:55Z (20 min wall, colab, GPU).
`arm_michaels_v2.json` fetched from R2 into a scratch dir, copied out alone and
committed (`03b18fbd`); `protocol.scorer.sha256` verified equal to
`6e50e025…2877b1` and `gates.hppnet` / `gates.proxy.ltas_abs_db` / `bars` all
present. The runner's own `round3/render/findings.md` was NOT committed (it is
`R3Bench`'s file for the round; the michaels copy stays in the job's artifact
tree, `s3://omnirun-artifacts/nv2-r3-score-michaels-ed8576/`).

Every number below is from the committed arm JSON, and the fit it scores is
**NOT converged** (`candidate.fits.michaels.converged = false`,
`lbfgs_restart_gain_per_cell = 9.0618e-03`, `grad_norm = 5430.15`).

| quantity | R3 | R2 | bar |
|---|---:|---:|---:|
| HPPNet equal-regime mean PIT MAE (rev/s) | **2.334258** | 2.456399 | ≤ 3.177994 → **within**, margin 0.843736 (R2: 0.721596) |
| aggregate ratio vs legacy 3.026661 | 0.771232 | 0.811587 | ≤ 1.05 |
| per-regime cruise (legacy 0.755783) | **0.693809** (ratio 0.918000) | 0.969167 (1.282334) | — |
| per-regime ramp (legacy 8.007098) | **3.145259** (0.392809) | 3.393256 (0.423781) | — |
| per-regime standby (legacy 0.317103) | **3.163707** (9.976916) | 3.006774 (9.482018) | — |
| proxy `ltas_abs_db` michaels cruise | **1.886305** | 2.149092 | ≤ 1.219668 → **FAIL**, margin −0.666637 (R2: −0.929423) |
| likelihood comb margin (nats/s) | **−862.83** (below oracle ✓) | −970.15 | below oracle |
| likelihood floor margin (nats/s) | +201.56 | +71.79 | — |

`pass = false` and `parity_pass = false` for the arm record as a whole because
it carries ONE rig: the DREGON cohort is empty in a michaels-only arm (same as
R2's michaels arm). The michaels frozen gate itself is `frozen_gate_pass =
true` with `cohort_complete = true` and all three regimes present.
