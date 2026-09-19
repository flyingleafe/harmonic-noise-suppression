# R3 DREGON humps — HPPNet probe job (submitted, awaiting harvest)

Written by `DregonHumps` before the job returned, so the job id and the exact
commands survive a dead harness.

## The question the job scores

Three rounds of DREGON candidates score HPPNet PIT MAE 78.8 → 72.3 → 71.9 rev/s
against a real 1.22 and a legacy 2.19. The hypothesis under test: HPPNet locks
onto BROAD harmonic humps (merged four-rotor lines + in-frame drift + a genuine
in-flight per-order width) that the v2 render — bench `gamma_rk` under 1 Hz at
every k ≤ 12, frozen — cannot produce at any gain. The arms separate WIDTH from
LEVEL: `gamma_rk` overridden to a k-independent 13/30/67/130 Hz at the fitted
comb level (V0–V3) and at the solved hump-fraction match (V0m, V4–V6), the same
match on the NEEDLE comb (`v2_matched`, the control that isolates width), a comb
level dose-response at the fitted widths (+6/+12/+18/+21/+24 dB), and the legacy
arm with its Lorentzian pedestal muted (V7, `coherence_k_half = 0`).

## The job

| field | value |
| --- | --- |
| job | `nv2-r3-humps-671f38` (placed on colab at submit) |
| backend | `colab`, `--gpus 1 --time 30m` |
| code SHA | `09f3fda5fcff9e91e8ff30bcbdc0c1e509c23a60` (`09f3fda5`, pushed to `origin/main`) |
| submitted | 2026-09-19, via `omnirun --daemon localhost:18787` (ssh tunnel) |
| submitted from | detached worktree `.worktrees/submit-DregonHumps` at `09f3fda5` |
| fit scored | `results/noise_v2/rounds/round3/fits/dregon_room2_floor__flight_floor_lowk.json` (`converged` **False**) |
| windows | `free-flight`, `hovering`, `updown` (frozen DREGON room-2 cruise supports) |
| seed | 2001, 8 mics, 21 arms × 3 windows |
| writes | `results/noise_v2/rounds/round3/dregon_humps/dregon_humps.json` (+ `findings.md`) |
| superseded | `nv2-r3-humps-44954e` at `dc94a037`, CANCELLED while still queued |

`nv2-r3-humps-44954e` was cancelled before it ever ran: the first cut solved the
level match on the RAW excess in the 64 Hz hump band, which is two thirds
floor-curvature bias, and it compared widths only at that match, where every
width gets a different comb shift. `09f3fda5` matches on null-subtracted
carrier-locked power over k = 1..8 at B = 16 Hz and adds the width sweep at a
FIXED +21 dB, so width is also compared at equal comb energy. No number from the
cancelled job exists.

## The command

```
cd .worktrees/submit-DregonHumps && COLUMNS=200 omnirun --daemon localhost:18787 submit \
  --backend colab --gpus 1 --time 30m --name nv2-r3-humps \
  --env R2_ACCOUNT_ID="$R2_ACCOUNT_ID" --env AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
  --env AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
  --outputs 'results/noise_v2/rounds/round3/dregon_humps/**' \
  -- bash -lc 'set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src:scripts; python scripts/noise_v2_render_dregon.py --study humps --probe --out results/noise_v2/rounds/round3/dregon_humps'
```

The three variables were expanded from the LOCAL `./.env` at call time
(`set -a; . ../../.env; set +a` in the submitting shell), so no secret appears in
this note, in the job command or in any log. They are required on a managed
backend: there is no `.env` on a colab node and the `r2://` checkpoint loader
derives bucket and endpoint from `R2_ACCOUNT_ID`.

## Harvest

```
omnirun --daemon localhost:18787 status nv2-r3-humps-671f38
omnirun --daemon localhost:18787 logs   nv2-r3-humps-671f38 | tail -40
set -a; . ./.env; set +a
aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  s3://omnirun-artifacts/nv2-r3-humps-671f38/outputs/results/ /tmp/r3humps_pull/
```

Sync into a SCRATCH dir. The harvested `dregon_humps.json` is the PROBE pass;
the committed record is the local spectral pass with the job's PIT merged onto
it by

```
PYTHONPATH=src python scripts/noise_v2_render_dregon.py --study humps --figures \
  --merge-probe /tmp/r3humps_pull/noise_v2/rounds/round3/dregon_humps/dregon_humps.json \
  --job nv2-r3-humps-671f38 --out results/noise_v2/rounds/round3/dregon_humps
```

`--merge-probe` compares the mic-0 band level of every shared arm first and dies
on a mismatch above 0.01 dB, so a PIT number can only be merged onto the audio it
was measured on.

Verify: `protocol.scorer.sha256 ==
6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1` (the probe
loader dies on a mismatch, so its presence IS the verification) and
`arms.real.pit_mae` against the frozen real scalars — free-flight 0.638129,
hovering 0.888408, updown 1.694930 (`round3/score/findings.md`).

If colab stalls > 40 min with no log lines, cancel and resubmit once on
`--backend kaggle` with the same three `--env` vars, and note it here.

## Local pass (the same script, CPU probe)

The frozen checkpoint is in the local `.cache/r2_checkpoints` at the matching
digest, so the same run was made on the laptop (CPU probe, ~4 s per arm) and
reproduces the frozen real PIT MAE exactly (free-flight 0.6381291 against
0.638129). The colab pass is the independent confirmation of that table, not its
only source.
