# Noise model v2 — round 3 fits (Model R3)

1 fit JSON(s) under `results/noise_v2/rounds/round3/fourmotor/fits/F2`. Every number below is read from a `noise-v2-fit/2` payload in that directory (a `/1` payload's per-order OU is mapped onto its equivalent Lorentzian width first); nothing is recomputed here. The `gamma` columns are the rotor-max width in Hz at that order.

## Per-support parameters

| support | mode | R | k_max | sigma_nu | lam | g(k=1) | g(k=2) | g(k=4) | g(k=8) | g(k=16) | g(k=32) | low-k check | pinned | carrier rev/s | floor dB | whittle nats | comb | floor | cells | conv | wall s |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|---|---|--:|--:|--:|--:|--:|:-:|--:|
| `bench_dregon_allMotors_70` | bench | 4 | 114 | 0.5301 | 44.644 | 13.68 | 0.2477 | 0.6313 | 2.278 | 6.287 | 41.46 | fail_above_floor | — | 64.642, 67.661, 68.754, 69.576 | -37.57 | -1.61014e+07 | -1.58235e+07 | -277877 | 1973192 | N | 4521 |

## Population median [IQR]

| set | n | sigma_nu | lam | gamma log-mean over orders (Hz) |
|---|--:|---|---|---|

## Multi-start restarts

Each support was fitted from several starts: start 0 from the data-driven initialisation, the others from a log-normal perturbation of the dynamics init (`OptimSpec.init_jitter`; a bare seed change is a no-op because a bench fit is deterministic). The REPORTED fit above is the start with the lowest polished objective. `best-median` and `best-worst` are that objective's advantage over the median and the worst start, per observed cell, against the same 0.0001 nats/cell tolerance the convergence test uses; `starts agree` is yes only when even the worst start is inside it. The dynamics columns are min / median / max over the starts.

| support | starts | best nats/cell | best-median | best-worst | starts agree | L-BFGS conv | sigma_nu | lam | gamma log-mean |
|---|--:|--:|--:|--:|:-:|:-:|---|---|---|
| `bench_dregon_allMotors_70` | 4 | -8.1601 | 0.000273 | 0.000337 | N | N | 0.53 / 0.554 / 0.558 | 44.6 / 47 / 47.4 | 1.95 / 2.09 / 2.66 |

0 of 1 supports have every start inside the tolerance. The widest disagreement is `bench_dregon_allMotors_70` at 0.000337 nats/cell, 3 times the tolerance. `lam` alone spans a factor of 1.06 (median over supports) and up to 1.06 across the starts of one support: on this evidence the R1 bench MAP problem is multi-modal, and a number computed from a single start is a draw from that multiplicity rather than an estimate.

## The low-order `gamma_rk` check

R3's one possible degeneracy: in the Brownian limit the shaft term is Lorentzian too, so a `gamma_rk` ramping as `k^2` would absorb it. PASS is every fitted width at `k <= 4` sitting within a factor of 3 of the window's resolution floor `1 / (2 T)`; `fail_k2_ramp` is `gamma_4 / gamma_1 >= 8`, and then `lam` takes the long-lag pin instead of the bench value.

| support | verdict | max gamma(k<=4) / resolution | gamma_4 / gamma_1 |
|---|:-:|--:|--:|
| `bench_dregon_allMotors_70` | fail_above_floor | 941 | 3.03 |

0 of 1 fits pass. Failing: `bench_dregon_allMotors_70`

## Prior edges and non-convergence

- `bench_dregon_allMotors_70`: lam = 44.64 outside the prior's central 95 % [0.271, 14.8]
- `bench_dregon_allMotors_70`: low-order gamma check fail_above_floor
- `bench_dregon_allMotors_70`: NOT converged (L-BFGS restart still improving)
