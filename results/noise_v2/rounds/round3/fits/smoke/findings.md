# Noise model v2 — round 3 fits (Model R3)

1 fit JSON(s) under `results/noise_v2/rounds/round3/fits/smoke`. Every number below is read from a `noise-v2-fit/2` payload in that directory (a `/1` payload's per-order OU is mapped onto its equivalent Lorentzian width first); nothing is recomputed here. The `gamma` columns are the rotor-max width in Hz at that order.

## Per-support parameters

| support | mode | R | k_max | sigma_nu | lam | g(k=1) | g(k=2) | g(k=4) | g(k=8) | g(k=16) | g(k=32) | low-k check | pinned | carrier rev/s | floor dB | whittle nats | comb | floor | cells | conv | wall s |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:-:|---|---|--:|--:|--:|--:|--:|:-:|--:|
| `bench_dregon_Motor2_60` | bench | 1 | 130 | 0.3970 | 0.170 | 0.0002483 | 0.008509 | 0.01453 | 0.04577 | 1.193 | 34.72 | pass | — | 58.144 | -57.08 | -7.66281e+06 | -7.38363e+06 | -279187 | 687752 | y | 209 |

## Population median [IQR]

| set | n | sigma_nu | lam | gamma log-mean over orders (Hz) |
|---|--:|---|---|---|
| DREGON single-motor bench (all throttles) | 1 | 0.397 [0.397, 0.397] | 0.1703 [0.1703, 0.1703] | 4.348 [4.348, 4.348] |

## Multi-start restarts

Each support was fitted from several starts: start 0 from the data-driven initialisation, the others from a log-normal perturbation of the dynamics init (`OptimSpec.init_jitter`; a bare seed change is a no-op because a bench fit is deterministic). The REPORTED fit above is the start with the lowest polished objective. `best-median` and `best-worst` are that objective's advantage over the median and the worst start, per observed cell, against the same 0.0001 nats/cell tolerance the convergence test uses; `starts agree` is yes only when even the worst start is inside it. The dynamics columns are min / median / max over the starts.

| support | starts | best nats/cell | best-median | best-worst | starts agree | L-BFGS conv | sigma_nu | lam | gamma log-mean |
|---|--:|--:|--:|--:|:-:|:-:|---|---|---|
| `bench_dregon_Motor2_60` | 4 | -11.1418 | 1.27e-05 | 0.000114 | N | y | 0.395 / 0.397 / 0.397 | 0.0495 / 0.113 / 0.17 | 4.28 / 4.32 / 4.59 |

0 of 1 supports have every start inside the tolerance. The widest disagreement is `bench_dregon_Motor2_60` at 0.000114 nats/cell, 1 times the tolerance. `lam` alone spans a factor of 3.44 (median over supports) and up to 3.44 across the starts of one support: on this evidence the R1 bench MAP problem is multi-modal, and a number computed from a single start is a draw from that multiplicity rather than an estimate.

## The low-order `gamma_rk` check

R3's one possible degeneracy: in the Brownian limit the shaft term is Lorentzian too, so a `gamma_rk` ramping as `k^2` would absorb it. PASS is every fitted width at `k <= 4` sitting within a factor of 3 of the window's resolution floor `1 / (2 T)`; `fail_k2_ramp` is `gamma_4 / gamma_1 >= 8`, and then `lam` takes the long-lag pin instead of the bench value.

| support | verdict | max gamma(k<=4) / resolution | gamma_4 / gamma_1 |
|---|:-:|--:|--:|
| `bench_dregon_Motor2_60` | pass | 0.349 | 58.5 |

Every fit passes the check.

## Prior edges and non-convergence

- `bench_dregon_Motor2_60`: lam = 0.17 outside the prior's central 95 % [0.271, 14.8]
