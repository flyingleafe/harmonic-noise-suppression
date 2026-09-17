# Basin / ridge diagnosis of the v2 objective in the dynamics parameters

Round 1, `results/noise_v2/rounds/round1/basin/`. Every number below is read from
a committed JSON in this directory; the file is named at each table. The scans
were produced by `scripts/noise_v2_basin.py` (schema `noise-v2-basin/1`) on
`uni-cpu`:

| job | what | wall | outputs |
| --- | --- | --- | --- |
| `nv2-r1-basin-mich-shaft-bb1368` | Michael's shaft grid + six 1-D profiles | 884.4 s + 501.2 s (226 + 352 objective evals) | `michaels_fly125_cruise__flight__grid_shaft.json`, `…__profiles.json` |
| `nv2-r1-basin-mich-order-354905` | Michael's odd per-order grid | 883.0 s (226 evals) | `michaels_fly125_cruise__flight__grid_order_odd.json` |
| `nv2-r1-basin-bench-bbed0e` | converged bench control, all three scans | 20.3 + 20.1 + 11.2 s | `bench_dregon_Motor1_80__bench__*.json` |
| `nv2-r1-basin-slices-a92306` | `sigma_nu` profiled at three candidate `lam` pins | 131.1 s (46 evals) | `michaels_fly125_cruise__flight__slices_sigma_nu_at_lam.json` |
| `nv2-r1-michaels-retry-a3bfe1` | the ONE permitted retry fit | see §8 | `../fits/michaels_fly125_cruise__flight_retry.json` |

Code at `bb1669c4` (scan) and `2e96fa29` (ridge-profiled verdict, slices); the
pin plumbing the retry needs is `463fd10b`.

## 1. The ridge coordinates, derived from `lag.py::r_tau`

`r_tau` is `exp(-shaft - order)` with

    shaft = k^2 V(tau) / 2,   V(tau) = 2 sigma_nu^2 / lam^2 * (lam|tau| - 1 + e^(-lam|tau|))
    order = sigma_eps^2 k^p * (1 - e^(-lam_eps|tau|)),   p = 1

(`V` is `revised_phase.integrated_ou_increment_var`, imported, not re-derived.)
Write `x = lam|tau|`.

**Shaft, diffusive limit `x >> 1`.** `lam|tau| - 1 + e^-x -> x - 1`, so

    V -> 2 (sigma_nu^2 / lam) |tau| - 2 sigma_nu^2 / lam^2 = 2 D |tau| - 2 D / lam,
    D = sigma_nu^2 / lam   [rad^2/s]

— Brownian phase with diffusion `D` plus a LAG-INDEPENDENT offset `2D/lam`. At
fixed `D` the entire lag dependence is fixed and the only residual `lam`
dependence is the constant coherence gain `exp(+k^2 D / lam)`, which vanishes as
`lam` grows. **Ridge coordinate: `D = sigma_nu^2 / lam`.**

**Shaft, frozen-rate limit `x << 1`.** The series in `integrated_ou_increment_var`
gives `V -> sigma_nu^2 tau^2`: a Gaussian line of sd `k sigma_nu / 2pi` Hz, with
`lam` ABSENT from the model altogether. **Ridge coordinate: `sigma_nu` itself,
and `lam` is not a parameter of the limit at all.**

The two branches meet at the crossover order `k* = sqrt(2) lam / sigma_nu`
(order `k`'s own decoherence lag `tau_k ~ sqrt(2)/(k sigma_nu)` equals `1/lam`).
At the first fit's point `k* = 33.72` (`grid_shaft.json:crossover_order_at_fit`),
i.e. orders above 34 of the modelled 81 are frozen-branch, below it diffusive.

**Per-order, saturated limit `lam_eps|tau| >> 1`.** `order -> sigma_eps^2 k^p`, a
lag-independent coherent fraction `exp(-sigma_eps^2 k^p)`. **Ridge coordinate:
`A = sigma_eps^2`**, with `lam_eps` unidentified above `1/tau_max`.

**Per-order, diffusive limit `lam_eps|tau| << 1`.** `order -> sigma_eps^2 lam_eps
k^p |tau|`: a per-order diffusion. **Ridge coordinate: the PRODUCT
`D_eps = sigma_eps^2 lam_eps`.** Which limit an order sits in is decided per
order: order `k` decoheres in the per-order term at `tau ~ 1/(A k lam_eps)`, so
at the first fit's odd values (`A = 0.354395`, `lam_eps_odd = 75.6331`) order 81
decoheres at 0.46 ms, far inside `1/lam_eps = 13.2 ms` — the high orders see only
the diffusive part, the low orders see the saturation.

**Scale of the window.** The flight lag grid is `tau_s_work = 0 .. 8191/64000 s`,
i.e. **0 .. 127.98 ms** (`spectrum.flight_grid`; 2048 taps at 16 kHz on the
4x work grid), NOT 1.024 s — 1.024 s is the bench floor-lag constant. The first
fit's `lam = 199.78` is a 5.01 ms correlation time, 3.9 % of the lag support.

## 2. What was evaluated, and in what units

`scripts/noise_v2_basin.py` loads a fit JSON, rebuilds the fit's OWN batch
(asserting the rebuilt `k_max` and cell count against the JSON; the fitted-point
objective reproduces the fit's own `whittle_nats` to the last stored digit (bench
control: scan `-12203279.952135311` against the fit JSON's
`-12203279.95213531`, 1e-16 relative), holds every block at the fitted values,
and moves only the dynamics block.

* `risk` = `model.whittle_risk`, the fit's likelihood term verbatim.
* `map` = `risk` minus the dynamics block's log-prior — the only prior term that
  moves under a dynamics scan, so MAP DIFFERENCES over a grid are exact. All
  numbers below are `map`, in **nats per observed cell of the evaluated batch**.
* Michael's scans use the L-BFGS polish's own deterministic 64-frame subset
  (`np.linspace` over the 256 pooled frames), **516096 cells** — the same frame
  set and the same denominator `fit.py` divides its restart gain by, so the fit's
  convergence tolerance `1e-4 nats/cell` and its observed restart gain
  `0.01875590 nats/cell` are directly comparable to every range quoted here.
  A full-256-frame scan would have cost 4x the 37 min of grid time for no change
  of geometry.
* The bench control uses all cells (1204568; one frame).
* Grids are 15x15: the ridge coordinate spans +-2 decades around its fitted
  value, the rate spans 0.1 .. 1000 s^-1 (the fitted rate is inserted if it falls
  outside). 1-D profiles are 21 points over +-2 decades.

Two caveats that belong next to the numbers. (i) The bench forward model fixes
its per-order lag-integration truncation from the PRIOR centre before the fit
(`model.bench_batch`, `bench_order_groups`), so the bench control scan carries
the same truncation the bench fit had — consistent, but it is a truncation.
(ii) A scan holds the 520-parameter profile block, the floor and the mic gains
at the fitted values; it is the objective's geometry in the dynamics block
alone, not a profile likelihood over all blocks.

## 3. Michael's FLY125 cruise, shaft term

`michaels_fly125_cruise__flight__grid_shaft.json`, first fit
`../fits/michaels_fly125_cruise__flight.json` (NOT converged, restart gain
0.01875590 nats/cell vs tol 1e-4, grad norm 8555, wall 6470 s, git
`43d7c5aaa29153f31872ce341bc3694c4fb419bb`). Fitted point: `sigma_nu = 8.378544`,
`lam = 199.7809`, hence `D = 0.3513850`.

The grid minimum sits at `D = 0.351385` (exactly the fitted `D`, grid point 8 of
15) and `lam = 268.27` (one grid step above the fitted 199.78), at
`-47.236544 nats/cell`.

**The ridge is the frozen-rate one.** Minimising over `D` at each `lam`, the
best `D` falls exactly as `1/lam` over lam 1.39 .. 138.9 — i.e. `sigma_nu =
sqrt(D lam)` stays constant at 6.987 rad/s — and then flattens
(`verdict.map.profiled_ridge_argmin`: 35.14, 35.14, 35.14, 35.14, 35.14, 18.20,
9.427, 4.882, 2.529, 1.310, 0.6784, 0.3514, 0.3514, 0.1820, 0.1820 against
lam 0.1 .. 1000). So the data identify `sigma_nu`, and `D` only at fixed `lam`.

**Flat along, curved across.** `verdict.map`:

| measure | value (nats/cell) |
| --- | --- |
| profiled over `D`, range over the whole lam axis (0.1 .. 1000) | 0.594464 |
| profiled over `D`, range over lam >= 1 | 0.0307066 |
| profiled row, lam 19.31 .. 1000 | 0.018564 (max) |
| profiled row, lam 138.9 .. 1000 | 0.014862 (max) |
| ACROSS the ridge: +-2 decades of `D` at lam = 268.27 | 1.11013 |
| 1-D `sigma_nu` profile range over +-2 decades | 1.99678, plateau at 1e-4: 0.00 decades |
| 1-D `lam` profile range over +-2 decades | 0.875334, plateau at 1e-4: 0.00 decades |

The profiled lam bands (`verdict.map.profiled_rate_band`) are: within 1e-4 — one
grid point (268.27); within 1e-2 — lam 71.97 .. 268.27; within 5e-2 — lam
1.389 .. 1000 (11 of 15 points, 2.86 decades).

**The decisive comparison.** The whole objective gain available along 1.7 decades
of `lam` (19.3 -> 1000) is **0.0186 nats/cell** — the same size as the
**0.0188 nats/cell** an L-BFGS restart still found, which is what failed the
convergence test. The `lam` prior cannot break the tie either: `LogNormal(log 5.5,
0.9)` over 516096 cells is `~2e-5 nats/cell per sd`, below the 1e-4 tolerance.
Neither the data nor the prior locate `lam`.

## 4. Michael's FLY125 cruise, odd per-order term

`michaels_fly125_cruise__flight__grid_order_odd.json`. Fitted
`A_odd = sigma_eps_odd^2 = 0.354395`, `lam_eps_odd = 75.6331`, product
`D_eps = 26.8040 rad^2/s per k`.

The fitted cell IS the grid's global minimum (`-47.239745 nats/cell`, grid point
(8, 11) of 15x15; the grid's argmin rate 71.97 is one step below the fitted
75.63 only because 75.63 is not a grid point). Curvature exists in both
directions of the chosen coordinates — 0.161583 along `lam_eps` at fixed `A`,
0.427175 across `A` at fixed `lam_eps`, and the 1-D profiles have zero 1e-4
plateau (ranges 1.293964 for `sigma_eps_odd`, 0.226193 for `lam_eps_odd`).

But the map's valley is the ANTI-diagonal `A lam_eps = const`, and profiling `A`
out leaves `lam_eps_odd` nearly as unconstrained as `lam`:

| measure | value (nats/cell) |
| --- | --- |
| profiled over `A`, range over the whole rate axis | 0.0617296 |
| profiled over `A`, range over `lam_eps` >= 1 | 0.0198915 |
| ACROSS: +-2 decades of `A` at `lam_eps` = 71.97 | 0.427175 |

Profiled bands: within 1e-4 — one grid point; within 1e-2 — `lam_eps_odd`
0.3728 .. 268.3 (2.86 decades); within 5e-2 — 0.1931 .. 1000.

So the identified per-order quantity is the **product `D_eps = A lam_eps =
26.80 rad^2/s per k`**, not the rate. This matters beyond the fit: the render's
odd-order coherent fraction is `exp(-A k)` (`lag.saturated_coherence`), which
depends on how the product is split, i.e. on the pin. That split is a labelled
R2 hypothesis, deliberately not tuned here.

## 5. Michael's FLY125 cruise, the six 1-D profiles

`michaels_fly125_cruise__flight__profiles.json`, 21 points over +-2 decades,
every other parameter held at the fitted value. "plateau" is the width in
decades of the connected region around the profile minimum that stays within the
fit's own convergence tolerance (1e-4 nats/cell).

| parameter | fitted | profile argmin | decades off | range (nats/cell) | plateau @1e-4 | identified? |
| --- | --- | --- | --- | --- | --- | --- |
| `sigma_nu` | 8.378544 | 8.378544 | +0.00 | 1.996778 | 0.00 | YES, sharply |
| `lam` | 199.7809 | 199.7809 | +0.00 | 0.875334 | 0.00 | at fixed `sigma_nu` only (see §3) |
| `sigma_eps_even` | 0.00161424 | 0.0642641 | +1.60 | 0.000233882 | 2.60 | NO |
| `sigma_eps_odd` | 0.595311 | 0.595311 | +0.00 | 1.293964 | 0.00 | YES (as `A`, see §4) |
| `lam_eps_even` | 0.0494033 | 0.782989 | +1.20 | 0.0000516729 | 4.00 (whole grid) | NO — range is BELOW the tolerance |
| `lam_eps_odd` | 75.6331 | 75.6331 | +0.00 | 0.226193 | 0.00 | at fixed `A` only (see §4) |

The even per-order term is effectively absent from the model at the fitted point:
`sigma_eps_even = 0.0016` gives a saturated coherence of 1.0000 at every order,
and the entire even block's leverage on the objective (5.2e-5 nats/cell for the
rate, 2.3e-4 for the amplitude) is at or below the convergence tolerance.

## 6. The converged bench control, `bench_dregon_Motor1_80`

`bench_dregon_Motor1_80__bench__{grid_shaft,grid_order_odd,profiles}.json`; the
fit (`../fits/bench_dregon_Motor1_80__bench.json`) IS converged (restart gain
1.662e-5 < 1e-4, wall 2829 s), at `sigma_nu = 0.836232`, `lam = 14.6539`,
`sigma_eps = (0.094676, 0.027426)`, `lam_eps = (107.4565, 0.174678)`.

| parameter | profile argmin | decades off | range (nats/cell) | plateau @1e-4 |
| --- | --- | --- | --- | --- |
| `sigma_nu` | 0.836232 | +0.00 | 2.177e+07 | 0.00 |
| `lam` | 14.6539 | +0.00 | 1.061e+07 | 0.00 |
| `sigma_eps_even` | 0.0946761 | +0.00 | 2.77554 | 0.00 |
| `sigma_eps_odd` | 0.0688908 | +0.40 | 0.101104 | 2.80 |
| `lam_eps_even` | 107.4565 | +0.00 | 2.02567 | 0.00 |
| `lam_eps_odd` | 1.10214 | +0.80 | 0.000187512 | 3.80 |

Four of the six sit in a genuine basin whose 1-D minimum is EXACTLY the fitted
value with no plateau at the tolerance; the odd per-order pair is flat here too
(its whole range, 1.9e-4 nats/cell, is barely above the tolerance) because on
this support the odd term is nearly switched off (`sigma_eps_odd = 0.027`).

The bench shaft grid still shows a ridge when profiled — `verdict.map`: profiled
range 0.0396165 over the whole lam axis, within-1e-2 band lam 0.1931 .. 71.97,
argmin lam 19.31 against a fitted 14.65 — with an across-ridge range of
1.235e+07 over +-2 decades of `D`. So ridge flatness in the rate is NOT what
separates a converged fit from a non-converged one; what separates them is that
here the restart gain (1.7e-5) is three orders of magnitude BELOW the depth
remaining along the ridge (0.040), while on Michael's the restart gain (0.0188)
EQUALS it (0.019-0.031). The bench fit had settled; the Michael's fit was still
crawling.

## 7. Verdict

* **Shaft term: a RIDGE, and the fitted point is ON it, not at a boundary.** The
  ridge is the frozen-rate one: `sigma_nu` is identified sharply (1.997
  nats/cell over +-2 decades, zero plateau), and `lam` is unidentified over
  2.86 decades (lam 1.39 .. 1000 within 0.031 nats/cell; 19.3 .. 1000 within
  0.0186). Across the ridge the objective moves 1.110 nats/cell. `log_box_hits`
  is 0 in the fit, so no coordinate was against its box: this is a ridge, not a
  boundary.
* **Odd per-order term: a RIDGE in `(A, lam_eps)` whose product is identified.**
  `D_eps = A lam_eps = 26.80` is pinned down (0.427 nats/cell across), the
  split is not (0.0199 nats/cell along, 2.86-decade 1e-2 band). The fitted point
  is the grid's minimum cell. Main's original hypothesis — a saturated term with
  `lam_eps_odd` unidentifiable because `1/lam_eps << window` — is not what the
  data show: the term is in its DIFFUSIVE limit for the orders that dominate the
  likelihood.
* **Even per-order term: unidentified in both coordinates**, with total leverage
  at or below the convergence tolerance. It is not a ridge, it is an absent term.
* **Identified by the data at all (Michael's cruise):** `sigma_nu`;
  `A_odd = sigma_eps_odd^2`; the product `A_odd lam_eps_odd`. **Not identified:**
  `lam`, `lam_eps_odd` separately from `A_odd`, `sigma_eps_even`, `lam_eps_even`.
  Of the six dynamics parameters the flight objective constrains two directions.
* **Consistency of the non-convergence.** The failed convergence test is
  consistent with the optimiser crawling along the shaft ridge: the total gain
  available along 1.7 decades of `lam` equals the restart gain that flagged it.
  This is consistency, not proof — the restart could also have been descending in
  the 520-parameter profile block, which these scans hold fixed.

## 8. The retry

Recipe, approved by Main with one change to the `lam` value (§8.1), run ONCE as
`nv2-r1-michaels-retry-a3bfe1` (`uni-cpu`, 8 cores, `--time 3h`, supports rebuilt
in-job, code at `2e96fa29`): R1Fit's recipe unchanged — `--set michaels-cruise`
(FLY125 only, 8 windows), frame stride 4, 256 pooled frames, 1500 Adam steps at
lr 0.02 with batch 8, 200 + 100 L-BFGS iterations on 64 frames, seed 0,
`--threads 8` — plus

    --pin lam=0.5 lam_eps_even=2.0 lam_eps_odd=75.63307088769216

`sigma_nu`, `sigma_eps_even`, `sigma_eps_odd`, the profile, the floor and the mic
gains stay free. **Pinning `lam` and fitting `sigma_nu` IS the ridge
reparameterisation**: at fixed `lam` the map `sigma_nu <-> D = sigma_nu^2/lam` is
a bijection, so no new parameterisation, no extra stage and no heuristic is
introduced. A pin is a CONSTANT of the model, not a tight prior: the guide
allocates no parameter for it and the log-prior counts nothing for it
(`model.sample_params`'s `pin`, `fit.py`'s `pinned_dynamics` record).

### 8.1 Why `lam = 0.5` and not the likelihood argmin

The likelihood's own best `lam` is 268.27 (§3). A 128 ms lag window cannot locate
`lam`, but the RENDER over seconds depends on it entirely, and the HPPNet gate
measures the render. The campaign's long-lag measurement (`82d0547e`) found the
`k = 2` shaft-phase variance still growing at 2 s lag (`q = 1.17`), i.e.
`tau_c >= 2 s`, `lam <= 0.5 s^-1`. Main's rule was: pin the physical value if it
costs less than 0.05 nats/cell against the argmin.

`michaels_fly125_cruise__flight__slices_sigma_nu_at_lam.json` profiles `sigma_nu`
(15 values over +-1 decade of 8.378544) at each candidate pin — necessary because
the 2-D grid's `D` axis is only +-2 decades wide, so at small `lam` its per-rate
minimum is an edge-limited upper bound:

| pinned `lam` | best `sigma_nu` | objective (nats/cell) | excess over argmin |
| --- | --- | --- | --- |
| 0.5 | 6.0299 | -47.196081 | 0.039377 |
| 5.5 (prior centre) | 6.0299 | -47.200306 | 0.035153 |
| 268.27 (likelihood argmin) | 8.3785 | -47.235458 | 0.000000 |

0.0394 < 0.05, so the physical pin is taken: **`lam = 0.5`, at a likelihood cost
of 0.0394 nats/cell** (394x the convergence tolerance, 3.5 % of the 1.110
nats/cell the objective moves across the ridge). `lam_eps_even = 2.0` is the
prior centre, chosen because the data express no preference whatsoever (5.2e-5
nats/cell over +-2 decades, below the tolerance). `lam_eps_odd = 75.63307` is the
grid argmin and equals the first fit's value; no long-lag evidence exists for the
odd orders to override it.

### 8.2 Retry outcome

`../fits/michaels_fly125_cruise__flight_retry.json` (code `2e96fa29`, job
`nv2-r1-michaels-retry-a3bfe1`, exit 0). **The retry ALSO fails the convergence
test** — recorded as such; no third attempt, per the stop rule.

| quantity | first fit | retry | delta |
| --- | --- | --- | --- |
| converged | false | **false** | — |
| L-BFGS restart gain (nats/cell, tol 1e-4) | 0.01875590 | 0.02172926 | +0.00297 |
| gradient norm at the reported point | 8555.25 | 2863.34 | /2.99 |
| wall (s) | 6470.07 (1.80 h) | 4799.24 (1.33 h) | -1670.8 |
| Adam / L-BFGS wall (s) | 2469.6 / 4000.5 | 1718.5 / 3080.8 | — |
| L-BFGS loss after restart, 64 frames / 516096 cells | -24378547.436 | **-24385986.196** | -7438.76, i.e. **0.014414 nats/cell BETTER** |
| full pooled objective `whittle_nats` / 2064384 cells | -11.8011177 | **-11.7880591** | **0.013059 nats/cell WORSE** |
| comb band / floor band (nats) | -23684606.8 / -677431.8 | -23659189.5 / -675891.2 | — |
| `sigma_nu` | 8.378544 | **5.985662** | -0.146 decades |
| `lam` | 199.7809 | **0.5 (pinned)** | — |
| `sigma_eps_even` | 0.00161424 | 0.00134537 | -0.079 decades |
| `sigma_eps_odd` | 0.595311 | **0.683521** | +0.060 decades |
| `lam_eps_even` | 0.0494033 | **2.0 (pinned)** | — |
| `lam_eps_odd` | 75.6331 | **75.6331 (pinned)** | — |
| derived `D = sigma_nu^2/lam` | 0.351385 | 71.6563 | (D is not the identified coordinate) |
| derived `D_eps,odd = A lam_eps` | 26.8040 | 35.3358 | +0.12 decades |
| `amp_exp` (speed-law exponent) | 10.019606 | 16.429708 | +6.41 |
| floor level (dB) | -32.791079 | -33.054688 | -0.264 |
| `log_box_hits` | 0 | 0 | no coordinate against its box |

Four things this says.

1. **The ridge prediction checks out quantitatively.** The `sigma_nu` slice at
   `lam = 0.5` (§8.1), computed with every other block HELD at the first fit's
   values, put the best `sigma_nu` at 6.0299. The retry, refitting the profile,
   floor and mic blocks freely, landed at 5.985662 — 0.73 % away. The shaft
   ridge is real and its coordinate is `sigma_nu`, exactly as derived.
2. **The pin's likelihood cost was recovered and then some, on the polish set.**
   The pin costs 0.039377 nats/cell at frozen nuisances; the retry ends 0.014414
   nats/cell BELOW the first fit on the same 64-frame, 516096-cell objective the
   convergence test uses. Since the first fit was itself unconverged, that
   difference measures optimiser progress as much as the pin.
3. **But on the full pooled objective the retry is 0.013059 nats/cell worse**,
   so what the polish gained on its 64 frames does not transfer to all 256. That
   is a property of the 64-frame polish, not of the pin, and it is the one number
   in this table that should worry R2.
4. **Removing every flat dynamics direction did NOT buy convergence.** With three
   of six dynamics coordinates pinned, the restart still found 0.0217 nats/cell —
   slightly MORE than before (0.0188) — while the gradient norm fell threefold.
   So the shaft ridge was not the only thing the optimiser was crawling along:
   the remaining non-convergence lives outside the dynamics block, in the
   520-parameter profile block, the floor, or the interaction between them and
   `amp_exp` (which moved from 10.02 to 16.43 — a large nuisance drift for a
   speed-law exponent). §7's last bullet stated this as the alternative
   explanation; the retry converts it from alternative to fact: the dynamics
   ridge is necessary but not sufficient to explain the non-convergence.

Consequences to carry into R2, not tuned here: the reported flight dynamics are
`sigma_nu` (identified), the pinned `lam` (a modelling choice, justified by the
long-lag evidence, costing 0.039 nats/cell of likelihood), and a per-order
product `D_eps,odd = 35.34` whose split into `A` and `lam_eps_odd` is set by the
pin. The render's odd-order coherent fraction `exp(-A k)` therefore also depends
on the pin (saturated coherence at k=1 moved 0.7016 -> 0.6268), which is a
labelled hypothesis for R2, not a measurement.

## 9. What this does not show

* The scans hold the profile, floor and mic blocks at the first fit's values.
  They are the geometry of the dynamics block, not a full profile likelihood.
* Michael's numbers are on the 64-frame polish subset (516096 of 2064384 cells).
  The subset is the one the convergence test itself uses, but it is a subset.
* Grid resolution is 0.2857 decades per step (15 points over 4 decades); the
  quoted argmins are grid points, not optimised minima, and the 1e-4 bands are
  reported as "one grid point" wherever the grid cannot resolve them.
