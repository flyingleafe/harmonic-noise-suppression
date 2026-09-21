# R5 DREGON fit, mode `flight_profile` — what the four restarts bought

Every number below is read from the committed payloads: the reduced fit
`dregon_room2_floor__flight_profile.json` and its four restarts under
`restarts/` (commit `43046ced`, code `1c6ac337`), R3's
`round3/fits/dregon_room2_floor__flight_floor_lowk.json` and the four
`round3/fits/bench_dregon_Motor{1..4}_70__bench.json` the dynamics and the
comparandum profile come from. The convergence label stands beside each one.

## The objective: better than R3 on the same cells, and NOT converged

| restart | Whittle nats/cell | polished loss | `converged` | `which_converged` | L-BFGS restart gain/cell (tol 1e-4) | grad norm | wall |
|---|---:|---:|---|---|---:|---:|---:|
| s0 | -8.712373 | -17931477.285089 | **false** | none | 2.120e-03 | 407.1 | 82 min |
| s1 | -8.712293 | -17931387.728376 | **false** | none | 2.514e-03 | 893.6 | 115 min |
| s2 | -8.712558 | -17931090.675246 | **false** | none | 2.341e-03 | 651.6 | 55 min |
| s3 | -8.712450 | -17931255.375777 | **false** | none | 1.679e-03 | 287.4 | 58 min |

The reported fit is **s0** — the restart with the lowest POLISHED
objective (-17931477.285089), which is the reduce rule
(`restarts.rule`), not the lowest Whittle term: s2's Whittle is lower
(-8.712558 nats/cell) and its prior term is worse.

* **-8.712373 nats/cell** over 2,056,320 cells, split
  -3.22246 (floor band, < 300 Hz, 71,400 cells) and
  -8.90985 (comb band, 1,984,920 cells).
* Restart spread: best − median 7.573e-05,
  best − worst 1.880e-04 nats/cell on the polished loss;
  the four Whittle terms themselves span 2.643e-04 nats/cell. The restarts
  agree to four decimal places and none of them is converged.
* **NOT converged**, all four: `converged=false`, `which_converged="none"`. The
  L-BFGS restart still bought 1.7–2.5e-3 nats/cell against the 1e-4 tolerance,
  i.e. the stopping test failed by more than an order of magnitude, and the
  gradient norm at the stop is 287–894. Every number in this file is a
  number from an unconverged optimum whose restarts nevertheless coincide.

### Against R3 `flight_floor_lowk`, the SAME pool and the SAME cells

| | R5 `flight_profile` | R3 `flight_floor_lowk` | delta |
|---|---:|---:|---:|
| Whittle nats/cell (2,056,320 cells) | **-8.712373** | -8.593117 | **-0.119256** |
| floor band (< 300 Hz) | -3.22246 | -2.45245 | -0.77001 |
| comb band | -8.90985 | -8.81400 | -0.09585 |
| convergence | not converged | not converged | — |

Freeing the per-order profile and freezing the dynamics buys
**0.119256 nats/cell** against freezing the profile and freeing the floor
scalars — the same model, the same five 8 s segments, the same 2,056,320
cells, so the difference is a MODE difference alone, and it is an improvement
in both bands. (The submit note's comparandum, −8.59398, is an arithmetic
slip: `-17670197.599347502 / 2056320` is **-8.593117**, which is what
this section quotes.)

## The comb: free per-order profile against the bench transplant

`profile_db` is `(4, 88)` dB, one ladder per rotor. The comparandum is each
rotor's OWN bench fit — rotor `r` against `Motor{r+1}` — because that is the
rotor-matched rule the frozen dynamics were transplanted by.

| k | r0 fit / bench / Δ | r1 fit / bench / Δ | r2 fit / bench / Δ | r3 fit / bench / Δ |
|---:|---|---|---|---|
| 1 | -52.97 / -58.45 / **+5.48** | -53.34 / -48.75 / **-4.58** | -45.37 / -51.13 / **+5.76** | -37.14 / -53.60 / **+16.46** |
| 2 | -47.03 / -49.36 / **+2.34** | -44.27 / -49.47 / **+5.20** | -42.40 / -46.73 / **+4.33** | -42.87 / -47.71 / **+4.84** |
| 3 | -72.00 / -62.27 / **-9.73** | -76.03 / -54.65 / **-21.38** | -74.46 / -73.82 / **-0.64** | -69.08 / -71.50 / **+2.42** |
| 4 | -60.01 / -58.06 / **-1.95** | -68.76 / -60.70 / **-8.06** | -56.93 / -57.35 / **+0.41** | -54.88 / -56.14 / **+1.26** |
| 5 | -68.80 / -71.56 / **+2.75** | -72.09 / -59.69 / **-12.40** | -63.72 / -77.98 / **+14.26** | -58.97 / -68.37 / **+9.39** |
| 6 | -66.67 / -58.22 / **-8.45** | -71.32 / -59.38 / **-11.94** | -60.10 / -58.72 / **-1.38** | -57.50 / -58.75 / **+1.25** |
| 7 | -56.69 / -76.34 / **+19.65** | -71.46 / -57.69 / **-13.77** | -65.12 / -79.26 / **+14.14** | -62.30 / -68.98 / **+6.68** |
| 8 | -65.73 / -61.72 / **-4.01** | -59.03 / -57.56 / **-1.47** | -60.23 / -59.02 / **-1.21** | -52.04 / -63.64 / **+11.60** |
| 9 | -54.00 / -67.62 / **+13.62** | -55.39 / -57.93 / **+2.54** | -54.18 / -65.98 / **+11.80** | -49.74 / -68.69 / **+18.95** |
| 10 | -66.40 / -64.48 / **-1.92** | -55.20 / -63.79 / **+8.59** | -55.93 / -65.48 / **+9.55** | -53.47 / -64.53 / **+11.06** |
| 11 | -77.15 / -66.09 / **-11.06** | -57.85 / -59.60 / **+1.75** | -72.48 / -56.70 / **-15.78** | -58.43 / -52.42 / **-6.02** |
| 12 | -59.67 / -61.18 / **+1.52** | -76.37 / -61.59 / **-14.79** | -68.41 / -60.47 / **-7.94** | -60.02 / -56.67 / **-3.35** |
| 13 | -62.64 / -62.34 / **-0.30** | -62.63 / -69.25 / **+6.62** | -62.59 / -47.04 / **-15.55** | -53.57 / -55.01 / **+1.44** |
| 14 | -60.04 / -63.96 / **+3.91** | -62.73 / -63.93 / **+1.20** | -57.21 / -56.07 / **-1.14** | -55.15 / -60.26 / **+5.11** |
| 15 | -59.42 / -72.09 / **+12.67** | -59.35 / -70.99 / **+11.64** | -64.01 / -64.82 / **+0.81** | -58.27 / -65.21 / **+6.93** |
| 16 | -61.48 / -63.23 / **+1.75** | -66.88 / -62.77 / **-4.11** | -57.16 / -63.55 / **+6.39** | -54.00 / -64.11 / **+10.11** |

* Mean over rotors and k ≤ 16: **+1.457 dB** with a 8.922 dB spread,
  range -21.38 … +19.65 dB. Over all 88 orders: +2.072 ± 8.441 dB.
  Per rotor, mean over k ≤ 16: r0 +1.642, r1 -3.435, r2 +1.487, r3 +6.134 dB.
* The free profile therefore sits **+2.07 dB above the bench ladder on average**
  and reorders it order by order: the deviations are ±10–20 dB at individual
  orders (worst -21.38 dB at r1 k=3, +19.65 dB at r0 k=7) with no
  common shape across the four rotors. The flight recording does not resolve
  per-order lines above k ≈ 8 (R3's registration study), so most of this
  ladder is the prior's `N(mu_F - 15, 8)` arm following the floor, not a
  measured line.
* At the orders the trackers read (k ≤ 8) the mean deviation is +0.852 dB
  with a 9.153 dB spread — per rotor r0 +0.76, r1 -8.55, r2 +4.46, r3 +6.74 dB.
* Restart-to-restart spread of `profile_db` (max − min over the four seeds):
  1.717 dB mean at k ≤ 8 (max 8.260 dB),
  0.894 dB mean over all orders (max 12.507 dB). The low orders
  are the least reproducible part of the comb, which is the part the trackers
  read.

## The rest of the free block

| parameter | R5 `flight_profile` | restarts s0..s3 | R3 `flight_floor_lowk` |
|---|---:|---|---:|
| `profile.amp_exp` | **-0.868132** | -0.8681, -0.9813, -0.9482, -0.7416 | 0.000000 (pinned) |
| `floor.floor_mean_db` | **-37.359981** | -37.360, -37.286, -37.366, -37.318 | -38.272988 |
| `floor.floor_tilt_db_oct` | **-5.552232** | -5.552, -5.622, -5.487, -5.281 | -7.201448 |
| `floor.floor_exp` | **15.321160** | 15.321, 15.290, 15.483, 15.367 | 22.816840 |
| `floor.floor_static_rel` | **2.580e-07** | 2.58e-07, 2.45e-09, 7.32e-05, 1.99e-08 | 3.807542 |
| `diagnostics.floor_level_db` | **-36.224495** | — | -36.692768 |

* **`span_pins`: nothing is pinned.** The pool's carrier span is
  1.743918 against the 1.5 threshold, so `pinned` is `[]` and `amp_exp`,
  `floor_exp` and `floor_static_rel` are all FREE here — exactly what the
  submit note said to verify after the harvest.
* `amp_exp` lands at -0.868 (restart range -0.981 … -0.742): the comb gets
  QUIETER with speed, where R3's floor-only fit had it pinned at 0 and R4's
  fit-to-legacy `flight` fit put it at +3.356.
* `floor_static_rel` collapses to 2.6e-07 (three of four restarts return exactly
  0.0): the flight floor is carried by the speed-dependent term alone, against
  R3's 3.808. `floor_exp` drops from 22.8 to 15.3 and the tilt flattens
  from -7.20 to -5.55 dB/oct.
* The microphone block: `mic_line_gain_db` is `(8, 4)`, -9.10 … +5.91 dB;
  `mic_gains_db` -2.92 … +6.18 dB.

## The frozen half is bit-frozen

`sigma_nu` = 0.7423180851884967 and `lam` = 0.06080034059031183 are IDENTICAL across
all four restarts (`restarts.params.*.max_over_min` = 1.0), as is the `gamma_hz`
log-mean, which is what a frozen dynamics block must do and the mode's own
test asserts. They are the log-means over the four
`bench_dregon_Motor1_70, bench_dregon_Motor2_70, bench_dregon_Motor3_70, bench_dregon_Motor4_70` R3 fits, with the per-line widths
rotor-matched (True) and the fits reaching 118 orders
([117, 118, 116, 115]), so the pooled fallback above this pool's k_max = 88
never fires.

The low-order width check passes (`pass`): the largest `gamma_rk`/(1/2T) at
k ≤ 4 is 0.0126, i.e. the frozen bench lines are far narrower than the
3.90625 Hz the 2048-point flight window can resolve. That is the property
this mode was built to keep and the reason the pin below is about LEVEL and
not about width.

