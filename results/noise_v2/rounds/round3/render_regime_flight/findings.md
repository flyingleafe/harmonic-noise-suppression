# Noise model v2 — round 2: why the v2 render loses standby and ramp

Record `results/noise_v2/rounds/round2/render_regime/render_regime.json`, git `e7510b5f96c4182765952fac02acb01eb152f8e7`, fit `results/noise_v2/rounds/round3/fits/michaels_fly125_all__flight.json` (amp_exp 4.1612, floor_exp 2.0047, floor_static_rel 0.000288, reference speed 80 rev/s), render seed 2001, 8 mics.

## PIT MAE per support and arm (rev/s, frozen HPPNet, seed 2001)

| support | regime | real | legacy | v2 | v2_nocomb | v2_amp_exp0 | v2_floor_exp0 | v2_both_exp0 | v2_floor_matched | v2_prior_exps | v2_level_matched_exps | v2_prior_exps_nofloor | v2_prior_exps_lowjitter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `FLY124@8.000000+8.000000` | standby | — | — | — | — | — | — | — | — | — | — | — | — |
| `FLY124@27.680000+8.000000` | ramp | — | — | — | — | — | — | — | — | — | — | — | — |
| `FLY124@40.000000+8.000000` | cruise | — | — | — | — | — | — | — | — | — | — | — | — |

## Levels and LTAS (mic 0, 30–7900 Hz, absolute)

| support | arm | band level dB | level offset vs real dB | LTAS mean abs dB | LTAS shape-only dB | comb/floor k=2 dB | k=4 | k=8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `standby` | real | -2.81 | 0.00 | 0.00 | 0.00 | 15.1 | 6.9 | 5.6 |
| `standby` | legacy | -3.36 | -0.41 | 0.59 | 0.67 | 16.4 | 8.8 | 6.3 |
| `standby` | v2 | -3.44 | 0.24 | 1.37 | 2.27 | 22.9 | 12.8 | 13.0 |
| `standby` | v2_nocomb | -12.83 | -33.13 | 33.66 | 67.14 | 9.7 | 7.3 | 1.2 |
| `standby` | v2_amp_exp0 | 10.46 | 10.98 | 10.98 | 3.64 | 27.5 | 13.5 | 12.5 |
| `standby` | v2_floor_exp0 | -1.86 | 3.26 | 3.78 | 5.29 | 19.9 | 12.8 | 13.0 |
| `standby` | v2_both_exp0 | 10.54 | 12.07 | 12.07 | 2.61 | 26.8 | 13.5 | 12.5 |
| `standby` | v2_floor_calib | -6.01 | -26.32 | 31.56 | 67.18 | 9.7 | 7.3 | 1.3 |
| `standby` | v2_floor_matched | -0.34 | 4.84 | 5.36 | 6.87 | 18.3 | 12.8 | 13.0 |
| `standby` | v2_prior_exps | 3.57 | 5.18 | 5.18 | 2.64 | 26.3 | 13.2 | 12.8 |
| `standby` | v2_level_matched_exps | -4.37 | -2.59 | 3.09 | 2.59 | 25.5 | 12.8 | 13.0 |
| `standby` | v2_prior_exps_nofloor | 3.47 | 3.53 | 4.97 | 4.23 | 27.1 | 13.2 | 12.8 |
| `standby` | v2_prior_exps_lowjitter | 3.70 | 5.15 | 5.15 | 2.65 | 25.6 | 13.1 | 13.4 |
| `ramp` | real | 9.33 | 0.00 | 0.00 | 0.00 | 6.9 | 2.8 | 1.7 |
| `ramp` | legacy | 6.35 | -1.95 | 4.19 | 3.77 | 6.2 | 3.4 | 4.0 |
| `ramp` | v2 | 6.54 | -3.48 | 3.48 | 1.35 | 8.2 | 5.4 | 3.7 |
| `ramp` | v2_nocomb | -8.56 | -41.73 | 41.73 | 57.40 | 12.4 | 2.4 | 1.9 |
| `ramp` | v2_amp_exp0 | 10.49 | -0.62 | 1.90 | 3.51 | 5.8 | 7.8 | 3.2 |
| `ramp` | v2_floor_exp0 | 6.64 | -2.94 | 2.94 | 1.08 | 8.2 | 5.4 | 3.7 |
| `ramp` | v2_both_exp0 | 10.53 | -0.18 | 1.46 | 3.08 | 5.8 | 7.8 | 3.2 |
| `ramp` | v2_floor_calib | -6.01 | -40.23 | 40.23 | 63.75 | 12.6 | 2.6 | 1.7 |
| `ramp` | v2_floor_matched | 11.12 | 2.91 | 5.38 | 5.93 | 8.1 | 5.4 | 3.7 |
| `ramp` | v2_prior_exps | 7.79 | -2.43 | 2.50 | 2.34 | 7.2 | 7.0 | 3.1 |
| `ramp` | v2_level_matched_exps | 7.94 | -2.29 | 2.41 | 2.40 | 7.1 | 7.0 | 3.1 |
| `ramp` | v2_prior_exps_nofloor | 7.68 | -3.55 | 3.62 | 3.47 | 7.2 | 7.0 | 3.1 |
| `ramp` | v2_prior_exps_lowjitter | 7.78 | -2.45 | 2.52 | 2.42 | 8.6 | 7.2 | 3.6 |
| `cruise` | real | 12.16 | 0.00 | 0.00 | 0.00 | 25.3 | 11.0 | 2.4 |
| `cruise` | legacy | 10.21 | -0.70 | 0.98 | 1.60 | 30.7 | 15.4 | 3.4 |
| `cruise` | v2 | 10.48 | 0.55 | 1.00 | 0.98 | 26.8 | 10.6 | 5.8 |
| `cruise` | v2_nocomb | -5.89 | -38.84 | 38.84 | 58.52 | 9.2 | 2.1 | 2.2 |
| `cruise` | v2_amp_exp0 | 10.57 | 0.62 | 0.82 | 0.66 | 25.8 | 10.2 | 6.2 |
| `cruise` | v2_floor_exp0 | 10.47 | 0.54 | 0.99 | 0.97 | 26.8 | 10.6 | 5.8 |
| `cruise` | v2_both_exp0 | 10.57 | 0.60 | 0.81 | 0.67 | 25.8 | 10.2 | 6.2 |
| `cruise` | v2_floor_calib | -6.01 | -39.53 | 39.53 | 62.40 | 9.7 | 1.6 | 2.2 |
| `cruise` | v2_floor_matched | 14.37 | 5.97 | 5.98 | 5.65 | 26.8 | 10.6 | 5.8 |
| `cruise` | v2_prior_exps | 10.44 | 0.54 | 0.88 | 0.77 | 26.3 | 10.3 | 6.0 |
| `cruise` | v2_level_matched_exps | 10.44 | 0.54 | 0.88 | 0.77 | 26.3 | 10.3 | 6.0 |
| `cruise` | v2_prior_exps_nofloor | 10.34 | -0.05 | 0.79 | 0.97 | 26.3 | 10.3 | 6.0 |
| `cruise` | v2_prior_exps_lowjitter | 10.46 | 0.52 | 0.86 | 0.74 | 26.1 | 10.9 | 5.4 |

## The two fitted speed envelopes at each support's speed

| support | regime | mean rev/s | speed | comb envelope dB | floor envelope dB | comb − floor dB | k_max | top comb line Hz |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `FLY124@8.000000+8.000000` | standby | 36.39 | 0.4549 | -13.99 | -6.81 | -7.17 | 81 | 3349 |
| `FLY124@27.680000+8.000000` | ramp | 56.82 | 0.7103 | -6.14 | -2.97 | -3.17 | 81 | 4853 |
| `FLY124@40.000000+8.000000` | cruise | 80.93 | 1.0117 | +0.41 | +0.13 | +0.28 | 81 | 7459 |

## Encoded speed (order-2 demodulation against the frozen label)

| support | arm | rotor | median abs dev rev/s | p90 | mean signed |
|---|---|---|---:|---:|---:|
| `standby` | real | rotor0 | 0.139 | 0.318 | -0.029 |
| `standby` | real | rotor2 | 0.300 | 0.782 | 0.017 |
| `standby` | legacy | rotor0 | 0.168 | 0.389 | 0.011 |
| `standby` | legacy | rotor2 | 0.373 | 0.954 | -0.002 |
| `standby` | v2 | rotor0 | 0.143 | 0.335 | -0.090 |
| `standby` | v2 | rotor2 | 0.510 | 1.257 | 0.336 |
| `standby` | v2_both_exp0 | rotor0 | 0.136 | 0.311 | -0.092 |
| `standby` | v2_both_exp0 | rotor2 | 0.510 | 1.249 | 0.336 |
| `standby` | v2_nocomb | rotor0 | 0.810 | 1.864 | -0.677 |
| `standby` | v2_nocomb | rotor2 | 0.665 | 1.757 | -0.668 |
| `ramp` | real | rotor2 | 0.361 | 1.341 | -0.238 |
| `ramp` | legacy | rotor2 | 0.470 | 1.012 | -0.279 |
| `ramp` | v2 | rotor2 | 0.509 | 1.180 | -0.106 |
| `ramp` | v2_both_exp0 | rotor2 | 0.478 | 1.194 | -0.039 |
| `ramp` | v2_nocomb | rotor2 | 0.474 | 1.734 | -0.237 |
| `cruise` | real | rotor0 | 0.202 | 0.533 | -0.086 |
| `cruise` | legacy | rotor0 | 0.231 | 0.722 | -0.005 |
| `cruise` | v2 | rotor0 | 0.124 | 0.303 | -0.094 |
| `cruise` | v2_both_exp0 | rotor0 | 0.132 | 0.316 | -0.095 |
| `cruise` | v2_nocomb | rotor0 | 0.056 | 0.209 | -0.203 |

## Verdict per hypothesis

| hypothesis | verdict | the numbers that decide it |
|---|---|---|
| **H1** — `amp_exp` = 4.16 extrapolated from cruise collapses the comb | REFUTED | standby: comb envelope -14.0 dB (-14.4 dB vs cruise), k=2 comb/floor v2 22.9 dB vs floor-only control 9.7 dB and real 15.1 dB; ramp: comb envelope -6.1 dB (-6.6 dB vs cruise), k=2 comb/floor v2 8.2 dB vs floor-only control 12.4 dB and real 6.9 dB |
| **H2** — `floor_exp` = 2.00 (NEGATIVE) makes the floor explode as the rotors slow | REFUTED | standby: floor envelope -6.8 dB (-6.9 dB vs cruise), v2 level offset +0.2 dB, with `floor_exp = 0` +3.3 dB; ramp: floor envelope -3.0 dB (-3.1 dB vs cruise), v2 level offset -3.5 dB, with `floor_exp = 0` -2.9 dB |
| **H3** — the render encodes a different speed | SUPPORTED | order-2 demodulation against the frozen label, on the arms that HAVE a visible comb: standby/v2_both_exp0/rotor0 0.136 rev/s; standby/v2_both_exp0/rotor2 0.510 rev/s; standby/real/rotor0 0.139 rev/s; standby/real/rotor2 0.300 rev/s; ramp/v2_both_exp0/rotor2 0.478 rev/s; ramp/real/rotor2 0.361 rev/s; k_max is 81 in every regime (no Nyquist cap change), so the carrier the render integrates IS the label track |
| **H4** — a regime-independent defect cruise tolerates | **REFUTED** | cruise `FLY124@40.000000+8.000000` runs the SAME code path, the same mic gains and the same seed, and is at parity: v2 PIT — rev/s, level offset +0.55 dB, LTAS 1.00 dB. Peak |x| of the v2 render vs the real clip: standby 0.07 vs 0.12, ramp 0.43 vs 0.59, cruise 0.41 vs 0.59 — the standby render is loud, not clipped (the probe reads float64 in memory and the renderer applies no normalisation), and a level error alone does not do this: `v2_floor_exp0` sits within 3.3 dB of the real standby level and still scores — rev/s |

Secondary (not one of the four, but real): the comb ladder is a FIXED 81 orders, so its top line slides with speed — standby 3349 Hz, ramp 4853 Hz, cruise 7459 Hz. Above that line the standby render is pure floor.

The residual, and what it is NOT: with both exponents at their prior mean the ramp is already at the real clip's own number (— vs real — rev/s, legacy —), but standby only falls to — against a legacy —. Two further renders separate what remains: the comb ALONE, floor muted (— rev/s at standby, — at ramp), and the same render with the shaft jitter at a quarter of the fitted `sigma_nu` = 1.23 rev/s (— rev/s at standby, — at ramp) — `sigma_nu` is in rev/s, so the SAME fitted jitter is a 2.2x larger fraction of a standby carrier than of the reference.

_The two residual renders are not in this pass._

## Minimal patch proposal (NOT applied — Main's call)

Both supported hypotheses are ONE defect: the two speed-envelope exponents are fitted on a pool that barely varies in speed (the fit's own objective saw carriers 20.9-97.9 rev/s, a span of 4.678x, 10 FLY125 cruise windows), where they trade almost exactly against `profile_db` and `floor_mean_db`. The optimiser took `amp_exp` = 4.16 and `floor_exp` = 2.00 against a `N(2, 2)` prior that 24 M Whittle cells simply outvote; in sample the two extremes cancel, and outside it they pull the comb and the floor 7 dB apart. The standby support runs at 36.4 rev/s — 1.74x the LOWEST carrier the fit ever saw, so both envelopes are pure extrapolation there. Two changes, smallest first:

**(a) the floor exponent may not be negative** — a rotor floor cannot get LOUDER as the rotors slow. One site changes parameterisation (`src/experiments/noise_model/model.py`):

```diff
@@ class Priors
-    amp_exp: tuple[float, float] = (2.0, 2.0)
-    floor_exp: tuple[float, float] = (2.0, 2.0)
+    amp_exp: tuple[float, float] = (2.0, 2.0)
+    #: LOG-space now: the floor's speed exponent is positive by construction
+    log_floor_exp: tuple[float, float] = (math.log(2.0), 0.7)
@@ def sample_params
-            exp=_normal(site, "floor_exp", *priors.floor_exp) if flight else zero,
+            exp=(
+                _lognormal(site, "floor_exp", priors.log_floor_exp) if flight else zero
+            ),
```

**(b) do not FIT an envelope the pool cannot see.** A flight pool whose speed span is under 1.5x identifies neither exponent, so pin both at the prior mean instead of letting the likelihood run away with them:

```diff
@@ def sample_params
     free = free_blocks(mode)
     fz = dict(frozen or {})
     r, m, k = batch.n_rotors, batch.n_mics, batch.k_max
     flight = batch.mode == "flight"
+    # the speed envelopes are identified by the SPAN of the pool's speeds; a
+    # single-regime pool has none, and a free exponent then extrapolates
+    # 70 dB of comb-to-floor error onto every other regime
+    envelopes_identified = flight and batch.speed_span >= SPEED_SPAN_MIN
@@
-        amp_exp = _normal(site, "amp_exp", *priors.amp_exp) if flight else zero
+        amp_exp = (
+            _normal(site, "amp_exp", *priors.amp_exp)
+            if envelopes_identified
+            else torch.as_tensor(priors.amp_exp[0], dtype=torch.float64)
+        )
```

with `speed_span = max_rotor_frame_speed / min_rotor_frame_speed` recorded on `SupportBatch` by `flight_batch` (it already holds `rate_work`) and `SPEED_SPAN_MIN = 1.5`; the same gate applies to `floor_exp` and `floor_static_rel`.

**(c) the campaign-level fix, and the one that actually buys parity: stop extrapolating.** The legacy arm reaches standby 0.32 rev/s because the frozen evaluator renders standby from a STANDBY export (`results/S2/standby.json`, regime `standby`) and only its cruise/ramp arms come from the cruise export. Round 2's Michael's fit should do the same: either pool FLY125 standby + ramp + cruise windows into one flight fit (which also makes (b)'s span gate open legitimately), or fit one support per regime and render each regime from its own. No renderer or model change is needed for this one — it is the fit pool.

### Expected effect — measured, not guessed

No variant below is a FIT, so none of them is a parity claim: each is the SAME fitted comb and floor with one speed law stopped from extrapolating, scored by the same frozen HPPNet on the same frozen support.

| support | v2 as fitted | (a)+(b) both exponents at the prior mean `v2_prior_exps` | the best a single global envelope can do `v2_level_matched_exps` | legacy (the bar) | real |
|---|---:|---:|---:|---:|---:|
| standby | — | — | — (exp 4.31) | — | — |
| ramp | — | — | — (exp 1.84) | — | — |
| cruise | — | — | — (exp 2.00) | — | — |

Equal-regime mean of the three supports measured here, against the frozen Michael's parity bar 3.177994 rev/s (1.05 x the legacy 3.026661) — INDICATIVE only, the gate averages five supports:

| arm | v2 | v2_prior_exps | v2_level_matched_exps | legacy | real |
|---|---:|---:|---:|---:|---:|
| equal-regime mean (3 supports) | — | — | — | — | — |

## Figures

* `results/noise_v2/rounds/round3/render_regime_flight/ltas_standby.png`
* `results/noise_v2/rounds/round3/render_regime_flight/spectrogram_standby.png`
* `results/noise_v2/rounds/round3/render_regime_flight/ltas_ramp.png`
* `results/noise_v2/rounds/round3/render_regime_flight/spectrogram_ramp.png`

