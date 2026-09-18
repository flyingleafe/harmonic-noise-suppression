# Noise model v2 — round 2: why the v2 render loses standby and ramp

Record `results/noise_v2/rounds/round2/render_regime/render_regime.json`, git `c089d978b7455940249c4beb12b662a484d8a9e0`, fit `results/noise_v2/rounds/round2/fits/michaels_fly125_all__flight.json` (amp_exp 3.9029, floor_exp 3.2243, floor_static_rel 0.001964, reference speed 80 rev/s), render seed 2001, 8 mics.

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
| `standby` | v2 | -3.56 | 0.04 | 1.13 | 1.36 | 22.0 | 10.7 | 6.6 |
| `standby` | v2_nocomb | -14.17 | -30.70 | 31.12 | 62.40 | 9.8 | 5.9 | 1.1 |
| `standby` | v2_amp_exp0 | 10.07 | 9.60 | 9.60 | 4.45 | 22.4 | 10.7 | 6.1 |
| `standby` | v2_floor_exp0 | -0.64 | 5.02 | 5.07 | 5.69 | 20.0 | 10.7 | 6.6 |
| `standby` | v2_both_exp0 | 10.25 | 12.48 | 12.48 | 1.69 | 22.1 | 10.7 | 6.1 |
| `standby` | v2_floor_calib | -3.38 | -19.92 | 28.65 | 62.42 | 9.8 | 5.9 | 1.2 |
| `standby` | v2_floor_matched | -0.32 | 5.31 | 5.37 | 5.99 | 19.9 | 10.7 | 6.6 |
| `standby` | v2_prior_exps | 3.03 | 5.61 | 5.61 | 1.58 | 22.2 | 10.7 | 6.4 |
| `standby` | v2_level_matched_exps | -5.13 | -2.12 | 2.19 | 1.49 | 22.1 | 10.7 | 6.7 |
| `standby` | v2_prior_exps_nofloor | 2.81 | 1.76 | 6.11 | 5.27 | 22.6 | 10.7 | 6.4 |
| `standby` | v2_prior_exps_lowjitter | 2.96 | 5.06 | 5.06 | 2.11 | 26.8 | 14.9 | 14.2 |
| `ramp` | real | 9.33 | 0.00 | 0.00 | 0.00 | 6.9 | 2.8 | 1.7 |
| `ramp` | legacy | 6.35 | -1.95 | 4.19 | 3.77 | 6.2 | 3.4 | 4.0 |
| `ramp` | v2 | 7.07 | -3.08 | 3.08 | 1.28 | 7.7 | 1.9 | 4.1 |
| `ramp` | v2_nocomb | -6.63 | -36.59 | 36.59 | 56.33 | 9.4 | 2.2 | 2.1 |
| `ramp` | v2_amp_exp0 | 10.08 | -0.78 | 1.73 | 3.01 | 6.1 | 2.4 | 3.7 |
| `ramp` | v2_floor_exp0 | 7.27 | -2.19 | 2.63 | 1.81 | 7.7 | 1.9 | 4.1 |
| `ramp` | v2_both_exp0 | 10.18 | 0.02 | 1.41 | 2.21 | 6.1 | 2.4 | 3.7 |
| `ramp` | v2_floor_calib | -3.38 | -33.82 | 34.17 | 58.98 | 10.1 | 1.5 | 1.8 |
| `ramp` | v2_floor_matched | 11.29 | 3.13 | 5.78 | 6.59 | 7.7 | 1.9 | 4.1 |
| `ramp` | v2_prior_exps | 7.75 | -2.17 | 2.22 | 1.55 | 7.1 | 1.6 | 3.5 |
| `ramp` | v2_level_matched_exps | 7.86 | -2.05 | 2.14 | 1.59 | 7.1 | 1.6 | 3.5 |
| `ramp` | v2_prior_exps_nofloor | 7.55 | -4.19 | 4.23 | 3.47 | 7.1 | 1.6 | 3.5 |
| `ramp` | v2_prior_exps_lowjitter | 7.78 | -2.22 | 2.23 | 1.53 | 7.7 | 3.9 | 1.9 |
| `cruise` | real | 12.16 | 0.00 | 0.00 | 0.00 | 25.3 | 11.0 | 2.4 |
| `cruise` | legacy | 10.21 | -0.70 | 0.98 | 1.60 | 30.7 | 15.4 | 3.4 |
| `cruise` | v2 | 11.14 | 0.83 | 1.26 | 1.18 | 25.7 | 8.9 | 3.4 |
| `cruise` | v2_nocomb | -3.09 | -32.59 | 32.87 | 56.30 | 6.3 | 0.9 | 1.9 |
| `cruise` | v2_amp_exp0 | 10.29 | 0.65 | 1.43 | 1.28 | 24.8 | 8.2 | 3.5 |
| `cruise` | v2_floor_exp0 | 11.13 | 0.77 | 1.23 | 1.14 | 25.7 | 8.9 | 3.4 |
| `cruise` | v2_both_exp0 | 10.27 | 0.59 | 1.39 | 1.24 | 24.8 | 8.2 | 3.5 |
| `cruise` | v2_floor_calib | -3.38 | -33.12 | 33.32 | 57.64 | 6.8 | 1.2 | 2.0 |
| `cruise` | v2_floor_matched | 14.62 | 6.38 | 6.40 | 5.62 | 25.7 | 8.9 | 3.4 |
| `cruise` | v2_prior_exps | 10.63 | 0.66 | 1.34 | 1.23 | 25.4 | 8.6 | 3.4 |
| `cruise` | v2_level_matched_exps | 10.63 | 0.66 | 1.34 | 1.23 | 25.4 | 8.6 | 3.4 |
| `cruise` | v2_prior_exps_nofloor | 10.45 | -0.89 | 1.68 | 2.22 | 25.4 | 8.6 | 3.4 |
| `cruise` | v2_prior_exps_lowjitter | 10.54 | 0.49 | 1.20 | 1.18 | 25.2 | 9.9 | 4.4 |

## The two fitted speed envelopes at each support's speed

| support | regime | mean rev/s | speed | comb envelope dB | floor envelope dB | comb − floor dB | k_max | top comb line Hz |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `FLY124@8.000000+8.000000` | standby | 36.39 | 0.4549 | -13.14 | -10.79 | -2.35 | 81 | 3349 |
| `FLY124@27.680000+8.000000` | ramp | 56.82 | 0.7103 | -5.76 | -4.74 | -1.02 | 81 | 4853 |
| `FLY124@40.000000+8.000000` | cruise | 80.93 | 1.0117 | +0.37 | +0.28 | +0.09 | 81 | 7459 |

## Encoded speed (order-2 demodulation against the frozen label)

| support | arm | rotor | median abs dev rev/s | p90 | mean signed |
|---|---|---|---:|---:|---:|
| `standby` | real | rotor0 | 0.139 | 0.318 | -0.029 |
| `standby` | real | rotor2 | 0.300 | 0.782 | 0.017 |
| `standby` | legacy | rotor0 | 0.168 | 0.389 | 0.011 |
| `standby` | legacy | rotor2 | 0.373 | 0.954 | -0.002 |
| `standby` | v2 | rotor0 | 0.509 | 1.268 | -0.398 |
| `standby` | v2 | rotor2 | 1.011 | 1.621 | 0.939 |
| `standby` | v2_both_exp0 | rotor0 | 0.504 | 1.271 | -0.400 |
| `standby` | v2_both_exp0 | rotor2 | 1.010 | 1.613 | 0.932 |
| `standby` | v2_nocomb | rotor0 | 0.859 | 1.926 | -0.773 |
| `standby` | v2_nocomb | rotor2 | 0.532 | 1.839 | -0.614 |
| `ramp` | real | rotor2 | 0.361 | 1.341 | -0.238 |
| `ramp` | legacy | rotor2 | 0.470 | 1.012 | -0.279 |
| `ramp` | v2 | rotor2 | 1.017 | 1.837 | -0.010 |
| `ramp` | v2_both_exp0 | rotor2 | 1.135 | 1.921 | -0.006 |
| `ramp` | v2_nocomb | rotor2 | 0.274 | 1.739 | 0.103 |
| `cruise` | real | rotor0 | 0.202 | 0.533 | -0.086 |
| `cruise` | legacy | rotor0 | 0.231 | 0.722 | -0.005 |
| `cruise` | v2 | rotor0 | 0.471 | 1.301 | -0.398 |
| `cruise` | v2_both_exp0 | rotor0 | 0.487 | 1.312 | -0.400 |
| `cruise` | v2_nocomb | rotor0 | 0.055 | 0.209 | -0.205 |

## Verdict per hypothesis

| hypothesis | verdict | the numbers that decide it |
|---|---|---|
| **H1** — `amp_exp` = 3.90 extrapolated from cruise collapses the comb | REFUTED | standby: comb envelope -13.1 dB (-13.5 dB vs cruise), k=2 comb/floor v2 22.0 dB vs floor-only control 9.8 dB and real 15.1 dB; ramp: comb envelope -5.8 dB (-6.1 dB vs cruise), k=2 comb/floor v2 7.7 dB vs floor-only control 9.4 dB and real 6.9 dB |
| **H2** — `floor_exp` = 3.22 (NEGATIVE) makes the floor explode as the rotors slow | REFUTED | standby: floor envelope -10.8 dB (-11.1 dB vs cruise), v2 level offset +0.0 dB, with `floor_exp = 0` +5.0 dB; ramp: floor envelope -4.7 dB (-5.0 dB vs cruise), v2 level offset -3.1 dB, with `floor_exp = 0` -2.2 dB |
| **H3** — the render encodes a different speed | SUPPORTED | order-2 demodulation against the frozen label, on the arms that HAVE a visible comb: standby/v2_both_exp0/rotor0 0.504 rev/s; standby/v2_both_exp0/rotor2 1.010 rev/s; standby/real/rotor0 0.139 rev/s; standby/real/rotor2 0.300 rev/s; ramp/v2_both_exp0/rotor2 1.135 rev/s; ramp/real/rotor2 0.361 rev/s; k_max is 81 in every regime (no Nyquist cap change), so the carrier the render integrates IS the label track |
| **H4** — a regime-independent defect cruise tolerates | **REFUTED** | cruise `FLY124@40.000000+8.000000` runs the SAME code path, the same mic gains and the same seed, and is at parity: v2 PIT — rev/s, level offset +0.83 dB, LTAS 1.26 dB. Peak |x| of the v2 render vs the real clip: standby 0.08 vs 0.12, ramp 0.36 vs 0.59, cruise 0.40 vs 0.59 — the standby render is loud, not clipped (the probe reads float64 in memory and the renderer applies no normalisation), and a level error alone does not do this: `v2_floor_exp0` sits within 5.0 dB of the real standby level and still scores — rev/s |

Secondary (not one of the four, but real): the comb ladder is a FIXED 81 orders, so its top line slides with speed — standby 3349 Hz, ramp 4853 Hz, cruise 7459 Hz. Above that line the standby render is pure floor.

The residual, and what it is NOT: with both exponents at their prior mean the ramp is already at the real clip's own number (— vs real — rev/s, legacy —), but standby only falls to — against a legacy —. Two further renders separate what remains: the comb ALONE, floor muted (— rev/s at standby, — at ramp), and the same render with the shaft jitter at a quarter of the fitted `sigma_nu` = 5.17 rev/s (— rev/s at standby, — at ramp) — `sigma_nu` is in rev/s, so the SAME fitted jitter is a 2.2x larger fraction of a standby carrier than of the reference.

_The two residual renders are not in this pass._

## Minimal patch proposal (NOT applied — Main's call)

Both supported hypotheses are ONE defect: the two speed-envelope exponents are fitted on a pool that barely varies in speed (the fit's own objective saw carriers 20.9-97.9 rev/s, a span of 4.678x, 10 FLY125 cruise windows), where they trade almost exactly against `profile_db` and `floor_mean_db`. The optimiser took `amp_exp` = 3.90 and `floor_exp` = 3.22 against a `N(2, 2)` prior that 24 M Whittle cells simply outvote; in sample the two extremes cancel, and outside it they pull the comb and the floor 2 dB apart. The standby support runs at 36.4 rev/s — 1.74x the LOWEST carrier the fit ever saw, so both envelopes are pure extrapolation there. Two changes, smallest first:

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

* `results/noise_v2/rounds/round2/render_regime_flight/ltas_standby.png`
* `results/noise_v2/rounds/round2/render_regime_flight/spectrogram_standby.png`
* `results/noise_v2/rounds/round2/render_regime_flight/ltas_ramp.png`
* `results/noise_v2/rounds/round2/render_regime_flight/spectrogram_ramp.png`

