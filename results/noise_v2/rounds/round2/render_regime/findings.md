# Noise model v2 — round 2: why the v2 render loses standby and ramp

Record `results/noise_v2/rounds/round2/render_regime/render_regime.json`, git `5b7efb2f49474a7cc80b2c5bdef2caf6ca702f6e`, fit `results/noise_v2/rounds/round1/fits/michaels_fly125_cruise__flight.json` (amp_exp 10.0196, floor_exp -10.1892, floor_static_rel 0.004928, reference speed 80 rev/s), render seed 2001, 8 mics.

HPPNet job `nv2-r2-regime-probe3-685b70` on `uni-gpushort` (frozen checkpoint sha256 verified in-job).

## PIT MAE per support and arm (rev/s, frozen HPPNet, seed 2001)

| support | regime | real | legacy | v2 | v2_nocomb | v2_amp_exp0 | v2_floor_exp0 | v2_both_exp0 | v2_floor_matched | v2_prior_exps | v2_level_matched_exps | v2_prior_exps_nofloor | v2_prior_exps_lowjitter |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `FLY124@8.000000+8.000000` | standby | 0.398 | 0.313 | 32.395 | 32.402 | 30.357 | 29.507 | 13.594 | 29.269 | 13.276 | 17.754 | 11.895 | 15.189 |
| `FLY124@27.680000+8.000000` | ramp | 3.171 | 9.649 | 15.725 | 29.851 | 4.775 | 3.043 | 3.344 | 26.917 | 3.210 | 3.181 | 3.210 | 2.776 |
| `FLY124@40.000000+8.000000` | cruise | 0.885 | 1.301 | 1.139 | 69.312 | 1.114 | 1.149 | 1.127 | 7.165 | 1.127 | 1.127 | 1.427 | 0.963 |

## Levels and LTAS (mic 0, 30–7900 Hz, absolute)

| support | arm | band level dB | level offset vs real dB | LTAS mean abs dB | LTAS shape-only dB | comb/floor k=2 dB | k=4 | k=8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `standby` | real | -2.81 | 0.00 | 0.00 | 0.00 | 15.1 | 6.9 | 5.6 |
| `standby` | legacy | -3.36 | -0.41 | 0.59 | 0.67 | 16.4 | 8.8 | 6.3 |
| `standby` | v2 | 29.51 | 34.14 | 34.14 | 15.05 | 4.4 | 1.9 | 1.5 |
| `standby` | v2_nocomb | 29.51 | 34.14 | 34.14 | 15.05 | 4.4 | 2.0 | 1.5 |
| `standby` | v2_amp_exp0 | 29.55 | 34.52 | 34.52 | 13.99 | 11.2 | 7.9 | 3.6 |
| `standby` | v2_floor_exp0 | -7.37 | -2.24 | 6.45 | 13.39 | 13.5 | 9.4 | 4.8 |
| `standby` | v2_both_exp0 | 10.03 | 10.58 | 10.58 | 3.70 | 23.7 | 11.1 | 8.0 |
| `standby` | v2_floor_calib | -7.46 | -2.82 | 7.03 | 15.05 | 4.4 | 2.0 | 1.5 |
| `standby` | v2_floor_matched | -2.78 | 2.10 | 7.66 | 14.26 | 10.9 | 7.3 | 3.3 |
| `standby` | v2_prior_exps | 3.09 | 3.76 | 5.12 | 3.65 | 23.7 | 11.2 | 8.2 |
| `standby` | v2_level_matched_exps | -4.84 | -3.94 | 4.33 | 3.48 | 23.8 | 11.3 | 8.6 |
| `standby` | v2_prior_exps_nofloor | 3.02 | 0.18 | 7.92 | 7.14 | 24.0 | 11.2 | 8.2 |
| `standby` | v2_prior_exps_lowjitter | 2.95 | 3.03 | 4.40 | 3.43 | 28.5 | 16.5 | 15.8 |
| `ramp` | real | 9.33 | 0.00 | 0.00 | 0.00 | 6.9 | 2.8 | 1.7 |
| `ramp` | legacy | 6.35 | -1.95 | 4.19 | 3.77 | 6.2 | 3.4 | 4.0 |
| `ramp` | v2 | 25.90 | 17.11 | 17.11 | 11.88 | -0.7 | 1.7 | 1.7 |
| `ramp` | v2_nocomb | 25.86 | 16.45 | 17.16 | 12.85 | 1.8 | 2.4 | 1.7 |
| `ramp` | v2_amp_exp0 | 25.97 | 17.55 | 17.55 | 10.48 | 4.6 | 4.0 | 1.5 |
| `ramp` | v2_floor_exp0 | 5.48 | -4.44 | 4.44 | 1.10 | 5.4 | 2.6 | 5.1 |
| `ramp` | v2_both_exp0 | 9.99 | -1.23 | 2.20 | 3.91 | 6.4 | 5.2 | 2.3 |
| `ramp` | v2_floor_calib | -7.46 | -16.73 | 16.73 | 12.91 | 0.9 | 1.7 | 1.1 |
| `ramp` | v2_floor_matched | 10.76 | 3.22 | 5.48 | 6.95 | 4.8 | 1.8 | 2.9 |
| `ramp` | v2_prior_exps | 7.28 | -3.39 | 3.39 | 3.04 | 7.8 | 3.6 | 3.2 |
| `ramp` | v2_level_matched_exps | 7.44 | -3.27 | 3.27 | 3.10 | 7.7 | 3.7 | 3.1 |
| `ramp` | v2_prior_exps_nofloor | 7.21 | -5.22 | 5.22 | 4.86 | 7.8 | 3.6 | 3.2 |
| `ramp` | v2_prior_exps_lowjitter | 6.96 | -3.45 | 3.45 | 2.78 | 9.3 | 5.8 | 4.1 |
| `cruise` | real | 12.16 | 0.00 | 0.00 | 0.00 | 25.3 | 11.0 | 2.4 |
| `cruise` | legacy | 10.21 | -0.70 | 0.98 | 1.60 | 30.7 | 15.4 | 3.4 |
| `cruise` | v2 | 10.37 | 0.17 | 0.59 | 0.67 | 26.4 | 11.7 | 3.0 |
| `cruise` | v2_nocomb | -6.43 | -15.03 | 15.03 | 11.19 | 0.7 | 1.8 | 1.0 |
| `cruise` | v2_amp_exp0 | 9.90 | -0.34 | 0.94 | 1.62 | 23.8 | 9.5 | 4.6 |
| `cruise` | v2_floor_exp0 | 10.35 | 0.03 | 0.61 | 0.74 | 26.4 | 11.7 | 3.0 |
| `cruise` | v2_both_exp0 | 9.88 | -0.53 | 1.12 | 1.80 | 23.8 | 9.5 | 4.6 |
| `cruise` | v2_floor_calib | -7.46 | -16.03 | 16.03 | 11.19 | 0.8 | 1.7 | 0.9 |
| `cruise` | v2_floor_matched | 14.33 | 7.02 | 7.02 | 5.52 | 25.5 | 9.3 | 2.0 |
| `cruise` | v2_prior_exps | 9.68 | -0.58 | 0.96 | 1.53 | 24.4 | 9.9 | 4.3 |
| `cruise` | v2_level_matched_exps | 9.68 | -0.58 | 0.96 | 1.53 | 24.4 | 9.9 | 4.3 |
| `cruise` | v2_prior_exps_nofloor | 9.59 | -1.92 | 2.28 | 2.84 | 24.4 | 9.9 | 4.4 |
| `cruise` | v2_prior_exps_lowjitter | 9.68 | -0.76 | 0.98 | 1.24 | 27.3 | 11.1 | 6.6 |

## The two fitted speed envelopes at each support's speed

| support | regime | mean rev/s | speed | comb envelope dB | floor envelope dB | comb − floor dB | k_max | top comb line Hz |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `FLY124@8.000000+8.000000` | standby | 36.39 | 0.4549 | -32.67 | +36.96 | -69.63 | 81 | 3349 |
| `FLY124@27.680000+8.000000` | ramp | 56.82 | 0.7103 | -14.61 | +15.46 | -30.07 | 81 | 4853 |
| `FLY124@40.000000+8.000000` | cruise | 80.93 | 1.0117 | +1.95 | +0.72 | +1.22 | 81 | 7459 |

## Encoded speed (order-2 demodulation against the frozen label)

| support | arm | rotor | median abs dev rev/s | p90 | mean signed |
|---|---|---|---:|---:|---:|
| `standby` | real | rotor0 | 0.139 | 0.318 | -0.029 |
| `standby` | real | rotor2 | 0.300 | 0.782 | 0.017 |
| `standby` | legacy | rotor0 | 0.168 | 0.389 | 0.011 |
| `standby` | legacy | rotor2 | 0.373 | 0.954 | -0.002 |
| `standby` | v2 | rotor0 | 0.740 | 1.837 | -0.569 |
| `standby` | v2 | rotor2 | 0.510 | 1.618 | -0.311 |
| `standby` | v2_both_exp0 | rotor0 | 0.276 | 0.630 | 0.001 |
| `standby` | v2_both_exp0 | rotor2 | 0.238 | 0.665 | 0.024 |
| `standby` | v2_nocomb | rotor0 | 0.739 | 1.835 | -0.568 |
| `standby` | v2_nocomb | rotor2 | 0.494 | 1.605 | -0.303 |
| `ramp` | real | rotor2 | 0.361 | 1.341 | -0.238 |
| `ramp` | legacy | rotor2 | 0.470 | 1.012 | -0.279 |
| `ramp` | v2 | rotor2 | 0.390 | 1.256 | -0.200 |
| `ramp` | v2_both_exp0 | rotor2 | 0.408 | 1.180 | -0.218 |
| `ramp` | v2_nocomb | rotor2 | 0.518 | 1.811 | -0.421 |
| `cruise` | real | rotor0 | 0.202 | 0.533 | -0.086 |
| `cruise` | legacy | rotor0 | 0.231 | 0.722 | -0.005 |
| `cruise` | v2 | rotor0 | 0.240 | 0.608 | 0.006 |
| `cruise` | v2_both_exp0 | rotor0 | 0.226 | 0.627 | -0.001 |
| `cruise` | v2_nocomb | rotor0 | 0.448 | 1.613 | -0.074 |

## Verdict per hypothesis

| hypothesis | verdict | the numbers that decide it |
|---|---|---|
| **H1** — `amp_exp` = 10.02 extrapolated from cruise collapses the comb | **SUPPORTED** | standby: comb envelope -32.7 dB (-34.6 dB vs cruise), k=2 comb/floor v2 4.4 dB vs floor-only control 4.4 dB and real 15.1 dB; ramp: comb envelope -14.6 dB (-16.6 dB vs cruise), k=2 comb/floor v2 -0.7 dB vs floor-only control 1.8 dB and real 6.9 dB |
| **H2** — `floor_exp` = -10.19 (NEGATIVE) makes the floor explode as the rotors slow | **SUPPORTED** | standby: floor envelope +37.0 dB (+36.2 dB vs cruise), v2 level offset +34.1 dB, with `floor_exp = 0` -2.2 dB; ramp: floor envelope +15.5 dB (+14.7 dB vs cruise), v2 level offset +17.1 dB, with `floor_exp = 0` -4.4 dB |
| **H3** — the render encodes a different speed | **REFUTED** | order-2 demodulation against the frozen label, on the arms that HAVE a visible comb: standby/v2_both_exp0/rotor0 0.276 rev/s; standby/v2_both_exp0/rotor2 0.238 rev/s; standby/real/rotor0 0.139 rev/s; standby/real/rotor2 0.300 rev/s; ramp/v2_both_exp0/rotor2 0.408 rev/s; ramp/real/rotor2 0.361 rev/s; k_max is 81 in every regime (no Nyquist cap change), so the carrier the render integrates IS the label track |
| **H4** — a regime-independent defect cruise tolerates | **REFUTED** | cruise `FLY124@40.000000+8.000000` runs the SAME code path, the same mic gains and the same seed, and is at parity: v2 PIT 1.139 rev/s, level offset +0.17 dB, LTAS 0.59 dB. Peak |x| of the v2 render vs the real clip: standby 12.46 vs 0.12, ramp 10.63 vs 0.59, cruise 0.41 vs 0.59 — the standby render is loud, not clipped (the probe reads float64 in memory and the renderer applies no normalisation), and a level error alone does not do this: `v2_floor_exp0` sits within 2.2 dB of the real standby level and still scores 29.507 rev/s |

Secondary (not one of the four, but real): the comb ladder is a FIXED 81 orders, so its top line slides with speed — standby 3349 Hz, ramp 4853 Hz, cruise 7459 Hz. Above that line the standby render is pure floor.

The residual, and what it is NOT: with both exponents at their prior mean the ramp is already at the real clip's own number (3.210 vs real 3.171 rev/s, legacy 9.649), but standby only falls to 13.276 against a legacy 0.313. Two further renders separate what remains: the comb ALONE, floor muted (11.895 rev/s at standby, 3.210 at ramp), and the same render with the shaft jitter at a quarter of the fitted `sigma_nu` = 8.38 rev/s (15.189 rev/s at standby, 2.776 at ramp) — `sigma_nu` is in rev/s, so the SAME fitted jitter is a 2.2x larger fraction of a standby carrier than of the reference.

**Neither moves it**: muting the floor gives 11.895 rev/s and quartering the jitter 15.189, against 13.276 with both envelopes pinned. The standby residual is therefore in the COMB ITSELF — a per-order profile fitted where 81 orders reach 7.5 kHz, reused where they reach 3.3 kHz — and no reparameterisation of the speed laws will remove it. That is what makes (c) (a standby support in the fit) load-bearing rather than optional.

## Minimal patch proposal (NOT applied — Main's call)

Both supported hypotheses are ONE defect: the two speed-envelope exponents are fitted on a pool that barely varies in speed (the fit's own objective saw carriers 68.2-97.9 rev/s, a span of 1.436x, 8 FLY125 cruise windows), where they trade almost exactly against `profile_db` and `floor_mean_db`. The optimiser took `amp_exp` = 10.02 and `floor_exp` = -10.19 against a `N(2, 2)` prior that 24 M Whittle cells simply outvote; in sample the two extremes cancel, and outside it they pull the comb and the floor 71 dB apart. The standby support runs at 36.4 rev/s — 0.53x the LOWEST carrier the fit ever saw, so both envelopes are pure extrapolation there. Two changes, smallest first:

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
| standby | 32.395 | 13.276 | 17.754 (exp 4.31) | 0.313 | 0.398 |
| ramp | 15.725 | 3.210 | 3.181 (exp 1.84) | 9.649 | 3.171 |
| cruise | 1.139 | 1.127 | 1.127 (exp 2.00) | 1.301 | 0.885 |

Equal-regime mean of the three supports measured here, against the frozen Michael's parity bar 3.177994 rev/s (1.05 x the legacy 3.026661) — INDICATIVE only, the gate averages five supports:

| arm | v2 | v2_prior_exps | v2_level_matched_exps | legacy | real |
|---|---:|---:|---:|---:|---:|
| equal-regime mean (3 supports) | 16.420 ✗ | 5.871 ✗ | 7.354 ✗ | 3.755 ✗ | 1.485 ✓ |

Read the DIFFERENCES, not the absolutes: the legacy arm itself scores 3.755 on this three-support subset against its own frozen 3.026661 over five supports (this subset's single ramp support is the harder one — legacy 9.649 here against the frozen legacy ramp reference 8.007098), so the subset is harsher than the gate and no row of it is a gate verdict.

## Figures

* `results/noise_v2/rounds/round2/render_regime/ltas_standby.png`
* `results/noise_v2/rounds/round2/render_regime/spectrogram_standby.png`
* `results/noise_v2/rounds/round2/render_regime/ltas_ramp.png`
* `results/noise_v2/rounds/round2/render_regime/spectrogram_ramp.png`

