# Noise model v2 — round 2: why the v2 DREGON arm renders to PIT MAE 75 rev/s

Record `results/noise_v2/rounds/round2/render_dregon/render_dregon.json`, git `206a1786156f0885736519e1cc1d96b60212180d`, fit `results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only.json` (mode `flight_floor_only`, `converged` None — NOT converged; amp_exp 0.0000, floor_exp 13.0041, floor_static_rel 0.4690, floor_mean_db -38.097, floor_tilt -8.359 dB/oct), render seed 2001, 8 mics.

The fit's floor level never moved from its initialisation: `diagnostics.init_floor_mean_db` -38.096769 vs `params.floor.floor_mean_db` -38.096769 (delta 0.00e+00 dB).

HPPNet job `nv2-r2-dregon-render-0aa14f` on `uni-gpushort`; checkpoint sha256 verified in-job.
Frozen scorer digest `6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1` (= `gates.SCORER_SHA256`).

## HPPNet PIT MAE per arm (rev/s, frozen HPPNet, seed 2001, 8 mics)

| arm | hovering | updown | what it isolates |
|---|---:|---:|---|
| `real` | 0.888 | 1.695 | reference |
| `legacy` | 1.895 | 2.499 | reference (the parity bar's own render) |
| `v2` | 80.927 | 78.848 | the failing arm |
| `v2_nocomb` | 80.925 | 78.894 | control: what the render scores with NO comb at all |
| `v2_nofloor` | 28.589 | 31.043 | control: is the frozen comb trackable on its own? |
| `v2_prior_exps` | 80.925 | 79.273 | H1: the R1 regime patch — are the exponents the defect here too? |
| `v2_floor_matched` | 68.916 | 50.579 | H3, floor side: is the floor level itself wrong? |
| `v2_comb_real_ratio` | 80.927 | 78.978 | the literal (e): match real's OWN measured k=2 comb-to-floor |
| `v2_comb_bench_ratio` | 28.780 | 32.364 | H3, comb side: restore the ratio the frozen comb was fitted under |
| `v2_comb_level_matched` | 2.029 | 4.366 | H3, comb side: the comb carries the real clip's own broadband level |

`v2` reproduces the scored audio: hovering max |float64 render - committed npz| 1.751e-07; updown max |float64 render - committed npz| 2.086e-07 (float32 storage rounding).

## Levels and LTAS (30–7900 Hz, absolute)

| support | arm | band level mic0 dB | mic-mean dB | offset vs real dB | LTAS abs dB | LTAS shape dB | broadband RMS |
|---|---|---:|---:|---:|---:|---:|---:|
| `hovering` | `real` | 15.30 | 13.30 | +0.00 | 0.00 | 0.00 | 0.0735 |
| `hovering` | `legacy` | 17.07 | 14.50 | +1.77 | 2.60 | 0.45 | 0.0843 |
| `hovering` | `v2` | 10.50 | 10.56 | -4.80 | 3.65 | 2.59 | 0.0552 |
| `hovering` | `v2_nocomb` | 10.44 | 9.97 | -4.86 | 3.86 | 2.44 | 0.0519 |
| `hovering` | `v2_nofloor` | -8.22 | 1.51 | -23.51 | 21.33 | 15.25 | 0.0186 |
| `hovering` | `v2_prior_exps` | 8.22 | 8.59 | -7.08 | 5.84 | 2.71 | 0.0436 |
| `hovering` | `v2_floor_matched` | 15.32 | 15.03 | +0.02 | 1.79 | 2.49 | 0.0928 |
| `hovering` | `v2_comb_real_ratio` | 10.52 | 10.70 | -4.78 | 3.61 | 2.63 | 0.0561 |
| `hovering` | `v2_comb_bench_ratio` | 30.05 | 39.74 | +14.75 | 17.47 | 13.76 | 1.5195 |
| `hovering` | `v2_comb_level_matched` | 15.31 | 23.51 | +0.01 | 5.19 | 8.01 | 0.2353 |
| `updown` | `real` | 16.43 | 14.53 | +0.00 | 0.00 | 0.00 | 0.0824 |
| `updown` | `legacy` | 16.19 | 13.39 | -0.23 | 4.59 | 1.07 | 0.0749 |
| `updown` | `v2` | 9.93 | 10.06 | -6.49 | 5.48 | 3.32 | 0.0515 |
| `updown` | `v2_nocomb` | 9.88 | 9.44 | -6.54 | 5.70 | 3.15 | 0.0481 |
| `updown` | `v2_nofloor` | -8.24 | 1.48 | -24.66 | 22.57 | 13.58 | 0.0186 |
| `updown` | `v2_prior_exps` | 8.05 | 8.40 | -8.38 | 7.33 | 3.43 | 0.0426 |
| `updown` | `v2_floor_matched` | 16.43 | 16.12 | +0.01 | 1.91 | 3.19 | 0.1038 |
| `updown` | `v2_comb_real_ratio` | 9.92 | 9.97 | -6.50 | 5.51 | 3.30 | 0.0510 |
| `updown` | `v2_comb_bench_ratio` | 29.15 | 38.83 | +12.72 | 15.29 | 12.80 | 1.3738 |
| `updown` | `v2_comb_level_matched` | 16.37 | 25.15 | -0.06 | 5.38 | 9.39 | 0.2850 |

## Order-tracked comb prominence (dB over the local floor)

Read at each frame's OWN label carrier (4096-point, hop 1024, peak = mean of ±1 bin, floor = median of the ±0.5·f0 bins excluding every rotor's k−1/k/k+1 line). A periodogram at the MEAN carrier is useless here: DREGON cruise carriers move 10–25 rev/s inside the 4 s window and smear every line into its neighbours.

| support | arm | k=2 mic0 | k=4 | k=8 | k=2 mic-mean | k=4 | k=8 |
|---|---|---:|---:|---:|---:|---:|---:|
| `hovering` | `real` | +0.33 | -0.13 | +0.32 | +5.83 | +1.39 | +0.02 |
| `hovering` | `legacy` | +5.84 | +2.60 | +1.75 | +6.50 | +3.06 | +0.73 |
| `hovering` | `v2` | -0.02 | +0.33 | +0.94 | +2.27 | +1.60 | +2.06 |
| `hovering` | `v2_nocomb` | -0.10 | +0.18 | +0.63 | +0.05 | +0.49 | +0.59 |
| `hovering` | `v2_nofloor` | +28.17 | +16.92 | +6.43 | +29.08 | +19.38 | +6.71 |
| `hovering` | `v2_prior_exps` | -0.01 | +0.34 | +0.95 | +3.22 | +2.16 | +2.71 |
| `hovering` | `v2_floor_matched` | -0.31 | +0.28 | +0.76 | +0.80 | +0.78 | +1.21 |
| `hovering` | `v2_comb_real_ratio` | +0.05 | +0.34 | +0.91 | +2.64 | +1.82 | +2.32 |
| `hovering` | `v2_comb_bench_ratio` | +22.82 | +15.51 | +6.27 | +26.79 | +18.54 | +6.71 |
| `hovering` | `v2_comb_level_matched` | +9.87 | +6.11 | +5.27 | +17.68 | +12.66 | +6.27 |
| `updown` | `real` | +0.27 | +0.42 | +0.32 | +1.91 | +0.52 | +0.18 |
| `updown` | `legacy` | +4.56 | +1.61 | +1.11 | +5.17 | +2.34 | +0.89 |
| `updown` | `v2` | +0.44 | -0.24 | +1.19 | +2.85 | +1.73 | +2.42 |
| `updown` | `v2_nocomb` | -0.21 | -0.42 | +0.56 | +0.09 | +0.48 | +0.50 |
| `updown` | `v2_nofloor` | +28.73 | +17.03 | +6.90 | +29.58 | +19.39 | +6.76 |
| `updown` | `v2_prior_exps` | +0.71 | -0.10 | +1.25 | +3.70 | +2.11 | +2.76 |
| `updown` | `v2_floor_matched` | +0.11 | -0.30 | +1.16 | +0.86 | +0.85 | +1.24 |
| `updown` | `v2_comb_real_ratio` | +0.35 | -0.26 | +1.24 | +2.55 | +1.63 | +2.32 |
| `updown` | `v2_comb_bench_ratio` | +23.33 | +15.02 | +6.80 | +27.19 | +18.62 | +6.69 |
| `updown` | `v2_comb_level_matched` | +12.45 | +8.19 | +6.31 | +19.99 | +14.19 | +6.42 |

Separated blocks of the same seed (`v2_nofloor` comb-only against `v2_nocomb` floor-only) give the exact per-line comb-to-floor, without any estimator:

| support | k | comb-to-floor per line dB | implied full-render prominence dB |
|---|---:|---:|---:|
| `hovering` | 2 | -12.05 | +0.29 |
| `updown` | 2 | -11.18 | +0.35 |

## The comb-to-floor the frozen comb was IDENTIFIED at (bench, own carriers)

| bench fit | carrier rev/s | floor_mean_db | comb-only band dB | floor-only band dB | comb−floor band dB | k=2 line ratio dB | full k=2 prominence dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `bench_dregon_Motor1_70` (`converged` None) | 68.314 | -50.13 | -9.48 | -18.97 | +9.50 | +32.83 | +30.84 |
| `bench_dregon_Motor2_70` (`converged` None) | 67.552 | -50.58 | 0.24 | -9.02 | +9.26 | +20.96 | +20.36 |
| `bench_dregon_Motor3_70` (`converged` None) | 68.634 | -49.30 | -9.72 | -9.42 | -0.30 | +7.81 | +6.94 |
| `bench_dregon_Motor4_70` (`converged` None) | 69.333 | -52.33 | -8.65 | -42.42 | +33.77 | +43.07 | +30.33 |
| **mean** | | -50.58 | | | +13.06 | +26.16 | |

## The fit's two speed laws at the bench carrier vs the DREGON cruise carriers

| support | carriers | mean speed (÷80) | comb envelope dB | floor envelope dB | comb−floor dB | k_max | top comb line Hz |
|---|---|---:|---:|---:|---:|---:|---:|
| `hovering` | bench ~68 rev/s | 0.8557 | +0.00 | -2.21 | +2.21 | 88 | 6101 |
| `hovering` | DREGON cruise | 1.0113 | +0.00 | +2.44 | -2.44 | 88 | 7488 |
| `hovering` | **bench → cruise shift** | | +0.00 | +4.65 | -4.65 | | |
| `updown` | bench ~68 rev/s | 0.8557 | +0.00 | -2.21 | +2.21 | 88 | 6101 |
| `updown` | DREGON cruise | 0.9913 | +0.00 | +1.60 | -1.60 | 88 | 7284 |
| `updown` | **bench → cruise shift** | | +0.00 | +3.80 | -3.80 | | |

## The comb's absolute level, term by term

render_noise centres mic_line_gain_db over mics (per rotor) and mic_gains_db over mics before use, so their ABSOLUTE values are discarded by the renderer; the only absolute comb scale a render has is profile_db.

| fit | orders | Σ_k line power, mic 0, all rotors dB | mic-mean dB | raw mic_line_gain_db mean (DISCARDED) | raw mic_gains_db mean (DISCARDED) |
|---|---:|---:|---:|---|---:|
| candidate (log-mean comb, 4 rotors) | 88 | -44.33 | -34.62 | 0.38, 0.19, 0.25, -0.19 | 0.33 |
| `bench_dregon_Motor1_70` (1 rotor) | 117 | -43.83 | -41.56 | 0.34 | -0.15 |
| `bench_dregon_Motor2_70` (1 rotor) | 118 | -34.27 | -35.53 | 0.78 | -0.05 |
| `bench_dregon_Motor3_70` (1 rotor) | 116 | -45.62 | -43.71 | 1.91 | -1.06 |
| `bench_dregon_Motor4_70` (1 rotor) | 115 | -42.92 | -39.44 | 2.18 | 0.60 |

Candidate minus the bench mean at mic 0: -2.67 dB — the four-rotor sum (+6.02 dB if the rotors were identical and the per-rotor mic gains neutral) minus what the dB-mean of the four bench profiles and the per-rotor mic-gain centring take back.

## The re-levelling shifts each variant applies (all solved, none searched)

| support | comb-only band dB | floor-only band dB | comb−floor dB | (d) floor→real level dB | (e) comb→real k=2 ratio dB | comb→bench ratio dB | comb→real band level dB |
|---|---:|---:|---:|---:|---:|---:|---:|
| `hovering` | -8.22 | 10.44 | -18.66 | +4.86 | +1.00 | +38.22 | +21.80 |
| `updown` | -8.24 | 9.88 | -18.12 | +6.54 | -0.69 | +37.35 | +23.57 |

## Encoded speed (order-2 demodulation against the frozen label)

| support | arm | rotor | median abs dev rev/s | p90 | mean signed |
|---|---|---|---:|---:|---:|
| `hovering` | `real` | rotor0 | 0.437 | 1.840 | 0.171 |
| `hovering` | `legacy` | rotor0 | 0.551 | 1.656 | -0.150 |
| `hovering` | `v2_nofloor` | rotor0 | 0.243 | 0.521 | -0.060 |
| `hovering` | `v2_comb_level_matched` | rotor0 | 0.287 | 0.644 | -0.082 |
| `updown` | `real` | rotor0 | 0.652 | 1.732 | 0.090 |
| `updown` | `legacy` | rotor0 | 0.470 | 1.731 | 0.174 |
| `updown` | `v2_nofloor` | rotor0 | 0.329 | 0.624 | -0.090 |
| `updown` | `v2_comb_level_matched` | rotor0 | 0.323 | 0.684 | -0.094 |

## Verdict per hypothesis

| hypothesis | verdict | the numbers that decide it |
|---|---|---|
| **H1** — the bench comb was fitted at ~68 rev/s and the cruise carriers differ, so the speed law / Nyquist geometry is wrong at cruise | **REFUTED as the cause** | `amp_exp` is 0.0 (a bench fit has one throttle and cannot identify a speed law), so the comb envelope shift bench → cruise is hovering +0.00, updown +0.00 dB; the floor's own law moves hovering +4.65, updown +3.80 dB, so the whole bench → cruise speed-law effect on comb−floor is under 5 dB against the -18.7 dB deficit measured. `k_max` is 88 at cruise vs 88 at the bench carrier (no Nyquist cap change). Pinning BOTH exponents at the prior mean leaves PIT at hovering 80.925, updown 79.273 rev/s against the unchanged hovering 80.927, updown 78.848 — the R1 regime patch does nothing here |
| **H2** — a units / per-rotor level mismatch in `mic_line_gain_db` / `mic_gains_db` | **REFUTED as a units bug, SUPPORTED as a missing degree of freedom** | `render_noise` mean-centres `mic_line_gain_db` over mics per rotor (`render.py:191`) and `mic_gains_db` over mics (`:192`), so their absolute values are DISCARDED: the only absolute comb scale a render has is `profile_db`. Σ_k line power at mic 0 is -44.33 dB for the candidate against a bench mean of -41.66 dB (-2.67 dB) — the transplanted comb carries the BENCH rig's absolute level and there is no parameter anywhere that could re-level it |
| **H3** — the floor-only flight fit's floor buries the comb | **SUPPORTED — this is the defect** | comb-only minus floor-only band level: hovering -18.66, updown -18.12 dB in the flight render against +13.06 dB on the bench where the comb was fitted; per-line at k=2, hovering -12.05, updown -11.18 dB against the bench +26.16 dB. Consequence: `v2` hovering 80.927, updown 78.848 rev/s is INDISTINGUISHABLE from `v2_nocomb` hovering 80.925, updown 78.894 — the render's comb contributes nothing at all, and the per-rotor PIT MAE equals the per-rotor label speed (the tracker returns no rotor). Restoring the comb level fixes it: comb → real band level hovering 2.029, updown 4.366, comb → bench ratio hovering 28.780, updown 32.364, against legacy hovering 1.895, updown 2.499 and real hovering 0.888, updown 1.695. Moving the FLOOR instead does not: hovering 68.916, updown 50.579 |
| **H4** — the render encodes a different speed than the label | **REFUTED** | order-2 demodulation of the comb-only render against the frozen label: hovering rotor0 0.243; updown rotor0 0.329 rev/s median absolute deviation — the comb sits on the label carrier by construction (`render_noise` integrates the carriers it is handed). Every frozen DREGON support is one 4 s cruise window scored whole (hovering 4.00, updown 4.00 s), so there is no standby-like frame inside the score window, and `k_max` 88 puts the top comb line at 7488 Hz, inside the band |

## Minimal patch proposal (NOT applied — Main's call)

One defect, one missing degree of freedom. The comb was frozen from four DREGON BENCH fits whose own floors sit at -50.58 dB, where the comb is +13.06 dB ABOVE the floor in band power and +26.16 dB above it per line at k=2. The flight fit is `mode=flight_floor_only`: its objective frees the FLOOR ONLY, so nothing in it ever balanced the transplanted comb against the flight floor — and its floor never even left its own initialisation (-38.0968 dB, `converged` None). On the DREGON cruise supports the same comb ends up -18.66 dB relative to the floor, i.e. the comb-to-floor ratio swung by 31.7 dB across the transplant. The renderer cannot absorb that: it mean-centres both `mic_line_gain_db` and `mic_gains_db`, so `profile_db` is the ONLY absolute comb scale and it is frozen. The minimal fix is therefore to give `flight_floor_only` exactly one more free scalar — a shared comb gain in dB — and let the flight likelihood set it:

```diff
@@ src/experiments/noise_model/model.py:99 class Priors
+    #: rig-to-rig level offset of a TRANSPLANTED comb, in dB. Zero-mean and
+    #: wide: it is a geometry/distance term (bench mic at ~1 m and one rotor
+    #: against a flight array at ~0.3 m and four), not a physical one.
+    comb_gain_db: tuple[float, float] = (0.0, 15.0)
@@ src/experiments/noise_model/model.py:162 free_blocks
     if mode == "flight_floor_only":
-        return ("floor", "mic")
+        # the comb arrives FROZEN from a bench rig, so its absolute level is
+        # that rig's; nothing downstream can re-level it, because
+        # render_noise mean-centres mic_line_gain_db over mics per rotor
+        # (render.py:191) and mic_gains_db over mics (:192) and profile_db is
+        # then the ONLY absolute comb scale. One shared scalar, no more.
+        return ("floor", "mic", "comb_gain")
@@ src/experiments/noise_model/model.py:586 sample_params
     if "profile" in free:
         profile_db = _normal(site, "profile_db", priors.profile_db[0], priors.profile_db[1], (r, k))
         amp_exp = _normal(site, "amp_exp", *priors.amp_exp) if flight else zero
     else:
         profile_db = take("profile", "profile_db", (r, k))
         amp_exp = take("profile", "amp_exp") if flight else zero
+    if "comb_gain" in free:
+        # folded into profile_db, so write_fit, render_noise and
+        # expected_periodogram need no change at all
+        profile_db = profile_db + _normal(site, "comb_gain_db", *priors.comb_gain_db)
```


### Expected effect — measured, not guessed

No variant below is a FIT, so none of them is a parity claim: each is the SAME frozen comb and the SAME fitted floor with ONE level moved, scored by the same frozen HPPNet on the same frozen supports and seed. What they bound is how much a single free comb gain can buy.

| arm | comb shift dB | hovering PIT | updown PIT |
|---|---:|---:|---:|
| `real` | — | 0.888 | 1.695 |
| `legacy` | — | 1.895 | 2.499 |
| `v2` | — | 80.927 | 78.848 |
| `v2_prior_exps` | +0.00 | 80.925 | 79.273 |
| `v2_floor_matched` | +0.00 | 68.916 | 50.579 |
| `v2_comb_real_ratio` | +1.00 | 80.927 | 78.978 |
| `v2_comb_bench_ratio` | +38.22 | 28.780 | 32.364 |
| `v2_comb_level_matched` | +21.80 | 2.029 | 4.366 |

Two of the five gate supports only, so this is INDICATIVE and not a gate verdict: the frozen DREGON target is 1.897063 rev/s and the round-2 arm scored 75.093968 over all five. With the comb re-levelled onto the real window's own band level these two score 2.029, 4.366 rev/s, against legacy 1.895, 2.499 and real 0.888, 1.695 on the same two. The level-matched comb shift is the counterfactual a free `comb_gain_db` would have to find; the prior above ({0, 15} dB) covers it.

`v2_comb_bench_ratio` OVERSHOOTS and is not the number to target. The four bench fits disagree wildly about their own floors — floor-only band level Motor1 -18.97, Motor2 -9.02, Motor3 -9.42, Motor4 -42.42 dB — so their mean k=2 comb-to-floor +26.16 dB is dominated by `Motor4` (+43.07 dB, floor-only -42.42 dB) against `Motor3` (+7.81 dB). The shift it implies (+38.22 dB) puts the render +17.5 dB over the real level and the PIT back up to 28.780, 32.364 rev/s. The real window's own BAND LEVEL is the well-posed target, and it is what a free `comb_gain_db` fitted on the flight supports would be pulled to.

The literal variant (e) — the comb re-levelled so its k=2 prominence matches the REAL clip's — is DEGENERATE on DREGON and is reported for completeness only. At mic 0 the real room-2 cruise clip has almost no discrete line prominence at the `motors_command` carrier (hovering +0.33 dB at k=2, updown +0.27 dB at k=2), so matching it asks the comb to stay where it is (hovering +1.00 dB, updown -0.69 dB). The 8-mic MEAN prominence is larger — real hovering +5.83, updown +1.91 dB at k=2 against `v2_nocomb` hovering +0.05, updown +0.09 dB — so real audio DOES carry a comb, spatially coherent enough to survive the 8-mic average, while the v2 render carries none. But it is a WEAK comb: HPPNet reaches hovering 0.888, updown 1.695 rev/s on it. The render's target is therefore not 'a prominent comb' but 'the real clip's own comb ENERGY', which is what the band-level variant sets.

## Figures

* `results/noise_v2/rounds/round2/render_dregon/ltas_hovering.png`
* `results/noise_v2/rounds/round2/render_dregon/spectrogram_hovering.png`
* `results/noise_v2/rounds/round2/render_dregon/ltas_updown.png`
* `results/noise_v2/rounds/round2/render_dregon/spectrogram_updown.png`
