# Noise v3 diagnosis: why the latents eat the rig (round-2 fits → round-3 design)

Inputs are the round-2 selected fits `results/noise_v3/fits_r2/<pool>__flight_v3.json` (DREGON s3,
cruise s1, standby s2), the round-1 restart with the same seed (`results/noise_v3/fits/restarts/`)
and the measured wander `results/noise_v3/wander/<rig>.json`.
Light numbers come from `scripts/noise_v3_diag.py` (`ltas-bias`, `fold`, `sanity`, `figs`), run on
the laptop under the cap. That script uses JSON plus one expected frame per fit, every rotor at the
middle of its pool's carrier span (70.8 / 83.0 / 30.6 rev/s). The pool-level numbers come from
`scripts/noise_v3_diag_rig.py`, a GPU job that rebuilds each pool exactly as the fit CLI built it.
Every table below is in the JSON next to this file.

## 1. Model sanity

### 1(a) Is the harmonic profile per rotor, and are per-rotor levels free?

**Yes.** The rig carries these blocks:

- One profile per rotor and order, `profile_db` (R, K): (4, 88) on DREGON, (4, 81) on cruise,
  (4, 130) on standby.
- One width per line, `gamma_hz` (R, K).
- A scalar `sigma_nu`.
- The floor spline `floor_shape_z` (14,), with control values `c = μ + σ_B L z`. μ and σ_B are
  measured constants; no floor level site exists.
- DREGON only: `wind_db` (8,).
- The speed laws `amp_exp`, `floor_exp`, `floor_static_rel`, pinned on cruise.

The latents per window are:

- `d` (R, B), one per rotor, added to all of its orders;
- `v` (R, K, B), one per line;
- `u` (B,) for the floor level;
- `u_j` (14, B), one per floor control point.

B is 16 blocks for an 8 s window and 8 for a 4 s window.

Each line has its own profile prior, `profile_db[r, k] ~ N(p̂_rk, 10 dB)`. p̂ is the pooled
observed level at k f_r (line plus floor, an upper bound). The prior has no rotor term, so
per-rotor levels are free. **The prior centre carries no per-rotor information.** The rotors share
their frequencies, so the p̂ group means differ between rotors by at most 2.4 dB on DREGON,
1.4 dB on cruise and 4.3 dB on standby. The fitted levels do differ per rotor, as the table shows.
Values are the per-rotor mean of `profile_db` over the order group, in dB, profile units.

| pool | orders | rotor 1 | rotor 2 | rotor 3 | rotor 4 | spread | spread of p̂ | static `d` (rotors 1–4) |
|---|---|---:|---:|---:|---:|---:|---:|---|
| DREGON | 1–2 | −48.2 | −44.3 | −41.9 | −38.4 | 9.8 | 2.4 | −0.81 / +0.37 / +1.50 / −1.83 |
| DREGON | 3–8 | −65.6 | −68.7 | −62.7 | −58.7 | 10.1 | 1.3 | |
| DREGON | 9–24 | −65.8 | −62.9 | −62.6 | −56.9 | 8.9 | 0.8 | |
| cruise | 1–2 | −40.4 | −44.6 | −53.4 | −59.1 | 18.6 | 0.0 | +0.09 / −0.02 / +2.29 / +3.33 |
| cruise | 3–8 | −58.8 | −53.4 | −58.8 | −62.6 | 9.2 | 1.4 | |
| cruise | 9–24 | −57.4 | −55.8 | −59.7 | −62.5 | 6.7 | 0.2 | |
| standby | 1–2 | −33.8 | −45.5 | −47.7 | −42.3 | 13.9 | 0.3 | +0.50 / +0.67 / −0.07 / +0.11 |

The rotor split comes from the likelihood alone, through the rotors' speed differences. It is
large: up to 18.6 dB at k 1–2 on cruise. The static `d` pulls part of it back on cruise: rotor 4
is 3.3 dB louder in the latents than in the rig. Fig `fig_1b_profiles.png` also shows a 15–20 dB
odd/even zig-zag on cruise rotor 1 at k ≤ 70.

### 1(b) The prior profile p̂ and the fall-off

The fall-off is regressed against log2(k f_r), in dB per octave with r², pooled over the rotors.
`fig_1b_profiles.png` shows it per rotor.

| pool | p̂ | fitted profile | profile + static d + v |
|---|---|---|---|
| DREGON | −3.15 (0.87) | −3.10 (0.25) | −2.75 (0.28) |
| cruise | −2.75 (0.67) | −2.47 (0.23) | −0.88 (0.04) |
| standby | −0.76 (0.03) | −2.58 (0.34) | −2.43 (0.39) |

**p̂ does encode the fall-off.** It is measured line by line on the pool, so it falls exactly as
the pooled spectrum falls: −3.1 dB/oct on DREGON and −2.8 on cruise. The fitted profiles fall
about as fast, but scatter around the trend by 6–7 dB. p̂ is not flat, and the latents do not
carry the fall-off (1(c)).

What p̂ gets wrong is its level. It is line plus floor, an upper bound, so the fit sits
7.5–10.8 dB under it on DREGON and cruise. The mean offset is −10.8 / −7.5 / −2.0 dB, with an sd
of 7.4 / 6.1 / 6.8 dB. Around a biased centre, the 10 dB sd is too wide to hold a line: all
352 DREGON lines together pay 299 nats in this prior.

### 1(c) The tilt test

The static part of v is the pool mean per line over every window and block. `fig_1c_static_tilt.png`
plots it.

| pool | static v vs log2 f: slope dB/oct (r²) | vs k: dB/order (r²) | per-rotor slopes (r²) | rms, residual sd dB | static u_j vs log2 f (r²) | static u |
|---|---|---|---|---|---|---|
| DREGON (k ≥ 9, σ_v > 0) | +0.38 (0.01) | +0.008 (0.00) | +0.02 / +0.39 / +0.95 / +0.18 (≤ 0.06) | 3.23, 2.97 | −0.94 (0.13) | −1.39 dB |
| cruise | +1.59 (0.34) | +0.104 (0.46) | +1.17 / +1.46 / +2.19 / +1.53 (0.14–0.56) | 3.60, 2.94 | −1.29 (0.36) | −2.71 dB |
| standby | +0.15 (0.01) | +0.003 (0.00) | +0.35 / +0.04 / +0.01 / +0.22 (≤ 0.05) | 2.16, 2.15 | −0.00 (0.00) | −1.46 dB |

- **DREGON and standby: no tilt.** The static v is a per-line scatter of 3.2 / 2.2 dB rms around
  zero (r² ≤ 0.06), not a fall-off the profile prior should have carried.
- **Cruise: a bend, not a fall-off.** Static v is −2.5 to −7.5 dB at 0.3–2 kHz and +2 to +9 dB
  above 3 kHz. The regression calls that +1.6 dB/oct (r² 0.34), with the wrong sign for "the
  latents carry the fall-off". Static u_j mirrors it: +2 to +7 dB below 200 Hz, then −5.7 / −14.9
  / −5.1 dB at 2.2 / 3.4 / 5.2 kHz. This is the floor/line swap of the runaway explainer. Above
  2 kHz the latents take power from the floor and give it to lines that the rig models 5–18 dB
  under its own floor.
- **Static u_j on DREGON** is a scatter of −3 to +5 dB, plus one notch: −21.3 dB at the 3.39 kHz
  control point, −6.6 dB at 5.2 kHz. That is no tilt (r² 0.13), and it is not smooth.

### 1(d) Prior against posterior, per rig block (round-2 selected fits)

The quadratic nats of each block are its −log prior without the normaliser. The rig −log prior
totals 5 137 / 4 737 / 4 589 nats and the OU total is 73 349 / 105 827 / 29 395 nats
(DREGON / cruise / standby).

| block | prior | DREGON | cruise | standby |
|---|---|---|---|---|
| γ/(0.01k) | HalfNormal(3) | median 3.02, p90 15.3, max 164; 34 % > 5; rms/scale 4.5; 3 559 nats | median 1.95, p90 19.4, max 75; 30 % > 5; rms/scale 4.6; 3 447 nats | median 3.97, p90 14.6, max 67; 42 % > 5; rms/scale 3.1; 2 447 nats |
| σ_ν | HalfNormal(0.6 rad/s) | 4.21 rad/s = 7.0 × scale; 24.7 nats | 3.82 = 6.4 ×; 20.3 nats | 0.35 = 0.6 ×; 0.2 nats |
| profile − p̂ | N(0, 10 dB) | mean −10.8, sd 7.4, range [−35, +8] dB; rms/scale 1.30; 299 nats | −7.5, 6.1, [−36, +5]; 0.96; 151 nats | −2.0, 6.8, [−16, +20]; 0.71; 131 nats |
| floor z | N(0, I), c = μ + σ_B L z | ‖z‖ 1.80; σ_B 17.6 dB; c − measured −18..0 dB; 1.6 nats | ‖z‖ 5.20; σ_B 3.9 dB; −12..+1 dB; 13.5 nats | ‖z‖ 1.63; σ_B 13.0 dB; −31..+9 dB; 1.3 nats |
| wind − measured | N(0, 6 dB) | −3.7..+3.8 dB; rms/scale 0.45; 0.8 nats | — | — |
| amp_exp | N(2, 1) | 0.82 (z −1.18); 0.7 nats | pinned 2 | 3.96 (z +1.96); 1.9 nats |
| floor_exp | LogN(ln 2, 0.5) | 7.32 (z +2.60); 3.4 nats | pinned 2 | 4.38 (z +1.57); 1.2 nats |
| floor_static_rel | LogN(−5.99, 1) | 2.2e-4 (z −2.43); 3.0 nats | pinned | 8.8e-5 (z −3.35); 5.6 nats |

The round-1 → round-2 move compares the same-seed restarts. Both fits run the identical round-0
rig step (latents at zero), and after it the rig does not move:

| pool | profile rms (max) move dB | log γ rms | σ_ν r1 → r2 | floor control values rms dB | wind rms dB |
|---|---|---|---|---|---|
| DREGON (s3) | 0.0014 (0.006) | 0.0014 | 4.75 → 4.21 | 1.49 | 0.007 |
| cruise (s1) | 0.0023 (0.018) | 0.0033 | 3.86 → 3.82 | 0.14 | — |
| standby (s2) | 0.0000 (0.0001) | 0.0000 | 0.348 → 0.348 | 0.011 | — |

Between rounds 1 and 2 the wander prior changed by a factor of 2–4 in σ², and 20 alternation
rounds ran. The profile and widths still moved by 0.001–0.02 dB. **The rig is the round-0 rig.**
The alternation's rig steps do not move it (§2).

What the table says per block:

- **Priors that never bind, with the fit far outside them.** σ_ν on DREGON and cruise is
  6.4–7.0 × its scale, for 20–25 nats. The γ tail reaches 67–164 × γ0 k, and 30–42 % of lines
  are over 5 γ0 k. γ is the costliest rig block (2.4–3.6k nats), but its tail is paid for, not
  prevented. `floor_exp` on DREGON is 7.3 (z +2.6).
- **Priors too flat to matter.** The profile prior pays 131–299 nats for 324–520 lines. Its sd
  of 10 dB is wider than the fitted scatter of 6–7 dB, and it is centred 7.5–10.8 dB above the
  fitted level. The floor z pays 1.3–13.5 nats. On DREGON and standby σ_B is 17.6 / 13.0 dB, so
  N(0, I) allows ±18 dB swings. The fitted floor sits up to 18 dB (DREGON) and 31 dB (standby)
  under the measured control values, at nearly zero cost.
- **The OU prior is the expensive side.** `v` costs 12.6k / 18.0k / 3.7k quadratic nats. The
  static part of every family is cheap in it: removing the pool means saves 12–16 % of the `v`
  nats (1 514 / 2 881 / 514), 15 / 35 / 3 % of `d` and 16 / 18 / 6 % of `u_j`.

### 1(e) The dB-wander Jensen question

`render` does **not** compensate. It draws fresh zero-mean OU tracks in dB and multiplies power by
10^{x/10}. So a family of sd s raises the mean power by exp((s ln10/10)²/2). With linear
interpolation between block knots, s² is taken as σ²(2/3 + ρ/3). The measured-σ numbers and
`experiments.noise_model.render.expected_periodogram` (the checks' "expectation") both use latents
at zero, the dB-median convention.

Table `ltas_bias_r2.json` gives band powers, in dB, of four spectra against the fit's own
block-mean spectrum: every window and block with its fitted latents, which is what the fit
explains of the real pool.

| pool | spectrum | 100–300 | 300–700 | 700–1.5k | 1.5–3k | 3–5k | 5–7k Hz |
|---|---|---:|---:|---:|---:|---:|---:|
| DREGON | render (fresh tracks) − fit | −0.05 | +0.39 | +0.89 | +1.80 | +1.30 | +1.83 |
| DREGON | latents at 0 − fit | −0.16 | −0.20 | −1.19 | −0.84 | −1.34 | −0.51 |
| DREGON | render + static part − fit | −0.02 | +0.29 | −0.08 | −0.10 | −0.35 | −0.03 |
| cruise | render − fit | +0.05 | +3.51 | +3.31 | +3.41 | +2.40 | +2.58 |
| cruise | latents at 0 − fit | −0.12 | −0.26 | +0.41 | +1.03 | +0.29 | +0.45 |
| cruise | render + static part − fit | +0.10 | +1.58 | +0.63 | +0.55 | +0.49 | +0.88 |
| cruise | render (RMS-matched) − fit | −1.40 | +2.05 | +1.85 | +1.96 | +0.95 | +1.13 |
| standby | render − fit | +3.61 | +2.19 | +1.77 | +1.54 | +3.04 | +2.04 |
| standby | render + static part − fit | +2.34 | +1.38 | +0.92 | −0.83 | +1.43 | +2.19 |

The Jensen factor of the fit's own wander is +0.35 dB on the lines at k 1–8 on DREGON (d only),
+2.1 / +2.9 / +2.5 dB at k 9–24 / 25–60 / 61+, and +1.5–2.4 dB on the floor. On Michael's
(mm1) it is +0.16 / +3.83 / +3.03 / +1.95 / +1.55 dB by order group (1–2 / 3–8 / 9–24 / 25–60
/ 61+) and +2.0–2.7 dB on the floor.

**The render is biased against what the fit explains**, by +0.4 to +1.8 dB (DREGON) and
+2.4 to +3.5 dB (cruise) above 300 Hz. The cause is not the missing mean correction on its own.
The fitted latents carry a static part, negative on average. Their pool mean cancels much of their
own Jensen term, so the latents-at-zero spectrum is within ±1.3 dB of the fit. The render never
draws that static part, then adds the full Jensen term of the prior σ on top of it. Folding the
static part back ("render + static part") brings DREGON within ±0.35 dB. Cruise and standby are
left with +0.1–2.3 dB, which is the prior σ (mm1) exceeding what the fitted tracks spread.

The RMS-matched cruise row (−1.4 / +2.05 / +1.85 / +1.96 / +0.95 / +1.13) reproduces most of
the Listen section's v3-minus-v2 LTAS excess on held-out FLY124
(−1.1 / +1.1 / +1.6 / +1.5 / +1.2 / +3.6). It explains 1.1 of the 3.6 dB at 5–7 kHz. It does not
explain the DREGON Listen deficit (−4.8 dB at 0.7–1.5 kHz, where this predicts +0.6 dB). That
deficit comes from the held-out window: the RMS matching is dominated by power below 100 Hz and
the wind term [inference].

**Decision: no render-side correction.** A mean correction in `render` would bake in the
unconverged convention of the r2 fits. The model's own convention is dB-median latents with a
zero-mean prior, and under it the render is right once the static part sits in the rig. The fix
is on the fit side: fold the static part into the rig (§4, the ridge step). For the r2 payloads
the same fold is an exact re-parametrisation, `scripts/noise_v3_diag.py fold` →
`results/noise_v3/diag/folded_r2/`:

- `profile_db += mean(d) + mean(v)`;
- `floor_mean_db += mean(u)` (the payload's μ, which the render reads);
- `z += L⁻¹ mean(u_j) / σ_B` (exact through L; ‖z‖ goes 1.8 → 168 on DREGON and 5.2 → 638 on
  cruise, because the static u_j notch is not smooth);
- latents minus the same pool means.

The fit's block spectra move by ≤ 8e-4 dB. ArmsV3 was told this at 01:10 (40 min budget from
00:47) and builds its v3 banks from `folded_r2`.
