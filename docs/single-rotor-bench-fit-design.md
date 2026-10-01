# Single-rotor bench fit (`bench_v4`) — design

Status: design, 2026-10-01. Not implemented. Target: fit the single-rotor
noise model found in `docs/experiments/static-rig-profiles.md` § Round 3 to
one stationary 8-mic recording with the noise-model stack
(`src/experiments/noise_model/`), so that the same machinery later fits
four-rotor benches and flights. Acceptance (set by Dmitrii): reproduce the
quick fits (`results/static_rig/single_rotor/{ou,amplitude_jitter.json}`) on
the 16 recordings where they held, do not fail on the 4 where they did
(M1_90, M2_80, M3_90, M4_90), fit the amplitude-modulation envelopes per
order jointly across mics, and degrade gracefully on wind-corrupted mics.

## 1. What the stack already has (map: `agent://NoiseModelStackMap`, 2026-10-01)

- `mode='bench'`: ONE whole-span periodogram per mic at 16 kHz (`supports.py`,
  `bench_batch` `model.py:919-1015`), carrier frozen by `refine_carrier`,
  Whittle risk over all bins and mics, `k_max_for_carrier`.
- Line lag law `lag.r_tau` (`data_processing/noise_model/lag.py:100-140`):
  `exp(−k² V_ν(τ)/2 − 2π γ_rk |τ|)` with `V_ν` the integrated-OU structure
  function — the OU shaft **is** the stack's law (σ_ν in rad/s, λ in 1/s);
  the per-line Wiener phase `γ_rk` is the extra.
- Floor: `mean + spline(14 control points, SE-GP) + 0·tilt`, measured `σ_B`;
  wind: `wind_db (M,)` × fixed `wind_shape(f)` (flat → 100 Hz, −12 dB/oct to
  400, cosine to 0 at 500 Hz) added to the floor.
- Mics: v2 scalar `mic_line_gain_db`, `mic_floor_db`; v3 none (channels
  normalised in the data by measured gains / per-band transfer).
- Fit: AutoDelta MAP, Adam 1500 → L-BFGS polish, restarts jitter the dynamics
  block; priors are `PriorsV3` subclasses registered in `PRIOR_SETS`.
- Render (`data_processing/noise_model/render.py`) draws the OU shaft and the
  Wiener phases, no amplitude modulation; `identifiability.LineShapes`
  tabulates the per-order shape from the same law.

## 2. Model (per recording, R = 1; nothing R-specific)

Expected periodogram of mic m at bin f:

    S_m(f) = Σ_k a_mk · [ K ⊛ L_k ](f − k s)  +  F(f) · E_m(f)  +  W_m(f)  +  N_m(f)

| term | parametrisation | sites / constants |
|---|---|---|
| shaft | `V_ν(τ)` of the OU speed error | `sigma_nu ~ LogNormal(log 0.4 rad/s, 0.5)`, `lam ~ LogNormal(log 12, 0.5)` (quick-fit population: σ_ν 0.04–0.09 rev/s, λ 9–15 s⁻¹) |
| per-line Wiener phase | `γ_rk` | **pinned 0** (the width law needs none; a variant frees it with `HalfNormal(0.3 Hz)` as a check) |
| AM pedestal | lag factor `exp(σ²_mk (e^{−2π γ_mk |τ|} − 1))` multiplying `r_tau` — the log-OU envelope's autocorrelation, = `L_γk + σ²_m L_(γk+γm)` to first order; **per order, shared by the mics** | `am_sigma2 (K,) ~ LogNormal(log 0.05, 1.5)`, `am_gamma_hz (K,) ~ LogNormal(log 0.5, 1.0)`; the direct measurement seeds them |
| line powers | `a_mk = 10^{(p_k + δ_mk)/10}`: rotor profile + per-(mic, order) deviation | `profile_db (K,) ~ N(measured peak level, 10)`, `mic_dev_db (M, K) ~ N(0, 2.4 dB)` (the measured per-run near-field scatter; its sd is a constant) |
| pair EQ | `E_m(f)`: measured broad per-(rotor, mic) transfer (`mic_variance.py`, W = 200 Hz), **constant**, applied to lines (inside `a_mk` via the data normalisation, as round 4's per-band transfer) and to the rotor floor | constant |
| rotor floor | `F(f)`: `floor_mean_db + spline`, measured σ_B | `floor_shape_z (14,) ~ N(0, I)` |
| ambient | `N_m(f)`: Welch PSD of the pre-spin-up (rotor-off) segment per mic, smoothed, **constant**; a new measurement in `supports.py` (the stack has no rotor-off marker) | constant |
| wind | `W_m(f)`: `wind_db (M,)` × `wind_shape` | `wind_db ~ N(measured low-band excess, 6)`; windy mics also get `mic_dev_db` sd widened to 6 dB below k = 8 so corrupted low orders do not pull `profile_db` |
| carrier | `s` | frozen by `refine_carrier` from the TABLE value |

Likelihood: the stack's Whittle risk over all bins 30 Hz – 7.9 kHz and all
mics, temperature 1. Resolution 0.03 Hz on 30 s, so the AM pedestals
(γ_m 0.2–1 Hz) and the OU shaft cores are resolved; orders up to
`k_max_for_carrier` (≈ 115 at 70 rev/s) at 16 kHz.

Why these choices: the AM pedestal enters the existing kernel as one more
multiplicative lag factor, so `expected_periodogram_from_atoms` is unchanged;
`profile_db + mic_dev_db` is the measured structure (mean profile + 2.4 dB
per-line pair scatter) and keeps weak lines regularised; the ambient as a
constant is the only honest floor on a rig whose gaps sit 0–6 dB above the
room; the pair EQ as a constant is what explains 45 % of the mic-to-mic
spread and nothing in the fit can absorb it otherwise.

## 3. Changes, by file

1. `data_processing/noise_model/lag.py`: `r_tau(..., am_sigma2=0, am_gamma_hz=0)`
   — the AM factor, default off (every existing call unchanged);
   `order_lag_support_s` accounts for it.
2. `experiments/noise_model/model.py`: `PriorsBench4(PriorsV3)` with the
   sites above; `free_blocks('bench_v4')`; `_sample_params_bench4`;
   `params_to_dict` gains `am.{sigma2, gamma_hz}`, `profile.mic_dev_db`,
   `ambient` (file reference), `pair_eq`.
3. `experiments/noise_model/spectrum.py`: bench forward model takes the AM
   pair through `r_tau`; additive `ambient (M, F)` after the floor; pair EQ
   applied as the round-4 transfer path.
4. `experiments/noise_model/supports.py`: rotor-off segment (pre-spin-up:
   everything before the strict span minus 0.5 s) → smoothed Welch PSD per
   mic; pair EQ loader (`results/static_rig/single_rotor/pair_eq.json`,
   produced once by a scratch script from the quick fits).
5. `experiments/noise_model/fit.py`: `_seeds_bench4` — σ_ν, λ from the
   width-law quick fit (or the prior median when it failed), AM from
   `amplitude_jitter.json`, profile from the OU peak reading, wind centres as
   v3; identifiability gate as v3.
6. `render.py`: draw the per-order log-OU envelope (shared by mics) when a
   record carries `am` — the single-rotor synthesis of
   `experiments/static_rig/single_rotor.synthesise` already does this; port
   the hop-rate AR(1) + linear blend.
7. `identifiability.LineShapes`: shape tabulation through the extended law.
8. CLI: `scripts/noise_v2_fit.py bench --mode bench_v4 --priors bench4
   --support dregon:motor_Motor1_70`; tests under `tests/experiments/noise_model/`
   for the lag factor (limit σ²→0 reproduces `r_tau`; first-order pedestal
   equals the two-Lorentzian shape) and for the ambient term.

## 4. Validation

- Synthetic: `single_rotor.synthesise` with known (σ_ν, λ, AM, profile) +
  recorded ambient → fit recovers σ_ν, λ within 20 %, AM σ² within 30 % on
  lines ≥ 15 dB over the floor, profile within 1 dB.
- The 16 good recordings: |ln σ_ν − ln σ_ν,quick| ≤ 0.3, λ within ×1.5,
  profile vs OU peak reading within 2 dB (median over lines ≥ 10 dB SNR).
- The 4 failed ones: λ stays in 5–30 s⁻¹ (no random-walk corner), HWHM at
  k = 80 within ×1.5 of the free per-order width.
- Wind: on M1 (ch5, 6 windy) the profile from the 6 clean mics changes by
  < 0.5 dB whether the windy mics are included or not.
- Cost: ≤ 10 min per recording on CPU (115 line kernels × 480 k lags per
  forward); GPU via `uni-gpushort` otherwise.

## 5. Open points

- Whether `mic_dev_db` should carry the measured same-parity neighbour
  correlation (ρ ≈ 0.25) — left independent in v1.
- The 2.3 kHz humps and the resolution-sharp spike atop the k ≈ 20 hump
  (static-rig doc § width law) are outside this model; the fit's residual
  map per order will show whether they matter.
- The decimation to 16 kHz drops orders 116–150 that the quick fits read; the
  bench profile stored for the generator keeps the quick-fit values there.
