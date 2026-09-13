# Revised-phase baseline instrumentation — proposals and round-2 resolutions

Status: **PROPOSED — baseline calibration has been run remotely; Main must still
approve before the manifest is frozen.** Round 1 built the instrument; round 2
(addenda 4-8) completed it against the ACTUAL candidate API and enforced every
frozen criterion. §1-§5 encode the proposal; §11-§18 record what round 2
resolved, with the numbers it measured; §19 records the actual `--prepare`
run on 2026-09-13.

Exact commands (the only two that matter for a freeze):

```
python scripts/stochastic_fit_revised_eval.py --manifest <frozen-manifest> --prepare
python scripts/stochastic_fit_revised_eval.py --manifest <frozen-manifest> --check
```

`--prepare` writes `<out_dir>/calibration.json`; `--check` writes
`<out_dir>/metrics.json` and `<out_dir>/gates.json` and exits nonzero on any
unmet gate. `--plan` writes `<out_dir>/plan.json`, `--verify-adapter` writes
`<out_dir>/adapter_verification.json`. `--mics N` limits the microphone set for
smoke runs only.

Artifacts:

| what | where |
|---|---|
| gate runner | `scripts/stochastic_fit_revised_eval.py` (`--plan` / `--prepare` / `--check` / `--verify-adapter`) |
| helpers | `src/experiments/stochastic_fit/revised_eval.py` |
| render AA + declared transfer | `src/experiments/stochastic_fit/stage2.py` (`antialias_filter`, `antialias`, `render_transfer_power`, `RENDER_TRANSFER_SPEC`, `render_from_export(normalize_rms=…)`) |
| proposed manifest | `docs/revised-phase-baseline-manifest-v1.json` |
| smoke manifests | `docs/revised-phase-baseline-manifest-smoke.json` (DREGON), `docs/revised-phase-baseline-manifest-smoke-michaels.json` (Michael) |
| tests | `tests/experiments/test_stochastic_fit_revised_eval.py`, `…_antialias.py`, `…_render_timebase.py` (73 tests, all passing) |
| smoke evidence | `results/revised_phase/smoke/{plan,calibration,metrics,gates,adapter_verification}.json`, `results/revised_phase/smoke_michaels/{plan,calibration,metrics,gates}.json` |

---

## 1. Nuisance aggregation (proposal)

**Rule.** For each export clip, first convert to **physical units** by folding in
that clip's own `scores.power_scale` (add `10·log10(scale)` dB to `profile_db`
and `floor_mean_db` — the two absolute levels; everything else is relative and
scale-free). Then average across clips:

* **linear power** (dB → linear → mean → dB) for `profile_db`,
  `floor_mean_db`, `floor_shape_db`, `mic_floor_db`, `mic_gain_db`,
  `gain_all_db`;
* **arithmetic** for the non-power quantities: `gamma0`, `gamma_slope`,
  `width_power`, `coherence_k_half`, `floor_tilt_db_oct`, `amp_exp`,
  `floor_exp`, `floor_static_rel`;
* the order ladder is truncated to the shortest clip's `n_harm` and the used
  count is recorded (`orders_used`);
* **clip-local latents are dropped, and the drop is recorded**: `h_db`,
  `floor_level_db`, `floor_tilt_gp`, `rps_offset`, `umod_db`, `carrier`. These
  are one window's posterior draw, not a rig property; carrying them onto
  another window would be a posterior replay, not a prediction.

**Why folding `power_scale` in matters.** The eight FLY125 cruise clips were
each fitted to a unit-mean periodogram with scales spanning 0.00291–0.00419
(−25.4 to −23.8 dB). Averaging the exported levels without folding the scales
in mixes three different unit systems. Measured on the real export: the folded
linear-power aggregate of `floor_mean_db` is **−32.648 dB**; a naive dB mean of
the unfolded values differs by design and has no physical meaning.

**Open point for Main.** `stage2.params_from_export` does **not** apply
`power_scale`, so the render path currently discards the fit's absolute level
entirely (and then `srn.synthesize(normalize_rms=0.1)` overwrites what is left
— see §6.2). I did not change that: it would move every existing number. The
predicted-`M` adapter does apply it, because `I/M` is meaningless otherwise.

## 2. Baseline ramp extrapolation (proposal)

No fitted ramp baseline exists and none is invented. Proposal, as encoded in
`baseline_map` of the manifest:

| regime | source | rule |
|---|---|---|
| standby | `results/S2/standby.json` | `match: aggregate`, `extrapolated: true`, labelled "FLY125 standby export applied to held-out FLY124 standby" |
| cruise | frozen cruise family | `match: aggregate`, `extrapolated: true`, labelled "FLY125 cruise export aggregate applied to FLY124 cruise" |
| **ramp** | the **same frozen cruise aggregate** | `match: aggregate`, `extrapolated: true`, labelled "FLY125 cruise export aggregate used for FLY124 RAMP — an explicit current-model extrapolation; no fitted ramp baseline exists" |

One fixed choice across the whole held-out recording; the runner **refuses** an
`aggregate` rule that does not carry both `extrapolated: true` and a label, so
an unlabelled extrapolation cannot slip through. DREGON uses
`match: identity` only.

**Escalation (needs a ruling).** The only standby export,
`results/S2/standby.json`, is **raw-era**: it has no `data` block, holds one
clip, and that clip is named `fly125_cruise_00` (the fit's `tag` was "cruise"
then). Its recording, start and rps-key therefore have to be *declared* in the
manifest, and the 1.78 s start I declared comes from `stage2`'s documented
single standby run, **not from the artifact**. Options: (a) accept the declared
provenance as-is, (b) re-fit standby on the refined label first, (c) drop
standby from the Michael gate. I recommend (b) if a fit is affordable, else (a)
with the declaration flagged in every output (`declared_provenance: true`,
which the runner already does).

## 3. Cohort and support split (proposal)

**DREGON (primary, cruise only).** Exactly the five no-source room2
recordings. Fit supports (one 16 s window each, from
`dregon_room2_cruise_refined.json`) are calibration. Held-out evaluation =
the first disjoint in-regime 8 s window per recording (`max_windows: 1`),
selected on the **same raw telemetry key used for scoring** so no refined label
enters the held-out path at all. Pairing is whole-recording: **5 clusters**.
room1 stays out of the primary gate.

Measured availability (`--plan`, all five recordings, `motors_command`,
min 65 rev/s): free-flight 5 in-regime windows / 3 disjoint; hovering 3 / 1.
The other three were not resolved in the smoke run; the full `--plan` will
report them and the runner **reports rather than pads** a shortfall.

**Michael's (held out, FLY124).** Standby / ramp / cruise, `max_windows: 2`
each, FLY125 is fit-only. Ramp windows are `min_seconds: 2.0`,
`stride_s: 1.0`, `45 ≤ rps ≤ 65`.

**Escalation (needs a ruling).** The frozen observation is a 16384-point
window (1.024 s at 16 kHz). A ramp window shorter than that carries **no
spectral frame at all**, so such a window is PIT-only and the runner records
why (`unavailable: …`). If FLY124's ramp is shorter than ~2 s, the Michael
composite/LTAS numbers for ramp do not exist; the MAE ratio still does. Main
should decide whether a PIT-only ramp is acceptable.

## 4. Manifest path and schema (proposal)

Path: **`docs/revised-phase-baseline-manifest-v1.json`** — committed, immutable
once frozen, versioned by filename; `--manifest` accepts any path, so Main may
relocate it. The runner records the manifest's **SHA-256** in every output and
`--check` **refuses** a calibration produced from a different manifest or a
different scorer checkpoint.

Schema (`"schema": "revised-phase-baseline/1"`), all keys required unless
marked:

```
scorer          {experiment, ckpt}                       → resolved path + SHA-256 recorded
observation     {sr, n_fft, hop, f_min, f_max}           → documentation; the code constants are frozen
gates           {alpha, bootstrap_seed, dregon_gap_fraction,
                 michaels_ratio_max, composite_tolerance}  → a missing key is a loud failure
null_variation  {rule, seeds[], metrics[]}               → ≥ 2 baseline render seeds
cohorts[]       {name, rig, role, gate, dataset, version, channels,
                 scoring_rps_key, recordings[],
                 family_selection{basis_recordings[], require_identity_coverage},
                 regimes[{regime, min_rps, max_rps,
                          evaluation{mode: auto|explicit, min_seconds, stride_s,
                                     max_windows | windows[]},
                          family?, select_family?}],
                 families{<family>{<regime>{path, provenance?}}},
                 baseline_map{<regime>{from_regime, match: identity|aggregate,
                                       extrapolated?, label?}}}
arms            {real, baseline, candidate?}             → kind: real | export_render | wav_dir
out_dir
```

`scoring_rps_key` must be raw telemetry; `rps_refined` and `auto` are refused
by `assert_raw_reference`. Measured: the five room2 flights publish
`motors_command` / `motors_command_raw` / `rps_refined` only (**no `rps`, no
`motors_measured`**), so their raw key is `motors_command`; FLY124/FLY125
publish `rps`.

## 5. Null-variation calibration (proposal — frozen before candidates, per addendum 3 §20)

**Rule.** `--prepare` renders the baseline arm with the manifest's seed list,
splits the seeds into two disjoint halves, pairs the halves per recording, and
takes the **one-sided upper bound of the clustered `|Δ|`** as that metric's
tolerance:

* **LTAS tolerance** — used by the baseline-calibrated LTAS gate.
* **PIT tolerance** — reported for information; the PIT gate's threshold is the
  0.70-gap point target, which needs no tolerance.
* **Composite tolerance** — stays at the manifest value (`0.0`). The composite
  score is a deterministic function of the export and the real clip: it has no
  render randomness, so its null variation is exactly zero.

Tolerances are written into `calibration.json` and read from there by
`--check`; a missing tolerance is a hard error, never a permissive default.
Measured in the smoke run (2 recordings, 2 seeds): LTAS `|Δ|` = 0.330 and
0.359 dB → tolerance **0.434 dB**; PIT `|Δ|` = 0.055 and 0.093 rev/s →
0.196 rev/s.

---

## 6. What the instrument measured, and three things Main should know

### 6.1 The DREGON real-vs-synthetic gap is positive (smoke scale)

Two recordings, one microphone, refined family, raw `motors_command`
reference, 8 s held-out windows, 2 render seeds:

| recording | real | baseline synth | gap |
|---|---:|---:|---:|
| free-flight_nosource_room2 | 0.8100 | 1.5362 | **+0.7262** |
| hovering_nosource_room2 | 0.9863 | 1.8402 | **+0.8539** |

Clustered mean gap 0.7900 rev/s, one-sided 95 % lower bound **0.3871 > 0**, so
the gap is distinguishable from zero and no escalation is triggered at this
scale. The full five-recording, eight-microphone number is a baseline job and
has not been run.

Family selection on calibration supports chose **refined** (composite
−806135 vs raw −799506 nats/s), and the raw family is also **ineligible**: it
names one of the cohort recordings, and modulo cycling does not exist in this
instrument.

### 6.2 The absolute LTAS error is dominated by the render's RMS normalization

Frozen addendum 3 §17 asks for absolute-level LTAS with no per-arm
normalization. Implemented — and on the real baseline it reads **12.4–14.4 dB**,
of which essentially all is a constant level offset: for
`hovering_nosource_room2`, `mean_abs_db = 12.381` and
`level_offset_db = −12.381` (every band low by the same amount), while the
shape-only secondary diagnostic is **0.843 dB**.

Cause, in the existing path: `render_from_export` → `srn.synthesize(…)` with
the default `normalize_rms=0.1`, so every synthetic arm is re-levelled to a
fixed RMS and the model's own absolute level never reaches the output (and was
already dropped by `params_from_export`, §1). Consequences:

* the gate still works as **non-regression**, because the offset is common to
  arms rendered through the same path;
* but the number is not a level-truthful error, and an arm that moves energy
  out of the LTAS bands can *improve* it: in the smoke check, the raw stand-in
  candidate read 5.70 dB against the baseline's 13.49 dB and **passed** the
  LTAS gate while failing PIT.

I did **not** change the render level: it would move every published number.
Proposal, for Main: either pass `normalize_rms=None` and apply the aggregated
physical level, or match the real clip's RMS per window, then re-freeze.

### 6.3 The predicted-`M` adapter agrees with its renderer to ~0.9 dB, Monte-Carlo limited

Required by addendum 2 §14. Frame-averaged, in-band (30–7900 Hz), FLY125
cruise aggregate, 8 s window, 2 microphones:

| comparison | level offset | shape mean abs | p95 |
|---|---:|---:|---:|
| adapter vs 6-seed render mean | −2.09 dB | **0.937 dB** | 3.08 dB |
| 3 seeds vs 3 other seeds (same params) | +0.06 dB | 0.772 dB | 1.91 dB |

The Monte-Carlo floor for the adapter comparison is ≈ 0.772/√2 ≈ 0.55 dB, so
there is a residual ~0.7 dB rms structural component. It is **the same in line
bins (0.939 dB) and floor bins (0.940 dB)** — i.e. no concentration at the
comb, which is what a wrong line-component structure would produce. Remaining
suspects, not chased: the renderer's `line_mode="fm"` atom placement, and the
44.1 kHz → AA → 16 kHz path the adapter does not model. The absolute level is
not comparable at all (§6.2).

The same check through the runner (`--verify-adapter`, DREGON free-flight 8 s
held-out window, 1 microphone, 3 seeds per half) gives the same picture for
both DREGON families, so this is a property of the adapter/renderer pair and
not of one export:

| family | level offset | shape mean abs | p95 | MC floor (3 v 3) |
|---|---:|---:|---:|---:|
| refined | −14.18 dB | 1.120 dB | 3.78 dB | 0.754 dB |
| raw | −7.10 dB | 1.028 dB | 3.44 dB | 0.763 dB |

---

## 7. Render anti-aliasing repair (done, measured)

`stage2.antialias` now designs the filter from an explicit specification
(`antialias_filter`: 7900 Hz pass / 8000 Hz stop / 100 dB stop /
**0.01 dB passband ripple**, Kaiser via `kaiserord`, 2829 taps, β = 10.06) and
applies it **once**, by centred overlap-add FFT convolution
(`scipy.signal.oaconvolve`, `mode="full"` sliced at the type-I group delay of
1414 samples). Measured on the actual application at 44.1 kHz:

| input | gain |
|---|---:|
| 100 / 1000 / 4000 / 7000 / 7800 / **7900** Hz | 0.000 dB (within the 0.01 dB ripple) |
| 7950 Hz (cutoff) | −6.02 dB — a single application, not `|H|²` |
| 8000 Hz | −100.4 dB |
| 9000 / 12000 Hz | −130.5 / −142.8 dB |

* an impulse returns at **the same sample index**, symmetric about it (zero
  phase);
* edges are a deterministic zero-extension, so **any** length works: a
  64-sample input returns 64 samples, bit-identical on repeat, whereas
  `filtfilt` raises `ValueError: The length of the input vector x must be
  greater than padlen, which is 8487`;
* cost is one FFT convolution instead of two direct passes over 2829 taps.

`clips.decimate` (the real-clip path) is untouched, no per-rotor mask was
reintroduced, and the claim is scoped to the stochastic-fit render path:
`stage2.render_from_export` and `stage1_bayes.render_from_export` are its only
callers.

## 8. Export/window pairing (done) and two legacy sites

Pairing is by `data.recordings` identity throughout the new instrument
(`ExportBundle.clips_of`, `baseline_params`, `candidate_params`): a recording
the export does not name yields **nothing** and raises, unless the manifest
declares a labelled extrapolation. Legacy exports without a `data` block must
have provenance declared, which is flagged in every output.

**Two pre-existing modulo sites are outside my ownership and were not edited**:
`scripts/_stage2_probe.py:149` (`entries[i % len(entries)]`) and
`scripts/_dregon_transfer.py:271` (`flight_fit["clips"].values()[i % …]`).
Both silently pair window *i* with clip *i mod n*. Say the word and I will fix
them.

## 9. Timebase (investigated, per section 5 — not modified)

`render_from_export` lines 366-370 convert the 16 kHz rotor track to the native
grid (`n_native = round(n16 / 16000 · 44100)` plus `np.interp`) and the
decimated output comes back at the requested duration. Confirmed by test, with
the renderer stubbed so the check is on the arithmetic, not on a render:
a 4 s / 64000-sample 16 kHz track drives the renderer with 176400 native
samples and returns 64000 samples at 16 kHz. The conversion is **correct and
unchanged**. The descriptive counterpart is also pinned: handing the function a
native-rate track produces a 2.76× too long clip, which is the documented
contract rather than a defect.

## 10. Conceptual complexity of the instrument (per the simplicity contract)

* New owned code: one helper module (~1500 lines with docstrings) and one CLI;
  no new abstraction layers, no new runner infrastructure, no new dependencies.
* Reused as-is: `_synthetic_probe.score` (zoo/HPPNet/PIT), `clips.*` loaders and
  decimator, `stage2.render_from_export`, `accept_stats.BANDS`,
  `model.CombSpectrum` as the predicted-`M` adapter, `data.periodogram` as the
  observation.
* Statistics: one estimator (a cluster mean), two bounds (t and cluster
  bootstrap, conservative side reported), no hierarchical model, no curvature,
  no nominal frame counts.
* The composite score is one expression over marginal terms plus an exposure
  weight; no dense STFT covariance anywhere.
* Runtime measured: `predicted_m` 21.6 s per 8 s window × 2 mics (frame-chunked
  to keep the `(rotor, order, frame, bin)` intermediates near 70 MB instead of
  ~1 GB); one 8 s × 2-mic render 6.6–7.1 s; the whole 2-recording, 1-mic smoke
  `--prepare` plus `--check` well under 20 minutes on the laptop.

---

# Round-2 resolutions (addenda 4-8)

## 11. `predicted_m`: option 1, labelled — the candidate uses the exact kernel

**Choice: addendum 4 §23 option 1, stated everywhere.** `predicted_m` stays the
LEGACY descriptive model's own forward law (`model.CombSpectrum`: frame-mean
carrier plus the known chirp-width covariate) and is labelled a
**historical-forward baseline diagnostic** in its docstring, in `__all__`
(`HISTORICAL_FORWARD_LABEL`), and as a field in every saved record that carries
a number from it (`model_source.observation_law`, `family_selection`,
`calibration.*.observation_law`, `adapter_verification.rows[].observation_law`).

Why not option 2 for the baseline: the shared kernel expects moving ATOMS, and
the legacy export has no atom representation — building one would replace the
old model's own forward law with a new one and the baseline would stop being the
baseline. What matters is that the CANDIDATE never goes through it: a
`model_family` export's spectrum comes from
`revised_phase.predict_spectrum(mode="prior")` and its audio from
`revised_phase.render_revised`, so the exact moving-window law is used wherever
the candidate is scored. The two laws are never described as the same
observation. Consequence Main should weigh: on the Michael RAMP the baseline's
spectral prediction is exactly the frame-mean approximation the contract
rejects for a candidate; the ramp composite comparison is therefore a
legacy-vs-candidate comparison of two different laws, and is labelled as such
in `metrics.json`.

Constants are now READ FROM `phase_kernel` (`N_FFT`, `HOP`, `SR`, `BAND_HZ`)
instead of redefined, and `stage2.SR` is asserted equal to `phase_kernel.SR` at
import, so the two adapters cannot drift.

## 12. Absolute level: the declared conversion, validated (addendum 4 §24, 7 §38, 8 §43)

`stage2.render_from_export` now takes `normalize_rms` and threads it to
`srn.synthesize`. **The default is `0.1`, exactly today's behaviour**, so
`_stage2_probe.py`, `_stage2_panels.py`, `_ab_render.py` and the training
streams are untouched. The evaluator passes `normalize_rms=None` explicitly on
every legacy arm — it is the only physical-units path.

**The declared conversion** (`revised_eval.to_renderer_units`), two effects
named separately:

1. **`normalize_rms`** — a common gain on the whole waveform. Measured on a
   planted clip: the default moved the tone integral and the floor by the same
   +11.1 dB (spread across tone and floor < 0.35 dB), i.e. it destroys absolute
   level and leaves line-to-floor ratios intact. Switched off, not compensated.
2. **The sample-rate term, `10·log10(native/sr) = +4.4032 dB`**, applied to
   `profile_db` and `floor_mean_db` (the two absolute levels) and to nothing
   else. Derivation: `I = |FFT(x·w)|²/Σw²` reads `PSD·sr/2`, so a floor shaped
   at 44.1 kHz reads `16000/44100` lower once decimated. In PHYSICAL terms the
   term belongs to the floor and not to a tone, but the legacy line parameter
   is not an amplitude — `CombSpectrum.line_power` is a band weight in
   periodogram·Hz (addendum 7 §38 conversion 1), an integral carrying the same
   `sr/2` — so in THIS parameterization it applies to both. The candidate's
   `profile_db` is a mean square (`A = √(2P)`) and must never receive it, which
   is why no level is ever transferred numerically between the families.

**Validation, planted and measured** (1 rotor, 40 orders at −10 dB, flat floor
at −40 dB, no speed law, all-coherent lines, `normalize_rms=None`, measured on
the frozen observation grid):

| quantity | as stored | term applied |
|---|---:|---:|
| floor mean, 4–6 kHz | −4.30 dB | **+0.10 dB** |
| line integral, order 5 | −4.38 dB | **+0.02 dB** |
| line integral, order 10 | −4.36 dB | **+0.05 dB** |
| line integral, order 20 | −4.27 dB | **+0.13 dB** |

Both tests are permanent (`test_stochastic_fit_render_timebase.py`).
**Legacy exported absolute levels are stale** without this conversion — every
published render they were checked against went through `normalize_rms=0.1` —
and `to_renderer_units` says so where they are consumed.

**Consequence on real data.** The absolute-level LTAS error of the DREGON
baseline fell from **12.4–14.4 dB (round 1, all of it a level offset) to
0.80–1.22 dB**, with the residual level offset now +1.19 dB instead of −12.38.
The addendum-3 §17 gate is now measuring a real quantity. It also flipped the
smoke decision the right way: the raw-era stand-in candidate, which PASSED the
round-1 LTAS gate at 5.70 dB against a 13.49 dB baseline (a laundering
artifact), now FAILS it (+0.58 dB against a 0.061 dB tolerance).

## 13. The declared render transfer, for the candidate's work grid

The candidate's landed `revised_phase.py` imports `RENDER_TRANSFER_SPEC` and
`render_transfer_power` from `stage2` — my file — so they are implemented
there: the power transfer of the render AA FIR (designed at the work rate)
times the UNCHANGED `resample_poly` decimator, both normalized to unit DC gain,
evaluated on the analysis grid.

Measured against a tone actually pushed through `antialias` + `clips.decimate`
at 64 → 16 kHz (permanent test):

| frequency | predicted | measured | difference |
|---|---:|---:|---:|
| 1000 Hz | +0.009 dB | +0.009 dB | 0.000 dB |
| 4000 Hz | +0.011 dB | +0.011 dB | 0.000 dB |
| 7000 Hz | −0.253 dB | −0.253 dB | 0.000 dB |
| 7500 Hz | −1.845 dB | −1.845 dB | 0.000 dB |
| 7900 Hz | −4.916 dB | −4.912 dB | +0.004 dB |

Attribution confirmed: at 7900 Hz the AA contributes ≤ 0.01 dB (its ripple
spec) and the whole −4.9 dB is `resample_poly`'s own rolloff. The legacy arm
does NOT get this multiplier, and must not: real clips and legacy renders both
pass through the same 44.1 → 16 kHz `clips.decimate`, so the fitted legacy
parameters already absorbed that chain.

Cross-worker note for Main: this is a NEW symbol pair in `stage2.py`, which I
own, added because the candidate's landed file imports it and would otherwise
not import at all (`ImportError: cannot import name 'RENDER_TRANSFER_SPEC'`,
hit on the first michaels smoke run). The decimator's own FIR is reproduced
from `resample_poly`'s documented default design because scipy exposes no
accessor for it; the measured-vs-predicted test above is what would catch a
scipy change, rather than a silent mismatch.

## 14. Exposure is now per ANALYSIS FRAME (addendum 5)

**Defect confirmed and fixed.** Round 1 counted exposure over whole 8 s clip
spans, so all ~110 overlapping analysis frames in a clip had weight 1 and
duplicating a single frame entry doubled its contribution.

**The convention now implemented**, and recorded in every composite block as
`exposure`:

* an observation is identified by **`(recording, absolute frame centre,
  n_fft)`**, the centre quantized to 0.1 ms (float64 resolves ~0.2 µs on
  DREGON's epoch clock, and genuine frames are a 64 ms hop apart);
* every frame carries the explicit **hop/window factor `hop / n_fft` = 1/16**;
* identical copies **split one unit of exposure** between them;
* the normalizer is the **unique observed support**: each unique frame owns the
  `hop` samples its cadence advances by, so `n_unique · hop / sr` seconds.

Properties, all three regression-tested: duplicating one frame entry changes
nothing; duplicating a whole clip changes nothing; the same material at half
the hop density gives the same risk per unique second (`per_frame · sr / n_fft`)
instead of twice as much. The stale prose claiming `HOP == N_FFT` and
window-level bookkeeping is corrected in the module docstring and in
`window_periodogram`. A fixed hop is recorded as a **decision-risk setting**,
never as information content, and the fit temperature is not touched here.

Numerically this rescaled the composite risk (DREGON baseline −767 050 → −54 478
nats/s of unique support); it is an absolute paired quantity, so only the
difference between arms enters a gate.

## 15. Regime supports: score the declared interval only (addendum 4 §26)

`regime_support` builds the scored sample mask from the RAW telemetry and the
manifest's frozen band; `pit_mae` averages only the output frames whose centres
lie inside it (reusing the tracker's own `align_corners=False` grid alignment
via `output_frame_centres`), and the composite keeps only the analysis frames
whose centres lie inside it. The mask's scored seconds, scored fraction,
contiguous intervals on the recording clock and the scored frame centres are
all saved as provenance. A support that selects no output frame RAISES rather
than falling back to a whole-window average.

**Measured fact that forced a second window mode.** FLY124's entire
45–65 rev/s ramp band holds **1.24 s of material, the longest contiguous run
0.99 s** — shorter than one 1.024 s analysis window. A window that is
in-regime throughout therefore does not exist, so `evaluation.mode: "support"`
selects a CONTEXT window (8 s, carrying the render, the analysis window and the
filter guard) centred on the pre-registered interval, while the scored mask
stays that interval. The smoke run scores FLY124's ramp on 0.99 s inside an 8 s
context window at 27.68 s. Two consequences Main should rule on:

* the ramp composite is a **centre-in-support** quantity: at NFFT 16384 each
  kept frame's window necessarily extends ±0.512 s into context. That is
  unavoidable at the frozen front end; the alternative is no spectral ramp
  number at all. It is labelled in the record;
* the ramp PIT MAE is read on ~31 output frames. It is a genuine ramp number,
  not a mostly-cruise average — the whole point of §26.

## 16. The seven instrument-review findings (addenda 6 and 8)

| # | finding | resolution |
|---|---|---|
| 1 | `--check` crashed on a candidate export | `export_family` dispatches on `model_family`. `read_export` refuses a revised file, `read_candidate_export` refuses a legacy one — each door rejects the other's file. A `revised_export` arm calls `render_revised` / `predict_spectrum(mode="prior")`, and its `physical_rps` is consumed for the **diagnostic-only** physical-shaft MAE (reported per window with `role: DIAGNOSTIC ONLY`, on the scored support, never a gate). `reference_rps` (raw telemetry) stays the primary target for every arm. |
| 2 | DREGON gate passed on partial cohorts | `dregon_pit_gate` takes `required_recordings` and fails `cohort_complete` before any bootstrap; a separate `cohort_completeness_gate` fails on any missing recording, any unexpected one, and any `SupportReport(sufficient=False)`. No intersection shrink. Regression-tested with two excellent clusters against the frozen five. |
| 3 | Michael's missing its §19 gates; `None` tolerance | added `michaels_fly124_composite_non_regression` and `michaels_fly124_absolute_ltas_non_regression` (`conditional_non_regression_gate`: mean delta AND one-sided upper bound within tolerance, explicitly conditional on FLY124). Null variation is now computed at recording level AND at block level, and the block-level tolerance is **pooled over every held-out block of the cohort across regimes** — a per-regime pool of one window yields nothing. A `None` tolerance produces a FAILED gate (`missing_tolerance_gate`), never a pass; regression-tested for all three gates. |
| 4 | `pit_mae` truncated to the shorter timeline | it now requires the exact frozen sample length (`sample_tolerance` defaults to 0 and any allowance is recorded) and the exact frozen microphone set, and RAISES otherwise. Regression-tested for a short arm, a narrow arm, and the stated tolerance. |
| 5 | no leakage check on candidate provenance | `candidate_leakage_guard` runs BEFORE the arm is measured: a revised export's `training_provenance.clips` are parsed and its `manifest_sha256` must equal the manifest's `fit_manifest_sha256` (absent → loud exit); a legacy or `wav_dir` candidate must declare its training clips or explicitly declare `no_training_supports`. Overlap with any scored support is a hard exit. The guard widens training spans by `training_guard_seconds` (default one analysis window) so an analysis/filter support that reaches into a scored window also counts. |
| 6 | exports and dataset versions not pinned | `frozen_inputs` records the manifest digest, the scorer checkpoint digest, **every export's SHA-256** and **every RESOLVED dataset version** (`dload`'s own resolution, not the manifest's `null`), and `--check` fails if any of them moved. |
| 7 | candidate could choose its own seeds | the `candidate.seeds` override is rejected: `--check` uses the frozen `null_variation.seeds` for every arm, and a declared list must equal it exactly. |

The dispatch is proven by execution, not by inspection: a hand-written
`model_family` export is driven through the runner's own `ArmModel` and reaches
the real `revised_phase.render_revised` (audio `(2, T)`, `physical_rps`
`(1, T)`, `eps_in_physical_rps: false`) and
`revised_phase.predict_spectrum(mode="prior")` on the frozen 16384/1024 grid,
with the shape asserted against `data.periodogram`'s own
(`test_the_candidate_arm_dispatches_to_the_real_revised_api`). The leakage
guard's fit-manifest pin and its three refusals are tested the same way
(`test_the_candidate_leakage_guard_pins_the_fit_manifest`).

## 17. Round-2 smoke evidence (instrument tests, not results)

**DREGON** (`docs/revised-phase-baseline-manifest-smoke.json`, 2 recordings,
1 mic, raw `motors_command`, 8 s held-out windows, frozen seeds 2001/2002):

| recording | real | baseline | candidate (raw-era stand-in) | gap |
|---|---:|---:|---:|---:|
| free-flight_nosource_room2 | 0.8100 | 1.5221 | 1.7134 | +0.7121 |
| hovering_nosource_room2 | 0.9863 | 1.8051 | 1.9587 | +0.8188 |

Clustered mean gap 0.765, one-sided lower bound **0.4287 > 0**. Baseline
absolute LTAS 1.22 / 0.80 dB; composite −54 478 / −53 318 nats/s; LTAS
null-variation tolerance 0.0608 dB. `--check` exits **1** with
`dregon_cruise_pit_mae`, `dregon_held_out_composite_delta` and
`dregon_baseline_calibrated_ltas` unmet and `dregon_room2_cruise_cohort_complete`
met.

**Michael** (`docs/revised-phase-baseline-manifest-smoke-michaels.json`, FLY124,
1 mic, one held-out block per regime, frozen seeds 2001/2002):

| regime | real | baseline | candidate | per-regime ratio |
|---|---:|---:|---:|---:|
| standby (8 s) | 0.2572 | 0.2702 | 8.4232 | 31.17 |
| ramp (0.99 s in an 8 s context) | 3.1790 | 3.9379 | 4.2812 | 1.087 |
| cruise (8 s) | 0.8316 | 1.1137 | 1.2326 | 1.107 |

Frozen arithmetic: `E_rig_baseline = 1.7739`, `E_rig_candidate = 4.6457`,
**aggregate ratio 2.6189 > 1.05 → fail**. The rejected mean-of-ratios form
would have read 11.12 — the standby-leverage failure mode addendum 8 §22 warns
about, visible in the same record. The conditional composite gate fails
(mean delta +65 265 nats/s at tolerance 0) and the conditional absolute-LTAS
gate fails (mean delta +6.30 dB at the pooled FLY124 block tolerance
0.1722 dB). `--check` exits **1**; the cohort-completeness gate passes.

**Adapter verification in physical units** (`--verify-adapter`, DREGON
free-flight 8 s held-out window, 1 mic, 3 seeds per half):

| family | level offset | shape mean abs | Monte-Carlo floor |
|---|---:|---:|---:|
| refined | **−0.41 dB** (round 1: −14.18) | 1.120 dB | 0.754 dB |
| raw | **−0.45 dB** (round 1: −7.10) | 1.028 dB | 0.763 dB |

With the declared conversion the legacy adapter and its own renderer now agree
in ABSOLUTE level to ~0.4 dB. The shape residual is unchanged (~1.0–1.1 dB
against a comparable MC floor of ~0.53 dB), still equal in line and floor bins,
so it remains a window/commutation-class residual and not a level or
line-structure error.

## 18. Main rulings applied (addenda 9-13)

1. **Standby provenance (§2)** — ACCEPTED as a conservative envelope: the whole
   known FLY125 standby event `[1.78, 15.28] s` (13.5 s) is declared in the
   manifest. The exact start is explicitly unknown; no refit or drop is forced.
   Candidate training supports must be non-overlapping with the scored supports.
2. **FLY124 ramp (§15)** — ACCEPTED as ONE fixed-event point benchmark: the
   longest contiguous 45-65 rev/s interval (~0.99 s) inside an 8 s context
   window centred at 27.68 s. The scored support is the pre-registered ramp
   interval; there are no fake independent ramps. The Michael ratio uses equal
   per-regime weight (mean of the three regime MAEs per arm, then ratio ≤ 1.05).
3. **Two observation laws (§11)** — ACCEPTED: the legacy arm uses the historical
   `model.CombSpectrum` forward law, labelled as such; the candidate arm uses
   `revised_phase.predict_spectrum(mode='prior')`. They share only the STFT
   front-end, the band and the units.
4. **Michael `gap_positive`** — INAPPLICABLE: the Michael cohort is one recording,
   so the DREGON-style positive-gap check is not used; the ratio gate and
   conditional non-regression gates are the binding Michael checks.

## 19. Composite temperature ``T = J / H`` (addendum 9 §45)

**Rule.** On the calibration supports of each rig, render ``B`` predictive draws
of the frozen legacy baseline through the same physical render path used for
scoring. For each draw compute the gain-direction score
``U_b = sum_i a_i (1 - I_b / M_i)`` with the actual unnormalized composite-risk
weights ``a_i = hop / n_fft`` (duplicate-split). Then ``J = sampleVar_b(U_b)``,
``H = sum_i a_i``, and ``T = J / H``. The fit divides: ``L / T``.

**Manifest pin.** ``composite_temperature.master_seed = 9001``,
``composite_temperature.B = 16``. These are internal predictive MC draws, not
model-training seeds. The calibration clip list is the actual fit-support set
resolved by ``--plan`` for the chosen family.

**Failure mode.** If ``J <= 0`` or non-finite, the calibration fails and no
value is frozen. There is no clamp and no fallback.

**Status.** Implemented in ``run_prepare`` and recorded per regime in
``calibration.json``. The actual numeric ``T`` must be computed remotely (the
rendering is too heavy for the laptop) and returned to Main before any
candidate fit. It is NOT tuned after seeing candidate results.

## 20. Held-out duration decision for the five DREGON recordings (addendum 13)

A ``--plan`` run on the full cohort with the frozen fit supports reports the
following largest disjoint cruise support per recording after excluding the
16 s fit window:

| recording | largest disjoint cruise support |
|---|---:|
| free-flight_nosource_room2 | 40.0 s |
| hovering_nosource_room2 | 12.0 s |
| updown_nosource_room2 | **4.0 s** |
| rectangle_nosource_room2 | 8.0 s |
| spinning_nosource_room2 | **4.0 s** |

**Decision:** freeze the scored duration at **4.0 s**, the largest common value
satisfied by all five recordings. The 8 s proposal is unavailable for the full
cohort. No recording is dropped, padded or repeated. The manifest is updated
accordingly.

## 21. Still open / still needing Main

1. **Legacy modulo sites** — `scripts/_stage2_probe.py:149` and
   `scripts/_dregon_transfer.py:271` still pair window *i* with clip *i mod n*`.
   Still outside my ownership, still unedited.
2. **Composite temperature value** — the actual ``T`` for each rig must be
   computed remotely and approved by Main before the candidate fit.
3. **Candidate training-support audit** — the non-overlapping standby/ramp/cruise
   supports for the candidate fit (e.g. FLY125 `[2,10]`, `[10,18]`, `[32,48]`)
   must be confirmed by the candidate owner before freeze.

---

## 19. Actual remote `--prepare` run (2026-09-13)

Command:

```bash
omnirun submit --backend uni-cpu --gpus 0 --time 4h --mem 64 --outputs results/revised_phase/baseline_v1 -- \
  bash -c 'mkdir -p results && curl -fsSL "https://d064390efe0e59d764d4f701b59d7b71.r2.cloudflarestorage.com/ml-data-new/revised-phase/s2-baseline-50796d9a.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1af5a5336e9feaec0646a6394a4b6a42%2F20260913%2Fauto%2Fs3%2Faws4_request&X-Amz-Date=20260913T015842Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=730b39e1f27e94df9fa980f76b701443556fb712991432a38a6c97b7c183498c" | tar xzf - -C results && PYTHONPATH=$PWD/src python scripts/stochastic_fit_revised_eval.py --manifest docs/revised-phase-baseline-manifest-v1.json --prepare --out results/revised_phase/baseline_v1'
```

Job ID: `bash-0e3e78` (uni-cpu, 64 GB RAM, exit 0, ~1.3 h wall time).

Input capsule: `s3://ml-data-new/revised-phase/s2-baseline-50796d9a.tar.gz` (SHA-256 of tar: `50796d9a…`).

Output: `results/revised_phase/baseline_v1/calibration.json` (606 KB).

Selected calibrated quantities:

| cohort | regime | composite T | B | master_seed | pit_tolerance | ltas_tolerance_db | note |
|---|---|---:|---:|---:|---:|---:|---|
| dregon_room2_cruise | cruise | 5.192 | 16 | 9001 | 0.0729 | 0.0535 | 5-recording cluster bootstrap |
| michaels_fly124 | standby | 15.214 | 16 | 9001 | — | — | insufficient clusters (only 2 windows) |
| michaels_fly124 | ramp | 15.214 | 16 | 9001 | 0.4004 | 0.1295 | 1 fixed-event window, cruise-extrapolated baseline |
| michaels_fly124 | cruise | 15.214 | 16 | 9001 | — | — | insufficient clusters (only 2 windows) |

Import provenance verified: all modules resolved under the pushed worktree
`.trees/5d51334311ae/src/experiments/stochastic_fit/`.

The manifest remains `PROPOSED`; freeze requires Main approval of the
4 s DREGON held-out duration, the FLY124 ramp fixed-event point, the standby
declared provenance, and the per-cohort composite temperature rule.
