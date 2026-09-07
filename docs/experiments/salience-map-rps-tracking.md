# Salience-Map Multi-F0 Tracking for RPS Prediction

**Status:** done | **Dates:** 2026-06-12 to 2026-06-15 | **Full report:** writing/reports/2026-06-15/ (run `make` for the PDF)

## Motivation

Direct RPS regression (SimpleConv / SimpleConvV2 + PIT loss) requires task-specific
architecture and training. This experiment asked whether an off-the-shelf *multi-pitch*
approach could match it with less engineering: run a multi-F0 salience-map model
(per-frequency-bin activation) over the drone audio, peak-pick the four rotor
fundamentals per frame, and track them into continuous RPS trajectories via Hungarian
(optimal bipartite) assignment. If salience models can reliably expose the four rotor
fundamentals, tracking should recover RPS "for free" without a custom regression head.

Two model families were adapted to emit salience: **LateDeep** (CNN over an HCQT
front-end, `multif0_salience`) and **Basic Pitch** (contour branch, `basic_pitch_salience`),
both trained with BCE against binary salience targets derived from GT RPS, native 16 kHz
throughout for comparability with SimpleConv.

## Results

**Tracking algorithm validation (GT round-trip, DREGON-LM-V4 valid, 16 samples ≥33 Hz):**
Hungarian tracking clearly beats greedy nearest-neighbor — per-frame PIT 1.19 Hz vs
2.78 Hz, global PIT 1.67 Hz vs 6.83 Hz. Irreducible quantization floor (per-frame PIT
directly on salience bins, no tracking) is 0.27 Hz.

**Baseline salience models on `DREGON-LM-V4/valid`** (30 clips × 8 channels, PIT eval,
`track_threshold=0.3`):

| Model | RMSE (Hz) | MAE frame (Hz) | R² |
|---|---|---|---|
| SimpleConvV2 (8ch, regression) | 1.62 | 1.08 | 0.93 |
| SimpleConv (8ch, regression) | 3.55 | — | 0.68 |
| `multif0_salience` (LateDeep) | 6.30 | 3.40 | 0.19 |
| `multif0_salience_fastest` (stacked HCQT) | 6.42 | 3.58 | 0.11 |
| `basic_pitch_salience` | 23.24 | 16.19 | −16.21 |

LateDeep is the best salience model but ~4× worse than SimpleConvV2 in RMSE; Basic
Pitch fails catastrophically (diffuse salience, wrong-octave locking). A round-trip
experiment (perfect binary targets → same tracking pipeline) put the coarse-grid
resolution floor at 2.5–3.0 Hz RMSE — worse than SimpleConvV2's *total* error, explaining
much of LateDeep's gap.

**Root cause identified:** on DREGON-LM-V4, rotor fundamentals cluster tightly
(p1–p99 = 69–89 Hz) and 55% of frames have two rotors <1 Hz apart — below the ~0.9 Hz
bin spacing of the coarse grids — so trajectories collapse to their mean.

**Narrow-band + super-resolution follow-on:** concentrated the HCQT input in the rotor
band (`fmin=55`, 1 octave, harmonics 1–4, 120 bins, 55–110 Hz) and added a
`FreqSuperResHead` that resamples onto a fine *linear* 360-bin output grid
(≈0.153 Hz/bin, 55–110 Hz), trained end-to-end with BCE. Results on the same
validation set:

| Model | RMSE (Hz) | MAE frame (Hz) | R² |
|---|---|---|---|
| `multif0_salience_narrow_sr` | **4.03** (was 6.30) | 2.34 (was 3.40) | **0.573** (was 0.19) |
| `basic_pitch_narrow_sr` | 11.66 (was 23.24) | — | −3.24 (was −16.21) |

`multif0_salience_narrow_sr` now sits *below* the old coarse-grid resolution floor
(2.5–3.0 Hz), confirming the finer output grid genuinely buys localization rather than
confident-but-wrong peaks. Per-rotor MAE spread compressed from 4.5 Hz (1.2–5.5 Hz
across rotors) to 1.8–2.9 Hz — the near-unison rotor collapse is fixed. Basic Pitch
improves but stays unusable (still negative R²).

## Conclusion

Narrow-band input + super-resolution output is the viable salience-tracking recipe:
it removes the rotor-collapse failure mode and narrows the gap to SimpleConvV2 from
~4× to ~2.5× in RMSE (4.03 Hz vs 1.62 Hz). The salience-map paradigm is not
fundamentally unsuited to closely-spaced rotor fundamentals — the original baselines
were simply mis-resolved. Basic Pitch remains unusable at any grid resolution
(designed for discrete musical notes, not continuous low-frequency F0 tracking).

SimpleConvV2 (direct regression + PIT) remains the recommended model going forward —
even the improved LateDeep variant doesn't surpass it, suggesting that after fixing
output resolution the salience approach is now limited by architectural capacity
(purely convolutional, same limitation as the simplest regression baseline) rather than
representation. Future investment should go to the regression family, not further
salience-map engineering.

## Implementation notes (moved from `src/models/AGENTS.md`, 2026-09-07)

`multif0_salience` and `basic_pitch_salience` (`src/models/salience_rps.py`)
output per-bin salience **logits** `(B, n_bins, T)` (flagged
`outputs_salience=True`), not RPS. The unified trainer routes them to a BCE path
by loss selection — a `conf/loss` entry that targets salience
(`src/losses/salience.py`) instead of the PIT-MSE loss of the direct-RPS
models. `rps_to_salience()` builds the per-bin target (precomputed/cached in
the dataset; blurred via a blur-bins parameter), trained with
`BCEWithLogitsLoss` (pos-weight parameter). At eval, `predict_rps()` does
`sigmoid → salience_to_rps_segmented` (Hungarian tracking, `--track_threshold`)
→ STFT grid, so the global-PIT metrics (PIT MSE/RMSE/MAE/R²) apply unchanged
and stay comparable to the SimpleConv family. Both run natively at 16 kHz. The
RPS↔salience helpers live in `multif0/utils.py`.

**Per-rotor layers (`n_maps`) — the `_l4` rows.** Both baselines take the same
`n_maps` option the three harmonic ports carry, and it changes the OUTPUT ONLY:
the front end, the trunk and every input grid stay as they are. `n_maps > 1`
widens the trunk's final 1x1 map convolution (`LateDeep.squishy[1]`,
`BasicPitch.contour_out`) and the channel width of `FreqSuperResHead`, then
stacks the maps along the codec's `(batch, freq, time)` output axis, width
`n_maps * out_bins`. It REQUIRES an explicit linear output grid
(`superres_out=True`), because the CRF band and the log-parabolic vertex fit
are both defined on a uniform axis and both input grids are log-spaced; the
constructor raises otherwise. Configs:
`conf/model/{multif0,basic_pitch}_salience_l4.yaml` paired with
`conf/{loss,metrics}/salience_layers_r150_h256.yaml` — the hop-256 twins of
the ports' `salience_layers_r150`, because these front ends emit salience at
hop 256 and the ports at hop 512.
`models.harmonic_ports.layer_readout.LayerCRFReadout` is mixed in before
`SalienceRPSPredictor`, so `predict_rps` becomes one CRF best path per layer
with no threshold and no Hungarian step. `n_maps=1` is the old model exactly
(same parameter names, shapes, shared-map decoder), locked by
`tests/models/test_salience_baseline_layers.py`. Widening costs 603 parameters
for `multif0_salience` and 1182 for `basic_pitch_salience`. Motivation and
measurement: `harmonic-multipitch-ports.md` § "Per-rotor layers".

Mixing the readout in made `models.harmonic_ports.__init__` import its five
models LAZILY (PEP 562 `__getattr__`): each of them imports
`models.salience_rps` for its base class, and `models.salience_rps` imports
`LayerCRFReadout` back out of that package. `layer_readout` itself depends on
nothing in the package, so exporting it eagerly and the models lazily removes
the cycle and leaves every import path unchanged.

**The zero convention (both directions).** A stopped rotor (`rps <= 0.1`) is
the only case the *target* leaves dark — a rotor that is slow but running is
quantized ONTO the lowest bin, not dropped, so a grid whose `fmin` sits above
the ramp speeds teaches a false speed there rather than losing the frame. On
the *decode* side a frame with no peak above `track_threshold` emits **0 rev/s
for every rotor**: silence == zero rotor speed, never a hold-over of the last
speed and never NaN (`_hungarian_tracking` / `_track_rotors`; tests in
`tests/models/test_salience_rps.py`). Track identity survives a dark frame —
only the emitted value is zeroed — so a momentary dropout does not restart the
tracks.

`multif0_salience`'s HCQT `fmin` defaults to **27.5 Hz (A0)** — matching
basic-pitch, low enough to cover rotor fundamentals below C1 — settable via
`--hcqt_fmin`. The grid descriptor is read back from the front-end, so changing
`fmin` auto-reshapes the salience target and the tracker. (At 16 kHz,
`fmin=27.5` auto-derives 4 harmonics `[1,2,3,4]`; lower `fmin` → more.)

`--fused_branches` runs LateDeep's two identical mag/phase branches as a single
grouped (`groups=2`) stack (`LateDeep(fused_branches=True)`): mathematically
identical (verified to float32 precision in the since-removed
`test_multif0.py` smoke test), one kernel launch per layer instead of two, and
the channel concat becomes free. Checkpoints convert between the two layouts
transparently via a `load_state_dict` pre-hook on `LateDeep`, so a model
trained either way loads either way. Same FLOPs — the win is launch overhead,
so benchmark on GPU (`scripts/bench.py --target grouped_branches`) before
relying on it; `groups=2` can regress on some cuDNN versions.

`--stacked_hcqt` (`LateDeepSalience(stacked=True)`, which rides through to
`build_frontend("hcqt", stacked=True)` → `HCQTFrontEnd(stacked=True)`) uses
`HCQTStacked_nnAudio`: **one** CQT (extra high bins) + harmonic freq-shifts of
mag and phase, instead of one CQT per harmonic. ~2× faster front-end on GPU
(`scripts/bench.py --target cqt`), same `(mag, dphase)` contract and grid. It
is a **lossy approximation** at higher harmonics (h=3 mag corr ~0.977), so the
features differ — **train from scratch**, do not load a non-stacked checkpoint
into it. Composes with `--fused_branches`.
