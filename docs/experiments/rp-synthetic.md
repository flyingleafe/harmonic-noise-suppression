# Synthetic training on the rp protocol

**Status:** in progress · started 2026-10-22

## Motivation

The rp protocol (`paper-real-protocol.md`) trains on real noise recordings
only. This campaign trains the same models on synthetic noise, synthesized
online, under the same protocol: the same model, loss, LR schedule, batch,
speech mixing and real validation panel. Each synthetic family is a "simple
variant" that keeps one property of real rotor noise and drops the rest. The
first variant is the **diverse static comb**:

- each rotor gets a harmonic comb whose profile stays constant within a window;
- profiles come from a wide population: even/odd power laws, Student-t line
  departures, drop-outs and rotor levels;
- each microphone gets a white floor that buries a drawn share of the lines;
- the rotors fly the fitted trajectory model under the reduced hyperprior
  (`rps-trajectory-model.md` § "Reduced hyperprior").

The stream is `conf/online_mix/rp_combdiv.yaml` and the draws are shown in
`docs/explainers/diverse-comb.qmd`.

## Setup

**Stream.** `rp_combdiv` is `rp_all` with the real noise replaced by
`kind: diverse_comb` (weight 2.0), next to the silence arm (0.4). Everything
else matches `rp_all`: 2 s clips, LibriSpeech train-clean-100 per channel at
U(−30, +5) dB, random_gain and spectral_recolor at p 0.5. Each comb render
is 4 s long and is reused for 40 draws, each draw cutting its own 2 s window
(`render_reuse: 40`, `render_duration_s: 4.0`), as in the `prior` stream.
The lines go through the noise model's line kernel (`noise_model.render`).

Measured on one thread, 8 mics, 2026-10-22:

| stream | cost per 2 s draw |
|---|---|
| comb | 0.006–0.012 s |
| `noise_prior_v1_5050` | 0.024–0.027 s |

**Validation** (`conf/validation/rp_real_combdiv.yaml`). Two parts:

- the rp real panel, unchanged: all 22 scores, with `real_macro` among them;
- `static_comb`: 512 mono items (64 draws × 8 mics) of the `rp_combdiv`
  stream, drawn once at the start of each run (`FixedSynthFrameDataset`).
  - Seed 771001; training uses 20261022.
  - 8 s clips, like the real panel.
  - One drone, flight and render per clip.
  - Speech and SNR as in the stream; augmentations stripped.
  - The silence arm is included at its stream weight.

**Stop.** `validation.control: [real_macro, static_comb]`. The run stops at
saturation (`min_rounds` 50, `final_patience` 30), and only once neither
control's 5-round median has improved for 30 rounds. A run that still
improves on the synthetic panel therefore keeps training while its real
scores get worse. Every primary score keeps both best checkpoints, `median`
and `raw`, including `best_static_comb_{median,raw}.ckpt`.

## Runs

| run | model | stream | job | real_macro | static_comb |
|---|---|---|---|---|---|
| `rp_pyrseplstm_k64_combdiv` | `hppnet_pyrseplstm_k64` (as `rp_pyrseplstm_k64_all`) | `rp_combdiv` | — | — | — |

## Results

Not run.

## Conclusion

Pending.
