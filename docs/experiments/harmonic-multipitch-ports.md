# Harmonic multi-pitch architectures ported to the linear STFT

**Configs:** `conf/experiment/{hf0,hppnet,hft}_*.yaml`
**Design note:** [`docs/harmonic-ports-design.md`](../harmonic-ports-design.md)
**Code:** `src/models/harmonic_ports/`

## Why this batch exists

The project's rotor-speed regressors read a spectrogram and emit four numbers.
The multi-pitch literature solves a structurally identical problem — several
simultaneous harmonic sources, unknown count, dense frame labels — with
architectures built around the harmonic structure itself. Three were ported:

| port | source | the harmonic device |
|---|---|---|
| `hf0` | [HarmoF0](https://github.com/WX-Wei/HarmoF0) | `MRDConv`, multi-rate dilated convolution |
| `hppnet` | [HPPNet](https://github.com/WX-Wei/HPPNet) | `HarmonicDilatedConv` + `FreqGroupLSTM` |
| `hft` | [hFT-Transformer](https://github.com/sony/hFT-Transformer) | per-note cross-attention over harmonics |

**All three assume a log-frequency axis, and this project's axis is linear.**
Their harmonic device is a fixed dilation pattern that only lands on harmonics
when frequency is logarithmic. Rather than resample the STFT onto a log grid —
which undersamples high harmonics badly (separating them would need 11,443 bins)
— the ports replace that device with the campaign's comb gather, which reads the
spectrum at `k * r` directly on the linear axis. The gather IS a
harmonic-dilated convolution on a linear axis, so this is the same idea in the
representation the data actually has.

## The cells

Each trunk runs on three curricula: static comb only (`_comb`), stochastic comb
only (`_stoch`), and the real-data curriculum R4 (`_r4`). The `_l4` suffix marks
the four-per-rotor Gaussian salience layers with a CRF readout, which replaced
the shared triangular-kernel map after that pair was measured to lose 8.24 rev/s
on a PERFECT target (`models.salience_crf`). Rows without `_l4` predate the
measurement and are kept for the comparison.

## Conclusion

Both convolutional ports beat the regressor baselines on the two synthetic
families; the transformer did not converge. Read those numbers with the caveat
that they were produced on the OLD synthetic streams, whose training and
validation distributions differ (`freq_scale` on every training sample against
an unaugmented validation set) and whose validation set was 12 clips from ONE
trajectory. The rebuilt streams and the grid that replaces this batch are
[SALV2](./salv2-speech-and-objective-grid.md).

## Implementation notes (moved from `src/models/AGENTS.md`, 2026-09-07)

### Why the output axis is a linear candidate-rate grid

Each port has ONE organ replaced: the log-frequency harmonic **shift** becomes
an explicit **gather** at `k*r` on the linear STFT
(`models.comb_salience.CombGather`). The measurements that reject the log axis
for this task live in `docs/harmonic-ports-design.md`; the short version is
that a log grid's separation-to-bandwidth ratio for two rotors `D` apart is
`D / (r * (2^(1/B) - 1))`, in which the harmonic index cancels, so a rotor pair
is resolved at every harmonic or at none, while a uniform STFT improves
linearly with `k`. The output axis is therefore the CANDIDATE RATE, not
frequency, and it is **linear** (a log rate grid spends its resolution at the
coarse end, where nothing needs it).

### Per-rotor layers (`n_maps: 4`) — the `_l4` rows

The framework's shared salience map is not a lossless encoding of this task:
`models.salience_crf` encodes real training telemetry and decodes it back — a
PERFECT target, no model involved — and returns it **8.24 rev/s** away on
average, with 39-45% of frames more than half a bin off, against **2.22e-16**
for Gaussian per-rotor layers read by a CRF plus log-parabolic fit. Three
causes: one map cannot hold four rotors whose pairs sit inside one 0.5 rev/s
bin; a triangular kernel has no exact sub-bin readout; and
`active = rps_grid > 0.1` encodes a stopped rotor as an ABSENCE, which is what
forces a decode threshold.

The `_l4` model configs (`conf/model/{harmof0,hppnet,hft}_rps_l4.yaml`) set
`n_maps: 4` and must be paired with `conf/loss/salience_layers_r150.yaml` and
`conf/metrics/salience_layers_r150.yaml`;
`models.harmonic_ports.layer_readout` then overrides `predict_rps` with one CRF
best path per layer — no threshold, no Hungarian step. The layers ride the
codec's `(batch, freq, time)` wire format stacked along the output axis (width
`4 * 300`), because a 4-D model output does not type-check through
`SalienceRPSCodec`. With `n_maps: 1` everything falls back to the old
shared-map path unchanged. `multif0_salience` and `basic_pitch_salience` take
the same option (their hop-256 twins are
`conf/{loss,metrics}/salience_layers_r150_h256.yaml`); widening costs 603 and
1182 parameters respectively, and the `n_maps=1` identity is locked by
`tests/models/test_salience_baseline_layers.py`.

### The `salience_rps` contract the ports satisfy

`forward(audio) -> (B, G, T)` logits, `outputs_salience = True`, BCE through
`losses.SalienceRPSBCELoss`, Hungarian tracking through the inherited
`predict_rps` — by declaring the rate grid as
`SalienceRPSPredictor.out_freqs`, the hook that already exists for a salience
axis decoupled from a log-spaced input CQT. Nothing in the task, the codec,
the loss or the tracker changes.

### The two controls (`harmof0_orig`, `hppnet_orig`)

The same two papers with NOTHING replaced — the published harmonic device, the
published front end, and a 352-bin log grid at 48 bins/octave from 27.5 Hz —
wired into the same `salience_rps` task. They exist because every port row so
far had been read against the direct REGRESSORS, which share neither the trunk
nor the output representation, so no measurement separated the substitution
from the trunk. Both emit a bit-identical grid, so
`conf/loss/salience_bce_orig.yaml` and `conf/metrics/salience_bce_orig.yaml`
serve both arms; the experiments are `hb_sal_{hf0,hppnet}_orig` and the batch
doc is `paper-regime-matrix.md` § "Block S", where they are level L0 of the
multi-pitch adaptation ladder and the `*_rps` ports are level L3. Their
harmonic blocks are checked bit-identical against the upstream source in
`tests/models/test_harmonic_orig.py`; the remaining deviations are seam-level
(the hop-512 frame grid, logits instead of a sigmoid, HPPNet's piano-specific
heads and its two pools) and are listed in the two module docstrings. Under
`f0 = rps` that log grid spans 27.5-4371 rev/s, of which rotors occupy bins
0-118 of 352, at 1.45% of the rate per bin — which is the cost the pair is
there to measure.

### What each port replaces

| Model | Paper | What was replaced |
|-------|-------|-------------------|
| `harmof0_rps` | HarmoF0, Wei et al. ISMIR 2022 | `MRDConv` (a 1x1 conv, a `round(log2(k)*B)`-bin shift, and a sum, per harmonic) → `CombGather` at `k*r` times a learned per-harmonic weight. Its blocks 2-4 keep their shape but their octave-sized dilations become plain dilated context convolutions along RATE, where an octave is not a fixed offset — a deliberate deviation, documented in the module docstring |
| `hppnet_rps` | HPPNet, Wei et al. ISMIR 2022 | `HarmonicDilatedConv` (eight log-axis dilated branches) → the same gather at `k*r`; `CNNTrunk` and `FreqGroupLSTM` kept, MPE head only |
| `hft_rps` | hFT-Transformer, Toyama et al. ISMIR 2023 | Nothing structural — hFT already holds ONE DECODER TOKEN PER NOTE that cross-attends to the frequency tokens (`attention = [batch, frame, heads, n_note, n_bin]`). The tokens become CANDIDATE RATES and the gather becomes a hard mask on that attention: a rate token sees only its own K harmonics. hFT's frequency self-attention ENCODER is deleted, because nothing else is visible; `attn_mode: bias` restores it over a pooled 256-token spectrum with the gather as a learned additive bias instead (variant (ii) of the design note). Only the frame/MPE head survives — onset, offset and velocity are piano-specific |

### Three traps of a rate grid that reaches 0

All handled inside `harmof0_rps.py` / `hft_rps.py`:

1. **The near-DC gather.** At 1.5 rev/s every harmonic lands inside the STFT
   window's DC mainlobe, reads far above a median floor computed from that same
   mainlobe, and wins. `f_min` (30 Hz by default) drops those reads. The
   classical scan never needed this because it searches 30-100 rev/s.
2. **Count normalization.** The classical head divides the summed evidence by
   the number of in-band harmonics, which is right in a narrow search band and a
   trap on a 0-150 grid: a candidate with three surviving harmonics wins on one
   lucky hit. Dividing by the constant `k_max` instead took the untrained
   score from 3/8 to 7/8 on a synthetic single-comb probe.
3. **The unresolvable candidate.** Consecutive harmonics of a rate are that
   rate apart in Hz, so below one STFT bin (3.906 Hz at n_fft 4096) they land
   in the same bin and the hypothesis is not a comb — its K reads are a handful
   of bins inside one strong low line's skirt. `f_min` does not catch it: those
   reads sit at 30-48 Hz, outside the DC mainlobe. `hft_rps.py`'s `r_min`
   (default `sr / n_fft`) drops every harmonic of such a candidate, and with it
   the untrained read of a 37 rev/s comb moves from 1.51 rev/s to 37.12; 45, 60,
   84.5 and 120 also land within one grid bin.
