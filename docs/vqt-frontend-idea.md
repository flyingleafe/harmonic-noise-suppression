# Idea: a variable-Q transform as the harmonic front end

2026-10-11. Deferred; not run. Follows `docs/pyramid-harmonic-frontend-design.md`
(the pyramid) and its § 8 literature anchors (VQT: Schörkhuber et al. 2014; PESTO).

## The slew/resolution balance, stated

Rotor at rate `r` (rev/s), slewing at `ṙ` (rev/s²). Order `k` sits at `f = k r` Hz
and moves at `k ṙ` Hz/s. Analyse it with a window of `T` seconds. Two errors in
the line position, both in rate units (Hz / k):

- resolution: `ε_res = 1 / (k T)` — finer with longer windows and higher orders;
- smear: the line moves `k ṙ T` Hz inside the window → `ε_slew = ṙ T`,
  **independent of k** (the Hz smear grows with k exactly as fast as the
  conversion to rate shrinks it).

Minimising `1/(kT) + ṙ T` gives

```
T*_k = 1 / sqrt(k ṙ),     ε*_k = 2 sqrt(ṙ / k),     Δf*(f) = sqrt(k ṙ) ∝ f^{1/2}
```

The optimal window shrinks with order as `k^-1/2`; achievable precision improves
with `k` only as `k^-1/2`, not `k^-1`.

| front end | Δf(f) | window vs order | verdict |
|---|---|---|---|
| single STFT | const | `T` const | resolution-limited at low k, smear-dominated at high k |
| CQT (Q = 68.8) | ∝ f | `T ∝ k^-1` | shrinks too fast: `ε_res = r/Q` is constant in k, the `1/k` lever is thrown away |
| pyramid | staircase ≈ ∝ f above 1.2 kHz (Q ≈ 1229), const 0.98 Hz below | `T ∝ k^-1` piecewise | right ballpark (long windows), wrong exponent, seams at level ceilings |
| optimum | ∝ f^{1/2} | `T ∝ k^-1/2` | needs a per-frequency window law: VQT / nonstationary Gabor frame |

Numbers at `ṙ = 3 rev/s²` (typical sustained): `T*_1 = 0.58 s`, `T*_10 = 0.18 s`,
`T*_60 = 75 ms`, `ε*_60 = 0.45 rev/s`. At the ramp extreme `ṙ = 460`:
`T*_60 = 6 ms`, `ε*_60 = 5.5 rev/s` — high orders are useless there and only low
orders with short windows survive, the regime where both the pyramid's fixed
long windows and the CQT's long low-frequency windows lose.

Caveat: `ṙ` is a property of the regime, so `Δf* ∝ sqrt(ṙ)` is regime-dependent.
One transform picks one `ṙ_typ`; the only argument for more than one window per
frequency (the pyramid's fusion, or two VQTs at two `ṙ` settings stacked as
channels) is covering two regimes at once.

## What the pyramid actually is, and what a VQT would change

A pyramid tap at `f` reads the finest level covering it, so its bandwidth is a
staircase: const 0.98 Hz below 1.2 kHz, then ≈ `f / 1229` above. The standard VQT
`Δf(f) = f/Q + γ` with `(Q, γ) ≈ (1229, 1 Hz)` is the smooth envelope of that
staircase (1.08 Hz at 100 Hz, 4.9 at 4.8 kHz, 7.5 at 8 kHz). The general
nonstationary Gabor frame with `Δf(f) = sqrt(f ṙ_typ / r_typ)` clamped to
`[1, 8] Hz` realises the optimum.

Gains:
- no seams (the design note flags kinks at `r = f_ℓ^hi / k`, inside the rotor
  band for `k ≥ 8`);
- no fusion and no nested redundancy: `∫ df/Δf ≈ 1229 · ln(1 + 8000/1229) ≈ 2.5k`
  bins vs 4.7k, one map instead of four + FPN;
- a line is the same number of bins wide everywhere *by construction* (bin
  spacing ∝ bandwidth), which is what the shared kernel bank needs; the pyramid
  has it only level-wise;
- the `k^-1/2` law can be realised exactly.

Costs / losses:
- the axis is neither linear nor log, so harmonic *dilations* do not apply; the
  proportional-tap gather (`tap_conv`) already takes an arbitrary frequency→bin
  table (`lo`, `frac`, `valid`), so it reuses unchanged with a linear rate grid;
- one window per frequency: the fusion's coarse-context-at-fine-resolution is
  gone; for fixed `f`, `k = f/r` varies 2× over the rotor band so `T*` varies
  1.4× — probably negligible, the convs can smooth;
- a kernel bank (sparse frequency-domain CQT trick, hop 512) instead of four
  FFTs; still negligible against the convs. nnAudio/librosa have CQT+γ; a GPU
  port on hop 512 with long low-frequency windows is small;
- orthogonal to the level-3 finding: the k ≈ 70 rotor-identity cue is a
  data/label property and will appear on any front end that resolves it.

## The arm

Same gather, same `RateConvLSTM` head, same unified regime and panel as the
pyramid arms (`docs/experiments/pyramid-harmonic.md` § Protocol). One front
end change: VQT with `Δf(f) = sqrt(f · ṙ_typ / r_typ)`, `ṙ_typ = 3 rev/s²`,
`r_typ = 80 rev/s`, clamped to `[1, 8] Hz`, hop 512, taps masked above 4.8 kHz
as in HRSPNet. Read per regime: ramps are where it should separate from the
pyramid; cruise should be level. Run after the pyramid grid in
`docs/handoff-pyramid-harmonic.md` § 6 is settled, not before.
