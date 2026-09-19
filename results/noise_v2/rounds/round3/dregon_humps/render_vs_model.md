# DREGON round 3: render against the fit's own forward model

`I` = periodogram of the v2 render (seeds 2001, 2002, 2003, 2004, power-averaged), `M` = `render.expected_periodogram` of the SAME fit on the same label track, `R` = the real clip's periodogram — one framing (2048/512), one set of absolute units, all 8 mics. Cells are classed by each frame's OWN label carriers: a comb cell is within 1 bin of `k f_r(t)` for some rotor; the three classes partition 30-7900 Hz.

| window | ratio | comb k<=8 | comb 8<k<=40 | floor |
|---|---|---:|---:|---:|
| `hovering` | I/M (render vs model) | -0.35 (-1.9/+1.1) | -0.34 (-1.9/+1.1) | -0.37 (-2.0/+1.1) |
| `hovering` | R/M (real vs model) | -2.21 (-8.2/+3.3) | -1.29 (-6.4/+3.2) | -0.46 (-5.2/+3.6) |
| `hovering` | R/I (real vs render) | -1.73 (-7.8/+4.0) | -0.84 (-6.1/+3.9) | -0.02 (-5.0/+4.4) |
| `updown` | I/M (render vs model) | -0.36 (-1.9/+1.1) | -0.32 (-1.9/+1.1) | -0.36 (-2.0/+1.1) |
| `updown` | R/M (real vs model) | +2.51 (-2.3/+6.8) | +1.07 (-3.4/+4.9) | +1.12 (-3.1/+4.6) |
| `updown` | R/I (real vs render) | +3.00 (-2.1/+7.5) | +1.49 (-3.2/+5.7) | +1.55 (-2.9/+5.4) |

Median dB with the quartiles beside it. Power-sum ratios (the same cells, total power rather than per-cell median):

| window | ratio | comb k<=8 | comb 8<k<=40 | floor |
|---|---|---:|---:|---:|
| `hovering` | I/M | +0.04 | +0.01 | +0.01 |
| `hovering` | R/M | +2.46 | +2.29 | +2.18 |
| `hovering` | R/I | +2.42 | +2.29 | +2.16 |
| `updown` | I/M | +0.05 | +0.02 | +0.00 |
| `updown` | R/M | +3.28 | +4.08 | +4.04 |
| `updown` | R/I | +3.23 | +4.06 | +4.04 |

Cell counts per window: comb_k1_8 45880, comb_k9_40 192544, floor 746144.
## Reading

1. **The render is faithful to its own forward model.** `I/M` is −0.35 dB
   median and +0.0 to +0.05 dB in power-sum, in EVERY class and both windows:
   comb k≤8, comb 8<k≤40 and floor alike. The −0.35 dB is exactly the
   median-vs-mean bias of the estimator (four seeds × Rayleigh cells: the
   median of a Gamma(4,1)/4 variate is 10 log10(0.918) = −0.37 dB), so the
   render puts on the wire what `expected_periodogram` says, cell for cell.
   There is **no render or forward-model bug**: not the four-rotor sum, not the
   mean-pinned mic gains, not comb_gain/low-order folding, not
   `grid_power_factor`, not a doubled transfer. Any of those would show as a
   class-dependent offset here and none does.
2. **The fit is not under-weighting the comb cells either.** `R/M` in power-sum
   is +2.46 (comb k≤8), +2.29 (8<k≤40), +2.18 (floor) dB on `hovering` and
   +3.28 / +4.08 / +4.04 dB on `updown`. The residual is a nearly UNIFORM
   +2 to +4 dB across all three classes — on `updown` the floor is missed by
   MORE than the comb. There is no hidden +20 dB comb in the real periodogram
   for a re-weighted objective to find.
3. **Therefore the +21 dB comb the tracker wants is not in the data.** The v2
   render reproduces the real window's per-cell power to within a few dB in
   every class, and HPPNet still scores 73 rev/s on it against 0.89 on the real
   clip. What HPPNet keys on is not per-cell power at the comb cells: it is
   structure the power spectrum does not determine (per-mic/cross-mic and
   cross-order phase coherence, temporal continuity of a line). The +21 dB
   re-level is a way to give the tracker enough per-cell contrast to lock, not
   a correction to the model.

This supersedes the objective-change diff sketched in `findings.md`
(§ R4 proposal): re-weighting the Whittle risk onto carrier-tracked cells
cannot move `comb_gain_db` by +21 dB, because those cells are already fitted to
within ~2 dB of the floor cells' own residual. The open R4 question is a
STATISTICS question, not a level or width question.
