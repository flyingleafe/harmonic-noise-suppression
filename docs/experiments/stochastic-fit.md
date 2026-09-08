# Fitting the stochastic rotor-noise family to real recordings

**Status:** in progress — 2026-09-08 → . Branch `stochastic-fit`, worktree
`.worktrees/stochastic-fit`. Code: `src/experiments/stochastic_fit/`. Textbook
explainer of the method (model, objective, references, procedure):
`docs/explainers/stochastic-fit.qmd` (render with `quarto render`; the OJS
figure needs an HTTP origin).

## Motivation

Regressors trained on the stochastic synthetic family alone
(`data_processing.stochastic_rotor_noise`, policy `conf/online_mix/salv2_stoch.yaml`)
score 8.08 rev/s all-MAE on the real panel; with real audio in the mix,
2.67 (`docs/experiments/stochastic-transfer.md`). Every previous attempt moved
one sampler range at a time and measured transfer. None helped. The question
this campaign answers first is prior to all of them: **at its best parameters
for a given real recording, how much of that recording does the family
explain, and where does it fail?** Two hypotheses with opposite fixes: wrong
ranges (retune the sampler) vs wrong family (change the model).

The instrument is a maximum-a-posteriori fit of the family's own spectral
model to a recording's eight-channel Hann-2048/512 periodogram under the
Whittle likelihood, rotor speeds held fixed to the refined references
(`pi_kalman_refine`, the 2026-09-06 linewidth-audit recipe). Everything the
sampler draws is a parameter; the GP drifts are latents with their whitened
priors. Explained fraction = (floor-only − fit) / (floor-only − LOO), where
the LOO reference is a leave-one-out periodogram smoother scored with the
analytic exponential-mean bias removed — the score a correct model reaches.

## Data

Bundle on R2 `artifacts/stochastic-fit/clips/` (manifest.json, 54 clips):
27 four-second training crops with acoustic references (17 DREGON room2, 10
FLY125; `.worktrees/jhtr-refinement/results/jhtr/trajectory-linewidth/refs`),
23 eight-second `nosource` clips of the frozen validation split (8 DREGON
room1 with measured shaft rate, 15 FLY124; stopped, spin-up, ramps and cruise;
refined on `uni-cpu`, job `stochfit-prepare-32bde4`), 4 renderer control clips
on real FLY125/room2 trajectories with their planted parameters.

## Results

### The instrument passes its controls

Four renderer clips (planted parameters, default sampler): excess over the
LOO reference −0.004 ± 0.01 nats/cell, explained 1.01–1.07; identical at the
short (150/150/300/60) and long (300/300/800/150) optimizer schedules, so the
fit is converged; residual profiles along the lines flat, half-order residual
1.00, coherence statistic 1.02–1.09 (exponential). ~15 s per clip on a Kaggle
P100 (PyPI torch 2.7 cu126 — the Kaggle image's torch has no sm_60 kernels).

Two findings about the instrument itself:

- **LOO replicates must be two hops apart.** The first reference averaged the
  ±1/±2-hop neighbours; at a quarter-window hop the adjacent Hann frame's
  periodogram is correlated with the cell (power correlation ≈ 0.43), so the
  smoother predicted every cell better than the truth could and all four
  controls showed the same +0.245 nats/cell "excess". Replicates at ±2/±4
  hops (half-window shifts, correlation ≈ 0.03) put the controls at zero;
  wider spans (±4/±6, ±4/±8) drift negative (−0.03 to −0.09) as the GP
  drift breaks stationarity. `fit.LOO_OFFSETS = (-4, -2, 2, 4)`.
- **Width-law parameters are only partly identifiable in the family
  itself.** The renderer floors every half width at 0.6 bins (4.7 Hz), so
  `gamma0` is observable only through orders where `gamma0 + slope*k`
  exceeds the floor; slopes come back roughly (0.57/0.95/0.99/0.67 vs planted
  0.30/0.70/0.72/0.68 on one control) and a rotor whose lines are buried gets
  arbitrary widths at no likelihood cost. The MAP drift curves are shrunk
  ~2× (fitted std 1.3–3.3 dB vs planted 3.8–5.3 at the 3 dB prior). The
  likelihood-level questions are unaffected; parameter readouts are
  restricted to visible lines (`diagnostics.fitted_parameter_summary`).

### The family, as sampled, explains half to two thirds of real recordings

Variant `family` (the renderer's exact parametrization, reference carriers,
no correction), Kaggle job `stochfit-family-b9f69d`, all 48 clips with
turning rotors (the two stopped-rotor clips have no comb and no defined
fraction):

| group | n | explained: median (min–max) | excess nats/cell: median (min–max) | floor−LOO (median) |
|---|---:|---|---|---:|
| DREGON room2 (crops) | 17 | 0.45 (0.37–0.54) | 0.33 (0.30–0.36) | 0.62 |
| DREGON room1 (validation) | 7 | 0.66 (0.48–0.73) | 0.29 (0.28–0.42) | 0.91 |
| FLY125 (crops) | 10 | 0.68 (0.64–0.74) | 0.31 (0.27–0.33) | 0.99 |
| FLY124 (validation) | 14 | 0.70 (0.56–0.75) | 0.26 (0.17–0.68) | 0.86 |
| renderer controls | 4 | 1.02 (1.01–1.07) | −0.006 (−0.02–0.00) | 0.30 |

The leftover ≈ 0.3 nats/cell is nearly constant within a group — a
structural misfit, not a bad draw; on the KL scale of the explainer it is a
factor-1.8 power error on every cell, or a factor 10 on one cell in five.
Per-clip numbers: `results/stochastic_fit/summary_family.json` of the job.

### Where it fails (residual `I/M` of the family fit, representative clips)

- **Low-order line shape.** Orders 1–8, residual along the line in units of
  the fitted half width: FLY125_06 R = 0.75 / 0.93 / 1.66 / **2.12** / 2.02 /
  1.34 / 0.76 at −3γ…+3γ; FLY124_31 0.86 / 1.16 / 1.45 / 1.38 / 1.20 / 0.84 /
  0.78. Real low harmonics are much sharper than the family can render: the
  renderer's 0.6-bin width floor plus bin-centre sampling cannot make a
  sub-bin tone. Orders 9–64 are flat (1 ± 0.1) — the mid-order widths are
  fine; the "widths grow too fast with order" hypothesis is not what the
  data say there.
- **Fluctuation statistics.** Normalized variance of the residual at line
  centres 1.4–3.1 at orders 1–8 and 65+ (controls 1.02–1.09): heavier than
  exponential, i.e. intermittent — consistent with the earlier flicker
  measurement (real 4–5 dB vs stochastic 3.4). Residual autocorrelation at
  lag 4 hops 0.06–0.15 at low orders (controls 0.00): amplitude variation the
  3 dB / 1.5 s GP drift does not follow.
- **Microphones** (model-free, `family` job periodograms, 200–4000 Hz;
  spread across the 8 mics in dB, median over clips):

  | group | floor proxy | line proxy | overall level | floor − level | line − level |
  |---|---:|---:|---:|---:|---:|
  | FLY125 | 12.8 | 13.0 | 12.2 | 1.2 | 1.0 |
  | FLY124 | 10.6 | 10.7 | 10.1 | 1.0 | 0.6 |
  | DREGON room1 | 4.4 | 6.8 | 10.1 | 9.2 | 3.4 |
  | DREGON room2 | 5.1 | 6.6 | 8.5 | 8.0 | 3.5 |

  On Michael's rig the microphones differ by ~12 dB in level with floor and
  lines moving *together*; the family's per-microphone gain applies to the
  lines only, over a common floor, and cannot represent this. On DREGON the
  floor and line spreads are decoupled (8–9 dB), which the family cannot
  represent either. (The fitted model's per-mic floor residual is not
  quoted: at K ≈ 266 the line masks cover 98–100 % of cells and the remainder
  is a selection-biased sliver; the `mic_floor` ablation is the evidence.)
- Half-order residual 1.05–1.17 at orders 1–8: a little sub-harmonic content.

### Ablations (in flight)

Kaggle jobs `stochfit-var-crops-7c047a` (27 crops × `family_rps`, `mic_floor`,
`sharp`, `gauss`, `free_gamma`, `drift6`, `extended`) and `stochfit-var-valid`
(23 validation clips × `family_rps`, `mic_floor`, `sharp`, `speed_law`,
`extended`; controls at the 6 dB drift prior). Results to follow.

## Conclusion

Pending the ablations.

## Gotchas

- Kaggle output collection through omnirun fails silently above ~2 GB (the
  job shows `succeeded`, `omnirun pull` returns nothing, and the kernel may
  still be RUNNING on Kaggle when omnirun says done). Check with
  `uvx --from kaggle kaggle kernels status/output <ref>`; result files are now
  slim (float16 dB spectrum, periodogram rebuilt offline from the R2 clip).
- Kaggle's image torch has no sm_60 kernels; the slim snapshot uses a uv env
  with PyPI `torch==2.7.0` (`scripts/kaggle_slim_stochfit.sh`, branch
  `kaggle-slim-stochfit`, worktree `.worktrees/kaggle-slim-stochfit`).
