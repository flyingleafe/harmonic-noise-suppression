---
experiment: hf0_l2_r2_s0
training_config: conf/experiment/hf0_l2_r2_s0.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `hf0_l2_r2_s0`

## Motivation

Supporting Block S comparison: hf0 L2, seed 0, trained from scratch
on the historical R2 honest-base adaptation protocol. Uses LateDeep L2's
`hb_silence_dload` policy, including its 50k-generated-chunk warm-up and
40,000 frames per validation, and the same frozen real validation targets.
There is no comb pretraining, warm-start checkpoint or additional seed.
HPPNet's recurrent width is 128 at both levels. Both levels use the same
four-layer output grid, CRF, loss and `rps_mae` checkpoint selection.
Native front ends and their harmonic coordinates remain the intended
L2/L3 difference; they are not claimed to be a single-operator ablation.

The historical R2 name is not the regressor ladder's DREGON-only R2 rung.
The adaptation results are presented separately from regressor results.
Old L0/L1 checkpoints used BCE selection; legacy L3 rows also differed in
warm-up and validation interval. This run supersedes the cancelled B
comb-curriculum protocol, not any original published measurement.

## Conclusion

Frozen real split, all mics, PIT MAE (rev/s): all **2.45** — zero 0.09,
below-30 11.51, DREGON ramp 4.27, FLY124 ramp 2.47, DREGON cruise 2.98,
FLY124 cruise 1.07; the best silence and below-grid rows of any model on the
split. Early-stopped at epoch 66, best at 45. The curve is noisy — the last
15 validations oscillate between 2.5 and 6 (median 3.25, IQR 2.34) — so 2.45
is the favourable end of a ~3.3 plateau; read it beside HPPNet L2 (2.27 on a
2.79 plateau). Beats the published L0 (10.79) and the matched L3
`hf0_l3_r2_s0` (11.54, which did not train) by the whole distance: the
per-rotor readout adapts HarmoF0, the comb gather does not. Its evaluation was
dumped by hand (`br2-hf0-l2-s0-eval-0df425`) because the chain's tenth
segment ended on the completed run without submitting it. Record:
`docs/experiments/paper-regime-matrix.md` § "B results".
