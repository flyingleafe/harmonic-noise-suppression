---
experiment: hppnet_l2_r2_s0
training_config: conf/experiment/hppnet_l2_r2_s0.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `hppnet_l2_r2_s0`

## Motivation

Supporting Block S comparison: hppnet L2, seed 0, trained from scratch
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

Frozen real split, all mics, PIT MAE (rev/s): all **2.27** — zero 1.07,
below-30 14.64, DREGON ramp 4.91, FLY124 ramp 2.72, DREGON cruise 2.07,
FLY124 cruise 0.77. Early-stopped at epoch 54, best at 33; the last 15
validations sit on a 2.79 plateau (IQR 0.27), so the number is stable. Beats
the matched L3 `hppnet_l3_r2_s0` (4.30) on every regime but FLY124 cruise
(0.77 vs 0.89), the published L0 (7.77) and LateDeep L2 on the same recipe
(3.83); at one seed it is the best learned row of the frozen split. The
per-rotor readout, not the comb gather, is what adapts HPPNet. Record:
`docs/experiments/paper-regime-matrix.md` § "B results".
