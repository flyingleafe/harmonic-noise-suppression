---
experiment: hf0_l3_r2_s0
training_config: conf/experiment/hf0_l3_r2_s0.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `hf0_l3_r2_s0`

## Motivation

Supporting Block S comparison: hf0 L3, seed 0, trained from scratch
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

Frozen real split, all mics, PIT MAE (rev/s): all **11.54** — zero 5.63,
below-30 24.60, DREGON ramp 30.36, FLY124 ramp 21.27, DREGON cruise 12.78,
FLY124 cruise 3.76. A from-scratch training failure, not a measurement of the
gather: `val/rps_mae` never left the 11-24 band (last-15 plateau 16.36, IQR
6.05), the best came at epoch 2 and patience stopped the run at 23. The
legacy-schedule `hf0_r2hb_l4` (7.90, plateau 9.76) shows the same behaviour
over 62 epochs; the `hf0_r4_l4` warm start from a comb-only stage is the
recipe that makes this port train. The matched L2 `hf0_l2_r2_s0` on the same
recipe reaches 2.45. Record: `docs/experiments/paper-regime-matrix.md`
§ "B results".
