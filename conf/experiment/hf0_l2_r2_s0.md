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

Results pending. No comparative accuracy conclusion is available yet.
