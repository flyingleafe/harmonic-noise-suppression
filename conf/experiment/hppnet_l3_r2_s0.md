---
experiment: hppnet_l3_r2_s0
training_config: conf/experiment/hppnet_l3_r2_s0.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `hppnet_l3_r2_s0`

## Motivation

Supporting Block S comparison: hppnet L3, seed 0, trained from scratch
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

Frozen real split, all mics, PIT MAE (rev/s): all **4.30** — zero 5.15,
below-30 28.29, DREGON ramp 9.91, FLY124 ramp 6.07, DREGON cruise 3.10,
FLY124 cruise 0.89. Early-stopped at epoch 38, best at 17, last-15 plateau
4.85 (IQR 0.35). Reproduces the legacy-schedule `hppnet_r2hb_l4` (4.18,
plateau 5.17), so that row was not a schedule artefact. Loses to the matched
L2 `hppnet_l2_r2_s0` (2.27) on every regime but FLY124 cruise: under the same
per-rotor readout, the comb gather on the linear STFT is worse than HPPNet's
own CQT + `HarmonicDilatedConv`. Record: `docs/experiments/paper-regime-matrix.md`
§ "B results".
