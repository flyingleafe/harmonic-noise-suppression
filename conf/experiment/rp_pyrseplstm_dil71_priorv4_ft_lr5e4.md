---
experiment: rp_pyrseplstm_dil71_priorv4_ft_lr5e4
training_config: conf/experiment/rp_pyrseplstm_dil71_priorv4_ft_lr5e4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrseplstm_dil71_priorv4_ft_lr5e4`

## Motivation

`rp_pyrseplstm_dil71_priorv4` (HRSPNet with 71 fixed dilated taps instead of the harmonic
gather) fine-tuned on real data with the HRSPNet recipe (`rp_pyrseplstm_k64_priorv4_ft_lr5e4`):
`rp_all` stream without label augmentation, fresh AdamW at max LR 5e-4, 1000 warm-up steps,
cosine to the 20 000-step cap, validation `rp_real`, from `best:uniform@rp_pyrseplstm_dil71_priorv4`.
Three seeds with fixed pretrained weights (Dmitrii, 2026-11-14): seed 0 is this experiment;
seeds 1 and 2 run as `experiment_name=rp_pyrseplstm_dil71_priorv4_ft_lr5e4_s<n> seed=<n>
+data.train.params.base_seed=20261015 / 20261016`, paired with the HRSPNet seeds. Read on the
held-out test split (`scripts/rp_test_eval.sh chain_synth`).

## Conclusion

Ran 2026-11-15 (seeds 0 / 1 / 2: `rp-pyrseplstm-dil71-prio-08dd93`, `-0553bd`, `-50d485`). Raw
validation minima 3.839 / 3.814 / 3.370 Hz (3.68 ± 0.26); test 5.36 / 5.15 / 5.00
(5.17 ± 0.18), the level of HPPNet-L2 (5.11 ± 0.55) and far above HRSPNet's 2.34 ± 0.01. The
harmonic gather is the component HRSPNet cannot do without. Table VIII, "HRSPNet, 71 dilated
taps".
