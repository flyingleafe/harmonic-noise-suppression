---
experiment: rp_hppnet_l2k64_convlstm_priorv4_ft_lr5e4
training_config: conf/experiment/rp_hppnet_l2k64_convlstm_priorv4_ft_lr5e4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_hppnet_l2k64_convlstm_priorv4_ft_lr5e4`

## Motivation

`rp_hppnet_l2k64_convlstm_priorv4` fine-tuned on real data with the HRSPNet recipe
(`rp_pyrseplstm_k64_priorv4_ft_lr5e4`): `rp_all` stream without label augmentation, fresh AdamW
at max LR 5e-4, 1000 warm-up steps, cosine to the 20 000-step cap, validation `rp_real`, from the
synthetic-selected checkpoint `best:uniform@rp_hppnet_l2k64_convlstm_priorv4`. Three seeds with
fixed pretrained weights (Dmitrii, 2026-11-14): seed 0 is this experiment; seeds 1 and 2 run as
`experiment_name=rp_hppnet_l2k64_convlstm_priorv4_ft_lr5e4_s<n> seed=<n>
+data.train.params.base_seed=20261015 / 20261016`, the stream seeds of the HRSPNet seeds of
Table IX. Read on the held-out test split against HRSPNet's 2.34 ± 0.01 Hz
(`scripts/rp_test_eval.sh`).

## Conclusion

Ran 2026-11-14 (seeds 0 / 1 / 2: `rp-hppnet-l2k64-convlstm-66fab0`, `-6c04ca`, `-42cebb`). Raw
validation minima 2.444 / 2.511 / 2.523 Hz (2.49 ± 0.04); test 2.73 / 2.91 / 3.16
(2.93 ± 0.22), against HRSPNet's 2.34 ± 0.01; MD2 zero 23.4 ± 2.4 Hz in every seed. Table VIII,
"+ ConvLSTM".
