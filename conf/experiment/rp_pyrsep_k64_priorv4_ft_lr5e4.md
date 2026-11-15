---
experiment: rp_pyrsep_k64_priorv4_ft_lr5e4
training_config: conf/experiment/rp_pyrsep_k64_priorv4_ft_lr5e4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrsep_k64_priorv4_ft_lr5e4`

## Motivation

`rp_pyrsep_k64_priorv4` (HRSPNet without its recurrence) fine-tuned on real data with the HRSPNet
recipe (`rp_pyrseplstm_k64_priorv4_ft_lr5e4`): `rp_all` stream without label augmentation, fresh
AdamW at max LR 5e-4, 1000 warm-up steps, cosine to the 20 000-step cap, validation `rp_real`,
from `best:uniform@rp_pyrsep_k64_priorv4`. Three seeds with fixed pretrained weights (Dmitrii,
2026-11-14): seed 0 is this experiment; seeds 1 and 2 run as
`experiment_name=rp_pyrsep_k64_priorv4_ft_lr5e4_s<n> seed=<n>
+data.train.params.base_seed=20261015 / 20261016`, paired with the HRSPNet seeds. Read on the
held-out test split (`scripts/rp_test_eval.sh`).

## Conclusion

Ran 2026-11-15 (seeds 0 / 1 / 2: `rp-pyrsep-k64-priorv4-ft-002d06`, `-b57e9a`, `-16410c`). Raw
validation minima 2.730 / 3.094 / 2.544 Hz (2.79 ± 0.28); test 4.32 / 4.76 / 3.93
(4.34 ± 0.42), against HRSPNet's 2.34 ± 0.01. The recurrence carries the DREGON zero, standby
and ramp frames (test 26.8 / 28.3 / 15.1 Hz against 0.9 / 13.2 / 5.1); MD2 changes little
(1.96 against 1.52). Table VIII, "HRSPNet without recurrence".
