---
experiment: rp_pyrsep_k64_priorv4
training_config: conf/experiment/rp_pyrsep_k64_priorv4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrsep_k64_priorv4`

## Motivation

Side ablation of the paper's chain table under the hard-pretraining recipe (Dmitrii,
2026-11-14): HRSPNet without its recurrence. `conf/model/hppnet_pyrsep_k64.yaml` is
`hppnet_pyrseplstm_k64` with the separable RateConvLSTM head replaced by the same-frame separable
rate kernel (`head: sepconv`), the change of the `rp_pyrsep_in_all` → `rp_pyrseplstm_all` pair at
$k \le 64$. Trained like `rp_pyrseplstm_k64_priorv4`: hard stream `rp_priorv4`, validation
`rp_real_priorv4`, batch 128, max LR 1e-3, 12 data workers, one H100. Fine-tuned in three seeds by
`rp_pyrsep_k64_priorv4_ft_lr5e4` and read on the held-out test split against HRSPNet's
2.34 ± 0.01 Hz.

## Conclusion

Pending.
