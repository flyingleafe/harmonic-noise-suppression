---
experiment: rp_pyrseplstm_k64_combdiv_ft_lr5e4
training_config: conf/experiment/rp_pyrseplstm_k64_combdiv_ft_lr5e4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrseplstm_k64_combdiv_ft_lr5e4`

## Motivation

The fine-tuned cell of the static-comb row of Table VIII: `rp_pyrseplstm_k64_combdiv` from its `best_static_comb_median.ckpt` (step 88 000; the table's synthetic-selected round, the argmin of the median, is 97 000), fine-tuned on the real `rp_all` stream with the 5e-4 recipe of `rp_pyrseplstm_k64_priorv4_ft_lr5e4`. Read against the hard fine-tune (2.15) and the real-only reference (2.52).

## Conclusion

Pending.
