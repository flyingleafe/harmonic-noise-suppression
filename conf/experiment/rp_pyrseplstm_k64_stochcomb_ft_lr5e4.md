---
experiment: rp_pyrseplstm_k64_stochcomb_ft_lr5e4
training_config: conf/experiment/rp_pyrseplstm_k64_stochcomb_ft_lr5e4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrseplstm_k64_stochcomb_ft_lr5e4`

## Motivation

The fine-tuned cell of the stochastic-comb row of Table VIII: `rp_pyrseplstm_k64_stochcomb` from its `best_stoch_comb_median.ckpt`, fine-tuned on the real `rp_all` stream with the 5e-4 recipe of `rp_pyrseplstm_k64_priorv4_ft_lr5e4` (1000 warm-up steps, cosine to 20 000). Read against the hard fine-tune (2.15) and the real-only reference (2.52).

## Conclusion

Pending.
