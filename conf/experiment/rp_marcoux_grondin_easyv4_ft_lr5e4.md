---
experiment: rp_marcoux_grondin_easyv4_ft_lr5e4
training_config: conf/experiment/rp_marcoux_grondin_easyv4_ft_lr5e4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_marcoux_grondin_easyv4_ft_lr5e4`

## Motivation

Real-data fine-tuning of `rp_marcoux_grondin_easyv4` with the grid's recipe: warm start from `best_near_macro_median`, `rp_all` without label augmentation, fresh AdamW at max LR 5e-4, 1 000 warm-up steps, cosine to the 20 000-step cap, real validation panel. Three seeds on fixed pretrained weights: seed 0 is this config; seeds 1 and 2 add `experiment_name=<exp>_s<n> seed=<n> +data.train.params.base_seed=2026101{5,6}`.

## Conclusion

Defined 2026-11-27; queued on `isambard` after its pretraining. Pending.
