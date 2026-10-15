---
experiment: rp_scv2_alllab
training_config: conf/experiment/rp_scv2_alllab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_scv2_alllab`

## Motivation

Label-altering augmentation reference: the SimpleConv-v2 regressor (`hb_scv2_mag_nogate`, `pit_mse` loss, `rps` metrics), trained on the `rp_all_labelaug` stream; compared against `rp_scv2_all`. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR 3.0e-4 from the family's LR range test (`results/rp_lr_range/rp_scv2_all.json`, read 2026-10-15: loss non-finite at 3.5e-3 (fp16 forward), a tenth of it).
