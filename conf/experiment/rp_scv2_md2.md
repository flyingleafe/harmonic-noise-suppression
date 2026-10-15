---
experiment: rp_scv2_md2
training_config: conf/experiment/rp_scv2_md2.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_scv2_md2`

## Motivation

Claim 5 (no cross-drone transfer): the SimpleConv-v2 regressor (`hb_scv2_mag_nogate`, `pit_mse` loss, `rps` metrics), trained on the `rp_md2` stream only and scored on both drones of the rp_real panel. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR 1.0e-3 from the family's LR range test (`results/rp_lr_range/rp_scv2_all.json`, read 2026-10-15: loss non-finite at 3.5e-3, a third of it; every earlier SCv2 run trained at 1e-3).
