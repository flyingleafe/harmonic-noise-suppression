---
experiment: rp_scv2_md2
training_config: conf/experiment/rp_scv2_md2.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_scv2_md2`

## Motivation

Paper, "Transfer between drones" (claim 5): the SimpleConv-v2 regressor (Conv+BiGRU in the paper; `hb_scv2_mag_nogate`, `pit_mse` loss, `rps` metrics), trained on the `rp_md2` stream only and scored on both drones of the rp_real panel. Since 2026-10-23 the run stops on the MD2 score only (`validation.control: [md2]`), so DREGON never steers it; the transfer cell is DREGON (all, cruise) at the round with the best five-round median of `md2`. Reference: `rp_scv2_all` (DREGON 3.82, DREGON cruise 2.64). Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Ran 2026-10-24 on `uni` (job `rp-scv2-md2-8ee152`, `b734dd28`); stopped by saturation at step 97 500. At the best `md2` median (step 84 000): MD2 1.30 (cruise 0.87), better than the both-drone `rp_scv2_all` (1.49); DREGON 31.75 (cruise 37.47) against 3.82 for the both-drone model. No transfer to DREGON. Paper: "other drone" row of the transfer table. Max LR 1e-3 — the LR every architecture we build on was published with (docs/experiments/paper-real-protocol.md § "Budget and LR").
