---
experiment: rp_scv2_dregon
training_config: conf/experiment/rp_scv2_dregon.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_scv2_dregon`

## Motivation

Paper, "Transfer between drones" (claim 5): the SimpleConv-v2 regressor (Conv+BiGRU in the paper; `hb_scv2_mag_nogate`, `pit_mse` loss, `rps` metrics), trained on the `rp_dregon` stream only and scored on both drones of the rp_real panel. Since 2026-10-23 the run stops on the DREGON score only (`validation.control: [dregon]`), so MD2 never steers it; the transfer cell is MD2 (all, cruise) at the round with the best five-round median of `dregon`. Reference: `rp_scv2_all` (MD2 1.49, MD2 cruise 0.93). Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Ran 2026-10-23/24 on `uni` (job `rp-scv2-dregon-f26b2b`, `b734dd28`); stopped by saturation at step 26 000. At the best `dregon` median (step 11 000): DREGON 3.45 (cruise 2.23), better than the both-drone `rp_scv2_all` (3.82); MD2 14.99 against 1.49 for the both-drone model, but MD2 cruise only 4.02 (standby 41.47, ramp 26.04). Paper: "other drone" row of the transfer table. Max LR 1e-3 — the LR every architecture we build on was published with (docs/experiments/paper-real-protocol.md § "Budget and LR").
