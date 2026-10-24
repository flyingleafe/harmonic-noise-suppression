---
experiment: rp_pyrseplstm_k64_md2lab
training_config: conf/experiment/rp_pyrseplstm_k64_md2lab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_pyrseplstm_k64_md2lab`

## Motivation

Paper, "Transfer between drones": HRSPNet (`hppnet_pyrseplstm_k64`, as `rp_pyrseplstm_k64_alllab`, its better stream on both drones) trained on the MD2 recordings alone (`rp_md2_labelaug`) and scored on DREGON. The run stops on the MD2 score only (`validation.control: [md2]`), so DREGON never steers it; the transfer cell is DREGON (all, cruise) at the round with the best five-round median of `md2`. Reference: `rp_pyrseplstm_k64_alllab` (DREGON 3.89, DREGON cruise 2.97). Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Ran 2026-10-24 on `uni` (job `rp-pyrseplstm-k64-md2lab-b481c0`, `b734dd28`); stopped by saturation at step 59 500. At the best `md2` median (step 44 500): MD2 0.82 (cruise 0.67), better than the both-drone `rp_pyrseplstm_k64_alllab` (1.14); DREGON 12.66 (cruise 13.31) against 3.89 for the both-drone model. No transfer to DREGON, though far less wrong there than `rp_scv2_md2` (31.75). Paper: "other drone" row of the transfer table.
