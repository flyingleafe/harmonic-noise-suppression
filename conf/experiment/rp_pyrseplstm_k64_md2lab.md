---
experiment: rp_pyrseplstm_k64_md2lab
training_config: conf/experiment/rp_pyrseplstm_k64_md2lab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_pyrseplstm_k64_md2lab`

## Motivation

Paper, "Transfer between drones": HRSPNet (`hppnet_pyrseplstm_k64`, as `rp_pyrseplstm_k64_alllab`, its better stream on both drones) trained on the MD2 recordings alone (`rp_md2_labelaug`) and scored on DREGON. The run stops on the MD2 score only (`validation.control: [md2]`), so DREGON never steers it; the transfer cell is DREGON (all, cruise) at the round with the best five-round median of `md2`. Reference: `rp_pyrseplstm_k64_alllab` (DREGON 3.89, DREGON cruise 2.97). Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-23; submitted to `uni`.
