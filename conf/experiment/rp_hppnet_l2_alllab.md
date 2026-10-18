---
experiment: rp_hppnet_l2_alllab
training_config: conf/experiment/rp_hppnet_l2_alllab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_hppnet_l2_alllab`

## Motivation

Label-altering augmentation reference: HPPNet at ladder level L2 (rp linear grid, `salience_layers_rp`), trained on the `rp_all_labelaug` stream; compared against `rp_hppnet_l2_all`. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Max LR 1e-3 — the LR every architecture we build on was published with (docs/experiments/paper-real-protocol.md § "Budget and LR"); the family's LR range test is reported in the appendix. Run history: three attempts on vast crashed/cancelled with the pilot-era L2 collapse (W&B `xrkuy0kj`, `t8o60jqx`); the post-fix attempt `ghziobzg` (2026-10-15, loss with bin 0 unweighted) was cancelled at step 1000 because its vast host ran the loader 9× slower than its sibling (1449 s vs 154 s per 500-step round — the host, not `rate_warp`: `rp_pyrseplstm_alllab` later ran the same stream at full speed). Resumed 2026-10-18 on `uni` from the R2 `train_state.pt` of `ghziobzg` with **`batch_size=64 grad_accum_steps=2`** — `uni` has no 80 GB card and HPPNet at 128 peaks at 26.5 GB allocated / 37 GB reserved; 64×2 is gradient-equivalent for HPPNet (instance norm is per-sample, no BatchNorm), so the protocol's "128, no accumulation" holds in effect.
