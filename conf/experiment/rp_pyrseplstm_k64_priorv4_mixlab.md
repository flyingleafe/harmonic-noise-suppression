---
experiment: rp_pyrseplstm_k64_priorv4_mixlab
training_config: conf/experiment/rp_pyrseplstm_k64_priorv4_mixlab.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrseplstm_k64_priorv4_mixlab`

## Motivation

HRSPNet trained from scratch on mixed data (Dmitrii, 2026-10-29): 50 % the hard synthetic stream (wide-prior drones with per-clip wind, `rp_priorv4`), 50 % the real stream with the label-altering rate warp (`rp_all_labelaug`), in one online mix (`conf/online_mix/rp_priorv4_mixlab.yaml`; the warp touches the real chunks only). Model, loss, LR schedule, batch, validation panel and stop rule are `rp_pyrseplstm_k64_priorv4`'s. Read against the real-only `rp_pyrseplstm_k64_alllab` and the pretrain-then-tune `rp_pyrseplstm_k64_priorv4_ft_lr5e4` on `real_macro`.

## Conclusion

Done 2026-10-30 (W&B `tykzoxrx`, stopped 78 500): `real_macro` 2.24 at step 55 000, `uniform` 1.59. This is 0.28 Hz under the real-only reference (2.52) and 0.09 Hz over the hard fine-tune (2.15), which keeps `uniform` only at 9.05. Batch doc: `docs/experiments/rp-synthetic.md` § "Mixed training".
