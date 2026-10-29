---
experiment: rp_pyrseplstm_k64_easyv4_mixlab
training_config: conf/experiment/rp_pyrseplstm_k64_easyv4_mixlab.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrseplstm_k64_easyv4_mixlab`

## Motivation

HRSPNet trained from scratch on mixed data (Dmitrii, 2026-10-29): 50 % the easy synthetic stream (drones drawn around the two fits, `rp_easyv4`), 50 % the real stream with the label-altering rate warp (`rp_all_labelaug`), in one online mix (`conf/online_mix/rp_easyv4_mixlab.yaml`; the warp touches the real chunks only). Model, loss, LR schedule, batch, validation panel and stop rule are `rp_pyrseplstm_k64_easyv4`'s. Read against the real-only `rp_pyrseplstm_k64_alllab` and the pretrain-then-tune `rp_pyrseplstm_k64_easyv4_ft_lr5e4` on `real_macro`.

## Conclusion

Pending.
