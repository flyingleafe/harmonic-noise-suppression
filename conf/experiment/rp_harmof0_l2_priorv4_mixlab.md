---
experiment: rp_harmof0_l2_priorv4_mixlab
training_config: conf/experiment/rp_harmof0_l2_priorv4_mixlab.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_harmof0_l2_priorv4_mixlab`

## Motivation

HarmoF0-L2 trained from scratch on mixed data (Dmitrii, 2026-10-29): 50 % the hard synthetic stream (wide-prior drones with per-clip wind, `rp_priorv4`), 50 % the real stream with the label-altering rate warp (`rp_all_labelaug`), in one online mix (`conf/online_mix/rp_priorv4_mixlab.yaml`; the warp touches the real chunks only). Model, loss, LR schedule, batch, validation panel and stop rule are `rp_harmof0_l2_priorv4`'s. Read against the real-only `rp_harmof0_l2_alllab` and the pretrain-then-tune `rp_harmof0_l2_priorv4_ft_lr5e4` on `real_macro`.

## Conclusion

Done 2026-10-30 (W&B `vtdal2ho`, stopped 64 000): `real_macro` 4.61 at step 30 000, `uniform` 9.97. This is far over the real-only reference (2.62). The error sits on the stopped rotors: zero frames at 9.54 (DREGON) and 23.77 (MD2). Batch doc: `docs/experiments/rp-synthetic.md` § "Mixed training".
