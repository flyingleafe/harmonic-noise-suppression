---
experiment: rp_hppnet_l2_priorv4_mixlab
training_config: conf/experiment/rp_hppnet_l2_priorv4_mixlab.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_hppnet_l2_priorv4_mixlab`

## Motivation

HPPNet-L2 trained from scratch on mixed data (Dmitrii, 2026-10-29): 50 % the hard synthetic stream (wide-prior drones with per-clip wind, `rp_priorv4`), 50 % the real stream with the label-altering rate warp (`rp_all_labelaug`), in one online mix (`conf/online_mix/rp_priorv4_mixlab.yaml`; the warp touches the real chunks only). Model, loss, LR schedule, validation panel and stop rule are `rp_hppnet_l2_priorv4`'s; batch 64 × 2 accumulation (= 128) so it fits any `uni` GPU. Read against the real-only `rp_hppnet_l2_alllab` and the pretrain-then-tune `rp_hppnet_l2_priorv4_ft_lr5e4` on `real_macro`.

## Conclusion

Done 2026-10-30 (W&B `hdy8j35o`, stopped 77 000): `real_macro` 4.17 at step 59 500, `uniform` 3.38. This is over the real-only reference (3.39). The error sits on the stopped rotors: zero frames at 22.88 (DREGON) and 17.63 (MD2). Batch doc: `docs/experiments/rp-synthetic.md` § "Mixed training".
