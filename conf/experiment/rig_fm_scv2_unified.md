---
experiment: rig_fm_scv2_unified
training_config: conf/experiment/rig_fm_scv2_unified.yaml
batch: docs/experiments/stochastic-fit.md
---

## Motivation

The transfer test of the rig-fitted stochastic family: SimpleConvV2 trained
under the unified regime on synthetic audio only, drawn 50/50 from the DREGON
and Michael's presets of `conf/online_mix/rig_fm_5050.yaml` (FM tone bank on a
jittering shared shaft, blade-passing emphasis, rig floor curves, per-mic
low-band modulation on DREGON, per-mic gain on Michael's), scored on the frozen
real validation panel. Compared with `real_r4_scv2_unified` (real data in
training; best `val/real_r3` 2.99) and the hand-ranged synthetic-only family
(`stoch_s1g_scv2`, 8.08 all-MAE).

## Conclusion

Pending.
