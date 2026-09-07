---
experiment: hppnet_l1_r2_s0
training_config: conf/experiment/hppnet_l1_r2_s0.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `hppnet_l1_r2_s0`

## Motivation

Block S level L1 for HPPNet: **the finer output grid only**, the step the
LateDeep row took as `hb_sal_multif0_nsr` (12.65 → 11.82) and the two ports
skipped. Now that the matched B batch showed the per-rotor readout (L2) is the
adaptation that matters (HPPNet 7.77 → 2.27) and the comb gather (L3) loses
under it, L1 is the missing rung between L0 and L2: it says how much of the
L0 → L2 gain is output *resolution* and how much is the per-rotor layers with
the CRF readout.

Architecture (`conf/model/hppnet_l1.yaml`): the published HPPNet — CQT,
`HarmonicDilatedConv`, `CNNTrunk`, `FreqGroupLSTM` — with ONE shared map, as
at L0, passed through `FreqSuperResHead` onto LateDeep L1's linear 20–130 rev/s
grid in 720 bins (0.153 rev/s per bin, against 1.45 % of the rate on the native
352-bin log grid). Decoding is L0's threshold + Hungarian on the shared map
(`rps_dump.py`'s shared-map readout); no Gaussian layers, no CRF. Loss and
monitor: `salience_bce_nsr_orig` — LateDeep L1's BCE settings (blur 2,
pos_weight auto) on the ports' hop-512 grid, monitored on `bce` because a
shared map has no `rps_mae` metric (L0 and LateDeep L1 selected the same way;
L2 selects on `rps_mae`, a disclosed difference).

Recipe otherwise identical to `hppnet_l2_r2_s0`: historical R2 honest-base
pool (`e12_real_fullflight` + `hb_silence_dload`, 50k-chunk unaugmented
warm-up), 40,000 frames per validation, 200 epochs, patience 20, batch 16,
seed 0, from scratch. The L2 clamp below the native 27.5 Hz carries over
(output bins 0–48 read the bottom native bin).

Defined 2026-09-07; **not yet run**. Evaluate as the B rows:
`scripts/rps_dump.py --sets real=conf/data/e12_real_fullflight.yaml
--experiments hppnet_l1_r2_s0 --out results/paper_review_B/hppnet_l1_r2_s0`
then `scripts/rps_regime_table.py` on the dump.

## Conclusion

Not run.
