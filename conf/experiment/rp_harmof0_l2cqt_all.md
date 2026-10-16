---
experiment: rp_harmof0_l2cqt_all
training_config: conf/experiment/rp_harmof0_l2cqt_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_harmof0_l2cqt_all`

## Motivation

Claim 2 side arm (Dmitrii, 2026-10-16): HarmoF0 L2 with HPPNet's CQT front end (`conf/model/harmof0_l2_cqt.yaml`) in place of the interpolated 64 ms STFT log spectrogram, everything else `rp_harmof0_l2_all`'s. HarmoF0 L2 (2.62) and HPPNet L2 (4.35) on the `all` stream differ in front end, harmonic organ, normalisation and head at once; this arm moves only the front end, on the stronger trunk. Expected to show what the CQT's 1/f window buys (slow airframes) and costs (standby/ramp transitions).

## Conclusion

Defined 2026-10-16; not yet run.
