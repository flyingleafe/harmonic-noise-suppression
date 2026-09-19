# DREGON round 3: does HPPNet track BROAD harmonic humps?

Windows free-flight, hovering, updown; seed 2001, 8 mics; fit `results/noise_v2/rounds/round3/fits/dregon_room2_floor__flight_floor_lowk.json` (`converged` None); git `e51ff57602ac769320bd52fb73d7fedc398522f5`; probe job `nv2-r3-humps-671f38`.

## Verdict

* **Is real DREGON's carrier-locked power BROADER than a needle comb's?** NO (0/3 windows).
* **Is that width the thing HPPNet needs — at the same hump fraction?** NO (0/3 windows: a widened comb beats the needle comb at the matched k<=8 carrier-locked power).
* **Is that width the thing HPPNet needs — at the same comb POWER (21 dB up, where the needle comb is tracked)?** NO (0/3 windows).
* **Is comb LEVEL the thing HPPNet needs?** YES (3/3 windows: re-levelling the needle comb alone at least halves the PIT MAE).

## The arms

| arm | tag | what it is |
|---|---|---|
| `real` | real | the real DREGON room-2 clip |
| `legacy` | legacy | legacy stage-2 baseline, identity-matched |
| `legacy_needle` | V7 | legacy with the Lorentzian PEDESTAL muted (coherence_k_half = 0: every order's power goes through the tone bank as a needle) |
| `v2` | v2 | the round-3 v2 candidate, unchanged (bench gamma_rk) |
| `v2_g13` | V0 | v2 with gamma_rk overridden to a k-independent 13 Hz, comb level as fitted |
| `v2_g30` | V1 | v2 with gamma_rk overridden to a k-independent 30 Hz, comb level as fitted |
| `v2_g67` | V2 | v2 with gamma_rk overridden to a k-independent 67 Hz, comb level as fitted |
| `v2_g130` | V3 | v2 with gamma_rk overridden to a k-independent 130 Hz, comb level as fitted |
| `v2_matched` | Vm | v2 at the FITTED (needle) widths with the comb re-levelled onto the real window's k<=8 hump fraction — the control that isolates width from level |
| `v2_g13_matched` | V0m | v2 at a k-independent 13 Hz with the comb re-levelled onto the real window's k<=8 hump fraction |
| `v2_g30_matched` | V4 | v2 at a k-independent 30 Hz with the comb re-levelled onto the real window's k<=8 hump fraction |
| `v2_g67_matched` | V5 | v2 at a k-independent 67 Hz with the comb re-levelled onto the real window's k<=8 hump fraction |
| `v2_g130_matched` | V6 | v2 at a k-independent 130 Hz with the comb re-levelled onto the real window's k<=8 hump fraction |
| `v2_plus6db` | L+6 | v2 at the fitted widths with the comb 6 dB up: the level dose-response |
| `v2_plus12db` | L+12 | v2 at the fitted widths with the comb 12 dB up: the level dose-response |
| `v2_plus18db` | L+18 | v2 at the fitted widths with the comb 18 dB up: the level dose-response |
| `v2_plus21db` | L+21 | v2 at the fitted widths with the comb 21 dB up: the level dose-response |
| `v2_plus24db` | L+24 | v2 at the fitted widths with the comb 24 dB up: the level dose-response |
| `v2_g13_plus21db` | W13 | v2 at a k-independent 13 Hz with the comb 21 dB up — the width comparison at EQUAL comb power, at the level where the needle comb is tracked |
| `v2_g30_plus21db` | W30 | v2 at a k-independent 30 Hz with the comb 21 dB up — the width comparison at EQUAL comb power, at the level where the needle comb is tracked |
| `v2_g67_plus21db` | W67 | v2 at a k-independent 67 Hz with the comb 21 dB up — the width comparison at EQUAL comb power, at the level where the needle comb is tracked |
| `v2_g130_plus21db` | W130 | v2 at a k-independent 130 Hz with the comb 21 dB up — the width comparison at EQUAL comb power, at the level where the needle comb is tracked |

## HPPNet PIT MAE (rev/s)

| arm | tag | gamma_rk | comb shift dB | free-flight | hovering | updown | mean |
|---|---|---|---:|---:|---:|---:|---:|
| `real` | real | as fitted | — | 0.638 | 0.888 | 1.695 | 1.074 |
| `legacy` | legacy | as fitted | — | 1.496 | 1.895 | 2.499 | 1.963 |
| `legacy_needle` | V7 | as fitted | — | 2.034 | 1.857 | 2.188 | 2.027 |
| `v2` | v2 | as fitted | +0.00 | 62.105 | 73.448 | 73.112 | 69.555 |
| `v2_g13` | V0 | 13 Hz | +0.00 | 80.370 | 80.874 | 77.198 | 79.481 |
| `v2_g30` | V1 | 30 Hz | +0.00 | 79.354 | 80.927 | 78.991 | 79.757 |
| `v2_g67` | V2 | 67 Hz | +0.00 | 80.369 | 80.066 | 79.275 | 79.903 |
| `v2_g130` | V3 | 130 Hz | +0.00 | 80.369 | 80.925 | 79.272 | 80.189 |
| `v2_matched` | Vm | as fitted | -1.73 | 75.386 | 71.511 | 79.273 | 75.390 |
| `v2_g13_matched` | V0m | 13 Hz | +4.04 | 77.940 | 58.385 | 77.246 | 71.190 |
| `v2_g30_matched` | V4 | 30 Hz | +42.00 | 13.966 | 8.461 | 7.725 | 10.050 |
| `v2_g67_matched` | V5 | 67 Hz | -12.00 | 80.368 | 80.925 | 79.271 | 80.188 |
| `v2_g130_matched` | V6 | 130 Hz | -12.00 | 80.367 | 80.925 | 79.271 | 80.188 |
| `v2_plus6db` | L+6 | as fitted | +6.00 | 41.822 | 49.692 | 48.285 | 46.600 |
| `v2_plus12db` | L+12 | as fitted | +12.00 | 30.370 | 31.850 | 38.911 | 33.710 |
| `v2_plus18db` | L+18 | as fitted | +18.00 | 4.341 | 2.151 | 8.812 | 5.101 |
| `v2_plus21db` | L+21 | as fitted | +21.00 | 5.118 | 2.759 | 6.800 | 4.893 |
| `v2_plus24db` | L+24 | as fitted | +24.00 | 6.449 | 5.776 | 6.450 | 6.225 |
| `v2_g13_plus21db` | W13 | 13 Hz | +21.00 | 27.057 | 20.562 | 13.678 | 20.432 |
| `v2_g30_plus21db` | W30 | 30 Hz | +21.00 | 37.616 | 34.122 | 27.444 | 33.061 |
| `v2_g67_plus21db` | W67 | 67 Hz | +21.00 | 28.263 | 25.802 | 25.957 | 26.674 |
| `v2_g130_plus21db` | W130 | 130 Hz | +21.00 | 25.375 | 26.453 | 29.656 | 27.161 |

`comb shift dB` is the shift applied to `profile_db`: the solved hump-fraction match for the `_matched` arms, the swept constant for the `v2_plusNdb` arms, 0 for the fitted level. Every arm is ONE seed (2001), so these are not the four-seed gate means.

## Hump fraction: power inside the k x fbar +- 64 Hz band set

Reference-free union of the twelve (eight) bands on each frame's own mean label carrier, as a fraction of 30-7900 Hz band power, mic-MEAN on the 2048/512 front end. At a DREGON cruise carrier (fbar ~ 80 Hz) the 128 Hz-wide bands OVERLAP: the union is nearly the contiguous 30-1150 Hz region, which is why its own bandwidth fraction and the concentration (their ratio) are beside it. `tracked k=2..8` is the tilt-corrected per-order excess the level match is solved on.

| window | arm | frac k<=8 | frac k<=12 | bandwidth frac k<=8 | concentration k<=8 | tracked k=2..8 @64 Hz (%) | mic0 frac k<=8 |
|---|---|---:|---:|---:|---:|---:|---:|
| `free-flight` | `real` | 0.8632 | 0.8893 | 0.0863 | 10.00 | +16.66 | 0.8775 |
| `free-flight` | `legacy` | 0.9373 | 0.9537 | 0.0863 | 10.86 | +74.79 | 0.9644 |
| `free-flight` | `legacy_needle` | 0.9381 | 0.9547 | 0.0863 | 10.87 | +96.43 | 0.9637 |
| `free-flight` | `v2` | 0.8780 | 0.9028 | 0.0863 | 10.17 | +21.53 | 0.8739 |
| `free-flight` | `v2_g13` | 0.8782 | 0.9028 | 0.0863 | 10.17 | +10.72 | 0.8734 |
| `free-flight` | `v2_g30` | 0.8739 | 0.8994 | 0.0863 | 10.12 | +5.57 | 0.8727 |
| `free-flight` | `v2_g67` | 0.8676 | 0.8950 | 0.0863 | 10.05 | +4.13 | 0.8725 |
| `free-flight` | `v2_g130` | 0.8611 | 0.8906 | 0.0863 | 9.97 | +4.96 | 0.8720 |
| `free-flight` | `v2_matched` | 0.8767 | 0.9016 | 0.0863 | 10.16 | +16.63 | 0.8737 |
| `free-flight` | `v2_g13_matched` | 0.8812 | 0.9055 | 0.0863 | 10.21 | +16.68 | 0.8737 |
| `free-flight` | `v2_g30_matched` | 0.8728 | 0.9014 | 0.0863 | 10.11 | +9.62 | 0.8754 |
| `free-flight` | `v2_g67_matched` | 0.8735 | 0.8988 | 0.0863 | 10.12 | +4.66 | 0.8730 |
| `free-flight` | `v2_g130_matched` | 0.8732 | 0.8986 | 0.0863 | 10.12 | +4.87 | 0.8730 |
| `free-flight` | `v2_plus6db` | 0.8861 | 0.9097 | 0.0863 | 10.26 | +47.73 | 0.8753 |
| `free-flight` | `v2_plus12db` | 0.8957 | 0.9178 | 0.0863 | 10.38 | +76.23 | 0.8792 |
| `free-flight` | `v2_plus18db` | 0.9010 | 0.9221 | 0.0863 | 10.44 | +90.25 | 0.8877 |
| `free-flight` | `v2_plus21db` | 0.9022 | 0.9231 | 0.0863 | 10.45 | +93.05 | 0.8930 |
| `free-flight` | `v2_plus24db` | 0.9029 | 0.9237 | 0.0863 | 10.46 | +94.49 | 0.8975 |
| `free-flight` | `v2_g13_plus21db` | 0.8904 | 0.9139 | 0.0863 | 10.31 | +37.85 | 0.8848 |
| `free-flight` | `v2_g30_plus21db` | 0.8728 | 0.9012 | 0.0863 | 10.11 | +9.37 | 0.8726 |
| `free-flight` | `v2_g67_plus21db` | 0.8413 | 0.8793 | 0.0863 | 9.75 | +1.51 | 0.8549 |
| `free-flight` | `v2_g130_plus21db` | 0.7979 | 0.8497 | 0.0863 | 9.24 | +3.35 | 0.8287 |
| `hovering` | `real` | 0.8752 | 0.9003 | 0.0869 | 10.08 | +22.28 | 0.9275 |
| `hovering` | `legacy` | 0.9356 | 0.9482 | 0.0869 | 10.77 | +74.70 | 0.9730 |
| `hovering` | `legacy_needle` | 0.9360 | 0.9491 | 0.0869 | 10.78 | +94.74 | 0.9736 |
| `hovering` | `v2` | 0.8802 | 0.9043 | 0.0869 | 10.13 | +21.35 | 0.8746 |
| `hovering` | `v2_g13` | 0.8761 | 0.9012 | 0.0869 | 10.09 | +9.53 | 0.8737 |
| `hovering` | `v2_g30` | 0.8738 | 0.8996 | 0.0869 | 10.06 | +4.93 | 0.8739 |
| `hovering` | `v2_g67` | 0.8688 | 0.8962 | 0.0869 | 10.00 | +3.88 | 0.8744 |
| `hovering` | `v2_g130` | 0.8621 | 0.8915 | 0.0869 | 9.93 | +3.39 | 0.8721 |
| `hovering` | `v2_matched` | 0.8805 | 0.9046 | 0.0869 | 10.14 | +22.31 | 0.8746 |
| `hovering` | `v2_g13_matched` | 0.8816 | 0.9063 | 0.0869 | 10.15 | +22.26 | 0.8741 |
| `hovering` | `v2_g30_matched` | 0.8703 | 0.9004 | 0.0869 | 10.02 | +7.32 | 0.8653 |
| `hovering` | `v2_g67_matched` | 0.8742 | 0.8995 | 0.0869 | 10.07 | +4.72 | 0.8740 |
| `hovering` | `v2_g130_matched` | 0.8738 | 0.8992 | 0.0869 | 10.06 | +4.47 | 0.8735 |
| `hovering` | `v2_plus6db` | 0.8891 | 0.9120 | 0.0869 | 10.24 | +47.33 | 0.8759 |
| `hovering` | `v2_plus12db` | 0.8992 | 0.9208 | 0.0869 | 10.35 | +75.92 | 0.8798 |
| `hovering` | `v2_plus18db` | 0.9044 | 0.9253 | 0.0869 | 10.41 | +90.36 | 0.8882 |
| `hovering` | `v2_plus21db` | 0.9055 | 0.9263 | 0.0869 | 10.43 | +93.30 | 0.8935 |
| `hovering` | `v2_plus24db` | 0.9061 | 0.9268 | 0.0869 | 10.43 | +94.82 | 0.8982 |
| `hovering` | `v2_g13_plus21db` | 0.8894 | 0.9135 | 0.0869 | 10.24 | +36.75 | 0.8818 |
| `hovering` | `v2_g30_plus21db` | 0.8703 | 0.9002 | 0.0869 | 10.02 | +7.13 | 0.8699 |
| `hovering` | `v2_g67_plus21db` | 0.8415 | 0.8795 | 0.0869 | 9.69 | -1.08 | 0.8628 |
| `hovering` | `v2_g130_plus21db` | 0.7981 | 0.8496 | 0.0869 | 9.19 | +0.09 | 0.8205 |
| `updown` | `real` | 0.8736 | 0.9052 | 0.0853 | 10.24 | +10.16 | 0.9170 |
| `updown` | `legacy` | 0.9363 | 0.9486 | 0.0853 | 10.97 | +66.32 | 0.9740 |
| `updown` | `legacy_needle` | 0.9391 | 0.9507 | 0.0853 | 11.00 | +94.39 | 0.9748 |
| `updown` | `v2` | 0.8782 | 0.9030 | 0.0853 | 10.29 | +23.24 | 0.8712 |
| `updown` | `v2_g13` | 0.8779 | 0.9023 | 0.0853 | 10.29 | +11.18 | 0.8718 |
| `updown` | `v2_g30` | 0.8723 | 0.8985 | 0.0853 | 10.22 | +4.88 | 0.8708 |
| `updown` | `v2_g67` | 0.8684 | 0.8955 | 0.0853 | 10.18 | +3.15 | 0.8703 |
| `updown` | `v2_g130` | 0.8597 | 0.8895 | 0.0853 | 10.07 | +4.54 | 0.8706 |
| `updown` | `v2_matched` | 0.8743 | 0.8995 | 0.0853 | 10.24 | +10.16 | 0.8711 |
| `updown` | `v2_g13_matched` | 0.8773 | 0.9018 | 0.0853 | 10.28 | +10.18 | 0.8717 |
| `updown` | `v2_g30_matched` | 0.8743 | 0.9030 | 0.0853 | 10.24 | +6.32 | 0.8687 |
| `updown` | `v2_g67_matched` | 0.8730 | 0.8983 | 0.0853 | 10.23 | +4.62 | 0.8710 |
| `updown` | `v2_g130_matched` | 0.8723 | 0.8979 | 0.0853 | 10.22 | +5.06 | 0.8712 |
| `updown` | `v2_plus6db` | 0.8877 | 0.9112 | 0.0853 | 10.40 | +50.87 | 0.8719 |
| `updown` | `v2_plus12db` | 0.8984 | 0.9203 | 0.0853 | 10.53 | +79.74 | 0.8750 |
| `updown` | `v2_plus18db` | 0.9040 | 0.9250 | 0.0853 | 10.59 | +93.32 | 0.8838 |
| `updown` | `v2_plus21db` | 0.9053 | 0.9260 | 0.0853 | 10.61 | +96.00 | 0.8897 |
| `updown` | `v2_plus24db` | 0.9060 | 0.9265 | 0.0853 | 10.62 | +97.37 | 0.8952 |
| `updown` | `v2_g13_plus21db` | 0.8907 | 0.9144 | 0.0853 | 10.44 | +37.92 | 0.8810 |
| `updown` | `v2_g30_plus21db` | 0.8738 | 0.9025 | 0.0853 | 10.24 | +6.38 | 0.8681 |
| `updown` | `v2_g67_plus21db` | 0.8447 | 0.8815 | 0.0853 | 9.90 | +1.37 | 0.8531 |
| `updown` | `v2_g130_plus21db` | 0.7947 | 0.8464 | 0.0853 | 9.31 | +0.88 | 0.8252 |

## Label-tracked per-order excess (% of band power, mic-mean)

Band minus a two-sided log-linear local baseline (the medians of the `B < |f - k fbar| <= 2.5 B` annuli, interpolated in log f), summed over frames before the ratio. `null` repeats the same read HALFWAY between two orders, at the same frequencies and under the same estimator: it is the zero this table is read against. An order whose annuli fall outside 30-7900 Hz is `—` (k=1 at B=64 and B=128, k=2 at B=128).

### B = 4 Hz

| window | arm | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 | k=11 | k=12 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free-flight` | `real` | +1.79 | +0.02 | +0.02 | +0.02 | +0.00 | +0.04 | -0.00 | +0.01 | +0.00 | +0.01 | +0.00 | +0.00 |
| `free-flight` | `real` null | -0.00 | +0.01 | +0.01 | -0.01 | +0.01 | -0.00 | +0.00 | +0.00 | -0.00 | +0.00 | -0.00 | -0.00 |
| `free-flight` | `legacy` | +10.78 | +0.09 | +0.04 | +0.05 | -0.01 | +0.02 | +0.01 | +0.01 | +0.00 | +0.00 | +0.00 | +0.00 |
| `free-flight` | `legacy_needle` | +14.16 | +0.32 | +0.51 | +0.35 | +0.04 | +0.04 | +0.01 | -0.02 | -0.01 | -0.01 | -0.00 | -0.00 |
| `free-flight` | `v2` | +2.03 | +0.05 | +0.06 | +0.01 | +0.01 | +0.01 | -0.00 | -0.00 | -0.00 | -0.00 | -0.00 | -0.00 |
| `free-flight` | `v2_matched` | +1.41 | +0.04 | +0.06 | +0.01 | +0.01 | +0.01 | -0.00 | -0.00 | -0.00 | -0.00 | -0.00 | +0.00 |
| `hovering` | `real` | +0.66 | +0.32 | -0.02 | +0.02 | +0.00 | +0.01 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.01 |
| `hovering` | `real` null | -0.05 | -0.07 | -0.06 | +0.01 | +0.01 | +0.01 | -0.00 | -0.00 | -0.00 | +0.00 | +0.00 | +0.00 |
| `hovering` | `legacy` | +6.05 | +0.09 | +0.00 | +0.01 | +0.00 | +0.00 | -0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| `hovering` | `legacy_needle` | +7.72 | +0.43 | +0.11 | -0.01 | -0.01 | -0.01 | -0.00 | -0.01 | +0.00 | +0.00 | -0.00 | +0.00 |
| `hovering` | `v2` | +1.33 | +0.24 | +0.03 | +0.01 | +0.01 | +0.01 | +0.00 | +0.00 | -0.00 | -0.00 | -0.00 | +0.00 |
| `hovering` | `v2_matched` | +1.40 | +0.26 | +0.03 | +0.01 | +0.01 | +0.01 | +0.00 | +0.00 | -0.00 | -0.00 | -0.00 | +0.00 |
| `updown` | `real` | +0.31 | +0.09 | +0.03 | +0.00 | +0.00 | +0.00 | +0.01 | +0.00 | +0.01 | +0.01 | +0.00 | +0.00 |
| `updown` | `real` null | +0.14 | +0.07 | +0.01 | -0.01 | +0.00 | -0.00 | -0.00 | -0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| `updown` | `legacy` | +8.71 | +0.06 | +0.01 | -0.01 | +0.00 | +0.00 | +0.00 | +0.00 | -0.00 | -0.00 | -0.00 | +0.00 |
| `updown` | `legacy_needle` | +15.54 | +0.24 | +0.09 | -0.03 | -0.01 | -0.02 | -0.00 | -0.01 | -0.00 | -0.00 | -0.00 | -0.00 |
| `updown` | `v2` | +2.99 | +0.25 | +0.03 | +0.01 | +0.01 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| `updown` | `v2_matched` | +0.83 | +0.08 | +0.03 | +0.01 | +0.01 | +0.01 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |

### B = 16 Hz

| window | arm | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 | k=11 | k=12 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free-flight` | `real` | +12.68 | +1.89 | +0.39 | -0.05 | +0.21 | +0.45 | -0.08 | +0.22 | +0.09 | +0.12 | +0.06 | +0.07 |
| `free-flight` | `real` null | -0.54 | -0.35 | +0.12 | +0.04 | -0.11 | -0.12 | +0.12 | -0.09 | -0.04 | -0.02 | -0.07 | -0.04 |
| `free-flight` | `legacy` | +56.43 | +7.30 | +1.07 | +0.93 | +0.31 | +0.30 | +0.09 | +0.17 | +0.05 | +0.08 | +0.01 | +0.01 |
| `free-flight` | `legacy_needle` | +67.61 | +14.62 | +3.08 | +2.57 | +0.96 | +0.78 | +0.22 | +0.36 | +0.17 | +0.19 | +0.03 | +0.03 |
| `free-flight` | `v2` | +16.37 | +3.17 | +0.20 | +0.17 | +0.12 | +0.03 | +0.02 | +0.01 | +0.03 | +0.02 | +0.03 | +0.01 |
| `free-flight` | `v2_matched` | +13.13 | +2.22 | +0.20 | +0.17 | +0.13 | +0.03 | +0.02 | +0.00 | +0.02 | +0.02 | +0.02 | +0.01 |
| `hovering` | `real` | +16.23 | +3.72 | +0.37 | +0.06 | -0.03 | +0.18 | -0.02 | +0.06 | +0.04 | +0.05 | +0.02 | +0.03 |
| `hovering` | `real` null | -1.22 | -0.76 | +0.10 | +0.12 | +0.07 | -0.06 | +0.06 | -0.02 | +0.00 | -0.00 | -0.01 | +0.04 |
| `hovering` | `legacy` | +60.08 | +5.79 | +0.86 | +0.49 | +0.11 | +0.05 | -0.02 | +0.03 | +0.01 | +0.00 | -0.01 | -0.00 |
| `hovering` | `legacy_needle` | +70.66 | +12.35 | +2.62 | +1.47 | +0.41 | +0.11 | -0.09 | +0.07 | +0.06 | +0.06 | -0.03 | +0.01 |
| `hovering` | `v2` | +16.62 | +2.98 | +0.21 | +0.14 | +0.13 | +0.03 | +0.01 | +0.01 | +0.02 | +0.01 | +0.01 | -0.01 |
| `hovering` | `v2_matched` | +17.29 | +3.16 | +0.21 | +0.14 | +0.12 | +0.03 | +0.01 | +0.01 | +0.02 | +0.01 | +0.01 | -0.01 |
| `updown` | `real` | +6.16 | +2.50 | +0.09 | +0.16 | +0.09 | +0.27 | -0.00 | +0.14 | +0.07 | +0.08 | +0.03 | +0.04 |
| `updown` | `real` null | -0.81 | -0.11 | +0.29 | +0.03 | -0.04 | -0.12 | +0.06 | -0.07 | +0.02 | -0.01 | -0.00 | +0.03 |
| `updown` | `legacy` | +51.83 | +5.42 | +1.04 | +0.39 | +0.02 | +0.02 | -0.04 | +0.01 | -0.02 | -0.02 | -0.02 | -0.01 |
| `updown` | `legacy_needle` | +70.95 | +11.90 | +2.86 | +1.21 | +0.24 | +0.03 | -0.13 | -0.05 | -0.06 | -0.06 | -0.04 | -0.02 |
| `updown` | `v2` | +17.14 | +3.45 | +0.20 | +0.13 | +0.12 | +0.05 | -0.01 | +0.00 | +0.02 | +0.02 | +0.02 | -0.01 |
| `updown` | `v2_matched` | +8.72 | +1.08 | +0.21 | +0.12 | +0.13 | +0.05 | -0.01 | -0.00 | +0.02 | +0.02 | +0.01 | -0.01 |

### B = 64 Hz

| window | arm | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 | k=11 | k=12 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free-flight` | `real` | — | +10.30 | +2.35 | +0.86 | +0.44 | +0.33 | +0.37 | +0.42 | +0.16 | +0.43 | -0.16 | +0.08 |
| `free-flight` | `real` null | +26.87 | +6.98 | +1.55 | +0.46 | +0.83 | +0.29 | +0.63 | +0.31 | +0.51 | +0.24 | +0.02 | +0.35 |
| `free-flight` | `legacy` | — | +8.67 | -0.54 | +0.28 | -0.05 | +0.11 | -0.20 | +0.16 | +0.18 | +0.28 | -0.17 | -0.02 |
| `free-flight` | `legacy_needle` | — | +13.25 | +2.11 | +2.54 | +1.09 | +0.89 | +0.40 | +0.59 | +0.40 | +0.58 | -0.01 | +0.17 |
| `free-flight` | `v2` | — | +10.16 | -0.36 | +0.75 | +0.39 | +0.18 | -0.07 | -0.04 | +0.03 | -0.03 | +0.06 | +0.08 |
| `free-flight` | `v2_matched` | — | +9.77 | -0.10 | +0.80 | +0.43 | +0.20 | -0.07 | -0.04 | +0.05 | -0.03 | +0.03 | +0.06 |
| `hovering` | `real` | — | +12.23 | +1.71 | +1.28 | +0.38 | +0.27 | +0.24 | +0.19 | +0.26 | +0.39 | -0.13 | +0.12 |
| `hovering` | `real` null | +32.67 | +8.14 | +1.48 | +0.56 | +0.34 | +0.25 | +0.23 | +0.20 | +0.47 | +0.16 | -0.02 | +0.17 |
| `hovering` | `legacy` | — | +7.72 | -0.07 | +0.23 | +0.24 | +0.12 | -0.13 | +0.10 | +0.18 | +0.26 | -0.10 | -0.05 |
| `hovering` | `legacy_needle` | — | +11.46 | +2.37 | +2.18 | +1.24 | +0.69 | +0.21 | +0.38 | +0.34 | +0.45 | -0.06 | +0.10 |
| `hovering` | `v2` | — | +10.15 | -0.35 | +0.69 | +0.43 | +0.13 | -0.07 | -0.01 | +0.03 | +0.00 | +0.05 | +0.08 |
| `hovering` | `v2_matched` | — | +10.23 | -0.40 | +0.68 | +0.43 | +0.13 | -0.07 | -0.01 | +0.03 | +0.00 | +0.06 | +0.08 |
| `updown` | `real` | — | +11.35 | +2.62 | +2.11 | +0.58 | +0.11 | +0.14 | +0.30 | +0.26 | +0.47 | -0.10 | +0.15 |
| `updown` | `real` null | +22.78 | +7.00 | +2.59 | +1.19 | +0.52 | +0.15 | +0.40 | +0.26 | +0.57 | +0.18 | +0.05 | +0.13 |
| `updown` | `legacy` | — | +8.68 | +0.35 | +0.19 | +0.31 | +0.23 | -0.08 | +0.10 | +0.19 | +0.24 | -0.06 | -0.04 |
| `updown` | `legacy_needle` | — | +10.99 | +2.41 | +2.35 | +1.21 | +0.69 | +0.28 | +0.39 | +0.40 | +0.48 | +0.08 | +0.13 |
| `updown` | `v2` | — | +10.89 | -0.24 | +0.71 | +0.39 | +0.13 | -0.05 | -0.02 | +0.03 | -0.02 | +0.05 | +0.12 |
| `updown` | `v2_matched` | — | +10.14 | +0.48 | +0.85 | +0.45 | +0.18 | -0.03 | -0.02 | +0.07 | -0.01 | -0.01 | +0.06 |

### B = 128 Hz

| window | arm | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 | k=11 | k=12 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free-flight` | `real` | — | +45.88 | +15.89 | +6.26 | +2.44 | +1.52 | +1.45 | +1.12 | +1.01 | +0.20 | +0.06 | +0.01 |
| `free-flight` | `real` null | — | +33.88 | +10.58 | +3.91 | +1.48 | +1.50 | +1.39 | +1.21 | +0.69 | +0.33 | +0.29 | +0.11 |
| `free-flight` | `legacy` | — | +74.63 | +12.50 | +2.14 | +1.03 | -0.28 | +0.36 | +0.55 | +0.86 | +0.32 | +0.07 | -0.22 |
| `free-flight` | `legacy_needle` | — | +80.38 | +18.66 | +5.55 | +4.83 | +2.44 | +2.11 | +1.60 | +1.65 | +0.84 | +0.50 | +0.18 |
| `free-flight` | `v2` | — | +50.49 | +12.85 | +4.07 | +1.80 | +0.48 | -0.32 | -0.49 | -0.39 | -0.06 | +0.10 | +0.25 |
| `free-flight` | `v2_matched` | — | +48.36 | +12.54 | +4.48 | +2.00 | +0.55 | -0.27 | -0.45 | -0.36 | -0.07 | +0.04 | +0.17 |
| `hovering` | `real` | — | +47.76 | +17.84 | +5.81 | +2.21 | +0.97 | +0.68 | +0.73 | +0.70 | +0.33 | +0.14 | -0.01 |
| `hovering` | `real` null | — | +39.09 | +11.48 | +3.69 | +1.26 | +0.56 | +0.77 | +0.87 | +0.59 | +0.25 | +0.19 | +0.06 |
| `hovering` | `legacy` | — | +73.71 | +10.58 | +2.30 | +0.59 | -0.17 | +0.06 | +0.44 | +0.68 | +0.27 | -0.07 | -0.21 |
| `hovering` | `legacy_needle` | — | +77.70 | +15.63 | +4.89 | +3.82 | +1.90 | +1.47 | +1.15 | +1.18 | +0.59 | +0.22 | +0.01 |
| `hovering` | `v2` | — | +48.28 | +12.57 | +3.92 | +1.75 | +0.46 | -0.27 | -0.40 | -0.35 | -0.07 | +0.03 | +0.20 |
| `hovering` | `v2_matched` | — | +48.70 | +12.62 | +3.84 | +1.72 | +0.44 | -0.28 | -0.41 | -0.36 | -0.07 | +0.04 | +0.21 |
| `updown` | `real` | — | +31.96 | +19.72 | +9.04 | +3.56 | +0.91 | +0.52 | +0.86 | +1.09 | +0.47 | +0.33 | -0.06 |
| `updown` | `real` null | — | +32.80 | +14.56 | +6.07 | +1.89 | +0.61 | +0.79 | +1.07 | +0.88 | +0.39 | +0.21 | -0.06 |
| `updown` | `legacy` | — | +60.31 | +11.64 | +2.74 | +0.68 | +0.05 | +0.19 | +0.42 | +0.58 | +0.26 | -0.06 | -0.16 |
| `updown` | `legacy_needle` | — | +66.01 | +15.28 | +4.82 | +3.82 | +2.01 | +1.47 | +1.09 | +1.09 | +0.54 | +0.26 | +0.07 |
| `updown` | `v2` | — | +37.18 | +13.17 | +4.06 | +1.84 | +0.49 | -0.25 | -0.44 | -0.42 | -0.09 | +0.07 | +0.28 |
| `updown` | `v2_matched` | — | +33.42 | +12.68 | +5.14 | +2.45 | +0.71 | -0.10 | -0.31 | -0.30 | -0.11 | -0.04 | +0.06 |

## Effective per-order half-width (Hz)

Fitted to the NULL-SUBTRACTED excess-vs-B curve over B = 4, 8, 16, 32, 64 Hz on the 4096/1024 front end (3.91 Hz per bin): a Lorentzian of half-width gamma keeps (2/pi) arctan(B/gamma) of its power inside +-B, so the GROWTH of the excess with B identifies gamma without ever resolving the line. The half-order null is subtracted at every B first, so a floor the two-sided baseline cannot follow drops out. The number carries the four-rotor spread (k x a few rev/s) and the in-frame drift with it: the `v2` row, whose fitted gamma_rk is under 1 Hz at every k <= 12, IS the instrumental floor this is read against. `>600` means the curve was still growing at B = 64 Hz — no width is identifiable below the order spacing; `—` means the order's carrier-locked power is at or below 0.5 % of band power, the estimator's own order-to-order scatter, so no width is fitted at all. The bracketed number is the assumption-free concentration, excess(4 Hz) / excess(64 Hz).

| window | arm | k=1 | k=2 | k=3 | k=4 | k=6 | k=8 | k=12 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `free-flight` | `real` | 4.5 (0.52) | 20.6 (0.12) | 61.0 (-0.05) | — | 4.4 (0.45) | 10.6 (0.20) | — |
| `free-flight` | `legacy` | 3.6 (0.56) | 3.1 | >600 | 1.8 | 5.1 | — | — |
| `free-flight` | `legacy_needle` | 2.0 (0.73) | 0.8 | 0.3 | 2.2 | 4.2 | — | — |
| `free-flight` | `v2` | 5.6 (0.46) | 15.2 (0.11) | — | — | — | — | — |
| `free-flight` | `v2_g13` | 45.8 (0.05) | >600 (-0.00) | — | — | — | — | — |
| `free-flight` | `v2_g30` | >600 (0.00) | >600 (-0.01) | — | — | — | — | — |
| `free-flight` | `v2_g67` | 69.7 (0.01) | >600 (0.00) | — | — | — | — | — |
| `free-flight` | `v2_g130` | >600 (-0.02) | >600 (-0.01) | 2.4 | — | — | — | — |
| `free-flight` | `v2_matched` | 7.5 (0.39) | 81.2 (0.06) | — | — | — | — | — |
| `free-flight` | `v2_g13_matched` | 24.9 (0.08) | >600 (0.01) | — | — | — | — | — |
| `free-flight` | `v2_g30_matched` | 18.7 (0.13) | >600 (-0.02) | >600 (0.00) | — | — | — | — |
| `free-flight` | `v2_g67_matched` | >600 (-0.01) | >600 (-0.01) | — | — | — | — | — |
| `free-flight` | `v2_g130_matched` | >600 (-0.01) | >600 (-0.01) | >600 | — | — | — | — |
| `free-flight` | `v2_plus6db` | 3.1 (0.63) | 3.7 (0.53) | — | — | — | — | — |
| `free-flight` | `v2_plus12db` | 2.3 (0.70) | 2.3 (2.62) | — | — | — | — | — |
| `free-flight` | `v2_plus18db` | 2.2 (0.71) | 1.9 (17.09) | — | — | — | — | — |
| `free-flight` | `v2_plus21db` | 2.2 (0.71) | 1.8 (114.11) | — | — | — | — | — |
| `free-flight` | `v2_plus24db` | 2.1 (0.72) | 1.8 | — | — | — | — | — |
| `free-flight` | `v2_g13_plus21db` | 17.0 (0.13) | 16.7 (0.04) | >600 (-0.02) | — | — | — | — |
| `free-flight` | `v2_g30_plus21db` | 20.6 (0.12) | >600 (-0.02) | >600 (0.01) | — | — | — | — |
| `free-flight` | `v2_g67_plus21db` | — | >600 (0.03) | >600 (0.05) | — | — | — | — |
| `free-flight` | `v2_g130_plus21db` | 2.2 (2.39) | 42.5 (-0.03) | >600 (0.04) | — | — | — | — |
| `hovering` | `real` | 2.8 (0.70) | 7.8 (0.22) | 14.6 (-0.20) | >600 (0.05) | — | — | — |
| `hovering` | `legacy` | 4.9 (0.42) | 4.4 | >600 | 0.3 | — | — | — |
| `hovering` | `legacy_needle` | 3.2 (0.56) | 2.8 | 1.6 | 4.8 | 21.4 (-0.07) | — | — |
| `hovering` | `v2` | 6.6 (0.42) | 14.1 (0.12) | — | — | — | — | — |
| `hovering` | `v2_g13` | 45.8 (0.04) | >600 (-0.01) | — | — | — | — | — |
| `hovering` | `v2_g30` | >600 (0.04) | >600 (-0.01) | — | — | — | — | — |
| `hovering` | `v2_g67` | >600 (-0.03) | >600 (-0.00) | — | — | — | — | — |
| `hovering` | `v2_g130` | >600 (0.02) | >600 (-0.02) | — | — | — | — | — |
| `hovering` | `v2_matched` | 6.2 (0.43) | 12.1 (0.13) | — | — | — | — | — |
| `hovering` | `v2_g13_matched` | 19.1 (0.12) | 21.8 (0.01) | — | — | — | — | — |
| `hovering` | `v2_g30_matched` | 11.2 (0.36) | >600 (0.01) | >600 (0.00) | — | — | — | — |
| `hovering` | `v2_g67_matched` | >600 (-0.03) | >600 (-0.01) | — | — | — | — | — |
| `hovering` | `v2_g130_matched` | >600 (-0.02) | >600 (-0.01) | — | — | — | — | — |
| `hovering` | `v2_plus6db` | 3.2 (0.60) | 3.4 (0.60) | — | — | — | — | — |
| `hovering` | `v2_plus12db` | 2.5 (0.68) | 2.0 (2.90) | — | — | — | — | — |
| `hovering` | `v2_plus18db` | 2.3 (0.70) | 1.6 (17.22) | — | — | — | — | — |
| `hovering` | `v2_plus21db` | 2.2 (0.71) | 1.5 (73.41) | — | — | — | — | — |
| `hovering` | `v2_plus24db` | 2.2 (0.71) | 1.5 | — | — | — | — | — |
| `hovering` | `v2_g13_plus21db` | 17.0 (0.14) | 12.3 (0.05) | >600 (-0.01) | — | — | — | — |
| `hovering` | `v2_g30_plus21db` | 12.3 (0.36) | >600 (0.01) | >600 (0.00) | — | — | — | — |
| `hovering` | `v2_g67_plus21db` | — | >600 (0.06) | >600 (0.01) | — | — | — | — |
| `hovering` | `v2_g130_plus21db` | >600 | >600 (-0.06) | 36.5 (-0.05) | — | — | — | — |
| `updown` | `real` | 16.1 (0.14) | 17.0 (0.14) | — | >600 (-0.01) | — | — | — |
| `updown` | `legacy` | 7.1 (0.26) | 5.1 | 1.7 | 0.3 | — | — | — |
| `updown` | `legacy_needle` | 4.1 (0.45) | 3.3 | 1.9 | 5.9 | — | — | — |
| `updown` | `v2` | 5.6 (0.45) | 12.8 (0.11) | — | — | — | — | — |
| `updown` | `v2_g13` | 29.6 (0.05) | >600 (-0.01) | — | — | — | — | — |
| `updown` | `v2_g30` | 57.6 (0.06) | >600 (-0.01) | — | — | — | — | — |
| `updown` | `v2_g67` | >600 (0.05) | >600 (-0.00) | — | — | — | — | — |
| `updown` | `v2_g130` | >600 (0.02) | >600 (-0.00) | — | — | — | — | — |
| `updown` | `v2_matched` | 16.1 (0.21) | >600 (0.01) | — | — | — | — | — |
| `updown` | `v2_g13_matched` | 34.4 (0.04) | >600 (-0.01) | — | — | — | — | — |
| `updown` | `v2_g30_matched` | 5.2 (0.46) | >600 (-0.00) | >600 (0.01) | — | — | — | — |
| `updown` | `v2_g67_matched` | >600 (0.01) | >600 (-0.01) | — | — | — | — | — |
| `updown` | `v2_g130_matched` | >600 (0.00) | >600 (-0.01) | — | — | — | — | — |
| `updown` | `v2_plus6db` | 3.1 (0.62) | 3.6 (0.55) | — | — | — | — | — |
| `updown` | `v2_plus12db` | 2.5 (0.68) | 2.2 (2.55) | — | — | — | — | — |
| `updown` | `v2_plus18db` | 2.3 (0.70) | 1.8 (12.36) | — | — | — | — | — |
| `updown` | `v2_plus21db` | 2.3 (0.70) | 1.7 (31.26) | — | — | — | — | — |
| `updown` | `v2_plus24db` | 2.3 (0.70) | 1.7 (106.44) | — | — | — | — | — |
| `updown` | `v2_g13_plus21db` | 16.1 (0.12) | 16.1 (0.05) | >600 (0.00) | — | — | — | — |
| `updown` | `v2_g30_plus21db` | 5.0 (0.45) | >600 (-0.01) | >600 (0.01) | — | — | — | — |
| `updown` | `v2_g67_plus21db` | 0.3 (1.95) | >600 (-0.00) | >600 (0.01) | — | — | — | — |
| `updown` | `v2_g130_plus21db` | 0.3 (0.62) | >600 (0.02) | >600 (-0.01) | — | — | — | — |

## Order-tracked profile: half-power width and inter-order contrast

Demodulated on each ROTOR's own label track (its order-k line at delta f = 0) and averaged over the four rotors and eight mics. The reference is the median of the profile in the VALLEY between two orders (0.4-0.6 fbar), the only floor reference available when the order spacing (~80 Hz) is comparable to the widths under test; a profile still above half power at +-0.5 fbar is `sat` — it has no measurable width inside one order spacing — and one with no peak at delta f = 0 at all is `no peak`, which is itself the reading.

| window | arm | k=2 FWHM / contrast | k=4 FWHM / contrast | k=8 FWHM / contrast |
|---|---|---|---|---|
| `free-flight` | `real` | sat / +0.77 dB | no peak / -0.13 dB | sat / +0.15 dB |
| `free-flight` | `legacy` | 23.3 Hz / +7.54 dB | 43.5 Hz / +3.38 dB | 45.4 Hz / +0.77 dB |
| `free-flight` | `legacy_needle` | 20.0 Hz / +37.59 dB | 34.2 Hz / +14.18 dB | 7.5 Hz / +4.39 dB |
| `free-flight` | `v2` | 18.1 Hz / +2.13 dB | sat / +0.25 dB | no peak / -0.13 dB |
| `free-flight` | `v2_g13` | sat / +0.18 dB | sat / +0.19 dB | no peak / -0.27 dB |
| `free-flight` | `v2_g30` | no peak / -0.43 dB | sat / +0.08 dB | no peak / -0.23 dB |
| `free-flight` | `v2_g67` | no peak / -0.89 dB | sat / +0.16 dB | no peak / -0.24 dB |
| `free-flight` | `v2_g130` | no peak / -0.68 dB | sat / +0.12 dB | no peak / -0.15 dB |
| `free-flight` | `v2_matched` | 17.9 Hz / +1.30 dB | sat / +0.21 dB | no peak / -0.16 dB |
| `free-flight` | `v2_g13_matched` | sat / +1.16 dB | sat / +0.24 dB | no peak / -0.27 dB |
| `free-flight` | `v2_g30_matched` | sat / +0.67 dB | no peak / -0.11 dB | no peak / -0.17 dB |
| `free-flight` | `v2_g67_matched` | no peak / -0.95 dB | sat / +0.15 dB | no peak / -0.23 dB |
| `free-flight` | `v2_g130_matched` | no peak / -0.87 dB | sat / +0.15 dB | no peak / -0.21 dB |
| `free-flight` | `v2_plus6db` | 19.2 Hz / +6.30 dB | sat / +0.59 dB | 4.7 Hz / +0.14 dB |
| `free-flight` | `v2_plus12db` | 19.5 Hz / +11.73 dB | 36.5 Hz / +1.74 dB | 6.2 Hz / +0.73 dB |
| `free-flight` | `v2_plus18db` | 19.6 Hz / +17.59 dB | 34.3 Hz / +4.53 dB | 6.9 Hz / +1.60 dB |
| `free-flight` | `v2_plus21db` | 19.6 Hz / +20.55 dB | 34.0 Hz / +6.50 dB | 7.2 Hz / +1.96 dB |
| `free-flight` | `v2_plus24db` | 19.6 Hz / +23.48 dB | 33.8 Hz / +8.57 dB | 7.3 Hz / +2.18 dB |
| `free-flight` | `v2_g13_plus21db` | 28.7 Hz / +4.18 dB | sat / +0.76 dB | no peak / -0.09 dB |
| `free-flight` | `v2_g30_plus21db` | sat / +0.68 dB | no peak / -0.10 dB | no peak / -0.19 dB |
| `free-flight` | `v2_g67_plus21db` | sat / +0.01 dB | no peak / -0.04 dB | no peak / -0.28 dB |
| `free-flight` | `v2_g130_plus21db` | sat / +0.10 dB | no peak / -0.16 dB | no peak / -0.20 dB |
| `hovering` | `real` | sat / +1.99 dB | no peak / -0.00 dB | no peak / -0.13 dB |
| `hovering` | `legacy` | 26.6 Hz / +6.68 dB | 44.1 Hz / +2.26 dB | 36.8 Hz / +0.76 dB |
| `hovering` | `legacy_needle` | 21.6 Hz / +35.59 dB | 18.0 Hz / +7.25 dB | 8.4 Hz / +3.65 dB |
| `hovering` | `v2` | 20.8 Hz / +1.84 dB | sat / +0.07 dB | no peak / -0.06 dB |
| `hovering` | `v2_g13` | sat / +0.26 dB | no peak / -0.13 dB | no peak / -0.26 dB |
| `hovering` | `v2_g30` | no peak / -0.91 dB | sat / +0.02 dB | no peak / -0.16 dB |
| `hovering` | `v2_g67` | no peak / -0.86 dB | sat / +0.04 dB | no peak / -0.12 dB |
| `hovering` | `v2_g130` | no peak / -0.84 dB | no peak / -0.08 dB | no peak / -0.16 dB |
| `hovering` | `v2_matched` | 20.7 Hz / +1.99 dB | sat / +0.08 dB | no peak / -0.05 dB |
| `hovering` | `v2_g13_matched` | 29.0 Hz / +2.52 dB | no peak / -0.09 dB | no peak / -0.25 dB |
| `hovering` | `v2_g30_matched` | sat / +0.58 dB | no peak / -0.07 dB | 10.4 Hz / +0.01 dB |
| `hovering` | `v2_g67_matched` | no peak / -1.00 dB | no peak / -0.03 dB | no peak / -0.19 dB |
| `hovering` | `v2_g130_matched` | no peak / -1.02 dB | no peak / -0.06 dB | no peak / -0.21 dB |
| `hovering` | `v2_plus6db` | 20.0 Hz / +5.93 dB | sat / +0.42 dB | 6.8 Hz / +0.25 dB |
| `hovering` | `v2_plus12db` | 19.9 Hz / +11.33 dB | 43.7 Hz / +1.55 dB | 7.3 Hz / +0.92 dB |
| `hovering` | `v2_plus18db` | 19.9 Hz / +17.15 dB | 34.0 Hz / +4.24 dB | 7.4 Hz / +1.74 dB |
| `hovering` | `v2_plus21db` | 19.9 Hz / +20.08 dB | 17.8 Hz / +6.09 dB | 7.4 Hz / +2.08 dB |
| `hovering` | `v2_plus24db` | 19.9 Hz / +22.98 dB | 17.7 Hz / +8.04 dB | 7.5 Hz / +2.30 dB |
| `hovering` | `v2_g13_plus21db` | 28.4 Hz / +4.65 dB | sat / +0.43 dB | 49.6 Hz / +0.14 dB |
| `hovering` | `v2_g30_plus21db` | sat / +0.45 dB | no peak / -0.03 dB | 9.9 Hz / +0.02 dB |
| `hovering` | `v2_g67_plus21db` | no peak / -0.52 dB | no peak / -0.18 dB | 21.9 Hz / +0.16 dB |
| `hovering` | `v2_g130_plus21db` | no peak / -0.02 dB | no peak / -0.08 dB | no peak / -0.01 dB |
| `updown` | `real` | 36.1 Hz / +1.18 dB | no peak / -0.06 dB | sat / +0.02 dB |
| `updown` | `legacy` | 28.1 Hz / +5.50 dB | 45.5 Hz / +1.58 dB | 40.9 Hz / +0.80 dB |
| `updown` | `legacy_needle` | 21.9 Hz / +34.30 dB | 16.2 Hz / +6.65 dB | 9.3 Hz / +4.78 dB |
| `updown` | `v2` | 20.5 Hz / +2.37 dB | sat / +0.02 dB | 15.2 Hz / +0.06 dB |
| `updown` | `v2_g13` | sat / +0.46 dB | sat / +0.03 dB | no peak / -0.09 dB |
| `updown` | `v2_g30` | no peak / -0.41 dB | no peak / -0.09 dB | no peak / -0.12 dB |
| `updown` | `v2_g67` | no peak / -0.75 dB | no peak / -0.05 dB | no peak / -0.11 dB |
| `updown` | `v2_g130` | no peak / -0.69 dB | sat / +0.01 dB | 14.3 Hz / +0.06 dB |
| `updown` | `v2_matched` | sat / +0.21 dB | no peak / -0.05 dB | no peak / -0.05 dB |
| `updown` | `v2_g13_matched` | sat / +0.28 dB | sat / +0.02 dB | no peak / -0.10 dB |
| `updown` | `v2_g30_matched` | sat / +0.50 dB | no peak / -0.34 dB | 16.4 Hz / +0.02 dB |
| `updown` | `v2_g67_matched` | no peak / -0.80 dB | no peak / -0.07 dB | no peak / -0.09 dB |
| `updown` | `v2_g130_matched` | no peak / -0.79 dB | no peak / -0.04 dB | no peak / -0.06 dB |
| `updown` | `v2_plus6db` | 20.0 Hz / +6.51 dB | sat / +0.34 dB | 9.7 Hz / +0.39 dB |
| `updown` | `v2_plus12db` | 19.8 Hz / +11.90 dB | 10.3 Hz / +1.49 dB | 8.2 Hz / +1.11 dB |
| `updown` | `v2_plus18db` | 19.7 Hz / +17.73 dB | 11.9 Hz / +4.23 dB | 8.3 Hz / +2.21 dB |
| `updown` | `v2_plus21db` | 19.7 Hz / +20.67 dB | 15.3 Hz / +6.17 dB | 8.3 Hz / +2.63 dB |
| `updown` | `v2_plus24db` | 19.7 Hz / +23.60 dB | 16.0 Hz / +8.16 dB | 8.3 Hz / +2.91 dB |
| `updown` | `v2_g13_plus21db` | 28.2 Hz / +4.55 dB | sat / +0.68 dB | 16.5 Hz / +0.21 dB |
| `updown` | `v2_g30_plus21db` | sat / +0.49 dB | no peak / -0.33 dB | 15.2 Hz / +0.02 dB |
| `updown` | `v2_g67_plus21db` | no peak / -0.16 dB | sat / +0.15 dB | no peak / -0.11 dB |
| `updown` | `v2_g130_plus21db` | no peak / -0.11 dB | 2.2 Hz / +0.05 dB | 11.8 Hz / +0.13 dB |

## The four-rotor merge: four sub-peaks, or one hump?

Demodulated on the four-rotor MEAN carrier, where rotor r's order-k line sits at k (f_r - fbar). A resolved pair must DIP between its two carriers; the statistic is the deepest inter-carrier dip in dB, and >= 3 dB is `split`.

| window | k | carrier span Hz | `real` | `legacy` | `v2` | `v2_matched` | `v2_g67_matched` |
|---|---:|---:|---|---|---|---|---|
| `free-flight` | 2 | 15.5 | merged (-0.01 dB) | merged (-0.04 dB) | merged (-0.04 dB) | merged (-0.03 dB) | merged (0.07 dB) |
| `free-flight` | 4 | 31.0 | merged (0.27 dB) | merged (-0.05 dB) | merged (0.04 dB) | merged (0.02 dB) | merged (0.03 dB) |
| `free-flight` | 8 | 62.0 | merged (0.18 dB) | merged (0.18 dB) | merged (0.10 dB) | merged (0.10 dB) | merged (0.13 dB) |
| `hovering` | 2 | 16.1 | merged (-0.02 dB) | merged (-0.10 dB) | merged (-0.19 dB) | merged (-0.19 dB) | merged (0.12 dB) |
| `hovering` | 4 | 32.1 | merged (0.54 dB) | merged (-0.01 dB) | merged (0.02 dB) | merged (0.02 dB) | merged (0.02 dB) |
| `hovering` | 8 | 64.2 | merged (0.19 dB) | merged (0.02 dB) | merged (0.45 dB) | merged (0.47 dB) | merged (0.38 dB) |
| `updown` | 2 | 14.3 | merged (-0.04 dB) | merged (-0.01 dB) | merged (-0.02 dB) | merged (-0.00 dB) | merged (0.15 dB) |
| `updown` | 4 | 28.6 | merged (0.16 dB) | merged (-0.02 dB) | merged (0.01 dB) | merged (0.01 dB) | merged (0.00 dB) |
| `updown` | 8 | 57.2 | merged (0.25 dB) | merged (0.26 dB) | merged (0.27 dB) | merged (0.22 dB) | merged (0.18 dB) |

## The hump-fraction level match

The shift solved on the exact block combination `floor_only + 10^(s/20) comb_only` of the same seed (the two blocks add to the full render to float round-off, checked per window below), so the `_matched` arms are real renders at a shift that was not searched by re-rendering.

| window | real target (%) | floor-only (%) | comb-only fitted (%) | shift fitted | shift 13 Hz | shift 30 Hz | shift 67 Hz | shift 130 Hz | block additivity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `free-flight` | +16.66 | +4.72 | +95.97 | -1.73 | +4.04 | +42.00 (bound) | -12.00 (bound) | -12.00 (bound) | 2.24e-08 / 0.324 |
| `hovering` | +22.28 | +4.73 | +96.39 | +0.30 | +7.68 | +42.00 (bound) | -12.00 (bound) | -12.00 (bound) | 2.98e-08 / 0.412 |
| `updown` | +10.16 | +5.00 | +98.77 | -5.95 | -0.91 | +42.00 (bound) | -12.00 (bound) | -12.00 (bound) | 2.05e-08 / 0.291 |

## The three windows

| window | mean carrier rev/s | per-rotor mean | per-rotor offset | in-window drift of the mean rev/s | legacy pedestal HWHM at k=1/2/4/8 Hz | legacy coherent share at k=1/2/4/8 |
|---|---:|---|---|---:|---|---|
| `free-flight` | 80.39 | 84.39, 76.64, 81.63, 78.89 | +4.00, -3.75, +1.24, -1.50 | 3.10 | 13.3/13.5/13.9/14.8 | 0.706/0.249/0.004/0.000 |
| `hovering` | 80.90 | 85.09, 77.06, 82.11, 79.34 | +4.19, -3.84, +1.21, -1.56 | 11.53 | 13.3/13.5/13.9/14.8 | 0.707/0.250/0.004/0.000 |
| `updown` | 79.31 | 82.77, 75.63, 80.88, 77.94 | +3.47, -3.68, +1.58, -1.37 | 17.45 | 13.3/13.5/13.9/14.8 | 0.623/0.151/0.001/0.000 |

The legacy DREGON cruise export's comb is a COHERENCE SPLIT: the share w_k = exp(-(k/coherence_k_half)^2) of each order's power is a needle and the rest is narrowband noise of half-width gamma0 + gamma_slope k — the pedestal. `coherence_k_half` = 1.696, so from k=3 up the legacy comb is essentially ALL pedestal. `legacy_needle` (V7) is the same export with `coherence_k_half = 0`, which puts every order's full power through the tone bank and leaves no pedestal at all.

## Reading

* **`free-flight`** — real's effective HWHM at k=1/2/4/8 is 4.5/20.6/—/10.6 Hz against the needle render's 5.6/15.2/—/— Hz. At the matched hump fraction (-1.73 dB on the needle comb) the PIT MAE is 75.386 rev/s; widening the SAME comb to 13 Hz -> 77.940, 30 Hz -> 13.966, 67 Hz -> 80.368, 130 Hz -> 80.367 rev/s. Real 0.638, legacy 1.496, legacy-needle 2.034, v2 as fitted 62.105.
  Per-order tracked excess at B=16 Hz (% of band power): real k=1 +12.68, k=2 +1.89, sum k=3..8 +1.14; v2 as fitted +16.37, +3.17, +0.55; v2 at the match +13.13, +2.22, +0.54.
* **`hovering`** — real's effective HWHM at k=1/2/4/8 is 2.8/7.8/600.0/— Hz against the needle render's 6.6/14.1/—/— Hz. At the matched hump fraction (+0.30 dB on the needle comb) the PIT MAE is 71.511 rev/s; widening the SAME comb to 13 Hz -> 58.385, 30 Hz -> 8.461, 67 Hz -> 80.925, 130 Hz -> 80.925 rev/s. Real 0.888, legacy 1.895, legacy-needle 1.857, v2 as fitted 73.448.
  Per-order tracked excess at B=16 Hz (% of band power): real k=1 +16.23, k=2 +3.72, sum k=3..8 +0.62; v2 as fitted +16.62, +2.98, +0.54; v2 at the match +17.29, +3.16, +0.53.
* **`updown`** — real's effective HWHM at k=1/2/4/8 is 16.1/17.0/600.0/— Hz against the needle render's 5.6/12.8/—/— Hz. At the matched hump fraction (-5.95 dB on the needle comb) the PIT MAE is 79.273 rev/s; widening the SAME comb to 13 Hz -> 77.246, 30 Hz -> 7.725, 67 Hz -> 79.271, 130 Hz -> 79.271 rev/s. Real 1.695, legacy 2.499, legacy-needle 2.188, v2 as fitted 73.112.
  Per-order tracked excess at B=16 Hz (% of band power): real k=1 +6.16, k=2 +2.50, sum k=3..8 +0.74; v2 as fitted +17.14, +3.45, +0.49; v2 at the match +8.72, +1.08, +0.50.


## R4 proposal

The three options this study was asked to decide between, each with the number that decides it:

**(i) free `gamma_rk` in the DREGON flight fit under a wide flight-specific prior — NOT SUPPORTED.** At the matched carrier-locked power the needle comb scores 75.386, 71.511, 79.273 rev/s and the same comb widened scores 13 Hz: 77.940, 58.385, 77.246; 30 Hz: 13.966, 8.461, 7.725; 67 Hz: 80.368, 80.925, 79.271; 130 Hz: 80.367, 80.925, 79.271 rev/s. Widening makes every window WORSE, so freeing gamma_rk cannot buy the gate: the widths the data does support are the measured effective HWHMs at k=1 (4.5, 2.8, 16.1 Hz) and k=2 (20.6, 7.8, 17.0 Hz) against the needle render's k=1 instrumental floor (5.6, 6.6, 5.6 Hz), i.e. a log-normal prior on gamma_rk with a median of a few Hz at k=1 and a factor-3 sd would cover them — and it is worth nothing to the gate.

At 130 Hz and 30 Hz and 67 Hz the match is NOT ATTAINABLE at any level, which is the sharpest number in this study: a DREGON cruise order spacing is fbar ~ 80 Hz, so a line of half-width 67 or 130 Hz is WIDER THAN THE GAP BETWEEN ORDERS. Its power lands as much halfway between two orders as on them — its carrier-locked fraction over k=1..8 is +1.28, -1.40, +1.82 % (67 Hz) and +3.00, +0.32, +0.49 % (130 Hz) of ITS OWN band power, against +95.97, +96.39, +98.77 % for the needle comb and a real-window target of +16.66, +22.28, +10.16 %. A comb that wide is not a quiet comb, it is not a comb.

The same comparison at EQUAL comb power, which is the one without a confound (the match gives each width a DIFFERENT shift): at 21 dB up, where the needle comb is tracked at 5.118, 2.759, 6.800 rev/s, the same comb widened scores 13 Hz: 27.057, 20.562, 13.678; 30 Hz: 37.616, 34.122, 27.444; 67 Hz: 28.263, 25.802, 25.957; 130 Hz: 25.375, 26.453, 29.656 rev/s — WORSE in every window.

**(ii) a per-order PEDESTAL term on top of the narrow line (a SECOND width per line — a complexity the user must approve) — NOT SUPPORTED.** The legacy model IS that term (coherence split: needle plus a 13.5 Hz HWHM pedestal at k=2) and `legacy_needle` (V7) is the same export with the pedestal switched off: free-flight 1.496 -> 2.034, hovering 1.895 -> 1.857, updown 2.499 -> 2.188 rev/s with it muted. A second width per line doubles the comb's parameter count (2 x R x K) and the only evidence for it is the pedestal's effect on the legacy arm, measured here.

**(iii) modelling the four-rotor merge explicitly — NOT NEEDED.** The render already carries four independent per-rotor tracks on the labels' own carriers; the split table above shows the v2 render and the real clip in the SAME state at every order measured, so there is nothing to add.

**What the numbers do support: the comb's LEVEL, and the objective that sets it.** Re-levelling the FITTED (needle) comb by -1.73, +0.30, -5.95 dB — nothing else changed — moves the PIT MAE from 62.105, 73.448, 73.112 to 75.386, 71.511, 79.273 rev/s, and the dose-response's own optimum (+21 dB) reaches 5.118, 2.759, 6.800 rev/s (mean 4.893) against legacy 1.496, 1.895, 2.499. `comb_gain_db` is ALREADY free in this fit and the flight Whittle objective chose -6.918 dB for it: the bottleneck is the objective, not the parameterisation.

```diff
  # src/experiments/noise_model/fit.py -- the flight objective
- # comb level is scored by the band-pooled Whittle risk alone, which is
- # dominated by the 30-300 Hz floor cells and buries the comb.
+ # score the comb level on the CARRIER-TRACKED cells only: the Whittle
+ # risk restricted to |f - k f_r(t)| <= 16 Hz, k = 1..K, plus the floor
+ # band as now. One extra term, no new parameter: comb_gain_db and the
+ # per-order low_order_gain_db are then pulled by the cells that
+ # actually carry the comb.
```

No model code changed in this study: every variant is a mutation of the fit payload (`_mutate`) rendered by the unchanged renderer.

## Provenance

Probe `hppnet_l2_r2_s0/best`, sha256 `6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1` (the loader dies on a mismatch, so a scored arm cannot exist without it). Fit `results/noise_v2/rounds/round3/fits/dregon_room2_floor__flight_floor_lowk.json`, mode `flight_floor_lowk`, `comb_gain_db` -6.918, `low_order_gain_db` 13.86, 3.67, -0.87, -4.89, 0.65, -9.20, -0.97, -5.70. Fitted `gamma_hz` mean over rotors at k = 1: 0.00136991 Hz, 12: 0.950316 Hz, 2: 0.00368294 Hz, 24: 4.21106 Hz, 4: 0.023683 Hz, 48: 13.4603 Hz, 8: 0.685055 Hz, 88: 15.9037 Hz. Record `e51ff57602ac769320bd52fb73d7fedc398522f5`, job `nv2-r3-humps-671f38`.

Figures: `results/noise_v2/rounds/round3/dregon_humps/humps_spectra.png`, `results/noise_v2/rounds/round3/dregon_humps/humps_order_profiles.png`, `results/noise_v2/rounds/round3/dregon_humps/humps_rotor_split.png`, `results/noise_v2/rounds/round3/dregon_humps/humps_pit_vs_level_width.png`.
## Three caveats and the bottom line (hand-added by `DregonHumps`)

This section is NOT produced by `noise_v2_render_dregon.py --study humps`; a
re-run overwrites the file and drops it.

1. **The 30 Hz match is a +42 dB clamp, not a match.** The solver's bounds are
   [-12, +42] dB. At 67 and 130 Hz the target is unreachable in the sense
   described above (those combs are wider than the gap between orders, so they
   carry almost no carrier-locked power at any level) and the solver clamps
   DOWN, muting them; at 30 Hz it clamps UP, to +42 dB, and the arm that comes
   out (V4, 10.050 rev/s mean) is not a width result — it carries 40 dB more
   comb than the needle arm it is compared with. Bound-clamped arms are flagged
   `match_at_bound` in the JSON and excluded from the width verdict; the clean
   width comparison is the equal-power one (`W13`/`W30`/`W67`/`W130` against
   `L+21`), where every arm carries the same +21 dB.
2. **`600.0` in a width cell is the fit grid's ceiling, i.e. "unresolved".**
   The width table prints it as `>600`; the per-window Reading paragraph prints
   the raw number. It means the null-subtracted excess was still growing at
   B = 64 Hz, not that a 600 Hz line was measured.
3. **The effective-HWHM reading of a QUIET comb is contaminated.** The needle
   render at its fitted level reads 5.6/6.6/5.6 Hz at k=1 and the SAME needle
   comb 21 dB up reads 2.2/2.2/2.3 Hz: the remainder is floor leaking into the
   estimator, not width. Real reads 4.5/2.8/16.1 Hz at k=1 and 20.6/7.8/17.0 Hz
   at k=2 — inside the same band as a needle, an order of magnitude below the
   30/67/130 Hz the hypothesis asked about.

**Bottom line.** HPPNet does not track broad harmonic humps on DREGON: there
are none to track (real's carrier-locked power is as narrow as a needle
render's, and its per-order excess above k=2 is at the estimator's own
±0.5 %-of-band noise), and manufacturing them makes every window worse at equal
comb power (4.893 rev/s mean needle against 20.4-33.1 rev/s widened). What the
tracker responds to is comb LEVEL: the same needle comb, 21 dB up and nothing
else changed, goes from 69.555 to 4.893 rev/s mean — 2.5x the legacy bar
(1.963) instead of 35x it. R4 should spend its freedom on the objective that
sets `comb_gain_db`, not on new width parameters.
