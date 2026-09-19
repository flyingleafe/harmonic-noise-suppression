# DREGON round 3: the coherence probe

Comb POWER is held at the fitted `M` in every synthetic arm — `render_phase` changes only how the line phases are drawn — so a PIT difference here is a statistics difference and nothing else. The `v2` arm is `render_phase` with every option off and reproduces `noise_model.render.render_noise` to 0.0e+00, 0.0e+00, 0.0e+00 on peaks of 0.324, 0.412, 0.291; seed 2001, 8 mics, probe job `nv2-r3-coh-70ae2c`.

| arm | tag | what it is |
|---|---|---|
| `real` | real | the real DREGON room-2 clip |
| `legacy` | legacy | legacy stage-2 baseline, identity-matched |
| `v2` | v2 | the round-3 v2 candidate, rendered by render_phase with every option off |
| `c1_split` | C1 | legacy-style coherence split (k_half 1.696, pedestal 13.068+0.211k Hz), same total power per order |
| `c2_order` | C2 | cross-ORDER phase lock: one phase per (mic, rotor) |
| `c3_mic` | C3 | cross-MIC lock: one phase per (rotor, order) |
| `c4_both` | C4 | cross-order AND cross-mic lock |
| `c5_both_shaftlock` | C5 | C4 plus gamma_rk = 0 for k <= 8 |

## PIT MAE (rev/s) and the two coherence statistics

| arm | tag | free-flight | hovering | updown | mean | inter-order coh (mean k=1..6) | inter-mic MSC k=1 | k=2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `real` | real | 0.638 | 0.888 | 1.695 | 1.074 | 0.073 | 0.020 | 0.025 |
| `legacy` | legacy | 1.496 | 1.895 | 2.499 | 1.963 | 0.090 | 0.222 | 0.039 |
| `v2` | v2 | 62.105 | 73.448 | 73.112 | 69.555 | 0.083 | 0.073 | 0.050 |
| `c1_split` | C1 | 80.372 | 77.249 | 76.436 | 78.019 | 0.075 | 0.067 | 0.032 |
| `c2_order` | C2 | 70.955 | 80.888 | 79.276 | 77.040 | 0.082 | 0.073 | 0.062 |
| `c3_mic` | C3 | 65.289 | 71.257 | 77.665 | 71.404 | 0.081 | 0.094 | 0.094 |
| `c4_both` | C4 | 70.500 | 71.817 | 70.781 | 71.033 | 0.082 | 0.094 | 0.099 |
| `c5_both_shaftlock` | C5 | 74.243 | 74.117 | 70.460 | 72.940 | 0.082 | 0.095 | 0.098 |

Inter-order phase coherence is `|<z_k z_{k+1}^* / |.|>|` over the 4 s window on the label's own carriers (time-domain demodulation, +-16 Hz), averaged over the four rotors, eight microphones and the five adjacent order pairs k=1..6 — it measures how STEADY the cross-order relative phase is. Inter-microphone MSC is the magnitude-squared coherence of the same demodulated envelope over the 28 microphone pairs. Both are 1 for a perfectly steady, spatially coherent line and fall towards 0 as the floor takes over the band.

## Per-window detail

| window | arm | PIT | inter-order coh | MSC k=1 | MSC k=2 |
|---|---|---:|---:|---:|---:|
| `free-flight` | `real` | 0.638 | 0.070 | 0.026 | 0.026 |
| `free-flight` | `legacy` | 1.496 | 0.091 | 0.281 | 0.064 |
| `free-flight` | `v2` | 62.105 | 0.083 | 0.068 | 0.052 |
| `free-flight` | `c1_split` | 80.372 | 0.075 | 0.068 | 0.031 |
| `free-flight` | `c2_order` | 70.955 | 0.081 | 0.067 | 0.061 |
| `free-flight` | `c3_mic` | 65.289 | 0.085 | 0.114 | 0.100 |
| `free-flight` | `c4_both` | 70.500 | 0.085 | 0.114 | 0.105 |
| `free-flight` | `c5_both_shaftlock` | 74.243 | 0.085 | 0.114 | 0.104 |
| `hovering` | `real` | 0.888 | 0.072 | 0.021 | 0.031 |
| `hovering` | `legacy` | 1.895 | 0.095 | 0.221 | 0.037 |
| `hovering` | `v2` | 73.448 | 0.082 | 0.074 | 0.054 |
| `hovering` | `c1_split` | 77.249 | 0.075 | 0.061 | 0.034 |
| `hovering` | `c2_order` | 80.888 | 0.083 | 0.074 | 0.060 |
| `hovering` | `c3_mic` | 71.257 | 0.081 | 0.095 | 0.091 |
| `hovering` | `c4_both` | 71.817 | 0.082 | 0.095 | 0.098 |
| `hovering` | `c5_both_shaftlock` | 74.117 | 0.082 | 0.094 | 0.097 |
| `updown` | `real` | 1.695 | 0.076 | 0.013 | 0.017 |
| `updown` | `legacy` | 2.499 | 0.085 | 0.165 | 0.015 |
| `updown` | `v2` | 73.112 | 0.083 | 0.078 | 0.043 |
| `updown` | `c1_split` | 76.436 | 0.073 | 0.072 | 0.033 |
| `updown` | `c2_order` | 79.276 | 0.082 | 0.078 | 0.065 |
| `updown` | `c3_mic` | 77.665 | 0.078 | 0.074 | 0.090 |
| `updown` | `c4_both` | 70.781 | 0.080 | 0.074 | 0.093 |
| `updown` | `c5_both_shaftlock` | 70.460 | 0.080 | 0.076 | 0.093 |

Probe `hppnet_l2_r2_s0/best`, sha256 `6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1`. Record `c8065a61ffa476a8bee7cda5d365fff4be930515`, job `nv2-r3-coh-70ae2c`.
## Reading

1. **No phase structure helps at the fitted comb level.** C1 78.019, C2 77.040,
   C3 71.404, C4 71.033, C5 72.940 rev/s mean against `v2` 69.555 — every arm
   is inside the spread of the failure, and the two best (C3, C4) are worse
   than the unmodified render on free-flight. That is the expected reading of
   the level study: the fitted comb sits ~19 dB under the floor in band power,
   so no arrangement of its phases is visible to anything. The statistics
   question cannot be asked at this level.
2. **The two statistics do not separate trackable from untrackable either.**
   The real window has the LOWEST inter-mic MSC of all eight arms (0.020 at
   k=1 against legacy 0.222 and v2 0.073) and the lowest inter-order phase
   coherence (0.073 against legacy 0.090), and it is the arm HPPNet tracks
   best (1.074 rev/s). Both statistics are computed in a +-16 Hz band that is
   floor-dominated in every arm, so what they mostly measure is comb-to-floor,
   not coherence; read as coherence they say the real comb is LESS spatially
   and cross-order coherent than the legacy render's, not more.
3. **What is left.** Combining this with `render_vs_model.md`: the render is
   faithful to M, the fit is faithful to the real periodogram to +2..+4 dB
   uniformly across cell classes, and neither width (`findings.md`) nor phase
   structure (here) moves the tracker at the fitted level. The only lever that
   has ever moved it is comb LEVEL (+21 dB: 69.555 -> 4.893 rev/s mean), which
   the real spectrum does not support. The next discriminating experiment is
   the same five arms REPEATED at +21 dB, where the comb is visible: that is
   the only regime in which a statistics difference can express itself, and it
   is one more job of the same size.
