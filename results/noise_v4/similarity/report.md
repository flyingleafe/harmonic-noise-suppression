# Noise model v4 — DREGON similarity battery

git `d1495169fee6`, render seeds 2001 (arm) and 2002 (seed floor), label `motors_command`, all 8 mics; wall 726 s. Real is the reference; signed numbers are arm − real. `<arm> seed floor` = the arm's seed-2001 render as reference against its seed-2002 render on the same trajectory: the distance a perfect generator would have from data that were one more draw from it. Every number is the mean over the five windows of the set (and over mics/bands/orders as stated).

## scoring windows (5 × 4 s)

### Spectral: old LTAS, line/floor LSD (dB), Itakura-Saito

| arm | LTAS mic0 \|Δ\| (old) | line \|Δ\| | line Δ | floor \|Δ\| | floor Δ | IS (linear bins, long-term PSD) | IS sym (linear) | IS octave-balanced | IS sym octave-balanced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 2.38 | 3.58 | -2.76 | 4.29 | -3.69 | 0.718 | 0.499 | 1.553 | 0.984 |
| nv3r4 seed floor | 0.10 | 0.89 | -0.01 | 1.07 | 0.09 | 0.036 | 0.036 | 0.036 | 0.036 |
| v4_stage_static | 2.38 | 3.23 | -1.71 | 4.09 | -3.39 | 0.568 | 0.414 | 1.178 | 0.786 |
| v4_stage_static seed floor | 0.16 | 0.89 | -0.00 | 1.05 | 0.01 | 0.038 | 0.038 | 0.035 | 0.035 |
| v4_cwind | 1.95 | 3.11 | -1.60 | 4.06 | -3.33 | 0.544 | 0.399 | 1.183 | 0.786 |
| v4_cwind seed floor | 0.25 | 0.94 | -0.12 | 1.19 | -0.15 | 0.040 | 0.040 | 0.046 | 0.044 |
| v4_laws_A | 2.61 | 3.73 | -1.78 | 4.65 | -4.03 | 0.863 | 0.621 | 1.320 | 0.896 |
| v4_laws_A seed floor | 0.26 | 1.00 | -0.10 | 1.23 | -0.16 | 0.046 | 0.046 | 0.051 | 0.050 |
| v4_laws_B | 1.91 | 3.65 | 0.98 | 3.73 | -2.26 | 0.369 | 0.441 | 0.827 | 0.717 |
| v4_laws_B seed floor | 0.24 | 0.98 | -0.05 | 1.21 | -0.13 | 0.046 | 0.047 | 0.052 | 0.051 |

### Spectro-temporal: 128 ms / 16 ms frames, third-octave log levels (24 bands, 31.5 Hz-7.9 kHz), mic mean

| arm | Fréchet (dB²) | FD mean term | FD cov term | co-mod ‖ΔR‖_F / n | off-diag r (ref) | off-diag r (arm) | band sd Δ | band sd \|Δ\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 543.4 | 309.4 | 234.0 | 0.433 | 0.444 | 0.119 | -1.98 | 2.00 |
| nv3r4 seed floor | 18.8 | 5.9 | 12.9 | 0.148 | 0.119 | 0.117 | -0.03 | 0.20 |
| v4_stage_static | 515.8 | 263.1 | 252.7 | 0.487 | 0.444 | 0.039 | -2.08 | 2.10 |
| v4_stage_static seed floor | 17.7 | 4.7 | 13.0 | 0.162 | 0.039 | 0.046 | 0.01 | 0.20 |
| v4_cwind | 465.5 | 296.8 | 168.7 | 0.408 | 0.444 | 0.153 | -1.58 | 1.71 |
| v4_cwind seed floor | 31.8 | 9.3 | 22.5 | 0.176 | 0.153 | 0.124 | -0.18 | 0.35 |
| v4_laws_A | 498.6 | 322.6 | 176.0 | 0.443 | 0.444 | 0.118 | -1.56 | 1.67 |
| v4_laws_A seed floor | 33.6 | 10.1 | 23.5 | 0.187 | 0.118 | 0.090 | -0.17 | 0.34 |
| v4_laws_B | 565.4 | 368.5 | 196.8 | 0.466 | 0.444 | 0.081 | -1.65 | 1.75 |
| v4_laws_B seed floor | 31.9 | 9.6 | 22.3 | 0.190 | 0.081 | 0.058 | -0.12 | 0.31 |

### Order-tracked lines: |Δ| (dB) per order group, absolute peak and prominence

| arm | peak k1-2 | peak k3-10 | peak k11-30 | peak k31-100 | peak all | prom k1-2 | prom k3-10 | prom k11-30 | prom k31-100 | prom all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 2.53 | 3.32 | 2.62 | 2.37 | 2.51 | 3.96 | 1.40 | 1.23 | 0.74 | 0.97 |
| nv3r4 seed floor | 0.65 | 0.62 | 0.61 | 0.55 | 0.57 | 0.93 | 0.77 | 0.75 | 0.68 | 0.71 |
| v4_stage_static | 2.70 | 3.21 | 2.41 | 2.42 | 2.49 | 4.50 | 1.68 | 1.66 | 0.86 | 1.18 |
| v4_stage_static seed floor | 0.52 | 0.74 | 0.71 | 0.57 | 0.62 | 0.71 | 0.88 | 0.82 | 0.67 | 0.72 |
| v4_cwind | 2.59 | 3.18 | 2.41 | 2.41 | 2.49 | 5.19 | 1.69 | 1.64 | 0.86 | 1.19 |
| v4_cwind seed floor | 0.59 | 0.76 | 0.70 | 0.57 | 0.62 | 1.05 | 0.89 | 0.80 | 0.67 | 0.72 |
| v4_laws_A | 3.32 | 3.57 | 2.85 | 3.06 | 3.07 | 6.35 | 2.22 | 2.12 | 0.99 | 1.45 |
| v4_laws_A seed floor | 0.55 | 0.87 | 0.79 | 0.58 | 0.65 | 0.98 | 0.98 | 0.85 | 0.66 | 0.74 |
| v4_laws_B | 6.26 | 3.73 | 3.01 | 2.56 | 2.84 | 8.77 | 2.66 | 2.33 | 1.03 | 1.62 |
| v4_laws_B seed floor | 0.48 | 0.93 | 0.80 | 0.57 | 0.65 | 0.85 | 1.03 | 0.85 | 0.66 | 0.74 |

### Time structure and space

| arm | W1 (dB) | W1 centred | \|Δsd\| | Δsd | \|Δ(p90−p10)\| | mod norm \|Δ\| (dB) | mod abs \|Δ\| (dB) | mic-spread \|Δsd\| | MSC \|Δ\| | MSC Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 3.43 | 1.84 | 2.11 | -2.11 | 5.62 | 7.76 | 13.72 | 1.48 | 0.034 | -0.003 |
| nv3r4 seed floor | 0.30 | 0.19 | 0.15 | -0.01 | 0.41 | 2.25 | 2.24 | 0.05 | 0.004 | -0.001 |
| v4_stage_static | 3.28 | 1.92 | 2.25 | -2.25 | 5.80 | 8.07 | 13.07 | 1.20 | 0.048 | -0.017 |
| v4_stage_static seed floor | 0.27 | 0.16 | 0.13 | 0.03 | 0.35 | 2.39 | 2.39 | 0.09 | 0.005 | 0.001 |
| v4_cwind | 3.15 | 1.62 | 1.76 | -1.68 | 4.60 | 7.17 | 11.67 | 1.22 | 0.050 | -0.021 |
| v4_cwind seed floor | 0.45 | 0.30 | 0.29 | -0.17 | 0.68 | 2.96 | 3.02 | 0.14 | 0.005 | 0.004 |
| v4_laws_A | 3.26 | 1.61 | 1.77 | -1.71 | 4.57 | 7.02 | 11.93 | 1.12 | 0.054 | 0.039 |
| v4_laws_A seed floor | 0.48 | 0.31 | 0.30 | -0.15 | 0.71 | 3.03 | 3.12 | 0.14 | 0.006 | 0.005 |
| v4_laws_B | 3.25 | 1.66 | 1.87 | -1.82 | 4.82 | 7.20 | 8.53 | 1.00 | 0.082 | 0.074 |
| v4_laws_B seed floor | 0.44 | 0.28 | 0.27 | -0.10 | 0.64 | 2.87 | 2.97 | 0.13 | 0.007 | 0.006 |

### Line bins |ΔdB| per octave band (mic mean)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | – | 2.83 | 4.28 | 4.94 | 3.62 | 3.17 | 3.25 | 2.97 |
| nv3r4 seed floor | – | 0.90 | 0.76 | 0.85 | 0.92 | 0.94 | 0.91 | 0.93 |
| v4_stage_static | – | 3.31 | 3.58 | 4.06 | 3.21 | 2.75 | 2.54 | 3.19 |
| v4_stage_static seed floor | – | 0.64 | 0.75 | 0.97 | 1.01 | 0.95 | 0.99 | 0.94 |
| v4_cwind | – | 3.15 | 3.50 | 4.17 | 2.60 | 2.64 | 2.52 | 3.18 |
| v4_cwind seed floor | – | 0.68 | 0.87 | 1.17 | 1.01 | 0.95 | 0.98 | 0.94 |
| v4_laws_A | – | 3.83 | 3.29 | 4.21 | 3.62 | 3.92 | 3.31 | 3.91 |
| v4_laws_A seed floor | – | 0.61 | 0.79 | 1.20 | 1.32 | 1.01 | 1.06 | 0.99 |
| v4_laws_B | – | 6.43 | 3.89 | 3.70 | 3.32 | 2.59 | 2.31 | 3.30 |
| v4_laws_B seed floor | – | 0.52 | 0.68 | 1.18 | 1.45 | 1.02 | 1.05 | 0.99 |

### Line bins ΔdB per octave band (mic mean, arm − real)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | – | -0.31 | -3.39 | -4.76 | -3.01 | -2.54 | -2.97 | -2.31 |
| nv3r4 seed floor | – | -0.15 | -0.08 | 0.16 | -0.07 | 0.03 | 0.01 | 0.00 |
| v4_stage_static | – | 1.91 | -2.52 | -3.51 | -1.68 | -1.95 | -1.90 | -2.28 |
| v4_stage_static seed floor | – | -0.12 | -0.01 | 0.21 | -0.17 | -0.03 | 0.06 | 0.06 |
| v4_cwind | – | 1.50 | -2.31 | -3.58 | -0.89 | -1.76 | -1.87 | -2.27 |
| v4_cwind seed floor | – | -0.10 | -0.24 | -0.26 | -0.27 | -0.06 | 0.05 | 0.06 |
| v4_laws_A | – | 3.18 | -1.22 | -3.39 | -2.31 | -3.37 | -2.65 | -2.67 |
| v4_laws_A seed floor | – | -0.07 | -0.19 | -0.25 | -0.34 | -0.05 | 0.08 | 0.08 |
| v4_laws_B | – | 6.40 | 0.99 | -1.43 | 0.21 | -0.50 | 0.59 | 0.56 |
| v4_laws_B seed floor | – | -0.03 | -0.11 | -0.17 | -0.28 | 0.02 | 0.10 | 0.08 |

### Floor cells |ΔdB| per octave band (mic mean)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 3.90 | 4.57 | 5.87 | 6.04 | 4.02 | 3.39 | 3.45 | 3.10 |
| nv3r4 seed floor | 0.87 | 0.89 | 1.04 | 1.09 | 1.19 | 1.20 | 1.14 | 1.13 |
| v4_stage_static | 4.58 | 4.56 | 5.41 | 5.34 | 3.58 | 3.12 | 2.81 | 3.32 |
| v4_stage_static seed floor | 0.81 | 0.86 | 0.94 | 1.08 | 1.19 | 1.19 | 1.17 | 1.14 |
| v4_cwind | 4.61 | 5.22 | 5.12 | 5.58 | 2.90 | 2.98 | 2.78 | 3.32 |
| v4_cwind seed floor | 0.94 | 1.18 | 1.16 | 1.51 | 1.23 | 1.19 | 1.17 | 1.13 |
| v4_laws_A | 4.52 | 5.03 | 4.88 | 5.87 | 4.45 | 4.68 | 3.74 | 4.05 |
| v4_laws_A seed floor | 0.93 | 1.16 | 1.13 | 1.51 | 1.42 | 1.26 | 1.24 | 1.18 |
| v4_laws_B | 4.24 | 4.53 | 4.26 | 4.74 | 3.38 | 2.82 | 2.41 | 3.48 |
| v4_laws_B seed floor | 0.90 | 1.10 | 1.07 | 1.41 | 1.50 | 1.28 | 1.25 | 1.18 |

### Floor cells ΔdB per octave band (mic mean, arm − real)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | -2.87 | -3.73 | -5.52 | -5.75 | -3.45 | -2.72 | -3.11 | -2.35 |
| nv3r4 seed floor | 0.32 | 0.26 | 0.11 | 0.05 | 0.03 | -0.03 | 0.00 | 0.01 |
| v4_stage_static | -3.49 | -3.86 | -5.09 | -4.94 | -2.63 | -2.56 | -2.24 | -2.32 |
| v4_stage_static seed floor | -0.10 | 0.08 | 0.09 | 0.06 | -0.13 | -0.04 | 0.05 | 0.04 |
| v4_cwind | -3.35 | -4.83 | -4.64 | -5.16 | -1.77 | -2.35 | -2.20 | -2.31 |
| v4_cwind seed floor | -0.12 | -0.11 | -0.44 | -0.35 | -0.21 | -0.06 | 0.05 | 0.04 |
| v4_laws_A | -3.26 | -4.63 | -4.39 | -5.49 | -4.03 | -4.46 | -3.24 | -2.72 |
| v4_laws_A seed floor | -0.11 | -0.11 | -0.42 | -0.29 | -0.39 | -0.11 | 0.07 | 0.08 |
| v4_laws_B | -2.96 | -4.08 | -3.67 | -4.11 | -1.90 | -1.76 | -0.05 | 0.49 |
| v4_laws_B seed floor | -0.10 | -0.08 | -0.37 | -0.19 | -0.41 | -0.05 | 0.08 | 0.08 |

### Frame band-level W1 (dB) per octave band (mic mean)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 3.31 | 2.65 | 3.30 | 4.96 | 3.67 | 3.16 | 3.48 | 2.92 |
| nv3r4 seed floor | 0.82 | 0.51 | 0.34 | 0.23 | 0.17 | 0.15 | 0.10 | 0.09 |
| v4_stage_static | 3.83 | 2.72 | 2.95 | 4.74 | 3.52 | 2.82 | 2.93 | 2.71 |
| v4_stage_static seed floor | 0.60 | 0.43 | 0.33 | 0.25 | 0.15 | 0.11 | 0.17 | 0.11 |
| v4_cwind | 4.11 | 2.71 | 2.59 | 4.38 | 3.07 | 2.71 | 2.92 | 2.70 |
| v4_cwind seed floor | 1.12 | 0.58 | 0.60 | 0.66 | 0.26 | 0.13 | 0.17 | 0.11 |
| v4_laws_A | 4.02 | 3.03 | 2.37 | 4.30 | 3.23 | 3.21 | 3.12 | 2.81 |
| v4_laws_A seed floor | 1.08 | 0.58 | 0.54 | 0.70 | 0.30 | 0.20 | 0.22 | 0.18 |
| v4_laws_B | 3.75 | 4.54 | 3.26 | 4.26 | 3.23 | 2.17 | 2.35 | 2.43 |
| v4_laws_B seed floor | 1.00 | 0.56 | 0.44 | 0.65 | 0.23 | 0.24 | 0.22 | 0.20 |

### Mean pairwise MSC, arm − real, per octave band

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | -0.013 | 0.045 | 0.040 | 0.018 | -0.004 | -0.080 | -0.024 | -0.009 |
| nv3r4 seed floor | -0.002 | 0.005 | -0.001 | 0.000 | -0.007 | -0.005 | 0.006 | -0.003 |
| v4_stage_static | -0.015 | 0.038 | 0.031 | 0.022 | -0.091 | -0.121 | -0.023 | 0.026 |
| v4_stage_static seed floor | 0.004 | -0.002 | -0.005 | 0.000 | -0.004 | -0.000 | 0.007 | 0.009 |
| v4_cwind | -0.019 | 0.044 | 0.024 | 0.011 | -0.102 | -0.125 | -0.024 | 0.026 |
| v4_cwind seed floor | 0.005 | 0.001 | 0.004 | 0.009 | -0.003 | 0.000 | 0.008 | 0.009 |
| v4_laws_A | -0.019 | 0.059 | 0.041 | 0.047 | -0.006 | -0.021 | 0.079 | 0.128 |
| v4_laws_A seed floor | 0.005 | -0.000 | 0.006 | 0.006 | 0.000 | 0.003 | 0.009 | 0.011 |
| v4_laws_B | -0.018 | 0.085 | 0.076 | 0.086 | 0.058 | 0.033 | 0.111 | 0.163 |
| v4_laws_B seed floor | 0.005 | -0.002 | 0.006 | 0.004 | 0.009 | 0.006 | 0.009 | 0.008 |

### IS(real‖arm) per octave band, long-term PSD (mic mean)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 1.113 | 1.595 | 3.860 | 3.177 | 0.916 | 0.635 | 0.669 | 0.463 |
| nv3r4 seed floor | 0.031 | 0.035 | 0.041 | 0.032 | 0.036 | 0.037 | 0.035 | 0.037 |
| v4_stage_static | 1.659 | 1.394 | 2.299 | 2.070 | 0.667 | 0.433 | 0.392 | 0.507 |
| v4_stage_static seed floor | 0.028 | 0.030 | 0.031 | 0.037 | 0.041 | 0.037 | 0.040 | 0.037 |
| v4_cwind | 1.658 | 1.894 | 2.064 | 2.224 | 0.349 | 0.390 | 0.382 | 0.505 |
| v4_cwind seed floor | 0.040 | 0.050 | 0.051 | 0.069 | 0.043 | 0.037 | 0.040 | 0.037 |
| v4_laws_A | 1.560 | 1.721 | 1.701 | 2.230 | 0.883 | 0.981 | 0.705 | 0.778 |
| v4_laws_A seed floor | 0.040 | 0.050 | 0.048 | 0.070 | 0.074 | 0.044 | 0.046 | 0.041 |
| v4_laws_B | 1.283 | 1.461 | 1.194 | 1.353 | 0.463 | 0.308 | 0.212 | 0.339 |
| v4_laws_B seed floor | 0.038 | 0.051 | 0.043 | 0.064 | 0.089 | 0.044 | 0.045 | 0.041 |

Line-bin smearing on the arms' (= real) label (fraction of frames whose label moves the bin's order by > 0.05, mean over line bins), per band: 31.25-62.5 –, 62.5-125 0.04, 125-250 0.14, 250-500 0.36, 500-1000 0.60, 1000-2000 0.79, 2000-4000 0.89, 4000-8000 0.95

## tuning windows (5 × 8 s)

### Spectral: old LTAS, line/floor LSD (dB), Itakura-Saito

| arm | LTAS mic0 \|Δ\| (old) | line \|Δ\| | line Δ | floor \|Δ\| | floor Δ | IS (linear bins, long-term PSD) | IS sym (linear) | IS octave-balanced | IS sym octave-balanced |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 1.72 | 2.66 | 0.77 | 2.49 | 0.72 | 0.170 | 0.196 | 0.240 | 0.283 |
| nv3r4 seed floor | 0.10 | 0.59 | -0.02 | 0.73 | -0.02 | 0.016 | 0.016 | 0.016 | 0.016 |
| v4_stage_static | 1.81 | 2.96 | 1.68 | 2.71 | 0.97 | 0.234 | 0.306 | 0.263 | 0.393 |
| v4_stage_static seed floor | 0.11 | 0.62 | -0.00 | 0.74 | -0.04 | 0.019 | 0.019 | 0.017 | 0.017 |
| v4_cwind | 1.85 | 2.98 | 1.75 | 2.70 | 0.94 | 0.231 | 0.308 | 0.267 | 0.398 |
| v4_cwind seed floor | 0.08 | 0.67 | -0.02 | 0.84 | 0.05 | 0.020 | 0.019 | 0.021 | 0.021 |
| v4_laws_A | 1.56 | 3.24 | 1.54 | 2.73 | 0.24 | 0.289 | 0.364 | 0.297 | 0.448 |
| v4_laws_A seed floor | 0.08 | 0.72 | -0.05 | 0.88 | 0.04 | 0.024 | 0.023 | 0.024 | 0.024 |
| v4_laws_B | 3.54 | 4.85 | 4.31 | 3.32 | 2.04 | 0.413 | 0.929 | 0.404 | 0.995 |
| v4_laws_B seed floor | 0.09 | 0.72 | -0.04 | 0.87 | 0.02 | 0.025 | 0.025 | 0.026 | 0.025 |

### Spectro-temporal: 128 ms / 16 ms frames, third-octave log levels (24 bands, 31.5 Hz-7.9 kHz), mic mean

| arm | Fréchet (dB²) | FD mean term | FD cov term | co-mod ‖ΔR‖_F / n | off-diag r (ref) | off-diag r (arm) | band sd Δ | band sd \|Δ\| |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 362.2 | 240.9 | 121.3 | 0.329 | 0.241 | 0.067 | -1.35 | 1.39 |
| nv3r4 seed floor | 8.7 | 2.5 | 6.2 | 0.106 | 0.067 | 0.070 | 0.00 | 0.14 |
| v4_stage_static | 402.6 | 279.4 | 123.2 | 0.349 | 0.241 | 0.019 | -1.36 | 1.38 |
| v4_stage_static seed floor | 8.9 | 2.5 | 6.4 | 0.111 | 0.019 | 0.023 | -0.01 | 0.14 |
| v4_cwind | 307.8 | 247.0 | 60.9 | 0.270 | 0.241 | 0.124 | -0.96 | 1.03 |
| v4_cwind seed floor | 12.0 | 3.1 | 8.9 | 0.120 | 0.124 | 0.130 | -0.01 | 0.20 |
| v4_laws_A | 334.6 | 272.1 | 62.5 | 0.290 | 0.241 | 0.108 | -0.92 | 0.98 |
| v4_laws_A seed floor | 13.5 | 3.5 | 10.1 | 0.132 | 0.108 | 0.111 | -0.00 | 0.20 |
| v4_laws_B | 733.8 | 660.5 | 73.3 | 0.308 | 0.241 | 0.073 | -0.99 | 1.04 |
| v4_laws_B seed floor | 13.6 | 3.6 | 10.0 | 0.133 | 0.073 | 0.076 | 0.01 | 0.20 |

### Order-tracked lines: |Δ| (dB) per order group, absolute peak and prominence

| arm | peak k1-2 | peak k3-10 | peak k11-30 | peak k31-100 | peak all | prom k1-2 | prom k3-10 | prom k11-30 | prom k31-100 | prom all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 3.81 | 2.94 | 2.29 | 1.79 | 2.04 | 2.99 | 1.38 | 0.89 | 0.50 | 0.71 |
| nv3r4 seed floor | 0.41 | 0.44 | 0.42 | 0.38 | 0.39 | 0.56 | 0.56 | 0.52 | 0.46 | 0.49 |
| v4_stage_static | 3.49 | 3.50 | 2.97 | 2.29 | 2.57 | 3.28 | 1.77 | 1.32 | 0.61 | 0.92 |
| v4_stage_static seed floor | 0.47 | 0.54 | 0.48 | 0.41 | 0.44 | 0.58 | 0.63 | 0.55 | 0.46 | 0.50 |
| v4_cwind | 3.48 | 3.36 | 3.09 | 2.30 | 2.58 | 3.62 | 1.65 | 1.31 | 0.61 | 0.91 |
| v4_cwind seed floor | 0.45 | 0.57 | 0.47 | 0.41 | 0.44 | 0.63 | 0.66 | 0.55 | 0.46 | 0.50 |
| v4_laws_A | 4.50 | 3.35 | 2.52 | 2.22 | 2.43 | 4.12 | 2.10 | 1.83 | 0.69 | 1.13 |
| v4_laws_A seed floor | 0.43 | 0.71 | 0.54 | 0.44 | 0.48 | 0.61 | 0.74 | 0.60 | 0.46 | 0.52 |
| v4_laws_B | 7.44 | 5.07 | 4.48 | 3.82 | 4.15 | 5.73 | 2.43 | 2.07 | 0.72 | 1.26 |
| v4_laws_B seed floor | 0.42 | 0.79 | 0.54 | 0.45 | 0.50 | 0.59 | 0.80 | 0.61 | 0.46 | 0.52 |

### Time structure and space

| arm | W1 (dB) | W1 centred | \|Δsd\| | Δsd | \|Δ(p90−p10)\| | mod norm \|Δ\| (dB) | mod abs \|Δ\| (dB) | mic-spread \|Δsd\| | MSC \|Δ\| | MSC Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 2.66 | 1.26 | 1.51 | -1.48 | 3.95 | 7.02 | 7.38 | 1.52 | 0.072 | -0.057 |
| nv3r4 seed floor | 0.19 | 0.13 | 0.10 | 0.02 | 0.29 | 1.34 | 1.38 | 0.04 | 0.004 | 0.001 |
| v4_stage_static | 2.84 | 1.27 | 1.54 | -1.53 | 4.00 | 7.34 | 7.07 | 1.18 | 0.089 | -0.073 |
| v4_stage_static seed floor | 0.21 | 0.12 | 0.10 | 0.02 | 0.27 | 1.34 | 1.39 | 0.06 | 0.002 | -0.001 |
| v4_cwind | 2.57 | 0.94 | 1.07 | -1.03 | 2.83 | 5.73 | 6.06 | 1.24 | 0.092 | -0.075 |
| v4_cwind seed floor | 0.28 | 0.19 | 0.15 | -0.01 | 0.38 | 1.99 | 2.08 | 0.06 | 0.002 | -0.001 |
| v4_laws_A | 2.54 | 0.92 | 1.04 | -1.01 | 2.76 | 5.49 | 6.48 | 1.29 | 0.073 | -0.016 |
| v4_laws_A seed floor | 0.29 | 0.20 | 0.15 | 0.01 | 0.42 | 2.01 | 2.09 | 0.06 | 0.004 | -0.002 |
| v4_laws_B | 4.48 | 0.98 | 1.13 | -1.10 | 2.97 | 5.84 | 6.08 | 1.49 | 0.075 | 0.020 |
| v4_laws_B seed floor | 0.28 | 0.19 | 0.15 | 0.02 | 0.41 | 1.90 | 1.96 | 0.05 | 0.005 | -0.003 |

### Line bins |ΔdB| per octave band (mic mean)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | – | 3.99 | 2.86 | 2.64 | 2.73 | 2.15 | 2.00 | 2.26 |
| nv3r4 seed floor | – | 0.47 | 0.58 | 0.57 | 0.63 | 0.63 | 0.62 | 0.63 |
| v4_stage_static | – | 4.46 | 2.38 | 2.96 | 3.49 | 2.31 | 2.31 | 2.84 |
| v4_stage_static seed floor | – | 0.53 | 0.50 | 0.62 | 0.65 | 0.68 | 0.68 | 0.66 |
| v4_cwind | – | 4.48 | 2.49 | 2.92 | 3.40 | 2.37 | 2.33 | 2.84 |
| v4_cwind seed floor | – | 0.55 | 0.66 | 0.82 | 0.66 | 0.68 | 0.67 | 0.66 |
| v4_laws_A | – | 5.37 | 3.13 | 3.20 | 3.01 | 2.54 | 2.21 | 3.23 |
| v4_laws_A seed floor | – | 0.52 | 0.62 | 0.84 | 0.85 | 0.80 | 0.71 | 0.71 |
| v4_laws_B | – | 7.96 | 5.11 | 4.47 | 4.45 | 3.18 | 4.05 | 4.69 |
| v4_laws_B seed floor | – | 0.48 | 0.58 | 0.81 | 0.93 | 0.83 | 0.71 | 0.73 |

### Line bins ΔdB per octave band (mic mean, arm − real)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | – | 1.54 | 0.61 | 0.81 | 0.95 | 0.58 | 0.48 | 0.41 |
| nv3r4 seed floor | – | -0.01 | -0.05 | 0.01 | -0.08 | -0.04 | -0.02 | 0.00 |
| v4_stage_static | – | 3.30 | 1.41 | 1.91 | 2.05 | 1.03 | 1.53 | 0.49 |
| v4_stage_static seed floor | – | 0.11 | -0.10 | 0.13 | -0.11 | 0.02 | 0.01 | -0.08 |
| v4_cwind | – | 3.06 | 1.64 | 1.52 | 2.79 | 1.19 | 1.56 | 0.50 |
| v4_cwind seed floor | – | 0.11 | -0.29 | 0.16 | -0.11 | 0.02 | 0.01 | -0.08 |
| v4_laws_A | – | 4.63 | 2.69 | 1.66 | 1.35 | -0.39 | 0.68 | 0.15 |
| v4_laws_A seed floor | – | 0.12 | -0.27 | 0.13 | -0.21 | 0.01 | 0.03 | -0.13 |
| v4_laws_B | – | 7.80 | 4.93 | 3.68 | 3.95 | 2.51 | 3.92 | 3.40 |
| v4_laws_B seed floor | – | 0.15 | -0.23 | 0.12 | -0.27 | 0.04 | 0.04 | -0.17 |

### Floor cells |ΔdB| per octave band (mic mean)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 2.37 | 2.33 | 2.60 | 3.13 | 2.87 | 2.25 | 2.10 | 2.25 |
| nv3r4 seed floor | 0.57 | 0.60 | 0.74 | 0.75 | 0.85 | 0.81 | 0.77 | 0.77 |
| v4_stage_static | 2.62 | 2.39 | 2.69 | 3.33 | 3.30 | 2.26 | 2.32 | 2.79 |
| v4_stage_static seed floor | 0.55 | 0.61 | 0.72 | 0.78 | 0.81 | 0.82 | 0.82 | 0.80 |
| v4_cwind | 2.70 | 2.33 | 2.79 | 3.26 | 3.13 | 2.30 | 2.33 | 2.79 |
| v4_cwind seed floor | 0.69 | 0.72 | 0.97 | 1.04 | 0.85 | 0.82 | 0.82 | 0.80 |
| v4_laws_A | 2.68 | 2.32 | 2.87 | 3.11 | 2.67 | 2.74 | 2.26 | 3.18 |
| v4_laws_A seed floor | 0.71 | 0.72 | 0.94 | 1.07 | 0.97 | 0.89 | 0.86 | 0.86 |
| v4_laws_B | 2.69 | 2.39 | 3.26 | 3.85 | 3.36 | 2.64 | 3.76 | 4.60 |
| v4_laws_B seed floor | 0.72 | 0.70 | 0.89 | 0.99 | 0.99 | 0.92 | 0.86 | 0.87 |

### Floor cells ΔdB per octave band (mic mean, arm − real)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 1.30 | 0.82 | 0.67 | 1.06 | 0.73 | 0.33 | 0.41 | 0.46 |
| nv3r4 seed floor | -0.07 | -0.03 | 0.02 | 0.02 | -0.04 | -0.06 | -0.02 | -0.01 |
| v4_stage_static | 0.48 | 0.61 | 1.06 | 1.89 | 1.40 | 0.48 | 1.34 | 0.53 |
| v4_stage_static seed floor | -0.09 | -0.04 | -0.04 | -0.01 | -0.05 | -0.01 | 0.03 | -0.08 |
| v4_cwind | 0.32 | -0.33 | 1.16 | 1.51 | 2.25 | 0.66 | 1.37 | 0.53 |
| v4_cwind seed floor | 0.14 | 0.06 | 0.25 | 0.07 | -0.04 | 0.00 | 0.03 | -0.09 |
| v4_laws_A | 0.46 | -0.12 | 1.43 | 1.11 | -0.02 | -1.44 | 0.34 | 0.16 |
| v4_laws_A seed floor | 0.13 | 0.04 | 0.25 | 0.08 | -0.10 | -0.02 | 0.06 | -0.14 |
| v4_laws_B | 0.82 | 0.44 | 2.17 | 2.52 | 2.15 | 1.28 | 3.53 | 3.40 |
| v4_laws_B seed floor | 0.10 | 0.02 | 0.22 | 0.08 | -0.15 | 0.01 | 0.07 | -0.18 |

### Frame band-level W1 (dB) per octave band (mic mean)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 2.93 | 3.36 | 3.05 | 3.55 | 2.67 | 1.96 | 2.08 | 1.71 |
| nv3r4 seed floor | 0.52 | 0.30 | 0.27 | 0.14 | 0.12 | 0.08 | 0.05 | 0.05 |
| v4_stage_static | 2.90 | 3.90 | 2.58 | 4.30 | 3.23 | 1.65 | 2.29 | 1.88 |
| v4_stage_static seed floor | 0.45 | 0.39 | 0.25 | 0.14 | 0.11 | 0.11 | 0.12 | 0.09 |
| v4_cwind | 2.55 | 3.50 | 1.97 | 3.33 | 3.30 | 1.71 | 2.30 | 1.89 |
| v4_cwind seed floor | 0.48 | 0.45 | 0.39 | 0.38 | 0.17 | 0.12 | 0.12 | 0.09 |
| v4_laws_A | 2.55 | 4.38 | 2.54 | 3.63 | 2.16 | 1.21 | 1.98 | 1.87 |
| v4_laws_A seed floor | 0.48 | 0.44 | 0.36 | 0.37 | 0.25 | 0.15 | 0.14 | 0.15 |
| v4_laws_B | 2.61 | 6.63 | 4.81 | 5.58 | 4.41 | 3.22 | 4.45 | 4.17 |
| v4_laws_B seed floor | 0.45 | 0.45 | 0.33 | 0.31 | 0.26 | 0.18 | 0.09 | 0.17 |

### Mean pairwise MSC, arm − real, per octave band

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | -0.027 | 0.047 | 0.014 | -0.042 | -0.196 | -0.165 | -0.074 | -0.011 |
| nv3r4 seed floor | -0.000 | 0.008 | 0.000 | -0.001 | -0.001 | -0.003 | 0.007 | -0.005 |
| v4_stage_static | -0.030 | 0.033 | 0.005 | -0.041 | -0.295 | -0.207 | -0.075 | 0.026 |
| v4_stage_static seed floor | -0.001 | 0.000 | -0.006 | -0.003 | 0.000 | -0.001 | 0.001 | 0.000 |
| v4_cwind | -0.028 | 0.041 | -0.002 | -0.048 | -0.306 | -0.210 | -0.075 | 0.026 |
| v4_cwind seed floor | -0.002 | -0.001 | 0.000 | -0.003 | 0.001 | -0.001 | 0.001 | 0.000 |
| v4_laws_A | -0.028 | 0.055 | 0.015 | -0.016 | -0.204 | -0.107 | 0.022 | 0.132 |
| v4_laws_A seed floor | -0.003 | -0.002 | 0.000 | -0.007 | -0.000 | -0.006 | 0.002 | -0.001 |
| v4_laws_B | -0.025 | 0.081 | 0.052 | 0.019 | -0.133 | -0.058 | 0.052 | 0.168 |
| v4_laws_B seed floor | -0.004 | -0.003 | -0.003 | -0.009 | -0.003 | -0.004 | 0.003 | -0.004 |

### IS(real‖arm) per octave band, long-term PSD (mic mean)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nv3r4 | 0.191 | 0.418 | 0.287 | 0.271 | 0.270 | 0.181 | 0.156 | 0.147 |
| nv3r4 seed floor | 0.014 | 0.014 | 0.019 | 0.014 | 0.017 | 0.017 | 0.017 | 0.016 |
| v4_stage_static | 0.278 | 0.304 | 0.230 | 0.307 | 0.360 | 0.202 | 0.179 | 0.247 |
| v4_stage_static seed floor | 0.014 | 0.016 | 0.016 | 0.016 | 0.018 | 0.019 | 0.020 | 0.019 |
| v4_cwind | 0.299 | 0.344 | 0.244 | 0.310 | 0.306 | 0.205 | 0.180 | 0.247 |
| v4_cwind seed floor | 0.019 | 0.018 | 0.029 | 0.026 | 0.019 | 0.019 | 0.020 | 0.019 |
| v4_laws_A | 0.289 | 0.356 | 0.269 | 0.337 | 0.284 | 0.325 | 0.189 | 0.327 |
| v4_laws_A seed floor | 0.020 | 0.018 | 0.027 | 0.027 | 0.031 | 0.026 | 0.022 | 0.023 |
| v4_laws_B | 0.270 | 0.465 | 0.417 | 0.504 | 0.458 | 0.285 | 0.380 | 0.451 |
| v4_laws_B seed floor | 0.022 | 0.018 | 0.025 | 0.025 | 0.039 | 0.029 | 0.021 | 0.025 |

Line-bin smearing on the arms' (= real) label (fraction of frames whose label moves the bin's order by > 0.05, mean over line bins), per band: 31.25-62.5 –, 62.5-125 0.01, 125-250 0.16, 250-500 0.41, 500-1000 0.66, 1000-2000 0.83, 2000-4000 0.91, 4000-8000 0.95

## Notes

- Lowest octave (31.25-62.5 Hz) holds no integer-order line at cruise (k = 1 is at 75-90 Hz), so its line column is empty (–); its floor column is the only sub-100 Hz reading.
- Line bins use each window's mean label (±0.05 order, 3.9 Hz bins). The label drifts: the smear fraction passes 0.5 above ~500 Hz and reaches 0.95 in 4-8 kHz, and because the frame-wise floor cells (`floor_cells`, ±5 Hz guard) follow the moving lines, nearly every bin above ~1 kHz is a floor cell in ≥ 25 % of frames — the line and floor sets overlap there (`line_bins`/`floor_bins` per band in the JSON), so the line/floor split is trustworthy below ~500 Hz only.
- Floor PSD: per bin the mean power over the frames where it is a floor cell; a bin enters only if it is a floor cell in ≥ 25 % of the frames (same cells on both sides: one label).
- Order levels: per 4096-point frame and rotor, the peak within ±0.08 order of k·f_r (frame-mean label) and its prominence over the median 0.3-0.5 orders away, frame-averaged in dB, per mic; orders up to 100 whose side window stays below 7.9 kHz. `accept_stats.order_profile` was not reused: it reads one constant rate per clip (single rotor, no drift).
- Modulation spectrum: the band envelope at the 62.5 Hz frame rate, Welch with ≤ 128-frame (2 s) segments zero-padded to 256, octave bins 0.5-1 … 16-31.25 Hz. `mod abs` is the contract's linear-power envelope with its mean removed (a level offset of x dB reads as 2x dB); `mod norm` divides by the mean first (modulation depth only) and is the level-free one.
- Coherence: Welch's MSC estimator (identical to `scipy.signal.coherence` with a 4096-point Hann) computed from one shared STFT at 75 % overlap; it is biased up by ≈ 1/n_frames.
- Model-vs-real comparisons render each arm on the REAL window's `motors_command` trajectory, so the arm's lines sit where the data's do. The seed floor renders the same payload on the same trajectory with a second seed, so it measures realisation noise only; it is a floor for the arm's own variability (an arm that is less variable than the data has a lower floor than the data would). The earlier half-split floor (real first half vs second half) was dropped: the halves follow different speed trajectories, so their lines sit at different frequencies, and they also differ by level drift.
- LTAS (old) is `revised_eval.ltas_deviation_db` (mic 0, 8192-point Welch, 7 bands 100-7900 Hz).
- Itakura-Saito: both forms read the LONG-TERM (whole-window) fine PSDs, so they are time-blind. The linear-bin mean weights every 3.9 Hz bin equally, so 2-8 kHz holds most of it; the octave-balanced form averages the eight per-octave means (per-octave values in report.json, `bands.is_octave`).
- Spectro-temporal (time-aware): per mic, 128 ms periodic-Hann frames at a 16 ms hop, log power summed in 24 third-octave bands (31.5 Hz-7.9 kHz). Fréchet distance between Gaussians fitted to the frame level vectors (dB², `scipy.linalg.sqrtm`, real part, ridge 1e-6 dB²), split into the mean term (long-term spectrum) and the covariance term (how band levels vary and co-vary over time). Co-modulation: the 24×24 correlation matrix of frame levels; ‖R_arm − R_ref‖_F / 24 and each side's mean off-diagonal correlation (broadband bursts raise it). In seed-floor rows 'ref' is the seed-2001 render.
