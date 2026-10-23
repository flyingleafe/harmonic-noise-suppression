# Noise model v4 — DREGON similarity battery

git `5cd104650db6`, render seed 2001, label `motors_command`, all 8 mics; wall 220 s. Real is the reference; signed numbers are arm − real. `real_self` = real first half vs real second half of each window (the noise floor a perfect generator would roughly reach; see Notes). Every number is the mean over the five windows of the set (and over mics/bands/orders as stated).

## scoring windows (5 × 4 s)

### Spectral: old LTAS, line/floor LSD (dB), Itakura-Saito

| arm | LTAS mic0 \|Δ\| (old) | line \|Δ\| | line Δ | floor \|Δ\| | floor Δ | IS(real‖arm) | IS sym |
|---|---:|---:|---:|---:|---:|---:|---:|
| real_self | 3.61 | 4.65 | -3.93 | 5.35 | -4.65 | 1.761 | 1.097 |
| nv3r4 | 2.38 | 3.58 | -2.76 | 4.29 | -3.69 | 0.718 | 0.499 |
| v4_stage_static | 2.38 | 3.23 | -1.71 | 4.09 | -3.39 | 0.568 | 0.414 |
| v4_cwind | 1.95 | 3.11 | -1.60 | 4.06 | -3.33 | 0.544 | 0.399 |
| v4_laws_A | 2.61 | 3.73 | -1.78 | 4.65 | -4.03 | 0.863 | 0.621 |
| v4_laws_B | 1.91 | 3.65 | 0.98 | 3.73 | -2.26 | 0.369 | 0.441 |

### Order-tracked lines: |Δ| (dB) per order group, absolute peak and prominence

| arm | peak k1-2 | peak k3-10 | peak k11-30 | peak k31-100 | peak all | prom k1-2 | prom k3-10 | prom k11-30 | prom k31-100 | prom all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | 2.93 | 5.23 | 4.61 | 3.50 | 3.87 | 4.27 | 1.69 | 1.80 | 1.11 | 1.38 |
| nv3r4 | 2.53 | 3.32 | 2.62 | 2.37 | 2.51 | 3.96 | 1.40 | 1.23 | 0.74 | 0.97 |
| v4_stage_static | 2.70 | 3.21 | 2.41 | 2.42 | 2.49 | 4.50 | 1.68 | 1.66 | 0.86 | 1.18 |
| v4_cwind | 2.59 | 3.18 | 2.41 | 2.41 | 2.49 | 5.19 | 1.69 | 1.64 | 0.86 | 1.19 |
| v4_laws_A | 3.32 | 3.57 | 2.85 | 3.06 | 3.07 | 6.35 | 2.22 | 2.12 | 0.99 | 1.45 |
| v4_laws_B | 6.26 | 3.73 | 3.01 | 2.56 | 2.84 | 8.77 | 2.66 | 2.33 | 1.03 | 1.62 |

### Time structure and space

| arm | W1 (dB) | W1 centred | \|Δsd\| | Δsd | \|Δ(p90−p10)\| | mod norm \|Δ\| (dB) | mod abs \|Δ\| (dB) | mic-spread \|Δsd\| | MSC \|Δ\| | MSC Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | 4.05 | 1.08 | 1.11 | -0.57 | 2.88 | 4.64 | 10.28 | 0.91 | 0.052 | 0.049 |
| nv3r4 | 3.43 | 1.84 | 2.11 | -2.11 | 5.62 | 7.76 | 13.72 | 1.48 | 0.034 | -0.003 |
| v4_stage_static | 3.28 | 1.92 | 2.25 | -2.25 | 5.80 | 8.07 | 13.07 | 1.20 | 0.048 | -0.017 |
| v4_cwind | 3.15 | 1.62 | 1.76 | -1.68 | 4.60 | 7.17 | 11.67 | 1.22 | 0.050 | -0.021 |
| v4_laws_A | 3.26 | 1.61 | 1.77 | -1.71 | 4.57 | 7.02 | 11.93 | 1.12 | 0.054 | 0.039 |
| v4_laws_B | 3.25 | 1.66 | 1.87 | -1.82 | 4.82 | 7.20 | 8.53 | 1.00 | 0.082 | 0.074 |

### Line bins |ΔdB| per octave band (mic mean)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | – | 3.43 | 4.90 | 6.53 | 5.44 | 4.81 | 4.22 | 3.23 |
| nv3r4 | – | 2.83 | 4.28 | 4.94 | 3.62 | 3.17 | 3.25 | 2.97 |
| v4_stage_static | – | 3.31 | 3.58 | 4.06 | 3.21 | 2.75 | 2.54 | 3.19 |
| v4_cwind | – | 3.15 | 3.50 | 4.17 | 2.60 | 2.64 | 2.52 | 3.18 |
| v4_laws_A | – | 3.83 | 3.29 | 4.21 | 3.62 | 3.92 | 3.31 | 3.91 |
| v4_laws_B | – | 6.43 | 3.89 | 3.70 | 3.32 | 2.59 | 2.31 | 3.30 |

### Line bins ΔdB per octave band (mic mean, arm − real)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | – | -1.81 | -4.39 | -6.16 | -5.02 | -4.33 | -3.43 | -2.37 |
| nv3r4 | – | -0.31 | -3.39 | -4.76 | -3.01 | -2.54 | -2.97 | -2.31 |
| v4_stage_static | – | 1.91 | -2.52 | -3.51 | -1.68 | -1.95 | -1.90 | -2.28 |
| v4_cwind | – | 1.50 | -2.31 | -3.58 | -0.89 | -1.76 | -1.87 | -2.27 |
| v4_laws_A | – | 3.18 | -1.22 | -3.39 | -2.31 | -3.37 | -2.65 | -2.67 |
| v4_laws_B | – | 6.40 | 0.99 | -1.43 | 0.21 | -0.50 | 0.59 | 0.56 |

### Floor cells |ΔdB| per octave band (mic mean)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | 4.65 | 5.30 | 6.75 | 7.44 | 5.57 | 5.07 | 4.57 | 3.41 |
| nv3r4 | 3.90 | 4.57 | 5.87 | 6.04 | 4.02 | 3.39 | 3.45 | 3.10 |
| v4_stage_static | 4.58 | 4.56 | 5.41 | 5.34 | 3.58 | 3.12 | 2.81 | 3.32 |
| v4_cwind | 4.61 | 5.22 | 5.12 | 5.58 | 2.90 | 2.98 | 2.78 | 3.32 |
| v4_laws_A | 4.52 | 5.03 | 4.88 | 5.87 | 4.45 | 4.68 | 3.74 | 4.05 |
| v4_laws_B | 4.24 | 4.53 | 4.26 | 4.74 | 3.38 | 2.82 | 2.41 | 3.48 |

### Floor cells ΔdB per octave band (mic mean, arm − real)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | -4.15 | -4.83 | -6.34 | -6.87 | -4.83 | -4.27 | -3.55 | -2.36 |
| nv3r4 | -2.87 | -3.73 | -5.52 | -5.75 | -3.45 | -2.72 | -3.11 | -2.35 |
| v4_stage_static | -3.49 | -3.86 | -5.09 | -4.94 | -2.63 | -2.56 | -2.24 | -2.32 |
| v4_cwind | -3.35 | -4.83 | -4.64 | -5.16 | -1.77 | -2.35 | -2.20 | -2.31 |
| v4_laws_A | -3.26 | -4.63 | -4.39 | -5.49 | -4.03 | -4.46 | -3.24 | -2.72 |
| v4_laws_B | -2.96 | -4.08 | -3.67 | -4.11 | -1.90 | -1.76 | -0.05 | 0.49 |

### Frame band-level W1 (dB) per octave band (mic mean)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | 4.29 | 3.62 | 3.72 | 6.04 | 4.44 | 3.80 | 3.66 | 2.83 |
| nv3r4 | 3.31 | 2.65 | 3.30 | 4.96 | 3.67 | 3.16 | 3.48 | 2.92 |
| v4_stage_static | 3.83 | 2.72 | 2.95 | 4.74 | 3.52 | 2.82 | 2.93 | 2.71 |
| v4_cwind | 4.11 | 2.71 | 2.59 | 4.38 | 3.07 | 2.71 | 2.92 | 2.70 |
| v4_laws_A | 4.02 | 3.03 | 2.37 | 4.30 | 3.23 | 3.21 | 3.12 | 2.81 |
| v4_laws_B | 3.75 | 4.54 | 3.26 | 4.26 | 3.23 | 2.17 | 2.35 | 2.43 |

### Mean pairwise MSC, arm − real, per octave band

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | 0.014 | -0.000 | 0.025 | 0.031 | 0.151 | 0.090 | 0.061 | 0.021 |
| nv3r4 | -0.013 | 0.045 | 0.040 | 0.018 | -0.004 | -0.080 | -0.024 | -0.009 |
| v4_stage_static | -0.015 | 0.038 | 0.031 | 0.022 | -0.091 | -0.121 | -0.023 | 0.026 |
| v4_cwind | -0.019 | 0.044 | 0.024 | 0.011 | -0.102 | -0.125 | -0.024 | 0.026 |
| v4_laws_A | -0.019 | 0.059 | 0.041 | 0.047 | -0.006 | -0.021 | 0.079 | 0.128 |
| v4_laws_B | -0.018 | 0.085 | 0.076 | 0.086 | 0.058 | 0.033 | 0.111 | 0.163 |

Line-bin smearing on the arms' (= real) label (fraction of frames whose label moves the bin's order by > 0.05, mean over line bins), per band: 31.25-62.5 –, 62.5-125 0.04, 125-250 0.14, 250-500 0.36, 500-1000 0.60, 1000-2000 0.79, 2000-4000 0.89, 4000-8000 0.95

## tuning windows (5 × 8 s)

### Spectral: old LTAS, line/floor LSD (dB), Itakura-Saito

| arm | LTAS mic0 \|Δ\| (old) | line \|Δ\| | line Δ | floor \|Δ\| | floor Δ | IS(real‖arm) | IS sym |
|---|---:|---:|---:|---:|---:|---:|---:|
| real_self | 1.45 | 2.42 | 0.91 | 2.57 | 0.79 | 0.175 | 0.246 |
| nv3r4 | 1.72 | 2.66 | 0.77 | 2.49 | 0.72 | 0.170 | 0.196 |
| v4_stage_static | 1.81 | 2.96 | 1.68 | 2.71 | 0.97 | 0.234 | 0.306 |
| v4_cwind | 1.85 | 2.98 | 1.75 | 2.70 | 0.94 | 0.231 | 0.308 |
| v4_laws_A | 1.56 | 3.24 | 1.54 | 2.73 | 0.24 | 0.289 | 0.364 |
| v4_laws_B | 3.54 | 4.85 | 4.31 | 3.32 | 2.04 | 0.413 | 0.929 |

### Order-tracked lines: |Δ| (dB) per order group, absolute peak and prominence

| arm | peak k1-2 | peak k3-10 | peak k11-30 | peak k31-100 | peak all | prom k1-2 | prom k3-10 | prom k11-30 | prom k31-100 | prom all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | 2.08 | 2.21 | 1.84 | 1.82 | 1.86 | 1.89 | 1.62 | 1.34 | 0.75 | 0.97 |
| nv3r4 | 3.81 | 2.94 | 2.29 | 1.79 | 2.04 | 2.99 | 1.38 | 0.89 | 0.50 | 0.71 |
| v4_stage_static | 3.49 | 3.50 | 2.97 | 2.29 | 2.57 | 3.28 | 1.77 | 1.32 | 0.61 | 0.92 |
| v4_cwind | 3.48 | 3.36 | 3.09 | 2.30 | 2.58 | 3.62 | 1.65 | 1.31 | 0.61 | 0.91 |
| v4_laws_A | 4.50 | 3.35 | 2.52 | 2.22 | 2.43 | 4.12 | 2.10 | 1.83 | 0.69 | 1.13 |
| v4_laws_B | 7.44 | 5.07 | 4.48 | 3.82 | 4.15 | 5.73 | 2.43 | 2.07 | 0.72 | 1.26 |

### Time structure and space

| arm | W1 (dB) | W1 centred | \|Δsd\| | Δsd | \|Δ(p90−p10)\| | mod norm \|Δ\| (dB) | mod abs \|Δ\| (dB) | mic-spread \|Δsd\| | MSC \|Δ\| | MSC Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | 1.78 | 0.60 | 0.57 | 0.21 | 1.49 | 4.56 | 6.36 | 0.45 | 0.020 | -0.011 |
| nv3r4 | 2.66 | 1.26 | 1.51 | -1.48 | 3.95 | 7.02 | 7.38 | 1.52 | 0.072 | -0.057 |
| v4_stage_static | 2.84 | 1.27 | 1.54 | -1.53 | 4.00 | 7.34 | 7.07 | 1.18 | 0.089 | -0.073 |
| v4_cwind | 2.57 | 0.94 | 1.07 | -1.03 | 2.83 | 5.73 | 6.06 | 1.24 | 0.092 | -0.075 |
| v4_laws_A | 2.54 | 0.92 | 1.04 | -1.01 | 2.76 | 5.49 | 6.48 | 1.29 | 0.073 | -0.016 |
| v4_laws_B | 4.48 | 0.98 | 1.13 | -1.10 | 2.97 | 5.84 | 6.08 | 1.49 | 0.075 | 0.020 |

### Line bins |ΔdB| per octave band (mic mean)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | – | 2.53 | 2.57 | 2.94 | 2.41 | 2.15 | 2.38 | 1.96 |
| nv3r4 | – | 3.99 | 2.86 | 2.64 | 2.73 | 2.15 | 2.00 | 2.26 |
| v4_stage_static | – | 4.46 | 2.38 | 2.96 | 3.49 | 2.31 | 2.31 | 2.84 |
| v4_cwind | – | 4.48 | 2.49 | 2.92 | 3.40 | 2.37 | 2.33 | 2.84 |
| v4_laws_A | – | 5.37 | 3.13 | 3.20 | 3.01 | 2.54 | 2.21 | 3.23 |
| v4_laws_B | – | 7.96 | 5.11 | 4.47 | 4.45 | 3.18 | 4.05 | 4.69 |

### Line bins ΔdB per octave band (mic mean, arm − real)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | – | -0.00 | 0.58 | 1.30 | 1.00 | 0.71 | 1.35 | 1.40 |
| nv3r4 | – | 1.54 | 0.61 | 0.81 | 0.95 | 0.58 | 0.48 | 0.41 |
| v4_stage_static | – | 3.30 | 1.41 | 1.91 | 2.05 | 1.03 | 1.53 | 0.49 |
| v4_cwind | – | 3.06 | 1.64 | 1.52 | 2.79 | 1.19 | 1.56 | 0.50 |
| v4_laws_A | – | 4.63 | 2.69 | 1.66 | 1.35 | -0.39 | 0.68 | 0.15 |
| v4_laws_B | – | 7.80 | 4.93 | 3.68 | 3.95 | 2.51 | 3.92 | 3.40 |

### Floor cells |ΔdB| per octave band (mic mean)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | 2.07 | 2.63 | 3.11 | 3.13 | 2.60 | 2.35 | 2.57 | 2.08 |
| nv3r4 | 2.37 | 2.33 | 2.60 | 3.13 | 2.87 | 2.25 | 2.10 | 2.25 |
| v4_stage_static | 2.62 | 2.39 | 2.69 | 3.33 | 3.30 | 2.26 | 2.32 | 2.79 |
| v4_cwind | 2.70 | 2.33 | 2.79 | 3.26 | 3.13 | 2.30 | 2.33 | 2.79 |
| v4_laws_A | 2.68 | 2.32 | 2.87 | 3.11 | 2.67 | 2.74 | 2.26 | 3.18 |
| v4_laws_B | 2.69 | 2.39 | 3.26 | 3.85 | 3.36 | 2.64 | 3.76 | 4.60 |

### Floor cells ΔdB per octave band (mic mean, arm − real)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | 0.10 | 0.50 | 0.65 | 0.80 | 0.86 | 0.67 | 1.37 | 1.37 |
| nv3r4 | 1.30 | 0.82 | 0.67 | 1.06 | 0.73 | 0.33 | 0.41 | 0.46 |
| v4_stage_static | 0.48 | 0.61 | 1.06 | 1.89 | 1.40 | 0.48 | 1.34 | 0.53 |
| v4_cwind | 0.32 | -0.33 | 1.16 | 1.51 | 2.25 | 0.66 | 1.37 | 0.53 |
| v4_laws_A | 0.46 | -0.12 | 1.43 | 1.11 | -0.02 | -1.44 | 0.34 | 0.16 |
| v4_laws_B | 0.82 | 0.44 | 2.17 | 2.52 | 2.15 | 1.28 | 3.53 | 3.40 |

### Frame band-level W1 (dB) per octave band (mic mean)

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | 2.00 | 2.48 | 1.98 | 2.39 | 1.36 | 1.20 | 1.44 | 1.38 |
| nv3r4 | 2.93 | 3.36 | 3.05 | 3.55 | 2.67 | 1.96 | 2.08 | 1.71 |
| v4_stage_static | 2.90 | 3.90 | 2.58 | 4.30 | 3.23 | 1.65 | 2.29 | 1.88 |
| v4_cwind | 2.55 | 3.50 | 1.97 | 3.33 | 3.30 | 1.71 | 2.30 | 1.89 |
| v4_laws_A | 2.55 | 4.38 | 2.54 | 3.63 | 2.16 | 1.21 | 1.98 | 1.87 |
| v4_laws_B | 2.61 | 6.63 | 4.81 | 5.58 | 4.41 | 3.22 | 4.45 | 4.17 |

### Mean pairwise MSC, arm − real, per octave band

| arm | 31.25-62.5 | 62.5-125 | 125-250 | 250-500 | 500-1000 | 1000-2000 | 2000-4000 | 4000-8000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real_self | -0.004 | -0.008 | -0.012 | 0.010 | -0.006 | -0.031 | -0.032 | -0.008 |
| nv3r4 | -0.027 | 0.047 | 0.014 | -0.042 | -0.196 | -0.165 | -0.074 | -0.011 |
| v4_stage_static | -0.030 | 0.033 | 0.005 | -0.041 | -0.295 | -0.207 | -0.075 | 0.026 |
| v4_cwind | -0.028 | 0.041 | -0.002 | -0.048 | -0.306 | -0.210 | -0.075 | 0.026 |
| v4_laws_A | -0.028 | 0.055 | 0.015 | -0.016 | -0.204 | -0.107 | 0.022 | 0.132 |
| v4_laws_B | -0.025 | 0.081 | 0.052 | 0.019 | -0.133 | -0.058 | 0.052 | 0.168 |

Line-bin smearing on the arms' (= real) label (fraction of frames whose label moves the bin's order by > 0.05, mean over line bins), per band: 31.25-62.5 –, 62.5-125 0.01, 125-250 0.16, 250-500 0.41, 500-1000 0.66, 1000-2000 0.83, 2000-4000 0.91, 4000-8000 0.95

## Notes

- Lowest octave (31.25-62.5 Hz) holds no integer-order line at cruise (k = 1 is at 75-90 Hz), so its line column is empty (–); its floor column is the only sub-100 Hz reading.
- Line bins use each window's mean label (±0.05 order, 3.9 Hz bins). The label drifts: the smear fraction passes 0.5 above ~500 Hz and reaches 0.95 in 4-8 kHz, and because the frame-wise floor cells (`floor_cells`, ±5 Hz guard) follow the moving lines, nearly every bin above ~1 kHz is a floor cell in ≥ 25 % of frames — the line and floor sets overlap there (`line_bins`/`floor_bins` per band in the JSON), so the line/floor split is trustworthy below ~500 Hz only.
- Floor PSD: per bin the mean power over the frames where it is a floor cell; a bin enters only if it is a floor cell in ≥ 25 % of frames on both sides.
- Order levels: per 4096-point frame and rotor, the peak within ±0.08 order of k·f_r (frame-mean label) and its prominence over the median 0.3-0.5 orders away, frame-averaged in dB, per mic; orders up to 100 whose side window stays below 7.9 kHz. `accept_stats.order_profile` was not reused: it reads one constant rate per clip (single rotor, no drift).
- Modulation spectrum: the band envelope at the 62.5 Hz frame rate, Welch with ≤ 128-frame (2 s) segments zero-padded to 256, octave bins 0.5-1 … 16-31.25 Hz. `mod abs` is the contract's linear-power envelope with its mean removed (a level offset of x dB reads as 2x dB); `mod norm` divides by the mean first (modulation depth only) and is the level-free one.
- Coherence: Welch's MSC estimator (identical to `scipy.signal.coherence` with a 4096-point Hann) computed from one shared STFT at 75 % overlap; it is biased up by ≈ 1/n_frames.
- real_self compares halves (2 s on the scoring set, 4 s on the tuning set) while the arms compare full windows (4 s / 8 s): the estimation noise of every measure is larger for the halves (fewer frames: 28 vs 61 fine frames on the scoring set; MSC bias higher), and the halves carry different labels (the per-window label drift is in the JSON), so real_self is a pessimistic floor. No measure needed a window-length change beyond the modulation spectrum's segment length (2 s → one segment on a 2 s half).
- The real flight audio is not stationary within a window: the real_self LTAS error is 4.8-5.7 dB on hovering, rectangle and spinning (mic 0 of hovering's halves differs by 3-7 dB per band; rectangle's label ramps 65 → 90 rev/s in its first 0.5 s), so real_self's level-sensitive columns (LTAS, line/floor Δ, W1, IS) measure that drift as well as estimation noise; its level-free columns (W1 centred, mod norm, MSC, prominence) are less affected and are the better floors.
- LTAS (old) is `revised_eval.ltas_deviation_db` (mic 0, 8192-point Welch, 7 bands 100-7900 Hz).
