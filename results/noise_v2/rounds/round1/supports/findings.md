# Noise model v2, round 1 — supports

Every support is one observation window at 16 kHz in the frozen periodogram
convention `|rfft(x*w)|^2 / sum(w^2)` (periodic Hann, one-sided,
`revised_eval.window_periodogram`): bench = ONE frame over the whole segment,
flight = NFFT 2048 / hop 512. Numbers below are read from `index.json`, which is
written by `scripts/noise_v2_supports.py build`.

Sets present: bench-points, dregon-bench, dregon-floor, michaels-cruise. 
Total supports cached: 179.

## bench-points

135 of 135 specs built (21.5 s, git `a4744d1b54a6`).

135 published points, 10 rigs, channel counts [1, 8], durations 12.0-30.0 s (median 30.0 s). Each is taken whole: the publishing derivation already cut it to its stationary span (<= 30 s).

| support | rig | dur (s) | mics | bins | rotors | carrier (rev/s) |
| --- | --- | --- | --- | --- | --- | --- |
| `bench_point_AVQ__S1_seq1_w1` | avq_quadrotor | 30.000 | 8 | 240001 | 4 | 77.910, 78.000, 78.000, 78.000 |
| `bench_point_AVQ__S1_seq1_w3` | avq_quadrotor | 30.000 | 8 | 240001 | 4 | 77.718, 77.790, 77.790, 77.790 |
| `bench_point_AVQ__S1_seq2_w0` | avq_quadrotor | 30.000 | 8 | 240001 | 4 | 97.604, 97.971, 97.780, 97.780 |
| `bench_point_AVQ__S1_seq2_w3` | avq_quadrotor | 30.000 | 8 | 240001 | 4 | 98.333, 98.240, 98.240, 98.240 |
| `bench_point_AVQ__S2_seq1_w3` | avq_quadrotor | 30.000 | 8 | 240001 | 4 | 90.482, 93.227, 93.480, 93.480 |
| `bench_point_AVQ__S2_seq1_w5` | avq_quadrotor | 30.000 | 8 | 240001 | 4 | 89.875, 92.562, 92.520, 92.520 |
| `bench_point_DREGON-bench__motor_Motor1_50` | dregon_mikrokopter_single_motor | 30.000 | 8 | 240001 | 1 | 49.000 |
| `bench_point_DREGON-bench__motor_Motor1_60` | dregon_mikrokopter_single_motor | 30.000 | 8 | 240001 | 1 | 58.690 |
| `bench_point_DREGON-bench__motor_Motor1_70` | dregon_mikrokopter_single_motor | 30.000 | 8 | 240001 | 1 | 68.300 |
| `bench_point_DREGON-bench__motor_Motor1_80` | dregon_mikrokopter_single_motor | 12.000 | 8 | 96001 | 1 | 78.060 |
| `bench_point_DREGON-bench__motor_Motor1_90` | dregon_mikrokopter_single_motor | 13.000 | 8 | 104001 | 1 | 88.180 |
| `bench_point_DREGON-bench__motor_Motor2_50` | dregon_mikrokopter_single_motor | 12.000 | 8 | 96001 | 1 | 48.360 |
| `bench_point_DREGON-bench__motor_Motor2_60` | dregon_mikrokopter_single_motor | 12.000 | 8 | 96001 | 1 | 58.070 |
| `bench_point_DREGON-bench__motor_Motor2_70` | dregon_mikrokopter_single_motor | 12.000 | 8 | 96001 | 1 | 67.520 |
| `bench_point_DREGON-bench__motor_Motor2_80` | dregon_mikrokopter_single_motor | 12.000 | 8 | 96001 | 1 | 77.180 |
| `bench_point_DREGON-bench__motor_Motor2_90` | dregon_mikrokopter_single_motor | 12.000 | 8 | 96001 | 1 | 86.700 |
| `bench_point_DREGON-bench__motor_Motor3_50` | dregon_mikrokopter_single_motor | 12.000 | 8 | 96001 | 1 | 49.065 |
| `bench_point_DREGON-bench__motor_Motor3_60` | dregon_mikrokopter_single_motor | 13.000 | 8 | 104001 | 1 | 58.900 |
| `bench_point_DREGON-bench__motor_Motor3_70` | dregon_mikrokopter_single_motor | 12.000 | 8 | 96001 | 1 | 68.560 |
| `bench_point_DREGON-bench__motor_Motor3_80` | dregon_mikrokopter_single_motor | 13.000 | 8 | 104001 | 1 | 78.820 |
| `bench_point_DREGON-bench__motor_Motor3_90` | dregon_mikrokopter_single_motor | 12.000 | 8 | 96001 | 1 | 88.240 |
| `bench_point_DREGON-bench__motor_Motor4_50` | dregon_mikrokopter_single_motor | 13.000 | 8 | 104001 | 1 | 49.740 |
| `bench_point_DREGON-bench__motor_Motor4_60` | dregon_mikrokopter_single_motor | 13.000 | 8 | 104001 | 1 | 59.660 |
| `bench_point_DREGON-bench__motor_Motor4_70` | dregon_mikrokopter_single_motor | 13.000 | 8 | 104001 | 1 | 69.310 |
| `bench_point_DREGON-bench__motor_Motor4_80` | dregon_mikrokopter_single_motor | 12.000 | 8 | 96001 | 1 | 79.380 |
| `bench_point_DREGON-bench__motor_Motor4_90` | dregon_mikrokopter_single_motor | 12.000 | 8 | 96001 | 1 | 89.140 |
| `bench_point_DREGON-bench__motor_allMotors_70` | dregon_mikrokopter_quad | 30.000 | 8 | 240001 | 4 | 64.647, 67.659, 68.740, 69.566 |
| `bench_point_DroneAudioSet__drone2_low_50cm_M_down_File3` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 102.453, 102.778, 102.778, 102.778 |
| `bench_point_DroneAudioSet__drone2_low_50cm_M_center_File3` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 102.500, 102.780, 102.780, 102.780 |
| `bench_point_DroneAudioSet__drone2_low_50cm_M_down_silence` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 104.542, 104.380, 104.380, 104.380 |
| `bench_point_DroneAudioSet__drone2_low_50cm_M_center_silence` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 104.688, 104.600, 104.600, 104.600 |
| `bench_point_DroneAudioSet__drone2_low_50cm_M_down_File2` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 104.000, 104.060, 104.060, 104.060 |
| `bench_point_DroneAudioSet__drone2_low_50cm_M_up_File2` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 104.375, 104.802, 104.680, 104.680 |
| `bench_point_DroneAudioSet__drone2_low_50cm_M_center_File2` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 105.261, 105.100, 105.100, 105.100 |
| `bench_point_DroneAudioSet__drone2_low_50cm_M_up_silence` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 104.537, 104.620, 104.620, 104.620 |
| `bench_point_DroneAudioSet__drone2_low_50cm_M_up_File6` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 105.698, 105.520, 105.520, 105.520 |
| `bench_point_DroneAudioSet__drone2_low_50cm_M_center_File6` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 105.835, 106.167, 106.060, 106.060 |
| `bench_point_DroneAudioSet__drone2_low_50cm_M_down_File6` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 105.653, 105.600, 105.600, 105.600 |
| `bench_point_DroneAudioSet__drone2_low_50cm_M_up_File3` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 102.328, 102.320, 102.320, 102.320 |
| `bench_point_DroneAudioSet__drone2_high_50cm_M_down_File3` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 100.925, 101.597, 101.360, 101.360 |
| `bench_point_DroneAudioSet__drone2_high_50cm_M_center_File3` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 24.561, 25.431, 25.819, 26.336 |
| `bench_point_DroneAudioSet__drone2_high_50cm_M_down_silence` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 94.453, 100.151, 100.380, 100.380 |
| `bench_point_DroneAudioSet__drone2_high_50cm_M_up_File1` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 135.975, 136.299, 136.370, 136.370 |
| `bench_point_DroneAudioSet__drone2_high_50cm_M_center_File1` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 136.250, 136.540, 136.540, 136.540 |
| `bench_point_DroneAudioSet__drone2_high_50cm_M_down_File2` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 103.795, 103.990, 103.990, 103.990 |
| `bench_point_DroneAudioSet__drone2_high_50cm_M_up_File2` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 103.788, 103.780, 103.780, 103.780 |
| `bench_point_DroneAudioSet__drone2_high_50cm_M_center_File2` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 103.927, 103.740, 103.740, 103.740 |
| `bench_point_DroneAudioSet__drone2_high_50cm_M_up_silence` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 100.179, 100.450, 100.450, 100.450 |
| `bench_point_DroneAudioSet__drone2_high_50cm_M_up_File6` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 47.631, 50.600, 52.755, 50.680 |
| `bench_point_DroneAudioSet__drone2_high_50cm_M_center_File6` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 50.700, 50.780, 50.780, 50.780 |
| `bench_point_DroneAudioSet__drone2_high_50cm_M_down_File1` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 135.969, 136.391, 135.660, 135.660 |
| `bench_point_DroneAudioSet__drone2_high_50cm_M_down_File6` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 101.175, 101.320, 101.320, 101.320 |
| `bench_point_DroneAudioSet__drone2_high_50cm_M_up_File3` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 24.273, 25.400, 26.290, 25.331 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_down_File3` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 106.484, 106.380, 106.380, 106.380 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_center_File3` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 106.662, 106.820, 106.820, 106.820 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_down_silence` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 108.413, 108.620, 108.620, 108.620 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_center_silence` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 108.534, 108.760, 108.760, 108.760 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_up_File4` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 25.763, 26.635, 27.312, 28.683 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_down_File2` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 106.546, 106.800, 106.800, 106.800 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_up_File2` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 106.656, 106.820, 106.820, 106.820 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_down_File5` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 109.345, 109.220, 109.220, 109.220 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_center_File2` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 106.771, 106.940, 106.940, 106.940 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_up_silence` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 108.409, 108.520, 108.520, 108.520 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_up_File6` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 108.821, 109.120, 109.120, 109.120 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_center_File6` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 108.982, 109.060, 109.060, 109.060 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_down_File6` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 108.773, 108.680, 108.680, 108.680 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_up_File5` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 27.360, 27.280, 27.280, 27.280 |
| `bench_point_DroneAudioSet__drone2_low_25cm_M_up_File3` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 106.500, 106.400, 106.400, 106.400 |
| `bench_point_DroneAudioSet__drone2_high_25cm_M_down_File3` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 129.282, 129.560, 129.560, 129.560 |
| `bench_point_DroneAudioSet__drone2_high_25cm_M_center_File3` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 129.466, 129.740, 129.740, 129.740 |
| `bench_point_DroneAudioSet__drone2_high_25cm_M_down_silence` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 125.975, 126.300, 126.300, 126.300 |
| `bench_point_DroneAudioSet__drone2_high_25cm_M_up_File1` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 134.204, 134.518, 134.320, 134.320 |
| `bench_point_DroneAudioSet__drone2_high_25cm_M_center_silence` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 126.109, 126.410, 126.410, 126.410 |
| `bench_point_DroneAudioSet__drone2_high_25cm_M_center_File1` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 134.316, 134.660, 134.540, 134.540 |
| `bench_point_DroneAudioSet__drone2_high_25cm_M_down_File2` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 130.875, 130.650, 130.650, 130.650 |
| `bench_point_DroneAudioSet__drone2_high_25cm_M_down_File5` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 129.448, 129.762, 129.840, 129.840 |
| `bench_point_DroneAudioSet__drone2_high_25cm_M_up_silence` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 125.979, 125.760, 125.760, 125.760 |
| `bench_point_DroneAudioSet__drone2_high_25cm_M_center_File5` | daset_drone2 | 30.000 | 1 | 240001 | 4 | 129.326, 129.750, 129.500, 129.500 |
| `bench_point_DroneAudioSet__drone2_high_25cm_M_up_File5` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 129.188, 129.300, 129.300, 129.300 |
| `bench_point_DroneAudioSet__drone2_high_25cm_M_up_File3` | daset_drone2 | 30.000 | 8 | 240001 | 4 | 129.250, 129.040, 129.040, 129.040 |
| `bench_point_DroneAudioSet__drone1_low_50cm_M_center_File3` | daset_drone1 | 30.000 | 1 | 240001 | 4 | 92.397, 92.788, 93.188, 92.720 |
| `bench_point_DroneAudioSet__drone1_low_50cm_M_center_silence` | daset_drone1 | 30.000 | 1 | 240001 | 4 | 84.975, 85.306, 87.260, 87.812 |
| `bench_point_DroneAudioSet__drone1_low_50cm_M_center_File1` | daset_drone1 | 30.000 | 1 | 240001 | 4 | 60.538, 62.099, 62.100, 62.100 |
| `bench_point_DroneAudioSet__drone1_low_50cm_M_up_silence` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 85.000, 87.676, 84.920, 84.920 |
| `bench_point_DroneAudioSet__drone1_low_50cm_M_up_File6` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 84.784, 88.558, 89.111, 84.740 |
| `bench_point_DroneAudioSet__drone1_low_50cm_M_center_File6` | daset_drone1 | 30.000 | 1 | 240001 | 4 | 84.764, 88.775, 84.910, 84.910 |
| `bench_point_DroneAudioSet__drone1_low_50cm_M_down_File6` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 84.777, 89.355, 89.355, 89.355 |
| `bench_point_DroneAudioSet__drone1_high_50cm_M_down_File3` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 118.463, 118.847, 118.540, 118.540 |
| `bench_point_DroneAudioSet__drone1_high_50cm_M_down_silence` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 118.654, 118.540, 118.540, 118.540 |
| `bench_point_DroneAudioSet__drone1_high_50cm_M_center_File1` | daset_drone1 | 30.000 | 1 | 240001 | 4 | 123.534, 124.050, 123.760, 123.760 |
| `bench_point_DroneAudioSet__drone1_high_50cm_M_down_File2` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 119.319, 119.708, 119.740, 119.740 |
| `bench_point_DroneAudioSet__drone1_high_50cm_M_up_File2` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 119.031, 119.458, 119.833, 120.005 |
| `bench_point_DroneAudioSet__drone1_high_50cm_M_down_File5` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 120.329, 121.271, 121.300, 121.300 |
| `bench_point_DroneAudioSet__drone1_high_50cm_M_center_File2` | daset_drone1 | 30.000 | 1 | 240001 | 4 | 119.641, 120.062, 120.020, 120.020 |
| `bench_point_DroneAudioSet__drone1_high_50cm_M_up_silence` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 118.557, 118.896, 118.580, 118.580 |
| `bench_point_DroneAudioSet__drone1_high_50cm_M_up_File6` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 118.750, 119.141, 119.180, 119.180 |
| `bench_point_DroneAudioSet__drone1_high_50cm_M_center_File6` | daset_drone1 | 30.000 | 1 | 240001 | 4 | 119.028, 119.557, 119.240, 119.240 |
| `bench_point_DroneAudioSet__drone1_high_50cm_M_center_File5` | daset_drone1 | 30.000 | 1 | 240001 | 4 | 120.797, 121.231, 121.689, 121.690 |
| `bench_point_DroneAudioSet__drone1_high_50cm_M_down_File6` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 118.229, 118.847, 119.180, 119.180 |
| `bench_point_DroneAudioSet__drone1_high_50cm_M_up_File5` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 120.875, 121.274, 121.200, 121.200 |
| `bench_point_DroneAudioSet__drone1_low_25cm_M_center_File3` | daset_drone1 | 30.000 | 1 | 240001 | 4 | 78.397, 82.708, 83.220, 84.028 |
| `bench_point_DroneAudioSet__drone1_low_25cm_M_down_silence` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 88.504, 88.993, 89.180, 89.180 |
| `bench_point_DroneAudioSet__drone1_low_25cm_M_down_File2` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 82.763, 85.375, 85.440, 85.440 |
| `bench_point_DroneAudioSet__drone1_low_25cm_M_up_File2` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 82.700, 85.425, 82.620, 82.620 |
| `bench_point_DroneAudioSet__drone1_low_25cm_M_down_File5` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 87.812, 87.920, 87.920, 87.920 |
| `bench_point_DroneAudioSet__drone1_low_25cm_M_center_File2` | daset_drone1 | 30.000 | 1 | 240001 | 4 | 79.147, 82.845, 83.263, 85.574 |
| `bench_point_DroneAudioSet__drone1_low_25cm_M_center_File5` | daset_drone1 | 30.000 | 1 | 240001 | 4 | 84.938, 87.869, 88.170, 88.170 |
| `bench_point_DroneAudioSet__drone1_low_25cm_M_down_File6` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 88.150, 88.500, 88.360, 88.360 |
| `bench_point_DroneAudioSet__drone1_low_25cm_M_up_File5` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 87.730, 87.920, 87.920, 87.920 |
| `bench_point_DroneAudioSet__drone1_high_25cm_M_down_File3` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 116.773, 117.454, 116.740, 116.740 |
| `bench_point_DroneAudioSet__drone1_high_25cm_M_center_File3` | daset_drone1 | 30.000 | 1 | 240001 | 4 | 116.875, 117.410, 117.220, 117.220 |
| `bench_point_DroneAudioSet__drone1_high_25cm_M_down_silence` | daset_drone1 | 17.000 | 8 | 136001 | 4 | 118.396, 118.815, 118.680, 118.680 |
| `bench_point_DroneAudioSet__drone1_high_25cm_M_up_File4` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 112.183, 112.538, 112.938, 112.620 |
| `bench_point_DroneAudioSet__drone1_high_25cm_M_down_File2` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 118.281, 118.704, 118.440, 118.440 |
| `bench_point_DroneAudioSet__drone1_high_25cm_M_up_File2` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 118.550, 118.927, 118.620, 118.620 |
| `bench_point_DroneAudioSet__drone1_high_25cm_M_center_File2` | daset_drone1 | 30.000 | 1 | 240001 | 4 | 118.688, 119.188, 118.820, 118.820 |
| `bench_point_DroneAudioSet__drone1_high_25cm_M_up_File6` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 119.263, 119.340, 119.340, 119.340 |
| `bench_point_DroneAudioSet__drone1_high_25cm_M_center_File4` | daset_drone1 | 30.000 | 1 | 240001 | 4 | 112.548, 112.977, 112.850, 112.850 |
| `bench_point_DroneAudioSet__drone1_high_25cm_M_up_File3` | daset_drone1 | 30.000 | 8 | 240001 | 4 | 116.688, 117.188, 116.780, 116.780 |
| `bench_point_SPCUP19-ChuMS-bench__3prop_repeat2` | spcup_ChuMS_3prop_bench | 30.000 | 8 | 240001 | 3 | 77.080, 77.521, 78.072 |
| `bench_point_SPCUP19-ChuMS-bench__2prop_repeat3` | spcup_ChuMS_2prop_bench | 30.000 | 8 | 240001 | 2 | 82.625, 83.029 |
| `bench_point_SPCUP19-egonoise__Idea_ssu__stationary_1` | spcup_Idea_ssu | 30.000 | 1 | 240001 | 4 | 92.208, 94.412, 97.328, 92.181 |
| `bench_point_SPCUP19-egonoise__Diagonal_Unloading__recordings__static__static_2m` | spcup_Diagonal_Unloading | 30.000 | 8 | 240001 | 4 | 99.704, 100.010, 100.010, 100.010 |
| `bench_point_SPCUP19-egonoise__AGH__ego-noise__single_rotors__0` | spcup_AGH | 17.000 | 1 | 136001 | 1 | 79.820 |
| `bench_point_SPCUP19-egonoise__AGH__ego-noise__single_rotors__1` | spcup_AGH | 16.000 | 1 | 128001 | 1 | 132.300 |
| `bench_point_SPCUP19-egonoise__AGH__ego-noise__single_rotors__2` | spcup_AGH | 16.000 | 1 | 128001 | 1 | 113.540 |
| `bench_point_SPCUP19-egonoise__AGH__ego-noise__single_rotors__3` | spcup_AGH | 16.000 | 1 | 128001 | 1 | 97.480 |
| `bench_point_SPCUP19-egonoise__AGH__ego-noise__single_rotors__5` | spcup_AGH | 16.000 | 1 | 128001 | 1 | 98.080 |
| `bench_point_SPCUP19-egonoise__AGH__ego-noise__single_rotors__6` | spcup_AGH | 17.000 | 1 | 136001 | 1 | 96.780 |
| `bench_point_SPCUP19-egonoise__AGH__ego-noise__single_rotors__7` | spcup_AGH | 16.000 | 1 | 128001 | 1 | 96.780 |
| `bench_point_SPCUP19-egonoise__AGH__static_clean__0` | spcup_AGH | 20.000 | 8 | 160001 | 4 | 47.657, 51.631, 52.268, 53.159 |
| `bench_point_SPCUP19-egonoise__AGH__static_clean__1` | spcup_AGH | 20.000 | 8 | 160001 | 4 | 142.520, 142.520, 142.520, 142.520 |
| `bench_point_SPCUP19-egonoise__AGH__static_clean__4` | spcup_AGH | 24.000 | 8 | 192001 | 4 | 51.358, 53.303, 50.435, 50.435 |
| `bench_point_SPCUP19-egonoise__AGH__static_clean__5` | spcup_AGH | 23.000 | 8 | 184001 | 4 | 48.263, 52.759, 53.282, 50.540 |
| `bench_point_SPCUP19-egonoise__AGH__static_corrupted__1` | spcup_AGH | 12.000 | 8 | 96001 | 4 | 108.675, 111.195, 111.300, 111.300 |

## dregon-bench

21 of 21 specs built (103.8 s, git `a4744d1b54a6`).

The +-1 Hz rule (order 60-80, residual averaged over 2 s, minimum 4 s): **18 of 21 recordings pass**.

Passing segments run 5.62-41.00 s (median 7.66 s, total 231.7 s of stationary bench material).

Failing recordings (kept with the most stationary 4 s window, flagged `FAIL`): `bench_dregon_Motor1_70` longest 7.29 s, worst residual 0.35 Hz; `bench_dregon_Motor2_80` longest 3.63 s, worst residual 0.11 Hz; `bench_dregon_Motor4_90` longest 10.50 s, worst residual 0.45 Hz

Longest +-1 Hz spans over all 21 recordings: min 3.63 s, median 7.29 s, max 41.00 s.

Carrier refinement (survey speed -> demodulated line): |shift| up to 0.0634 rev/s, median 0.0095 rev/s. At order 70 a 0.01 rev/s error already displaces the line by 0.7 Hz, most of the tolerance, which is why the carrier is refined before the rule is applied.

| support | pass | segment (s) | dur (s) | longest +-1 Hz (s) | order | survey (rev/s) | carrier (rev/s) | shift | residual std (Hz) | mics | bins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bench_dregon_Motor1_50` | PASS | 2.00-43.00 | 41.000 | 41.00 | [63] | 49.0000 | 49.0073 | +0.0073 | 0.201 | 8 | 328001 |
| `bench_dregon_Motor1_60` | PASS | 2.05-8.81 | 6.763 | 6.76 | [74] | 58.6900 | 58.7534 | +0.0634 | 0.174 | 8 | 54103 |
| `bench_dregon_Motor1_70` | FAIL | 35.71-43.00 | 7.293 | 7.29 | [62] | 68.3000 | 68.3174 | +0.0174 | 0.153 | 8 | 58342 |
| `bench_dregon_Motor1_80` | PASS | 2.00-23.00 | 21.000 | 21.00 | [67] | 78.0600 | 78.0596 | -0.0004 | 0.201 | 8 | 168001 |
| `bench_dregon_Motor1_90` | PASS | 2.00-9.66 | 7.664 | 7.66 | [61] | 88.1800 | 88.1844 | +0.0044 | 0.318 | 8 | 61314 |
| `bench_dregon_Motor2_50` | PASS | 5.59-23.00 | 17.407 | 17.41 | [63] | 48.3600 | 48.4086 | +0.0486 | 0.232 | 8 | 139260 |
| `bench_dregon_Motor2_60` | PASS | 15.90-23.00 | 7.098 | 7.10 | [63] | 58.0700 | 58.0979 | +0.0279 | 0.070 | 8 | 56787 |
| `bench_dregon_Motor2_70` | PASS | 15.86-23.00 | 7.136 | 7.14 | [70] | 67.5200 | 67.5225 | +0.0025 | 0.106 | 8 | 57087 |
| `bench_dregon_Motor2_80` | FAIL | 18.30-22.30 | 4.000 | 3.63 | [63] | 77.1800 | 77.1895 | +0.0095 | 0.051 | 8 | 32001 |
| `bench_dregon_Motor2_90` | PASS | 17.38-23.00 | 5.620 | 5.62 | [72] | 86.7000 | 86.7182 | +0.0182 | 0.226 | 8 | 44964 |
| `bench_dregon_Motor3_50` | PASS | 13.91-23.00 | 9.087 | 9.09 | [63] | 49.0650 | 49.1086 | +0.0436 | 0.273 | 8 | 72698 |
| `bench_dregon_Motor3_60` | PASS | 16.31-23.00 | 6.687 | 6.69 | [70] | 58.9000 | 58.9305 | +0.0305 | 0.195 | 8 | 53495 |
| `bench_dregon_Motor3_70` | PASS | 17.03-23.00 | 5.971 | 5.97 | [63] | 68.5600 | 68.5746 | +0.0146 | 0.280 | 8 | 47771 |
| `bench_dregon_Motor3_80` | PASS | 2.00-23.00 | 21.000 | 21.00 | [65] | 78.8200 | 78.8134 | -0.0066 | 0.271 | 8 | 168001 |
| `bench_dregon_Motor3_90` | PASS | 16.71-23.00 | 6.293 | 6.29 | [70] | 88.2400 | 88.2489 | +0.0089 | 0.165 | 8 | 50345 |
| `bench_dregon_Motor4_50` | PASS | 2.00-23.00 | 21.000 | 21.00 | [63] | 49.7400 | 49.7686 | +0.0286 | 0.328 | 8 | 168001 |
| `bench_dregon_Motor4_60` | PASS | 11.72-23.00 | 11.279 | 11.28 | [62] | 59.6600 | 59.6802 | +0.0202 | 0.166 | 8 | 90236 |
| `bench_dregon_Motor4_70` | PASS | 15.89-23.00 | 7.114 | 7.11 | [68] | 69.3100 | 69.3183 | +0.0083 | 0.187 | 8 | 56909 |
| `bench_dregon_Motor4_80` | PASS | 17.29-23.00 | 5.715 | 5.71 | [60] | 79.3800 | 79.3760 | -0.0040 | 0.125 | 8 | 45721 |
| `bench_dregon_Motor4_90` | FAIL | 4.89-15.39 | 10.499 | 10.50 | [69] | 89.1400 | 89.1223 | -0.0177 | 0.154 | 8 | 83995 |
| `bench_dregon_allMotors_70` | PASS | 14.43-38.27 | 23.847 | 23.85 | [67, 64, 63, 63] | 64.6471, 67.6588, 68.7396, 69.5655 | 64.6396, 67.6618, 68.7379, 69.5718 | -0.0075, +0.0030, -0.0017, +0.0063 | 0.237, 0.234, 0.247, 0.280 | 8 | 190775 |

## dregon-floor

10 of 10 specs built (8.9 s, git `a4744d1b54a6`).


| support | recording | start (s) | dur (s) | label | mics | frames | rotors | carrier mean (rev/s) | carrier span (rev/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flight_dregon_free-flight_nosource_room2@1512727397.205+4_motors_command` | free-flight_nosource_room2 | 1512727397.205 | 4.000 | `motors_command` | 8 | 122 | 4 | 84.42, 76.61, 81.64, 78.87 | 75.08-85.83 |
| `flight_dregon_free-flight_nosource_room2@1512727403.205+8_motors_command` | free-flight_nosource_room2 | 1512727403.205 | 8.000 | `motors_command` | 8 | 247 | 4 | 84.18, 77.31, 82.54, 80.34 | 74.74-87.03 |
| `flight_dregon_hovering_nosource_room2@1511903905.394+4_motors_command` | hovering_nosource_room2 | 1511903905.394 | 4.000 | `motors_command` | 8 | 122 | 4 | 85.08, 76.98, 82.09, 79.27 | 74.96-86.98 |
| `flight_dregon_hovering_nosource_room2@1511903911.394+8_motors_command` | hovering_nosource_room2 | 1511903911.394 | 8.000 | `motors_command` | 8 | 247 | 4 | 84.85, 76.14, 81.91, 79.10 | 74.36-86.43 |
| `flight_dregon_rectangle_nosource_room2@1511905725.953+4_motors_command` | rectangle_nosource_room2 | 1511905725.953 | 4.000 | `motors_command` | 8 | 122 | 4 | 84.28, 76.71, 82.02, 79.77 | 69.05-90.00 |
| `flight_dregon_rectangle_nosource_room2@1511905731.953+8_motors_command` | rectangle_nosource_room2 | 1511905731.953 | 8.000 | `motors_command` | 8 | 247 | 4 | 85.77, 76.36, 82.73, 79.74 | 74.56-88.30 |
| `flight_dregon_spinning_nosource_room2@1511905200.978+4_motors_command` | spinning_nosource_room2 | 1511905200.978 | 4.000 | `motors_command` | 8 | 122 | 4 | 84.78, 76.39, 82.44, 79.62 | 72.00-87.10 |
| `flight_dregon_spinning_nosource_room2@1511905206.978+8_motors_command` | spinning_nosource_room2 | 1511905206.978 | 8.000 | `motors_command` | 8 | 247 | 4 | 83.34, 78.46, 81.60, 81.59 | 75.03-87.11 |
| `flight_dregon_updown_nosource_room2@1511903578.348+4_motors_command` | updown_nosource_room2 | 1511903578.348 | 4.000 | `motors_command` | 8 | 122 | 4 | 82.94, 75.73, 81.03, 78.06 | 67.07-85.61 |
| `flight_dregon_updown_nosource_room2@1511903584.348+8_motors_command` | updown_nosource_room2 | 1511903584.348 | 8.000 | `motors_command` | 8 | 247 | 4 | 84.31, 76.69, 81.51, 79.69 | 61.61-89.03 |

## michaels-cruise

13 of 13 specs built (4.5 s, git `a4744d1b54a6`).


| support | recording | start (s) | dur (s) | label | mics | frames | rotors | carrier mean (rev/s) | carrier span (rev/s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `flight_michaels_FLY124@8.000+8_rps` | FLY124 | 8.000 | 8.000 | `rps` | 8 | 247 | 4 | 31.64, 36.18, 41.35, 36.38 | 31.52-41.58 |
| `flight_michaels_FLY124@16.000+8_rps` | FLY124 | 16.000 | 8.000 | `rps` | 8 | 247 | 4 | 31.66, 36.18, 41.32, 36.42 | 31.45-41.51 |
| `flight_michaels_FLY124@27.680+8_rps` | FLY124 | 27.680 | 8.000 | `rps` | 8 | 247 | 4 | 57.58, 54.90, 59.90, 54.85 | 31.39-96.58 |
| `flight_michaels_FLY124@40.000+8_rps` | FLY124 | 40.000 | 8.000 | `rps` | 8 | 247 | 4 | 92.08, 74.76, 79.58, 77.27 | 68.47-97.74 |
| `flight_michaels_FLY124@56.000+8_rps` | FLY124 | 56.000 | 8.000 | `rps` | 8 | 247 | 4 | 91.57, 73.96, 81.93, 75.33 | 71.59-93.93 |
| `flight_michaels_FLY125@16.000+8_rps_refined` | FLY125 | 16.000 | 8.000 | `rps_refined` | 8 | 247 | 4 | 89.96, 74.30, 81.42, 75.39 | 69.09-95.95 |
| `flight_michaels_FLY125@32.000+8_rps_refined` | FLY125 | 32.000 | 8.000 | `rps_refined` | 8 | 247 | 4 | 91.80, 74.91, 80.99, 75.23 | 72.59-94.38 |
| `flight_michaels_FLY125@48.000+8_rps_refined` | FLY125 | 48.000 | 8.000 | `rps_refined` | 8 | 247 | 4 | 89.41, 75.01, 81.32, 77.44 | 68.75-93.50 |
| `flight_michaels_FLY125@64.000+8_rps_refined` | FLY125 | 64.000 | 8.000 | `rps_refined` | 8 | 247 | 4 | 91.06, 74.48, 82.84, 74.48 | 69.57-94.14 |
| `flight_michaels_FLY125@96.000+8_rps_refined` | FLY125 | 96.000 | 8.000 | `rps_refined` | 8 | 247 | 4 | 90.09, 74.62, 81.29, 76.74 | 71.46-93.83 |
| `flight_michaels_FLY125@112.000+8_rps_refined` | FLY125 | 112.000 | 8.000 | `rps_refined` | 8 | 247 | 4 | 91.62, 74.05, 80.97, 75.98 | 69.57-97.41 |
| `flight_michaels_FLY125@128.000+8_rps_refined` | FLY125 | 128.000 | 8.000 | `rps_refined` | 8 | 247 | 4 | 91.61, 74.21, 81.80, 73.92 | 70.72-95.16 |
| `flight_michaels_FLY125@144.000+8_rps_refined` | FLY125 | 144.000 | 8.000 | `rps_refined` | 8 | 247 | 4 | 91.57, 74.24, 82.86, 75.27 | 69.71-97.10 |

## Bytes

* `bench-points`: 1749.6 MB of `.npz` (power as float32)
* `dregon-bench`: 102.1 MB of `.npz` (power as float32)
* `dregon-floor`: 76.1 MB of `.npz` (power as float32)
* `michaels-cruise`: 132.3 MB of `.npz` (power as float32)

The `.npz` caches are NOT committed (`results/**` is gitignored and they are hundreds of MB); `index.json` and this file are. Every row carries its `spec`, so a consumer rebuilds one support with `supports.load_support(<spec>)` and a whole set with

```bash
set -a; . ./.env; set +a          # R2 credentials for the dload stream
PYTHONPATH=src python scripts/noise_v2_supports.py build --set <set>
```

Sourcing the credentials is not optional: a remote job's worktree is a bare git checkout with NO `.env` in it (omnirun ships the secrets to `$JOB_DIR/.env` and sources them in its own bootstrap), so a build whose environment carries no R2 keys now stops on the credential check instead of streaming into a botocore `NoCredentialsError`.
