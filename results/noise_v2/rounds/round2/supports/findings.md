# Noise model v2 — supports

Every support is one observation window at 16 kHz in the frozen periodogram
convention `|rfft(x*w)|^2 / sum(w^2)` (periodic Hann, one-sided,
`revised_eval.window_periodogram`): bench = ONE frame over the whole segment,
flight = NFFT 2048 / hop 512. Numbers below are read from `index.json`, which is
written by `scripts/noise_v2_supports.py build`.

BENCH STATIONARITY RULE REVISION 2 (2026-09-17). A candidate sample must now also
lie where the band-limited (30 Hz - 7.9 kHz) median level over a sliding
`BENCH_MIN_SEGMENT_S` window is within `BENCH_LEVEL_TOL_DB` = 6 dB of the
recording's loudest such window, the accepted window is certified by the line
margin measured INSIDE it (not over the recording), the carrier is refined on the
window and FROZEN (the fit has no carrier parameter), and revision 1's wide-band
residual test is gone. Revision 1 scored frequency-residual stationarity alone,
which silence satisfies perfectly, and put 12 of the 21 DREGON bench windows
10-33 dB below the loudest window of their own recording — see
`results/noise_v2/rounds/round1/bench_diag/findings.md`. `level_deficit_db`,
`line_margin_db` and `carrier_recording_rev_s` in `index.json` are the new
per-recording evidence.

Sets present: bench-points, dregon-bench. 
Total supports cached: 156.

## bench-points

135 of 135 specs built (21.5 s, git `7a7303014e64`).

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

21 of 21 specs built (78.4 s, git `7a7303014e64`).

The rev-2 rule (order 60-80, residual +-1 Hz averaged over 2 s, level gate within 6 dB, in-window margin >= 3 dB, minimum 4 s): **15 of 21 recordings pass**.

Passing segments run 11.66-37.82 s (median 12.04 s, total 248.1 s of stationary bench material).

Failing recordings (window kept and cached, flagged `FAIL`; under rule rev 2 a bench recording fails on the LINE MARGIN measured inside its own window, against 3 dB): `bench_dregon_Motor1_70` margin 2.39 dB (recording 2.70 dB), window 31.69 s, level deficit 0.37 dB; `bench_dregon_Motor1_90` margin 2.51 dB (recording 3.22 dB), window 12.79 s, level deficit 0.62 dB; `bench_dregon_Motor2_70` margin 2.35 dB (recording 3.44 dB), window 11.80 s, level deficit 0.31 dB; `bench_dregon_Motor3_90` margin 2.60 dB (recording 4.48 dB), window 11.80 s, level deficit 0.88 dB; `bench_dregon_Motor4_70` margin 1.49 dB (recording 3.30 dB), window 11.99 s, level deficit 0.71 dB; `bench_dregon_Motor4_90` margin 1.13 dB (recording 2.67 dB), window 11.64 s, level deficit 1.23 dB

Level gate (rev 2): every window sits 0.29-1.23 dB below the loudest 4 s window of its own recording, against 0.2-32.8 dB under rev 1 (`results/noise_v2/rounds/round1/bench_diag/census.json`). In-window line margins run 1.13-18.15 dB.

Longest +-1 Hz spans over all 21 recordings: min 11.64 s, median 11.99 s, max 37.82 s.

Carrier refinement (survey speed -> demodulated line): |shift| up to 0.0746 rev/s, median 0.0166 rev/s. At order 70 a 0.01 rev/s error already displaces the line by 0.7 Hz, most of the tolerance, which is why the carrier is refined before the rule is applied.

| support | pass | segment (s) | dur (s) | longest +-1 Hz (s) | level deficit (dB) | in-window margin (dB) | order | survey (rev/s) | carrier (rev/s) | shift vs survey | shift vs R1 | residual std (Hz) | mics | bins |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bench_dregon_Motor1_50` | PASS | 5.72-37.27 | 31.550 | 31.55 | 0.37 | 8.71 | [63] | 49.0000 | 49.0151 | +0.0151 | +0.0079 | 0.308 | 8 | 252401 |
| `bench_dregon_Motor1_60` | PASS | 4.15-41.97 | 37.820 | 37.82 | 0.48 | 10.56 | [74] | 58.6900 | 58.7561 | +0.0661 | +0.0027 | 0.187 | 8 | 302561 |
| `bench_dregon_Motor1_70` | FAIL | 4.30-35.99 | 31.690 | 31.69 | 0.37 | 2.39 | [62] | 68.3000 | 68.3166 | +0.0166 | -0.0009 | 0.152 | 8 | 253521 |
| `bench_dregon_Motor1_80` | PASS | 3.98-15.69 | 11.710 | 11.71 | 0.42 | 3.45 | [67] | 78.0600 | 78.0528 | -0.0072 | -0.0069 | 0.132 | 8 | 93681 |
| `bench_dregon_Motor1_90` | FAIL | 4.40-17.19 | 12.790 | 12.79 | 0.62 | 2.51 | [61] | 88.1800 | 88.1783 | -0.0017 | -0.0061 | 0.220 | 8 | 102321 |
| `bench_dregon_Motor2_50` | PASS | 3.83-16.10 | 12.270 | 12.27 | 0.39 | 16.05 | [63] | 48.3600 | 48.4346 | +0.0746 | +0.0261 | 0.380 | 8 | 98161 |
| `bench_dregon_Motor2_60` | PASS | 4.07-16.06 | 11.990 | 11.99 | 0.50 | 18.15 | [63] | 58.0700 | 58.1441 | +0.0741 | +0.0462 | 0.276 | 8 | 95921 |
| `bench_dregon_Motor2_70` | FAIL | 3.85-15.65 | 11.800 | 11.80 | 0.31 | 2.35 | [70] | 67.5200 | 67.5348 | +0.0148 | +0.0123 | 0.134 | 8 | 94401 |
| `bench_dregon_Motor2_80` | PASS | 4.21-16.63 | 12.420 | 12.42 | 0.29 | 7.11 | [63] | 77.1800 | 77.2105 | +0.0305 | +0.0210 | 0.227 | 8 | 99361 |
| `bench_dregon_Motor2_90` | PASS | 5.27-16.93 | 11.660 | 11.66 | 0.49 | 5.44 | [72] | 86.7000 | 86.7188 | +0.0188 | +0.0006 | 0.186 | 8 | 93281 |
| `bench_dregon_Motor3_50` | PASS | 3.04-14.87 | 11.830 | 11.83 | 0.54 | 10.65 | [63] | 49.0650 | 49.1353 | +0.0703 | +0.0268 | 0.188 | 8 | 94641 |
| `bench_dregon_Motor3_60` | PASS | 3.75-16.33 | 12.580 | 12.58 | 0.71 | 9.91 | [70] | 58.9000 | 58.9490 | +0.0490 | +0.0185 | 0.187 | 8 | 100641 |
| `bench_dregon_Motor3_70` | PASS | 4.87-16.75 | 11.880 | 11.88 | 0.43 | 13.88 | [63] | 68.5600 | 68.6026 | +0.0426 | +0.0280 | 0.224 | 8 | 95041 |
| `bench_dregon_Motor3_80` | PASS | 4.06-16.44 | 12.380 | 12.38 | 0.60 | 3.96 | [65] | 78.8200 | 78.8251 | +0.0051 | +0.0117 | 0.152 | 8 | 99041 |
| `bench_dregon_Motor3_90` | FAIL | 4.84-16.64 | 11.800 | 11.80 | 0.88 | 2.60 | [70] | 88.2400 | 88.2622 | +0.0222 | +0.0133 | 0.111 | 8 | 94401 |
| `bench_dregon_Motor4_50` | PASS | 3.76-15.80 | 12.040 | 12.04 | 0.40 | 10.69 | [63] | 49.7400 | 49.8057 | +0.0657 | +0.0370 | 0.366 | 8 | 96321 |
| `bench_dregon_Motor4_60` | PASS | 2.83-14.66 | 11.830 | 11.83 | 0.79 | 5.55 | [62] | 59.6600 | 59.6914 | +0.0314 | +0.0112 | 0.177 | 8 | 94641 |
| `bench_dregon_Motor4_70` | FAIL | 3.81-15.80 | 11.990 | 11.99 | 0.71 | 1.49 | [68] | 69.3100 | 69.3222 | +0.0122 | +0.0039 | 0.166 | 8 | 95921 |
| `bench_dregon_Motor4_80` | PASS | 5.30-17.01 | 11.710 | 11.71 | 0.78 | 4.34 | [60] | 79.3800 | 79.3781 | -0.0019 | +0.0022 | 0.092 | 8 | 93681 |
| `bench_dregon_Motor4_90` | FAIL | 5.44-17.08 | 11.640 | 11.64 | 1.23 | 1.13 | [69] | 89.1400 | 89.1266 | -0.0134 | +0.0043 | 0.091 | 8 | 93121 |
| `bench_dregon_allMotors_70` | PASS | 3.85-38.25 | 34.400 | 34.40 | 1.15 | 5.12, 5.11, 5.12, 9.37 | [67, 64, 63, 63] | 64.6471, 67.6588, 68.7396, 69.5655 | 64.6363, 67.6596, 68.7364, 69.5652 | -0.0108, +0.0008, -0.0032, -0.0003 | -0.0033, -0.0022, -0.0015, -0.0066 | 0.276, 0.242, 0.249, 0.232 | 8 | 275201 |

## Bytes

* `bench-points`: 1749.6 MB of `.npz` (power as float32)
* `dregon-bench`: 137.3 MB of `.npz` (power as float32)

The `.npz` caches are NOT committed (`results/**` is gitignored and they are hundreds of MB); `index.json` and this file are. Every row carries its `spec`, so a consumer rebuilds one support with `supports.load_support(<spec>)` and a whole set with

```bash
set -a; . ./.env; set +a          # R2 credentials for the dload stream
PYTHONPATH=src python scripts/noise_v2_supports.py build --set <set>
```

Sourcing the credentials is not optional: a remote job's worktree is a bare git checkout with NO `.env` in it (omnirun ships the secrets to `$JOB_DIR/.env` and sources them in its own bootstrap), so a build whose environment carries no R2 keys now stops on the credential check instead of streaming into a botocore `NoCredentialsError`.
