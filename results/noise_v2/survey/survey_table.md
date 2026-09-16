## Corpora

| Corpus | Location | Type | Rigs | Telemetry | Speed source | fs / ch / duration | Licence | Verdict | Reason |
|---|---|---|---|---|---|---|---|---|---|
| DREGON single-motor bench | dload:DREGON-frames (recordings motor_Motor{1-4}_{50..90}, motor_allMotors_70) | static single-motor bench; motor clamped, one throttle per recording | 1 airframe (MikroKopter), 4 individually driven motors + 1 all-motors run | none logged; the THROTTLE SETPOINT is in the recording name | estimated (Welch harmonic sum + odd-harmonic octave check, tol from the two-half spread) — cross-checked against the validated throttle law rate = 0.975*throttle + 0.37 rev/s | 8 ch, 44.1 kHz, 25-45 s files with 11-35 s of motor-on | research use (INRIA DREGON) | USABLE (20/21 recordings) | one throttle per recording, one rotor per recording, and a non-acoustic reference (the throttle law) to score the estimator against |
| SPCUP19 AGH single rotors | dload:SPCUP19-egonoise (AGH/ego-noise/single rotors/0..7) | static single-rotor takes, one mic | 1 (AGH quadrotor, model unspecified) | none | estimated (same estimator); cross-checked against the accepted blind-campaign readings (blind-corpus-annotation.md) | 1 ch, 44.1 kHz, 16.4-17.6 s | free for personal, educational and academic use only | USABLE (7/8 recordings) | single-source static takes; the blind campaign already accepted all 8 under the one-source rule, so the readings are independently checkable |
| SPCUP19 static / hover (quadrotor) | dload:SPCUP19-egonoise (AGH static clean/corrupted, Diagonal_Unloading static, Idea_ssu stationary, Shout_COOEE StaticSubmission1/2) | four-rotor static / hover windows on 4 team rigs | 4 (AGH quadrotor, DJI Phantom 4 PRO, DJI Phantom 4 GL300C, Intel Aero RTF) | none | none accepted (estimator margins below the 3 dB rule) | 1-8 ch, 44.1/48 kHz, 11.5-85.9 s | free for personal, educational and academic use only | REJECTED (0/18 recordings) | four combs inside a few rev/s make every rival comb score almost as well as the accepted one, so no window clears the 3 dB margin — the same geometry that flattened the blind campaign's ridge clearance (no SPCUP static window cleared its off-comb null there either) |
| SPCUP19 ChuMS propeller rig | dregon.inria.fr SPCUP19_ChuMS_data.zip -> UAV_rotor_recordings.mat (TestResults.Test(1..9)) | STATIC PROPELLER RIG: 1, 2 or 3 propellers running, 3 repeats each, 8 calibrated mics on a 1 m arc (MicPositions in the .mat) | 1 rig (Skylark M4-680 / Dotterel) in 3 propeller-count conditions | none; the .mat records the propeller COUNT and per-mic OASPL, no RPM | estimated (same estimator) | 8 ch, 44.1 kHz, 73-75 s, calibrated in Pa | free for personal, educational and academic use only | PARTLY USABLE (0/9 recordings) | a genuine bench rig that the published SPCUP19-egonoise frames expose only as 216 unlabelled arrays (Freq/SPL/RawTruncatedCalibrated per mic); the recordings are stationary but the rig runs several propellers at uncontrolled speeds, so only the runs that clear the 3 dB margin are kept |
| DroneAudioSet (drone-only) | HuggingFace ahlab-drone-project/DroneAudioSet, subset drone-only/ (28 parquet shards, 3.1 GiB, 168 recordings); published as dload:DroneAudioSet (88.4 GiB, all subsets) | rig-mounted static: the quadcopter is bolted to an aluminium frame at 1.5 m and run at a fixed throttle (the paper's hover emulation) | 2 (DJI F450 'D_large', 450 mm wheelbase, 9.4x5.0 props; DJI F330 'D_small', 330 mm, 8x4.5 props) x 2 throttles (low/high) x mic distance 25/50 cm | none; throttle is a two-level label. The paper states spectral lines 168/235 Hz (D_large low/high) and 156/259 Hz (D_small low/high) | estimated (same estimator); cross-checked against the paper's stated lines via the measured blade-pass frequency 2*f | 8 ch (M_up / M_down arrays) or 1 ch (M_center Soundskrit), 16 kHz, 30-152 s | MIT | PARTLY USABLE (52/168 recordings) | the largest static multichannel drone corpus and the only one with two airframes; the recordings are stationary but the four rotors are not near-equal on every mic, so the reading is accepted per recording |
| AVQ constant-throttle ego-noise | dload:AVQ (sequences S1_seq1/S1_seq2/S1_seq3/S2_seq1; the same recordings are in dload:AVQ-egonoise at channel 0, 16 kHz) | BENCH/CONSTANT-THROTTLE EGO-NOISE: the four sequences the AVQ spec table marks Type = EO with Drone = constant — S1 seq1 50 % (120 s), S1 seq2 100 % (120 s), S1 seq3 150 % (40 s), S2 seq1 100 % (210 s) — read in consecutive 30 s windows. The other eight sequences are excluded by type: S2_seq2 is EO at DYNAMIC throttle, S2_seq5/seq6 are constant 100 % but speech+ego-noise MIXTURES, and S1_seq5/S2_seq3/seq4/seq7/seq8 are speech with the drone muted or dynamic | 1 quadrotor (onboard 8-mic circular array + camera, QMUL) | none logged; the THROTTLE SETTING per sequence is in the spec table (50/100/150 %), which is what makes these four sequences bench class | estimated (same estimator, multi-rotor mode) | 8 ch, 44.1 kHz; 40-210 s per sequence -> 30 s windows | free for academic/research use (courtesy of Lin Wang, QMUL) | USABLE (4/16 recordings) | the first survey pass refused the whole corpus as 'free flight' without reading the spec table. The table's Drone column is explicit: four sequences hold ego-noise only at a CONSTANT throttle setting, which is the bench-class definition used here (stationary window, one speed per rotor). The blind-campaign octave warning (median fvk_ratio_double 1.044) was measured on the FLIGHT sequences, not on these |
| drone_audio (Al-Emadi IWCMC 2019) | data/drone_audio, dload:drone_audio (Binary_Drone_Audio/yes_drone, 1332 clips; Multiclass bebop_1/membo_1) | indoor propeller recordings of a Parrot Bebop and a Parrot Mambo, cut into 1 s clips; the sibling unknown/ class is ESC-50 + white noise + silence | 2 (Bebop, Mambo) | none | none accepted | 1 ch, 16 kHz, 0.65-1.02 s per clip | none stated (citation request only, IWCMC 2019 paper) | REJECTED (0/24 recordings) | the clips are 1.02 s, an eighth of the 8 s stationary window the tolerance rule needs, and there is no take-level grouping that would let clips be re-joined; the sampled clips also fail the margin rule outright |
| zenodo_drone_noises | data/zenodo_drone_noises (all_drone_noises.zip -> noises-train-drones/n116..n120, noises-test-drones/n121..n122) | 7 unlabelled drone-noise clips; no rig, session or setup recorded anywhere in the zip (no README, no metadata file) | unknown (one unknown rig per file at best) | none | none accepted for a rig; per-file estimates reported | 1 ch, 8 kHz (n121/n122) or 44.1 kHz, 40-215 s | none stated | REJECTED (1/7 recordings) | no rig identity, so a per-rig noise parameter cannot be attached to the point even where the estimator reads a speed; the recordings also drift (flight, not bench) and mostly fail the margin rule |
| KAIST-rotating-acoustic (control) | dload:KAIST-rotating-acoustic | industrial rotating-machine testbed, mono bench, dataset-stated 3010 RPM | 1 (not a drone) | the stated nominal 3010 RPM = 50.167 rev/s | paper-stated nominal; the estimator is scored against it | 1 ch, 51.2 kHz, 60 s, 5 recordings (fault + severity per file) | CC BY 4.0 | CONTROL ONLY (0/5 recordings) | kept as the out-of-domain control the blind campaign used: bearing fault lines are real, strong and NOT octaves of the shaft, so this corpus measures whether the gate refuses what it cannot read. Not a drone fit point |
| DronePrint (Kolamunna et al. 2021) | OSF repository via github.com/DronePrint/DronePrint (not in this repo) | far-field FREE FLIGHT: 5 drone classes recorded with a RODE NTG4 shotgun mic at ~20 m altitude within a 50 m radius, plus YouTube-scraped clips | 5 recorded (Bebop 2, Mavic Pro, Phantom 4 Pro, Spark, Matrice 100) + 15 online classes | none | none | 1 ch, 44.1 kHz | open (OSF), attribution | REJECTED (not downloaded) | free flight at 20 m with a directional ground mic: the comb decoheres, the speed is unknown and the manoeuvre is unconstrained — class G of the blind-corpus plan (mono far field, refused by default) |
| MAVD | two unrelated datasets answer to this name: MAVD (Mandarin audio-visual with depth, github.com/SpringHuo/MAVD) and MAVD-Traffic (Montevideo audio-visual traffic) | neither is drone ego-noise: speech+depth corpus / street-traffic corpus | none | n/a | n/a | n/a | MAVD: access by e-mail request to the authors | REJECTED (not a drone-noise corpus) | the name does not resolve to a rotor-noise dataset; the speech variant is also gated behind an e-mail request, which the survey rules out |
| DroneNoise Database (Salford) | salford.figshare.com/articles/dataset/DroneNoise_Database/22133411 (figshare article 22133411, 175 files, 742 MB) | field OVERFLIGHT campaign (Edzell, Scotland, 2022-08-17): sUAS flying over a ground microphone array, 3 events per configuration | several sUAS types (file stems Ed_3p / Ed_Fp / Ed_M3 / Ed_Yn) | none published with the audio (no RPM channel in the file set) | none | ground mics M1..M?, ~45-68 s WAV per event | CC BY 4.0 | REJECTED (metadata only, not downloaded) | overflight recordings are Doppler-shifted and non-stationary by construction, and no rotor speed is published, so no fit point can be built from them |
| ESC-50 | github.com/karolpiczak/ESC-50 (already inside data/drone_audio as the unknown/ negatives) | 2000 five-second environmental clips, 50 classes; no drone class (the closest are helicopter and chainsaw) | none | none | none | 1 ch, 44.1 kHz, 5 s | CC BY-NC 3.0 | REJECTED | not a drone corpus at all; it is the negative-class pool of drone_audio and carries no rotating-source speed |

## Fit points (one row per usable recording)

| Corpus | Recording | Rig | Condition | f_i rev/s (one per resolved rotor) | f̄ rev/s | Tol rev/s | Spread rev/s | n_res / n_rotors | Octave | Margin dB | Margin family-only dB | Half Δ rev/s | Window s | ch | fs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AVQ | S1_seq1_w1 | avq_quadrotor | constant throttle 50%, ego-noise only | 78.00 | 78.00 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 4.68 | 4.68 | 0.02 | 30.00 | 8 | 44100 |
| AVQ | S1_seq1_w3 | avq_quadrotor | constant throttle 50%, ego-noise only | 77.79 | 77.79 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 4.37 | 4.37 | 0.06 | 30.00 | 8 | 44100 |
| AVQ | S1_seq2_w0 | avq_quadrotor | constant throttle 100%, ego-noise only | 97.60, 97.97 | 97.78 | 0.54 | 0.37 | 2 / 4 (unresolved) | as_found | 3.79 | 3.79 | 0.54 | 30.00 | 8 | 44100 |
| AVQ | S1_seq2_w3 | avq_quadrotor | constant throttle 100%, ego-noise only | 98.24 | 98.24 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 4.80 | 4.80 | 0.12 | 30.00 | 8 | 44100 |
| DREGON-bench | motor_Motor1_50 | dregon_mikrokopter_single_motor | bench Motor1 throttle 50% | 49.00 | 49.00 | 0.25 | 0.00 | 1 / 1 | as_found | 4.74 | 4.74 | 0.06 | 16.00 | 8 | 44100 |
| DREGON-bench | motor_Motor1_60 | dregon_mikrokopter_single_motor | bench Motor1 throttle 60% | 58.69 | 58.69 | 0.25 | 0.00 | 1 / 1 | halved | 5.36 | 5.36 | 0.01 | 16.00 | 8 | 44100 |
| DREGON-bench | motor_Motor1_70 | dregon_mikrokopter_single_motor | bench Motor1 throttle 70% | 68.30 | 68.30 | 0.25 | 0.00 | 1 / 1 | halved | 9.75 | 9.75 | 0.08 | 16.00 | 8 | 44100 |
| DREGON-bench | motor_Motor1_80 | dregon_mikrokopter_single_motor | bench Motor1 throttle 80% | 78.06 | 78.06 | 0.25 | 0.00 | 1 / 1 | as_found | 8.40 | 8.40 | 0.06 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor1_90 | dregon_mikrokopter_single_motor | bench Motor1 throttle 90% | 88.18 | 88.18 | 0.35 | 0.00 | 1 / 1 | as_found | 10.01 | 10.01 | 0.35 | 13.00 | 8 | 44100 |
| DREGON-bench | motor_Motor2_50 | dregon_mikrokopter_single_motor | bench Motor2 throttle 50% | 48.36 | 48.36 | 0.25 | 0.00 | 1 / 1 | as_found | 4.03 | 4.03 | 0.04 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor2_60 | dregon_mikrokopter_single_motor | bench Motor2 throttle 60% | 58.07 | 58.07 | 0.31 | 0.00 | 1 / 1 | halved | 5.52 | 5.52 | 0.31 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor2_70 | dregon_mikrokopter_single_motor | bench Motor2 throttle 70% | 67.52 | 67.52 | 0.25 | 0.00 | 1 / 1 | as_found | 5.56 | 5.56 | 0.00 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor2_80 | dregon_mikrokopter_single_motor | bench Motor2 throttle 80% | 77.18 | 77.18 | 0.28 | 0.00 | 1 / 1 | as_found | 7.17 | 7.17 | 0.28 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor2_90 | dregon_mikrokopter_single_motor | bench Motor2 throttle 90% | 86.70 | 86.70 | 0.25 | 0.00 | 1 / 1 | as_found | 9.76 | 9.76 | 0.10 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor3_50 | dregon_mikrokopter_single_motor | bench Motor3 throttle 50% | 49.06 | 49.06 | 0.28 | 0.00 | 1 / 1 | as_found | 5.14 | 5.14 | 0.28 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor3_60 | dregon_mikrokopter_single_motor | bench Motor3 throttle 60% | 58.90 | 58.90 | 0.25 | 0.00 | 1 / 1 | as_found | 5.28 | 5.28 | 0.01 | 13.00 | 8 | 44100 |
| DREGON-bench | motor_Motor3_70 | dregon_mikrokopter_single_motor | bench Motor3 throttle 70% | 68.56 | 68.56 | 0.25 | 0.00 | 1 / 1 | halved | 8.20 | 8.20 | 0.03 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor3_80 | dregon_mikrokopter_single_motor | bench Motor3 throttle 80% | 78.82 | 78.82 | 0.25 | 0.00 | 1 / 1 | as_found | 8.88 | 8.88 | 0.00 | 13.00 | 8 | 44100 |
| DREGON-bench | motor_Motor3_90 | dregon_mikrokopter_single_motor | bench Motor3 throttle 90% | 88.24 | 88.24 | 0.26 | 0.00 | 1 / 1 | as_found | 11.45 | 11.45 | 0.26 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor4_50 | dregon_mikrokopter_single_motor | bench Motor4 throttle 50% | 49.74 | 49.74 | 0.25 | 0.00 | 1 / 1 | as_found | 4.59 | 4.59 | 0.02 | 13.00 | 8 | 44100 |
| DREGON-bench | motor_Motor4_60 | dregon_mikrokopter_single_motor | bench Motor4 throttle 60% | 59.66 | 59.66 | 0.33 | 0.00 | 1 / 1 | halved | 5.11 | 5.11 | 0.33 | 13.00 | 8 | 44100 |
| DREGON-bench | motor_Motor4_70 | dregon_mikrokopter_single_motor | bench Motor4 throttle 70% | 69.31 | 69.31 | 0.25 | 0.00 | 1 / 1 | halved | 7.27 | 7.27 | 0.03 | 13.00 | 8 | 44100 |
| DREGON-bench | motor_Motor4_80 | dregon_mikrokopter_single_motor | bench Motor4 throttle 80% | 79.38 | 79.38 | 0.25 | 0.00 | 1 / 1 | as_found | 9.25 | 9.25 | 0.02 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor4_90 | dregon_mikrokopter_single_motor | bench Motor4 throttle 90% | 89.14 | 89.14 | 0.25 | 0.00 | 1 / 1 | as_found | 9.94 | 9.94 | 0.02 | 12.00 | 8 | 44100 |
| DroneAudioSet | drone1_high_25cm_M_center_File2 | daset_drone1 | rig-mounted static, throttle high, mic M_center at 25cm | 118.69, 119.19 | 118.82 | 0.32 | 0.50 | 2 / 4 (unresolved) | as_found | 5.61 | 5.61 | 0.32 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone1_high_25cm_M_center_File3 | daset_drone1 | rig-mounted static, throttle high, mic M_center at 25cm | 116.88, 117.41 | 117.22 | 0.88 | 0.54 | 2 / 4 (unresolved) | as_found | 5.06 | 5.06 | 0.88 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone1_high_25cm_M_up_File2 | daset_drone1 | rig-mounted static, throttle high, mic M_up at 25cm | 118.62 | 118.62 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 5.72 | 5.72 | 0.14 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone1_high_25cm_M_up_File3 | daset_drone1 | rig-mounted static, throttle high, mic M_up at 25cm | 116.69, 117.19 | 116.78 | 0.70 | 0.50 | 2 / 4 (unresolved) | as_found | 4.32 | 4.32 | 0.70 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_center_File2 | daset_drone1 | rig-mounted static, throttle high, mic M_center at 50cm | 119.64, 120.06 | 120.02 | 0.72 | 0.42 | 2 / 4 (unresolved) | as_found | 5.87 | 5.87 | 0.72 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_center_File3 | daset_drone1 | rig-mounted static, throttle high, mic M_center at 50cm | 59.39 | 59.39 | 0.25 | 0.00 | 1 / 4 (unresolved) | halved | 5.74 | 5.74 | 0.16 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_center_File5 | daset_drone1 | rig-mounted static, throttle high, mic M_center at 50cm | 121.69 | 121.69 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 5.92 | 5.92 | 0.24 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_center_File6 | daset_drone1 | rig-mounted static, throttle high, mic M_center at 50cm | 119.24 | 119.24 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 5.32 | 5.32 | 0.12 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_up_File2 | daset_drone1 | rig-mounted static, throttle high, mic M_up at 50cm | 119.46, 119.83 | 120.00 | 0.25 | 0.38 | 2 / 4 (unresolved) | as_found | 4.45 | 4.45 | 0.10 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_up_File3 | daset_drone1 | rig-mounted static, throttle high, mic M_up at 50cm | 59.36 | 59.36 | 0.25 | 0.00 | 1 / 4 (unresolved) | halved | 5.80 | 5.80 | 0.14 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_up_File5 | daset_drone1 | rig-mounted static, throttle high, mic M_up at 50cm | 120.88, 121.27 | 121.20 | 0.25 | 0.40 | 2 / 4 (unresolved) | as_found | 4.92 | 4.92 | 0.20 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_up_File6 | daset_drone1 | rig-mounted static, throttle high, mic M_up at 50cm | 118.75, 119.14 | 119.18 | 0.64 | 0.39 | 2 / 4 (unresolved) | as_found | 4.27 | 4.27 | 0.64 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_up_silence | daset_drone1 | rig-mounted static, throttle high, mic M_up at 50cm | 118.58 | 118.58 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 5.70 | 5.70 | 0.05 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone1_low_25cm_M_center_File3 | daset_drone1 | rig-mounted static, throttle low, mic M_center at 25cm | 78.40, 82.71, 83.22 | 82.86 | 0.25 | 4.82 | 3 / 4 (unresolved) | as_found | 4.77 | 4.77 | 0.06 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone1_low_25cm_M_center_File5 | daset_drone1 | rig-mounted static, throttle low, mic M_center at 25cm | 88.17 | 88.17 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 3.56 | 1.58 | 0.00 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_center_File1 | daset_drone2 | rig-mounted static, throttle high, mic M_center at 25cm | 134.32, 134.66 | 134.54 | 0.84 | 0.34 | 2 / 4 (unresolved) | as_found | 5.12 | 5.12 | 0.84 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_center_File2 | daset_drone2 | rig-mounted static, throttle high, mic M_center at 25cm | 65.42 | 65.42 | 0.25 | 0.00 | 1 / 4 (unresolved) | halved | 5.67 | 5.67 | 0.13 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_center_File3 | daset_drone2 | rig-mounted static, throttle high, mic M_center at 25cm | 129.74 | 129.74 | 0.60 | 0.00 | 1 / 4 (unresolved) | as_found | 9.67 | 9.67 | 0.60 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_center_File5 | daset_drone2 | rig-mounted static, throttle high, mic M_center at 25cm | 129.33, 129.75 | 129.50 | 0.54 | 0.42 | 2 / 4 (unresolved) | as_found | 6.86 | 6.86 | 0.54 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_down_File2 | daset_drone2 | rig-mounted static, throttle high, mic M_down at 25cm | 130.65 | 130.65 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 5.94 | 5.94 | 0.16 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_down_File3 | daset_drone2 | rig-mounted static, throttle high, mic M_down at 25cm | 129.56 | 129.56 | 0.66 | 0.00 | 1 / 4 (unresolved) | as_found | 7.55 | 7.55 | 0.66 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_down_File5 | daset_drone2 | rig-mounted static, throttle high, mic M_down at 25cm | 129.45, 129.76 | 129.84 | 0.25 | 0.31 | 2 / 4 (unresolved) | as_found | 5.44 | 5.44 | 0.24 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_up_File1 | daset_drone2 | rig-mounted static, throttle high, mic M_up at 25cm | 134.32 | 134.32 | 0.74 | 0.00 | 1 / 4 (unresolved) | as_found | 5.85 | 5.85 | 0.74 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_up_File2 | daset_drone2 | rig-mounted static, throttle high, mic M_up at 25cm | 65.31 | 65.31 | 0.25 | 0.00 | 1 / 4 (unresolved) | halved | 5.37 | 5.37 | 0.11 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_up_File3 | daset_drone2 | rig-mounted static, throttle high, mic M_up at 25cm | 129.04 | 129.04 | 0.40 | 0.00 | 1 / 4 (unresolved) | as_found | 10.03 | 10.03 | 0.40 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_up_File5 | daset_drone2 | rig-mounted static, throttle high, mic M_up at 25cm | 129.30 | 129.30 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 6.79 | 6.79 | 0.04 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_50cm_M_center_File1 | daset_drone2 | rig-mounted static, throttle high, mic M_center at 50cm | 136.54 | 136.54 | 0.78 | 0.00 | 1 / 4 (unresolved) | as_found | 6.11 | 6.11 | 0.78 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone2_high_50cm_M_center_File2 | daset_drone2 | rig-mounted static, throttle high, mic M_center at 50cm | 103.74 | 103.74 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 3.81 | 3.81 | 0.08 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone2_high_50cm_M_up_File1 | daset_drone2 | rig-mounted static, throttle high, mic M_up at 50cm | 136.37 | 136.37 | 0.72 | 0.00 | 1 / 4 (unresolved) | as_found | 5.86 | 5.86 | 0.72 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_50cm_M_up_File2 | daset_drone2 | rig-mounted static, throttle high, mic M_up at 50cm | 103.78 | 103.78 | 0.42 | 0.00 | 1 / 4 (unresolved) | as_found | 3.95 | 3.95 | 0.42 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_center_File2 | daset_drone2 | rig-mounted static, throttle low, mic M_center at 25cm | 106.94 | 106.94 | 0.36 | 0.00 | 1 / 4 (unresolved) | as_found | 4.39 | 4.39 | 0.36 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_center_File3 | daset_drone2 | rig-mounted static, throttle low, mic M_center at 25cm | 106.82 | 106.82 | 0.56 | 0.00 | 1 / 4 (unresolved) | as_found | 6.16 | 6.16 | 0.56 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_center_File5 | daset_drone2 | rig-mounted static, throttle low, mic M_center at 25cm | 54.68 | 54.68 | 0.25 | 0.00 | 1 / 4 (unresolved) | halved | 6.55 | 6.55 | 0.01 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_center_File6 | daset_drone2 | rig-mounted static, throttle low, mic M_center at 25cm | 109.06 | 109.06 | 0.26 | 0.00 | 1 / 4 (unresolved) | as_found | 7.19 | 7.19 | 0.26 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_center_silence | daset_drone2 | rig-mounted static, throttle low, mic M_center at 25cm | 108.76 | 108.76 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 8.31 | 8.31 | 0.02 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_down_File2 | daset_drone2 | rig-mounted static, throttle low, mic M_down at 25cm | 106.80 | 106.80 | 0.54 | 0.00 | 1 / 4 (unresolved) | as_found | 4.72 | 4.72 | 0.54 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_down_File3 | daset_drone2 | rig-mounted static, throttle low, mic M_down at 25cm | 106.38 | 106.38 | 0.36 | 0.00 | 1 / 4 (unresolved) | as_found | 5.38 | 5.38 | 0.36 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_down_File5 | daset_drone2 | rig-mounted static, throttle low, mic M_down at 25cm | 109.22 | 109.22 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 5.78 | 5.78 | 0.04 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_down_File6 | daset_drone2 | rig-mounted static, throttle low, mic M_down at 25cm | 108.68 | 108.68 | 0.30 | 0.00 | 1 / 4 (unresolved) | as_found | 6.75 | 6.75 | 0.30 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_down_silence | daset_drone2 | rig-mounted static, throttle low, mic M_down at 25cm | 108.62 | 108.62 | 0.53 | 0.00 | 1 / 4 (unresolved) | as_found | 8.39 | 8.39 | 0.53 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_up_File2 | daset_drone2 | rig-mounted static, throttle low, mic M_up at 25cm | 106.82 | 106.82 | 0.33 | 0.00 | 1 / 4 (unresolved) | as_found | 3.97 | 3.97 | 0.33 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_up_File3 | daset_drone2 | rig-mounted static, throttle low, mic M_up at 25cm | 106.40 | 106.40 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 5.32 | 5.32 | 0.08 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_up_File5 | daset_drone2 | rig-mounted static, throttle low, mic M_up at 25cm | 27.28 | 27.28 | 0.25 | 0.00 | 1 / 4 (unresolved) | halved_twice | 7.48 | 7.48 | 0.01 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_up_File6 | daset_drone2 | rig-mounted static, throttle low, mic M_up at 25cm | 109.12 | 109.12 | 0.48 | 0.00 | 1 / 4 (unresolved) | as_found | 7.22 | 7.22 | 0.48 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_up_silence | daset_drone2 | rig-mounted static, throttle low, mic M_up at 25cm | 108.52 | 108.52 | 0.32 | 0.00 | 1 / 4 (unresolved) | as_found | 9.07 | 9.07 | 0.32 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_50cm_M_center_File2 | daset_drone2 | rig-mounted static, throttle low, mic M_center at 50cm | 105.10 | 105.10 | 0.57 | 0.00 | 1 / 4 (unresolved) | as_found | 4.29 | 4.29 | 0.57 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone2_low_50cm_M_center_File3 | daset_drone2 | rig-mounted static, throttle low, mic M_center at 50cm | 102.78 | 102.78 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 4.19 | 4.19 | 0.24 | 30.00 | 1 | 16000 |
| DroneAudioSet | drone2_low_50cm_M_down_File2 | daset_drone2 | rig-mounted static, throttle low, mic M_down at 50cm | 104.06 | 104.06 | 0.29 | 0.00 | 1 / 4 (unresolved) | as_found | 4.66 | 4.66 | 0.29 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_50cm_M_down_File3 | daset_drone2 | rig-mounted static, throttle low, mic M_down at 50cm | 102.78 | 102.78 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 5.57 | 5.57 | 0.14 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_50cm_M_down_silence | daset_drone2 | rig-mounted static, throttle low, mic M_down at 50cm | 104.38 | 104.38 | 0.28 | 0.00 | 1 / 4 (unresolved) | as_found | 3.32 | 3.32 | 0.28 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_50cm_M_up_File2 | daset_drone2 | rig-mounted static, throttle low, mic M_up at 50cm | 104.38, 104.80 | 104.68 | 0.68 | 0.43 | 2 / 4 (unresolved) | as_found | 3.03 | 3.03 | 0.68 | 30.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_50cm_M_up_File3 | daset_drone2 | rig-mounted static, throttle low, mic M_up at 50cm | 102.32 | 102.32 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 4.29 | 4.29 | 0.19 | 30.00 | 8 | 16000 |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__0 | spcup_AGH | single_rotor | 79.82 | 79.82 | 0.25 | 0.00 | 1 / 1 | as_found | 5.47 | 5.47 | 0.16 | 16.00 | 1 | 44100 |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__1 | spcup_AGH | single_rotor | 132.30 | 132.30 | 0.25 | 0.00 | 1 / 1 | as_found | 6.10 | 6.10 | 0.10 | 16.00 | 1 | 44100 |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__2 | spcup_AGH | single_rotor | 113.54 | 113.54 | 0.25 | 0.00 | 1 / 1 | as_found | 5.99 | 5.99 | 0.00 | 16.00 | 1 | 44100 |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__3 | spcup_AGH | single_rotor | 97.48 | 97.48 | 0.25 | 0.00 | 1 / 1 | as_found | 4.52 | 4.52 | 0.02 | 16.00 | 1 | 44100 |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__5 | spcup_AGH | single_rotor | 98.08 | 98.08 | 0.32 | 0.00 | 1 / 1 | as_found | 5.18 | 5.18 | 0.32 | 16.00 | 1 | 44100 |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__6 | spcup_AGH | single_rotor | 96.78 | 96.78 | 0.25 | 0.00 | 1 / 1 | as_found | 5.02 | 5.02 | 0.02 | 16.00 | 1 | 44100 |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__7 | spcup_AGH | single_rotor | 96.78 | 96.78 | 0.25 | 0.00 | 1 / 1 | as_found | 5.71 | 5.71 | 0.00 | 16.00 | 1 | 44100 |
| zenodo_drone_noises | n117 | zenodo_unknown_n117 | unlabelled drone-noise clip (noises-train-drones) | 97.14 | 97.14 | 0.25 | 0.00 | 1 / 4 (unresolved) | as_found | 4.96 | 4.96 | 0.10 | 30.00 | 1 | 44100 |

## Rejected recordings

| Corpus | Recording | Reading rev/s | Margin dB | Reason |
|---|---|---|---|---|
| AVQ | S1_seq1_w0 | 78.00 | 5.57 | half-to-half 6.56 rev/s > 1.0 |
| AVQ | S1_seq1_w2 | 39.46 | 2.23 | harmonic-sum margin 2.23 dB < 3 dB |
| AVQ | S1_seq2_w1 | 16.33 | -0.29 | harmonic-sum margin -0.29 dB < 3 dB; half-to-half 80.92 rev/s > 1.0 |
| AVQ | S1_seq2_w2 | 97.34 | 2.35 | harmonic-sum margin 2.35 dB < 3 dB; half-to-half 80.84 rev/s > 1.0 |
| AVQ | S1_seq3_w0 | 111.38 | 3.33 | half-to-half 1.50 rev/s > 1.0 |
| AVQ | S2_seq1_w0 | 88.98 | 2.19 | harmonic-sum margin 2.19 dB < 3 dB; half-to-half 4.24 rev/s > 1.0 |
| AVQ | S2_seq1_w1 | 45.48 | -0.81 | harmonic-sum margin -0.81 dB < 3 dB |
| AVQ | S2_seq1_w2 | 45.24 | -0.43 | harmonic-sum margin -0.43 dB < 3 dB; half-to-half 48.50 rev/s > 1.0 |
| AVQ | S2_seq1_w3 | 93.48 | 0.74 | harmonic-sum margin 0.74 dB < 3 dB |
| AVQ | S2_seq1_w4 | 45.03 | -0.67 | harmonic-sum margin -0.67 dB < 3 dB |
| AVQ | S2_seq1_w5 | 92.52 | 1.58 | harmonic-sum margin 1.58 dB < 3 dB |
| AVQ | S2_seq1_w6 | 93.12 | 0.49 | harmonic-sum margin 0.49 dB < 3 dB; half-to-half 48.24 rev/s > 1.0 |
| DREGON-bench | motor_allMotors_70 | 68.66 | 0.77 | harmonic-sum margin 0.77 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_center_File1 | 120.02 | 0.91 | harmonic-sum margin 0.91 dB < 3 dB; half-to-half 2.26 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_center_File4 | 112.85 | 2.10 | harmonic-sum margin 2.10 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_center_File5 | 106.20 | 1.64 | harmonic-sum margin 1.64 dB < 3 dB; half-to-half 9.36 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_center_File6 | 119.26 | 1.62 | harmonic-sum margin 1.62 dB < 3 dB; half-to-half 34.48 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_center_silence | 118.70 | 4.42 | half-to-half 39.48 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_down_File1 | 118.92 | 1.51 | harmonic-sum margin 1.51 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_down_File2 | 118.44 | 2.73 | harmonic-sum margin 2.73 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_down_File3 | 116.74 | 1.66 | harmonic-sum margin 1.66 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_down_File4 | 112.48 | 1.57 | harmonic-sum margin 1.57 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_down_File5 | 100.50 | 0.43 | harmonic-sum margin 0.43 dB < 3 dB; half-to-half 15.01 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_down_File6 | 119.50 | 0.26 | harmonic-sum margin 0.26 dB < 3 dB; half-to-half 11.24 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_down_silence | 118.68 | 2.11 | harmonic-sum margin 2.11 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_up_File1 | 120.02 | 1.98 | harmonic-sum margin 1.98 dB < 3 dB; half-to-half 6.09 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_up_File4 | 112.62 | 1.81 | harmonic-sum margin 1.81 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_up_File5 | 105.94 | 2.19 | harmonic-sum margin 2.19 dB < 3 dB; half-to-half 9.46 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_up_File6 | 119.34 | 2.00 | harmonic-sum margin 2.00 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_up_silence | 118.46 | 5.16 | half-to-half 101.66 rev/s > 1.0 |
| DroneAudioSet | drone1_high_50cm_M_center_File1 | 123.76 | 2.91 | harmonic-sum margin 2.91 dB < 3 dB |
| DroneAudioSet | drone1_high_50cm_M_center_File4 | 110.39 | 1.51 | harmonic-sum margin 1.51 dB < 3 dB; half-to-half 3.80 rev/s > 1.0 |
| DroneAudioSet | drone1_high_50cm_M_center_silence | 118.98 | 5.55 | half-to-half 1.01 rev/s > 1.0 |
| DroneAudioSet | drone1_high_50cm_M_down_File1 | 123.12 | 1.73 | harmonic-sum margin 1.73 dB < 3 dB; half-to-half 7.73 rev/s > 1.0 |
| DroneAudioSet | drone1_high_50cm_M_down_File2 | 119.74 | 1.85 | harmonic-sum margin 1.85 dB < 3 dB |
| DroneAudioSet | drone1_high_50cm_M_down_File3 | 118.54 | 2.18 | harmonic-sum margin 2.18 dB < 3 dB |
| DroneAudioSet | drone1_high_50cm_M_down_File4 | 132.22 | -0.25 | harmonic-sum margin -0.25 dB < 3 dB; half-to-half 9.96 rev/s > 1.0 |
| DroneAudioSet | drone1_high_50cm_M_down_File5 | 121.30 | 1.56 | harmonic-sum margin 1.56 dB < 3 dB |
| DroneAudioSet | drone1_high_50cm_M_down_File6 | 119.18 | 1.82 | harmonic-sum margin 1.82 dB < 3 dB |
| DroneAudioSet | drone1_high_50cm_M_down_silence | 118.54 | 2.43 | harmonic-sum margin 2.43 dB < 3 dB |
| DroneAudioSet | drone1_high_50cm_M_up_File1 | 123.58 | 2.72 | harmonic-sum margin 2.72 dB < 3 dB; half-to-half 98.41 rev/s > 1.0 |
| DroneAudioSet | drone1_high_50cm_M_up_File4 | 110.71 | 2.44 | harmonic-sum margin 2.44 dB < 3 dB; half-to-half 4.10 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_center_File1 | 82.72 | 1.74 | harmonic-sum margin 1.74 dB < 3 dB; half-to-half 3.01 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_center_File2 | 82.86 | 2.32 | harmonic-sum margin 2.32 dB < 3 dB |
| DroneAudioSet | drone1_low_25cm_M_center_File4 | 126.80 | 0.49 | harmonic-sum margin 0.49 dB < 3 dB; half-to-half 30.45 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_center_File6 | 88.54 | 2.31 | harmonic-sum margin 2.31 dB < 3 dB; half-to-half 21.48 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_center_silence | 89.30 | 1.86 | harmonic-sum margin 1.86 dB < 3 dB; half-to-half 124.10 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_down_File1 | 87.01 | 0.46 | harmonic-sum margin 0.46 dB < 3 dB; half-to-half 1.19 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_down_File2 | 85.44 | 1.04 | harmonic-sum margin 1.04 dB < 3 dB |
| DroneAudioSet | drone1_low_25cm_M_down_File3 | 42.08 | 0.36 | harmonic-sum margin 0.37 dB < 3 dB |
| DroneAudioSet | drone1_low_25cm_M_down_File4 | 66.19 | -0.36 | harmonic-sum margin -0.36 dB < 3 dB; half-to-half 60.50 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_down_File5 | 87.92 | 1.18 | harmonic-sum margin 1.18 dB < 3 dB |
| DroneAudioSet | drone1_low_25cm_M_down_File6 | 88.36 | 2.26 | harmonic-sum margin 2.26 dB < 3 dB |
| DroneAudioSet | drone1_low_25cm_M_down_silence | 89.18 | 1.63 | harmonic-sum margin 1.63 dB < 3 dB |
| DroneAudioSet | drone1_low_25cm_M_up_File1 | 86.98 | 1.46 | harmonic-sum margin 1.46 dB < 3 dB; half-to-half 3.32 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_up_File2 | 82.62 | 1.67 | harmonic-sum margin 1.67 dB < 3 dB |
| DroneAudioSet | drone1_low_25cm_M_up_File3 | 82.54 | 3.33 | half-to-half 61.68 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_up_File4 | 126.50 | 0.41 | harmonic-sum margin 0.41 dB < 3 dB; half-to-half 69.08 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_up_File5 | 87.92 | 2.53 | harmonic-sum margin 2.53 dB < 3 dB |
| DroneAudioSet | drone1_low_25cm_M_up_File6 | 44.35 | -0.92 | harmonic-sum margin -0.92 dB < 3 dB; half-to-half 5.80 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_up_silence | 16.16 | -2.96 | harmonic-sum margin -2.96 dB < 3 dB; half-to-half 123.71 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_center_File1 | 62.10 | 1.05 | harmonic-sum margin 1.05 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_center_File2 | 83.05 | 1.56 | harmonic-sum margin 1.56 dB < 3 dB; half-to-half 1.36 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_center_File3 | 92.72 | 1.95 | harmonic-sum margin 1.95 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_center_File4 | 142.44 | 0.59 | harmonic-sum margin 0.59 dB < 3 dB; half-to-half 2.66 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_center_File5 | 87.19 | 2.11 | harmonic-sum margin 2.11 dB < 3 dB; half-to-half 4.02 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_center_File6 | 84.91 | 2.85 | harmonic-sum margin 2.85 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_center_silence | 84.96 | 1.21 | harmonic-sum margin 1.21 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_down_File1 | 133.38 | 0.78 | harmonic-sum margin 0.78 dB < 3 dB; half-to-half 9.08 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_down_File2 | 66.56 | -0.96 | harmonic-sum margin -0.96 dB < 3 dB; half-to-half 16.88 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_down_File3 | 92.38 | 3.97 | window 2.00 s < 8 s; window too short for a two-half stability test |
| DroneAudioSet | drone1_low_50cm_M_down_File4 | 66.03 | -0.05 | harmonic-sum margin -0.05 dB < 3 dB; half-to-half 61.40 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_down_File5 | 86.96 | 0.88 | harmonic-sum margin 0.88 dB < 3 dB; half-to-half 3.66 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_down_File6 | 89.36 | 0.03 | harmonic-sum margin 0.03 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_down_silence | 56.52 | -0.36 | harmonic-sum margin -0.36 dB < 3 dB; half-to-half 2.89 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_up_File1 | 124.26 | 0.30 | harmonic-sum margin 0.30 dB < 3 dB; half-to-half 4.94 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_up_File2 | 18.41 | -1.73 | harmonic-sum margin -1.73 dB < 3 dB; half-to-half 22.43 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_up_File3 | 23.19 | -1.60 | harmonic-sum margin -1.60 dB < 3 dB; half-to-half 69.35 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_up_File4 | 16.22 | -0.14 | harmonic-sum margin -0.14 dB < 3 dB; half-to-half 4.08 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_up_File5 | 86.98 | 1.74 | harmonic-sum margin 1.74 dB < 3 dB; half-to-half 4.18 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_up_File6 | 84.74 | 1.90 | harmonic-sum margin 1.90 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_up_silence | 84.92 | 1.54 | harmonic-sum margin 1.54 dB < 3 dB |
| DroneAudioSet | drone2_high_25cm_M_center_File4 | 128.45 | 0.72 | harmonic-sum margin 0.72 dB < 3 dB; half-to-half 2.76 rev/s > 1.0 |
| DroneAudioSet | drone2_high_25cm_M_center_File6 | 127.18 | 1.29 | harmonic-sum margin 1.29 dB < 3 dB; half-to-half 22.82 rev/s > 1.0 |
| DroneAudioSet | drone2_high_25cm_M_center_silence | 126.41 | 0.29 | harmonic-sum margin 0.29 dB < 3 dB |
| DroneAudioSet | drone2_high_25cm_M_down_File1 | 134.44 | 4.58 | half-to-half 1.30 rev/s > 1.0 |
| DroneAudioSet | drone2_high_25cm_M_down_File4 | 128.00 | 0.15 | harmonic-sum margin 0.15 dB < 3 dB; half-to-half 4.20 rev/s > 1.0 |
| DroneAudioSet | drone2_high_25cm_M_down_File6 | 127.20 | 3.02 | half-to-half 22.88 rev/s > 1.0 |
| DroneAudioSet | drone2_high_25cm_M_down_silence | 126.30 | 1.47 | harmonic-sum margin 1.47 dB < 3 dB |
| DroneAudioSet | drone2_high_25cm_M_up_File4 | 127.30 | 0.87 | harmonic-sum margin 0.87 dB < 3 dB; half-to-half 3.86 rev/s > 1.0 |
| DroneAudioSet | drone2_high_25cm_M_up_File6 | 127.14 | 1.92 | harmonic-sum margin 1.92 dB < 3 dB; half-to-half 22.80 rev/s > 1.0 |
| DroneAudioSet | drone2_high_25cm_M_up_silence | 125.76 | 0.22 | harmonic-sum margin 0.22 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_center_File3 | 25.36 | 0.74 | harmonic-sum margin 0.74 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_center_File4 | 122.42 | 3.42 | half-to-half 9.03 rev/s > 1.0 |
| DroneAudioSet | drone2_high_50cm_M_center_File5 | 104.32 | 2.26 | harmonic-sum margin 2.26 dB < 3 dB; half-to-half 78.49 rev/s > 1.0 |
| DroneAudioSet | drone2_high_50cm_M_center_File6 | 50.78 | 0.48 | harmonic-sum margin 0.48 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_center_silence | 50.27 | 0.56 | harmonic-sum margin 0.56 dB < 3 dB; half-to-half 75.98 rev/s > 1.0 |
| DroneAudioSet | drone2_high_50cm_M_down_File1 | 135.66 | 2.27 | harmonic-sum margin 2.27 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_down_File2 | 103.99 | 2.27 | harmonic-sum margin 2.27 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_down_File3 | 101.36 | 0.05 | harmonic-sum margin 0.05 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_down_File4 | 122.36 | 1.77 | harmonic-sum margin 1.78 dB < 3 dB; half-to-half 2.49 rev/s > 1.0 |
| DroneAudioSet | drone2_high_50cm_M_down_File5 | 103.78 | 1.66 | harmonic-sum margin 1.65 dB < 3 dB; half-to-half 25.54 rev/s > 1.0 |
| DroneAudioSet | drone2_high_50cm_M_down_File6 | 101.32 | -0.55 | harmonic-sum margin -0.55 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_down_silence | 100.38 | 0.14 | harmonic-sum margin 0.14 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_up_File3 | 25.33 | 1.56 | harmonic-sum margin 1.56 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_up_File4 | 122.22 | 2.79 | harmonic-sum margin 2.79 dB < 3 dB; half-to-half 9.40 rev/s > 1.0 |
| DroneAudioSet | drone2_high_50cm_M_up_File5 | 104.14 | 2.43 | harmonic-sum margin 2.43 dB < 3 dB; half-to-half 78.40 rev/s > 1.0 |
| DroneAudioSet | drone2_high_50cm_M_up_File6 | 50.68 | 0.02 | harmonic-sum margin 0.02 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_up_silence | 100.45 | 0.96 | harmonic-sum margin 0.96 dB < 3 dB |
| DroneAudioSet | drone2_low_25cm_M_center_File1 | 97.98 | 0.54 | harmonic-sum margin 0.54 dB < 3 dB; half-to-half 4.17 rev/s > 1.0 |
| DroneAudioSet | drone2_low_25cm_M_center_File4 | 109.60 | 0.04 | harmonic-sum margin 0.04 dB < 3 dB; half-to-half 23.98 rev/s > 1.0 |
| DroneAudioSet | drone2_low_25cm_M_down_File1 | 93.98 | 0.78 | harmonic-sum margin 0.78 dB < 3 dB; half-to-half 1.35 rev/s > 1.0 |
| DroneAudioSet | drone2_low_25cm_M_down_File4 | 82.66 | 0.42 | harmonic-sum margin 0.42 dB < 3 dB; half-to-half 66.54 rev/s > 1.0 |
| DroneAudioSet | drone2_low_25cm_M_up_File1 | 97.79 | 0.14 | harmonic-sum margin 0.14 dB < 3 dB; half-to-half 1.16 rev/s > 1.0 |
| DroneAudioSet | drone2_low_25cm_M_up_File4 | 27.30 | 1.38 | harmonic-sum margin 1.38 dB < 3 dB |
| DroneAudioSet | drone2_low_50cm_M_center_File1 | 91.24 | 0.54 | harmonic-sum margin 0.54 dB < 3 dB; half-to-half 30.68 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_center_File4 | 147.72 | -0.03 | harmonic-sum margin -0.03 dB < 3 dB; half-to-half 42.73 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_center_File5 | 102.97 | 1.46 | harmonic-sum margin 1.46 dB < 3 dB; half-to-half 2.68 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_center_File6 | 106.06 | 1.95 | harmonic-sum margin 1.95 dB < 3 dB |
| DroneAudioSet | drone2_low_50cm_M_center_silence | 104.60 | 1.80 | harmonic-sum margin 1.80 dB < 3 dB |
| DroneAudioSet | drone2_low_50cm_M_down_File1 | 104.30 | 0.21 | harmonic-sum margin 0.21 dB < 3 dB; half-to-half 47.24 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_down_File4 | 136.24 | -0.30 | harmonic-sum margin -0.30 dB < 3 dB; half-to-half 59.14 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_down_File5 | 88.34 | -0.20 | harmonic-sum margin -0.20 dB < 3 dB; half-to-half 4.36 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_down_File6 | 105.60 | 2.93 | harmonic-sum margin 2.93 dB < 3 dB |
| DroneAudioSet | drone2_low_50cm_M_up_File1 | 15.22 | 0.06 | harmonic-sum margin 0.06 dB < 3 dB; half-to-half 129.07 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_up_File4 | 17.11 | 0.29 | harmonic-sum margin 0.29 dB < 3 dB; half-to-half 92.26 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_up_File5 | 102.76 | 0.51 | harmonic-sum margin 0.51 dB < 3 dB; half-to-half 55.33 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_up_File6 | 105.52 | 1.34 | harmonic-sum margin 1.34 dB < 3 dB |
| DroneAudioSet | drone2_low_50cm_M_up_silence | 104.62 | 2.30 | harmonic-sum margin 2.30 dB < 3 dB |
| KAIST-rotating-acoustic | 0Nm_BPFI_03 | 107.25 | -6.24 | harmonic-sum margin -6.24 dB < 3 dB |
| KAIST-rotating-acoustic | 0Nm_BPFI_10 | 107.20 | -7.75 | harmonic-sum margin -7.75 dB < 3 dB |
| KAIST-rotating-acoustic | 0Nm_BPFO_03 | 22.83 | 8.07 | half-to-half 98.56 rev/s > 1.0 |
| KAIST-rotating-acoustic | 0Nm_BPFO_10 | 60.94 | -0.10 | harmonic-sum margin -0.10 dB < 3 dB |
| KAIST-rotating-acoustic | 0Nm_Normal | 120.84 | -2.74 | harmonic-sum margin -2.74 dB < 3 dB; half-to-half 1.80 rev/s > 1.0 |
| SPCUP19-ChuMS-bench | 1prop_repeat1 | 100.48 | 0.48 | harmonic-sum margin 0.48 dB < 3 dB; half-to-half 12.38 rev/s > 1.0 |
| SPCUP19-ChuMS-bench | 1prop_repeat2 | 25.74 | -1.26 | harmonic-sum margin -1.26 dB < 3 dB; half-to-half 69.40 rev/s > 1.0 |
| SPCUP19-ChuMS-bench | 1prop_repeat3 | 99.28 | 0.24 | harmonic-sum margin 0.24 dB < 3 dB |
| SPCUP19-ChuMS-bench | 2prop_repeat1 | 84.38 | 0.42 | harmonic-sum margin 0.42 dB < 3 dB |
| SPCUP19-ChuMS-bench | 2prop_repeat2 | 83.72 | 0.99 | harmonic-sum margin 0.99 dB < 3 dB |
| SPCUP19-ChuMS-bench | 2prop_repeat3 | 82.86 | 2.06 | harmonic-sum margin 2.06 dB < 3 dB |
| SPCUP19-ChuMS-bench | 3prop_repeat1 | 76.36 | 0.70 | harmonic-sum margin 0.71 dB < 3 dB; half-to-half 1.40 rev/s > 1.0 |
| SPCUP19-ChuMS-bench | 3prop_repeat2 | 76.68 | 1.88 | harmonic-sum margin 1.88 dB < 3 dB; half-to-half 1.22 rev/s > 1.0 |
| SPCUP19-ChuMS-bench | 3prop_repeat3 | 78.22 | 0.96 | harmonic-sum margin 0.96 dB < 3 dB |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__4 | 77.66 | 1.62 | harmonic-sum margin 1.62 dB < 3 dB |
| SPCUP19-egonoise | AGH__static_clean__0 | 50.35 | -0.21 | harmonic-sum margin -0.21 dB < 3 dB |
| SPCUP19-egonoise | AGH__static_clean__1 | 142.52 | -0.07 | harmonic-sum margin -0.07 dB < 3 dB |
| SPCUP19-egonoise | AGH__static_clean__2 | 113.22 | -0.35 | harmonic-sum margin -0.35 dB < 3 dB; half-to-half 81.96 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_clean__3 | 25.09 | -0.03 | harmonic-sum margin -0.03 dB < 3 dB; half-to-half 71.45 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_clean__4 | 50.44 | -0.00 | harmonic-sum margin -0.00 dB < 3 dB |
| SPCUP19-egonoise | AGH__static_clean__5 | 50.54 | 0.10 | harmonic-sum margin 0.10 dB < 3 dB |
| SPCUP19-egonoise | AGH__static_clean__6 | 50.36 | -0.00 | harmonic-sum margin -0.00 dB < 3 dB; half-to-half 92.88 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_clean__7 | 149.88 | -0.12 | harmonic-sum margin -0.12 dB < 3 dB; half-to-half 100.13 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_clean__8 | 142.63 | 0.01 | harmonic-sum margin 0.01 dB < 3 dB; half-to-half 5.69 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_clean__9 | 135.88 | -0.28 | window 7.00 s < 8 s; harmonic-sum margin -0.28 dB < 3 dB; window too short for a two-half stability test |
| SPCUP19-egonoise | AGH__static_corrupted__0 | 96.74 | -0.58 | harmonic-sum margin -0.58 dB < 3 dB; half-to-half 14.54 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_corrupted__1 | 111.30 | 1.44 | harmonic-sum margin 1.44 dB < 3 dB |
| SPCUP19-egonoise | Diagonal_Unloading__recordings__static__static_10m | 98.64 | 0.00 | harmonic-sum margin 0.00 dB < 3 dB; half-to-half 7.65 rev/s > 1.0 |
| SPCUP19-egonoise | Diagonal_Unloading__recordings__static__static_2m | 100.01 | 2.00 | harmonic-sum margin 2.00 dB < 3 dB |
| SPCUP19-egonoise | Idea_ssu__stationary_1 | 92.18 | 1.73 | harmonic-sum margin 1.73 dB < 3 dB |
| SPCUP19-egonoise | Idea_ssu__stationary_2 | 96.78 | 0.41 | harmonic-sum margin 0.41 dB < 3 dB; half-to-half 1.52 rev/s > 1.0 |
| SPCUP19-egonoise | Shout_COOEE__SPCUP19_Shout_COOEE_StaticSubmission1 | 127.30 | -0.07 | harmonic-sum margin -0.07 dB < 3 dB |
| SPCUP19-egonoise | Shout_COOEE__SPCUP19_Shout_COOEE_StaticSubmission2 | 127.06 | -0.22 | harmonic-sum margin -0.22 dB < 3 dB; half-to-half 4.20 rev/s > 1.0 |
| drone_audio | B_S2_D1_067-bebop | 143.68 | -0.04 | window 1.02 s < 8 s; harmonic-sum margin -0.03 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_068-bebop | 144.74 | 0.21 | window 1.02 s < 8 s; harmonic-sum margin 0.21 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_069-bebop | 119.28 | -0.29 | window 1.02 s < 8 s; harmonic-sum margin -0.29 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_070-bebop | 76.56 | -0.88 | window 1.02 s < 8 s; harmonic-sum margin -0.88 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_071-bebop | 134.16 | 0.08 | window 1.02 s < 8 s; harmonic-sum margin 0.08 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_072-bebop | 147.92 | 3.28 | window 1.02 s < 8 s; window too short for a two-half stability test |
| drone_audio | B_S2_D1_073-bebop | 137.62 | 1.55 | window 1.02 s < 8 s; harmonic-sum margin 1.55 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_074-bebop | 145.58 | 1.66 | window 1.02 s < 8 s; harmonic-sum margin 1.66 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_075-bebop | 147.72 | 1.80 | window 1.02 s < 8 s; harmonic-sum margin 1.80 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_076-bebop | 131.30 | -0.64 | window 1.02 s < 8 s; harmonic-sum margin -0.64 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_077-bebop | 147.16 | 1.36 | window 1.02 s < 8 s; harmonic-sum margin 1.36 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_078-bebop | 74.25 | 0.22 | window 1.02 s < 8 s; harmonic-sum margin 0.22 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_079-bebop | 43.94 | 0.32 | window 1.02 s < 8 s; harmonic-sum margin 0.32 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_080-bebop | 137.58 | -0.24 | window 1.02 s < 8 s; harmonic-sum margin -0.24 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_081-bebop | 134.96 | 1.68 | window 1.02 s < 8 s; harmonic-sum margin 1.68 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_082-bebop | 140.84 | -0.38 | window 1.02 s < 8 s; harmonic-sum margin -0.38 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_083-bebop | 146.52 | -0.28 | window 1.02 s < 8 s; harmonic-sum margin -0.28 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_084-bebop | 149.08 | 1.00 | window 1.02 s < 8 s; harmonic-sum margin 1.00 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_085-bebop | 122.74 | -1.05 | window 1.02 s < 8 s; harmonic-sum margin -1.05 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_086-bebop | 132.47 | -0.07 | window 1.02 s < 8 s; harmonic-sum margin -0.07 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_087-bebop | 132.90 | 0.57 | window 1.02 s < 8 s; harmonic-sum margin 0.57 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_088-bebop | 142.86 | 0.81 | window 1.02 s < 8 s; harmonic-sum margin 0.81 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_089-bebop | 147.10 | 1.98 | window 1.02 s < 8 s; harmonic-sum margin 1.98 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_090-bebop | 149.76 | -0.96 | window 1.02 s < 8 s; harmonic-sum margin -0.96 dB < 3 dB; window too short for a two-half stability test |
| zenodo_drone_noises | n116 | 39.48 | 2.31 | harmonic-sum margin 2.31 dB < 3 dB |
| zenodo_drone_noises | n118 | 111.18 | 2.40 | harmonic-sum margin 2.39 dB < 3 dB; half-to-half 1.08 rev/s > 1.0 |
| zenodo_drone_noises | n119 | 93.36 | 0.45 | harmonic-sum margin 0.45 dB < 3 dB |
| zenodo_drone_noises | n120 | 88.48 | 0.86 | harmonic-sum margin 0.86 dB < 3 dB; half-to-half 40.49 rev/s > 1.0 |
| zenodo_drone_noises | n121 | 105.69 | 1.80 | harmonic-sum margin 1.80 dB < 3 dB; half-to-half 58.29 rev/s > 1.0 |
| zenodo_drone_noises | n122 | 144.86 | 0.13 | harmonic-sum margin 0.13 dB < 3 dB |
