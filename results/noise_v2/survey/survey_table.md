## Corpora

| Corpus | Location | Type | Rigs | Telemetry | Speed source | fs / ch / duration | Licence | Verdict | Reason |
|---|---|---|---|---|---|---|---|---|---|
| DREGON single-motor bench | dload:DREGON-frames (recordings motor_Motor{1-4}_{50..90}, motor_allMotors_70) | static single-motor bench; motor clamped, one throttle per recording | 1 airframe (MikroKopter), 4 individually driven motors + 1 all-motors run | none logged; the THROTTLE SETPOINT is in the recording name | estimated (Welch harmonic sum + odd-harmonic octave check, tol from the two-half spread) — cross-checked against the validated throttle law rate = 0.975*throttle + 0.37 rev/s | 8 ch, 44.1 kHz, 25-45 s files with 11-35 s of motor-on | research use (INRIA DREGON) | USABLE (20/21 recordings) | one throttle per recording, one rotor per recording, and a non-acoustic reference (the throttle law) to score the estimator against |
| SPCUP19 AGH single rotors | dload:SPCUP19-egonoise (AGH/ego-noise/single rotors/0..7) | static single-rotor takes, one mic | 1 (AGH quadrotor, model unspecified) | none | estimated (same estimator); cross-checked against the accepted blind-campaign readings (blind-corpus-annotation.md) | 1 ch, 44.1 kHz, 16.4-17.6 s | free for personal, educational and academic use only | USABLE (7/8 recordings) | single-source static takes; the blind campaign already accepted all 8 under the one-source rule, so the readings are independently checkable |
| SPCUP19 static / hover (quadrotor) | dload:SPCUP19-egonoise (AGH static clean/corrupted, Diagonal_Unloading static, Idea_ssu stationary, Shout_COOEE StaticSubmission1/2) | four-rotor static / hover windows on 4 team rigs | 4 (AGH quadrotor, DJI Phantom 4 PRO, DJI Phantom 4 GL300C, Intel Aero RTF) | none | none accepted (estimator margins below the 3 dB rule) | 1-8 ch, 44.1/48 kHz, 11.5-85.9 s | free for personal, educational and academic use only | REJECTED (0/18 recordings) | four combs inside a few rev/s make every rival comb score almost as well as the accepted one, so no window clears the 3 dB margin — the same geometry that flattened the blind campaign's ridge clearance (no SPCUP static window cleared its off-comb null there either) |
| SPCUP19 ChuMS propeller rig | dregon.inria.fr SPCUP19_ChuMS_data.zip -> UAV_rotor_recordings.mat (TestResults.Test(1..9)) | STATIC PROPELLER RIG: 1, 2 or 3 propellers running, 3 repeats each, 8 calibrated mics on a 1 m arc (MicPositions in the .mat) | 1 rig (Skylark M4-680 / Dotterel) in 3 propeller-count conditions | none; the .mat records the propeller COUNT and per-mic OASPL, no RPM | estimated (same estimator) | 8 ch, 44.1 kHz, 73-75 s, calibrated in Pa | free for personal, educational and academic use only | PARTLY USABLE (1/9 recordings) | a genuine bench rig that the published SPCUP19-egonoise frames expose only as 216 unlabelled arrays (Freq/SPL/RawTruncatedCalibrated per mic); the recordings are stationary but the rig runs several propellers at uncontrolled speeds, so only the runs that clear the 3 dB margin are kept |
| DroneAudioSet (drone-only) | HuggingFace ahlab-drone-project/DroneAudioSet, subset drone-only/ (28 parquet shards, 3.1 GiB, 168 recordings); published as dload:DroneAudioSet (88.4 GiB, all subsets) | rig-mounted static: the quadcopter is bolted to an aluminium frame at 1.5 m and run at a fixed throttle (the paper's hover emulation) | 2 (DJI F450 'D_large', 450 mm wheelbase, 9.4x5.0 props; DJI F330 'D_small', 330 mm, 8x4.5 props) x 2 throttles (low/high) x mic distance 25/50 cm | none; throttle is a two-level label. The paper states spectral lines 168/235 Hz (D_large low/high) and 156/259 Hz (D_small low/high) | estimated (same estimator); cross-checked against the paper's stated lines via the measured blade-pass frequency 2*f | 8 ch (M_up / M_down arrays) or 1 ch (M_center Soundskrit), 16 kHz, 30-152 s | MIT | PARTLY USABLE (50/168 recordings) | the largest static multichannel drone corpus and the only one with two airframes; the recordings are stationary but the four rotors are not near-equal on every mic, so the reading is accepted per recording |
| AVQ | dload:AVQ / AVQ-egonoise | onboard 8-ch array, FREE FLIGHT (2 sessions, 12 sequences) | 1 quadrotor | none (blind VK pseudo-labels only, dload:AVQ-egonoise-vkrps) | none | 8 ch, 44.1 kHz, 12 sequences (705 s of pure ego-noise) | free for academic/research use (courtesy of Lin Wang, QMUL) | REJECTED | free flight, so no recording has one speed per rotor to fit to, and the corpus is the hardest case on record: 175 of 187 blind windows are octave-suspect and its median fvk_ratio_double 1.044 is below the calibrated 1.065 cut (blind-corpus-annotation.md). Not a bench/static candidate by type |
| drone_audio (Al-Emadi IWCMC 2019) | data/drone_audio, dload:drone_audio (Binary_Drone_Audio/yes_drone, 1332 clips; Multiclass bebop_1/membo_1) | indoor propeller recordings of a Parrot Bebop and a Parrot Mambo, cut into 1 s clips; the sibling unknown/ class is ESC-50 + white noise + silence | 2 (Bebop, Mambo) | none | none accepted | 1 ch, 16 kHz, 0.65-1.02 s per clip | none stated (citation request only, IWCMC 2019 paper) | REJECTED (0/24 recordings) | the clips are 1.02 s, an eighth of the 8 s stationary window the tolerance rule needs, and there is no take-level grouping that would let clips be re-joined; the sampled clips also fail the margin rule outright |
| zenodo_drone_noises | data/zenodo_drone_noises (all_drone_noises.zip -> noises-train-drones/n116..n120, noises-test-drones/n121..n122) | 7 unlabelled drone-noise clips; no rig, session or setup recorded anywhere in the zip (no README, no metadata file) | unknown (one unknown rig per file at best) | none | none accepted for a rig; per-file estimates reported | 1 ch, 8 kHz (n121/n122) or 44.1 kHz, 40-215 s | none stated | REJECTED (1/7 recordings) | no rig identity, so a per-rig noise parameter cannot be attached to the point even where the estimator reads a speed; the recordings also drift (flight, not bench) and mostly fail the margin rule |
| KAIST-rotating-acoustic (control) | dload:KAIST-rotating-acoustic | industrial rotating-machine testbed, mono bench, dataset-stated 3010 RPM | 1 (not a drone) | the stated nominal 3010 RPM = 50.167 rev/s | paper-stated nominal; the estimator is scored against it | 1 ch, 51.2 kHz, 60 s, 5 recordings (fault + severity per file) | CC BY 4.0 | CONTROL ONLY (0/5 recordings) | kept as the out-of-domain control the blind campaign used: bearing fault lines are real, strong and NOT octaves of the shaft, so this corpus measures whether the gate refuses what it cannot read. Not a drone fit point |
| DronePrint (Kolamunna et al. 2021) | OSF repository via github.com/DronePrint/DronePrint (not in this repo) | far-field FREE FLIGHT: 5 drone classes recorded with a RODE NTG4 shotgun mic at ~20 m altitude within a 50 m radius, plus YouTube-scraped clips | 5 recorded (Bebop 2, Mavic Pro, Phantom 4 Pro, Spark, Matrice 100) + 15 online classes | none | none | 1 ch, 44.1 kHz | open (OSF), attribution | REJECTED (not downloaded) | free flight at 20 m with a directional ground mic: the comb decoheres, the speed is unknown and the manoeuvre is unconstrained — class G of the blind-corpus plan (mono far field, refused by default) |
| MAVD | two unrelated datasets answer to this name: MAVD (Mandarin audio-visual with depth, github.com/SpringHuo/MAVD) and MAVD-Traffic (Montevideo audio-visual traffic) | neither is drone ego-noise: speech+depth corpus / street-traffic corpus | none | n/a | n/a | n/a | MAVD: access by e-mail request to the authors | REJECTED (not a drone-noise corpus) | the name does not resolve to a rotor-noise dataset; the speech variant is also gated behind an e-mail request, which the survey rules out |
| DroneNoise Database (Salford) | salford.figshare.com/articles/dataset/DroneNoise_Database/22133411 (figshare article 22133411, 175 files, 742 MB) | field OVERFLIGHT campaign (Edzell, Scotland, 2022-08-17): sUAS flying over a ground microphone array, 3 events per configuration | several sUAS types (file stems Ed_3p / Ed_Fp / Ed_M3 / Ed_Yn) | none published with the audio (no RPM channel in the file set) | none | ground mics M1..M?, ~45-68 s WAV per event | CC BY 4.0 | REJECTED (metadata only, not downloaded) | overflight recordings are Doppler-shifted and non-stationary by construction, and no rotor speed is published, so no fit point can be built from them |
| ESC-50 | github.com/karolpiczak/ESC-50 (already inside data/drone_audio as the unknown/ negatives) | 2000 five-second environmental clips, 50 classes; no drone class (the closest are helicopter and chainsaw) | none | none | none | 1 ch, 44.1 kHz, 5 s | CC BY-NC 3.0 | REJECTED | not a drone corpus at all; it is the negative-class pool of drone_audio and carries no rotating-source speed |

## Fit points (one row per usable recording)

| Corpus | Recording | Rig | Condition | Speed(s) rev/s | Tol rev/s | Spread rev/s | Rotors seen | Octave | Margin dB | Half Δ rev/s | Window s | ch | fs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DREGON-bench | motor_Motor1_50 | dregon_mikrokopter_single_motor | bench Motor1 throttle 50% | 49.00 | 0.25 | 0.00 | 1 | as_found | 4.74 | 0.06 | 16.00 | 8 | 44100 |
| DREGON-bench | motor_Motor1_60 | dregon_mikrokopter_single_motor | bench Motor1 throttle 60% | 58.69 | 0.25 | 0.00 | 1 | halved | 5.36 | 0.01 | 16.00 | 8 | 44100 |
| DREGON-bench | motor_Motor1_70 | dregon_mikrokopter_single_motor | bench Motor1 throttle 70% | 68.30 | 0.25 | 0.00 | 1 | halved | 9.75 | 0.08 | 16.00 | 8 | 44100 |
| DREGON-bench | motor_Motor1_80 | dregon_mikrokopter_single_motor | bench Motor1 throttle 80% | 78.06 | 0.25 | 0.00 | 1 | as_found | 8.40 | 0.06 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor1_90 | dregon_mikrokopter_single_motor | bench Motor1 throttle 90% | 88.18 | 0.35 | 0.00 | 1 | as_found | 10.01 | 0.35 | 13.00 | 8 | 44100 |
| DREGON-bench | motor_Motor2_50 | dregon_mikrokopter_single_motor | bench Motor2 throttle 50% | 48.36 | 0.25 | 0.00 | 1 | as_found | 4.03 | 0.04 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor2_60 | dregon_mikrokopter_single_motor | bench Motor2 throttle 60% | 58.07 | 0.31 | 0.00 | 1 | halved | 5.52 | 0.31 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor2_70 | dregon_mikrokopter_single_motor | bench Motor2 throttle 70% | 67.52 | 0.25 | 0.00 | 1 | as_found | 5.56 | 0.00 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor2_80 | dregon_mikrokopter_single_motor | bench Motor2 throttle 80% | 77.18 | 0.28 | 0.00 | 1 | as_found | 7.17 | 0.28 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor2_90 | dregon_mikrokopter_single_motor | bench Motor2 throttle 90% | 86.70 | 0.25 | 0.00 | 1 | as_found | 9.76 | 0.10 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor3_50 | dregon_mikrokopter_single_motor | bench Motor3 throttle 50% | 49.06 | 0.28 | 0.00 | 1 | as_found | 5.14 | 0.28 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor3_60 | dregon_mikrokopter_single_motor | bench Motor3 throttle 60% | 58.90 | 0.25 | 0.00 | 1 | as_found | 5.28 | 0.01 | 13.00 | 8 | 44100 |
| DREGON-bench | motor_Motor3_70 | dregon_mikrokopter_single_motor | bench Motor3 throttle 70% | 68.56 | 0.25 | 0.00 | 1 | halved | 8.20 | 0.03 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor3_80 | dregon_mikrokopter_single_motor | bench Motor3 throttle 80% | 78.82 | 0.25 | 0.00 | 1 | as_found | 8.88 | 0.00 | 13.00 | 8 | 44100 |
| DREGON-bench | motor_Motor3_90 | dregon_mikrokopter_single_motor | bench Motor3 throttle 90% | 88.24 | 0.26 | 0.00 | 1 | as_found | 11.45 | 0.26 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor4_50 | dregon_mikrokopter_single_motor | bench Motor4 throttle 50% | 49.74 | 0.25 | 0.00 | 1 | as_found | 4.59 | 0.02 | 13.00 | 8 | 44100 |
| DREGON-bench | motor_Motor4_60 | dregon_mikrokopter_single_motor | bench Motor4 throttle 60% | 59.66 | 0.33 | 0.00 | 1 | halved | 5.11 | 0.33 | 13.00 | 8 | 44100 |
| DREGON-bench | motor_Motor4_70 | dregon_mikrokopter_single_motor | bench Motor4 throttle 70% | 69.31 | 0.25 | 0.00 | 1 | halved | 7.27 | 0.03 | 13.00 | 8 | 44100 |
| DREGON-bench | motor_Motor4_80 | dregon_mikrokopter_single_motor | bench Motor4 throttle 80% | 79.38 | 0.25 | 0.00 | 1 | as_found | 9.25 | 0.02 | 12.00 | 8 | 44100 |
| DREGON-bench | motor_Motor4_90 | dregon_mikrokopter_single_motor | bench Motor4 throttle 90% | 89.14 | 0.25 | 0.00 | 1 | as_found | 9.94 | 0.02 | 12.00 | 8 | 44100 |
| DroneAudioSet | drone1_high_25cm_M_center_File3 | daset_drone1 | rig-mounted static, throttle high, mic M_center at 25cm | 116.70, 116.88, 117.19, 117.43 | 0.67 | 0.73 | 4 | as_found | 5.50 | 0.67 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone1_high_25cm_M_up_File3 | daset_drone1 | rig-mounted static, throttle high, mic M_up at 25cm | 116.82, 117.00, 117.25, 117.55 | 0.64 | 0.73 | 4 | as_found | 4.75 | 0.64 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_center_File2 | daset_drone1 | rig-mounted static, throttle high, mic M_center at 50cm | 119.63, 120.24, 120.91, 121.09 | 0.58 | 1.46 | 4 | as_found | 5.06 | 0.58 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_center_File3 | daset_drone1 | rig-mounted static, throttle high, mic M_center at 50cm | 59.14, 59.27, 59.42, 59.60 | 0.25 | 0.46 | 4 | halved | 5.68 | 0.04 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_center_File5 | daset_drone1 | rig-mounted static, throttle high, mic M_center at 50cm | 120.79, 121.34, 121.64, 122.01 | 0.42 | 1.22 | 4 | as_found | 5.61 | 0.42 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_center_File6 | daset_drone1 | rig-mounted static, throttle high, mic M_center at 50cm | 118.84, 119.02, 119.51, 119.75 | 0.25 | 0.92 | 4 | as_found | 5.49 | 0.02 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_center_silence | daset_drone1 | rig-mounted static, throttle high, mic M_center at 50cm | 117.74, 118.23, 118.65, 119.08 | 0.80 | 1.34 | 4 | as_found | 5.59 | 0.80 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_up_File3 | daset_drone1 | rig-mounted static, throttle high, mic M_up at 50cm | 59.08, 59.17, 59.33, 59.42 | 0.25 | 0.34 | 4 | halved | 5.90 | 0.04 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_up_File5 | daset_drone1 | rig-mounted static, throttle high, mic M_up at 50cm | 120.79, 121.09, 121.28, 121.52 | 0.25 | 0.73 | 4 | as_found | 4.35 | 0.10 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_up_File6 | daset_drone1 | rig-mounted static, throttle high, mic M_up at 50cm | 118.35, 118.65, 118.96, 119.57 | 0.25 | 1.22 | 4 | as_found | 4.68 | 0.14 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone1_high_50cm_M_up_silence | daset_drone1 | rig-mounted static, throttle high, mic M_up at 50cm | 117.92, 118.10, 118.47, 118.65 | 0.84 | 0.73 | 4 | as_found | 5.34 | 0.84 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone1_low_25cm_M_center_File3 | daset_drone1 | rig-mounted static, throttle low, mic M_center at 25cm | 82.46, 82.70, 84.17, 84.41 | 0.25 | 1.95 | 4 | as_found | 4.02 | 0.04 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_center_File1 | daset_drone2 | rig-mounted static, throttle high, mic M_center at 25cm | 133.67, 134.52, 134.70, 135.07 | 0.78 | 1.40 | 4 | as_found | 4.54 | 0.78 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_center_File2 | daset_drone2 | rig-mounted static, throttle high, mic M_center at 25cm | 65.02, 65.14, 65.39, 65.59 | 0.25 | 0.57 | 4 | halved | 5.80 | 0.02 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_center_File5 | daset_drone2 | rig-mounted static, throttle high, mic M_center at 25cm | 128.95, 129.31, 129.56, 129.72 | 0.25 | 0.77 | 4 | as_found | 6.63 | 0.06 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_down_File1 | daset_drone2 | rig-mounted static, throttle high, mic M_down at 25cm | 134.03, 134.24, 134.48, 134.64 | 0.98 | 0.61 | 4 | as_found | 5.00 | 0.98 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_down_File2 | daset_drone2 | rig-mounted static, throttle high, mic M_down at 25cm | 129.96, 130.94 | 0.25 | 0.98 | 2 | as_found | 7.09 | 0.12 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_down_File3 | daset_drone2 | rig-mounted static, throttle high, mic M_down at 25cm | 128.95, 129.35, 129.64 | 0.30 | 0.69 | 3 | as_found | 8.20 | 0.30 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_down_File5 | daset_drone2 | rig-mounted static, throttle high, mic M_down at 25cm | 129.09, 129.27, 129.46, 129.76 | 0.42 | 0.67 | 4 | as_found | 6.01 | 0.42 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_up_File1 | daset_drone2 | rig-mounted static, throttle high, mic M_up at 25cm | 134.40, 134.58, 134.77, 135.01 | 1.00 | 0.61 | 4 | as_found | 4.78 | 1.00 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_up_File2 | daset_drone2 | rig-mounted static, throttle high, mic M_up at 25cm | 65.12, 65.25, 65.46, 68.60 | 0.25 | 3.48 | 4 | halved | 6.63 | 0.06 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_up_File3 | daset_drone2 | rig-mounted static, throttle high, mic M_up at 25cm | 128.99, 129.23, 129.39, 129.60 | 0.25 | 0.61 | 4 | as_found | 9.81 | 0.08 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_25cm_M_up_File5 | daset_drone2 | rig-mounted static, throttle high, mic M_up at 25cm | 128.85, 129.03, 129.21, 129.46 | 0.40 | 0.61 | 4 | as_found | 6.75 | 0.40 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_50cm_M_center_File1 | daset_drone2 | rig-mounted static, throttle high, mic M_center at 50cm | 135.56, 135.74, 136.23, 136.84 | 0.40 | 1.28 | 4 | as_found | 8.95 | 0.40 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone2_high_50cm_M_center_File2 | daset_drone2 | rig-mounted static, throttle high, mic M_center at 50cm | 103.39, 103.94, 104.13, 104.68 | 0.25 | 1.28 | 4 | as_found | 4.60 | 0.12 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone2_high_50cm_M_center_File5 | daset_drone2 | rig-mounted static, throttle high, mic M_center at 50cm | 103.76, 104.25, 104.68, 104.92 | 0.25 | 1.16 | 4 | as_found | 3.59 | 0.12 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone2_high_50cm_M_down_File1 | daset_drone2 | rig-mounted static, throttle high, mic M_down at 50cm | 135.99, 136.41 | 0.36 | 0.43 | 2 | as_found | 5.99 | 0.36 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_50cm_M_down_File2 | daset_drone2 | rig-mounted static, throttle high, mic M_down at 50cm | 103.76, 106.32, 106.61 | 0.56 | 2.85 | 3 | as_found | 3.56 | 0.56 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_50cm_M_up_File1 | daset_drone2 | rig-mounted static, throttle high, mic M_up at 50cm | 135.50, 135.99, 136.54, 136.78 | 0.25 | 1.28 | 4 | as_found | 9.58 | 0.05 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_high_50cm_M_up_File2 | daset_drone2 | rig-mounted static, throttle high, mic M_up at 50cm | 102.97, 103.27, 103.76, 104.19 | 0.25 | 1.22 | 4 | as_found | 4.69 | 0.12 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_center_File2 | daset_drone2 | rig-mounted static, throttle low, mic M_center at 25cm | 105.65, 105.83, 106.38, 107.06 | 0.25 | 1.40 | 4 | as_found | 4.60 | 0.20 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_center_File3 | daset_drone2 | rig-mounted static, throttle low, mic M_center at 25cm | 106.45, 106.81 | 0.46 | 0.37 | 2 | as_found | 4.87 | 0.46 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_center_File5 | daset_drone2 | rig-mounted static, throttle low, mic M_center at 25cm | 54.43, 54.52, 54.86, 55.01 | 0.25 | 0.58 | 4 | halved | 4.92 | 0.04 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_center_File6 | daset_drone2 | rig-mounted static, throttle low, mic M_center at 25cm | 108.34, 108.64, 108.95, 109.13 | 0.49 | 0.79 | 4 | as_found | 5.84 | 0.49 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_center_silence | daset_drone2 | rig-mounted static, throttle low, mic M_center at 25cm | 107.97, 108.58, 108.83, 109.13 | 0.25 | 1.16 | 4 | as_found | 8.52 | 0.05 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_down_File2 | daset_drone2 | rig-mounted static, throttle low, mic M_down at 25cm | 106.75, 106.93, 107.18 | 0.25 | 0.43 | 3 | as_found | 4.78 | 0.00 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_down_File3 | daset_drone2 | rig-mounted static, throttle low, mic M_down at 25cm | 105.88, 106.16, 106.40, 106.61 | 0.48 | 0.73 | 4 | as_found | 5.31 | 0.48 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_down_File5 | daset_drone2 | rig-mounted static, throttle low, mic M_down at 25cm | 108.89, 109.42, 109.58, 109.74 | 0.25 | 0.85 | 4 | as_found | 4.70 | 0.16 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_down_File6 | daset_drone2 | rig-mounted static, throttle low, mic M_down at 25cm | 108.22, 108.46, 108.89, 109.25 | 0.25 | 1.04 | 4 | as_found | 5.71 | 0.14 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_down_silence | daset_drone2 | rig-mounted static, throttle low, mic M_down at 25cm | 107.97, 108.22, 108.40, 108.76 | 0.25 | 0.79 | 4 | as_found | 8.16 | 0.16 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_up_File3 | daset_drone2 | rig-mounted static, throttle low, mic M_up at 25cm | 105.83, 106.63, 106.99 | 0.58 | 1.16 | 3 | as_found | 4.30 | 0.58 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_up_File5 | daset_drone2 | rig-mounted static, throttle low, mic M_up at 25cm | 27.22, 27.34, 27.39, 27.46 | 0.25 | 0.24 | 4 | halved_twice | 6.43 | 0.07 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_up_File6 | daset_drone2 | rig-mounted static, throttle low, mic M_up at 25cm | 107.79, 108.22, 108.89, 109.31 | 0.25 | 1.53 | 4 | as_found | 5.92 | 0.24 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_25cm_M_up_silence | daset_drone2 | rig-mounted static, throttle low, mic M_up at 25cm | 107.67, 108.03, 108.40, 108.70 | 0.27 | 1.04 | 4 | as_found | 8.92 | 0.27 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_50cm_M_center_File3 | daset_drone2 | rig-mounted static, throttle low, mic M_center at 50cm | 99.43, 102.11, 102.48, 102.72 | 0.25 | 3.30 | 4 | as_found | 3.18 | 0.12 | 16.00 | 1 | 16000 |
| DroneAudioSet | drone2_low_50cm_M_down_File2 | daset_drone2 | rig-mounted static, throttle low, mic M_down at 50cm | 103.56, 103.88, 104.09, 107.95 | 0.25 | 4.39 | 4 | as_found | 4.04 | 0.14 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_50cm_M_down_File3 | daset_drone2 | rig-mounted static, throttle low, mic M_down at 50cm | 101.97, 102.21, 102.50 | 0.25 | 0.53 | 3 | as_found | 5.06 | 0.04 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_50cm_M_down_File6 | daset_drone2 | rig-mounted static, throttle low, mic M_down at 50cm | 105.55, 105.75, 106.12, 110.96 | 0.25 | 5.41 | 4 | as_found | 3.06 | 0.10 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_50cm_M_up_File2 | daset_drone2 | rig-mounted static, throttle low, mic M_up at 50cm | 104.19, 104.61, 104.92, 105.29 | 0.25 | 1.10 | 4 | as_found | 3.46 | 0.14 | 16.00 | 8 | 16000 |
| DroneAudioSet | drone2_low_50cm_M_up_File3 | daset_drone2 | rig-mounted static, throttle low, mic M_up at 50cm | 99.30, 102.29, 102.54, 102.84 | 0.25 | 3.54 | 4 | as_found | 3.24 | 0.08 | 16.00 | 8 | 16000 |
| SPCUP19-ChuMS-bench | 3prop_repeat2 | spcup_ChuMS_3prop_bench | 3 propellers. Repeat:2 | 76.92, 77.26, 77.51 | 0.56 | 0.59 | 3 | as_found | 3.26 | 0.56 | 16.00 | 8 | 44100 |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__0 | spcup_AGH | single_rotor | 79.82 | 0.25 | 0.00 | 1 | as_found | 5.47 | 0.16 | 16.00 | 1 | 44100 |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__1 | spcup_AGH | single_rotor | 132.30 | 0.25 | 0.00 | 1 | as_found | 6.10 | 0.10 | 16.00 | 1 | 44100 |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__2 | spcup_AGH | single_rotor | 113.54 | 0.25 | 0.00 | 1 | as_found | 5.99 | 0.00 | 16.00 | 1 | 44100 |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__3 | spcup_AGH | single_rotor | 97.48 | 0.25 | 0.00 | 1 | as_found | 4.52 | 0.02 | 16.00 | 1 | 44100 |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__5 | spcup_AGH | single_rotor | 98.08 | 0.32 | 0.00 | 1 | as_found | 5.18 | 0.32 | 16.00 | 1 | 44100 |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__6 | spcup_AGH | single_rotor | 96.78 | 0.25 | 0.00 | 1 | as_found | 5.02 | 0.02 | 16.00 | 1 | 44100 |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__7 | spcup_AGH | single_rotor | 96.78 | 0.25 | 0.00 | 1 | as_found | 5.71 | 0.00 | 16.00 | 1 | 44100 |
| zenodo_drone_noises | n118 | zenodo_unknown_n118 | unlabelled drone-noise clip (noises-train-drones) | 110.48, 110.74, 110.90, 111.37 | 0.72 | 0.88 | 4 | as_found | 3.38 | 0.72 | 16.00 | 1 | 44100 |

## Rejected recordings

| Corpus | Recording | Reading rev/s | Margin dB | Reason |
|---|---|---|---|---|
| DREGON-bench | motor_allMotors_70 | 68.56 | 0.51 | harmonic-sum margin 0.51 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_center_File1 | 113.12 | 1.12 | harmonic-sum margin 1.12 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_center_File2 | 118.49 | 4.87 | half-to-half 1.48 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_center_File4 | 112.88 | 1.13 | harmonic-sum margin 1.13 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_center_File5 | 106.00 | 0.48 | harmonic-sum margin 0.48 dB < 3 dB; half-to-half 10.66 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_center_File6 | 119.53 | 1.04 | harmonic-sum margin 1.04 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_center_silence | 118.56 | 4.52 | half-to-half 39.38 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_down_File1 | 119.16 | 1.65 | harmonic-sum margin 1.65 dB < 3 dB; half-to-half 3.18 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_down_File2 | 118.76 | 2.43 | harmonic-sum margin 2.43 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_down_File3 | 116.66 | 2.26 | harmonic-sum margin 2.26 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_down_File4 | 112.46 | 1.48 | harmonic-sum margin 1.48 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_down_File5 | 106.06 | 0.20 | harmonic-sum margin 0.20 dB < 3 dB; half-to-half 13.98 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_down_File6 | 108.08 | 0.86 | harmonic-sum margin 0.86 dB < 3 dB; half-to-half 11.32 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_down_silence | 118.71 | 1.84 | harmonic-sum margin 1.84 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_up_File1 | 112.90 | 0.43 | harmonic-sum margin 0.43 dB < 3 dB; half-to-half 7.09 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_up_File2 | 118.76 | 5.40 | half-to-half 1.32 rev/s > 1.0 |
| DroneAudioSet | drone1_high_25cm_M_up_File4 | 112.52 | 1.22 | harmonic-sum margin 1.22 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_up_File5 | 106.24 | -0.06 | harmonic-sum margin -0.06 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_up_File6 | 119.32 | 1.34 | harmonic-sum margin 1.34 dB < 3 dB |
| DroneAudioSet | drone1_high_25cm_M_up_silence | 118.70 | 4.73 | half-to-half 78.75 rev/s > 1.0 |
| DroneAudioSet | drone1_high_50cm_M_center_File1 | 123.64 | 1.84 | harmonic-sum margin 1.84 dB < 3 dB; half-to-half 50.02 rev/s > 1.0 |
| DroneAudioSet | drone1_high_50cm_M_center_File4 | 111.16 | 1.41 | harmonic-sum margin 1.41 dB < 3 dB; half-to-half 15.62 rev/s > 1.0 |
| DroneAudioSet | drone1_high_50cm_M_down_File1 | 117.28 | 0.42 | harmonic-sum margin 0.42 dB < 3 dB; half-to-half 6.14 rev/s > 1.0 |
| DroneAudioSet | drone1_high_50cm_M_down_File2 | 120.06 | 1.69 | harmonic-sum margin 1.69 dB < 3 dB |
| DroneAudioSet | drone1_high_50cm_M_down_File3 | 118.77 | 2.76 | harmonic-sum margin 2.76 dB < 3 dB |
| DroneAudioSet | drone1_high_50cm_M_down_File4 | 132.14 | -0.25 | harmonic-sum margin -0.25 dB < 3 dB; half-to-half 11.68 rev/s > 1.0 |
| DroneAudioSet | drone1_high_50cm_M_down_File5 | 121.62 | 1.45 | harmonic-sum margin 1.45 dB < 3 dB; half-to-half 1.28 rev/s > 1.0 |
| DroneAudioSet | drone1_high_50cm_M_down_File6 | 119.24 | 2.19 | harmonic-sum margin 2.19 dB < 3 dB |
| DroneAudioSet | drone1_high_50cm_M_down_silence | 118.38 | 2.32 | harmonic-sum margin 2.32 dB < 3 dB |
| DroneAudioSet | drone1_high_50cm_M_up_File1 | 123.46 | 1.73 | harmonic-sum margin 1.73 dB < 3 dB; half-to-half 1.42 rev/s > 1.0 |
| DroneAudioSet | drone1_high_50cm_M_up_File2 | 120.02 | 3.74 | half-to-half 1.26 rev/s > 1.0 |
| DroneAudioSet | drone1_high_50cm_M_up_File4 | 110.76 | 1.18 | harmonic-sum margin 1.17 dB < 3 dB; half-to-half 12.94 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_center_File1 | 82.68 | 0.75 | harmonic-sum margin 0.75 dB < 3 dB; half-to-half 3.08 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_center_File2 | 82.76 | 0.54 | harmonic-sum margin 0.54 dB < 3 dB; half-to-half 40.04 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_center_File4 | 126.61 | 0.20 | harmonic-sum margin 0.21 dB < 3 dB; half-to-half 3.29 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_center_File5 | 88.24 | 1.07 | harmonic-sum margin 1.07 dB < 3 dB |
| DroneAudioSet | drone1_low_25cm_M_center_File6 | 66.35 | -0.03 | harmonic-sum margin -0.03 dB < 3 dB |
| DroneAudioSet | drone1_low_25cm_M_center_silence | 25.39 | -2.22 | harmonic-sum margin -2.22 dB < 3 dB; half-to-half 49.61 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_down_File1 | 86.87 | 0.72 | harmonic-sum margin 0.72 dB < 3 dB; half-to-half 1.08 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_down_File2 | 82.68 | -0.37 | harmonic-sum margin -0.37 dB < 3 dB; half-to-half 2.78 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_down_File3 | 82.78 | 1.25 | harmonic-sum margin 1.25 dB < 3 dB |
| DroneAudioSet | drone1_low_25cm_M_down_File4 | 66.10 | -0.14 | harmonic-sum margin -0.14 dB < 3 dB; half-to-half 60.65 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_down_File5 | 88.04 | -0.10 | harmonic-sum margin -0.10 dB < 3 dB; half-to-half 3.29 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_down_File6 | 88.52 | 2.76 | harmonic-sum margin 2.76 dB < 3 dB |
| DroneAudioSet | drone1_low_25cm_M_down_silence | 88.91 | 1.34 | harmonic-sum margin 1.34 dB < 3 dB |
| DroneAudioSet | drone1_low_25cm_M_up_File1 | 82.60 | 0.47 | harmonic-sum margin 0.47 dB < 3 dB; half-to-half 1.04 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_up_File2 | 82.66 | 0.70 | harmonic-sum margin 0.70 dB < 3 dB; half-to-half 2.78 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_up_File3 | 82.54 | 2.47 | harmonic-sum margin 2.47 dB < 3 dB; half-to-half 61.65 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_up_File4 | 21.06 | 0.70 | harmonic-sum margin 0.71 dB < 3 dB; half-to-half 111.36 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_up_File5 | 84.66 | -0.16 | harmonic-sum margin -0.16 dB < 3 dB; half-to-half 3.28 rev/s > 1.0 |
| DroneAudioSet | drone1_low_25cm_M_up_File6 | 22.10 | -0.46 | harmonic-sum margin -0.46 dB < 3 dB |
| DroneAudioSet | drone1_low_25cm_M_up_silence | 25.38 | -2.10 | harmonic-sum margin -2.10 dB < 3 dB; half-to-half 123.71 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_center_File1 | 62.22 | 0.37 | harmonic-sum margin 0.37 dB < 3 dB; half-to-half 4.89 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_center_File2 | 82.84 | 2.11 | harmonic-sum margin 2.11 dB < 3 dB; half-to-half 1.28 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_center_File3 | 92.84 | 2.52 | harmonic-sum margin 2.52 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_center_File4 | 142.91 | 0.91 | harmonic-sum margin 0.91 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_center_File5 | 87.08 | 1.14 | harmonic-sum margin 1.14 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_center_File6 | 84.79 | 0.50 | harmonic-sum margin 0.50 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_center_silence | 84.94 | 0.80 | harmonic-sum margin 0.80 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_down_File1 | 133.18 | 0.30 | harmonic-sum margin 0.30 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_down_File2 | 66.60 | -1.07 | harmonic-sum margin -1.07 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_down_File3 | 92.64 | 2.84 | window 2.00 s < 8 s; harmonic-sum margin 2.84 dB < 3 dB; window too short for a two-half stability test |
| DroneAudioSet | drone1_low_50cm_M_down_File4 | 66.04 | -0.17 | harmonic-sum margin -0.17 dB < 3 dB; half-to-half 8.72 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_down_File5 | 86.74 | 0.02 | harmonic-sum margin 0.02 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_down_File6 | 84.88 | 0.39 | harmonic-sum margin 0.39 dB < 3 dB; half-to-half 4.34 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_down_silence | 84.82 | 0.74 | harmonic-sum margin 0.74 dB < 3 dB; half-to-half 31.88 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_up_File1 | 62.12 | 0.48 | harmonic-sum margin 0.48 dB < 3 dB; half-to-half 4.79 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_up_File2 | 41.41 | -0.38 | harmonic-sum margin -0.38 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_up_File3 | 23.24 | -0.97 | harmonic-sum margin -0.97 dB < 3 dB; half-to-half 69.34 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_up_File4 | 17.80 | -0.33 | harmonic-sum margin -0.33 dB < 3 dB; half-to-half 124.76 rev/s > 1.0 |
| DroneAudioSet | drone1_low_50cm_M_up_File5 | 86.90 | 0.79 | harmonic-sum margin 0.79 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_up_File6 | 89.06 | -0.04 | harmonic-sum margin -0.05 dB < 3 dB |
| DroneAudioSet | drone1_low_50cm_M_up_silence | 85.22 | 2.03 | harmonic-sum margin 2.03 dB < 3 dB |
| DroneAudioSet | drone2_high_25cm_M_center_File3 | 129.80 | 9.21 | half-to-half 96.99 rev/s > 1.0 |
| DroneAudioSet | drone2_high_25cm_M_center_File4 | 127.92 | 0.98 | harmonic-sum margin 0.98 dB < 3 dB; half-to-half 1.90 rev/s > 1.0 |
| DroneAudioSet | drone2_high_25cm_M_center_File6 | 104.10 | 2.21 | harmonic-sum margin 2.22 dB < 3 dB |
| DroneAudioSet | drone2_high_25cm_M_center_silence | 125.94 | -0.23 | harmonic-sum margin -0.23 dB < 3 dB |
| DroneAudioSet | drone2_high_25cm_M_down_File4 | 126.38 | 0.81 | harmonic-sum margin 0.81 dB < 3 dB; half-to-half 1.56 rev/s > 1.0 |
| DroneAudioSet | drone2_high_25cm_M_down_File6 | 103.92 | -0.30 | harmonic-sum margin -0.30 dB < 3 dB |
| DroneAudioSet | drone2_high_25cm_M_down_silence | 125.70 | 0.97 | harmonic-sum margin 0.97 dB < 3 dB; half-to-half 10.45 rev/s > 1.0 |
| DroneAudioSet | drone2_high_25cm_M_up_File4 | 126.42 | 2.14 | harmonic-sum margin 2.14 dB < 3 dB; half-to-half 1.36 rev/s > 1.0 |
| DroneAudioSet | drone2_high_25cm_M_up_File6 | 103.94 | 1.26 | harmonic-sum margin 1.26 dB < 3 dB |
| DroneAudioSet | drone2_high_25cm_M_up_silence | 126.12 | 0.02 | harmonic-sum margin 0.02 dB < 3 dB; half-to-half 22.84 rev/s > 1.0 |
| DroneAudioSet | drone2_high_50cm_M_center_File3 | 25.41 | 2.78 | harmonic-sum margin 2.78 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_center_File4 | 122.30 | 1.08 | harmonic-sum margin 1.08 dB < 3 dB; half-to-half 1.34 rev/s > 1.0 |
| DroneAudioSet | drone2_high_50cm_M_center_File6 | 50.75 | 1.83 | harmonic-sum margin 1.83 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_center_silence | 50.25 | 1.55 | harmonic-sum margin 1.55 dB < 3 dB; half-to-half 75.63 rev/s > 1.0 |
| DroneAudioSet | drone2_high_50cm_M_down_File3 | 101.38 | 1.44 | harmonic-sum margin 1.44 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_down_File4 | 122.32 | 1.70 | harmonic-sum margin 1.70 dB < 3 dB; half-to-half 1.08 rev/s > 1.0 |
| DroneAudioSet | drone2_high_50cm_M_down_File5 | 103.80 | 1.80 | harmonic-sum margin 1.80 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_down_File6 | 101.32 | 0.85 | harmonic-sum margin 0.85 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_down_silence | 100.44 | 1.08 | harmonic-sum margin 1.08 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_up_File3 | 25.33 | 2.33 | harmonic-sum margin 2.33 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_up_File4 | 122.13 | 1.28 | harmonic-sum margin 1.28 dB < 3 dB; half-to-half 1.10 rev/s > 1.0 |
| DroneAudioSet | drone2_high_50cm_M_up_File5 | 26.01 | 2.01 | harmonic-sum margin 2.01 dB < 3 dB; half-to-half 78.22 rev/s > 1.0 |
| DroneAudioSet | drone2_high_50cm_M_up_File6 | 50.63 | 1.98 | harmonic-sum margin 1.98 dB < 3 dB |
| DroneAudioSet | drone2_high_50cm_M_up_silence | 50.17 | 2.34 | harmonic-sum margin 2.34 dB < 3 dB |
| DroneAudioSet | drone2_low_25cm_M_center_File1 | 97.44 | -1.20 | harmonic-sum margin -1.20 dB < 3 dB |
| DroneAudioSet | drone2_low_25cm_M_center_File4 | 82.28 | 2.57 | harmonic-sum margin 2.57 dB < 3 dB |
| DroneAudioSet | drone2_low_25cm_M_down_File1 | 93.24 | 1.53 | harmonic-sum margin 1.53 dB < 3 dB |
| DroneAudioSet | drone2_low_25cm_M_down_File4 | 81.70 | 1.23 | harmonic-sum margin 1.23 dB < 3 dB; half-to-half 25.08 rev/s > 1.0 |
| DroneAudioSet | drone2_low_25cm_M_up_File1 | 17.11 | -1.81 | harmonic-sum margin -1.81 dB < 3 dB |
| DroneAudioSet | drone2_low_25cm_M_up_File2 | 26.69 | 3.57 | half-to-half 80.23 rev/s > 1.0 |
| DroneAudioSet | drone2_low_25cm_M_up_File4 | 27.31 | 2.34 | harmonic-sum margin 2.34 dB < 3 dB |
| DroneAudioSet | drone2_low_50cm_M_center_File1 | 150.01 | -0.04 | harmonic-sum margin -0.04 dB < 3 dB; half-to-half 67.03 rev/s > 1.0; mapped shaft rate 150.01 rev/s outside 15-150 |
| DroneAudioSet | drone2_low_50cm_M_center_File2 | 105.08 | 2.92 | harmonic-sum margin 2.92 dB < 3 dB |
| DroneAudioSet | drone2_low_50cm_M_center_File4 | 17.11 | 0.25 | harmonic-sum margin 0.25 dB < 3 dB; half-to-half 12.03 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_center_File5 | 101.04 | 1.06 | harmonic-sum margin 1.06 dB < 3 dB; half-to-half 1.73 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_center_File6 | 105.72 | 2.94 | harmonic-sum margin 2.94 dB < 3 dB |
| DroneAudioSet | drone2_low_50cm_M_center_silence | 104.60 | 2.40 | harmonic-sum margin 2.40 dB < 3 dB |
| DroneAudioSet | drone2_low_50cm_M_down_File1 | 104.88 | 1.27 | harmonic-sum margin 1.27 dB < 3 dB |
| DroneAudioSet | drone2_low_50cm_M_down_File4 | 144.42 | -0.27 | harmonic-sum margin -0.27 dB < 3 dB; half-to-half 12.74 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_down_File5 | 105.20 | 0.90 | harmonic-sum margin 0.90 dB < 3 dB; half-to-half 1.83 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_down_silence | 104.42 | 2.93 | harmonic-sum margin 2.93 dB < 3 dB |
| DroneAudioSet | drone2_low_50cm_M_up_File1 | 15.24 | 0.16 | harmonic-sum margin 0.16 dB < 3 dB; half-to-half 16.05 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_up_File4 | 17.11 | 1.78 | harmonic-sum margin 1.78 dB < 3 dB; half-to-half 12.03 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_up_File5 | 15.00 | -0.98 | harmonic-sum margin -0.98 dB < 3 dB; half-to-half 91.16 rev/s > 1.0 |
| DroneAudioSet | drone2_low_50cm_M_up_File6 | 105.58 | 1.43 | harmonic-sum margin 1.43 dB < 3 dB |
| DroneAudioSet | drone2_low_50cm_M_up_silence | 104.56 | 2.41 | harmonic-sum margin 2.41 dB < 3 dB |
| KAIST-rotating-acoustic | 0Nm_BPFI_03 | 107.25 | -6.24 | harmonic-sum margin -6.24 dB < 3 dB |
| KAIST-rotating-acoustic | 0Nm_BPFI_10 | 107.20 | -7.75 | harmonic-sum margin -7.75 dB < 3 dB |
| KAIST-rotating-acoustic | 0Nm_BPFO_03 | 22.83 | 8.07 | half-to-half 98.56 rev/s > 1.0 |
| KAIST-rotating-acoustic | 0Nm_BPFO_10 | 60.94 | -0.10 | harmonic-sum margin -0.10 dB < 3 dB |
| KAIST-rotating-acoustic | 0Nm_Normal | 120.84 | -2.74 | harmonic-sum margin -2.74 dB < 3 dB; half-to-half 1.80 rev/s > 1.0 |
| SPCUP19-ChuMS-bench | 1prop_repeat1 | 100.48 | 0.48 | harmonic-sum margin 0.48 dB < 3 dB; half-to-half 12.38 rev/s > 1.0 |
| SPCUP19-ChuMS-bench | 1prop_repeat2 | 25.74 | -1.26 | harmonic-sum margin -1.26 dB < 3 dB; half-to-half 69.40 rev/s > 1.0 |
| SPCUP19-ChuMS-bench | 1prop_repeat3 | 99.28 | 0.24 | harmonic-sum margin 0.24 dB < 3 dB |
| SPCUP19-ChuMS-bench | 2prop_repeat1 | 84.80 | 0.33 | harmonic-sum margin 0.33 dB < 3 dB |
| SPCUP19-ChuMS-bench | 2prop_repeat2 | 80.88 | -0.50 | harmonic-sum margin -0.50 dB < 3 dB; half-to-half 1.54 rev/s > 1.0 |
| SPCUP19-ChuMS-bench | 2prop_repeat3 | 83.06 | 2.63 | harmonic-sum margin 2.63 dB < 3 dB |
| SPCUP19-ChuMS-bench | 3prop_repeat1 | 76.80 | -0.03 | harmonic-sum margin -0.03 dB < 3 dB; half-to-half 2.06 rev/s > 1.0 |
| SPCUP19-ChuMS-bench | 3prop_repeat3 | 80.46 | 0.12 | harmonic-sum margin 0.12 dB < 3 dB |
| SPCUP19-egonoise | AGH__ego-noise__single_rotors__4 | 77.66 | 1.62 | harmonic-sum margin 1.62 dB < 3 dB |
| SPCUP19-egonoise | AGH__static_clean__0 | 115.50 | 0.11 | harmonic-sum margin 0.11 dB < 3 dB; half-to-half 24.62 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_clean__1 | 50.15 | -0.53 | harmonic-sum margin -0.53 dB < 3 dB; half-to-half 91.58 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_clean__2 | 49.43 | -0.21 | harmonic-sum margin -0.21 dB < 3 dB; half-to-half 64.59 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_clean__3 | 141.20 | -0.07 | harmonic-sum margin -0.07 dB < 3 dB; half-to-half 65.38 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_clean__4 | 124.18 | -0.06 | harmonic-sum margin -0.06 dB < 3 dB; half-to-half 42.34 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_clean__5 | 50.56 | 0.28 | harmonic-sum margin 0.28 dB < 3 dB; half-to-half 66.23 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_clean__6 | 49.96 | 0.12 | harmonic-sum margin 0.12 dB < 3 dB |
| SPCUP19-egonoise | AGH__static_clean__7 | 149.72 | -0.20 | harmonic-sum margin -0.20 dB < 3 dB; half-to-half 114.68 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_clean__8 | 50.84 | -0.46 | harmonic-sum margin -0.46 dB < 3 dB; half-to-half 8.96 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_clean__9 | 113.68 | -0.16 | window 7.00 s < 8 s; harmonic-sum margin -0.16 dB < 3 dB; window too short for a two-half stability test |
| SPCUP19-egonoise | AGH__static_corrupted__0 | 112.74 | -0.24 | harmonic-sum margin -0.24 dB < 3 dB; half-to-half 36.90 rev/s > 1.0 |
| SPCUP19-egonoise | AGH__static_corrupted__1 | 111.30 | 1.46 | harmonic-sum margin 1.46 dB < 3 dB |
| SPCUP19-egonoise | Diagonal_Unloading__recordings__static__static_10m | 98.76 | 0.29 | harmonic-sum margin 0.29 dB < 3 dB; half-to-half 11.03 rev/s > 1.0 |
| SPCUP19-egonoise | Diagonal_Unloading__recordings__static__static_2m | 99.94 | 2.03 | harmonic-sum margin 2.03 dB < 3 dB |
| SPCUP19-egonoise | Idea_ssu__stationary_1 | 92.16 | 1.76 | harmonic-sum margin 1.76 dB < 3 dB |
| SPCUP19-egonoise | Idea_ssu__stationary_2 | 96.82 | 1.96 | harmonic-sum margin 1.96 dB < 3 dB; half-to-half 4.02 rev/s > 1.0 |
| SPCUP19-egonoise | Shout_COOEE__SPCUP19_Shout_COOEE_StaticSubmission1 | 140.48 | 0.01 | harmonic-sum margin 0.01 dB < 3 dB; half-to-half 30.68 rev/s > 1.0 |
| SPCUP19-egonoise | Shout_COOEE__SPCUP19_Shout_COOEE_StaticSubmission2 | 99.20 | 0.03 | harmonic-sum margin 0.03 dB < 3 dB; half-to-half 40.68 rev/s > 1.0 |
| drone_audio | B_S2_D1_067-bebop | 148.72 | 0.07 | window 1.02 s < 8 s; harmonic-sum margin 0.07 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_068-bebop | 144.20 | 0.72 | window 1.02 s < 8 s; harmonic-sum margin 0.72 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_069-bebop | 143.08 | -0.49 | window 1.02 s < 8 s; harmonic-sum margin -0.49 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_070-bebop | 90.76 | -0.71 | window 1.02 s < 8 s; harmonic-sum margin -0.71 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_071-bebop | 126.48 | -0.17 | window 1.02 s < 8 s; harmonic-sum margin -0.17 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_072-bebop | 147.38 | 1.58 | window 1.02 s < 8 s; harmonic-sum margin 1.58 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_073-bebop | 137.28 | -0.07 | window 1.02 s < 8 s; harmonic-sum margin -0.07 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_074-bebop | 144.46 | 0.10 | window 1.02 s < 8 s; harmonic-sum margin 0.10 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_075-bebop | 147.26 | 0.83 | window 1.02 s < 8 s; harmonic-sum margin 0.83 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_076-bebop | 143.52 | -0.26 | window 1.02 s < 8 s; harmonic-sum margin -0.26 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_077-bebop | 127.52 | -0.72 | window 1.02 s < 8 s; harmonic-sum margin -0.72 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_078-bebop | 148.74 | -0.54 | window 1.02 s < 8 s; harmonic-sum margin -0.54 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_079-bebop | 136.62 | -0.38 | window 1.02 s < 8 s; harmonic-sum margin -0.38 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_080-bebop | 36.46 | -0.66 | window 1.02 s < 8 s; harmonic-sum margin -0.66 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_081-bebop | 133.98 | -0.56 | window 1.02 s < 8 s; harmonic-sum margin -0.56 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_082-bebop | 142.78 | 0.41 | window 1.02 s < 8 s; harmonic-sum margin 0.42 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_083-bebop | 122.76 | -1.15 | window 1.02 s < 8 s; harmonic-sum margin -1.15 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_084-bebop | 148.74 | 1.13 | window 1.02 s < 8 s; harmonic-sum margin 1.13 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_085-bebop | 145.98 | -0.21 | window 1.02 s < 8 s; harmonic-sum margin -0.21 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_086-bebop | 132.38 | 0.10 | window 1.02 s < 8 s; harmonic-sum margin 0.09 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_087-bebop | 136.38 | -0.65 | window 1.02 s < 8 s; harmonic-sum margin -0.65 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_088-bebop | 143.08 | 0.05 | window 1.02 s < 8 s; harmonic-sum margin 0.05 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_089-bebop | 146.98 | 1.40 | window 1.02 s < 8 s; harmonic-sum margin 1.40 dB < 3 dB; window too short for a two-half stability test |
| drone_audio | B_S2_D1_090-bebop | 149.96 | -0.22 | window 1.02 s < 8 s; harmonic-sum margin -0.22 dB < 3 dB; window too short for a two-half stability test |
| zenodo_drone_noises | n116 | 39.58 | 1.66 | harmonic-sum margin 1.66 dB < 3 dB |
| zenodo_drone_noises | n117 | 97.04 | 2.50 | harmonic-sum margin 2.50 dB < 3 dB |
| zenodo_drone_noises | n119 | 93.46 | 1.62 | harmonic-sum margin 1.62 dB < 3 dB |
| zenodo_drone_noises | n120 | 89.40 | 0.09 | harmonic-sum margin 0.09 dB < 3 dB; half-to-half 7.38 rev/s > 1.0 |
| zenodo_drone_noises | n121 | 31.72 | -1.05 | harmonic-sum margin -1.05 dB < 3 dB |
| zenodo_drone_noises | n122 | 95.06 | 0.42 | harmonic-sum margin 0.42 dB < 3 dB; half-to-half 6.19 rev/s > 1.0 |
