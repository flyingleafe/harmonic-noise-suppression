# A per-rig generative model of rotor-speed trajectories

**Status:** in progress — 2026-09-15 → . Campaign `rps-trajectory-model`, branch
`main`. Code: `src/experiments/rps_traj/{data,stats,model,baseline,posterior,flight}.py`,
drivers `scripts/rps_traj_{fit,compare,real_stats,posterior}.py`, exploration and
diagnostics `scripts/_rps_traj_{explore,diag}.py`, tables
`scripts/_rps_traj_rounds_table.py`. Artefacts: `results/rps_traj/`. Every table
below is printed by `python scripts/_rps_traj_rounds_table.py` from those
artefacts — re-run it and paste after each round instead of editing numbers.

## Motivation

The training streams that teach an RPS predictor what a rotor-speed trajectory
looks like draw that trajectory from a hand-written synthesiser
(`src/data_processing/rps_synthesis.py`). The transfer pair
(`docs/experiments/rig-sampler-transfer-pair.md`) measured what that costs: with
the noise family widened around two real acoustic fits, the best real MAE is
5.37 rev/s (hard arm) and 5.72 (easy), against 2.99 for a model trained on real
audio. The whole residual gap sits in the transitions, where both synthetic arms
collapse the four rotors onto nearly one speed — output spread 0.28 (easy) and
0.19 (hard) against 4.71 for the real-trained model. The trajectories are too
smooth and too rigid, so the campaign asks whether a fitted generative model of
the trajectory itself closes that gap.

**The recollection, and its correction.** The campaign was opened on the
understanding that the streams sample a MIXTURE of the OU generator and the
intermittent generator. `agent://SamplerScout` refutes that, with file
references:

- Selection is deterministic, not a mixture: `rps_kind != "full_flight"` calls
  `generate_intermittent`, `"full_flight"` calls `generate_full_flight`
  (`src/data_processing/stochastic_rotor_noise.py:2129-2168`, mirrored in
  `rotor_spectral_model.py:379-425` and `generated_noise.py:79-104`). Every named
  policy states `kind: full_flight`. The 0.4/0.4/0.2 mixture in those policies is
  over noise SOURCES, not over trajectory models.
- The control-to-rotor half of the recollection is correct: all generators build
  four control modes and map them with the 4x4 `MIXER` of
  `src/tracking/rotors.py:18-30`; the inverse is `MIXER.T @ w / 4`.
- Nothing in the streams is FITTED. `fit_config` exists for the OU generator
  (`rps_synthesis.py:148-205`) and is never invoked; the intermittent and
  full-flight profiles are hand-coded constants, and the flight-phase ranges are
  documented as "loosely calibrated" (`rps_synthesis.py:585-604`).
- No reusable empirical per-rotor autocorrelation or cross-correlation
  diagnostic existed anywhere in `src/`, `scripts/`, `notebooks/` or `docs/`.
  Inter-rotor correlation was induced structurally by the mixer and never
  matched against data.

So the campaign needs three things that did not exist: real per-rotor telemetry
from more than two rigs, a frozen rule that says whether one trajectory model is
better than another, and a model with a likelihood.

## Data

`agent://DatasetSurvey` surveyed every public corpus that might carry
time-indexed ACTUAL per-rotor speed. Inclusion rule: measured or actual
individual-rotor speed, not PWM, throttle, thrust or force command.
`rev/s = rpm/60`; eRPM needs a pole-pair division first.

### Accepted

| Dataset | Vehicle, rotors | Speed channel; units; native rate | Corpus | Licence / access |
|---|---|---|---|---|
| Blackbird | 1 custom quad, 4 | ROS `blackbird/MotorRPM`, `float32[] rpm`; RPM; ~190 Hz | 168 flights / >10 h published | MIT (tooling); host dead, see below |
| NeuroBEM | 1 custom quad, 4 | `processed_data/*.csv` `mot 1..4`; rad/s; 400 Hz | 96 flights / 1 h 15 min | Not stated; open HTTP |
| PI-TCN (ARPL/NYU) | 1 micro quad 0.25 kg, 4 | `/dragonfly17/motor_rpm`, `float64[4] rpm`; RPM; 100 Hz | 68 flights / 58 min 03 s | GPL-3.0 code, data not stated |
| VID (ZJU) | 1 DJI M100 derivative, 4 | `/m100withm3508/m3508_m1..m4`, `int16 rpm`; RPM; ~1 kHz CAN | 22 sequences / 42 min 55 s | AGPL-3.0 repo; passworded share |
| DREGON | 1 MikroKopter quad, 4 | `motor.measured`, `motor.command` (4 columns each); rev/s; ~1 kHz grid | 11 named in-flight conditions plus benches | Personal/educational/academic only |
| NanoBench (IDSIA) | 1 Crazyflie 2.1 Brushless, 4 | `motor.m1_rpm..m4_rpm`, published as rad/s; 100 Hz | 15 runs / 75 k samples | Not stated; GitHub LFS |
| Pelican (AscTec) | 1 AscTec Pelican, 4 | MATLAB `Motors`; AscTec integer 0-218, NOT calibrated RPM; 100 Hz | 54 flights / 3 h 50 min | Not stated; open HTTP |

Michael's DJI Matrice 100 (FLY124/FLY125 plus the held-out FLY103/FLY108) was
already in the project and is the eighth candidate: DatCon ESC feedback at
29.41 Hz, calibrated by `data_processing.sources.michaels.read_motor_speeds`.

### Rejected

| Candidate | Reason |
|---|---|
| DronePropA (`S235234092500321X`, Mendeley `ftdyxrr3c5`) | PWM/ESC COMMAND only: `QDrone_data` rows 47-54 are motor command and ESC command in volt %. No RPM, RPS or eRPM field |
| Drone Sensor Fusion Dataset (IEEE DataPort) | Login wall; only "actuator control signals" documented, no speed field or schema visible. PX4 CAN log `esc_report.esc_rpm`, but capability is not dataset evidence |
| IDF_DS (Zenodo 16992976) | Fixed wing (SpeedyBee F405 INAV, Pixhawk 6X), actuator commands and states only, no per-rotor RPM |
| Race Against the Machine / TII racing | Per-motor NORMALISED THRUST feedback (`motors_thrust_*.csv`), no motor-RPM column in the released schema |
| D3Set | `actuator.csv` carries PWM; the per-rotor values are network output, not feedback |
| UZH-FPV, EuRoC MAV | No motor-speed topic in the documented bag or CSV types |
| PX4 Flight Review public logs | `esc_report.esc_rpm` is optional per airframe; no way to select complete 4-motor records from the landing page. A route, not a dataset |
| Betaflight Blackbox corpus | Firmware can log bidirectional-DSHOT RPM; no surveyable public collection exists |
| Kaggle "Drone Flight Video with Telemetry (GPS & ESC)" | No exposed column or sample establishes per-rotor feedback |
| Pelican, at ingest | The 0-218 integer is an uncalibrated autopilot estimate from motor current, not rev/s. Highest duration of any candidate, excluded on units |

### Access records

Recorded verbatim, because each one changes what the campaign can claim.

- **Blackbird — canonical host dead, one flight recoverable worldwide.**
  Checked 2026-09-15. `blackbird-dataset.mit.edu` is NXDOMAIN (dead since
  ~2024); repo issues #28 (maintainer, 2024-04-10: "links are dead, moving from
  AWS to another solution"), #30, #31, #32, #34 are open with no replacement
  URL, and `sequenceDownloader.py` still points at the dead host. The legacy S3
  buckets `ijrr-blackbird-dataset` and `blackbird-dataset-static-index` exist but
  return `AccessDenied` for `ListBucket` and for every key tried (path-style and
  accelerate, with and without the `BlackbirdDatasetData/` prefix);
  `blackbird-dataset.mit.edu` as a bucket is `NoSuchBucket`. The academic
  torrent (`eb542a23…c2656`, 4.79 TB, `BlackbirdDatasetData`) parses to 747
  files = 560 `.tar` of camera images plus 187 `.mp4`, ZERO csv/bag/yaml/txt
  entries, 0 of 4 seeders — consistent with repo issue #1 (2018-10-24, "ROS bag
  files and IMU data files are missing from the torrent"). Wayback CDX holds
  only `videos/` directory listings. HuggingFace, Zenodo, Kaggle, OpenDataLab,
  Harvard Dataverse, archive.org and all 30 GitHub forks are empty of
  telemetry. One third-party copy survives and is pinned: the publisher's
  verbatim `bagToCsv` export of `clover/yawForward/maxSpeed5p0`
  (8,017,574 B) vendored in `fbanelli/learned-acceleration-estimator`. The
  module keeps the full 45-flight subset list and re-fetches automatically if
  MIT restores the host.
- **PI-TCN — README Drive ids 404, all 68 bags recovered.** Both ids in the
  README (`1b1PFSBlKTdrlTIurYNpTJWWEx1KIJzuR` bags,
  `1s7nSqATpCS849csSdkHNL0-VLZwdNzg4` PDFs) return HTTP 404 "Page not found"
  through `/file/d/<id>/view`, `/uc?id=`, `/open?id=`, `docs.google.com/uc` and
  `drive.usercontent.google.com/download`. Not a quota or virus interstitial:
  the control id `0B9P1L--7Wd2vNm9zMTJWOGxobkU` still returns 200 through the
  same path. `gdown` 6.1.0 fails even on its own public test file in this venv.
  The 68 ORIGINAL bags, original names and full topic set, are redistributed in
  `data/pi_tcn/rosbags/` of `data.zip` (id `1BB-r63qgiqB5uJ5xVbTcfR-6j9rXCFCA`,
  2,043,979,141 B) of the authors' follow-up repo
  `arplaboratory/long-horizon-dynamics`. 68 bags / 61.35 min matches the
  README's "68 trajectories, 58'03''".
- **VID — the passworded share works non-interactively.** QNAP share API,
  `ezEncode` is plain base64, so `ep = base64("viddataset@2021")` =
  `dmlkZGF0YXNldEAyMDIx`. (1) `POST /share.cgi` with
  `ssid=<ssid>&func=check_passwd&ep=<ep>` returns `{"status":1}`; (2)
  `GET /share.cgi?func=get_list&ssid=<ssid>&ep=<ep>&fid=<ssid>&path=/<folder>…`
  returns the listing with exact file sizes; (3)
  `GET /share.cgi?ssid=<ssid>&openfolder=forcedownload&ep=<ep>&fid=<ssid>&path=/<folder>&filename=<name>`
  returns the file, with `Range` honoured at 13-26 MB/s (one shared cap;
  parallel streams add nothing). The publisher bundles the D435 streams into the
  flight bags and offers no telemetry-only variant, and 4706 of the hover bag's
  4751 chunks carry motor messages, so selective download is useless: 39 GB of
  bag bytes were STREAMED through an in-process rosbag1 reader and only the
  motor records were written (22 MB of `*.motors.npz` kept).
- **IEEE DataPort drone-sensor-fusion — login wall.** Subscription or login
  required, no speed field visible in the exposed schema, skipped rather than
  guessed.
- **DronePropA — PWM/ESC command only** (see the rejection table). This is the
  ScienceDirect/Data in Brief candidate the campaign started from.
- **Zenodo 16992976 — fixed wing, no rotor RPM**, rejected.
- **Pelican — units.** Excluded despite the largest disclosed duration: the
  "actual" channel is an autopilot estimate on an uncalibrated 0-218 integer
  scale, not rev/s.
- **Licences.** NeuroBEM, NanoBench and PI-TCN publish NO dataset licence
  (NeuroBEM carries only a copyright line and a citation request; the NanoBench
  repo reports `license: null` as of 2026-09-15). Blackbird is MIT and VID
  AGPL-3.0 for their TOOLING, not for the data. All are used here for research
  with attribution, recorded verbatim in each `PROVENANCE['license']`, each
  derivation note and `docs/data-catalog.md`. This is a publication decision,
  not a code one.

### The seven ingested rigs

Every rig arrives as a published `tdframe-v1` frames dataset with one `rps`
Series per flight, pinned in `dload.lock`:

| Rig | Derivation | `dload.lock` pin | Native rate | Speed source |
|---|---|---|---|---|
| `michaels` | `michaels-frames`, `michaels-test-frames` | `8e9d1495…cca6e46`, `353cc523…9f338991` | 29.41 Hz | DatCon ESC feedback, 1 RPM steps |
| `dregon` | `DREGON-frames` | `261b0997…2ba89` | ~1 kHz grid (`measured` ~45 Hz effective, `command` 670 Hz) | MikroKopter `motor.measured` / `motor.command` |
| `neurobem_quad` | `NeuroBEM-frames` | `39c14d5c…3af4f88` | 400 Hz (one flight at 164 Hz) | Betaflight ESC feedback |
| `pitcn_quad` | `PITCN-frames` | `d2494a91…8effd01ab8` | 100 Hz | ESC telemetry, 4 RPM quantised |
| `nanobench_cf21b` | `NanoBench-frames` | `3aab8eb2…7294215cd6` | 100 Hz | Bidirectional DSHOT eRPM / 6 |
| `vid_m100` | `VID-frames` | `3d63961c…12cbf2b48` | ~1 kHz | M3508/C620 CAN feedback |
| `blackbird_quad` | `Blackbird-frames` | `c08a5c8d…c9fb7f4632` | 187 Hz measured (190 nominal) | Optical motor encoders (tachometer) |

Shard sizes: NeuroBEM-frames 247 samples / 55.2 MiB, PITCN-frames 68 / 11.3 MiB,
NanoBench-frames 15 / 2.3 MiB, VID-frames 8 / 23.6 MiB, Blackbird-frames 1 /
1.5 MiB. All five new pins were verified by a clean-cache pull.

Unit provenance is established per rig rather than assumed: NeuroBEM rad/s /
2 pi; NanoBench eRPM / 6 pole pairs then rad/s / 2 pi (hover 1700 rad/s =
271 rev/s = 16.2 kRPM against the publisher's own 45 g mass and
`Kt = 3.72e-8`); PI-TCN rpm / 60 (hover check `sqrt(mg/4/k_f)` = 11832 RPM =
197 rev/s against measured per-flight means 185-230 rev/s); VID rpm / 60 (the
publisher's `rpmconvert.cpp` copies the field unscaled, and commanded
`target_rpm/60` agrees with measured to <0.1 rev/s); Blackbird rpm / 60 (hover
`sqrt(0.915 * 9.81 / 4 / 2.27e-8)` = 9.94 kRPM = 166 rev/s against measured
means 164-183 rev/s).

Rotor axes are permuted to the MIXER order `[RFront, LFront, LBack, RBack]` at
load time (`ROTOR_TO_MIXER`), so the mode projection means the same thing on
every rig. Blackbird is deliberately absent from that table — MIT's `MotorRPM`
column order is undocumented (repo issue #26, unanswered), so it is inferred
from the diagonal-pair correlation signature: summed diagonal correlation 1.723
for the published order against -0.566 and -0.606 for the other two pairings, a
margin of 2.3 against a sampling-noise scale of 0.02 (`PAIRING_MARGIN` 0.05).

The corpus, on the frozen airborne rule
(`python scripts/_rps_traj_rounds_table.py`):

| rig | flights | recorded (s) | airborne (s) | segments | overall mean (rev/s) | rotor var ((rev/s)^2) | NaN |
|---|---|---|---|---|---|---|---|
| blackbird_quad | 1 | 209 | 199 | 1 | 181.1 | 432 587 513 589 | 0 |
| dregon | 10 | 495 | 429 | 10 | 80.5 | 7.5 12.6 5.0 7.0 | 0 |
| michaels | 4 | 506 | 395 | 4 | 78.4 | 52.2 20.3 23.8 21.4 | 0 |
| nanobench_cf21b | 15 | 751 | 668 | 19 | 277.7 | 461 409 369 463 | 0 |
| neurobem_quad | 247 | 4520 | 2479 | 157 | 216.8 | 2989 2919 2788 3112 | 0 |
| pitcn_quad | 68 | 3682 | 3345 | 75 | 201.9 | 237 241 364 179 | 0 |
| vid_m100 | 4 | 528 | 397 | 18 | 87.7 | 22.0 20.8 20.5 35.6 | 0.00134 |
| **total** | **349** | **10692** | **7912** | **284** | | | |

The rigs span a factor of 3.5 in hover level (78.4 to 277.7 rev/s) and a factor
of 600 in per-rotor variance (5.0 to 3112 (rev/s)^2). NeuroBEM loses 2041 s of
its 4520 s to the airborne rule: its median segment is 7.25 s, so the 1 s
erosion at each end plus the 5 s minimum removes many segments entirely. It is
still the largest airborne corpus in the campaign.

### The common 100 Hz grid

One analysis rate, frozen (`src/experiments/rps_traj/data.py`): **100 Hz**,
linear interpolation from the first native sample. The reasoning:

- The consumer is the acoustic renderer's STFT grid, 16 kHz / 512 = 31.25 Hz.
  100 Hz is more than 3x that.
- 100 Hz is at or above every trustworthy native band in the corpus. DREGON's
  `measured` is usable to ~5 Hz, Michael's to ~8 Hz, and the fastest genuine ESC
  feedback in the corpus is 400 Hz.
- The model is continuous-time, so it samples at any rate. The grid is an
  analysis choice, not a model limit.
- Telemetry faster than 100 Hz is low-passed at 40 Hz first (zero-phase
  4th-order Butterworth, `ANTIALIAS_HZ`/`ANTIALIAS_ORDER`), so decimation cannot
  fold rotor-speed ripple into the modelled band.
- Michael's 29.41 Hz log is interpolated as is and has no content above its own
  Nyquist of ~15 Hz. Any `michaels` comparison is informative below ~15 Hz only.

Gaps wider than two native sample periods stay NaN instead of being bridged; the
statistics drop NaNs pairwise.

## What the telemetry says

`scripts/_rps_traj_explore.py` measured the second-order structure of every
series the project held before the new ingests, on the frozen pipeline (100 Hz
grid, airborne rule, Welch 10 s Hann 50 %, per-rotor per-run demeaned,
duration-weighted pooling). Full record:
`results/rps_traj/explore/findings.md` and `summary.json`.

**Channel fidelity comes first, because it gates every spectral claim.**

- DREGON's `motor.measured` is a ~45 Hz sample-and-hold, not a 1002 Hz
  measurement: 95.3 % of consecutive native samples are bit-identical, the value
  changes 45 times/s (median hold 19 ms) and takes 52 distinct values per flight
  spaced 0.30 rev/s (0.35 % of the operating point). `motor.command` in the same
  file updates at 670 Hz with a 2.00e-4 rev/s step. On the same room1 flights
  the `measured - command` residual is flat at 2.99e-2 (rev/s)^2/Hz over
  0.5-10 Hz and accounts for 1 % / 14 % / 106 % / 114 % of the measured band
  variance in <0.5 / 0.5-5 / 5-15 / >15 Hz. `results/rps_traj/explore/dregon_channel_fidelity.png`
  shows it directly: the left panel overlays measured, command and residual PSD
  on the native 1 kHz grid with the residual rising above both above a few tens
  of hertz, and the right panel shows 1 s of rotor RFront in which `measured` is
  a visible staircase of flat steps while `command` varies continuously.
  **DREGON measured is trustworthy below ~5 Hz only.**
- Michael's log has the opposite defect: fine in amplitude, coarse in time.
  Every value is an integer RPM (16.7 mrev/s, 0.02 % of the operating point,
  20x finer than DREGON's 0.30 rev/s), but the log rate is 29.41 Hz and it is
  partly held — P(delta = 0) is 0.41-0.43 on the four flights, median hold
  68 ms, effective update 17 Hz. Conditional on changing, the median jump is
  29 RPM against a 7 RPM unconditional median: an asynchronous ESC update
  resampled into the log. Usable bandwidth ~8 Hz. The 12-14.7 Hz band sits
  1068-2418x above the 1 RPM quantisation floor of 1.57e-6 (rev/s)^2/Hz and
  continues (or flattens onto) the 2-8 Hz power-law trend instead of rolling off
  into Nyquist, so that tail is folding and hold-resampling, not shaft motion.

**Band fractions.** Fraction of the 0-50 Hz PSD integral, rotor mean, in
<0.5 / 0.5-5 / 5-15 / >15 Hz: michaels 42.8 / 46.4 / 2.4 / 0.1 %;
dregon room1 measured 39.3 / 35.8 / 12.4 / 8.6 %; dregon room1 command
51.0 / 39.8 / 3.3 / 1.1 %; dregon room2 command 41.7 / 40.8 / 8.3 / 3.3 %. The
two right-hand columns are artefacts on both rigs (interpolation residue on
michaels, the staircase on DREGON). The comparable columns agree within 12
percentage points: **40-50 % of the variance is below 0.5 Hz and 36-40 % in
0.5-5 Hz.**

**Mode dominance.** The collective/yaw pair carries almost everything: common +
yaw is 78 % of rotor variance on michaels, 78 % on room1 measured, 81 % on room1
command, 69 % on room2 command; roll and pitch sit about 10 dB below collective
across the band. The eigenbasis is NOT the mixer basis — the leading eigenvector
projects on common 0.55 and yaw 0.67 on michaels, and eigenvalue spread
lambda1/lambda4 is 5.1-14.5x. A quad trims yaw with the same two diagonal rotors
it uses for thrust, which is also why the coupling is diagonal-pair coupling:
zero-lag correlation for diagonal against adjacent pairs is 0.57 vs 0.03
(michaels), 0.56 vs 0.37 (room1 measured), 0.38 vs -0.03 (room2 command), with
cross-spectral phases near 0 or 180 degrees below 0.5 Hz, so the coupling is
instantaneous rather than lagged. `results/rps_traj/explore/michaels_eigen.png`
shows both bases side by side with their variance shares (e1 48.5 %, e2 31.2 %;
common 40.3 %, yaw 37.9 %) and the cosine table that says e1 is a common/yaw
mixture. **Consequence: the loud pair must be free to rotate inside the
collective-yaw plane, and the 4x4 covariance needs exactly two free
off-diagonals.**

The four modes do NOT share a spectral shape. Fraction of each mode's own
variance below 0.5 Hz (common / roll / pitch / yaw): michaels 28/37/44/57 %,
room1 measured 39/30/32/54 %, room1 command 45/45/47/70 %, room2 command
23/35/50/65 %. On michaels the yaw mode is the reddest and collective dominates
from ~0.7 Hz up. One shared spectrum scaled per mode cannot fit this.

**ACF time scales.** Rotor-mean autocorrelation: michaels 0.980 at 0.02 s, 0.82
at 0.1 s, 0.16 at 1 s, 0.007 at 10 s, 1/e crossing 0.43 s; room1 measured
0.907 / 0.78 / 0.23 / -0.036, 1/e at 0.49 s; room1 command
0.978 / 0.87 / 0.31 / -0.039, 1/e at 0.69 s; room2 command
0.919 / 0.71 / 0.29 / -0.075, 1/e at 0.65 s. Two time scales, so no single
exponential (AR(1)) fits.

**Heavy tails.** Excess kurtosis of the 0.01 s and 0.5 s increments (Gaussian =
0): michaels 13.5 and 2.7; room1 measured 2.9 and 13.9; room1 command 46.4 and
18.1; room2 command 27.3 and 5.1. Michael's 0.01 s figure is really its native
0.034 s increment and is inflated by the hold spike at delta = 0, so its
cleanest number is the 0.5 s lag at 2.7 — still far from Gaussian. DREGON is
unambiguous at both lags. **Gaussian increments are excluded at every lag the
data can measure.**

**Resonance.** Using a robust cubic baseline in log frequency and requiring 3 dB
prominence with both -3 dB crossings inside the band, the only clean resonance
in the corpus is in the DREGON COMMAND channel: room2 command peaks at 3.00 Hz,
+4.3 dB, Q ~ 6.0, with a second peak at 6.70 Hz; room1 command at 5.20 Hz,
+4.4 dB, Q ~ 1.1. That is the MikroKopter attitude loop. Michael's ESC feedback
shows no hump worth the name (best +2.0 dB at 8.70 Hz, no -3 dB crossing inside
the band): a DJI M100's rotor speed is a nearly featureless power law, broadband
slope -20.9 dB/decade. DREGON measured's 19 Hz bump is the staircase — it is
absent from `command` on the same flights. **A resonance must therefore be a
per-rig parameter that is allowed to vanish.**

Coupling is band- and channel-dependent in a way that is itself a measurement
statement: mean magnitude-squared coherence over the six pairs falls with
frequency on michaels (0.33 → 0.27 → 0.20) and room1 measured
(0.26 → 0.23 → 0.13) but RISES on room2 command (0.28 → 0.43 → 0.58), because
the 3 Hz attitude-loop resonance is common-mode. "Independent above 5 Hz" is a
property of the measurement chain, not of the vehicle; every series is coupled
below 1 Hz, so independent per-rotor noise is wrong there.

Two identity findings came out of the same pass. FLY103/FLY108 are the same
physical vehicle as FLY124/FLY125 — all four DatCon headers carry flight
controller serial `mcID(SN)|041DE70827`, `ACType|M100`, `mcVer|v3.1.6.108` and
the same firmware date — and the yaw trim (the airframe's own torque imbalance)
agrees to a 1.20 rev/s spread about 5.43 rev/s while roll and pitch trim move by
up to 5.37 rev/s. They are therefore pooled into the single rig `michaels`
(`MICHAELS_TEST_IS_SAME_RIG`).

**What the model must reproduce**, as the exploration stated it: (i) 39-51 % of
variance below 0.5 Hz with a power-law shoulder, not a flat spectrum; (ii) a
collective+yaw-dominated 4x4 covariance with lambda1/lambda4 ~ 9 and
diagonal-pair correlation above adjacent-pair; (iii) a two-time-scale ACF (1/e
at 0.4-0.6 s with a 1-10 s tail); (iv) heavy-tailed increments at every lag; (v)
an OPTIONAL per-rig resonance (0 dB for the M100, +4 dB at 3.0 Hz for the
MikroKopter). A Gaussian per-rotor AR(1) reproduces none of them.

Target (ii) survived the five new rigs only in part. Diagonal correlation
exceeds adjacent on 4 of 7 rigs — michaels 0.570 vs 0.070, dregon 0.521 vs
0.254, vid_m100 0.500 vs -0.549, blackbird_quad 0.861 vs -0.293 — but
nanobench_cf21b has HIGHER adjacent correlation (0.570 vs 0.673) and
neurobem_quad (0.765 vs 0.799) and pitcn_quad (0.386 vs 0.346) show no contrast.
NeuroBEM flies up to 65 km/h and NanoBench and PI-TCN fly system-identification
trajectories (chirp, random, square) that deliberately excite roll and pitch,
swamping the yaw-diagonal signature. The model's `M diag(var_modes) M^T` form
covers this without a special case: adjacent above diagonal is simply roll and
pitch variance above yaw variance.

## The frozen judge

`src/experiments/rps_traj/stats.py` is the campaign's judge. Every candidate,
from the first strawman to whatever reaches the paper, is summarised by the same
`TrajStats`, compared by the same five-family `discrepancy` and accepted by the
same `passes`. Changing any definition invalidates every number the campaign has
produced, so the definitions are fixed here.

**Grid and support.** 100 Hz (`RATE_HZ`). Only samples inside
`airborne_segments` enter any statistic. Synthetic trajectories go through the
identical segmentation, so a model that never leaves the ground is scored on
nothing.

**Airborne rule** (`src/experiments/rps_traj/data.py`, frozen, never tuned per
rig): let `m(t)` be the mean over rotors; the threshold is
`0.5 * percentile(m, 90)` (`AIRBORNE_FRACTION`, `AIRBORNE_PERCENTILE`); a sample
is airborne when EVERY rotor exceeds it; each run is then eroded by
`ERODE_S` = 1.0 s at both ends and kept only if at least `MIN_RUN_S` = 5.0 s
remains.

**Lags.** `LAG_SAMPLES` is 24 log-spaced lags from 0.05 s to 10 s, rounded to the
100 Hz grid; the rounding is part of the definition. A segment shorter than twice
a lag contributes nothing at that lag.

**The five families** (`FAMILIES`, in report order):

| Family | Statistic | Discrepancy |
|---|---|---|
| `overall_mean` | Mean rev/s over all airborne samples of all rotors, sample-weighted | `abs(delta)` in rev/s |
| `rotor_mean` | The same per rotor | RMS over rotors of `abs(delta mu_r)` in rev/s |
| `rotor_var` | Per-rotor variance over all airborne samples POOLED over flights, about the pooled per-rotor mean | RMS over rotors of `abs(ln(var_model / var_real))`, scale-free |
| `acf` | Per segment, per rotor, within-segment demeaned, BIASED normalised autocorrelation `sum x_t x_{t+k} / sum x_t^2` at the 24 lags, averaged with weight = segment length | RMS over rotors x lags of `abs(delta rho)` |
| `xcorr` | Zero-lag Pearson correlation between rotors on within-segment-demeaned data pooled over all segments | RMS over the 6 off-diagonal pairs of `abs(delta c)` |

Pooling in `rotor_var` is deliberate: a model that reproduces within-flight
wobble but puts every flight at the same hover level is wrong, and this is the
family that says so. NaNs are dropped pairwise everywhere.

**Pass rule** (`passes`): no family may regress beyond `STRICT_EPS` = 1e-9, and
at least `MIN_STRICT_FAMILIES` = 3 must strictly improve. That combination is
what stops a candidate from trading ACF for variance and calling it progress.

**The estimator** (`stats_from_samples`). Model statistics are computed by
drawing the REAL flight durations `n_rep` times over — one sampler call per
duration, with `default_rng([seed, rep, i])`, so the result is a deterministic
function of `(sampler, durations, seed)` — and running the samples through the
same `airborne_segments`. Matching the real durations matters because the ACF is
biased-normalised and segment-length weighted. Rounds 2 and later use
`n_rep` = 40 with ANTITHETIC OFFSETS: consecutive sampler calls are paired and
the second flight of a pair reuses the negated per-flight offset of the first.
Each flight keeps its marginal `N(0, diag(s^2))` law, so `rotor_var`, `acf` and
`xcorr` are unchanged in distribution, but the offsets over an even number of
flights cancel exactly and the pooled model mean is `mu` instead of carrying an
`s / sqrt(N)` error. That error was worth 26.8 rev/s on `neurobem_quad` at
`n_rep` = 5 (visible as round 1's `overall_mean` 26.79) and no affordable
`n_rep` removes it, while the baseline has no offset term and pays nothing.

**Known bias in the shortest lags, stated rather than patched.** The seven lags
below 0.2 s are biased by the loggers in OPPOSITE directions: michaels is
INFLATED (a 0.05 s lag spans 1.47 native samples, so on the 100 Hz grid that lag
is mostly interpolation), DREGON is DEFLATED (additive quantise-and-hold noise
of variance v scales every nonzero-lag ACF by `1/(1 + v/sigma^2)`). Size, from
DREGON room1's within-flight measured-against-command A/B: rotor-mean ACF 0.907
vs 0.978 at 0.02 s and 0.78 vs 0.87 at 0.1 s, converged by ~0.2 s. Those lags
stay in the definition because a model must reproduce what the data does, but a
candidate that matches lags >= 0.2 s and misses only the shortest ones is closer
to the truth than its `acf` number alone suggests.

### The baseline the judge compares against

`src/experiments/rps_traj/baseline.py` fits the CURRENT synthesiser per rig, so
the new model has to beat the incumbent at its best, not at its hand-set
defaults. `generate_intermittent` (Poisson-gated rectangular pulses through a
first-order lag plus OU jitter, mixed through `MIXER`) has no tractable
likelihood, so it is fitted by **simulated moment matching** against the frozen
statistics:

- 22 free parameters (`PARAM_NAMES`): per control mode
  `(trim, cruise_std, maneuver_std, rate_hz, mean_maneuver_s)`, plus the
  airframe's `motor_tau` and `cruise_tau`. Positive fields are optimised in log
  space, the roll/pitch/yaw trims in linear space because they cross zero.
  `aggressiveness` is held at 1.0 (exactly redundant with the free maneuver rate
  and amplitude) and `rps_min`/`rps_max` are held at a wide bracket where the
  clip never bites.
- Objective: the frozen families collapsed by fixed `OBJECTIVE_WEIGHTS` =
  `rotor_var + acf + xcorr + 0.1 rotor_mean + 0.1 overall_mean`. The three
  unit-weighted families are dimensionless and O(0.1-1); the two mean families
  are in rev/s and would otherwise dominate at 10-100x, so they carry 0.1 —
  enough to pin the trims, not enough to buy a mean at the cost of the dynamics.
- Common random numbers: every evaluation simulates the real durations with the
  same seed, so the objective is exactly deterministic. It is not smooth (the
  pulse stream re-shuffles as the maneuver rate moves), so only pattern searches
  apply.
- Optimiser: **Powell, 300 evaluations**, chosen by measurement. At that budget
  Powell reaches 0.693 on michaels and 0.494 on dregon where Nelder-Mead reaches
  1.161 and 0.565 at the same ~135 s, because a 23-point simplex in 22
  dimensions spends its whole budget being built.
- Cost is controlled through `n_rep`, never duration: `search_n_rep` drops 2 to 1
  when the corpus would exceed `SEARCH_SIM_BUDGET_S` = 6000 simulated s per
  evaluation (NeuroBEM and PI-TCN only), because the ACF is segment-length
  weighted and shortening a duration would change the target itself.
  `fitting_durations` drops flights below `MIN_RUN_S + 2 ERODE_S` = 7 s, which
  the airborne rule discards anyway (115 of NeuroBEM's 247).
- `idle_rps` is not identifiable from the objective (the airborne rule erases
  warm-up), so it is read off the real pre-takeoff plateau.
- The returned fit is the better of (start, search result) at `n_rep` = 5, so the
  baseline is never handed to `passes` worse than the stock profile.

Fitted objectives, start to end: michaels 1.347 → 0.693, dregon 1.573 → 0.494,
neurobem_quad 3.524 → 1.683, pitcn_quad 1.526 → 0.649, nanobench_cf21b
1.229 → 0.403, vid_m100 1.385 → 1.010, blackbird_quad 2.836 → 1.207. Two
properties of these fits matter when reading a verdict:

- **The baseline is not a pulse-shaped straw man.** Where it can, the fit
  converts "rare pulses" into "continuous jitter": michaels roll and vid_m100
  roll/pitch drive `maneuver_std` and `mean_maneuver_s` to the bounds floor
  while `cruise_std` grows 10x, and dregon pushes `cruise_tau` to its 1 ms floor
  (white jitter, i.e. it fits the 45 Hz staircase). That is the incumbent
  admitting the real series is continuous wander.
- **The 0.1 weight lets the baseline's mean drift when variance is cheap**:
  blackbird `rotor_mean` 0.18 → 1.16 rev/s and NeuroBEM 0.10 → 0.41 rev/s
  (0.64 % and 0.19 % of the operating point). Both are wins on the weighted
  objective, but `passes` counts `rotor_mean` as a family, so a candidate can
  beat the baseline's dynamics and still be blocked on a mean it has to match.

## The model

Rotor speeds in rev/s, mixer order `[RFront, LFront, LBack, RBack]`
(`src/experiments/rps_traj/model.py`, round 3 form):

```
w(t)   = mu + delta_flight + M R(theta) v(t) + e(t)
v_i(t) = OU(tau_slow_i, sigma_slow_i) + CAR2(f0_i, zeta_i, sigma_osc_i)
```

- `M` is `tracking.rotors.MIXER`, columns `[common, roll, pitch, yaw]`.
- `R(theta)` rotates the (common, yaw) plane and is the identity on roll and
  pitch. The exploration found the two loud modes to be a collective/yaw
  MIXTURE, so the loud pair must be free to rotate inside that plane. `theta` is
  reported in `(-pi/4, pi/4]`; `theta -> theta + pi/2` with a swap of the mode-0
  and mode-3 parameters leaves the model invariant.
- `CAR2` is the white-noise-driven damped harmonic oscillator
  `x'' + 2 zeta w0 x' + w0^2 x = xi`, `w0 = 2 pi f0`, with `sigma_osc` its
  stationary standard deviation. It is `f^-4` above its corner and flat below
  it, and with `zeta` free it is also the optional per-rig resonance the
  exploration asked for: `zeta < 1/sqrt(2)` peaks at `f0` (the MikroKopter's
  3 Hz loop), `zeta ~ 1` is a Matern-3/2-like shoulder, `zeta > 1` is overdamped
  (the M100's featureless power law).
- `e(t)` is per-rotor iid white MEASUREMENT noise with one standard deviation
  `sigma_w`. It carries what a rig's telemetry chain adds on top of the shaft —
  DREGON's 45 Hz staircase, michaels' hold-resampling and interpolation residue
  — and is labelled "measurement" precisely so a training sampler can drop it.
- `delta_flight ~ N(0, diag(s^2))` per ROTOR, one draw per flight. The frozen
  `rotor_var` pools all airborne samples about the POOLED mean, so
  flight-to-flight level differences are part of what is scored, and they are
  per-rotor: michaels' rotor 0 carries 2.5x its neighbours' variance purely
  because its between-flight trim moves 17 rev/s.

**Parameters** (30, `Params`): `mu` (4, rev/s), `theta` (rad), and per mode
`tau_slow` (s), `sigma_slow` (rev/s), `f0` (Hz), `zeta`, `sigma_osc` (rev/s) =
20, plus `sigma_w` (rev/s) and `s` (4, rev/s). `mu` and `s` are ESTIMATED from
the pooled airborne rotor means and the per-flight offsets, so the optimiser
vector is 22-dimensional.

**Hard ranges, imposed by parametrisation rather than by an optimiser box.**
`f0` in (0.05, 20) Hz, `zeta` in (0.2, 3.0), and
`tau_slow = tau_c + (10 - tau_c) sigmoid(u)` with `tau_c = 1/(2 pi f0)`, so the
oscillator is ALWAYS the faster component (no label-switching gauge) and
`tau_slow` never exceeds 10 s — half the block length and the longest lag the
frozen ACF scores. Slower variation is `delta_flight`'s job, and that term is
estimated rather than fitted, so the cap removes a double count rather than
capability.

**One object, not two.** Both components are defined by their EXACT
discrete-time state space on the grid, `Phi = expm(A dt)` and
`Q = P - Phi P Phi^T` with `P` the continuous stationary covariance (exact,
because the process is stationary), and the spectrum the likelihood evaluates is
computed from that same `(Phi, Q, H)`:

```
S(w) = (2/fs) H (e^{iw} I - Phi)^-1 Q (e^{-iw} I - Phi^T)^-1 H^T
```

so the sampler cannot drift away from the likelihood. This is also why the
continuous-time Lorentzian formula is not used: on a 100 Hz grid it misses the
power folded down from above Nyquist (74 % low at 39.5 Hz for `tau` = 0.3 s),
which an earlier fit paid for by inflating `sigma_w` and shortening `tau_slow`.

**Why the likelihood is cheap.** `M^T M = 4 I`, so `A = M R(theta) = 2 Q` with
`Q` orthogonal, and
`S(f) = A diag(S_i(f)) A^T + c I = Q diag(4 S_i(f) + c) Q^T` with
`c = sigma_w^2 / (fs/2)`. The model spectral matrix is diagonalised by the
rotated mixer basis at every frequency, so `log det S` and `tr(S^-1 I)` close in
scalar form with no 4x4 factorisation in the optimiser loop, and the periodogram
enters only through five real arrays
(`(M^T I M)_{00}, _{03}, _{11}, _{22}, _{33}`), block-averaged once before the
fit.

**Fit protocol per round** (Whittle MAP, rounds 1-3). Airborne segments are cut
into `BLOCK_S` = 20 s Hann blocks at 50 % overlap; a segment shorter than
`MIN_BLOCK_S` = 5 s contributes nothing and a segment between 5 s and 20 s
contributes one whole-segment block. Each block is block-mean detrended, and
from round 3 the bins below `MIN_BIN_RAYLEIGH` = 2/T are dropped (0.1 Hz for a
20 s block), because the first Rayleigh bins of a Hann-tapered demeaned block
are leakage-dominated and they were exactly the bins the round-2 fit used to
justify parking unresolvable slow power. Priors are log-normal:
`TAU_SLOW_PRIOR` (2.0 s, 1.5), `SIGMA_PRIOR` (1.0, 3.0), `SIGMA_W_PRIOR`
(0.1, 3.0). The per-rig Whittle band is `FIT_BAND_HZ`. Eight random restarts
around a data-driven start; the reported fit is the best posterior.

## Rounds

Each round changes the model or the protocol, refits all seven rigs, re-scores
against the fitted baseline and is judged by `passes`. Tables printed by
`python scripts/_rps_traj_rounds_table.py`; `!` marks a regression, and a round
PASSES a rig only with no regression and at least three strict improvements.

### Round 1 — two OU modes per control mode

Design: `v_i` = sum of two independent OU processes; per-flight offsets in MODE
space; Whittle band 0.02-5 Hz on michaels and dregon (the telemetry-trustworthy
band) and 0.02-40 Hz elsewhere; estimator `n_rep` = 5.

| rig | overall_mean (rev/s) | rotor_mean (rev/s) | rotor_var (log) | acf | xcorr | strict | verdict |
|---|---|---|---|---|---|---|---|
| michaels | 0.1172 -> 1.0590 ! | 0.2015 -> 1.0638 ! | 0.4349 -> 0.5240 ! | 0.0613 -> 0.0928 ! | 0.1746 -> 0.1142 | 1/5 | FAIL |
| dregon | 0.1325 -> 0.1084 | 0.1554 -> 0.1250 | 0.3366 -> 0.4764 ! | 0.0646 -> 0.0865 ! | 0.1725 -> 0.2987 ! | 2/5 | FAIL |
| neurobem_quad | 0.7246 -> 26.7906 ! | 0.7751 -> 26.8488 ! | 1.1845 -> 0.5425 | 0.3361 -> 0.0575 | 0.0752 -> 0.4997 ! | 2/5 | FAIL |
| pitcn_quad | 0.2067 -> 0.1870 | 0.2188 -> 0.3538 ! | 0.2592 -> 0.2474 | 0.1154 -> 0.2487 ! | 0.2313 -> 0.1243 | 3/5 | FAIL |
| nanobench_cf21b | 0.2109 -> 0.3494 ! | 0.3033 -> 0.3574 ! | 0.1076 -> 0.1932 ! | 0.2004 -> 0.3914 ! | 0.0721 -> 0.0489 | 1/5 | FAIL |
| vid_m100 | 0.0075 -> 0.0236 ! | 0.0956 -> 0.5044 ! | 0.3147 -> 0.3115 | 0.1451 -> 0.0666 | 0.5221 -> 0.1328 | 3/5 | FAIL |
| blackbird_quad | 0.2919 -> 3.1655 ! | 0.6674 -> 3.3573 ! | 0.1387 -> 0.6908 ! | 0.1531 -> 0.4445 ! | 0.6818 -> 0.1948 | 1/5 | FAIL |

0/7 PASS (no family may regress and >= 3 must strictly improve; `!` marks a
regression). Estimator: `n_rep` 5, seed 0, 100 Hz grid.

Three diagnosed causes:

1. **Band against statistics.** The fit optimised 0.02-5 Hz on michaels and
   dregon while the judge scores the FULL band. Power the fit never saw was free
   to be wrong, and `rotor_var` and `acf` paid for it.
2. **Equal per-rotor variance.** Offsets drawn in MODE space give every rotor
   the same between-flight variance, which contradicts michaels directly (rotor
   0 carries 2.5x its neighbours' variance because its trim moves 17 rev/s).
   `rotor_var` cannot be matched from mode space.
3. **`tau_slow` is unidentifiable and `mu` is noisy.** The slower OU has no
   spectral evidence at the corner it wants, and with `n_rep` = 5 the pooled
   model mean carries an `s / sqrt(N)` error that reaches 26.8 rev/s on
   neurobem_quad — the `overall_mean` and `rotor_mean` columns of that row are
   estimator noise, not model error.

### Round 2 — full band, rotor-space offsets, antithetic estimator

Design changes: Whittle band 0.02-40 Hz on every rig, so the fit sees what the
judge scores; per-flight offsets moved to ROTOR space with per-rotor `s`;
`tau_slow` capped at 10 s; estimator `n_rep` = 40 with antithetic offsets.

| rig | overall_mean (rev/s) | rotor_mean (rev/s) | rotor_var (log) | acf | xcorr | strict | verdict |
|---|---|---|---|---|---|---|---|
| michaels | 0.1088 -> 0.0678 | 0.1431 -> 0.1751 ! | 0.3911 -> 0.1422 | 0.0562 -> 0.2944 ! | 0.2537 -> 0.1204 | 3/5 | FAIL |
| dregon | 0.1078 -> 0.0051 | 0.1201 -> 0.0153 | 0.3313 -> 0.4218 ! | 0.0622 -> 0.0675 ! | 0.1749 -> 0.3156 ! | 2/5 | FAIL |
| neurobem_quad | 0.6624 -> 6.9339 ! | 0.6907 -> 6.9503 ! | 1.2048 -> 0.2347 | 0.3351 -> 0.0640 | 0.0736 -> 0.5697 ! | 2/5 | FAIL |
| pitcn_quad | 0.2361 -> 0.0985 | 0.2554 -> 0.1364 | 0.2620 -> 0.1555 | 0.1164 -> 0.2500 ! | 0.2322 -> 0.1315 | 4/5 | FAIL |
| nanobench_cf21b | 0.0170 -> 0.0762 ! | 0.1455 -> 0.2075 ! | 0.0983 -> 0.1622 ! | 0.1997 -> 0.3848 ! | 0.0666 -> 0.0486 | 1/5 | FAIL |
| vid_m100 | 0.0038 -> 0.0416 ! | 0.0547 -> 0.1440 ! | 0.3372 -> 0.1834 | 0.1386 -> 0.0627 | 0.5219 -> 0.1353 | 3/5 | FAIL |
| blackbird_quad | 0.8440 -> 0.0311 | 0.9668 -> 1.0080 ! | 0.1810 -> 0.2042 ! | 0.1594 -> 0.4204 ! | 0.7014 -> 0.0929 | 2/5 | FAIL |

0/7 PASS (no family may regress and >= 3 must strictly improve; `!` marks a
regression). Estimator: `n_rep` 40, seed 0, 100 Hz grid.

Diagnosed cause, measured with `scripts/_rps_traj_diag.py`
(`results/rps_traj/diag/summary.json`, per mode and per octave): **a sum of OU
processes cannot be flat and then steeper than `f^-2`.** The measured mode
spectra are FLAT below ~0.3 Hz and then fall at -23 to -61 dB/decade. A
Lorentzian sum can never fall faster than -20 dB/decade, so asked for
plateau-then-steep with a family that cannot be steep, the MAP pushed its corner
below the fit band and bought a long straight `f^-2` line. On michaels that put
7.8x too much variance below 0.2 Hz, 2-8x too little in 0.2-5 Hz, and an ACF
decaying with a 2.5 s time constant against a real 0.25 s (visible as `acf`
0.0562 → 0.2944). Two secondary findings from the same pass: 88 % of the Whittle
bins lie in 5-40 Hz, so the likelihood's weight sits where the variance does
not, and the detrending choice (block mean, segment mean, linear) costs nothing.

### Round 3 — OU plus CAR(2) per mode

Design changes: the second OU becomes a CAR(2) oscillator
(`f0`, `zeta`, `sigma_osc`), which is `f^-4` above its corner, flat below it, and
free to be a resonance or overdamped; the michaels Whittle band is capped at
12 Hz, because the two-decade dive its spectra take at 25-30 Hz is the
interpolation null of the 29.41 Hz log rate and no rational spectrum can follow
it; bins below 2/T are dropped.

| rig | overall_mean (rev/s) | rotor_mean (rev/s) | rotor_var (log) | acf | xcorr | strict | verdict |
|---|---|---|---|---|---|---|---|
| michaels | 0.1088 -> 0.0763 | 0.1431 -> 0.1609 ! | 0.3911 -> 0.1469 | 0.0562 -> 0.0449 | 0.2537 -> 0.1242 | 4/5 | FAIL |
| dregon | 0.1078 -> 0.0065 | 0.1201 -> 0.0099 | 0.3313 -> 0.2498 | 0.0622 -> 0.1010 ! | 0.1749 -> 0.4606 ! | 3/5 | FAIL |
| neurobem_quad | 0.6624 -> 8.7158 ! | 0.6907 -> 8.7260 ! | 1.2048 -> 0.2479 | 0.3351 -> 0.1031 | 0.0736 -> 0.6379 ! | 2/5 | FAIL |
| pitcn_quad | 0.2361 -> 0.0016 | 0.2554 -> 0.1658 | 0.2620 -> 0.4438 ! | 0.1164 -> 0.2498 ! | 0.2322 -> 0.3852 ! | 2/5 | FAIL |
| nanobench_cf21b | 0.0170 -> 0.0910 ! | 0.1455 -> 0.2823 ! | 0.0983 -> 0.0831 | 0.1997 -> 0.2444 ! | 0.0666 -> 0.0878 ! | 1/5 | FAIL |
| vid_m100 | 0.0038 -> 0.0373 ! | 0.0547 -> 0.1141 ! | 0.3372 -> 0.0880 | 0.1386 -> 0.2063 ! | 0.5219 -> 0.1558 | 2/5 | FAIL |
| blackbird_quad | 0.8440 -> 0.0713 | 0.9668 -> 0.1032 | 0.1810 -> 0.1213 | 0.1594 -> 0.1560 | 0.7014 -> 0.2807 | 5/5 | **PASS** |

1/7 PASS (no family may regress and >= 3 must strictly improve; `!` marks a
regression). Estimator: `n_rep` 40, seed 0, 100 Hz grid.

**The family works where the oscillator stays in the shoulder.**
`blackbird_quad` passes 5/5 with nothing regressed. michaels is 4/5 with its
only regression in `rotor_mean` (0.1431 → 0.1609 rev/s, 0.21 % of a 78 rev/s
operating point), and its ACF now BEATS the baseline, 0.0449 against 0.0562 and
against round 2's 0.2944. Round 3 improved 26 of the 35 family discrepancies
over round 2. In-band variance per mode is within 11 % in 0.2-1 Hz on all four
michaels modes and within 30 % in 1-40 Hz, so the plateau-then-steep shape is
reproduced.

**Diagnosed cause of the remaining six failures: component role confusion.**
Nothing in the round-3 parametrisation makes the oscillator be the shoulder and
the OU the plateau.

1. **The oscillator chases per-rotor high-frequency junk.** dregon fitted
   `f0` = [10.8, 12.9, 19.6, 19.2] Hz, pitcn_quad [0.56, 14.9, 15.0, 0.52] with
   EVERY `zeta` at 0.20-0.29 (the 0.2 bound), nanobench mode 1 at
   `f0` = 12.8 Hz with `zeta` = 0.20, vid_m100 modes 1-2 at 11.9 and 12.2 Hz
   with `zeta` = 0.20. A narrow resonator up there wins many bins and leaves the
   whole 0.02-1 Hz shape to a bare OU, which is round 2's failure again, one mode
   at a time. That high-frequency content is measurement chain on exactly the
   rigs where it happens. The two rigs whose oscillators stayed in the shoulder
   are michaels (band capped at 12 Hz, `f0` 0.6-5.1 Hz, `zeta` 0.94-1.56) and
   blackbird (`f0` 0.5-2.3 Hz) — the two best rigs.
2. **The oscillator can also run to the LOWER `f0` bound and duplicate the OU.**
   neurobem_quad's pitch mode fitted `f0` = 0.06 Hz with `tau_slow` = 9.93 s,
   giving 6.6x too much variance below 0.2 Hz.
3. **`tau_slow` still parks at its bound**: michaels pitch 7.97 s, dregon yaw
   6.18 s, neurobem 9.3-10.0 s, nanobench 7.1-9.5 s, vid 4.2-7.9 s. The 2/T bin
   cut removed the leakage bins but also the only spectral EVIDENCE about
   1.6-10 s structure — for a 20 s block the lowest kept bin is 0.1 Hz, so a
   corner below that is unidentifiable by construction. Measured on simulated
   data: the fit recovered `sigma_slow^2/tau_slow` = 4.47 against a true 4.50
   while getting each factor wrong by 29 % and 68 %.
4. **On nanobench and vid the slow OU was switched off entirely**
   (`sigma_slow` ~ 1e-3 on 3 of 4 and 2 of 4 modes), so the fit found one
   component sufficient while the frozen ACF scores lags out to 10 s.

The diagnostic figures show this directly.
`results/rps_traj/diag/michaels_spectra.png` is a 2x2 log-log panel per control
mode ("real (as fitted)" against "model") in which common, roll and yaw track
the smooth model across the band while pitch starts above the real curve and
crosses it near 0.08-0.1 Hz. `results/rps_traj/diag/michaels_acf.png` is the
matching 2x2 ACF panel (real per-segment demeaned, real 20 s block demeaned,
model samples through the frozen estimator, analytic model prediction, with 1/e
marked): common and yaw agree closely at short lags, roll is modestly low, and
pitch is the one clear failure — the model stays clearly positive out to 10 s
where the real per-segment curve has reached zero. That is the `tau_slow` = 7.97 s
mode of cause 3, seen in the time domain.

One calibration was measured and NOT committed, pending review:
`mu := mu - (model retained rotor_mean - real rotor_mean)`, iterated twice,
which corrects the airborne-rule SELECTION BIAS (the rule keeps only samples
above 0.5 p90 of the model's own pure-airborne sample, dropping them from the
low side, whereas a real recording's p90 is taken over a whole flight including
ground). It drives `overall_mean` and `rotor_mean` to 0.0000 on every rig with
`rotor_var` and `acf` unchanged to three decimals, and would make michaels PASS
5/5, blackbird PASS 5/5, vid_m100 4/5 — 2/7 PASS with one rig one family short.

Retention (the fraction of sampled airborne samples the frozen rule keeps) is
0.95-0.99 on five rigs, 0.917 on vid_m100 and **0.525 on neurobem_quad**. The
neurobem `overall_mean` and `rotor_mean` columns above are that selection bias,
not a level error.

## Round 4 — Kalman likelihood

**Status: done** (7 uni-cpu jobs at commit `374c9f7d`). Design, as decided by the
user after reading the round-3 diagnosis:

- **Exact Kalman likelihood** in place of the Whittle approximation, so the
  1.6-10 s structure has evidence instead of a prior. The block-Whittle
  likelihood cannot see a corner below the lowest kept Rayleigh bin; a
  state-space likelihood evaluated on the whole airborne segment can. Blocks are
  30 s with a 5 s burn-in whose innovations are not scored, all blocks of a
  shape are one batched recursion, and the gain comes from the differentiable
  discrete Riccati fixed point.
- **Per-rotor OU measurement term** in place of the single iid white `sigma_w`.
  The high-frequency content the oscillator was chasing is per-rotor measurement
  chain (DREGON's staircase, michaels' hold-resampling, PI-TCN's 4 RPM
  quantisation), so the measurement model gets a coloured, per-rotor term and the
  oscillator is left with the shoulder.
- **Fit rate per rig** instead of one 100 Hz fit grid for all: michaels is fitted
  at 25 Hz, which is below the interpolation null of its 29.41 Hz log and removes
  the band cap workaround.
- **ESC floor and ceiling clamp**, so a sampled trajectory cannot leave the
  physically reachable rotor-speed range.

| rig | overall_mean (rev/s) | rotor_mean (rev/s) | rotor_var (log) | acf | xcorr | strict | verdict |
|---|---|---|---|---|---|---|---|
| michaels | 0.0725 -> 0.0503 | 0.1217 -> 0.1065 | 0.4841 -> 0.1429 | 0.0614 -> 0.0510 | 0.1352 -> 0.0468 | 5/5 | **PASS** |
| dregon | 0.1078 -> 0.0451 | 0.1201 -> 0.0483 | 0.3313 -> 0.1401 | 0.0622 -> 0.0857 ! | 0.1749 -> 0.0766 | 4/5 | FAIL |
| neurobem_quad | 0.6624 -> 3.6537 ! | 0.6907 -> 3.6936 ! | 1.2048 -> 0.1387 | 0.3351 -> 0.1550 | 0.0736 -> 0.1423 ! | 2/5 | FAIL |
| pitcn_quad | 0.2361 -> 0.0019 | 0.2554 -> 0.0503 | 0.2620 -> 0.1531 | 0.1164 -> 0.1220 ! | 0.2322 -> 0.0418 | 4/5 | FAIL |
| nanobench_cf21b | 0.0170 -> 0.0947 ! | 0.1455 -> 0.1065 | 0.0983 -> 0.1027 ! | 0.1997 -> 0.2644 ! | 0.0666 -> 0.0318 | 2/5 | FAIL |
| vid_m100 | 0.0038 -> 0.0375 ! | 0.0547 -> 0.1125 ! | 0.3372 -> 0.1271 | 0.1386 -> 0.1702 ! | 0.5219 -> 0.1702 | 2/5 | FAIL |
| blackbird_quad | 0.8440 -> 0.3235 | 0.9668 -> 0.3387 | 0.1810 -> 0.2185 ! | 0.1594 -> 0.1004 | 0.7014 -> 0.0949 | 4/5 | FAIL |

1/7 PASS (no family may regress and >= 3 must strictly improve; `!` marks a regression). Estimator: `n_rep` 40, seed 0, 100 Hz grid.

**Read.** The exact likelihood is what the model needed: michaels PASSES all
five families, and the term that did it is the coloured measurement process —
its `xcorr` fell from the baseline's 0.135 to 0.047 because the per-rotor
incoherent junk no longer has to be absorbed by the shaft modes, and
`rotor_mean` beat the baseline with no mean calibration at all. Four rigs
(dregon, pitcn_quad, blackbird_quad) reach 4/5. The one systematic defect left
is the mean families on the three rigs with large between-flight level
differences, and the airborne-rule retention column of
`results/rps_traj/rounds/round4.json` names the mechanism: neurobem keeps only
68.6 % of its sampled rotor samples, because a per-rotor offset of ~41 rev/s on
all four rotors manufactures rotor spreads no real flight has, and the frozen
rule drops those samples from the low side only.

Per-rig cost (uni-cpu, one job per rig): michaels 240 s / 468 iterations,
nanobench 291 s, vid_m100 582 s, neurobem 979 s, pitcn 1546 s, dregon 1937 s,
blackbird ~4400 s. The gradient is a hand-written adjoint rather than autograd
through the recursion, which is a measured 19x: autograd's backward was 94 % of
the cost (18.2 s of a 19.4 s gradient on neurobem) at ~160 us of graph-traversal
overhead per time step.

## Round 5 — the per-flight offset splits in two

**Status: done** (laptop rescore; the DYNAMICS ARE THE ROUND-4 FITS, only the
offset model changed, which is why `rounds/round5.json` carries
`"rescored": true` and `dynamics_from`). `delta_flight = 1 c + r` with
`c ~ N(0, s_c^2)` common to all four rotors and `r ~ N(0, diag(s_r^2))` per
rotor, both moment-estimated from the population covariance of the per-flight
airborne rotor means: `s_c^2` is the mean of its six off-diagonal entries
(floored at 0) and `s_r^2 = max(diag - s_c^2, 0)`.

| rig | overall_mean (rev/s) | rotor_mean (rev/s) | rotor_var (log) | acf | xcorr | strict | verdict |
|---|---|---|---|---|---|---|---|
| michaels | 0.0725 -> 0.1856 ! | 0.1217 -> 0.2024 ! | 0.4841 -> 0.2349 | 0.0614 -> 0.0508 | 0.1352 -> 0.0473 | 3/5 | FAIL |
| dregon | 0.1078 -> 0.0369 | 0.1201 -> 0.0433 | 0.3313 -> 0.0946 | 0.0622 -> 0.0857 ! | 0.1749 -> 0.0765 | 4/5 | FAIL |
| neurobem_quad | 0.6624 -> 1.1082 ! | 0.6907 -> 1.1103 ! | 1.2048 -> 0.0643 | 0.3351 -> 0.1076 | 0.0736 -> 0.0977 ! | 2/5 | FAIL |
| pitcn_quad | 0.2361 -> 0.0571 | 0.2554 -> 0.0668 | 0.2620 -> 0.1244 | 0.1164 -> 0.1220 ! | 0.2322 -> 0.0418 | 4/5 | FAIL |
| nanobench_cf21b | 0.0170 -> 0.1027 ! | 0.1455 -> 0.1088 | 0.0983 -> 0.1014 ! | 0.1997 -> 0.2644 ! | 0.0666 -> 0.0318 | 2/5 | FAIL |
| vid_m100 | 0.0038 -> 0.0190 ! | 0.0547 -> 0.0458 | 0.3372 -> 0.0793 | 0.1386 -> 0.1702 ! | 0.5219 -> 0.1702 | 3/5 | FAIL |
| blackbird_quad | 0.8440 -> 0.3235 | 0.9668 -> 0.3387 | 0.1810 -> 0.2185 ! | 0.1594 -> 0.1004 | 0.7014 -> 0.0949 | 4/5 | FAIL |

0/7 PASS (no family may regress and >= 3 must strictly improve; `!` marks a regression). Estimator: `n_rep` 40, seed 0, 100 Hz grid.

**Read.** The split does exactly what it was meant to do where the diagnosis
pointed, and costs elsewhere. neurobem's retention rises 0.686 -> 0.873, its
`overall_mean` falls 3.65 -> 1.11, its `rotor_var` 0.139 -> 0.064 and its `acf`
0.155 -> 0.108 — its offset is 42.1 rev/s common with only 0-10.7 rev/s
per-rotor, i.e. it really was one level difference between gentle and aggressive
flights. dregon, pitcn_quad and vid_m100 also improve. But michaels LOSES its
round-4 PASS: its offset splits as 4.4 rev/s common plus [5.5, 0, 2.0, 0]
per rotor, and forcing part of its rotor-0 excess through a common term costs
both mean families (0.050 -> 0.186 and 0.107 -> 0.202) while `rotor_var` gets
worse too (0.143 -> 0.235). So the honest summary of round 5 is: the right fix
for the rigs whose offset is common-mode, the wrong one for the rig whose offset
is genuinely one rotor's trim. A per-rig choice between the two offset models —
or a full offset covariance — is the obvious next step, and it needs no refit of
the dynamics.

## Rig posterior

`src/experiments/rps_traj/posterior.py` fits a distribution OVER RIGS, so a
training stream can draw "a plausible drone" rather than one of the fitted
seven. Seven rigs is not much to learn a 32-dimensional distribution from, so
the density is deliberately the simplest thing that can be sampled: a DIAGONAL
Gaussian. What makes it defensible is the reparametrisation, not the density —
`rig_vector` makes every coordinate dimensionless:

- `log mean(mu)` — the one scale coordinate;
- three RELATIVE trims `(M^T mu / 4)[1:] / mean(mu)`, i.e. roll, pitch and yaw
  trim as a fraction of the hover level;
- `theta`, `u_slow`, `log f0` and `logit zeta`, already dimensionless;
- every `sigma` and the per-rotor offset `s` as `log(sigma / mean(mu))`, i.e. as
  a relative fluctuation.

Sampling maps back through the same transform, so a draw is "a drone this big,
fluctuating this much RELATIVE to its size". That matters because the rigs differ
in absolute size by 3.6x: a Gaussian over raw parameters would mostly encode how
big the drone is, and a draw combining one rig's hover level with another's
absolute jitter would be nonsense. Four invariants survive the round trip by
construction: `mu > 0`, `f0` inside its range (clipped, because that coordinate
is a plain log), `zeta` inside its range and `tau_slow` inside
`(1/(2 pi f0), 10 s)` (both by their sigmoid parametrisations, so no clipping is
needed however wide the Gaussian gets).

Artefacts: `results/rps_traj/posterior.json` (round 3: `rig_posterior`, 30
coordinates, fitted on the 7 rigs, `|z|max` 1.83-2.38) and
`results/rps_traj/draws/0..11.json` plus `draws.json` — 12 draws, each validated
over 60 s at 100 Hz with every rotor strictly positive (4 redraws in total),
drawn hover levels 67-573 rev/s. Re-run
`python scripts/rps_traj_posterior.py` after a round; the dimension and the rig
list are printed by `python scripts/_rps_traj_rounds_table.py`.

**Read this as a sampler, not as a population.** It is fitted on SEVEN rigs, six
of which are one airframe each with one flight-controller tune; the diagonal
Gaussian has no covariance structure between coordinates that the rigs
demonstrably share (a rig's four modes are not independent); and two of the seven
contribute one and four flights.

## Conclusion so far

The exploration answered the question the campaign opened with: real rotor-speed
trajectories are a two-time-scale, collective/yaw-dominated, heavy-tailed,
optionally resonant process, and the incumbent synthesiser reproduces none of
those five properties even at its best fitted parameters. The frozen judge and
the fitted baseline make "better" a decidable question, and three rounds have
narrowed the model class by elimination:

- a sum of OU processes is excluded by the measured spectral slope (round 2);
- OU plus CAR(2) reaches the measured shape and PASSES where the oscillator
  stays in the shoulder — blackbird 5/5, michaels 4/5 with an ACF better than
  the baseline's (round 3);
- the remaining failures are one mechanism, not six: nothing assigns the
  oscillator to the shoulder and the OU to the plateau, so on rigs with loud
  per-rotor measurement chains the oscillator goes hunting at 10-20 Hz and the
  slow structure falls back to a bare OU whose time constant is not identifiable
  from block-Whittle bins.

Round 4 attacks exactly those two levers (a likelihood that can see slow
structure, a measurement term that can absorb high-frequency chain artefacts).

### Open questions

- **Heavy tails.** The model is Gaussian throughout, while every series in the
  corpus has excess kurtosis 2.7-46 at the lags we can measure. The frozen five
  families are all second-order, so they cannot see this: a model can pass and
  still produce increments that are too well behaved. Either the judge gains a
  tail family or the innovation gains a mixing/stochastic-volatility layer, and
  the choice should be made deliberately rather than by default.
- **A regime layer.** Everything above is fitted on AIRBORNE, roughly stationary
  segments. The transfer pair says the residual sim-to-real gap lives in the
  TRANSITIONS, and the incumbent's full-flight envelope (ground, spin-up,
  warm-up idle, take-off, cruise, landing, spin-down) is the part of it that is
  "loosely calibrated". A stationary cruise model, however good, does not close
  that gap on its own. `ground_level` already measures the real idle plateau per
  flight (michaels 8.2, dregon 16.0, vid_m100 12.7, blackbird 71.9 rev/s;
  neurobem, pitcn and nanobench mostly start airborne).
- **Single-flight and few-flight rigs.** blackbird_quad is ONE 199 s flight and
  its per-flight offset `s` fitted to exactly zero, because a single flight
  carries no between-flight information; michaels and vid_m100 have four flights
  each. A rig's `s` below `S_MIN_FLIGHTS` = 3 flights is switched off entirely.
  The one rig that passes 5/5 is therefore also the one with the least
  information, and that should not be over-read.
- **Measurement chains differ per rig, and the model has one term for them.**
  DREGON is a 45 Hz staircase quantised to 0.30 rev/s, michaels a 17 Hz
  asynchronous update resampled at 29.41 Hz with no anti-alias filter, PI-TCN
  4 RPM fixed point, NanoBench eRPM/6, Blackbird an optical encoder. Round 4's
  per-rotor OU measurement term is the first attempt to model that rather than
  to band-limit around it.
- **Two rigs are commanded, not measured.** DREGON room2 has `command` only, and
  the campaign's own reading is to trust `command` below 5 Hz and claim nothing
  above it. The only clean resonance in the corpus lives in that channel.

### What this does NOT change

**The training sampler is untouched.** `src/data_processing/rps_synthesis.py`,
the online-mix policies and every `rig_*` stream still draw trajectories exactly
as before. Nothing in `src/experiments/rps_traj/` is wired into training, and no
`conf/online_mix/*` or `conf/experiment/*` file was modified by this campaign.
The new model has PASSED on 1 of 7 rigs (2 of 7 with an uncommitted mean
calibration), which is not a mandate to replace the incumbent; the decision to
route a fitted trajectory model into the streams waits on review, and on the
round-4 verdict. `baseline.py` fits the incumbent but never edits it.
