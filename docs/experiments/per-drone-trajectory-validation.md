# Per-drone motor-speed generation: fixed training and validation sets

**Status:** evaluated, 2026-11-23. [Interactive validation traces and scores](../../writing/reports/2026-11-23_per-drone-validation/index.html) · [Quarto source](../../writing/reports/2026-11-23_per-drone-validation/index.qmd) · [earlier development and confirmation](event-clock-response.md).

## Question

How does the same regulated whole-flight law compare with the old chunkwise template and VAR generators when **each of the seven measured four-rotor datasets** supplies its own training and validation subset? A result on just MD2, VID and DREGON does not answer that question. This run changes the evaluation population, not the fitted family or scoring rule to make it win.

## Split and coverage

| Drone | Train recording IDs (count) | Validation recording IDs (count) | Validation tracks | Records with a 16 s cruise score |
|---|---:|---:|---:|---:|
| MD2 | 2 | 2 | 2 | 2 |
| DREGON | 4 | 1 | 1 | 1 |
| VID M100 | 4 | 2 | 2 | 2 |
| Blackbird | 1 early span | 1 later span, same flight | 1 | 1 |
| NeuroBEM | 76 | 19 | 50 | 11 |
| PI-TCN | 54 | 14 | 14 | 11 |
| NanoBench CF2.1B | 12 | 3 | 3 | 3 |

- MD2 uses the source split: FLY124/125 train, FLY103/108 validate. DREGON uses only measured shaft speeds for all arms, withholding `free-flight_nosource_room1`. No command reconstruction trains a competing model.
- VID trains on all four original flights and validates on the two additional circular flights. This is a **different training set**, not a new independent test after the earlier prospective comparison. NanoBench uses its publisher's split: chirp/random/square train; the three melon runs validate.
- The 20% splits for DREGON, NeuroBEM and PI-TCN are seeded on physical IDs before checking landmark or spectral support. All NeuroBEM segments of a recording stay together; no train/validation window shuffling. Only one Blackbird recording is available: train on its first 146.52 s, omit 16 s, and validate on its last 46.85 s. Rotor permutation is inferred from its training span alone. Blackbird references are disjoint training windows, not independent flights.
- Four rigs do not supply the complete ground-to-ground phase panel. Unseen phases are not assigned a fictitious score. Single-motor VID motor tests do not define a joint four-rotor process and Pelican's uncalibrated autopilot units are not shaft rev/s. All seven eligible datasets are included.

## Matched fits and scoring

Six fits per rig: regulated bridge, template with two-level switching cruise, phase VAR(2), phase VAR(4), phase VAR(2) with three-state MS-VAR cruise, and phase VAR(2) with three-state VAR-HSMM cruise. Independent fits, not previously fitted all-data checkpoints. A worker receives only its training waveforms and the target schedules; the evaluator retains the validation waveforms. All arms report a digest of the same packed training data and the same ordered training IDs. The bridge receives motors-on/takeoff/landing only. Historical phase-VAR arms also receive finer **timing-only** landmarks; they never receive target rotor levels. For records that open airborne, there is no synthetic takeoff: the plain VAR begins with its fitted stationary lag distribution, and the bridge begins in cruise.

At 25 Hz, the same 32 named draws per target supply the frozen joint-path/excursion energy score on short phases and mean-speed plus complex cross-spectrum score on settled cruise. Reference scales use the rig's training donors, at most one donor per physical recording; cruise feature calibration uses only training recordings of other rigs. Aggregate origins within segment, segments within recording, recordings within phase, and phases within rig. Scores on a one-phase rig and a five-phase rig answer different questions.

An additional **raw RMS path energy score in rev/s** covers validation tracks too short for a 16 s spectrum. It uses identical finite times across all six methods, reports unsupported time explicitly, and is not a replacement for the spectral metric. In the interactive plots every target is shown, using the first draw with matched axes and no best-draw selection. A blank part of a baseline trace denotes a phase not estimable from its training records.

Reproduction, with numerical libraries limited to one thread per process:

```bash
R=results/rps_traj/per_drone
python -m experiments.rps_traj.per_drone_data prepare --root "$R"
# On university CPUs: pass only pack/*.train.{json,npz} to this command.
python -m experiments.rps_traj.per_drone_run --root "$R" --jobs 4
# After pulling fitted models and validation draws:
python -m experiments.rps_traj.per_drone_data score --root "$R"
python -m experiments.rps_traj.per_drone_paths --root "$R"
make -C writing/reports/2026-11-23_per-drone-validation
```

## Results

The regulated bridge had the lowest phase-balanced validation score on **4/7** drones and beat the old template on **5/7**. It did not consistently beat all baselines. Lower is better; the first three rows average five observed phases. The other rows score cruise only.

| Drone | Bridge | Template | Phase VAR(2) | Phase VAR(4) | MS-VAR hybrid | VAR-HSMM hybrid |
|---|---:|---:|---:|---:|---:|---:|
| MD2 | **1.266** | 1.476 | 3.254 | 4.048 | 3.245 | 3.245 |
| VID M100 | **0.375** | 0.600 | 0.548 | 0.538 | 0.542 | 0.531 |
| DREGON | **0.590** | 1.613 | 1.066 | 1.731 | 1.064 | 1.063 |
| NeuroBEM | **0.791** | 0.819 | 2.920 | 1.808 | 0.875 | 0.876 |
| PI-TCN | 0.461 | 0.481 | 0.461 | 0.472 | 0.450 | **0.443** |
| NanoBench | 2.435 | 1.748 | 2.483 | 2.105 | 1.473 | **1.431** |
| Blackbird | 0.467 | 0.440 | 0.458 | **0.391** | 0.753 | 0.652 |

The bridge won four of five phase means on MD2 (not standby) and DREGON (not cruise), and two on VID (takeoff and cruise). On Blackbird the training collective is not itself low (184.0 rev/s median in the central half of training), but the fitted cruise equilibrium is 125.6 rev/s. The middle validation window averages 184.2 rev/s measured versus a median 128.9 over the 32 generated bridge paths. On VID the first loaded-round draw contains much larger opposing motor excursions than that measured trajectory. A low ensemble score does not imply that every draw resembles the individual observation.

The all-trace diagnostic changes some rankings: the bridge has the lowest RMS path energy on DREGON and NeuroBEM, but not on the other five rigs. Blackbird has 1047/1171 common samples; its early training span did not supply a landing law for the old template. All other rigs have full time support. The primary score has 306 eligible recording/phase/method rows, with zero failed rows; the supplementary score includes all 73 segments from the 42 assigned validation recording IDs/spans.

For the three five-phase rigs the bridge stores 256–276 coefficients versus 1,934–3,140 values for the template. On Blackbird, PI-TCN and NanoBench the old template fits no transition curves and stores only 35 values; there the bridge is **not** the smaller model. The report includes all six sizes per rig.

University runs: `drone-prepare-804e2f2-cea8d4` (split and calibration), `drone-fit-b1657bd-cca4e9` (42/42 fits). Exact tables: `results/rps_traj/per_drone/{phase_scores.csv,whole_path_scores.csv}`. Recording IDs, hashes and exclusions: `pack/manifest.json` and `pack/*.valid.json`; 32-draw ensembles and fit provenance: `runs/<rig>/`. In the [seven-drone overview](../../writing/reports/2026-11-23_per-drone-validation/assets/every-drone.png), each row is the first validation ID at least 16 s long, not a selected successful draw.

## Follow-on hypothesis: separate the operating level from its fluctuations

Protocol fixed before fitting `lifting_control`. This is an **exploratory** comparison: the preceding validation results are known. Do not call another win on these same recordings an independent replication. The family is fitted solely on the existing `.train.npz` files, on university CPUs. No predictor, probability, phase clock or variance is chosen from an individual validation recording. No per-drone or per-phase selection between model families.

**Training evidence.** Blackbird training has one simultaneous arming/takeoff and no landing, yet the regulated law stores 236 coefficients; PI-TCN and NanoBench training have no arming or landing and store 276. The Blackbird training median collective in settled flight is about 182 rev/s, far above the previous fit's equilibrium. Across rigs, training detail is neither white nor confined to one timescale: median lag-one correlations in the four control coordinates range from 0.76 to 0.99; the most prominent training frequencies in 128-sample Welch windows are about 0.2–0.8 Hz. These measurements motivate modeling level and band dynamics, not changing the evaluation metric.

**One candidate.** Represent each commanded phase by its exact arithmetic mean and invertible multiscale deviations from predictions based on neighboring coarse means. Fit the event-aware reference, persistent flight heterogeneity, cross-mode covariance, scale-dependent fluctuations and signed two-stage maneuver grammar in one complete-flight transformed-data likelihood. An unsupported transition gets no stored empirical curve. In an airborne-only law the planned maximum is 34 independent learned numbers: four means, sixteen log-scale coefficients, six correlations, at most two between-flight level variances, one persistent activity multiplier, three dwell/branch probabilities, one active-noise multiplier and one signed-drive amplitude. Count any additional learned event or clock values on other rigs. Compare complete exported parameter counts, not just the coefficients active in a chosen window.

**Decision rule.** Retain the old 42 fits and the previous scores unchanged. Fit the new law once on each of the same seven training sets; verify parameter and trajectory replay, and report training-only free-running operating levels. Compare all 19 supported rig/phase means on identical validation recordings and windows against the strongest of the previous six methods in that cell. Separately report numerical wins, compactness versus the smallest comparator per rig, and physical/visual checks. Better pooled means or selectively combining methods do not satisfy the requested all-regime criterion. A failed numerical or compactness gate disqualifies promotion; no preference study can rescue it.

For context, replacing the generated ensemble by actual training-donor windows on these splits is better than the strongest previous fit in 14/19 cells, worse in five (MD2 startup/takeoff/landing, VID standby, NanoBench cruise). At NanoBench the donor reference scores 1.502 versus 1.431 for VAR-HSMM; at Blackbird it scores 0.310 versus 0.391 for VAR(4). This calculation uses validation observations for *scoring*, not fitting. It is a reference distribution, not a lower bound or a permissible compact generator.

**Implementation check before opening the successor's validation scores.** In a synthetic example with 140–150 rev/s cruise the first event design placed its hover level near 58 rev/s. Although each root was an exact phase mean, the *same coefficient* still set the steep transient slope. Details then dominated the three observed means. For repeated events we separated constant idle/airborne levels from the fitted transient deviations (up to eight additional coefficients). For sparse events we conditioned the edge shape on the realized phase mean: subtract the known edge's detail-to-root ratio times the observed root on analysis, add it back on synthesis. This is another determinant-one triangular transform, with no fitted coefficients. It prevents one fast edge from re-estimating the operating level. No correction was selected using the successor's validation scores.

The physical shaft is exactly stopped during a commanded-off interval, while raw off telemetry enters the objective through a fixed 1 rev/s sensor error law. For powered samples, the density is on floored inverse-softplus mixer coordinates, not an exact censored raw-speed likelihood. On a 65-sample case with both flight styles and three maneuver blocks, explicit enumeration and the dynamic program agreed in log density to $6\times10^{-14}$.

**Result: reject `lifting_control` as the replacement.** It is smaller than every comparator on all seven rigs (31–205 learned scalars), but wins only **1/19** matched rig/phase cells against the previous best: DREGON takeoff 0.730 vs 0.759. Corrected Blackbird training mean speed is 179.4 rev/s generated vs 181.6 measured (eight draws), and its full-path error improves to 13.87 from 37.00 rev/s for the regulated bridge. Nevertheless its cruise spectral score is 0.752 vs 0.391 for phase VAR(4). Centering fixes a real physical error, not the multiscale distribution as a whole.

Against the user-requested five historical comparators alone (excluding the regulated predecessor), it wins **5/19** cells: MD2 cruise, VID takeoff and cruise, and DREGON standby and takeoff. The regulated predecessor wins 11/19 against those same historical methods. The 1/19 figure above is the stricter successor-promotion gate, not a redefinition of the original comparator set.

| Rig | Lifting coefficients | Lifting phase-balanced score | Lowest previous phase-balanced score |
|---|---:|---:|---:|
| MD2 | 205 | 2.457 | 1.266 (regulated bridge) |
| VID M100 | 203 | 1.078 | 0.375 (regulated bridge) |
| DREGON | 171 | 1.012 | 0.590 (regulated bridge) |
| NeuroBEM | 34 | 2.530 | 0.791 (regulated bridge) |
| PI-TCN | 34 | 0.463 | 0.443 (VAR-HSMM hybrid) |
| NanoBench | 34 | 1.527 | 1.431 (VAR-HSMM hybrid) |
| Blackbird | 31 | 0.752 | 0.391 (phase VAR(4)) |

The lifting law beats the regulated bridge on NanoBench's cruise score (1.527 vs 2.435), but does not beat the VAR-HSMM hybrid. On NeuroBEM 210/1,600 draws exceed 1.25 times the training maximum, reaching 1,118 rev/s. The [matched first draws](../../results/rps_traj/per_drone/trials/lifting_control/every-drone.png) show the level correction on Blackbird, a calmer VID cruise, and a large late NeuroBEM excursion. No visual superiority is claimed. [All per-phase scores](../../results/rps_traj/per_drone/trials/lifting_control/comparison.csv) · [whole-path diagnostic](../../results/rps_traj/per_drone/trials/lifting_control/whole_path_scores.csv) · [parameter counts, envelopes and paired results](../../results/rps_traj/per_drone/trials/lifting_control/acceptance.json).

Reproduction: `per_drone_run --arms lifting_control` on the locked training packs, then `python -m experiments.rps_traj.per_drone_trial`. Jobs: `drone-fit-0a98445-eead9e` (unchanged all-airborne PI-TCN/NanoBench fits), `drone-fit-4ba6397-b00f90` (separated level/shape, three full-flight rigs), `drone-fit-108f9cf-949bcd` (root-conditioned edge, Blackbird and NeuroBEM). The unchanged old fit ensemble and its 51 recording/phase scores replayed identically; only the candidate was newly evaluated for the primary score.
