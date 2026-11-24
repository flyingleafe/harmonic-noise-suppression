# Unified whole-flight motor-speed response

**Status:** tested, 2026-11-22 — smaller unified likelihood model; **all-regime superiority rejected on this panel**. This is a trajectory experiment, not an acoustic or flight-control experiment.

## Question and frozen decision rule

Does sharing a small set of motor responses across start-up, standby, take-off, cruise and landing replace the rig-specific sampled templates **without losing any regime**? Does a manoeuvre parent that controls the order of reusable cruise actions help relative to a flat action process? The experiment is a flight-held-out development comparison, not a blind test of new rigs.

* Fit once per training fold to whole recordings at 25 Hz; the conditioning input exposes only motors-on, take-off, landing and recording length. No test idle plateau, arrival, cruise level, fitted transition curve, action label or test waveform is a model input. Preserve motor-speed history across schedule boundaries. Compare the hierarchy to a flat version with the same local response family and flight-anchor model.
* Reuse the rig splits, 13 measured targets, 32 free-running draws, support windows, train-only scale factors, seeds and proper ensemble scores from [the fixed-schedule comparison](rps-flight-regimes.md#held-out-comparison-on-the-commanded-phase-schedule). Score all five phases. In cruise, use the absolute four motor means and complex cross-spectrum, not the visualization or a one-step likelihood. Keep DREGON command-only recordings in the original training pool if they have a usable coarse schedule; never use them as measured targets. Report any exclusions and failures.
* A win **in every regime** means the hierarchy's mean score in each of the five phases is strictly below the best of the template, both phase VARs, the MS-VAR hybrid and the VAR-HSMM hybrid. Give each rig equal weight; a lower average over phases is not sufficient. Report the per-phase rival and paired differences. Do not tune on the test scores, change the folds, or select the nicest draw.
* Count learned scalars, separately noting stored nonparametric template samples, derived quantities and fixed structural choices. Thousands of smoothed coordinates do not imply thousands of independent statistical degrees of freedom. The new model must use fewer total fitted coefficients, not just a smaller serialized file.
* Visual endpoint: use draw 0 on the published support and include the whole comparison panel, including failures. Evaluate mean speed, rotor spread, transition shape/timing and cruise variability on a common scale. A visual example or an unblinded author inspection is not independent perceptual validation.

## Model and estimator

At each 25 Hz sample the observed signed speeds are rotated into collective, roll-, pitch- and yaw-like speed coordinates. Velocity increments follow a stable, second-order diagonal autoregression toward an action-dependent target. The four momentum and four restoring coefficients are shared across all phases. There are 13 action labels (including rest and idle), with 13 × 4 disturbance SDs, nine free four-mode target offsets, and two latent pairs of idle and hover vectors shared across the whole recording. These are **two population components**, not one parameter per training flight. They are initialized on training data, then jointly updated along with their mixture weight, response and grammar. No sampled waveform is a fitted parameter.

The supplied schedule consists of rest, motors-on to take-off, take-off to landing, and rundown. At a schedule change only the discrete grammar restarts; the response and its velocity continue. Start-up can pulse then release, repeat a pulse or settle into idle (absorbing until take-off). In airborne cruise the ordered graph can hold, issue an isolated signed drive, or alternate signed drives; these parent contexts reuse the same drive and hold emissions. The flat graph keeps the same action repertoire, emission law and duration family but removes the parent-specific successor rules. Both use negative-binomial dwell laws via tied geometric clock stages, not one free probability per possible duration.

Fit the penalized likelihood of **whole training records**, conditional on the first two speeds and the coarse event schedule. The E-step sums over action paths by sparse forward–backward in log space and over the two persistent anchor pairs by a single mixture sum per flight. The M-step updates grammar counts, anchor probabilities, disturbance variances, the shared stable response, and anchor/offset regressions; it retains improvements to the joint training objective only. Default budget: 30 generalized EM steps, single initialization, early stop at a training-only tolerance. This is an observed-speed likelihood rather than a separate latent shaft/logger Kalman model. The primary draws are from the fitted signed law, **not** feedback-clamped. Report negative speeds as a failure mode; `sample(..., shaft_only=True)` is an optional nonnegative output projection, not the likelihood evaluated here.

**Secondary diagnostic, not a selection criterion.** Compare flat and hierarchical conditional whole-flight log densities and cruise ordering on the same targets. In each 16 s cruise window, subtract the yaw-contrast mean, average over three samples, and detect signed excursions beyond the median donor RMS (floor 0.05 rev/s). Count excursion onsets and opposite-sign successors within 1 s. Subtract the expected reversal count if excursion signs were shuffled at fixed onsets and counts. Score the two rates per second with the fair ensemble energy score. No pilot action or semantic manoeuvre labels are inferred as ground truth.

The original hierarchy has 113 emission/anchor parameters + 46 graph parameters = **159**; the flat ablation has 113 + 28 = **141**. The nested two-response alternative adds eight, for **167/149** respectively. Template fits store 1,814–3,146 fitted values across these folds. The phase VAR(2) and VAR(4) use 277 and 469 coefficients including the training-derived cap; hybrid counts are 427–523 (MS-VAR), 607–1,267 (VAR-HSMM), counting the four cruise levels. Comparison is to nominal learned/stored size, not a claim about the effective degrees of freedom of the smoothed curves.

## Results

All 26 final selected fits converged by the training tolerance (29–104 steps for the hierarchical model, 22–92 for the flat one). Training BIC selected separate transient and cruise response coefficients in 8/13 hierarchical and 7/13 flat folds. No fitted model uses pointwise transition templates. All 130 new flight × phase × arm scores are finite; all 65 cell supports, ensemble sizes and donor scale factors match the published comparison. No failed fit or draw was removed.

| Generator | Start-up | Standby | Take-off | Cruise | Landing | Balanced |
|---|---:|---:|---:|---:|---:|---:|
| Unified hierarchy, training-BIC choice | 2.17 | 2.39 | 0.86 | 2.17 | 2.08 | 1.93 |
| Unified flat, same choice | 2.13 | 2.51 | 0.83 | 2.22 | 2.08 | 1.96 |
| Previous phase templates + switching cruise | 2.04 | 0.67 | 0.76 | 2.15 | 0.69 | 1.26 |
| Phase VAR(2) | 1.89 | 1.25 | 0.89 | 2.52 | 1.32 | 1.57 |
| Phase VAR(4) | 1.55 | 1.72 | 0.82 | 2.39 | 1.25 | 1.55 |
| VAR(2) + MS-VAR | 1.89 | 1.25 | 0.88 | 2.11 | 1.32 | 1.49 |
| VAR(2) + VAR-HSMM | 1.89 | 1.25 | 0.88 | 2.11 | 1.32 | 1.49 |

Lower is better. The strictly shared-response hierarchy, before the eight-coefficient extension, scored 2.24 balanced (3.71 standby). The training-selected version improves on that but beats **none of the five phase-wise best scores**. Versus the template it is 0.67 worse on the balanced score (descriptive 95% paired interval 0.33–1.20 worse; 1/13 flight wins). The hierarchy narrowly improves over its flat ablation (0.023 lower balanced, 8/13 flight wins); its mean held-out conditional log density is −0.804 vs −0.838 nats per four-mode step. The secondary order-feature energy scores are 0.257 vs 0.260. These differences do not establish recovery of physical manoeuvres. On cruise, the new hierarchy does beat both the template and the HSMM hybrid on MD2 and VID separately, but not on DREGON (1.01 vs 0.64 and 0.60). Do not replace the existing generator on these results.

**Visual and physical checks.** On the preselected MD2 FLY124, the measured start-up collective peak is 50.8 rev/s. Across 32 draws the new model's median is 24.6 (10th–90th percentile 3.6–56.9), against 48.4 (44.8–51.8) for the template. The observed maximum landing rotor range is 66.8 rev/s; new median 29.3, template 50.5. The free-running model does not reliably place the start-up peaks or reproduce the landing fan-out. Signed draws produce at least one negative rotor on 2.67% of time steps (equal-rig average); 1.35% have a rotor below −1 rev/s. The most negative draw is −201.5 rev/s (FLY103 during start-up), not a rounding artefact. As a post-hoc **readout diagnostic**, scoring the nonnegative output projection instead changes balanced score from 1.93 to 1.92 and does not win a phase. Stable response poles and better one-step likelihood are not sufficient for physically plausible free-running excursions.

**Interpretation.** A single likelihood and compact parameterization were achieved. Evidence does not support visual or metric superiority. The fit can explain short events partly by large action disturbances when conditioned on real previous speeds; duration and innovation uncertainty accumulate during generation. This is a possible mechanism, not a demonstrated unique cause.

**Post-hoc timing diagnostic, not a competing generator.** Use the held-out speeds to infer the posterior distribution of (a) its two-point anchor alone, or (b) its anchor and entire hidden action path via forward-filter/backward-sampling. Generate new AR innovations and run from rest without further access to the measured speeds. Revealing only the anchor changes the five scores by less than 0.02 apiece. Revealing both changes standby from 2.39 to 1.42 and cruise from 2.17 to 1.88, but start-up goes from 2.17 to **2.83** and landing from 2.08 to **2.03**. Even with this privileged information the transient response does not reproduce the template's landing. Changing dwell times or choosing a flight anchor alone is therefore not a sufficient repair in this diagnostic. The next question is whether a bounded, repeatable within-action drive is being represented as random acceleration; test that on independent recordings, not by optimizing these held-out scores.

Under `results/rps_traj/unified/`: `comparison.json` (rig, phase, component, paired bootstrap and diagnostic results), `unified_models.json` (130 scored cells and provenance), `fits/`, `draws/`, `example_FLY124.png`, `flight_comparison.html` (13 flights × 5 phases, eight panels including measured, method names masked until revealed). Source modules: `experiments.rps_traj.unified_{grammar,response,experiment,report,diagnostics}`. The focused response/grammar suite passed 26 tests, including exact likelihood/posterior checks, stability, latent-anchor persistence, generation parity and an ordered-versus-flat synthetic test.

## Compute

One seed. Heavy fits belong on university CPU compute nodes, not the laptop or login nodes. Inspect convergence from training objectives; the held-out phase score never selects a checkpoint. Results go to `results/rps_traj/unified/` separately from the frozen comparison.

The initial university job `unified-flight-58e43bb8-1d4439` used four CPU cores (one thread per fit; no GPUs) on `uni-cpu`. Environment from pushed commit `befb85918039`; worker modules and fold packs archived by SHA-256 under `artifacts/unified-flight-response/worker/` in the project artifact store. Workers read only their own fold's training speeds, not that fold's held-out speed values. Evaluation NPZs and reference scores were not staged. The three `submission*.json` records include commands and exact bundle hashes. Baseline score SHA-256 begins `e53e3b41b33c5412`.

**Training-only decisions before inspecting held-out scores.** All 13 hierarchical fits hit the initial 30-step cap. A second job (`unified-flight-9f211f9c-58dbe1`) resumed both grammars with identical optimizer settings and a 300-step total cap; all 26 shared-response fits reached their training tolerance in 22–109 steps. The original model is retained as an ablation. In the initial VID fold 0 hierarchy there was little posterior evidence for the idle action (2 effective samples) and the idle anchor was about 71 rev/s; other start-up action offsets compensated. We therefore allowed a single additional set of four restoring and four momentum coefficients, shared by **all actions except settled-cruise hold and the two signed drives**. No new target curves or phase-by-phase estimators. Before examining the shared-response test scores, launched both flat and ordered fits with this alternative, selecting it per fold only if it lowered `−2 × training marginal log likelihood + k log(number of valid time steps)`. This is a training complexity criterion, not model evidence for the number of physical states. The second candidate starts with a faster transient attractor and is otherwise fitted by the same whole-flight EM. Final job: `unified-flight-e782d8b5-bfdd51`.

Reproduction (fit commands run on a compute node):

```bash
export PYTHONPATH="$PWD/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
python -m experiments.rps_traj.unified_experiment pack
python -m experiments.rps_traj.unified_report --baseline-traces-only
for rig in michaels vid_m100 dregon; do
  for pack in results/rps_traj/unified/pack/"${rig}"__*.train.npz; do
    fold=${pack##*__}; fold=${fold%.train.npz}
    for arm in hierarchical flat; do
      python -m experiments.rps_traj.unified_experiment fit --rig "$rig" --fold "$fold" --arm "$arm" --max-iter 300 --select-response
    done
  done
done
python -m experiments.rps_traj.unified_experiment collect
MPLBACKEND=Agg python -m experiments.rps_traj.unified_report
python -m experiments.rps_traj.unified_oracle  # optional, uses test speeds; not a model score
```
