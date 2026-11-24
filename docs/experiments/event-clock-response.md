# Event-driven whole-flight response

**Status:** regulated bridge frozen on 2026-11-23 after passing all five development phase means. This continues [the unified-response experiment](unified-flight-response.md). Previous models and plots were inspected on the same evaluation panel.

## Test

Does a time-varying drive within a commanded event fix the previous model's free-running transition failures without giving up a compact, fitted whole-flight probability law? The first hierarchy reached 1.93 balanced versus 1.26 for the template. Even revealing its inferred action sequence did not rescue start-up/landing. Here the coarse schedule sets the event clock but does not give the target's arrival, idle speed, overshoot, peak or cruise level.

Two new families are investigated: a functional mixed-effects clock response with an autoregressive residual, and an event-driven hidden semi-Markov programme with persistent flight-level effects. Use the same raw four-channel training records, with measured versus command-only source recorded. Any source conditioning or special treatment of command-only observations must be disclosed. No fit uses held-out motor speeds. Both models must sample physical nonnegative speeds and learn the time course under a single whole-flight likelihood, not a separately frozen estimator for each chunk. Count every learned parameter (including candidate-specific duration or scale parameters); a stored donor trajectory does not count as a compact generator.

Keep the 13 measured target flights, original 65 phase cells, all 32 draws and the exact existing training-only normalizers, equal rig/phase weights and baseline file hashes. A candidate must individually beat every historical method on **each** phase, not a per-phase oracle between candidates: start-up <1.54915338, standby <0.66970603, take-off <0.76079404, cruise <2.10572534, landing <0.69233202 (compare full precision in code). Fewer fitted values than the smallest historical comparator, 277, in every fold. Show the same preselected FLY124, draw 0, and masked comparisons on every rig and phase. A close numerical win is a descriptive development result: report uncertainty, failed candidates and inspect visually. Do not call visual plausibility independently validated without a genuine independent judgement.

## Reproduce

Use the original `results/rps_traj/unified/pack` (13 training and evaluation fold packs) and the published `schedule_eval/schedule_models.json`. Run `python -m experiments.rps_traj.clock_experiment prepare`; stage the training NPZs together with source flags and target **schedules only** to `uni-cpu`. Fit each rig/fold/family with `clock_experiment fit --rig <rig> --fold <fold> --family <family>` and collect fits/draws back into `results/rps_traj/event_response`. `clock_experiment collect` calls the original scorer, asserts matching reference scales and writes `comparison.json` plus the masked flight comparison. Numerical evidence and all trial records are kept separately from the template and first unified model.

## First two trials

The following are measured development-panel scores, not train-likelihood rankings. Smaller is better. Each entry is a single generator, never a per-phase choice among generators.

| Generator | Startup | Standby | Takeoff | Cruise | Landing | Largest fitted law |
|---|---:|---:|---:|---:|---:|---:|
| Strongest old method per column | 1.549 | 0.670 | 0.761 | 2.106 | 0.692 | — |
| Mixed-effect clock AR | **1.200** | 2.766 | **0.606** | 2.201 | 0.873 | 222 |
| Programme-response HSMM | 2.433 | 1.984 | **0.661** | 2.167 | 0.700 | 219 |

`clock-mixed-407bdf3-36c571` and `program-response-924bb9e-3fca81` each produced all thirteen fits and ensembles. The mixed-effect model's cruise score beats the VAR-HSMM hybrid on MD2 (0.531 vs 0.542) and VID (4.961 vs 5.177), but loses on DREGON (1.112 vs 0.598). Its persistent class is not in-flight switching. The programme model does support ordered within-flight switching; its near-tie on landing is **not a win**. Both use a finite floor on near-zero observations rather than an exact censored stopped-motor likelihood.

One diagnosis is directly visible in training fold 0: measured DREGON holds about 15.3 rev/s within 0.2 s of arming and waits 1.80–4.20 s for takeoff; VID climbs to about 13.3 rev/s over 2.2 s and takes off at 3.04–3.24 s. The mixed-effect model assigns the first six seconds to startup noise irrespective of settling, so neither rig gets its stationary standby variance before takeoff in these training records. This is a structural issue, not a reason to remove standby or enlarge the donor scale.

Subsequent candidates test a multiresolution autoregressive bridge with a semi-Markov cruise child, and a controller with a source-aware equilibrium and a distinct relaxation law. These are development iterations. All use the same folds and original scoring code. The historical phase-VAR arms additionally see `phase_labels(marks)` (including observed arrival); candidates see only motors-on, takeoff, landing and recording length. The stricter original benchmark is retained. An all-phase win on this panel would still need replication on new flights and independent visual judgement before a claim of superiority.

## Multiscale bridge

`bridge-program-c4a4f81-fe5381`: 265–266 fitted values; all 416 complete draws finite, minimum 0, maximum 159.85 rev/s. Balanced scores **1.385 / 2.107 / 0.613 / 2.020 / 0.851**, in phase order. It beats every old arm on startup, takeoff and cruise, not standby or landing. For cruise the paired difference against VAR-HSMM is 0.086 in the bridge's favour (within-rig flight bootstrap 95% interval 0.025–0.140; 6/13 individual flights). This is a difference in equally weighted rig means, not consistent superiority per flight. The interval does not account for model search.

In a draw-0 inspection **before opening the whole-flight key**: on MD2, C has the most convincing stationary trim and shutdown; A never quite switches off. On VID, B shows the intermittent rather than continual cruise excursions; D has the more convincing touchdown. On DREGON, C holds the plateau and shuts down cleanly; A and B surge just before stopping, D wanders during arming. These are three examples judged by the implementer, not an independent human preference study. Every historical trace on the complete plates was replayed from its fold and checked against its five archived scored windows.

The revealed key identifies the preferred MD2 and DREGON full traces as the template; VID B is the VAR-HSMM hybrid and VID D the template. The bridge is A/A/D on those original plates. This inspection does not support a visual-win claim for the bridge despite the lower average cruise distance. Future candidate comparisons use separate keyed directories to keep the original inspection auditable.

The duration-profiled bridge (`settled-bridge-7f6be9d-9d7077`, 276 fitted values) excludes command-only records, profiles the response widths on joint training likelihood and restarts the *disturbance* at clock boundaries. Scores: 2.156 / 2.161 / 0.647 / 1.993 / 0.992. DREGON standby improves but VID worsens; a single discrete variance boundary still confounds rising and steady speeds. In VID fold 0 the fit selects a 0.96 s settling response even though training telemetry continues to rise for about 2.2 s. Do not promote it. The population feedback variant instead propagates standardized disturbances across boundaries and estimates a time-varying variance profile; its time bins are fixed across rigs.

### Standardized-disturbance feedback

The population bridge fits time-dependent disturbance scales for collective and differential control, and carries its residual across command changes (no restarts). It wins overall at **1.131 versus 1.263** for the template; phase scores 0.917 / 1.149 / 0.596 / 2.000 / 0.991. It nevertheless produces a physically implausible 593 rev/s maximum. On MD2 training schedules, multiplying the old residual by a much greater touchdown scale could amplify an existing cruise fluctuation by more than 60×. This model is rejected regardless of its overall score.

The nonamplifying bridge limits the transfer of an existing *physical residual*, not the sampled speed: a noisier next state can generate new variation, but does not multiply the old fluctuation. It sets the support of its arming basis from the training **schedule lengths**, not from a separately fitted idle target. `bounded-bridge-e9fa1f1-099a79`: 274 parameters, 0–171.92 rev/s across all draws, overall **0.992**; phase scores **0.805 / 0.942 / 0.584 / 2.002 / 0.626**. It beats all baselines on four of five phase means. The landing advantage is 0.066 (descriptive 95% paired bootstrap −0.056 to 0.172); not a statistically resolved advantage. In the masked complete plates MD2 B shows believable stationary trim and a clean shutdown; DREGON A and B both hold idle and stop sharply. VID C captures intermittent mild maneuvers while D is much more active than this particular measured flight.

On the training VID fold-0 schedules this generator still produces median standby rotor spreads of about 2–3.5 rev/s, versus measured 0.3–1.1 rev/s, despite a collective mean near 13 rev/s. The transient and idle poles were tied. Test two additional fitted ground-regulation poles (one collective, one shared by contrasts), inside the same likelihood. Also test an alternative physical parameterization with total thrust proportional to the sum of squared motor speeds and a three-dimensional log-ratio torque allocation. Neither change alters the metric, sample seed, or old comparators.

## Locked model: regulated feedback bridge

Job `regulated-bridge-82b132a-4b7332`. **One model**, fitted once per fold, beats each of the five historical generators on each of the five balanced phase means. Allowing one ground pole for collective speed and one shared by the three contrasts is enough: VID fold 0 estimates 0.003 and −0.091, respectively, rather than the 0.94–0.95 transition poles. The DREGON fold-0 ground estimates are 0.174 and 0.156. MD2 estimates 0.95 for both: unlike VID, its standby trim persists. No idle level, arming waveform or transition library is fitted separately.

| Phase | Regulated bridge | Strongest historical | Historical − bridge (95% paired interval) | Flights won |
|---|---:|---:|---:|---:|
| Startup | 0.631 | 1.549, phase VAR(4) | 0.918 [0.702, 1.152] | 12/13 |
| Standby | 0.618 | 0.670, template | 0.051 [−0.301, 0.286] | 9/13 |
| Takeoff | 0.585 | 0.761, template | 0.176 [0.035, 0.325] | 9/13 |
| Cruise | 2.002 | 2.106, VAR-HSMM hybrid | 0.103 [0.030, 0.163] | 10/13 |
| Landing | 0.625 | 0.692, template | 0.068 [−0.042, 0.166] | 8/13 |

Balanced: **0.892 vs 1.263** for the template (29.3% lower). Finite nonnegative speeds in all 416 draws, maximum 108.31 / 171.90 / 99.95 rev/s on MD2 / VID / DREGON. These extrema are an audit, not certified hardware limits. The mechanism is a mixture of three flight equilibria and activity styles, with a signed two-stage dwell controller over 1.28 s cruise blocks. Causal feedback during commanded transitions and recursive midpoint prediction in cruise enter the same complete-record penalized likelihood on the floored, transformed motor measurements. EM marginalizes both the persistent style and the maneuver sequence. Optimization is capped at 80 iterations; global optimality is not assumed.

The alternative in squared-speed collective thrust and log-ratio motor allocation (`allocation-bridge-aa2c17-61e82e`, 276 values) scored 1.509 / 0.970 / 0.591 / 2.005 / 0.874; positive thrust allocation did not yield a better generator. The separately implemented variational controller (`controller_law`) optimizes an ELBO and has no complete, confirmed benchmark result. It was not used for selection after the lock.

**Masked review.** Two model-assisted reviewers saw no keys or scores. On complete draw-0 flights the first reviewer ranked the regulated bridge first on MD2 and DREGON and third on VID, behind a tie between the template and phase VAR(4). Its VID draw showed large opposing maneuvers absent in the measured indoor recording; the reviewer distinguished this from nonphysical motion. On 15 independently shuffled phase windows the second reviewer selected the bridge in 5, the template in 6.5, phase VAR(4) in 2.5 and the VAR-HSMM hybrid in 1 (a tie splits one vote). Full-trajectory plausibility improved in two examples; the window review did not favor the bridge over the template overall. This was not a human preference study and does not establish uniform visual superiority.

After the lock, all ten remaining complete draw-0 development flights were reviewed with independently shuffled methods, using the same criteria and without showing keys. Across all thirteen flights the first-place shares were **regulated bridge 6, template 4.5, phase VAR(4) 1.5, and VAR-HSMM hybrid 1**: the most first-place votes, not a majority. The bridge was first on 3/4 MD2, 1/4 VID and 2/5 DREGON. This is still a model-assisted visual check, and the three additional reviewers each saw flights on only one rig. The complete traces and votes are in `complete_review/regulated_bridge/{all_flights,whole_preferences.json}`. On the VID rig 16/128 complete draws exceeded 1.25× the fastest training rotor speed of their fold, versus 0/4 measured targets; neither MD2 nor DREGON had such a draw. The generator imposes no hardware speed cutoff. See `trials/regulated_bridge/speed_envelope.json`.

The exact numbers and 13-flight zoomable review are in `results/rps_traj/event_response/trials/regulated_bridge/{comparison.json,flight_comparison.html}`; masked full-flight plates and their key in `results/rps_traj/event_response/complete_review/regulated_bridge/`. For reconstruction run `python -m experiments.rps_traj.full_flight_review references`, then `render --families regulated_bridge`. Each legacy draw-0 path is checked against all its archived scored windows.

### Compact deployment, separate from reported fits

There are 276 independent scalars per fitted rig: 144 response/level coefficients, 30 feedback coefficients, 28 scale variances, 8 non-reference style activity factors, 8 signed-pulse/variance parameters, 9 duration/branch probabilities, 2 independent mixture masses, 46 relative error scales and 1 arming-support value derived from training schedules. The template stores 1,814–3,146 fitted numbers across folds; phase VAR(2) has 277. `RegulatedBridge.coefficients()` exports the law without optimizer traces or redundant fixed/tied values. The portable generator requires 2,208 bytes of coefficient data in float64, plus the NumPy header. The model definition and fixed knot grids are code shared by every rig.

`regulated_deploy prepare` makes training-only packs from the unique 4 MD2, 4 VID and 5 DREGON development flights. Run `clock_experiment fit --out results/rps_traj/event_response/deployment --rig <rig> --fold all --family regulated_bridge` per rig, then `regulated_deploy export`. These deployment fits are **not** used for any reported score. Example with 1 s initially off, 10 s armed before takeoff, 40 s airborne and an 8 s touchdown/rundown recording:

```bash
python -m experiments.rps_traj.regulated_deploy draw \
  results/rps_traj/event_response/deployment/laws/michaels.npy flight.npz \
  --seconds 1 10 40 8 --draws 3 --seed 2026
```

`flight.npz` contains `rps[draw, rotor, time]` in rev/s, `phase[time]` and `fs_hz=25`. Stage inputs specify the clock, not the realized motor speeds.

## Prospective check, reserved before acquisition

The 13 original targets are development data for model-family selection. Reserve **VID `indoor_loaded_round` and `indoor_loadless_round`** from the publisher, selected by maneuver and payload metadata alone. These are not among the four development sequences. No new speeds, plots or derived scores enter training or model selection. If a candidate satisfies all five development thresholds, freeze that family and source revision first. Evaluate on both with the existing VID fold-0 fits (same three physical donors for all methods), the established clock/32-draw protocol and the unchanged scorer. These two flights check transfer to a new maneuver on VID, not three-rig generalization. Failure must be reported; do not recycle them as a fresh holdout for the next candidate.

**Outcome:** the balanced advantage transferred; the all-regime advantage did not. Job `reserve-vid-round-pair-44d5bf` extracted the two previously unused recordings. The evaluator verified the locked checkpoint and replayed the original fold-0 ensemble before opening the reserved inputs. No cells excluded: two flights × five phases × six generators = 60 scored rows in `results/rps_traj/confirmation/scores.json`.

| Phase | Locked bridge | Best other fitted arm on the pair |
|---|---:|---:|
| Startup | 0.490 | 0.297, phase VAR(2), tied with its hybrids |
| Standby | 0.388 | 0.251, phase VAR(4) |
| Takeoff | **0.306** | 0.668, phase VAR(4) |
| Cruise | **0.125** | 0.566, VAR-HSMM hybrid |
| Landing | 0.832 | 0.481, template |
| Balanced across phases | **0.428** | 0.488, phase VAR(4) |

The template scored 0.571 balanced. Takeoff and cruise wins occurred on both flights. The bridge lost startup and landing on both; its standby win on the loadless flight did not outweigh the loss on the loaded one. The supported claim is a compact whole-flight fit with better balanced scores on both panels, not superiority on every regime of previously unseen flights. The model and metrics were not changed in response to this check.
