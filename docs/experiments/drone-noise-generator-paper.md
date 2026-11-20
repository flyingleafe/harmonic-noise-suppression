# Standalone rotor-noise generator paper

**Status:** neural-baseline extension delivered and verified on 2026-11-19:
15 current arms / 1560 fidelity pairs; 14 control methods / 672 readouts.
PTR, its published HPN comparator and DDSP-Piano are evaluated and included
in the paper and explainer. Recording warp is excluded from current comparison
panels and listening players. EONE code, fits and historical evidence remain
unchanged; all 115 frozen files passed verification.
Current proof: `results/noise_generator_paper/neural_extension/final_verification.json`.
The detailed IEEE-SPL-styled working draft is 22 pages, not an SPL-length
submission. Original-campaign proof remains
`results/noise_generator_paper/final_verification.json`.

**Current manuscript workflow:** edit `writing/papers/2026-08_wrapup/src/generator.tex`.
The wrap-up workspace's `tectonic -X watch` rebuilds both papers; the
generator PDF is `build/generator/generator.pdf`. The delivery records below
retain the original Quarto-era paths and checks.

## Motivation

The supervisor suggested separating the generator from the rotor-speed estimation
paper. Candidate contributions are physics-informed noise modelling, controllable
rotor-speed trajectories and corresponding audio, and calibration-data efficiency.
They are hypotheses about the paper's contribution, not established novelty or
performance claims. Literature and measurements may justify narrowing or dropping
any of them.

## Acceptance and boundaries

- Study prior art extensively before selecting and running new scientific
  evaluations; retain encountered papers and every full text read in the
  bibliography service under `drone-noise-generator`.
- Implement and run quantitative comparisons against at least two representative
  published methods, using metrics justified by that literature.
- Associate each retained contribution with a quantitative experiment. Do not
  describe limited-data calibration as few-shot learning without evidence.
- Deliver a complete Quarto manuscript in
  `writing/papers/2026-11_drone-noise-generator/`, a rendered PDF, and a
  self-contained LaTeX export that compiles independently for later Overleaf use.
  The final manuscript must contain no result placeholders.
- Original campaign boundary: leave the wrap-up manuscript and raw recordings
  untouched; no commits, pushes, paid compute, hardware integration or new
  recordings. Later generator-only source publication and the EONE extension
  were explicitly authorized below. Preserve unrelated changes and historical
  evidence. University/free compute only; heavy CPU work runs remotely.
- At most **five** substantive evaluation-and-revision rounds, including an
  initial or unsuccessful scientific campaign. Escalate inaccessible required
  data/baselines or an unavoidable scope expansion rather than silently reducing
  acceptance criteria.

## Literature and source audit

The initial library search was `drone propeller rotor noise synthesis auralization
generative model spectral validation` (2026-11-11). It found existing full texts
on small-rotor tonal aeroacoustics and drone audition, but the closest initial
procedural synthesis lead, *Physically Derived Sound Synthesis Model of a
Propeller* (2017; DOI `10.1145/3123514.3123524`), was metadata-only.
At that initial stage, independent primary-source studies were investigating
physical/procedural and learned/statistical/controllable synthesis; baseline
and metric selection followed those studies, as recorded below.

Existing implementation/evidence to reuse or audit:

- `docs/explainers/noise-model-v4.qmd`: one renderer/model with separate
  single-rotor bench and free-flight fitting procedures; no room/propagation
  model or calibrated cross-microphone phase relationships.
- `src/data_processing/noise_model/render.py`: the corrected local-speed shaft
  law and mean-power-normalized log-amplitude process. Its implemented amplitude
  is proportional to `exp(a - sigma2_amp)`, not the `exp(a - sigma2_amp/2)`
  displayed in the older explainer. The new manuscript must derive its
  normalization from the implemented stochastic model, not copy that expression.
- `src/experiments/noise_model/similarity.py` and
  `scripts/noise_v4_similarity_report.py`: existing line/floor spectral,
  order-tracked, modulation, distributional and spatial diagnostics. These are
  reusable instruments, not automatically evidence of metrics adopted by the
  synthesis literature. The old report uses only DREGON and compares model
  variants rather than two external published methods.
- `docs/data-catalog.md`: DREGON single-rotor bench recordings and pure-noise
  flights; MD2 FLY125/FLY124 multichannel recordings and FLY103/FLY108 mono test
  recordings. Existing model development used some validation recordings, so a
  calibration holdout must not be misrepresented as a previously unseen blind
  test. Raw test recordings must never enter calibration.

### Synchronized inputs and document-toolchain preflight

`results/noise_generator_paper/input_provenance.json` records pinned datasets
and the accepted model inputs. Both MD2 regime fits were retrieved from the
immutable collected outputs of `nv4-md2-cruise-fpg4-caa813`; their SHA-256 hashes
match the accepted-fit identifiability record. The accepted DREGON model was
retrieved from the base metadata of `easy-v4-banks@80fa07c85d53`; every parameter
matches the historically accepted local bench-plus-wind payload exactly.
The two DREGON scoring jobs did not collect their input payload, so the
published bank, not those scoring output trees, is its synchronized source.

A temporary document smoke verified Quarto 1.7.34 PDF rendering with Tectonic
0.15.0 and a separate `latex` output format. A clean directory containing only
the exported `main.tex` and `refs.bib` compiled with `tectonic main.tex`, including
BibTeX and a resolved numeric citation. Use explicit matching `pdf` and `latex`
format settings and scalar `natbiboptions: numbers`; the initial list-valued
natbib options failed compilation. This is a toolchain probe, not a completed
manuscript or a scientific evaluation round.

## Literature decision and evaluation protocol

This decision was recorded before the first new scientific evaluation.
The bibliography project is `drone-noise-generator`; the two literature studies
covered recording-based, procedural, aeroacoustic, learned and spectral-only
methods. Full texts and their bibliographic identities were checked, including
correction of three destructive fuzzy-title PDF misassociations in the service.
The import bug was reported; genuine pre-existing records were restored.

### Published baselines

1. **Heutschi et al. (2020)**, *Synthesis of real world drone signals based on
   lab recordings*, DOI <https://doi.org/10.1051/aacus/2020023>: native-rate
   reference audio, speed-dependent resampling and third-octave gain
   `E_i = S_i (RPM - RPM_ref)`. Fit its spectral gains from calibration only.
   The original method changes the whole recording's pitch and cannot control
   its rotors independently. An onboard comparison omits far-field propagation
   and uses observed mean rotor speed rather than simulating a flight controller.
2. **Yang et al. (2025)**, *Neural Drone Localization Exploiting Signal Synthesis
   of Real-World Audio Data*, DOI
   <https://doi.org/10.23919/eusipco63237.2025.11226465>, official MIT code
   <https://github.com/SOUNDS-RESEARCH/DroneSynthesis2DOA>: pulse-event rotor
   synthesis, stochastic timing, white noise and spectral equalization.
   Preserve the published source-stage mechanics and document target-calibration
   and telemetry-input adaptations. The official `getFullDroneSignal` passes
   the same macro RPM trajectory to every rotor; feeding separately measured
   rotor trajectories is an explicit stronger-baseline extension, not an
   existing independent-control API. Render the source at its native operating
   rate before conversion to the benchmark rate: a normalized low-pass cutoff
   of 0.1 must not silently become 800 Hz merely by running it at 16 kHz.

**Dreier and Vorländer (2024)**, DOI
<https://doi.org/10.1051/aacus/2024026>, is close prior art, not an exact
reproduction baseline: the paper and thesis do not specify the temporal
bandwidth/update/correlation law of Gaussian AM/FM. Their fitted Mavic
coefficients are not required for refitting, but supplying an unspecified
temporal law would be our completion. Their named metrics are ANSI S3.4-2007
N5 loudness, Aures sharpness and ECMA-418-2 tonality, not generic interchangeable
loudness/tonality implementations. The later Ramos-Romero 2024/2025 auralization
papers remain an explicit full-text access gap; do not claim exhaustive coverage
or priority over them.

### First comparison: recording-held-out MD2 synthesis

- Calibration: the existing pinned MD2 FLY125 recording. Evaluation: FLY124.
  No FLY103/FLY108 test material enters calibration. Use all eight channels and
  the published, already calibrated/refined RPS tracks, without reapplying
  timing or speed corrections.
- Preserve native-rate calibration audio for the resampling baseline; a
  16-kHz-only reference would incorrectly discard high-frequency content that
  can move into the evaluation band under a downward pitch shift.
- Audio evaluation rate: 16 kHz. Calibration windows: 8 s; evaluation windows:
  4 s; no overlap. Anchor the grid at the first jointly covered rotor-on
  interval; retain only complete windows whose four rotor speeds stay at or
  above 20 rev/s. Record exact timestamps, dataset hashes and exclusions.
- Report cruise, standby and transitions separately. Cruise is the primary
  comparison within the familiar operating range of the published source
  models; wider-range results characterize transfer, not an unqualified
  state-of-the-art ranking.
- All methods may use FLY125; no fitting, per-test gain correction or parameter
  selection uses FLY124. The existing stochastic model is frozen at its accepted
  calibration. Record the extra prior/calibration measurements it used rather
  than misrepresenting a short optimizer pool as its total data requirement.
  FLY124 has appeared during earlier model development, so this is a
  recording-held-out calibration-transfer diagnostic, not a new blind test.
- Use one experiment seed. Independent noise draws within that seeded
  evaluation estimate stochastic rendering variability; they are not new
  training seeds. Average microphones and draws within a window, then report
  variation across non-overlapping windows without treating microphones as
  independent airframes.
- Primary spectral instrument: RMS log-power spectral distance in dB on
  Welch spectra, 4096-point Hann, hop 1024, 125--7200 Hz. The definition follows
  the log-spectral-distance family used for rotor-spectrum prediction; do not
  claim that the drone auralization papers share a standard cross-method LSD
  protocol. Add third-octave levels, harmonic/floor and temporal-distribution
  diagnostics so average-spectrum agreement cannot hide missing line structure.
- Psychoacoustic descriptors, if reported, use an explicitly named algorithm at
  a nominal reference playback level with one common gain applied to the real
  and generated signals. They are not measurements of absolute aircraft SPL
  or loudness. Do not infer human perceptual equivalence without a listening
  experiment.

### Contribution criteria

Physics-informed structure, controllability and limited-data calibration already
have precedents. The candidate contribution is the specific calibrated
rotor-synchronous stochastic model and its measured behavior against the two
published source-model families. User control is a measured capability, not by
itself a novelty claim, and does not imply a remote-controller dynamics model.
Any calibration-data-efficiency claim requires budgeted refitting of **every**
target-data-dependent preprocessing and fitting step; the frozen accepted fits
alone cannot establish few-shot calibration. Unsupported candidate contributions
will be narrowed or dropped, as authorized, rather than filled with estimates.

### Exercised implementation and remote-data preflight

The source-stage implementation is in
`src/experiments/generator_paper/{cases,baselines,metrics,run}.py`, with the
thin `scripts/generator_paper_eval.py` driver and the existing `utils.gridrun`
unit harness. Synthetic instrumentation checks matched Yang's upstream
rectangular-pulse output exactly and verified the dB gain convention. Three
deterministic boundary regressions passed (metric gain units, silence handling
and fractional-period clock accuracy); the source-only Ruff check and the
driver's language-server diagnostics passed.

The comparison also reports a **phase-carry Yang variant**: integer period
rounding in the upstream event loop accumulates a speed bias, whereas retaining
the fractional interval does not. It is an explicit stronger-baseline
implementation variant. Both pulse variants fit their spectral EQ on FLY125
and select residual micromodulation range from `{0, 2, 5, 10}` Hz peak-to-peak
and white-noise/pulse ratio from `{0.002, 0.01, 0.05}` by calibration-only LSD.
All twelve candidate scores are retained; none is selected on FLY124.
A coherent-line ablation keeps the proposed model's calibrated powers and
floor but sets shaft, log-amplitude and phase variances to zero without refitting.
Every evaluated arm is reported.

Remote execution uses existing omnirun jobs plus a private, SHA-256-addressed
tar/R2 source artifact. Omnirun's native unpushed bundle requires a commit,
which is not authorized; manual rsync/sbatch would bypass the recorded runner.
The archive does not contain `.env`, raw datasets or the existing manuscript,
and it is unpacked into a job-owned ordinary directory rather than editing
tracked shared-worker source.

The first data-only preflight (`genpaper-data-preflight-9d5d87`) stopped before
loading recordings: GPFS denied creating a source hash beneath `.snapshots`.
No permissions were changed. The corrected ordinary-directory preflight,
`genpaper-data-probe2-756a43`, succeeded and its output was synchronized:

- FLY125: **21** eight-second calibration windows (168 s); their four-second
  subwindows are used for source-model calibration.
- FLY124: **26** four-second evaluation windows (104 s): **16 cruise**,
  **6 standby**, **4 transition**.
- Every clip has 8 microphones, 4 aligned RPS tracks, 16 kHz analysis audio,
  and preserved 44.1 kHz native calibration audio.
- Source-module paths point inside the immutable execution snapshot, and both
  accepted MD2 fit hashes were verified on the worker.

Evidence: `results/noise_generator_paper/{implementation_smoke.json,`
`preflight_submissions.json,code/snapshot_manifest.json}` and
`omnirun-outputs/genpaper-data-probe2-756a43/results/noise_generator_paper/`
`preflight2/preflight.json`. These are instrumentation/data checks, not
scientific generator-comparison rounds.

## Evaluation round ledger

**Scientific evaluation rounds completed: 5 / 5 (four succeeded, one failed).** Jobs use
`uni-cpu`, SHA-256-addressed source snapshots
(`results/noise_generator_paper/code/`), one seed (0) and four draws per window.
Successful outputs are synchronized before analysis; round 4 produced no
comparison outputs and its failure log is retained separately.

| Round | Job | Source snapshot | What ran | Outcome |
|---|---|---|---|---|
| 1 | `genpaper-round1-md2-c4e2af` | `c25b9d55…` | all five arms on the 26 FLY124 windows (520 per-draw comparisons) | statistics verified (`round1_verified_statistics.json`); both Yang variants at the upper white-ratio bound of the first grid |
| 2 | `genpaper-round2-yang-bb5321` | `2f744151…` | the two Yang arms only, calibration-only search widened to 6 × 8 = 48 candidates | both optima inside the widened grid (integer: 5 Hz, 0.05; phase carry: 10 Hz, 0.05), i.e. unchanged selections and unchanged FLY124 scores; round-2 raw files are the reported Yang results |
| 3 | `genpaper-round3-control-9dbebe` | `7ba52925…` | observable control probe: rotor 0 at 70/72/74 rev/s, others 80/88/96; peaks read back from generated audio, no refitting | proposed 48/48 matched, 0.0595 rev/s; Yang 48/48 (0.167, 0.111 rev/s); recording warp 12/48, rotor 0 never matched |
| 4 | `genpaper-round4-gp-610210` | `d082778f…` | matched FLY125 coefficient-GP fit, planned white/coloured FLY124 comparison and controls | failed during float32 GP factorisation (`NotPSDError`), before comparison scores; `round4_failure.log`; unsuccessful round counts |
| 5 | `genpaper-round5-gp-f64-aad5d5` | `dd3f8a6f…` | same data, model, subset, optimiser and metrics; GP calculations/checkpoint reload changed to float64 | succeeded; 208 GP fidelity pairs and 96 control-command readouts synchronized and independently verified |

Data-only preflights (`genpaper-data-preflight-9d5d87`, failed before loading
data; `genpaper-data-probe2-756a43`) are instrumentation, not rounds.

## Original five-round results and conclusion

Canonical numbers: `results/noise_generator_paper/final_evidence.json`
(method sources with raw-file hashes, regime statistics, control summary,
contribution decisions, limitations); fitted comparator state:
`final_fit_reports.json`. Every mean/SD was independently recomputed from the
per-draw records and every control error from the observed peaks. Log-PSD
distance, dB, mean ± sample SD over window means (16 cruise, 6 standby,
4 transition windows):

| Source model | Cruise | Standby | Transition |
|---|---:|---:|---:|
| Proposed stochastic model | 2.50 ± 0.71 | 2.21 ± 0.06 | 2.78 ± 1.11 |
| Coherent-line ablation (no refit) | 2.42 ± 0.79 | 5.91 ± 0.26 | 2.80 ± 1.14 |
| Recording warp + calibrated EQ (Heutschi) | 3.60 ± 1.11 | 3.65 ± 0.10 | 3.05 ± 0.88 |
| Yang, integer-period events | 2.99 ± 0.64 | 3.11 ± 0.13 | 2.45 ± 0.64 |
| Yang, fractional-period carry | 2.99 ± 0.61 | 3.06 ± 0.21 | 2.42 ± 0.64 |
| Coefficient GP, white residual | 5.03 ± 1.44 | 2.91 ± 0.09 | 4.32 ± 0.87 |
| Coefficient GP, coloured residual | 2.85 ± 1.13 | 5.08 ± 0.22 | 2.54 ± 0.44 |

- **No blanket win.** The proposed model is lower than the published-family
  adaptations in cruise and standby and has the lowest cruise visible-order
  line errors, but the Yang variants are lower on transitions, and the
  recording warp has the lowest cruise normalized modulation-spectrum error
  (3.49 vs 4.24 dB): every synthesised arm under-modulates.
- **Ablation is regime dependent.** Removing the temporal variances without
  refitting is marginally better in cruise but collapses standby (floor too
  low by 11.5 dB). It is not evidence that a refitted deterministic model
  would be worse.
- **Control** is a measured capability, not a novelty claim: the Yang variants
  and both GP controls also reach 48/48. The GP controls have lower readback
  error (0.00558/0.00583 rev/s) than the proposed model (0.0595 rev/s).
- **The GP residual choice is regime dependent.** Colouring improves cruise
  and transitions but worsens standby. The coloured GP is closer on transitions
  than the proposed model. These are coefficient-space adaptation results, not
  a demonstration of superiority over the original joint waveform SVGP.
- **Contribution decisions** (recorded in the evidence file): retain the
  specific stochastic/calibration formulation and its measured behaviour, not
  physics-informed or controllable priority; **drop the data-efficiency /
  few-shot claim** (the accepted fits use auxiliary FLY125 measurements and
  external DREGON-informed priors; 168 s of calibration windows is not a
  budgeted calibration ablation; Heutschi already calibrates from two
  recordings). Levels stay relative (nominal 80 dB playback for sharpness), no
  listening study, Dreier not claimed reproduced.

These are descriptive case-study results on one airframe with a non-blind
evaluation recording, not significance or perceptual-equivalence claims.

## Lee et al. (2026) Bayesian GP comparator

Lee, Ko, Seshadri & Rauleder, JASA 159(4):3418–3435, DOI
<https://doi.org/10.1121/10.0043469>, is close prior art, not excluded for a
larger multirotor. The original PDF was already present at
`bibliography/quietdrones2026/lineage/3418_1_10.0043469-jasa.pdf`; it was
imported as full text under bibliography UUID
`4d9d6491-24da-5c4a-977e-0ab228e7fce8`, project `drone-noise-generator`, and
read completely before round 4.

The published joint waveform SVGP uses a BPF-informed Fourier kernel times
separable Matérn spatial/velocity factors, DWT preprocessing, phase alignment,
predictive log likelihood and a Gaussian broadband likelihood. It evaluates
CONA-simulated waveforms, not real onboard transfer, using DTW, FFT spectra,
ISO 532-1/SQAT loudness and psychoacoustic annoyance. Active training-location
selection is also prior art for economical calibration design.

Existing `jasa_gp.py` and `egonoise_gp.py` instead use projected coefficients
and independent ARD Matérn ExactGPs. The legacy real-recording adapter also
failed a tiny phase-alignment probe: phase 0.7 rad requested at zero became
1.39115 rad. Its old split mixes FLY125/FLY124, and the augmentation pool
adds shell-position projection and final RMS normalisation. None is an
unqualified paper-exact, matched-calibration result.

`src/experiments/generator_paper/gp_baseline.py` therefore reuses the existing
coefficient-GP factory and phase convention as an explicitly adapted control:
64 uniformly selected native-rate one-second FLY125 frames, 60 BPF harmonics
per rotor, 512 microphone--frame points, 481 outputs, seven inputs (xyz and
four speeds), Adam 200 steps at 0.05. White and coloured residual variants
share one frozen checkpoint; no test-time gain matching or shell projection.
All departures from the original joint GP are in the fit report and manuscript.

Round 4's covariance factorisation failed in float32 after jitter up to 1e-4;
round 5 used float64 throughout the GP and checkpoint reload, not a larger
jitter or a hidden fallback, and completed successfully. All floating GP state
and training tensors in the synchronized checkpoint are float64; its stored
reload probe reproduces exactly. The raw GP arm aggregations independently
recompute with maximum absolute difference 0, and all control peaks were
re-detected from saved PSDs. `control_combined/` concatenates the method axes of
rounds 3 and 5 after verifying identical grids, conditions, seeds and readout
rules; its provenance retains both original probes and hashes.

The tiny synthetic smoke also exercises fitting, checkpoint reproduction,
frequency readback, out-of-band rejection and rejection of a smoke checkpoint
by the production benchmark. Its results are not real-data fidelity evidence.

Exact dated literature-study snapshots and source provenance are retained in
`results/noise_generator_paper/literature/`; the new GP review supplements,
rather than retroactively exhausts, the initial review. Kim et al. (2026),
DOI `10.1121/10.0042382`, and Jeong et al. (2021), DOI `10.1063/5.0071850`,
are additional retained abstract-supported stochastic-rotation precedents;
their full texts were not acquired, and no priority claim is made over them.

## Manuscript

`writing/papers/2026-11_drone-noise-generator/`: Quarto source with explicit
`pdf`/`latex` formats, a deterministic `prepare.py` (hash checks, independent
recomputation, stored-waveform and control-peak checks, prose-claim checks)
and an `export.py` producing a self-contained Overleaf directory.

## Original campaign verification

- Seven reported arms, 728 per-draw fidelity comparisons; six control generators,
  288 commanded-rotor readouts (252 matched, with missing support preserved).
- Every reported mean and sample SD recomputes from synchronized raw records
  with maximum absolute difference zero; stored example scores and all control
  peaks are checked again by `prepare.py`.
- All 36 comparative prose checks pass. The 220 numeric table cells were checked
  against extracted PDF text, including component checks for wrapped cells.
- Quarto and independently compiled ZIP PDFs have identical normalized text.
  No undefined citations/references or missing resources; minor nonblocking
  table-alignment warnings are below 0.2 pt, with no visually clipped content.
- Ruff passes; targeted pytest: three passed (16 existing warnings). The synthetic
  GP smoke and the successful real round 5 prove the changed GP paths; the earlier
  failed float32 round remains in the ledger rather than being discarded.
- Canonical source and handoff bundle:
  `writing/papers/2026-11_drone-noise-generator/main.qmd`, `main.pdf`,
  `overleaf.zip`. Use XeLaTeX for subsequent Overleaf editing.

The old rotor-speed manuscript and raw recordings were not edited. The retained
claim is the specific stochastic/calibration formulation and its measured
tradeoffs, not first physical structure/control, few-shot advantage, blind
generalisation or superiority over the original joint waveform GP.

## EONE extension requested on 2026-11-17

The user identified reference 18 (Doerfler, Kuntz and Zimmer) as a useful
rotating-machine baseline and requested implementation, evaluation, paper
integration and matching explainer samples. This is a new authorized extension,
not an unrecorded sixth run of the original capped campaign. The separate
2026-11-16 request also authorized adding and pushing the generator LaTeX
snapshot to the wrap-up submodule; commit `cb5807b` added `generator.tex` without
including the user's existing tracking-paper edits.

### Primary-source analysis before evaluation

The source is *Gradient-Based Learning of Parametric Engine Sound
Representations for Real-Time Resynthesis and Tuning on Embedded Systems*,
arXiv:2606.21521 / AES 2026 Automotive Audio paper 27. It is retained as full
text in the bibliography service, UUID
`3c5f134f-9419-5b72-9743-9a455063093b`, PDF SHA-256
`5c7b066c9a731c298a6cff82c82965560220e2d1fdc9264d7442bbbd09d09df3`.
Its reference graph was expanded. The official paper HTML, demo page and demo
repository were inspected before implementation; no synthesizer, trainer or
pretrained checkpoint was linked from those resources. This is a bounded
availability finding, not a claim about the authors' private implementation.

The deployed EONE representation is directly relevant: positive RPM/torque gain
curves projected by linear control masks drive an additive order bank and
64 ERB cosine-filtered, random-phase noise bands. The pretrained encoder-decoder
only supplies initialization for the target curve-bank fit. The paper specifies
phase accumulation, IRFFT noise synthesis, multi-resolution STFT and harmonic
losses, but leaves several normalization and fine-tuning choices unspecified.
The primary source overrides an initial PaperQA shortlist that missed the
synthesis-equation chunk. Source evidence, omissions and fixed choices are in
`results/noise_generator_paper/literature/eone2026_primary_review.json`.

### Fixed adaptation and hypothesis

The hypothesis is that this learned order/noise representation is a relevant
source-model comparator on the same known rotating machine. The experiment
compares complete calibrated implementations, not architecture alone or the
published pretrained model's generalization.

- Method ID: `doerfler2026_eone`; display name: **EONE-RPS curve bank**.
- No torque proxy is fabricated. The torque factor is fixed to one; positive
  40-point RPM curves span 0–10000 RPM.
- Four rotor sources replace the engine carrier. Shaft orders 1–128 correspond
  to half-step multipliers on blade-pass frequency; this explicitly extends the
  paper's 64 components to cover the rotor benchmark band.
- Each microphone has its own gain curves. Rotor excitations are shared across
  microphones, with no propagation-delay or arbitrary-position claim.
- The 64 ERB bands use independent random-phase IRFFT loops of 65536 samples,
  zero DC/Nyquist and unit RMS. These choices fix unspecified basis units and
  avoid duration-dependent noise gain. No output level is matched to test audio.
- Initialization is flat and calibration-RMS-based, not an invented pretrained
  encoder. All 168 calibration seconds enter 42 non-overlapping four-second
  pieces. Hyperparameters were fixed before FLY124 evaluation.
- The production fit uses 50 epochs of AdamW, one-cycle cosine max LR 0.01,
  multi-resolution FFT sizes 32768 down to 32, harmonic-energy supervision
  (FFT 65536, window 16384, hop 256) and normalized curve-curvature regularization.
  The complete loss definitions and numerical choices are in the module and
  fit report; the best epoch is selected by calibration objective only.
- Existing FLY125 → FLY124 split, pinned frames/RPS, 26 evaluation windows,
  seed 0, four draws, all fidelity diagnostics and the 48-command waveform
  control probe are unchanged. Published EONE engine LSD uses a different
  STFT/range and is not numerically interchangeable with the benchmark metric.

Acceptance is a completed finite, reproducible calibration plus all 104
fidelity pairs and 48 control readouts, including unfavorable outcomes or
missing peaks. Numerical failure or nonconvergence must be recorded and
diagnosed, not hidden by removing components, losses or evaluation cases.
No other baseline is refitted and no FLY124 score selects a setting.

### Implementation verification and submitted run

`src/experiments/generator_paper/eone_baseline.py` implements differentiable
synthesis, direct curve fitting, atomic completed-fit checkpoint export,
reload verification and the target-free renderer. The existing evaluator and
control driver reuse one frozen checkpoint. Nine focused regressions passed
(six EONE behaviors and three existing generator checks), along with Ruff and
both CLI smoke commands. The gain-ramp test checks the actual relative output
without pinning RNG draw order.

University offer probes showed long queues, so the run uses free Kaggle T4
placement with a two-hour allocation and max cost zero. Job
`genpaper-round6-eone-d30594` first exercises a tiny CUDA calibration/reload,
then fits on FLY125 and runs the full fidelity and frozen-fit control comparison.
Its source overlay SHA-256 is
`0d6137c5bbf56c30046b2882f83099312d71ed11a268f86b1f0e627b33625c4e`,
derived from the prior immutable comparison snapshot with only the new core,
runner/control hooks and regression file replaced. The dirty research checkout
was not committed. Submission:
`results/noise_generator_paper/eone_round6_submission.json`.

### Completed calibration and comparison

Job `genpaper-round6-eone-d30594` succeeded on a free Kaggle T4 and its outputs
were pulled before analysis. The fit completed all 50 epochs / 2100 updates in
595.7 seconds. Its fixed-draw calibration objective fell from 3.10468 to 1.66163
(46.5%). The selected epoch is the final budgeted epoch, not evidence of full
optimization convergence. Fidelity and control used the same completed
checkpoint, SHA-256
`d6e27f420ac9874529f834bb07598a2f1eca969a5dc87a7f1711b01cc0aa28ac`.

All 26 evaluation keys and their full provenance match the original benchmark;
the three saved reference audio/RPS examples are bitwise identical. All 104
new fidelity pairs were recomputed. The expanded comparison now has eight
reported arms / 832 fidelity pairs.

Welch log-PSD error, dB; mean (sample SD) across windows, lower is better:

| Regime | Proposed stochastic | EONE-RPS curve bank |
|---|---:|---:|
| Cruise, 16 windows | 2.50 (0.71) | 2.18 (0.91) |
| Standby, 6 windows | 2.21 (0.06) | 5.49 (0.20) |
| Transition, 4 windows | 2.78 (1.11) | 2.77 (1.69) |
| All, 26 windows | 2.48 (0.69) | 3.04 (1.66) |

EONE also has lower cruise visible-peak, prominence, line-bin and floor errors.
The previous cruise line-fidelity advantage is therefore withdrawn, not
preserved by excluding the new comparator. Our standby advantage remains;
the fractional-carry pulse model has the lowest transition log-PSD error.
EONE's standby floor is too loud, unlike the too-quiet coherent ablation:
these failures do not support one common missing-fluctuation explanation.
This remains one known-airframe, non-blind parameter-transfer diagnostic,
not a population, perceptual-equivalence or architecture-only result.

The frozen EONE fit supplies 44/48 waveform control readouts, with conditional
speed MAE 0.02156 rev/s. All four missing peaks are on commanded rotor 0
(8/12 matches); no draw supports its complete three-setpoint slope.
Missing slopes remain undefined. The combined seven-method probe has
336 commands; all are independently re-detected from saved PSDs. No test-time
fitting or target level matching is introduced.

### Updated deliverables and verification

- `writing/reports/2026-11-15_generator-baselines-explainer/index.html` explains
  the curve-bank synthesis and exact fitting objective/settings, with all eight
  methods plus real recordings on the same three trajectories. All 27 embedded
  four-second WAVs decoded and advanced in offline Brave; every spectrogram
  decoded. PCM16 exports match the saved microphone-0 waveforms at one common
  gain per case to half a quantization step, without clipping. Offline native
  MathML, mobile layout, exclusive playback and the no-JavaScript fallback pass.
- `writing/papers/2026-11_drone-noise-generator/main.pdf` and `overleaf.zip`
  include the method, revised conclusions, all result tables, a matched
  real/EONE spectrogram figure and `samples.html`. Forty-five numerical prose
  guards pass; all 252 numeric/coverage table cells survive in the PDF.
  The final ZIP compiles with Tectonic in a clean directory and has the same
  logical PDF text as the canonical render. No overfull boxes or unresolved
  references were reported. The 19-page draft is not an SPL-ready submission.
- Targeted pytest passed nine tests (16 existing warnings); Ruff, both CLI help
  paths, tiny CUDA calibration/reload and the full scientific run passed.
- The additional generator-only LaTeX source, three figures and embedded sample
  companion were refreshed and pushed to the wrap-up submodule as commit
  `c0ae9fc` on `main`, preserving the intervening tracking-paper commit
  `6029824`. `generator.tex` compiles independently and differs in logical PDF
  text only by the namespaced companion filename. No parent-repository commit
  or push was made.

## Regime-matched baseline extension requested on 2026-11-17

### Motivation and fixed comparison

The user identified a calibration asymmetry: our generator has independently
fitted standby and cruise experts, whereas the external baselines were fitted
jointly across speeds. The user requested the same paired fitting and blending
for every external family. This tests calibration treatment against the six
existing pooled adaptations; it does not add a pooled-fit version of ours.
The prior pooled results remain reported and unchanged.

Before observing new scores, the protocol is fixed as follows:

- Read the exact support specifications from the two hash-verified accepted
  FLY125 fit artifacts, not an approximate regular window grid. Standby uses
  `[2,6)`, `[6,10)`, `[10,14)` seconds; cruise uses eight eight-second windows
  starting at 16, 32, 48, 64, 96, 112, 128 and 144 seconds.
- Every family gets separate standby and cruise fits on these same core
  supports: Heutschi, both Yang timing variants, the shared white/coloured GP
  state, and EONE-RPS. Existing family hyperparameters and calibration losses
  remain fixed. EONE uses 50 epochs per expert; the GP uses all 12 standby and
  64 cruise one-second frames within its existing frame cap.
- Reuse the proposed generator's exact gate: a samplewise smoothstep in the
  slowest rotor's RPS, zero at or below 45 and one at or above 65 rev/s.
  Independent expert streams combine with square-root gains, interpolating
  power. An inactive expert is not rendered.
- Heutschi needs a sufficiently long standby reference to support time warping.
  Its three contiguous four-second supports are joined into one 12-second
  reference/calibration clip. This uses no additional samples, looping, padding
  or test audio; constituent provenance is retained.
- Frozen expert states, not test-time refits, serve both fidelity and control.
  FLY124 evaluation windows, four draws, seed zero, metrics and 48 cruise
  commands are unchanged. The six new IDs end in `_regimes`.
- The matched primary panel contains ours, its no-refit coherent ablation and
  six matched adaptations. A separate panel retains the original eight arms.
  Combined evidence will contain 14 arms / 1456 fidelity pairs and 13 control
  generators / 624 commands when all runs finish; those are acceptance counts,
  not completed results.
- CPU calibration/scoring uses `uni-cpu`; EONE CUDA work uses free Kaggle or
  `uni-gpushort`. No paid compute or parent-repository commit/push is authorized.

This matches the core calibration support and regime composition, not every
method's auxiliary measurements, priors, parameter capacity, loss or optimizer.
It must not be presented as an architecture-only or total-data-budget controlled
comparison, and forcing a published-family adaptation through our regime wrapper
is explicitly a benchmark extension rather than a paper-exact reproduction.

### Implementation, real-input smoke and submitted workloads

The paired adapter, explicit-support loader, evaluator and frozen control path
are implemented. Seventeen focused regressions pass (16 existing warnings);
Ruff and both actual CLI help paths pass. A real-data preflight reads exactly
the 11 accepted support windows at native 44.1 kHz and proves that all 26
evaluation provenance records and the three saved reference audio/RPS arrays
remain unchanged.

The first actual paired-Heutschi transition smoke exposed a calibration failure,
not a reason to retune on evaluation audio: the independently fitted standby EQ
extrapolates to a generated RMS 84.176 dB above the reference. The waveform is
finite, but its nominal-playback low-frequency level exceeds MoSQITo's
ISO 532-1:2017 A.3 domain. The old loudness wrapper therefore raised before the
otherwise valid spectral results could be stored.

The harness now checks that exact domain before the same pinned MoSQITo kernels:
any 25–250 Hz third-octave level above 120 dB makes the pair's psychoacoustic
comparisons undefined. It does not cap, rescale or refit the generator, average
only surviving microphones, or hide the spectral error. Other numerical errors
still propagate. Valid-domain loudness and sharpness match the original public
API exactly in the smoke check. A regression covers one failed microphone
without partial sharpness credit. The fixed protocol records this amendment
before formal submissions.

With the same frozen Heutschi states and identical output, the real evaluator
now completes the first transition case, retaining its 75.395-dB log-PSD error.
The actual control CLI also reloads the same manifest with a deliberately absent
calibration-data path and reports the actual 11 supports. These are runtime
smokes, not an additional scientific arm or a test-selected setting.

| Submitted job | Backend | Reported matched families | Fidelity pairs | Control readouts |
|---|---|---|---:|---:|
| `genpaper-r7-regimes-cpu-170b8a` | `uni-cpu` | Warp, two pulse variants, two shared-state GP variants | 520 | 240 |
| `genpaper-r7-regimes-eone-3881e2` | Free Kaggle T4 | EONE-RPS | 104 | 48 |

The counts above are the submitted workloads, not a claim of completion. Both
jobs use immutable source overlay SHA-256
`1fa4290112aa212c204cff4ffb36aef26397c79ac7fb82ba1ea58cef8d681e07`,
over pushed base `7ae5b37c6093a701e57eea0f835e645870c89ac4`.
Only the seven feature/regression files replace the prior immutable source;
the other declared runtime dependencies match it exactly. No dirty-checkout
commit was made. The CUDA job first exercises fitting both tiny experts,
moving the frozen bundle, deleting its input data and reloading/rendering it.

Protocol and provenance:
`results/noise_generator_paper/regime_round7_protocol.json`,
`regime_round7_preflight.json`, `regime_round7_gain_diagnostic.json` and
`regime_round7_submission.json`. The original eight-arm evidence is retained as
`evidence_before_regime_round7.json`.

For the listening companion, failed-level outputs remain present with their
scored source hashes, but receive explicitly labelled, separately attenuated
previews rather than quantizing the real recording and every valid generator to
silence. All valid sounds still share one common case gain; no playback gain is
used in scoring. Raw spectrograms share a reference-anchored scale and label
display saturation. These presentation rules expose, rather than repair, the
failed gain.

### Intermediate CUDA-arm snapshot (superseded by the full comparison below)

`genpaper-r7-regimes-eone-3881e2` succeeded and was pulled before analysis.
The paired CUDA smoke passed. Both production experts completed 50 epochs:
150 updates on standby and 800 on cruise; both selected epoch 50. Their finite
budgets do not establish optimization convergence. All 104 fidelity pairs were
independently recomputed, all 48 control readouts re-detected from saved PSDs,
and the portable frozen pair reproduced a saved waveform on the laptop CPU
within a maximum absolute error of `6.71e-8`.

The evaluation provenance and metric definitions match the native comparison.
EONE's observed treatment change is:

| EONE treatment | Cruise LSD (dB) | Standby LSD (dB) | Transition LSD (dB) | All-window LSD (dB) | Control matches |
|---|---:|---:|---:|---:|---:|
| Native pooled | 2.18 | 5.49 | 2.77 | 3.04 | 44/48 |
| Standby/cruise experts | 2.38 | 2.38 | 2.80 | 2.45 | 28/48 |

These were descriptive, within-recording means at the intermediate checkpoint,
not significance or generalization claims. The lower standby error accompanied
worse cruise/control results, not a uniform improvement. The five CPU variants
were not yet available at that checkpoint. Independent CUDA-arm checks and
exact statistics:
`results/noise_generator_paper/regime_round7_eone_analysis.json`.

### Completed matched comparison and interpretation

Both round-7 jobs succeeded and were pulled before analysis. All six new
104-pair fidelity results were independently recomputed; the native eight
artifacts and every native mean/SD remain unchanged. Every evaluation-window
provenance record matches the original benchmark, and all six saved real
audio/RPS examples from the two jobs match the original arrays bit for bit.
The combined control probe appends only the method axis; all 624 readouts were
re-detected from saved PSDs at `1e-9` tolerance.

Matched-core panel, Welch log-PSD error in dB (window mean; lower is better):

| Generator | Cruise | Standby | Transition | All windows | Control matches |
|---|---:|---:|---:|---:|---:|
| Proposed stochastic | 2.50 | 2.21 | 2.78 | 2.48 | 48/48 |
| Recording warp + EQ, two experts | 3.16 | 2.25 | 38.92 | 8.45 | 25/48 |
| Yang integer, two experts | 2.81 | 2.25 | 29.64 | 6.81 | 48/48 |
| Yang phase carry, two experts | 2.82 | 2.27 | 28.24 | 6.60 | 48/48 |
| GP white, two experts | 5.13 | 3.26 | 4.40 | 4.59 | 48/48 |
| GP coloured, two experts | 2.88 | 2.39 | 3.38 | 2.84 | 48/48 |
| EONE-RPS, two experts | 2.38 | 2.38 | 2.80 | 2.45 | 28/48 |

The user's calibration concern was substantive: the closest standby baseline
is now only about 0.04 dB above ours. Our lowest standby/transition means have
small margins within the observed window spread, not evidence of a clear or
population-level advantage. Matched EONE has the lowest cruise/all-window means.
The native pooled EONE cruise mean and native phase-carry pulse transition mean
remain lower than every primary matched arm; those variants are not discarded.

The bundled treatment lowers standby error for five of six external variants
but increases transition error for all six. Warp and both pulse variants have
particularly large transition errors. The same gate does not prevent their
speed-linear gain laws from being evaluated beyond a regime's fit speeds;
extrapolation is a plausible explanation, not an isolated causal experiment.
We did not cap gains, retune on FLY124, change optimizer budgets or replace
unfavourable outputs. The four warp pairs outside the ISO loudness domain keep
their large spectral errors and explicit unsupported psychoacoustic diagnostics.

Only the proposed model, pulse variants and GP variants have full matched
control coverage. Matched EONE has one commanded-rotor match in twelve and
three matches in twelve on rotor 3, with no complete rotor-0 response slope.
Low conditional speed error is not a control pass. The no-refit coherent
ablation keeps the proposed model's regime pair and remains a within-model
comparison, not a separately optimized deterministic competitor.

### Verified delivery

- Canonical evidence: `results/noise_generator_paper/final_evidence.json`.
  Both native and matched panels, all fit/source hashes, partial support and
  control provenance are retained. Analysis:
  `regime_round7_analysis.json`; native-retention check:
  `regime_round7_native_statistics.json`.
- Manuscript: `writing/papers/2026-11_drone-noise-generator/main.qmd`,
  `main.pdf`, `main.tex` and `overleaf.zip`. Seventy-eight numerical prose
  guards pass; all 504 numeric/coverage cells of the primary/native result
  tables survive in the PDF. A fresh ZIP-only Tectonic compile matches the
  canonical content. No overfull boxes or unresolved references/citations
  were observed. This 25-page draft is not an SPL-ready submission.
- Explainer: `writing/reports/2026-11-15_generator-baselines-explainer/index.html`,
  also copied as the paper's `samples.html`. All 45 four-second WAVs played to
  completion offline in installed Brave, all spectrograms decoded, and native
  MathML, mobile layout, exclusive playback, selector pauses and the
  no-JavaScript fallback passed. No browser runtime errors or offline HTTP
  resource dependencies were observed.
- Audio exports match the scored microphone-0 waveforms at their declared
  gains to half a PCM16 step, without clipping. Ordinary sounds share one case
  gain. Three excessive-gain transition outputs have explicit standalone
  previews: the matched warp and both pulse variants, attenuated by an extra
  87.06, 69.79 and 63.55 dB respectively. The shared-gain PCM16 reference errors
  would otherwise be 97.4%, 25.6% and 12.5%; the presentation tolerance is 1%.
  These previews do not change any score and are labelled not level-comparable.
- Targeted pytest: 17 passed (16 existing warnings); changed-file Ruff passes;
  import-linter: all three contracts kept. Actual paired evaluation/control
  CLIs and the CUDA fit/freeze/move/reload/blend smoke were exercised.
- Generator-only source, five figures and the verified companion were refreshed,
  independently compiled and pushed to the wrap-up submodule as `c2aa388`,
  preserving the tracking paper and previous user history. No parent-repository
  commit or push was made.

Retained claims are the explicit stochastic formulation, calibration procedure,
competitive measured fidelity and its tradeoffs. Generic physical superiority,
unique control, “best overall”, data-efficiency, few-shot, perceptual-equivalence,
blind-generalization and complete-published-EONE claims are not supported.


## Native neural-baseline extension authorized on 2026-11-19

### Motivation and fixed protocol

The user requested removing the recording-warp comparator from the current
paper and explainer, retaining existing EONE results without changing EONE,
and adding the selected additional candidates. The new `isambard` backend
is explicitly available for this work. This extension is separately
authorized; it does not rewrite the original five-round boundary or discard
the historical warp results.

- New matched-only methods: `doerfler2026_ptr_regimes`,
  `doerfler2026_hpn_regimes`, and `renault2023_piano_regimes`.
- PTR and HPN reuse the official source at
  `rdoerfler/ptr-model@9eed6b0d26e1d65cb6b892a125215dca32c97a37`.
  Four intact single-source generators are summed and fitted jointly to each
  mono microphone recording. The native spectral loss and harmonic loss are
  retained; the latter is reduced over all four supplied tracks against the
  complete mixture, not invented isolated targets. Fixed native torque and
  an explicit fundamental-frequency unit conversion require no new controller.
  Pulse shaping and resonators are not replaced by drone-specific modules.
- DDSP-Piano reuses
  `lrenault/ddsp-piano@e868b7ccd3fe31b39132048a72561d7fcf1b465f`
  and DDSP 3.7.0 with the published `dafx22.gin` architecture. Four continuous
  pitch trajectories, fixed onset velocity, inactive pedals and native
  microphone/environment identifiers replace the MIDI data representation;
  no per-source loudness targets, F0 loss or new conditioning network is added.
- Every new method fits separate standby/cruise experts on the same accepted
  FLY125 core supports (12 s / 64 s) as round 7, then uses the existing
  square-root power blend and independent expert streams. Native training
  framing/overlap and any unused support tails are recorded, not represented
  as identical effective sample exposure.
- Before evaluation, PTR/HPN use the official code's 100-epoch training and
  component/controller/optimizer defaults. The source uses AdamW weight decay
  0.005 and one-cycle maximum learning rate 0.0002, which differ from parts of
  the paper's prose. DDSP-Piano uses the published amplitude/frequency/amplitude
  phase order, 2+1+3 epochs, 1000 steps per epoch, batches 6/3/6 and the
  native Adam learning rates/regularizers. This is an explicit bounded update
  budget on our smaller corpus, not reproduction of the authors' data/compute.
  Complete losses, actual updates and source/paper discrepancies are retained.
- Evaluation is unchanged: all 26 FLY124 four-second windows, eight microphones,
  16 kHz, seed zero and four stochastic draws. The same 48-command readback
  measurement is applied to each new method. No FLY103/FLY108 input, label
  re-correction, per-test gain fitting or evaluation-based selection is allowed.
  The target extension is 312 fidelity pairs and 144 control readouts.
- `results/noise_generator_paper/neural_extension/eone_frozen.json` records 115
  pre-change hashes, including EONE source, both EONE output trees and the
  original canonical evidence. New evidence is written separately to
  `neural_extension_evidence.json`; old EONE numbers and waveform arrays are
  copied, never rerun. Historical warp files remain archived.
- Native TensorFlow and project runtimes are separated by real aligned array
  artifacts when their Python/package requirements differ. Native inference
  receives only RPS, fixed controls, blend gains and frozen weights; reference
  audio is consumed only by the separate scoring stage.

### Runtime preflight

`isambard` connectivity and free GH200 placement were verified. The first
environment-only job, `genpaper-isambard-env-20d74f`, failed before running
project code because its older base commit required a nonexistent ARM64
`torchcodec` wheel. No scientific fitting or evaluation occurred.
The corrected job, `genpaper-isambard-env-18579c`, uses pushed base
`d9b898dfa4b272bc818b8a675bf7a1ea46bbf3b5`; its report was pulled before
inspection. It verified aarch64 Python 3.12.16, CUDA-enabled Torch 2.7.0,
torchaudio 2.7.0, and an NVIDIA GH200 120GB, plus installed Apptainer.
TensorFlow/TorchLPC and the CUDA-toolkit environment remain native-runtime
prerequisites, not reasons to replace the published models.


### Submitted native comparisons

The three jobs below were submitted together through the existing omnirun
client, using pushed base `d9b898d` and immutable source snapshot
`4ed9297475ff1a011f804a27ba6868f2743c2cbfa83a06f7c9d41e6ff6afa1e6`.
Each requests one free isambard GPU, eight CPU cores, 64 GB RAM and an
eight-hour limit. The source archive was checked to exclude credentials and
raw dataset trees. The existing EONE source hash matches its frozen preimage.

| Method | Job | Current evidence state |
|---|---|---|
| PTR, two experts | `genpaper-r8-ptr-b77374` | succeeded, pulled and verified: 104 pairs / 48 readouts |
| Published HPN, two experts | `genpaper-r8-hpn-1b276c` | succeeded, pulled and verified: 104 pairs / 48 readouts |
| DDSP-Piano, two experts | `genpaper-r8-piano-tracing-da80fb` | succeeded, pulled and verified: 104 pairs / 48 readouts |

Initial submission manifest:
`results/noise_generator_paper/neural_extension/round8_submission.json`.
Every job first performs a real native CUDA fit, moves/reloads its checkpoint
bundle and compares seeded renders of the same speeds before production.
The smoke uses one microphone and one calibration support per regime; it is
instrumentation, not a reduced scientific comparison. Production still uses
all allowed supports, all eight microphones and all fixed evaluation/control
cases. CUDA LPC atomics are checked within an explicit numerical tolerance,
not represented as bitwise deterministic.

Local verification before submission: the focused continuous-pitch/control
test passed (three native slow round trips were deselected); the stage and
merge CLIs were launched; changed-file Ruff passed after the three reported
import/Python-compatibility issues were corrected. This is not a claim that
the remote native fitting or scientific comparison has completed.

For fixed-phase native generators, the common square-root-gain blend need
not be an exact interpolation of expected powers: correlated tonal outputs
retain a cross term. The shared gain rule is unchanged, while this limitation
is explicit in both documents. No phase randomisation or source-model change
is introduced to hide it.


### First collected result and native-runtime correction

HPN completed successfully on isambard and was collected before analysis.
The complete 26-window/four-draw grid gives 104 fidelity pairs and all 48
control readouts; every control target was matched. Both frozen experts are
marked completed with production settings. The moved-checkpoint CUDA smoke
reproduced both seeded waveforms exactly in that run. All three saved
reference audio arrays and RPS tracks are bit-for-bit equal to the existing
common reference examples. Independent summary/check record:
`results/noise_generator_paper/neural_extension/hpn_verified.json`.

The first DDSP-Piano job reached the real native GPU model build but failed
before any training update: TensorFlow 2.15/Keras invoked legacy `CudnnRNN`,
while cuDNN 9 requires explicit sequence lengths. This is a runtime API
compatibility failure, not a model-quality result or a completed scientific
arm. Full failure evidence is retained in
`neural_extension/piano_cudnn9_failure.log`.

Retry `genpaper-r8-piano-legacy-eb008e` uses immutable snapshot
`2bef8503038191a350b74075ae21059980bf025c025e1a99c55c3955d701519b`
and the official NVIDIA 23.12 TensorFlow container (TensorFlow 2.14, Python
3.10 and cuDNN 8.9.7). Its system-site-packages virtual environment preserves
the container's paired TensorFlow/Keras libraries. The native model, losses,
calibration supports, training phases and update budget are unchanged.
Submission: `neural_extension/piano_runtime_retry_submission.json`.

The 23.12 retry passed the GPU GRU diagnostic but stopped in the first native
`Trainer.train_step` before its first update. Its unconstrained dependency
resolution installed `typing_extensions==4.16.0`; TensorFlow's tracing protocol
then raised the documented `_DictWrapper.__dict__` error. The upstream diagnosis
and original `<4.6` constraint are in
[TensorFlow issue 60687](https://github.com/tensorflow/tensorflow/issues/60687).
Full observed trace: `neural_extension/piano_typing_failure.log`.

Retry `genpaper-r8-piano-tracing-f7dba4` uses source snapshot
`c0b8562ec6aabcb89d3354be36b59c0bbf0822fec9f02a0716eda7aee138a16d`.
It reuses the already built 23.12 container and the failed job's private
environment, pins `typing_extensions==4.5.0` and compatible companion
dependencies through `neural_extension/piano_runtime_requirements.txt`,
and repeats the actual native fit/reload smoke before production. No source
patch, graph rewrite, model/loss change, shortened training budget or quality
claim is made. Submission: `neural_extension/piano_tracing_retry_submission.json`.

The first dependency-pinned retry stopped at a host-side interpreter-path
check: the reused virtual environment's Python is a symlink to an absolute
path inside its container. A host `Path.is_file()` incorrectly rejected it.
Retry `genpaper-r8-piano-tracing-da80fb`, snapshot
`d8751a152cd7daea8a6d2e18f84c9a38ee63d357cce0bce08708bd909e94cb59`,
uses `os.path.lexists()` for that entry; the following actual container
execution still validates the interpreter. This is an instrumentation-only
correction. The failed check never reached dependency installation or fitting.
Records: `neural_extension/piano_container_symlink_failure.log` and
`neural_extension/piano_tracing_retry2_submission.json`.

Backend status checked on the next user status request confirms that PTR
succeeded at `2026-11-19T05:02:50Z` and the final DDSP-Piano retry succeeded
at `2026-11-19T03:17:12Z`. The two local `omnirun wait` processes had remained
blocked and did not deliver those completion notifications; they were stopped
after the authoritative backend check. Both output trees are being pulled
with the native client before analysis. The successful Piano execution uses
the pinned legacy dependency stack, without changing the model or fitting
budget. HPN remains the only newly collected and verified arm at this point.

### Collected native comparison and bounded conclusion

All three selected arms are now collected and verified. Each contributes 104
fidelity pairs (26 FLY124 windows, four draws) and 48 control readouts. All 144
new control targets matched. Both experts completed their fixed production
budget: PTR/HPN 100 epochs per microphone, with 100 standby and 300 cruise
updates; DDSP-Piano 6000 updates per expert across the published three phases.
The native moved-checkpoint GPU smokes reproduced the seeded outputs exactly
in these runs. PTR's six and Piano's eight fitted-artifact hashes passed;
their three saved reference audio/RPS examples are bit-identical to the
existing common reference, and all 48 readouts per method were independently
recomputed from the stored PSDs with zero discrepancy.

Verification records:
`neural_extension/hpn_verified.json`,
`neural_extension/ptr_verified.json`,
`neural_extension/piano_verified.json`.

The separate `results/noise_generator_paper/neural_extension_evidence.json`
adds 312 fidelity pairs and 144 control readouts while preserving the archived
raw evidence. All 115 frozen EONE/historical files passed the merger's before
and after hash checks; EONE statistics, method-source and control fields are
unchanged. The merger retains the old combined control evidence's external
example-root references and historical verification string rather than
assuming these fields have the native per-job schema.

LSD below is the mean of per-window four-draw means, in dB; lower is better.
The EONE row is the existing matched result, not a new fit.

| Matched method | All 26 windows | Standby | Transition | Cruise |
|---|---:|---:|---:|---:|
| Proposed stochastic model | 2.4751 | 2.2080 | 2.7772 | 2.4997 |
| Existing EONE adaptation | 2.4458 | 2.3775 | 2.7976 | 2.3835 |
| Published HPN comparator | 2.9928 | 3.6523 | 2.8289 | 2.7864 |
| DDSP-Piano | 3.6187 | 4.4864 | 3.2954 | 3.3741 |
| PTR | 5.4348 | 7.5079 | 5.3045 | 4.6899 |

The proposed generator has lower LSD than each new candidate in every regime,
but is **not** best overall: the existing matched EONE adaptation remains
slightly better on overall and cruise LSD. These fixed-budget, one-seed fits
do not establish universal architectural superiority, a few-shot learning
curve or perceptual equivalence. The control result establishes recovery of
the commanded blade-pass peaks, not realism or flight dynamics.

Recording warp and its matched variant remain archived for traceability but
are excluded from current comparison panels, rankings and listening players.
The current presentation has 15 arms: nine primary matched arms, five
additional pooled arms and the coherent-phase ablation. New candidates have
matched fits only; no artificial pooled result is created.

### Rendered delivery and verification

- Explainer:
  `writing/reports/2026-11-15_generator-baselines-explainer/index.html`.
  The self-contained HTML includes fitting procedures, synthesis equations,
  common-trajectory spectrograms and 48 embedded four-second audio players.
- Manuscript: `writing/papers/2026-11_drone-noise-generator/main.pdf`,
  with Quarto source in `main.qmd` and standalone generated `main.tex`.
- Overleaf-ready source bundle:
  `writing/papers/2026-11_drone-noise-generator/overleaf.zip`.
  It was extracted into a fresh directory and compiled with Tectonic without
  Quarto. The paper's `samples.html` is byte-identical to the explainer.

Preparation independently verified all 1560 current fidelity pairs, all 672
current control readouts, and the 48 exported players. Installed Brave via
CDP played all nine new method/regime combinations with networking disabled:
each decoded audio and spectrogram; changing the method paused the previous
player. All 48 native players report four-second durations and no media errors;
the selector contains no recording-warp arm. A 390-pixel viewport has no
document-wide horizontal overflow; plots retain their labelled local
horizontal scrolling. Browser proof:
`neural_extension/browser_verification.json`.

Both Quarto PDF and standalone ZIP builds completed. The new provenance-table
horizontal overflow was corrected, and PDF pages 1, 19 and 22 were visually
inspected. TeX still reports an 8.33-point overfull vertical box plus font and
underfull-box diagnostics; the inspected control/discussion page is not
clipped. The 22-page detailed draft preserves the requested explanations and
results but requires length reduction before an IEEE SPL submission. No claim
of submission-length compliance is made.

The focused control test passed before submission (three native slow tests
were deselected); each remote arm additionally passed its actual GPU
fit/move/reload smoke. The two final integration edits passed targeted Ruff.
No new fit or generator render occurred during document preparation.
No commit, push or wrap-up-submodule change was made in this extension.
Artifact hashes and complete verification:
`neural_extension/final_verification.json`.

