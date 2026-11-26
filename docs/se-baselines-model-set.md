# SE baselines — model set

**Status:** decided by Dmitrii on 2026-11-25 · the missing models were ported
the same day (§ "Port status"), none is trained · supersedes § "Architecture
set" of `se-baselines-plan.md` (July 2026, 5 models). The training regime
(learning rate, schedule) and the validation split are still open; this
document fixes only *which models*.

## Scope

A model qualifies if it is a neural network that takes one microphone signal
and returns enhanced speech, with no reference sensor and no rotor-speed
input. The literature behind the list is in the bibliography under the project
tag `harmonic-noise-suppression`.

## Decided set (19 models)

"In repo" means model code only. Every speech-enhancement experiment config
(`f1_*`, `f2_*`, `a1_*`, `a2_*`) is legacy and refused by `train.py`, so each
model also needs a new config.

| # | Model | Source | Size | Public code | In repo | Work needed |
|---|---|---|---|---|---|---|
| | **Drone-specific** | | | | | |
| 1 | Edge-BS-RoFormer | Liu et al., Drones 2025, doi:10.3390/drones9060386 | — | — | yes (`a1_edge_bs_rof_fa` model) | config |
| 2 | SMoLnet-T | Tan & Khong, APSIPA 2024, doi:10.1109/APSIPAASC63619.2025.10848922 | 192k / 254k / 317k for 1 / 2 / 3 transformers | none | ported (`smolnet_t.yaml`) | experiment config |
| 3 | SMoLnet | Tan, Nguyen, Khong, APSIPA 2019, doi:10.1109/APSIPAASC47483.2019.9023324 | 224k | none | ported (`smolnet.yaml`) | experiment config |
| 4 | Adapter-FRCRN | Chen et al., IWAENC 2024, arXiv:2405.10022 | 0.3M trainable | adapters: none; backbone: ModelScope FRCRN | ported (`adapter_frcrn.yaml`; `frcrn.yaml` for the backbone control) | experiment config; stage and select the DNS-pretrained backbone |
| 5 | DroFiT | Lee et al., Interspeech 2026, arXiv:2509.16945 | 168k | demo page only | ported (`drofit.yaml`) | experiment config |
| 6 | DroneAudioNet | Gupta et al., arXiv:2608.00875 | 26.4M trained of 238M | `github.com/DroneAudioNet/DroneAudioNet-Code` (built on AudioSep; no licence) | ported (`droneaudionet.yaml`) | experiment config; stage the weights |
| | **Generic models with drone-noise evidence** | | | | | |
| 7 | DCUNet | best of 12 in Mukhutdinov et al., IEEE Access 2023, doi:10.1109/ACCESS.2023.3253719 | 3.53M there | — | yes (`a1_baseline_dcunet` model) | config |
| 8 | DCCRN | same comparison | — | — | yes | config |
| 9 | DPTNet | same comparison (runner-up) | 2.6M in the SMoLnet-T comparison | — | yes | config |
| 10 | DPT-FSNet | Dang, Chen, Zhang, ICASSP 2022, arXiv:2104.13002; best of 7 in Chen et al., ICINC 2025, doi:10.1117/12.3056881 | 0.88M there | `github.com/dangf15/DPT-FSNet` (no licence file) | ported (`dpt_fsnet.yaml`) | experiment config |
| | **Network behind a harmonic-noise filter** | | | | | |
| 11 | cMPDR + DNN | Bologni et al., EUSIPCO 2026, arXiv:2602.12986 | small CRNN or ULCNet | `github.com/narrietal/cMPDR_DNN` | ported (`cmpdr_crn.yaml`, `cmpdr_ulcnet.yaml`, `cmpdr_filter.yaml`) | experiment config |
| | **Generic state of the art, discriminative** | | | | | |
| 12 | TF-GridNet | Wang et al., TASLP 2023 | 8.38M | ESPnet | yes (port identical to ESPnet V1) | config |
| 13 | TF-Locoformer | Saijo et al., IWAENC 2024 | 15.0M (medium) | Apache-2.0, single file | ported (`tflocoformer.yaml`) | experiment config |
| 14 | BSRNN | Yu et al., Interspeech 2023 | 32.0M at 16 kHz (URGENT 2026 baseline) | `urgent-challenge/urgent2026_challenge_track1` (Apache-2.0) | ported (`bsrnn.yaml`) | experiment config |
| 15 | MP-SENet | Lu et al., Interspeech 2023 | 2.26M (current release) | official | ported: current release with discriminator and losses (`mpsenet_official.yaml`); the 1.71M hybrid stays for stored checkpoints | two-optimizer training loop; experiment config |
| 16 | HTDemucs, fine-tuned | Rouard et al., ICASSP 2023 | — | official | yes | config; reported as the pretrained-prior control, not as speech-enhancement state of the art |
| | **Generic state of the art, generative** | | | | | |
| 17 | SGMSE+ | Richter et al., TASLP 2023 | — | `sp-uhh/sgmse` | yes (identical to official) | checkpoint fix first (below), then config |
| 18 | StoRM | Lemercier et al., TASLP 2023 | 2 × 27.8M | MIT | ported (`storm.yaml`) | training loop for generative models; experiment config |
| 19 | FlowSE | Lee et al., ICASSP 2025 | 65.6M | `seongq/flowmse` (no licence) | ported (`flowse.yaml`) | training loop for generative models; experiment config |

Notes on single rows:

- **SMoLnet-T (2).** The paper does not state the loss, the sampling rate or
  the number of attention heads. SMoLnet (3) used mean-squared error on the
  complex spectrum at 16 kHz with a 2048-sample window. The published margin
  over SMoLnet is small: +2.3 dB SI-SDR at −32 dB, +0.02 ESTOI at −23 dB.
- **DPTNet (9).** The July plan ruled it out on Paper 1's result. It is back
  by the 2026-11-25 decision.
- **DCCRN (8).** Its original reason was to be the control for the
  rotor-conditioned DCCRN, which is set aside. It stays as a plain baseline.
- **DPT-FSNet (10).** Chen et al. tested at 0 to 15 dB only.
- **SGMSE+ (17).** The July score ("non-viable", F1 Finding 5) is void: the
  exponential-moving-average weights were not in the saved `state_dict`, so
  evaluation ran on untrained weights. A fix with a regression test exists,
  uncommitted, in `.worktrees/se-baselines`.

## Port status (2026-11-25)

Each port is model code, a model config and a test file. None has been
trained here. "Parity" is the largest difference between the port and the
upstream code on the same weights and the same input, relative to the largest
upstream value, in single precision unless stated.

| # | Model | Config (`conf/model/`) | Parameters: port / published | Parity with upstream | Published weights that load |
|---|---|---|---|---|---|
| 2 | SMoLnet-T | `smolnet_t.yaml` | 282,198 / 317k (three transformers) | no upstream code | none published |
| 3 | SMoLnet | `smolnet.yaml` | 224,194 / 224k | no upstream code | none published |
| 4 | Adapter-FRCRN | `adapter_frcrn.yaml`, `frcrn.yaml` | 287,096 trainable / ≈0.3M; 14,030,220 total | backbone: 0 on published weights; no reference for the adapters | ModelScope DNS-pretrained FRCRN |
| 5 | DroFiT | `drofit.yaml` | 167,884 / 168k | no upstream code | none published |
| 6 | DroneAudioNet | `droneaudionet.yaml` | 26,448,042 / 26.4M trained | published drone weights: 6e-6 in eval; AudioSep/random: up to 4.3e-5 in train | the authors' fine-tuned checkpoint; the AudioSep base |
| 10 | DPT-FSNet | `dpt_fsnet.yaml` | 879,460 / 0.88M | 3e-6 | none published |
| 11 | cMPDR + DNN | `cmpdr_crn.yaml`, `cmpdr_ulcnet.yaml`, `cmpdr_filter.yaml` | CRNN 181,945 / 0.46M; ULCNet 685,147 / 0.68M | filter 5e-15 in double precision; CRNN 3e-6; ULCNet 1e-5 in 99.99 % of bins | none published |
| 13 | TF-Locoformer | `tflocoformer.yaml` | 14,973,346 / 15.0M | 6e-7 | DNS 2020 checkpoint |
| 14 | BSRNN | `bsrnn.yaml` | 33,628,912 / "32.0M" | 0 | URGENT 2026 baseline checkpoint |
| 15 | MP-SENet, current release | `mpsenet_official.yaml` | generator 2,263,372 / 2.26M; discriminator 181,650 | generator 1e-6; discriminator 0; losses 4e-6 | `g_best_vb`, `g_best_dns` |
| 18 | StoRM | `storm.yaml` | predictor 27,342,350 and score network 27,776,038 / about 27.8M each | 0 for outputs, loss and the sampler at 3 and 10 steps | not tried (1.1 GB) |
| 19 | FlowSE | `flowse.yaml` | 65,590,822 / not printed in the paper | 0 for the network, three training steps and the sampler | not tried (1.2 GB) |

What to know before training each:

- **SMoLnet-T.** Built from the paper's equations it is 5 to 11 % smaller than
  published (182k / 232k / 282k against 192k / 254k / 317k); the readings
  tried are in the module docstring. Assumed, because the paper is silent:
  8 attention heads, 16 kHz, mean-squared error on the complex spectrum.
- **SMoLnet.** A long input is processed in non-overlapping 9-frame blocks, as
  the paper trains; `context_frames: null` runs it fully convolutional in time.
- **Adapter-FRCRN.** Set `checkpoint` to the DNS-pretrained backbone for
  this comparison; `null` builds random weights. The default trains 66,616
  adapter and 220,480 attention parameters. It zeros the up-projection only:
  each adapter starts as an identity residual without killing its weight
  gradients. Frozen normalisation layers stay in eval mode.
- **DroFiT.** Reconstructed from the paper alone. One free width was set to
  hit the published count, and two of the paper's four ablation counts are
  not reproduced.
- **DroneAudioNet.** The authors' code has no licence, so the residual head is
  written from the paper; the trunk is adapted from AudioSep (MIT). The text
  query is a stored embedding, so nothing of CLAP runs. The model works at
  32 kHz inside. The fine-tuned weights carry no stated licence.
- **DPT-FSNet.** Upstream has no licence; the port is written from the paper.
  The reference parity run uses real/imaginary STFT components as specified
  in the paper, not the magnitude/phase inputs of unmodified upstream.
- **cMPDR + DNN.** The filter estimates its cyclic frequencies once per
  utterance, from at most 2 s, assumes they stay constant, and processes
  nothing above 2.5 kHz. On a swept fundamental the upstream code itself
  filters 12 bins against 33 on a stationary one. The released CRNN has 182k
  parameters although the paper's table says 0.46M.
- **TF-Locoformer.** Medium configuration, following the released code (no ½
  factor in the Macaron blocks). An all-zero input gives NaN, as upstream.
- **BSRNN.** The URGENT 2026 baseline (196 channels, 6 layers), not the
  paper's 16 kHz model. Upstream's "32.0M" is in units of 2^20 and leaves out
  the GroupNorm parameters. The published checkpoint does not preserve level;
  upstream rescales every output file.
- **MP-SENet.** Upstream's attention and recurrent layers mix the utterances
  of a batch (a bug its README admits). The default
  `transformer_layout: per_utterance` removes the mixing and equals upstream
  at batch 1; `upstream` reproduces it.
- **StoRM.** The moving-average weights are part of `state_dict`. The config
  takes the paper's sampler (50 steps with a corrector) and the code's Hann
  window and loss weights. Upstream trains the predictor alone before the
  joint stage.
- **FlowSE.** Upstream has no licence: the flow path, loss and sampler are
  written from the paper, on the NCSN++ backbone of `models.sgmse`. Five Euler
  steps. The moving-average weights are part of `state_dict`.

Checkpoint averaging needs an optimizer-step policy before training. StoRM's
moving average is one update behind the optimizer in a normal training step.
FlowSE matched upstream with one step per forward; that equivalence does not
extend to gradient accumulation or skipped steps.

What the ports do not include:

- Experiment configs: none is written, because the regime is open.
- A training loop for StoRM and FlowSE. Like SGMSE+ they return their loss
  from `forward(mix, target)`, which `train.py` does not drive;
  `scripts/train_sgmse.py` hard-codes the SGMSE+ config and clips gradients,
  which neither upstream does.
- A two-optimizer loop for MP-SENet and its discriminator.
- Mixed-precision or GPU verification. SMoLnet (2019) and DroFiT report
  double-precision training; the TF-Locoformer DNS recipe uses fp32.

Weights are under `.cache/pretrained/<name>/` (git-ignored); cluster jobs
need them staged. The upstream clones that the slow parity tests read are
under `/tmp/<name>_upstream` and do not survive a reboot.

Verification: all 12 newly ported models passed their CPU test files in
separate processes, and the integrated factory listing passed all five
`tests/models/test_registry_listing.py` checks. A combined run including the
legacy SE tests was killed at the worker's 4 GiB memory cap before pytest
produced a summary; it is not a combined-suite pass.

## Set aside for now

- **Rotor-conditioned models**: Gulli et al. 2025 (Wave-U-Net, DCCRN, DCUNet
  with measured rotor speed; repo variants `b1_*`, `b2_*`).
- **Generative optionals**: Schrödinger bridge (Jukić et al. 2024), Diffusion
  Buffer (in repo), AnyEnhance.

## Not in the set

- **Cited only** (drone noise, single channel, neural, no code or weak
  evidence): speech-harmonic gating transformer (Wang et al., ICIC 2025), GAN
  noise subtraction (Premachandra & Kunisada 2023), diffusion enhancer for ASR
  (Miura et al., ICASSP 2026), two-stage residual network (Lei 2025),
  complex-spectrogram CNN (Kim & Kim 2020), SEGAN for ego-noise (Spadini et
  al. 2019); the remaining models of the two comparisons (FC, VAE, PHASEN,
  Wave-U-Net, Demucs, SEGAN, Conv-TasNet, DPRNN; Uformer, DPCRN, FSPEN, GTCRN).
- **Left out**: Mamba models (SEMamba, USEMamba, dual-branch Mamba;
  `mamba-ssm` builds from source only, a risk on Isambard's aarch64);
  ZipEnhancer, SepReformer, MossFormer2, CMGAN, DB-AIAT, FullSubNet,
  SepFormer, MetricGAN (same class as rows 12–15); FlowSE of Wang et al.,
  UNIVERSE++, LLaSE-G1, GenSE (vocoder or language-model generators).
- **Not valid for this list**: D-CrossNet (Zhang et al. 2026) needs a
  noise-reference microphone and a parabolic reflector; multi-microphone
  methods; classical methods.

## Outside this list (unchanged)

- Anchors: noisy input and Wiener filter (`scripts/se_eval.py`).
- Proposed, not decided: harmonic subtraction from measured or
  tracker-estimated rotor speed, assembled from `src/tracking/decompose.py`
  and `src/experiments/kalman_harmonic/`. Closest published precedent on
  DREGON: the speed-correlated harmonics cancellation in DOANet (Qayyum et
  al., EURASIP JASM 2020), which takes the first harmonic as 0.98 × rotor
  speed and subtracts up to ten band-passed harmonics.

## Check for other machine-noise work (2026-11-25, quick)

Question: are there monaural neural methods aimed at engine, industrial or
aircraft noise, not drones, that the survey missed? Searched: 14 keyword
queries, 25 title searches, 7 web searches; abstracts only.

| Noise | Found | What it is |
|---|---|---|
| Aircraft cockpit | Huang & Wu, Chinese J. Electronics 2023 and Aerospace 2024 | LSTM mask with teacher–student training; temporal-convolution network with attention |
| Airport apron | Zhang et al., ISAIC 2022 | Conv-TasNet with added attention |
| Industrial | Bi-DecoNet, Guan et al., Electronics Letters 2026 | 0.54M dual-decoder transformer; scored on VoiceBank+DEMAND and TIMIT |
| MRI scanner | Chou et al., arXiv:2608.16125 | test of three off-the-shelf enhancers, no new model |
| Helicopter | 2023 headset papers; Zhou et al. 2025 | classical filtering; active noise control |
| Engine | cMPDR + DNN | already row 11 |

Nothing found for dental drills, sirens, ships, tanks, construction machinery
or electric-vehicle motors. None of the finds is built around the harmonic
structure of the noise; they apply generic architectures to domain noise and
publish no code. None was added.

Adjacent class, not added: networks designed for extremely low SNR with any
noise — UNetGAN (Hao et al., Interspeech 2019), DisCoGAN (Shetu, Habets,
Brendel, ICASSP 2025), NRSRNet (Zhang et al., Applied Acoustics 2026).

## Open points

- **Budget.** The Isambard plan budgets 6 models × 3 regimes = 2,700 GPU-hours
  at 150 per run. Nineteen models × 3 regimes at that rate is 8,550. Rows 2,
  3, 5 and 11 are small models and should cost far less (not measured). Rows
  4 and 10 are small in trained parameters but not in compute: row 4 runs the
  full frozen FRCRN, and row 10 needs 17 GMACs per second of audio in Chen et
  al.
- **Training regime and validation split** are not decided.
