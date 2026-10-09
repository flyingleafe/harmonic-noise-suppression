# Handoff: the pyramid harmonic front end, the k ≤ 84 arms, and why level 3 hurts

Campaign of 2026-10-07/09 on `main`. Everything is committed; the two round-2
training runs have finished. Read this, then
`docs/experiments/pyramid-harmonic.md` (the campaign record, every number
with its job id), then `docs/pyramid-harmonic-frontend-design.md` (the design
note) and `docs/explainers/tap-conv.qmd` (the fused gather kernel).

## 1. What exists, in one paragraph

HPPNet's block order on a **nested multi-resolution STFT pyramid**
(`models.harmonic_ports.hppnet_pyramid.HPPNetPyramid`, registry name
`hppnet_pyramid`): four STFTs on one hop (16384/8192/4096/2048 at hop 512,
each kept from DC to 1200·2^l Hz), shared conv blocks on each level's native
grid, top-down FPN fusion, then **harmonic taps that are gathers** — for
hypothesis rate `r` the k-th harmonic is read at `k·r` Hz off the finest
level covering it, with one `(C_out, C_in)` weight per tap shared across
rates (`ProportionalTapConv` over `tap_conv`). Output grid is level 0's bin
width, 0.977 rev/s. Heads: `conv1x1` (variant C) or `RateConvLSTM` (variant
A: bidirectional ConvLSTM along time whose gates convolve along rate with a
±16-bin reach, sized from the telemetry's largest sustained slew). The CQT
HPPNet-L2 (`HPPNetOrig`) gained the same options (`head`, `harmonic_k_max`)
and a fused forward of its published `HarmonicDilatedConv`. Both families
run their taps through one shared fused Triton kernel.

## 2. Results (unified panel, `val/real_overall`, one seed)

| arm | front end · taps · head | best real_overall | job |
|---|---|---|---|
| `real_r4_hppnet_pyr_unified` | pyramid, k ≤ 32, 307-bin grid, conv1x1 | 2.69 @ 5 (noisy, ~2.9 typical) | cancelled r68 |
| `real_r4_hppnet_pyrlstm_unified` | pyramid, k ≤ 32, ConvLSTM | **2.17** @ 34 | cancelled r40 |
| `real_r4_hppnet_l2nolstm_unified` | CQT L2, k ≤ 9, conv1x1 | 2.45 @ 24 | finished r77 |
| `real_r4_hppnet_l2convlstm_unified` | CQT L2, k ≤ 9, ConvLSTM (±15 log-bins) | **2.16** @ 23 | cancelled r77 |
| `real_r4_hppnet_pyrk84_unified` | pyramid, k ≤ 84, 154-bin grid, conv1x1 | 2.67 @ 13 (W&B `rnzenk2s`, finished, 79 epochs) | `hppnet-pyrk84-r-f83ad1` |
| `real_r4_hppnet_l2k84nolstm_unified` | CQT L2 to 7.04 kHz, k ≤ 84, conv1x1 | 2.31 @ 6 (W&B `g8egadgl`, finished, 66 epochs) | `hppnet-l2k84nolstm-r3-939d4b` |

References: `hppnet_l2_r2_s0` 2.27 on the frozen split; `nv2_mixed_hppnet_l2`
2.25 on this panel. The published-form `real_r4_hppnet_l2_unified`
(CQT + FreqGroupLSTM, same regime) has **still never been run**.

Readings. (i) A ConvLSTM head is worth 0.3–0.5 on either front end and lands
both at the published L2's level without the per-bin LSTM. (ii) Extending
the taps to k ≤ 84 helps the CQT immediately (2.45 → 2.31) and the pyramid
not at all (2.69 → 2.67). (iii) Both k ≤ 84 conv-only runs peak early and
drift up (pyrk84 last epochs ~2.95, l2k84 ~2.5): the panel's
`best_real_overall` is an early-epoch alias on a noisy curve.

## 3. The diagnosis (the part that matters)

Measured on the frozen real split with the taps zeroed in the trained
checkpoints — no retraining (`scripts/tap_ablation_regimes.py`, uni-cpu job
`tap-ablation-regimes-660f0b`, per-frame PIT MAE, rev/s):

| zeroed | pyrk84 all · dregon · michaels | l2k84 all · dregon · michaels |
|---|---|---|
| none | 2.68 · 3.33 · 1.73 | 2.31 · 2.70 · 1.74 |
| taps k ≥ 61 | **2.22** · 2.59 · 1.69 | 2.31 · 2.58 · 1.92 |
| taps k ≥ 31 | 3.31 · 3.59 · 2.90 | 4.53 · 5.79 · 2.68 |
| taps k ≥ 16 | 4.65 · 5.65 · 3.20 | 11.1 · 17.3 · 2.03 |

1. The pyramid *does* read its high harmonics: k = 31–60 (level 2,
   2.4–4.8 kHz) are worth 1.1 rev/s, k = 16–30 another 1.3 — the same shape
   as the CQT (2.2 / 6.6). What it does not profit from is **level 3**
   (k ≥ 61, > 4.8 kHz): those rows cost it 0.46 overall and 0.74 on DREGON,
   while the CQT's are neutral. Level 2's gain and level 3's harm cancel —
   that is the null panel delta. Without its level-3 rows the pyramid conv
   arm (2.22) is ahead of the CQT conv arm (2.31) and at the ConvLSTM arms.
2. What the harm looks like (`notebooks/rps_tracking.ipynb` § "Round-2
   pyramid arms", six cruise frames): on every DREGON frame both full-k
   models put one of the two slow rotors (~75 rev/s) onto the fast pair
   (~85); zeroing k ≥ 61 brings it back. Michael's frames: all variants
   identical at 0.7.
3. Why: **the dominant high-order line belongs to a different rotor in the
   two rooms.** DREGON's per-rotor comb has strong lines at k = 14, 42, 70
   (motor-pole multiples); the bright ~6 kHz line is k ≈ 70. Rate-synchronous
   spectra: in room 1 (validation) it is rotor 0's, the fastest; in room 2
   (training: `free-flight`, `hovering`, `rectangle` room 2) it is rotor 3's,
   third fastest. The per-rotor output layers learn it as a rotor-identity
   cue and in room 1 it points at the wrong rotor.
4. Why the pyramid suffers more than the CQT: room 2 has no
   `motors_measured`, its labels are the **command**, which sits 0.5–3 %
   above the acoustic rate per rotor (fitted ×0.978 / ×0.969 / ×0.983 /
   ×0.995; room 1's measured track is a uniform ×0.9935). At k = 70 a 2 %
   label error is 110 Hz = 14 bins on level 3 (7.8 Hz) and 1.4 bins on the
   CQT (1.45 %/bin): the CQT's log axis makes a label error the same
   fraction of a bin at every harmonic, the pyramid's linear levels amplify
   it with k. The high taps are trained against positions that miss the
   lines by 10+ bins.
5. Refuted along the way: slew smearing in long windows (the k = 70 tap has
   the same 4.3–5 dB contrast at a 1 s window as at 16 ms, so level 3's
   short window buys nothing on this data); "no high harmonics in DREGON";
   tap-table / mask / normalisation bugs (all checked, § 5 of the campaign
   doc); grid-offset sensitivity (none at the output).
6. One config regression, not the cause but a confound: on the 154-bin grid
   `block_4`'s `2r` tap is masked for r > 73.2 rev/s (the whole DREGON
   cruise band); the 307-bin k ≤ 32 arm had it to 148.

## 4. Code map (what was added, where)

- `src/models/harmonic_ports/hppnet_pyramid.py` — `PyramidSTFT`,
  `upsample2_along_freq`, `build_tap_table`, `ProportionalTapConv`,
  `HPPNetPyramid` (`head`, `max_slew`, `k_max`, `k_sub`, `f_min/f_max`).
- `src/models/harmonic_ports/rate_convlstm.py` — `RateConvLSTM`; the
  per-step `gx[:, :, s]` slice backward was 85 % of its time, `gx.unbind(2)`
  once fixed it (1131 → 190 ms, T4).
- `src/models/harmonic_ports/tap_conv.py` — `tap_conv`, `shift_tap_table`
  (torch path: `(F, B·T, C)` layout, chunked GEMM);
  `tap_conv_triton.py` — fused fwd / dx (relaxed fp32 atomics) / dw kernels,
  dispatched on CUDA unless `TAP_CONV_TORCH=1`; `tests/models/test_tap_conv_triton.py`
  (CUDA only). A100 B=64: pyramid step 498 → 293 ms; CQT k ≤ 84 step
  551 → 239 ms (`HarmonicDilatedConv.FUSED_FROM = 16`; the published k ≤ 9
  stays on the branch sum). sm_75 (T4) is not a target.
- `src/models/harmonic_ports/hppnet_orig.py` — `HPPNetOrig(head=, harmonic_k_max=)`,
  `HarmonicDilatedConv` fused forward on the unchanged branch parameters
  (fp64-equivalent tests in `tests/models/test_harmonic_orig.py`).
- `conf/model/hppnet_pyramid{,_convlstm,_k84}.yaml`,
  `hppnet_l2_{nolstm,convlstm}.yaml`, `hppnet_l2k84_nolstm.yaml`;
  `conf/{loss,metrics}/salience_layers_pyr{,154}.yaml`; the six
  `conf/experiment/real_r4_hppnet_*_unified.{yaml,md}`.
- `scripts/bench.py` targets `hppnet_pyramid`, `hppnet_l2`, `rate_convlstm`,
  `tap_conv` (`BENCH_PROFILE=1`); `scripts/tap_ablation_regimes.py`.
- `src/zoo/frame_model.py` — a relative `results_root` now resolves against
  the repo root, so `zoo.load(exp, ckpt="<local name>")` works from
  `notebooks/`.
- `docs/explainers/tap-conv.qmd` (+ rendered HTML, gitignored): the kernel
  explainer. Written by a delegate; its `_rows` shared-memory rationale, the
  `index_add_` thread-mapping schematic and the pass counts are the
  delegate's estimates, not measurements.

## 5. Traps, all paid for once

- `omnirun submit` refuses a dirty tree; a detached worktree of
  `origin/main` at `/tmp/hns-submit` (with `.env` copied in) is the
  submission point. Resubmitting a resume while the first one is still
  provisioning produces two live runs on one W&B id — check `omnirun ps`
  before any second submit.
- vast rentals die at provisioning often; `--gpu-type A100` only (a P40
  rental ran at 10 s/it).
- `zoo.checkpoints(max_age_s=10**9)` serves a stale listing for experiments
  it has never seen; the notebook now refreshes once in that case.
- `best_real_overall.ckpt` on R2 is aliased on the median-smoothed score and
  moves while a run trains; the ablation script re-fetches it and the
  notebook cells use `cache=False`.
- Raw telemetry ranges (2–4 rev/s within a 1 s window) are not what the
  acoustics show; do not size windows from them.
- The synthetic `stochastic_*` panel views are anti-validation for ranking
  (prior session's rule); the pyramid conv arms score 30–40 on `static_*`,
  the ConvLSTM arm 14.

## 6. What is genuinely open

1. **Next pyramid arm**, not run: (a) `k_max 60` + `out_bins 307` (taps stop
   at 4.8 kHz, `2r` restored) — the cheapest test of § 3; (b) a top level at
   the 4096 window to 8 kHz (3.9 Hz bins, the four rotors' k ≈ 70 lines
   ~20 Hz apart are resolvable there, not at 7.8); (c) either with the
   ConvLSTM head, which is the only head that has reached 2.17.
2. **Labels**: room 2's command labels are 1–3 % off per rotor; a
   per-rotor refined label (or `rps_refined`, which Dmitrii distrusts for
   slew statistics but which may be right for scale) would remove the
   k-amplified misplacement that hits the pyramid. Worth one arm.
3. The rotor-identity cue (which rotor carries the k = 70 line) is a
   dataset property; any model with per-rotor layers can learn it. A
   permutation-invariant readout, or rotor-shuffling augmentation across
   recordings, is the structural answer.
4. `real_r4_hppnet_l2_unified` (published CQT + FreqGroupLSTM, this regime)
   as the proper anchor.
5. Kernel: the `dx` atomics (19 ms of the 34 ms tap backward) — an
   inverse-table gather would remove them if the taps need to be faster.

## 7. Records

- Campaign doc with every job id, bench and probe: `docs/experiments/pyramid-harmonic.md`.
- Regime ladder JSON: `results/regime_decomp/tap_ablation.json` on the
  cluster (`omnirun pull tap-ablation-regimes-660f0b` timed out three times;
  the numbers are in the job log and the campaign doc).
- Local ablated checkpoints: `results/<experiment>/drop_k61.ckpt`
  (untracked, rebuilt by `scripts/tap_ablation_regimes.py`).
- Probe scripts were throwaway (`/tmp/probe_pyr*.py`); their findings are
  in the campaign doc § "Round-2 diagnosis" with enough detail to redo them.
