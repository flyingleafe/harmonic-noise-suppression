# `src/tracking` performance — knobs, kernel paths, measured tables, the guard

**Status:** record of the issue #16 optimization campaign and the 2026-08 consolidation (R1/R2) ·
**Moved from** `src/tracking/AGENTS.md` (2026-09) · **Earlier profiling:**
`docs/vk-order-tracking-design.md` §8 (2026-07-20, pre-consolidation fast paths) ·
**Guard driver:** `scripts/tracking_ref.py` · **Reference:** `results/tracking_ref/`

## Knobs

The demodulation transforms dominate `pi_kalman_refine` and `vk_envelopes`. Every one of them is the same kernel — `dsp.zoom_bands`, reached through the one driver `dsp.demod` — and these knobs control it, all with measured defaults:

| Knob | Default | What it does |
|------|---------|--------------|
| `TRACKING_FFT_WORKERS` (env), `thread_pool(n)` (context manager), `pi_kalman_refine(threads=n)` | 1 (or `OMP_NUM_THREADS`) | Torch CPU threads, clamped to the process's CPU affinity. The name is historical — it is the tracking stack's ONE thread knob. The default stays 1 because oversubscribing on a restricted Slurm allocation thrashes, so offline and interactive callers must opt in. Bit-identical. |
| `TRACKING_DEMOD_BUDGET_MB` (env), `dsp.DEMOD_BUDGET_BYTES` / `dsp.DEVICE_BUDGET_BYTES` | 64 MB on CPU, 512 MB elsewhere | Working set of one demodulation flush, i.e. how many harmonics share a transform. On CPU this is a **cache** knob, not a memory-headroom knob: channels are already batched jointly, so bigger flushes amortize nothing and only leave cache. On a device it is the memory bound instead (the full `(8, 40, 256000)` complex64 bank is 655 MB). Bit-identical. |
| `TRACKING_DEVICE` (env), `dsp_config(device=...)` | `cpu` | Torch device — `cuda` moves the carriers, the products and the transforms onto the GPU, and switches the peel to its clip-long tiling. |
| `vk_tracking.LS_TILE_BYTES` | 256 KB | Working set of one gain-fit tile in the peel. A CPU cache knob: the tile is streamed six times, so it must stay resident. Tiles are whole 0.25 s blocks, so the floor is one block; off CPU the tile is the whole clip. Bit-identical. |
| `TRACKING_PAD` (env), `dsp_config(pad=...)` | `exact` | `fast` grows the envelope grid to the next 5-smooth length. NOT bit-identical (a zero tail lengthens the circular convolution), so it is opt-in. |

There is **no backend knob**. The scipy/pocketfft transform and the numpy peel core were deleted in the 2026-08 consolidation: torch runs the same arithmetic on CPU and on CUDA, so the second implementation bought a summation order and a maintenance surface, nothing else.

Three properties the optimization campaign relies on:

- The off-comb noise probe is sliced out of the on-comb spectrum (a constant frequency offset is a pure bin shift), so a demodulation costs one forward FFT, not two. The probe offset is snapped to the bin grid.
- Envelopes are **complex64** — the transform's own precision. Code that needs float64 (variances, gates, the Kalman) uses `_abs2` / `_increment_phase`, which compute in float64 from the components rather than widening the bank.
- The `band_env = 0.45` band always fits inside the decimated Nyquist range (`floor(0.45 n) <= (n - 1) // 2` for every `n`), so the zoom identity holds on degenerate grids too and the kernel needs no short-clip special case.

## Which kernel path

`zoom_bands` has two ways to lift a band out of the spectrum, and the choice is automatic:

- **Uniform** (one half-width and one shift for every row): two slices — `narrow` at DC, a cached `index_select` when the band is shifted. No index tensor built per call, which is what keeps the many small demods of `pi_kalman_refine` cheap.
- **Per row** (the `band_mode="k_scaled"` / `probe_mode="clean"` paths): one `gather` with a cached `(rows, n_envp)` wrapped index plus a keep mask.

A caller that hands over one entry per track when every entry is the same (`demod_bank`'s fixed probe offset is the common case) is collapsed onto the uniform path by `_collapse` — same bins, cheaper kernel.

## CPU, before and after the consolidation

Measured on the frozen 16 s / 8-mic clip, this laptop, `scripts/tracking_ref.py --bench --bench-vk`. "before" is the scipy transform plus the numpy peel at the same thread count:

| Stage | before, 1 thread | after, 1 thread | before, 4 threads | after, 4 threads |
|-------|------------------|-----------------|-------------------|------------------|
| `zoom_lp_decimate` (8, T) | 11.2 ms | 12.2 ms (+9 %) | 7.0 ms | 5.9 ms (**-16 %**) |
| `demod_bank` K=40 | 541 ms | 630 ms (+16 %) | 395 ms | 307 ms (**-22 %**) |
| `vk_envelopes` | 20.4 s | 18.7 s (**-8 %**) | 15.4 s | 13.9 s (**-10 %**) |
| `ls_project_envelopes` | 2.41 s | 2.59 s (+7 %) | 2.72 s | 2.58 s (**-5 %**) |
| `pi_kalman_refine` (full) | 3.88 s | 4.30 s (+11 %) | 2.91 s | 2.31 s (**-21 %**) |
| one peeled application, end to end | 24.0 s | 20.1 s (**-16 %**) | — | — |

The one-thread column is where torch was expected to lose — its per-call tensor/gather/copy overhead against pocketfft's single-thread SIMD, which the pre-consolidation table below measured at +46 % on `pi_kalman_refine`. Caching the band indices and taking the uniform-band slice path bought most of that back (+11 %). At four threads the consolidated path wins across the board. `vk_envelopes` improves at both counts because its coupling cross-phasors now go through the same kernel as everything else.

`pad="fast"` still earns its keep only when the bad factor is in `n_env` (1009 -> 1024) and is worth **nothing** when it is in `stride` — `n_pad` is a multiple of `stride` by construction, so a stride like `round(44100 / 62.5) = 706 = 2 * 353` poisons every admissible length. The 44.1 kHz Bluestein trap is fixed by choosing an `fs_env` whose stride factorizes, not by padding (the torch transform, which used to be the other fix, is now the only transform).

## On a GPU

```bash
TRACKING_DEVICE=cuda python <whatever>
```

Re-measured **after** the consolidation on `uni-gpushort` (job `r3-gpu-fdbf49`, 2026-08-06), same
frozen clip, `--bench --bench-devices cpu,cuda --bench-workers 1,4 --bench-vk`. All four columns
come from ONE job on ONE node, so read across a row and never against the laptop table above — the
node's CPU is the slower machine.

| Stage | cpu, 1 thr | cpu, 4 thr | cuda, 1 thr | cuda, 4 thr | cuda vs best cpu |
|-------|-----------|-----------|-------------|-------------|------------------|
| `zoom_lp_decimate` (8, T) | 17.3 ms | 5.7 ms | 3.4 ms | 3.3 ms | 1.7x |
| `_demod_bank` K=40 | 830.6 ms | 299.0 ms | 54.6 ms | 16.4 ms | **18x** |
| `vk_envelopes` | 23.5 s | 13.1 s | 10.7 s | 10.8 s | 1.2x |
| `ls_project_envelopes` | 5.85 s | 5.83 s | 264 ms | 183 ms | **32x** |
| `pi_kalman_refine` (full) | 6.14 s | 2.26 s | 415 ms | 396 ms | **5.7x** |

The consolidation did not cost the GPU anything. Against the pre-consolidation cuda leg (job
`bash-01dc95`, same node class) every stage held or improved: `_demod_bank` 24.4 -> 16.4 ms,
`pi_kalman_refine` 0.47 -> 0.40 s, `ls_project_envelopes` 0.44 -> 0.18 s, `vk_envelopes`
11.1 -> 10.7 s. That job's scipy/CPU column also lands on top of this job's torch/CPU column
(16.8 / 849 ms / 5.85 / 26.0 / 4.76 s against 17.3 / 831 ms / 6.14 / 23.5 / 5.85 s), which is the
laptop table's conclusion measured a second time: torch on CPU is the scipy path's equal, so
deleting the second implementation cost nothing.

Two things this run measured that the old one could not, both from the thread columns:

- **The peel is memory bound, and now it is proven.** `ls_project_envelopes` is the one stage whose
  CPU time does not move with thread count at all (5.85 s -> 5.83 s). Its 32x on a GPU is bandwidth,
  not arithmetic — exactly what the peel section below claims, and the reason no demod knob touches it.
- **`vk_envelopes` is now the whole cost, and the transform is not why.** On cuda it too ignores the
  thread count (10.7 s -> 10.8 s) and it is **82 %** of one end-to-end application (11.3 s of 13.8 s
  in the self-check leg). What is left in it is the scipy banded Cholesky plus the numpy cross-phasor
  host work — the two terms named under "What is left after the demod", neither of which is a
  transform. Any further GPU work on this stack starts there.

The same job ran `--self-check --device cuda`, which is how the CUDA path's CORRECTNESS is verified
(the cpu/exact and cuda legs in one process). It **passed**, so the R1/R2 consolidation introduced no
device bug at the numpy in/out seams or in the device-keyed caches:

| array | max abs | of scale | verdict |
|-------|---------|----------|---------|
| `env_z` | 3.44e-8 | 2.21e-7 | close |
| `env_x` | 1.40e-6 | 3.37e-6 | close |
| `env_x_ls` | 1.07e-6 | 9.05e-6 | close |
| `r_next` | 5.10e-6 rev/s | 5.52e-8 | close |
| `env_valid` | 0 flips | — | identical |
| `env_bw_track`, `env_t_env`, `r0` | 0 | — | identical |

End to end in that leg: 28.05 s on cpu/exact against 13.79 s on cuda (`vk_envelopes` 16.5 -> 11.3 s,
`ls_project_envelopes` 5.88 -> 0.37 s).

## The peel (`ls_project_envelopes`)

It has no transform in it at all, so no demod knob touches it, and after the demod work it was the largest remaining term: 6.9-11 s per application, a Python loop over 160 tracks. The fix is not to break the loop — tracks are fitted **sequentially against a running residual** and reordering them into independent fits is measurably worse (see the docstring) — but to see that everything *inside* one iteration is independent: 64 blocks x 8 channels.

`_ls_project` is the one core. It runs the carrier recursion on the device, keeps every host sync out of the track loop (the "track is all zero" test and the clip counter are precomputed or accumulated on the device), and sweeps tiles of `LS_TILE_BYTES` so the five block sums and the residual update stay cache-local — the tiling insight the deleted numpy core was built on, carried across. The naive form streamed six clip-long float64 arrays through DRAM per track, ~40 GB of traffic for the frozen clip. Off CPU the tile is the whole clip, because a GPU has no cache to block for and pays for kernel launches instead.

The peel's own floor is the residual traffic itself: it is read twice and written once per track, and no reordering that keeps the sequential guarantee can avoid that.

## What is left after the demod (profiled before the consolidation, 4 threads)

| Call | Total | Transform | Not the transform |
|------|-------|-----------|-------------------|
| `pi_kalman_refine` | 3.36 s | 2.19 s FFT + 0.52 s carrier `mul` | **0.15 s** — gating, observations, Kalman/RTS all together |
| `vk_envelopes` | 13.35 s | 6.00 s FFT | 3.98 s banded Cholesky, 2.04 s cross-pair phasors + bookkeeping, 0.66 s seam conversions |
| `ls_project_envelopes` | 2.5 s | none | 0.62 s residual update, 0.53 s `<resid, p/q>`, 0.48 s basis, 0.44 s Gram, 0.22 s Re/Im split |

So the "Python-side gating becomes the bottleneck" worry from issue #16 does **not** materialize: `_rw_kalman_rts` is 22 ms and the whole gating/observation layer is 4 % of `pi_kalman_refine`. What is left is the coupled-group **banded Cholesky**, the dominant term of `vk_envelopes` once the FFT is on a GPU. The banded solve stays on scipy on purpose: `torch.linalg` has no banded Cholesky, the systems are `g * n_env` up to ~64000 unknowns so dense is out, and a block-tridiagonal solver of our own is exactly the bespoke code this project does not write for a 4 s term.

One smaller lever, measured but not taken: the coupling cross-phasors are still built in numpy and shipped to the device per flush (2.0 s of host work plus ~10 GB of transfer on the GPU run — fusing them into `demod`'s recursion is the same pattern).

## The guard

`scripts/tracking_ref.py` diffs a frozen 16 s DREGON cruise window (`results/tracking_ref/`) array by array:

- `--compare [--exact]` against the stored `.npz`. Tolerance mode uses the per-array `TOL` bar (scale-relative for the envelopes, an absolute 1e-4 rev/s for `r_next`, **zero flips** for the gate masks).
- `--self-check --device cuda` runs the cpu/exact leg and the selected device in ONE process and diffs them — no 100 MB `.npz` to ship, which is how the GPU is verified.
- `--bench [--bench-devices cpu,cuda] [--bench-workers 1,4] [--bench-vk] [--bench-json PATH]` reports per-stage wall times.

The stored `.npz` was captured on the scipy transform and the numpy peel, so **`--exact` no longer passes and is not expected to**: the consolidation changed the summation order of every transform. Tolerance mode does pass, and the measured moves against that reference are

| array | max abs | of scale | bar |
|-------|---------|----------|-----|
| `env_z` | 2.36e-8 | 1.5e-7 | 1e-5 |
| `env_x` | 3.03e-7 | 7.3e-7 | 1e-3 |
| `env_x_ls` | 1.88e-7 | 1.6e-6 | 1e-3 |
| `r_next` | 6.57e-6 rev/s | — | 1e-4 rev/s |
| `env_valid` | 0 flips | — | 0 flips |

which is the same order the old scipy->torch backend swap showed — the arithmetic did not change, only where it runs.

`env_x` carries a looser bar than `env_z` for a reason: at `bw_hz = 1` the VK normal equations have `rho^2 ~ 4e5` and a condition number ~1e7, so the solve amplifies the demod's complex64 rounding by one to three orders depending on the clip. `r_next` moves 6.6e-6 rev/s and no gate flips — and `r_next` plus the gates are what the tracker consumes.

