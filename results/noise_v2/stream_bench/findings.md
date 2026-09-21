# `noise_v2` stream throughput

Numbers: `bench.json`, produced by `bench.py` (which documents the harness).
Machine: laptop, Intel Core Ultra 7 165U, 14 logical cores, `MemoryMax=10G`.
Policies: `conf/online_mix/_noise_v2_smoke.yaml` (two `noise_v2` sources at
0.4 / 0.4 plus silence 0.2) against `conf/online_mix/rig_easy_5050.yaml` (the
same weights with two `stochastic` sources), both at `render_reuse: 48`.
Windows are 2 s unless stated; `build_noise_stream` iterated through a torch
`DataLoader`.

## Per render (one window, reuse bypassed)

| source | mics | window | median ms | CPU-s per audio-s |
| --- | --- | --- | --- | --- |
| `noise_v2` | 1 | 2 s | 1373 | 0.69 |
| `noise_v2` | 8 | 2 s | 1973 | 0.99 |
| `noise_v2` | 8 | 4 s | 4455 | 1.11 |
| `stochastic` | 8 | 2 s | 416 | 0.21 |
| `stochastic` | 8 | 4 s | 676 | 0.17 |

The v2 renderer costs **4.7x** the stochastic family's at 8 mics and is
**linear in window length** (0.99 -> 1.11 CPU-s per audio-s from 2 s to 4 s),
so the 4 s arms pay twice the 2 s figure. It is only 1.4x cheaper at 1 mic than
at 8: the cost is the per-(rotor, order) trig pass on the 64 kHz work grid,
which every microphone shares, not the per-mic accumulation.

## Stream, 64 windows of 2 s (the spec'd run, build included)

| source | mics | workers | samples/s | peak RSS/worker |
| --- | --- | --- | --- | --- |
| `noise_v2` | 1 | 0 | 14.0 | 641 MB |
| `noise_v2` | 1 | 4 | 6.6 | 431 MB |
| `noise_v2` | 8 | 0 | 12.1 | 683 MB |
| `noise_v2` | 8 | 4 | 5.6 | 475 MB |
| `stochastic` | 8 | 0 | 49.2 | 695 MB |
| `stochastic` | 8 | 4 | 13.0 | 484 MB |

**These rows measure startup, not throughput.** At `render_reuse: 48`, 64
windows contain one or two renders per source; what dominates is each worker's
pipeline build (reading the fits, `dload:rps-traj-fits`, the first render), and
four workers pay it four times over a run too short to amortise it — which is
why `num_workers: 4` is SLOWER than 0 here. A training run never sees this.

## Stream, 384 windows of 2 s, 8 mics (steady state)

| source | workers | samples/s | audio-s/s | peak RSS/worker |
| --- | --- | --- | --- | --- |
| `noise_v2` | 0 | 16.7 | 33.4 | 693 MB |
| `noise_v2` | 4 | 33.7 | 67.4 | 475 MB |
| `stochastic` | 0 | 71.6 | 143.2 | 701 MB |
| `stochastic` | 4 | 71.9 | 143.8 | 485 MB |

`noise_v2` is **4.3x slower** than the legacy pool per worker and **2.1x** with
four workers (the laptop's four workers do not scale linearly; the stochastic
pool is already memory-bandwidth bound at 72 samples/s and gains nothing).
**Peak RSS is the same source to source** — 475 vs 485 MB per worker — so the
v2 renderer is a CPU cost and not a memory one; a 12-worker node needs the same
~6 GB of loader RSS either way.

Decomposing the single-process steady rate (8 mics, 2 s): 59.8 ms per sample,
of which the render is `0.8 / 48 x 1973 ms = 32.9 ms` (the 0.8 is the two
weighted `noise_v2` sources; the silence source never renders) and the
remaining **26.9 ms is the pipeline itself** — frame construction, the rps
event series, the tensor copy. So at reuse 48 the render is 55% of the loader's
work at 2 s, and ~65% at 4 s (74.3 ms render against ~40 ms of pipeline).

## Is a batch-128, 12-worker A100 run render-bound at `render_reuse: 48`?

**Yes**, at the arms' 4 s windows, unless the A100 step exceeds ~1.2 s.

Per worker at 4 s / 8 mics: `0.8 / 48 x 4455 ms = 74.3 ms` of render plus ~40 ms
of pipeline = ~114 ms per sample, i.e. 8.8 samples/s. Twelve workers supply
~105 samples/s, so a batch of 128 arrives every **1.22 s**, of which **0.79 s
is the renderer**. A DCCRN/DCUNet-class step on a batch of 128 x 4 s x 8 ch is
far under a second on an A100, so the GPU waits on the loader and the loader
waits on the renderer. (Caveat: these are laptop-CPU walls. The verdict holds
for any node whose per-core speed is within ~2x of this machine's; it is the
ratio of 0.79 s of render supply to the step time that decides it, so a node
with much faster cores would need the same arithmetic redone.)

What would clear it:

* **`render_reuse` ~190** brings the render share to ~19 ms per sample, i.e.
  ~59 ms per sample and ~203 samples/s on 12 workers — a batch of 128 every
  0.63 s, with the render down to 30% of the loader. Reuse cannot go much
  further: below ~28 ms per sample the *pipeline* binds, not the renderer.
* **A preset bank with many entries** is what makes a high reuse safe: at reuse
  48 and `render_pool = 4 x reuse` the stream already circulates 192 distinct
  clips per source, and raising reuse without raising the diversity of the rigs
  behind them buys throughput with repetition. The bank builder is the next
  step for exactly this reason.
* Rendering at 1 mic costs only 1.4x less, so dropping channels is not a lever.

## Reproduce

```
systemd-run --user --scope -p MemoryMax=10G \
    .venv/bin/python results/noise_v2/stream_bench/bench.py
```
