"""Throughput of the `noise_v2` online-mix source against the legacy stochastic one.

Run from the repository root::

    systemd-run --user --scope -p MemoryMax=10G \\
        .venv/bin/python results/noise_v2/stream_bench/bench.py

What is measured, and why exactly this:

* ``build_noise_stream`` on a policy's ``sources.noise`` list, iterated through
  a torch ``DataLoader`` at ``num_workers`` 0 and 4 — the harness the training
  loop uses, so the number includes the pipeline's own overhead and the
  worker-to-parent transfer of the rendered audio, not just the renderer;
* 64 windows of 2 s per configuration, at 1 and at 8 microphones;
* the 64-window rate END TO END, each worker's pipeline build included
  (reading the fits, pulling ``rps-traj-fits``, the first render). At
  ``render_reuse: 48`` only one or two windows of the 64 are actually rendered,
  so this rate is build-dominated — which is why there is also a STEADY
  section: ``N_STEADY`` windows, eight full reuse cycles per source, long
  enough that the build amortises and the measured cost is the one a training
  run pays;
* the PER-RENDER wall, timed on the pool object with reuse bypassed (a stream
  at ``render_reuse: 48`` renders one window in 48, so no stream rate can be
  inverted into a render cost);
* peak resident set of the worst DataLoader worker. Every configuration runs in
  its OWN subprocess, because ``RUSAGE_CHILDREN.ru_maxrss`` is a high-water
  mark over the whole process lifetime and would otherwise carry the previous
  configuration's peak.

Writes ``bench.json`` beside itself; ``findings.md`` reads it.
"""

from __future__ import annotations

import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, IterableDataset

from data_processing.online_mixing import build_noise_stream

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
SR = 16000
WINDOW_S = 2.0
N_SAMPLES = 64
#: Windows of the STEADY section: eight full reuse cycles for each of the two
#: weighted sources (0.4 x 384 / 48 = 3.2 renders per source... at 48-window
#: reuse the whole run is 6.4 renders, enough that the build is a few percent).
N_STEADY = 384

V2_POLICY = ROOT / "conf/online_mix/_noise_v2_smoke.yaml"
LEGACY_POLICY = ROOT / "conf/online_mix/rig_easy_5050.yaml"

#: rig_easy_5050's preset bank is an 8-MIC bank (``fixed_mic_gain_db`` is
#: (8, 4)) and ``StochasticNoisePool`` refuses a bank whose mic count differs
#: from the policy's, so the 1-mic column exists for ``noise_v2`` only. The
#: 8-mic column is the one the comparison rests on: the arms are 8-mic.
MIC_COUNTS = {"noise_v2": (1, 8), "stochastic": (8,)}


def _specs(source: str) -> list[dict[str, Any]]:
    path = V2_POLICY if source == "noise_v2" else LEGACY_POLICY
    return yaml.safe_load(path.read_text())["sources"]["noise"]


def _with_mics(specs: list[dict[str, Any]], n_mics: int) -> list[dict[str, Any]]:
    out = []
    for spec in specs:
        spec = dict(spec)
        if spec.get("kind") == "silence":
            spec["n_channels"] = n_mics
        elif "n_mics" in spec or spec.get("kind") in ("noise_v2", "stochastic"):
            spec["n_mics"] = n_mics
        out.append(spec)
    return out


class NoiseStream(IterableDataset):
    """``n`` rendered noise windows, split evenly over the DataLoader workers.

    Each item carries its worker id so the parent can see how the windows
    interleave; the audio is transferred exactly as a training loader would.
    """

    def __init__(self, specs: list[dict[str, Any]], *, n_mics: int, seed: int, n: int):
        self.specs = _with_mics(specs, n_mics)
        self.seed = int(seed)
        self.n = int(n)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, nw = (0, 1) if info is None else (int(info.id), int(info.num_workers))
        t0 = time.perf_counter()
        stream, _ = build_noise_stream(
            self.specs, sample_rate=SR, window_s=WINDOW_S, seed=self.seed + 7919 * wid
        )
        it = iter(stream)
        share = self.n // nw + (1 if wid < self.n % nw else 0)
        for index in range(share):
            frame = next(it)
            audio = torch.from_numpy(np.ascontiguousarray(np.asarray(frame["audio"].data)))
            yield audio, wid, index, time.perf_counter() - t0


def stream_case(source: str, n_mics: int, num_workers: int, seed: int, n: int) -> dict[str, Any]:
    dataset = NoiseStream(_specs(source), n_mics=n_mics, seed=seed, n=n)
    loader = DataLoader(dataset, batch_size=None, num_workers=num_workers)
    t0 = time.perf_counter()
    count = 0
    for audio, _wid, _index, _elapsed in loader:
        assert audio.shape[0] == n_mics
        count += 1
    wall = time.perf_counter() - t0
    return dict(
        source=source,
        n_mics=n_mics,
        num_workers=num_workers,
        samples=count,
        wall_s=round(wall, 3),
        samples_per_s=round(count / wall, 3),
        audio_s_per_s=round(count * WINDOW_S / wall, 2),
        peak_rss_worker_mb=round(
            (
                resource.getrusage(resource.RUSAGE_SELF)
                if num_workers == 0
                else resource.getrusage(resource.RUSAGE_CHILDREN)
            ).ru_maxrss
            / 1024.0,
            1,
        ),
        peak_rss_parent_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1),
    )


def render_case(
    source: str, n_mics: int, n_renders: int = 8, window_s: float = WINDOW_S
) -> dict[str, Any]:
    """Wall per ONE rendered window, on the pool object, reuse bypassed."""
    from data_processing.online_mixing import _build_engine

    spec = _with_mics(_specs(source), n_mics)[0]
    # `render` is each pool's own per-window entry point, below the
    # `NoiseEngine` protocol's `sample_timeframe`, so the reuse pool and the
    # Frame construction stay out of the timing.
    engine: Any = _build_engine(spec, window_s=window_s, sample_rate=SR)
    rng = np.random.default_rng(0)
    engine.render(rng, window_s)  # warm the cached filter design / lazy imports
    walls = []
    for _ in range(n_renders):
        t0 = time.perf_counter()
        engine.render(rng, window_s)
        walls.append(time.perf_counter() - t0)
    return dict(
        source=source,
        kind=spec["kind"],
        n_mics=n_mics,
        window_s=window_s,
        n_renders=n_renders,
        render_ms_median=round(1e3 * float(np.median(walls)), 1),
        render_ms_min=round(1e3 * float(np.min(walls)), 1),
        render_ms_max=round(1e3 * float(np.max(walls)), 1),
        cpu_s_per_audio_s=round(float(np.median(walls)) / float(window_s), 4),
    )


def _run_isolated(case: dict[str, Any]) -> dict[str, Any]:
    """One case in its own process, so its RSS high-water mark is its own."""
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--case", json.dumps(case)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    if "--case" in sys.argv:
        case = json.loads(sys.argv[sys.argv.index("--case") + 1])
        kind = case.pop("case")
        print(json.dumps(render_case(**case) if kind == "render" else stream_case(**case)))
        return

    v2, legacy = _specs("noise_v2"), _specs("stochastic")
    out: dict[str, Any] = dict(
        meta=dict(
            window_s=WINDOW_S,
            n_samples=N_SAMPLES,
            n_steady=N_STEADY,
            sample_rate=SR,
            v2_policy=str(V2_POLICY.relative_to(ROOT)),
            legacy_policy=str(LEGACY_POLICY.relative_to(ROOT)),
            render_reuse=dict(
                noise_v2=int(v2[0].get("render_reuse", 1)),
                stochastic=int(legacy[0].get("render_reuse", 1)),
            ),
            one_mic_note=(
                "rig_easy_5050's preset bank is 8-mic (fixed_mic_gain_db (8, 4)) and the legacy "
                "pool refuses a bank whose mic count differs from the policy's, so only noise_v2 "
                "carries a 1-mic row"
            ),
            torch_threads=torch.get_num_threads(),
        ),
        render_wall=[],
        stream=[],
        steady=[],
    )
    for source, mics in MIC_COUNTS.items():
        for n_mics in mics:
            row = _run_isolated(dict(case="render", source=source, n_mics=n_mics))
            out["render_wall"].append(row)
            print("render", row, flush=True)
    # the arms run 4 s windows; this is the linearity check the A100 estimate
    # in findings.md extrapolates through.
    for source in MIC_COUNTS:
        row = _run_isolated(dict(case="render", source=source, n_mics=8, n_renders=4, window_s=4.0))
        out["render_wall"].append(row)
        print("render", row, flush=True)
    for source, mics in MIC_COUNTS.items():
        for n_mics in mics:
            for workers in (0, 4):
                row = _run_isolated(
                    dict(
                        case="stream",
                        source=source,
                        n_mics=n_mics,
                        num_workers=workers,
                        seed=20260921,
                        n=N_SAMPLES,
                    )
                )
                out["stream"].append(row)
                print("stream", row, flush=True)
    for source in MIC_COUNTS:
        for workers in (0, 4):
            row = _run_isolated(
                dict(
                    case="stream",
                    source=source,
                    n_mics=8,
                    num_workers=workers,
                    seed=20260921,
                    n=N_STEADY,
                )
            )
            out["steady"].append(row)
            print("steady", row, flush=True)
    (OUT / "bench.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", OUT / "bench.json")


if __name__ == "__main__":
    main()
