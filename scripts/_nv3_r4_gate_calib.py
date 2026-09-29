"""Calibration record of the per-rotor identifiability gate (noise-model-v3 round 4).

Writes ``results/noise_v3r4/identifiability/calibration.json``:

* ``equivalence``: :func:`identifiability.decompose`'s mic-mean spectrum (the
  EXACT reference) against the mic mean of :func:`render.expected_periodogram_regimes`,
  array response included, on a ramp through the standby/cruise blend;
* ``anchors``: the round-4 anchors (DREGON; Michael's standby + cruise pair), as
  fitted and with the speed laws pinned at the round-4 bank's (6, 6), on a frame
  pool of the round-4 hard stream: per rotor the p10 / p50 over eligible frames of
  the 1st/2nd/3rd best EXACT line contrast, and the (fast) gate's verdict;
* ``accuracy``: the fast contrasts (:func:`identifiability.line_contrasts`)
  against the exact ones, on the pool's frames and on the same frames held at
  their centre speed, for the default background (+-10 neighbours) and the two
  nearest per other rotor;
* ``prior``: acceptance of prior draws (``rig_prior.draw_rig``) of the DREGON and
  Michael's cruise fits at T = 2 and 1 dB, and the seconds per draw.

    PYTHONPATH=src python scripts/_nv3_r4_gate_calib.py [--draws 24] [--frames 64]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from data_processing.noise_v2_pool import NoiseV2Pool
from experiments.noise_model import identifiability as ID
from experiments.noise_model import render as RD
from experiments.noise_model import rig_prior as RP
from experiments.noise_model import rig_sampler as RS

FITS = Path("results/noise_v3/fits_r4")
POLICY = Path("conf/online_mix/noise_v3r4_hard_5050.yaml")
BANK = Path("results/noise_v3r4/rig_sampler/banks/noise_v3r4_hard_n2048.json")
OUT = Path("results/noise_v3r4/identifiability/calibration.json")


def load(name: str) -> dict:
    return json.loads((FITS / f"{name}__flight_v3.json").read_text())


def stream_windows(n: int, seed: int) -> list[np.ndarray]:
    cfg = yaml.safe_load(POLICY.read_text())
    src = next(s for s in cfg["sources"]["noise"] if s.get("kind") == "noise_v2")
    src["preset_bank"] = str(BANK)
    pool = NoiseV2Pool.from_config(src, duration_s=2.0, sample_rate=16000)
    rng = np.random.default_rng(seed)
    return [
        pool.sample_rps(rng, 2.0, pool.entries[int(rng.integers(len(pool.entries)))])
        for _ in range(n)
    ]


def equivalence(mich: dict) -> dict:
    t = np.arange(32000) / 16000.0
    rps = np.array([30.0, 42.0, 55.0, 70.0])[:, None] + 8.0 * t[None, :]
    starts = np.arange(1 + (rps.shape[1] - 2048) // 512) * 512
    ref = RD.expected_periodogram_regimes(mich, rps, n_mics=8).mean(axis=0)
    dec = ID.decompose(mich, rps, starts)
    return dict(
        carrier="rotors 30/42/55/70 rev/s + 8 rev/s^2, 2 s; array response on",
        n_frames=int(starts.size),
        max_abs_db=float(np.max(np.abs(10.0 * np.log10(dec.power / ref)))),
    )


def exact_contrasts(fit, pool: ID.FramePool) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per window: ``(contrast_db, centre_hz)`` ``(R, n, K)``, the EXACT
    ``10 log10(S / (S - L))`` of every line at its centre bin."""
    out = []
    for w, st in zip(pool.windows, pool.starts, strict=True):
        dec = ID.decompose(fit, w, st)
        n_r, n, kk, _ = dec.lines.shape
        hz = dec.centre_rps[:, :, None] * np.arange(1, kk + 1)
        b = np.clip(np.rint(hz * pool.n_fft / pool.sr).astype(int), 0, dec.power.shape[1] - 1)
        ri, ni = np.meshgrid(np.arange(n_r), np.arange(n), indexing="ij")
        line = dec.lines[ri[..., None], ni[..., None], np.arange(kk), b]
        s = dec.power[ni[..., None], b]
        out.append((10.0 * np.log10(s / np.maximum(s - line, 1e-300)), hz))
    return out


def accuracy(fit, pool: ID.FramePool, exact: list, cfg: ID.GateConfig) -> dict:
    """Fast minus exact contrast (dB) over in-band lines above 0.5 dB in either,
    and how many 2 dB decisions differ."""
    diffs, flips, lines = [], 0, 0
    for (fast, hz), (ex, _) in zip(ID.line_contrasts(fit, pool, cfg), exact, strict=True):
        kk = min(fast.shape[2], ex.shape[2])
        band = (hz[:, :, :kk] >= cfg.f_min_hz) & (hz[:, :, :kk] <= cfg.f_max_hz)
        f, e = fast[:, :, :kk][band], ex[:, :, :kk][band]
        diffs += (f - e)[(f > 0.5) | (e > 0.5)].tolist()
        flips += int(((f >= cfg.threshold_db) != (e >= cfg.threshold_db)).sum())
        lines += int(band.sum())
    d = np.asarray(diffs)
    return dict(
        median_db=round(float(np.median(d)), 3),
        p95_db=round(float(np.percentile(d, 95)), 2),
        max_db=round(float(d.max()), 2),
        decision_flips=flips,
        lines=lines,
    )


def best_contrasts(fit, pool: ID.FramePool, cfg: ID.GateConfig) -> dict[str, list]:
    """Per rank 1-3 and rotor, over the rotor's eligible frames: the p10 / p50
    of the rank-th best line contrast (``10 log10(S / (S - L))`` at the line
    centre, in-band orders), and the order ``k`` that holds that rank most often."""
    vals: list[list[list[float]]] = [[[] for _ in range(pool.n_rotors)] for _ in range(3)]
    ords: list[list[list[int]]] = [[[] for _ in range(pool.n_rotors)] for _ in range(3)]
    for (c, hz), (w, st) in zip(
        exact_contrasts(fit, pool), zip(pool.windows, pool.starts, strict=True), strict=True
    ):
        if st.size == 0:
            continue
        e = ID.pool_coverage(ID.FramePool((w,), (st,), 0), cfg).masks[0]
        n_r = c.shape[0]
        c = c.copy()
        c[(hz < cfg.f_min_hz) | (hz > cfg.f_max_hz)] = -np.inf
        order = np.argsort(c, axis=2)[:, :, ::-1][:, :, :3]  # (R, n, 3) best first
        top = np.take_along_axis(c, order, axis=2)
        for rank in range(3):
            for r in range(n_r):
                vals[rank][r] += top[r, :, rank][e[r]].tolist()
                ords[rank][r] += (order[r, :, rank][e[r]] + 1).tolist()
    out: dict[str, list] = {}
    for rank, label in enumerate(("1st", "2nd", "3rd")):
        out[f"{label}_contrast_db_p10_p50"] = [
            np.round(np.percentile(v, [10, 50]), 2).tolist() if v else None for v in vals[rank]
        ]
        out[f"{label}_modal_order"] = [
            int(np.bincount(o).argmax()) if o else None for o in ords[rank]
        ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument(
        "--anchors-only",
        action="store_true",
        help="recompute only the anchors block and merge it into the existing record",
    )
    args = ap.parse_args()
    torch.set_num_threads(4)
    pin = RS.ROUND4_SPEED_LAW_PIN
    dregon = load("dregon_room2_floor")
    mich = {"standby": load("michaels_fly125_standby"), "cruise": load("michaels_fly125_cruise")}
    pool = ID.frame_pool(stream_windows(32, 0), n_frames=args.frames, seed=1)
    cfg = ID.GateConfig()
    out: dict = dict(
        pool=dict(
            policy=str(POLICY),
            windows=32,
            window_s=2.0,
            n_frames=pool.n_frames,
            coverage=ID.pool_coverage(pool, cfg).eligible.tolist(),
        ),
        config=cfg.__dict__,
        equivalence=None if args.anchors_only else equivalence(mich),
        anchors={},
        prior={},
        accuracy={},
    )
    pinned = {
        "dregon (6,6)": RS.pin_speed_laws(dregon, pin=pin),
        "michaels (6,6)": {k: RS.pin_speed_laws(v, pin=pin) for k, v in mich.items()},
    }
    for name, fit in {"dregon": dregon, "michaels": mich, **pinned}.items():
        res = ID.gate(fit, pool, cfg)
        out["anchors"][name] = dict(
            **best_contrasts(fit, pool, cfg),
            passed=res.passed,
            reason=res.reason,
        )
        print(name, json.dumps(out["anchors"][name]))
    if args.anchors_only:
        rec = json.loads(OUT.read_text())
        rec["anchors"] = out["anchors"]
        OUT.write_text(json.dumps(rec, indent=1))
        print(f"merged anchors into {OUT}")
        return
    # the same windows, each held at its speed at sample 2048: frames without chirp
    const = ID.frame_pool(
        [np.repeat(w[:, [2048]], w.shape[1], axis=1) for w in pool.windows],
        n_frames=args.frames,
        seed=1,
    )
    checks = {
        **pinned,
        "prior dregon [7,1]": RP.draw_rig(dregon, np.random.default_rng([7, 1])),
        "prior michaels-cruise [7,0]": RP.draw_rig(mich["cruise"], np.random.default_rng([7, 0])),
    }
    for name, fit in checks.items():
        for label, frames in (("constant speed", const), ("stream", pool)):
            out["accuracy"][f"{name}, {label}"] = accuracy(
                fit, frames, exact_contrasts(fit, frames), cfg
            )
            print(name, label, out["accuracy"][f"{name}, {label}"])
    for base in ("dregon_room2_floor", "michaels_fly125_cruise"):
        for t in (2.0, 1.0):
            t0 = time.perf_counter()
            accept = RP.gate_acceptance(pool, ID.GateConfig(threshold_db=t))
            try:  # all draws accepted is the only way not to exhaust n = max_attempts
                acc, rej = RP.sample_rigs(
                    load(base), n=args.draws, seed=0, accept=accept, max_attempts=args.draws
                )
            except RP.SamplingExhausted as short:
                acc, rej = short.accepted, short.rejected
            key = f"{base} T={t:g}"
            out["prior"][key] = dict(
                accepted=len(acc),
                drawn=len(acc) + len(rej),
                s_per_draw=round((time.perf_counter() - t0) / args.draws, 2),
            )
            print(key, out["prior"][key])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
