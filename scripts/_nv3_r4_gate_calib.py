"""Calibration record of the per-rotor identifiability gate (noise-model-v3 round 4).

Writes ``results/noise_v3r4/identifiability/calibration.json``:

* ``equivalence``: :func:`identifiability.decompose`'s mic-mean spectrum against
  :func:`render.expected_periodogram` / ``_regimes`` (array response removed:
  it averages out of the mic mean) on a ramp through the standby/cruise blend;
* ``anchors``: the round-4 anchors (DREGON; Michael's standby + cruise pair), as
  fitted and with the speed laws pinned at the round-4 bank's (6, 6), on a frame
  pool of the round-4 hard stream: per rotor the p10 / p50 over eligible frames of
  the 3rd-best line contrast (the largest ``T`` at which the frame passes), and
  the default gate's verdict;
* ``prior``: acceptance of prior draws (``rig_prior.draw_rig``) of the DREGON and
  Michael's cruise fits at T = 2 and 1 dB, and the seconds per draw.

    PYTHONPATH=src python scripts/_nv3_r4_gate_calib.py [--draws 24] [--frames 64]
"""

from __future__ import annotations

import argparse
import copy
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
    bare = {k: copy.deepcopy(v) for k, v in mich.items()}
    for v in bare.values():
        v["params"].pop("array_response", None)
    t = np.arange(32000) / 16000.0
    rps = np.array([30.0, 42.0, 55.0, 70.0])[:, None] + 8.0 * t[None, :]
    starts = np.arange(1 + (rps.shape[1] - 2048) // 512) * 512
    ref = RD.expected_periodogram_regimes(bare, rps, n_mics=8).mean(axis=0)
    dec = ID.decompose(mich, rps, starts)
    return dict(
        carrier="rotors 30/42/55/70 rev/s + 8 rev/s^2, 2 s",
        n_frames=int(starts.size),
        max_abs_db=float(np.max(np.abs(10.0 * np.log10(dec.power / ref)))),
    )


def third_best(fit, pool: ID.FramePool, cfg: ID.GateConfig) -> list[list[float] | None]:
    third: list[list[float]] = [[] for _ in range(pool.n_rotors)]
    for w, st in zip(pool.windows, pool.starts, strict=True):
        if st.size == 0:
            continue
        e = ID.pool_coverage(ID.FramePool((w,), (st,), 0), cfg).masks[0]
        dec = ID.decompose(fit, w, st)
        n_r, n, kk, _ = dec.lines.shape
        hz = dec.centre_rps[:, :, None] * np.arange(1, kk + 1)
        b = np.clip(np.rint(hz * cfg.n_fft / cfg.sr).astype(int), 0, dec.power.shape[1] - 1)
        ri, ni = np.meshgrid(np.arange(n_r), np.arange(n), indexing="ij")
        line = dec.lines[ri[..., None], ni[..., None], np.arange(kk), b]
        s = dec.power[ni[..., None], b]
        c = 10.0 * np.log10(s / np.maximum(s - line, 1e-300))
        c[(hz < cfg.f_min_hz) | (hz > cfg.f_max_hz)] = -np.inf
        top3 = np.sort(c, axis=2)[:, :, -3]
        for r in range(n_r):
            third[r] += top3[r][e[r]].tolist()
    return [np.round(np.percentile(t, [10, 50]), 2).tolist() if t else None for t in third]


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--draws", type=int, default=24)
    ap.add_argument("--frames", type=int, default=64)
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
        equivalence=equivalence(mich),
        anchors={},
        prior={},
    )
    pinned = {
        "dregon (6,6)": RS.pin_speed_laws(dregon, pin=pin),
        "michaels (6,6)": {k: RS.pin_speed_laws(v, pin=pin) for k, v in mich.items()},
    }
    for name, fit in {"dregon": dregon, "michaels": mich, **pinned}.items():
        res = ID.gate(fit, pool, cfg)
        out["anchors"][name] = dict(
            third_best_contrast_db_p10_p50=third_best(fit, pool, cfg),
            passed=res.passed,
            reason=res.reason,
        )
        print(name, out["anchors"][name])
    for base in ("dregon_room2_floor", "michaels_fly125_cruise"):
        for t in (2.0, 1.0):
            t0 = time.perf_counter()
            acc, rej = RP.sample_guarded(
                load(base),
                pool,
                n=args.draws,
                seed=0,
                config=ID.GateConfig(threshold_db=t),
                max_attempts=args.draws,
            )
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
