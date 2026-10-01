"""Static rig profiles: per-rig speed tables, harmonic profiles and the
point-source test, from the tracking job's unit JSONs (``results/static_rig/
tracks/raw``), the round-1 full-record reads (``results/static_rig/speeds/raw``)
and the rig geometry (``results/static_rig/geometry``).

    python scripts/static_rig_report.py            # -> results/static_rig/report/

Light (pure JSON arithmetic); runs on the laptop.
"""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/static_rig"
C_SOUND = 343.0
#: A line enters the profile / geometry fits when its SNR is at least this on
#: every microphone used.
SNR_MIN_DB = 6.0
COH_MIN = 0.8
MAX_ORDER = 40

GEOM_RIG = {
    "dregon": "dregon",
    "spcup_ku_leuven": "ku_leuven",
    "spcup_maverick": "maverick",
    "avq": "avq",
    "daset_drone1": "daset_f450",
    "daset_drone2": "daset_f330",
}


def _load(d: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]


def observations() -> list[dict[str, Any]]:
    """One record per (recording or DroneAudioSet member) with its rotors."""
    out = []
    for r in _load(RES / "tracks/raw"):
        if "members" in r:
            for m, v in r["members"].items():
                out.append(
                    {
                        **{k: r[k] for k in ("dataset", "rig", "throttle", "mic_dist", "take")},
                        "key": v["key"],
                        "member": m,
                        "group": r["key"],
                        **v,
                    }
                )
        else:
            out.append(r)
    return out


def geometry(obs: dict[str, Any]) -> dict[str, Any] | None:
    name = GEOM_RIG.get(obs["rig"])
    if name is None:
        return None
    g = json.loads((RES / "geometry" / f"{name}.json").read_text())
    if obs["dataset"] == "DroneAudioSet":
        tag = {"M_up": "up", "M_down": "down", "M_center": "centre"}[obs["member"]]
        cfg = g["configs"][f"{tag}_{obs['mic_dist']}"]
    else:
        cfg = g["configs"][g["recordings"][obs["key"]]]
    if cfg.get("mic_pos") is None or cfg.get("rotor_pos") is None:
        return None
    return cfg


def _lines(rot: dict[str, Any], n_ch: int):
    """(order, power_db (C,), snr_db (C,), phase (C,), coherence (C,)) per order."""
    for k, p, s, ph in zip(
        rot["orders"], rot["power_db"], rot["snr_db"], rot["phase_rel"], strict=True
    ):
        if k > MAX_ORDER:
            continue
        p = np.array([np.nan if v is None else v for v in p], dtype=float)
        yield k, p, np.asarray(s, dtype=float), np.asarray(ph["phase"]), np.asarray(ph["coherence"])


def valid(rot: dict[str, Any], o: dict[str, Any]) -> bool:
    """Per-track evidence: the recording's VK fit explains > 20 % of the signal
    (final residual ratio < 0.8), and a distinct track with >= 5 orders at median-mic SNR
    >= SNR_MIN_DB, speed std < 3 rev/s, and < 20 % of its frames within
    0.5 rev/s of the search band's edges (a track parked on an edge is junk)."""
    if not rot["distinct"] or rot["std"] >= 3.0 or o["residual_ratios"][-1] >= 0.8:
        return False
    good = sum(1 for s in rot["snr_db"] if np.nanmedian(np.asarray(s, dtype=float)) >= SNR_MIN_DB)
    lo, hi = o.get("rate_range", (-np.inf, np.inf))
    t = np.asarray(rot["track"])
    edge = float(np.mean((t - lo < 0.5) | (hi - t < 0.5)))
    return good >= 5 and edge < 0.2


def speed_table(obs_all: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for o in obs_all:
        rows.append(
            {
                "rig": o["rig"],
                "key": o.get("group", o["key"]),
                "member": o.get("member"),
                "n_distinct": o["n_distinct"],
                "residual": o["residual_ratios"][-1],
                "means": [round(r["mean"], 3) for r in o["rotors"]],
                "valid": [valid(r, o) for r in o["rotors"]],
                "stds": [round(r["std"], 3) for r in o["rotors"]],
                "ranges": [[round(v, 2) for v in r["range"]] for r in o["rotors"]],
            }
        )
    return {"rows": rows}


def tracker_vs_fullrecord(obs_all: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = {(r["dataset"], r["key"]): r for r in _load(RES / "speeds/raw")}
    out = []
    for o in obs_all:
        f = full.get((o["dataset"], o["key"]))
        if f is None or o["rig"] not in ("dregon", "spcup_agh") or o.get("n_rotors") != 1:
            continue
        fs = f["modes"]["rotors"][0]["speed"] if f["modes"]["rotors"] else None
        out.append(
            {
                "key": o["key"],
                "full": fs,
                "tracked": o["rotors"][0]["mean"],
                "tracked_std": o["rotors"][0]["std"],
            }
        )
    return out


def profiles(obs_all: list[dict[str, Any]]) -> dict[str, Any]:
    """Per rig and order: median over (recording, distinct rotor, mic) of the
    line power relative to that rotor's strongest order (dB), with IQR and
    count; only lines at SNR >= SNR_MIN_DB on the median mic."""
    acc: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for o in obs_all:
        rig = o["rig"] + (f"/{o['member']}" if o.get("member") else "")
        for rot in o["rotors"]:
            if not valid(rot, o):
                continue
            per_k = {}
            for k, p, s, _, _ in _lines(rot, 0):
                if np.nanmedian(s) >= SNR_MIN_DB and np.isfinite(np.nanmedian(p)):
                    per_k[k] = float(np.nanmedian(p))
            if len(per_k) < 3:
                continue
            top = max(per_k.values())
            for k, v in per_k.items():
                acc[rig][k].append(v - top)
    out = {}
    for rig, by_k in acc.items():
        out[rig] = {
            int(k): {
                "median_db": float(np.median(v)),
                "q25": float(np.percentile(v, 25)),
                "q75": float(np.percentile(v, 75)),
                "n": len(v),
            }
            for k, v in sorted(by_k.items())
        }
    return out


def _dist(cfg: dict[str, Any]) -> np.ndarray:
    m = np.asarray(cfg["mic_pos"], dtype=float)
    r = np.asarray(cfg["rotor_pos"], dtype=float)
    return np.linalg.norm(m[None, :, :] - r[:, None, :], axis=2)  # (rotor, mic)


def _min_mics(n_mic: int) -> int:
    return max(3, math.ceil(0.6 * n_mic))


def _assign(lines: list, D: np.ndarray, perms: list) -> tuple:
    """Track -> rotor-position permutation best fitting pure 1/r (no gains)."""

    def cost(perm: tuple) -> float:
        tot = 0.0
        for ti, _, p in lines:
            d = p + 20.0 * np.log10(D[perm[ti]])
            ok = np.isfinite(d)
            tot += float(np.sum((d[ok] - d[ok].mean()) ** 2))
        return tot

    return min(perms, key=cost)


def _fit(rows: list, n_mic: int, gain: bool, dist: bool) -> tuple[np.ndarray, float]:
    """Mic gains (C,) and alpha on ``rows`` = [(p (C,) NaN-masked, r (C,))]."""
    y, li, mi, dl = [], [], [], []
    for i, (p, r) in enumerate(rows):
        ok = np.isfinite(p)
        y.append(p[ok])
        li.append(np.full(int(ok.sum()), i))
        mi.append(np.nonzero(ok)[0])
        dl.append(20.0 * np.log10(r[ok]))
    yv, lv, mv, dv = (np.concatenate(a) for a in (y, li, mi, dl))
    cols = [np.eye(len(rows))[lv]]
    if gain:
        cols.append(np.eye(n_mic)[mv][:, 1:])
    if dist:
        cols.append(-dv[:, None])
    b, *_ = np.linalg.lstsq(np.hstack(cols), yv, rcond=None)
    g = np.zeros(n_mic)
    if gain:
        g[1:] = b[len(rows) : len(rows) + n_mic - 1]
    return g, (float(b[-1]) if dist else 0.0)


def _rms(rows: list, g: np.ndarray, alpha: float) -> float:
    """Across-mic RMS residual of ``rows`` under fixed gains and alpha."""
    res: list[float] = []
    for p, r in rows:
        d = p - g + alpha * 20.0 * np.log10(r)
        ok = np.isfinite(d)
        res.extend((d[ok] - d[ok].mean()).tolist())
    return float(np.sqrt(np.mean(np.square(res)))) if res else float("nan")


def _rig_lines(o: dict[str, Any], D: np.ndarray) -> list:
    """(track, order, masked level (C,), freq, phase, coherence) per usable line."""
    n_mic = D.shape[1]
    rots = [r for r in o["rotors"] if valid(r, o)]
    out = []
    for ti, rot in enumerate(rots):
        for k, p, s, ph, coh in _lines(rot, n_mic):
            if p.size != n_mic:
                continue
            p = np.where(s >= SNR_MIN_DB, p, np.nan)
            if np.isfinite(p).sum() >= _min_mics(n_mic):
                out.append((ti, k, p, k * rot["mean"], ph, coh))
    return out


def _perm_for(o: dict[str, Any], lines: list, D: np.ndarray) -> tuple:
    if o.get("n_rotors") == 1 and o["rig"] == "dregon":
        return (int(o["motor"]) - 1,)
    n_tr = 1 + max(ln[0] for ln in lines)
    perms = list(itertools.permutations(range(D.shape[0]), n_tr))
    return _assign([(ln[0], ln[1], ln[2]) for ln in lines], D, perms)


def _collect(lines, D, perm, rows, perr, pnull, rng) -> None:
    n_mic = D.shape[1]
    for ti, k, p, f, ph, coh in lines:
        r = D[perm[ti]]
        rows.append((k, p, r))
        ref = int(np.nanargmax(p))
        good = (coh >= COH_MIN) & (np.arange(n_mic) != ref)
        if not good.any():
            continue
        pred = -2 * math.pi * f * (r - r[ref]) / C_SOUND
        perr.extend(np.abs(np.angle(np.exp(1j * (ph - pred))))[good].tolist())
        q = r[rng.permutation(n_mic)]
        predn = -2 * math.pi * f * (q - q[ref]) / C_SOUND
        pnull.extend(np.abs(np.angle(np.exp(1j * (ph - predn))))[good].tolist())


def _score(rows: list, perr: list[float], pnull: list[float]) -> dict[str, Any]:
    """Held-out across-mic RMS (fit on even orders, score odd, and back)."""
    n_mic = rows[0][1].size
    folds = [[(p, r) for k, p, r in rows if k % 2 == m] for m in (0, 1)]
    out: dict[str, Any] = {"n_lines": len(rows), "n_mics": n_mic}
    out["one_over_r_spread_db"] = float(np.median([np.ptp(20 * np.log10(r)) for _, _, r in rows]))
    for name, gain, dist in (("none", 0, 0), ("gain", 1, 0), ("dist", 0, 1), ("both", 1, 1)):
        held, alphas = [], []
        for a, b in ((0, 1), (1, 0)):
            if not folds[a] or not folds[b]:
                continue
            g, al = _fit(folds[a], n_mic, bool(gain), bool(dist))
            held.append(_rms(folds[b], g, al))
            alphas.append(al)
        out[f"rms_{name}"] = float(np.mean(held)) if held else None
        if dist:
            out[f"alpha_{name}"] = [round(a, 3) for a in alphas]
    out["phase_median_abs_err_rad"] = float(np.median(perr)) if perr else None
    out["phase_null_median_abs_err_rad"] = float(np.median(pnull)) if pnull else None
    out["phase_n"] = len(perr)
    return out


def point_source(obs_all: list[dict[str, Any]]) -> dict[str, Any]:
    """Level = line const + mic gain - alpha*20*log10 r on cells at SNR >=
    SNR_MIN_DB (>= 60 % of mics, min 3). Tracks go to rotor positions by the
    best pure-1/r permutation per recording. Gains/alpha fitted on even
    orders, scored on odd ones and vice versa. Phase error vs
    -2 pi f (r_c - r_ref)/c at coherence >= COH_MIN vs a mic-permuted null."""
    rows: dict[str, list] = defaultdict(list)
    perr: dict[str, list[float]] = defaultdict(list)
    pnull: dict[str, list[float]] = defaultdict(list)
    rng = np.random.default_rng(0)
    for o in obs_all:
        cfg = geometry(o)
        D = _dist(cfg) if cfg is not None else None
        lines = _rig_lines(o, D) if D is not None and D.shape[1] >= 3 else []
        if not lines:
            continue
        assert D is not None
        rig = GEOM_RIG[o["rig"]] + (f"/{o['member']}" if o.get("member") else "")
        perm = _perm_for(o, lines, D)
        _collect(lines, D, perm, rows[rig], perr[rig], pnull[rig], rng)
    return {rig: _score(rr, perr[rig], pnull[rig]) for rig, rr in rows.items()}


def profile_tables(obs_all: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Per rig: every distinct rotor's speed and its full (order x mic) line
    power and SNR tables, as measured (no collapsing)."""
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for o in obs_all:
        for rot in o["rotors"]:
            if not valid(rot, o):
                continue
            out[o["rig"]].append(
                {
                    "recording": o.get("group", o["key"]),
                    "member": o.get("member"),
                    "throttle": o.get("throttle"),
                    "speed_mean": rot["mean"],
                    "speed_std": rot["std"],
                    "orders": rot["orders"],
                    "power_db": rot["power_db"],
                    "snr_db": rot["snr_db"],
                }
            )
    return dict(out)


def main() -> int:
    obs_all = observations()
    out_dir = RES / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "speeds": speed_table(obs_all),
        "tracker_vs_fullrecord": tracker_vs_fullrecord(obs_all),
        "profiles": profiles(obs_all),
        "point_source": point_source(obs_all),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    (out_dir / "profiles_full.json").write_text(json.dumps(profile_tables(obs_all)))
    print(json.dumps({k: report[k] for k in ("tracker_vs_fullrecord", "point_source")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
