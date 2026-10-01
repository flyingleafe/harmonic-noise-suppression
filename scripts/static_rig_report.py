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
import re
import sys
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


#: Tracking outputs, later directories replacing earlier units of the same
#: name (the AGH-array rerun in the 3-blade shaft band, AVQ S2_seq1 at full length).
TRACK_DIRS = (
    "tracks/raw",
    "tracks_agh/raw",
    "tracks_s2/raw",
    "tracks_das/raw",
    "tracks_das2/raw",
)


def _uid(dataset: str, key: str) -> str:
    """``scripts/static_rig.py`` unit id of one recording file."""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", f"{dataset}__{key}")


def observations() -> list[dict[str, Any]]:
    """One record per (recording or DroneAudioSet member) with its rotors."""
    units: dict[str, dict[str, Any]] = {}
    for d in TRACK_DIRS:
        for p in sorted((RES / d).glob("*.json")):
            units[p.name] = json.loads(p.read_text())
    out = []
    for r in units.values():
        if "members" in r:
            for m, v in r["members"].items():
                out.append(
                    {
                        **{k: r[k] for k in ("dataset", "rig", "throttle", "mic_dist", "take")},
                        "n_rotors": r["n_rotors"],
                        "rate_range": r["rate_range"],
                        **v,
                        "uid": _uid(r["dataset"], v["key"]),
                        "member": m,
                        "group": r["key"],
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
    """(order, power_db (C,), snr_db (C,), phase (C,), coherence (C,), ref) per
    order; phase and coherence are relative to mic ``ref`` (the tracker's
    loudest raw cell)."""
    for k, p, s, ph in zip(
        rot["orders"], rot["power_db"], rot["snr_db"], rot["phase_rel"], strict=True
    ):
        if k > MAX_ORDER:
            continue
        p = np.array([np.nan if v is None else v for v in p], dtype=float)
        s = np.asarray(s, dtype=float)
        yield k, p, s, np.asarray(ph["phase"]), np.asarray(ph["coherence"]), int(ph["ref"])


def _candidate(rot: dict[str, Any], o: dict[str, Any]) -> bool:
    """A distinct track with speed std < 3 rev/s and < 20 % of its frames
    within 0.5 rev/s of the search band's edges (a track parked on an edge is
    junk)."""
    if not rot["distinct"] or rot["std"] >= 3.0:
        return False
    lo, hi = o.get("rate_range", (-np.inf, np.inf))
    t = np.asarray(rot["track"])
    return float(np.mean((t - lo < 0.5) | (hi - t < 0.5))) < 0.2


def valid(rot: dict[str, Any], o: dict[str, Any]) -> bool:
    """Per-track evidence: a :func:`_candidate` with >= 5 orders at median-mic
    SNR >= SNR_MIN_DB, in a recording whose VK fit explains > 20 % of the
    signal (final residual ratio < 0.8)."""
    if not _candidate(rot, o) or o["residual_ratios"][-1] >= 0.8:
        return False
    good = sum(1 for s in rot["snr_db"] if np.nanmedian(np.asarray(s, dtype=float)) >= SNR_MIN_DB)
    return good >= 5


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
            for k, p, s, *_ in _lines(rot, 0):
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
    """(track, order, masked level (C,), freq, phase, coherence, ref) per usable line."""
    n_mic = D.shape[1]
    rots = [r for r in o["rotors"] if valid(r, o)]
    out = []
    for ti, rot in enumerate(rots):
        for k, p, s, ph, coh, ref in _lines(rot, n_mic):
            if p.size != n_mic:
                continue
            p = np.where(s >= SNR_MIN_DB, p, np.nan)
            if np.isfinite(p).sum() >= _min_mics(n_mic):
                out.append((ti, k, p, k * rot["mean"], ph, coh, ref))
    return out


def _perm_for(o: dict[str, Any], lines: list, D: np.ndarray, n_tr: int) -> tuple:
    if o.get("n_rotors") == 1 and o["rig"] == "dregon":
        return (int(o["motor"]) - 1,)
    perms = list(itertools.permutations(range(D.shape[0]), n_tr))
    return _assign([(ln[0], ln[1], ln[2]) for ln in lines], D, perms)


def _collect(lines, D, perm, rows, perr, pnull, rng) -> None:
    """Level rows of ``lines``; phase errors where the reference mic's cell is
    usable (SNR >= SNR_MIN_DB) and the other mic's coherence >= COH_MIN."""
    n_mic = D.shape[1]
    for ti, k, p, f, ph, coh, ref in lines:
        r = D[perm[ti]]
        rows.append((k, p, r))
        good = (coh >= COH_MIN) & (np.arange(n_mic) != ref)
        if not np.isfinite(p[ref]) or not good.any():
            continue
        pred = -2 * math.pi * f * (r - r[ref]) / C_SOUND
        perr.extend(np.abs(np.angle(np.exp(1j * (ph - pred))))[good].tolist())
        q = r[rng.permutation(n_mic)]
        predn = -2 * math.pi * f * (q - q[ref]) / C_SOUND
        pnull.extend(np.abs(np.angle(np.exp(1j * (ph - predn))))[good].tolist())


def _score(
    fold: dict[int, dict[str, list]], perr: list[float], pnull: list[float]
) -> dict[str, Any]:
    """Held-out across-mic RMS: for each parity ``a``, gains/alpha (and the
    recordings' track -> rotor permutations) come from order parity ``a``, the
    RMS from the other parity's lines."""
    test_all = [r for d in fold.values() for r in d["test"]]
    n_mic = test_all[0][1].size
    out: dict[str, Any] = {"n_lines": len(test_all), "n_mics": n_mic}
    out["one_over_r_spread_db"] = float(
        np.median([np.ptp(20 * np.log10(r)) for _, _, r in test_all])
    )
    for name, gain, dist in (("none", 0, 0), ("gain", 1, 0), ("dist", 0, 1), ("both", 1, 1)):
        held, alphas = [], []
        for d in fold.values():
            if not d["train"] or not d["test"]:
                continue
            g, al = _fit([(p, r) for _, p, r in d["train"]], n_mic, bool(gain), bool(dist))
            held.append(_rms([(p, r) for _, p, r in d["test"]], g, al))
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
    SNR_MIN_DB (>= 60 % of mics, min 3). Cross-fitted by order parity: on
    parity ``a`` each recording's tracks go to rotor positions by the best
    pure-1/r permutation and the gains/alpha are fitted; the other parity's
    lines are scored under both (a track with no training line is dropped).
    Phase error of the scored lines vs -2 pi f (r_c - r_ref)/c at coherence
    >= COH_MIN, against a mic-permuted null."""
    fold: dict[str, dict[int, dict[str, list]]] = defaultdict(
        lambda: {a: {"train": [], "test": []} for a in (0, 1)}
    )
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
        n_tr = 1 + max(ln[0] for ln in lines)
        for a in (0, 1):
            train = [ln for ln in lines if ln[1] % 2 == a]
            if not train:
                continue
            seen = {ln[0] for ln in train}
            test = [ln for ln in lines if ln[1] % 2 != a and ln[0] in seen]
            perm = _perm_for(o, train, D, n_tr)
            _collect(train, D, perm, fold[rig][a]["train"], [], [], rng)
            _collect(test, D, perm, fold[rig][a]["test"], perr[rig], pnull[rig], rng)
    return {
        rig: _score(f, perr[rig], pnull[rig])
        for rig, f in fold.items()
        if any(d["test"] for d in f.values())
    }


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


#: AGH 4-rotor array takes: the round-1 full-record search (2-blade
#: assumption, 15-260 rev/s) does not see their 3-blade combs, and neither the
#: tracker (40-100 rev/s) nor a shaft-rate harmonic sieve finds a dominant comb.
#: The only sourced band: take 4's blade-pass lines in AGH.yaml
#: (209.6/214.3/216.0/217.4 Hz), /3 for the shaft.
AGH_ARRAY_BAND = {"SPCUP19-frames__AGH__ego-noise__mic_array__4": (209.6 / 3, 217.4 / 3)}


def recordings(obs_all: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per recording file: resolved R/R, partial or unresolved, and for the
    last two the shaft band ``[s_lo, s_hi]`` holding every rotor, for the
    combined comb read. The band comes from the candidate tracks (5th-95th
    percentile of their frames) when at least one track is valid, else from
    the round-1 full-record speed modes (±0.5 %), else from the full-record
    comb rate ± ``spread_frac`` (the speed-mode search window); AGH arrays
    only from :data:`AGH_ARRAY_BAND`, DroneAudioSet only from tracks."""
    full = {(r["dataset"], r["key"]): r for r in _load(RES / "speeds/raw")}
    out = {}
    for o in obs_all:
        uid = o.get("uid") or _uid(o["dataset"], o["key"])
        n_r, n_v = int(o["n_rotors"]), sum(valid(r, o) for r in o["rotors"])
        rec: dict[str, Any] = {"rig": o["rig"], "n_rotors": n_r, "n_valid": n_v}
        rec["verdict"] = "resolved" if n_v == n_r else "partial" if n_v else "unresolved"
        if n_v < n_r:
            cand = [np.asarray(r["track"]) for r in o["rotors"] if _candidate(r, o)]
            f = full.get((o["dataset"], o["key"]))
            band: tuple[float, float] | None = None
            src = None
            if "mic_array" in o["key"] and o["rig"] == "spcup_agh":
                if uid in AGH_ARRAY_BAND:
                    band, src = AGH_ARRAY_BAND[uid], "AGH.yaml blade-pass lines / 3"
            elif n_v and cand:
                band = (
                    float(min(np.percentile(t, 5) for t in cand)),
                    float(max(np.percentile(t, 95) for t in cand)),
                )
                src = "tracks"
            elif o["dataset"] == "DroneAudioSet":
                pass  # round-1 DAS reads scatter 15-250 rev/s: no fallback
            elif f is not None and f["modes"]["rotors"]:
                s = [m["speed"] for m in f["modes"]["rotors"]]
                band, src = (0.995 * min(s), 1.005 * max(s)), "full_record_modes"
            elif f is not None:
                c, w = f["comb"]["f0_comb"], f["params"]["spread_frac"]
                band, src = (c * (1 - w), c * (1 + w)), "full_record_comb"
            if band is not None:
                rec["band"] = {"s_lo": float(band[0]), "s_hi": float(band[1]), "source": src}
        out[uid] = rec
    return out


def combined_profiles(recs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per partial/unresolved recording read by ``static_rig.py combined``:
    total comb power per (order, mic) at SNR >= SNR_MIN_DB (else null), and
    the symmetric per-rotor profile, total - 10 log10 R dB."""
    out = {}
    for r in _load(RES / "combined/raw"):
        rec = recs.get(r["uid"])
        if rec is None or rec.get("band") != r["band"]:
            continue  # resolved since, or read in a band no longer used
        p = np.array(r["power_db"], dtype=float)
        s = np.array(r["snr_db"], dtype=float)
        p = np.where(s >= SNR_MIN_DB, p, np.nan)
        out[r["uid"]] = {
            **rec,
            "orders": r["orders"],
            "total_db": [
                [None if not np.isfinite(v) else round(float(v), 2) for v in row] for row in p
            ],
            "per_rotor_db": [
                [
                    None
                    if not np.isfinite(v)
                    else round(float(v - 10 * math.log10(rec["n_rotors"])), 2)
                    for v in row
                ]
                for row in p
            ],
            "snr_db": [[round(float(v), 2) for v in row] for row in s],
            "n_orders_measured": int(
                np.sum(np.sum(np.isfinite(p), axis=1) >= min(p.shape[1], _min_mics(p.shape[1])))
            ),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    obs_all = observations()
    recs = recordings(obs_all)
    if (argv if argv is not None else sys.argv[1:]) == ["bands"]:
        path = RES / "combined/bands.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        bands = {u: r["band"] for u, r in recs.items() if "band" in r}
        path.write_text(json.dumps(bands, indent=1))
        print(f"{len(bands)} bands -> {path}")
        return 0
    out_dir = RES / "report"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "recordings": recs,
        "speeds": speed_table(obs_all),
        "tracker_vs_fullrecord": tracker_vs_fullrecord(obs_all),
        "profiles": profiles(obs_all),
        "point_source": point_source(obs_all),
        "combined": combined_profiles(recs),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    (out_dir / "profiles_full.json").write_text(json.dumps(profile_tables(obs_all)))
    print(json.dumps({k: report[k] for k in ("tracker_vs_fullrecord", "point_source")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
