"""Speed laws on the DREGON bench ladder (4 motors x 5 throttles, 48-89 rev/s).

Per harmonic order: the power exponent alpha_k of the line (``P_k ~ s^alpha_k``)
from the stored OU-shape profiles; the floor: its exponent at fixed frequency
per band (with and without the rotor-off ambient), its exponent at fixed ORDER
(the floor under harmonic k), and which axis - fixed f or f/s (Strouhal) - lets
one scalar exponent explain a motor's five throttles.  Pooled regressions carry
one level offset per motor.  Output: ``results/static_rig/single_rotor/speed_laws.{json,png}``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "decoh", ROOT / "scripts" / "noise_v2_order_decoherence.py"
)
assert _spec is not None and _spec.loader is not None
decoh = importlib.util.module_from_spec(_spec)
sys.modules["decoh"] = decoh
_spec.loader.exec_module(decoh)

FS = 44100.0
SR_DIR = ROOT / "results/static_rig/single_rotor"
PROF = SR_DIR / "profile"
THR = (50, 60, 70, 80, 90)
MOTORS = (1, 2, 3, 4)
BANDS = [
    (100, 200),
    (200, 400),
    (400, 800),
    (800, 1600),
    (1600, 3200),
    (3200, 6400),
    (6400, 12800),
    (12800, 20000),
]
K_MAX = 100
S_REF = 70.0


def pooled_slope(groups: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float, float]:
    """Slope of y on x with one free offset per group: ``(q, se, rms residual)``."""
    X = np.concatenate([x - x.mean() for x, _ in groups])
    Y = np.concatenate([y - y.mean() for _, y in groups])
    q = float((X * Y).sum() / (X * X).sum())
    r = Y - q * X
    se = float(np.sqrt((r**2).sum() / (len(X) - len(groups) - 1) / (X * X).sum()))
    return q, se, float(np.sqrt((r**2).mean()))


def load() -> dict[str, dict]:
    singles = json.load(open(SR_DIR / "dregon_singles.json"))["recordings"]
    audio = dict(decoh.dregon_motor_audio(lambda rid: rid.startswith("motor_Motor")))
    recs: dict[str, dict] = {}
    for rid, a in sorted(audio.items()):
        prof = json.load(open(PROF / f"{rid}.json"))
        mics = [m for m in range(a.shape[0]) if m not in set(singles[rid]["wind_mics"])]
        a0, a1 = prof["span"]
        seg = a[mics, a0:a1]
        f, p = signal.welch(seg - seg.mean(1, keepdims=True), fs=FS, nperseg=8192, axis=-1)
        act0, _ = decoh.active_window(a)
        off = a[mics, : max(int(act0 - 0.5 * FS), int(FS))]
        _, pn = signal.welch(off - off.mean(1, keepdims=True), fs=FS, nperseg=8192, axis=-1)
        recs[rid] = dict(
            s=float(prof["s"]),
            f=f,
            p=p,
            p_off=pn,
            line_db=10 * np.log10(np.maximum(np.asarray(prof["amp2"])[mics], 1e-30)).mean(0),
        )
    return recs


def band_floor_db(f: np.ndarray, p: np.ndarray, lo: float, hi: float, pct: float = 20) -> float:
    sel = (f >= lo) & (f < hi)
    if sel.sum() < 3:
        return float("nan")
    return float(10 * np.log10(np.percentile(p[:, sel], pct, axis=1)).mean())


def groups_for(recs, value) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for m in MOTORS:
        rids = [f"motor_Motor{m}_{t}" for t in THR]
        out.append(
            (
                np.array([10 * np.log10(recs[r]["s"]) for r in rids]),
                np.array([value(recs[r]) for r in rids]),
            )
        )
    return out


def main() -> None:
    recs = load()
    out: dict = {"speeds": {r: v["s"] for r, v in recs.items()}}

    # ---- tonal: alpha_k per order ----
    alpha = [pooled_slope(groups_for(recs, lambda v, i=i: v["line_db"][i])) for i in range(K_MAX)]
    out["alpha_k"] = [
        dict(k=i + 1, alpha=q, se=se, rms_db=rms) for i, (q, se, rms) in enumerate(alpha)
    ]
    per_motor = {}
    for m in MOTORS:
        rids = [f"motor_Motor{m}_{t}" for t in THR]
        xs = np.array([10 * np.log10(recs[r]["s"]) for r in rids])
        L = np.array([recs[r]["line_db"][:K_MAX] for r in rids])
        per_motor[m] = [pooled_slope([(xs, L[:, i])])[0] for i in range(K_MAX)]
    out["alpha_k_per_motor"] = per_motor
    k = np.arange(1, K_MAX + 1)
    a = np.array([q for q, _, _ in alpha])
    even = (k % 2 == 0) & (k <= 40) & (k % 21 != 0)
    odd = (k % 2 == 1) & (k >= 3) & (k <= 39) & (k % 21 != 0)
    motor = np.isin(k, [63, 84])
    fam = {
        "bpf_k2": dict(alpha=float(a[1]), se=float(alpha[1][1])),
        "even_4_40": dict(
            median=float(np.median(a[even])),
            q10=float(np.percentile(a[even], 10)),
            q90=float(np.percentile(a[even], 90)),
        ),
        "odd_3_39": dict(
            median=float(np.median(a[odd])),
            q10=float(np.percentile(a[odd], 10)),
            q90=float(np.percentile(a[odd], 90)),
        ),
        "motor_63_84": dict(values=[float(x) for x in a[motor]]),
    }
    # one exponent for the family vs its own per order
    for name, sel in (("even_4_40", even), ("odd_3_39", odd)):
        q1 = float(np.median(a[sel]))
        own, single = [], []
        for i in np.where(sel)[0]:
            g = groups_for(recs, lambda v, i=i: v["line_db"][i])
            X = np.concatenate([x - x.mean() for x, _ in g])
            Y = np.concatenate([y - y.mean() for _, y in g])
            own.append(np.sqrt(((Y - a[i] * X) ** 2).mean()))
            single.append(np.sqrt(((Y - q1 * X) ** 2).mean()))
        fam[name]["rms_single_db"] = float(np.mean(single))
        fam[name]["rms_own_db"] = float(np.mean(own))
    out["families"] = fam

    # ---- floor at fixed frequency, per band, raw and ambient-subtracted ----
    def floor_rows(v, b, net):
        lo, hi = BANDS[b]
        sel = (v["f"] >= lo) & (v["f"] < hi)
        q20 = np.percentile(v["p"][:, sel], 20, axis=1) / 0.223
        if not net:
            return float(10 * np.log10(q20).mean())
        n = np.median(v["p_off"][:, sel], axis=1) / 0.693
        return float(10 * np.log10(np.maximum(q20 - n, 0.05 * q20)).mean())

    out["floor_fixed_f"] = []
    for b, (lo, hi) in enumerate(BANDS):
        raw = pooled_slope(groups_for(recs, lambda v, b=b: floor_rows(v, b, False)))
        net = pooled_slope(groups_for(recs, lambda v, b=b: floor_rows(v, b, True)))
        over_amb = np.mean(
            [
                floor_rows(v, b, False)
                - 10
                * np.log10(
                    np.median(v["p_off"][:, (v["f"] >= lo) & (v["f"] < hi)], axis=1) / 0.693
                ).mean()
                for v in recs.values()
            ]
        )
        out["floor_fixed_f"].append(
            dict(
                lo=lo,
                hi=hi,
                b=raw[0],
                se=raw[1],
                b_net=net[0],
                se_net=net[1],
                over_ambient_db=float(over_amb),
            )
        )

    # ---- floor under harmonic k (fixed order) ----
    def floor_under_k(v, kk):
        f, p, s = v["f"], v["p"], v["s"]
        df = f[1]
        c = kk * s
        sel = (f > c - 60) & (f < c + 60) & (np.abs(f - c) > 2 * df)
        return float(10 * np.log10(np.percentile(p[:, sel], 20, axis=1)).mean())

    out["floor_fixed_order"] = []
    for kk in range(2, K_MAX + 1):
        q, se, rms = pooled_slope(groups_for(recs, lambda v, kk=kk: floor_under_k(v, kk)))
        out["floor_fixed_order"].append(dict(k=kk, beta=q, se=se, rms_db=rms))

    # ---- axis test: fixed f vs f/s, per motor and sub-band ----
    edges = 300 * 2 ** (np.arange(0, 40) / 6)
    edges = edges[edges < 18000]
    fc = np.sqrt(edges[:-1] * edges[1:])
    grid = np.arange(-2, 8.01, 0.1)

    def axis_resid(curves, spd, xsel, beta):
        Y = []
        for t, (xa, ya) in curves.items():
            Y.append(np.interp(np.log10(fc[xsel]), np.log10(xa), ya) - beta * np.log10(spd[t]))
        Y = np.array(Y)
        return float(10 * np.sqrt(((Y - Y.mean(0)) ** 2).mean()))

    out["axis_test"] = []
    for lo, hi in [(300, 16000), (300, 1500), (1500, 4000), (4000, 16000)]:
        for m in MOTORS:
            spd = {t: recs[f"motor_Motor{m}_{t}"]["s"] for t in THR}
            cf, cs = {}, {}
            for t in THR:
                v = recs[f"motor_Motor{m}_{t}"]
                cf[t] = (
                    fc,
                    np.array(
                        [
                            band_floor_db(v["f"], v["p"], edges[i], edges[i + 1]) / 10
                            for i in range(len(fc))
                        ]
                    ),
                )
                xe = edges / S_REF * v["s"]  # the same x = f/s bands, in Hz at this speed
                cs[t] = (
                    fc,
                    np.array(
                        [
                            band_floor_db(v["f"], v["p"], xe[i], xe[i + 1]) / 10
                            for i in range(len(fc))
                        ]
                    ),
                )
            xsel = (fc >= lo) & (fc < hi)
            bf = min(grid, key=lambda b: axis_resid(cf, spd, xsel, b))
            bs = min(grid, key=lambda b: axis_resid(cs, spd, xsel, b))
            out["axis_test"].append(
                dict(
                    lo=lo,
                    hi=hi,
                    motor=m,
                    fixed_f=dict(b=float(bf), resid_db=axis_resid(cf, spd, xsel, bf)),
                    strouhal=dict(beta=float(bs), resid_db=axis_resid(cs, spd, xsel, bs)),
                )
            )

    # ---- b(f) = b0 + b1 log2(f / 2 kHz): one tilt parameter per motor ----
    out["floor_tilt_law"] = []
    for m in MOTORS:
        rids = [f"motor_Motor{m}_{t}" for t in THR]
        xs = np.array([10 * np.log10(recs[r]["s"]) for r in rids])
        Yb = np.array(
            [
                [
                    band_floor_db(recs[r]["f"], recs[r]["p"], edges[i], edges[i + 1])
                    for i in range(len(fc))
                ]
                for r in rids
            ]
        )
        bk = np.array([pooled_slope([(xs, Yb[:, i])])[0] for i in range(len(fc))])
        sel = (fc >= 300) & (fc < 16000)
        A = np.c_[np.ones(sel.sum()), np.log2(fc[sel] / 2000.0)]
        coef, *_ = np.linalg.lstsq(A, bk[sel], rcond=None)
        resid_scalar = float(np.sqrt(((bk[sel] - bk[sel].mean()) ** 2).mean()))
        resid_tilt = float(np.sqrt(((bk[sel] - A @ coef) ** 2).mean()))
        out["floor_tilt_law"].append(
            dict(
                motor=m,
                b0_at_2khz=float(coef[0]),
                b1_per_octave=float(coef[1]),
                sd_b_scalar=resid_scalar,
                sd_b_tilt=resid_tilt,
            )
        )

    SR_DIR.mkdir(parents=True, exist_ok=True)
    (SR_DIR / "speed_laws.json").write_text(json.dumps(out, indent=1))

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    se = np.array([s for _, s, _ in alpha])
    col = np.where(k % 21 == 0, "tab:red", np.where(k % 2 == 0, "tab:blue", "tab:orange"))
    ax[0].errorbar(k, a, yerr=se, fmt="none", ecolor="0.7", lw=0.8)
    ax[0].scatter(k, a, c=col, s=14, zorder=3)
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set(
        xlabel="order k",
        ylabel="power exponent α_k (P ∝ s^α)",
        title="lines: blue even, orange odd, red 21·n",
        ylim=(-6, 12),
        xlim=(0, K_MAX + 1),
    )
    ax[0].grid(alpha=0.3)
    fo = out["floor_fixed_order"]
    ax[1].errorbar(
        [d["k"] for d in fo],
        [d["beta"] for d in fo],
        yerr=[d["se"] for d in fo],
        fmt="o",
        ms=3,
        color="tab:green",
        ecolor="0.7",
    )
    ax[1].set(
        xlabel="order k (floor under k·s)",
        ylabel="β (fixed order)",
        title="floor under harmonic k vs speed",
        ylim=(-3, 6),
    )
    ax[1].grid(alpha=0.3)
    ff = out["floor_fixed_f"]
    fcen = [np.sqrt(d["lo"] * d["hi"]) for d in ff]
    ax[2].errorbar(
        fcen,
        [d["b"] for d in ff],
        yerr=[d["se"] for d in ff],
        fmt="o-",
        color="tab:purple",
        label="raw 20th pct",
    )
    ax[2].plot(fcen, [d["b_net"] for d in ff], "s--", color="tab:gray", label="ambient removed")
    for d in out["floor_tilt_law"]:
        ax[2].plot(
            fcen,
            d["b0_at_2khz"] + d["b1_per_octave"] * np.log2(np.array(fcen) / 2000.0),
            lw=0.6,
            color="0.6",
        )
    ax[2].set(
        xscale="log",
        xlabel="frequency (Hz)",
        ylabel="b (fixed f)",
        title="floor at fixed frequency vs speed",
        ylim=(0, 7),
    )
    ax[2].legend()
    ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(SR_DIR / "speed_laws.png", dpi=130)

    print(json.dumps(out["families"], indent=1))
    for d in out["floor_fixed_f"]:
        print(
            f"floor {d['lo']:>5}-{d['hi']:<5} b={d['b']:4.1f}±{d['se']:3.1f} net {d['b_net']:4.1f}  over ambient {d['over_ambient_db']:4.1f} dB"
        )
    for d in out["floor_tilt_law"]:
        print(
            f"M{d['motor']}: b(f) = {d['b0_at_2khz']:.2f} + {d['b1_per_octave']:.2f} log2(f/2k); sd of b over bands: scalar {d['sd_b_scalar']:.2f} -> tilt {d['sd_b_tilt']:.2f}"
        )
    for d in out["axis_test"]:
        print(
            f"axis {d['lo']:>5}-{d['hi']:<5} M{d['motor']}: fixed-f b={d['fixed_f']['b']:4.1f} resid {d['fixed_f']['resid_db']:.2f} dB | f/s β={d['strouhal']['beta']:4.1f} resid {d['strouhal']['resid_db']:.2f} dB"
        )


if __name__ == "__main__":
    main()
