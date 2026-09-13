"""What the fit actually learned, as figures: one page of parameters per fit.

The fit's export IS the renderer's parameter vector, so these panels are not an
interpretation — they are the object that gets rendered. One PNG per fit with
six axes (per-order profile, width law and coherent share, floor shape, the
per-microphone pattern, the speed law, the per-clip scores), plus a JSON of the
same numbers for the explainer's tables.

    python scripts/_fit_params_panel.py --fit results/S2/cruise_8clip_refined.json \
        --label "FLY125 cruise, refined labels" --slug cruise_refined
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/explainers/refit")
ROTOR_COLOURS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")


def _arr(x: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(x, dtype=np.float64))


def _profile(params: dict[str, Any]) -> np.ndarray:
    """(R, K) static line level in dB, including the time mean of any drift."""
    profile = np.atleast_2d(_arr(params["profile_db"]))
    h = _arr(params.get("h_db", 0.0))
    if h.ndim == 3:
        profile = profile + h.mean(axis=-1)[: profile.shape[0]]
    return profile


def summarise(summary: dict[str, Any]) -> dict[str, Any]:
    """Scores and tied parameters, one row per clip."""
    rig = summary.get("rig") or {}
    rows = []
    for cid, entry in summary["clips"].items():
        s, p = entry["scores"], entry["params"]
        rows.append(
            dict(
                clip=cid,
                group=entry.get("group"),
                nll_fit=round(float(s["nll_fit"]), 4),
                nll_loo=round(float(s["nll_loo"]), 4),
                excess_over_loo=round(float(s["excess_over_loo"]), 4),
                n_cells=int(s["n_cells"]),
                k_half=round(float(p.get("coherence_k_half", 0.0)), 3),
                gamma0_hz=round(float(_arr(p["gamma0"])[0]), 4),
                gamma_slope_hz=round(float(_arr(p["gamma_slope"])[0]), 4),
                amp_exp=round(float(p.get("amp_exp", 0.0)), 3),
                floor_exp=round(float(p.get("floor_exp", 0.0)), 3),
                floor_mean_db=round(float(p.get("floor_mean_db", 0.0)), 2),
            )
        )
    return dict(
        data=summary.get("data"),
        seconds=summary.get("seconds"),
        rig_ties=summary.get("rig_ties") or summary.get("rig_spec"),
        tied=dict(
            gamma0_hz=rig.get("gamma0"),
            gamma_slope_hz=rig.get("gamma_slope"),
            width_power=rig.get("width_power"),
            amp_exp=rig.get("amp_exp"),
            floor_exp=rig.get("floor_exp"),
            floor_tilt_db_oct=rig.get("floor_tilt_db_oct"),
        ),
        clips=rows,
    )


def figure(summary: dict[str, Any], label: str, slug: str) -> Path:
    clips = list(summary["clips"].items())
    first = clips[0][1]["params"]
    profile = _profile(first)
    n_rotors, n_orders = profile.shape
    k = np.arange(1, n_orders + 1)

    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"{label} — the fitted renderer parameters", fontsize=13)

    # 1. per-order line profile, every clip faint, rotor means solid
    a = ax[0, 0]
    for _cid, entry in clips:
        pr = _profile(entry["params"])
        for r in range(pr.shape[0]):
            a.plot(k, pr[r], color=ROTOR_COLOURS[r % 4], alpha=0.18, lw=0.8)
    mean = np.mean([_profile(e["params"]) for _c, e in clips], axis=0)
    for r in range(n_rotors):
        a.plot(k, mean[r], color=ROTOR_COLOURS[r % 4], lw=1.8, label=f"rotor {r}")
    a.set_xscale("log")
    a.set_xlabel("harmonic order $k$")
    a.set_ylabel("line level (dB)")
    a.set_title(f"per-order profile — {n_rotors} rotors x {n_orders} orders")
    a.grid(alpha=0.3)
    a.legend(fontsize=8)

    # 2. the width law and the coherent share on one axis
    a = ax[0, 1]
    for _cid, entry in clips:
        p = entry["params"]
        gamma = np.atleast_2d(_arr(p["gamma"]))
        a.plot(k[: gamma.shape[1]], gamma[0], color="#1f77b4", alpha=0.25, lw=0.9)
    a.set_xlabel("harmonic order $k$")
    a.set_ylabel(r"pedestal half width $\gamma_k$ (Hz)", color="#1f77b4")
    a.tick_params(axis="y", labelcolor="#1f77b4")
    a.grid(alpha=0.3)
    b = a.twinx()
    for _cid, entry in clips:
        kh = float(entry["params"].get("coherence_k_half", 0.0))
        if kh > 0:
            b.plot(k, np.exp(-((k / kh) ** 2)), color="#d62728", alpha=0.35, lw=0.9)
    b.set_ylabel(r"coherent share $w_k$", color="#d62728")
    b.tick_params(axis="y", labelcolor="#d62728")
    khs = [float(e["params"].get("coherence_k_half", 0.0)) for _c, e in clips]
    a.set_title(
        f"width law and coherent share — $k_{{1/2}}$ {min(khs):.2f}-{max(khs):.2f}", fontsize=10
    )

    # 3. the broadband floor: shape at its knots, per clip
    a = ax[0, 2]
    for _cid, entry in clips:
        p = entry["params"]
        hz = _arr(p["floor_ctrl_hz"])
        db = _arr(p["floor_shape_db"]) + float(p.get("floor_mean_db", 0.0))
        a.plot(hz, db, marker="o", ms=3, lw=1.2, alpha=0.7)
    a.set_xscale("log")
    a.set_xlabel("Hz")
    a.set_ylabel("floor level (dB)")
    tilts = [float(e["params"].get("floor_tilt_db_oct", 0.0)) for _c, e in clips]
    a.set_title(
        f"floor shape at 14 knots — tilt {np.mean(tilts):+.2f} dB/oct",
        fontsize=10,
    )
    a.grid(alpha=0.3)

    # 4. the per-microphone structure the probe reads
    a = ax[1, 0]
    mg = _arr(first.get("mic_gain_db", 0.0))
    if mg.ndim == 2:
        for r in range(mg.shape[1]):
            a.plot(mg[:, r], "o-", color=ROTOR_COLOURS[r % 4], label=f"line gain, rotor {r}")
    mf = _arr(first.get("mic_floor_db", 0.0))
    if mf.size > 1:
        a.plot(mf - mf.mean(), "s--", color="#111111", label="floor offset")
    ga = _arr(first.get("gain_all_db", 0.0))
    if ga.size > 1:
        a.plot(ga - ga.mean(), "^:", color="#7f7f7f", label="common gain")
    a.set_xlabel("microphone")
    a.set_ylabel("dB")
    a.set_title("per-microphone pattern (first clip)", fontsize=10)
    a.grid(alpha=0.3)
    a.legend(fontsize=7)

    # 5. the speed law: what happens to level when the rotor speeds up
    a = ax[1, 1]
    speeds = np.linspace(20.0, 100.0, 200)
    for _cid, entry in clips:
        p = entry["params"]
        q = float(p.get("amp_exp", 2.5))
        qf = float(p.get("floor_exp", q))
        a.plot(speeds, 10 * q * np.log10(speeds / 80.0), color="#1f77b4", alpha=0.3, lw=0.9)
        a.plot(speeds, 10 * qf * np.log10(speeds / 80.0), color="#2ca02c", alpha=0.3, lw=0.9)
    qs = [float(e["params"].get("amp_exp", 0.0)) for _c, e in clips]
    qfs = [float(e["params"].get("floor_exp", 0.0)) for _c, e in clips]
    a.axvline(80.0, color="#999999", lw=0.8, ls=":")
    a.set_xlabel("rotor speed (rev/s)")
    a.set_ylabel("level re 80 rev/s (dB)")
    a.set_title(
        f"speed law — lines $q$ {np.mean(qs):.2f}, floor $q$ {np.mean(qfs):.2f}",
        fontsize=10,
    )
    a.grid(alpha=0.3)

    # 6. the scores, per clip, against their own leave-one-out reference
    a = ax[1, 2]
    idx = np.arange(len(clips))
    nll = [float(e["scores"]["nll_fit"]) for _c, e in clips]
    loo = [float(e["scores"]["nll_loo"]) for _c, e in clips]
    excess = [float(e["scores"]["excess_over_loo"]) for _c, e in clips]
    a.bar(idx - 0.2, nll, 0.4, label="fit", color="#1f77b4")
    a.bar(idx + 0.2, loo, 0.4, label="LOO smoother", color="#999999")
    for i, e in zip(idx, excess):
        a.annotate(f"{e:+.2f}", (i, min(nll[i], loo[i])), fontsize=7, ha="center", va="top")
    a.set_xticks(idx)
    a.set_xticklabels([c.split("_")[-1] for c, _e in clips], fontsize=7)
    a.set_xlabel("clip (annotated: excess over LOO)")
    a.set_ylabel("nats / cell")
    a.set_title(
        f"Whittle score — median excess {np.median(excess):+.3f} nats/cell",
        fontsize=10,
    )
    a.grid(alpha=0.3, axis="y")
    a.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = OUT / f"params_{slug}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", type=Path, action="append", required=True)
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--slug", action="append", required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {}
    for path, label, slug in zip(args.fit, args.label, args.slug, strict=True):
        summary = json.loads(path.read_text())
        png = figure(summary, label, slug)
        index[slug] = dict(label=label, fit=str(path), figure=png.name, **summarise(summary))
        print(f"{slug}: {png}  ({len(summary['clips'])} clips)", flush=True)
    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    print(f"wrote {OUT / 'index.json'}")


if __name__ == "__main__":
    main()
