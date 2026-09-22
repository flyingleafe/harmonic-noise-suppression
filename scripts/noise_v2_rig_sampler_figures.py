"""The four figures of the v2 rig-sampler study.

Reads ``results/noise_v2/rig_sampler/structure.json`` (phase A: tiers, ladder,
real-window band levels, coverage ladder) and the two built banks, and writes
into ``results/noise_v2/rig_sampler/``:

``widths_ladder.png``
    Each coordinate's between-restart, between-rotor and between-rig spread, as
    the STRENGTH at which a strength-1 draw reaches the higher tiers. The
    figure's point is that one scalar cannot walk every coordinate up together.
``coverage.png``
    Per rig: the cloud's 1/3-octave envelope against every real window, and the
    coverage ladder that chose the strength.
``profiles.png``
    The sampled harmonic profiles against their anchors, per rig, plus the
    realised spread of the decomposed coordinates.
``path_cloud.png``
    The hard bank: realised mixing coordinate against U(0, 1), and the cloud's
    coordinates moving from DREGON to Michael's along it.

``PYTHONPATH=src python scripts/noise_v2_rig_sampler_figures.py``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from experiments.noise_model import rig_sampler as RS  # noqa: E402

OUT = Path("results/noise_v2/rig_sampler")
RIG_COLOUR = {"dregon": "#1f77b4", "michaels": "#d62728"}


def _save(fig: Figure, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def widths_ladder(st: dict[str, Any]) -> Path:
    rows = st["ladder"]
    keys = sorted(rows, key=lambda k: rows[k]["strength_to_rotor"] or 0.0)
    y = np.arange(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), width_ratios=(1.25, 1.0))

    ax = axes[0]
    for i, key in enumerate(keys):
        row = rows[key]
        ax.plot(
            [row["between_restart"], row["between_rotor"], row["between_rig"]],
            [i, i, i],
            "-",
            color="0.75",
            lw=1.2,
            zorder=1,
        )
        ax.plot(row["between_restart"], i, "o", ms=6, color="#2b7", zorder=3)
        ax.plot(row["between_rotor"], i, "s", ms=6, color="#e8a", zorder=3)
        ax.plot(row["between_rig"], i, "D", ms=6, color="#36c", zorder=3)
    ax.set_xscale("log")
    ax.set_yticks(y, keys, fontsize=8)
    ax.set_xlabel("per-draw sigma (coordinate units)")
    ax.grid(alpha=0.25, axis="x", which="both")
    ax.set_title("the three measured tiers", fontsize=10)
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="", color="#2b7", label="between-restart (= strength 1)"),
            Line2D([], [], marker="s", ls="", color="#e8a", label="between-rotor (4 bench motors)"),
            Line2D([], [], marker="D", ls="", color="#36c", label="between-rig (DREGON vs FLY125)"),
        ],
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.11),
        ncol=2,
    )

    ax = axes[1]
    to_rotor = [rows[k]["strength_to_rotor"] for k in keys]
    to_rig = [rows[k]["strength_to_rig"] for k in keys]
    ax.barh(y - 0.2, to_rotor, height=0.38, color="#e8a", label="strength reaching between-rotor")
    ax.barh(y + 0.2, to_rig, height=0.38, color="#36c", label="strength reaching between-rig")
    chosen = st.get("coverage_ladder", {}).get("chosen_strength")
    if chosen:
        ax.axvline(chosen, color="k", ls="--", lw=1.4, label=f"strength used ({chosen:g})")
    ax.set_xscale("log")
    ax.set_yticks(y, [""] * len(keys))
    ax.set_xlabel("strength multiplier")
    ax.grid(alpha=0.25, axis="x", which="both")
    ax.set_title("one scalar cannot reach every tier at once", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, -0.11), ncol=2)
    fig.suptitle(
        "v2 rig sampler: measured widths and the strength ladder "
        f"(strength 1 = between-restart; level_db is between-clip, {st['between_clip']['level_db']:.2f} dB)",
        fontsize=11,
    )
    return _save(fig, "widths_ladder.png")


def coverage_figure(st: dict[str, Any], banks: dict[str, dict[str, Any]]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), width_ratios=(1, 1, 0.9))
    for ax, rig in zip(axes[:2], ("dregon", "michaels"), strict=True):
        centres, real = RS.real_bands_of(st, rig)
        cloud = np.asarray(banks["easy"]["bands"][rig], dtype=np.float64)
        lo, hi = cloud.min(axis=0), cloud.max(axis=0)
        ax.fill_between(
            centres, lo, hi, color=RIG_COLOUR[rig], alpha=0.22, label="easy cloud (min-max)"
        )
        ax.plot(centres, cloud.mean(axis=0), color=RIG_COLOUR[rig], lw=1.8, label="cloud mean")
        hard = np.asarray(banks["hard"]["bands"], dtype=np.float64)
        ax.plot(
            centres, hard.min(axis=0), color="0.35", lw=1.0, ls=":", label="hard cloud (min-max)"
        )
        ax.plot(centres, hard.max(axis=0), color="0.35", lw=1.0, ls=":")
        for i, row in enumerate(real):
            ax.plot(
                centres,
                row,
                color="k",
                lw=1.0,
                alpha=0.75,
                label="real windows" if i == 0 else None,
            )
        ax.axvline(RS.LEVEL_BAND_HZ[0], color="0.4", lw=0.8, ls="--")
        ax.set_xscale("log")
        ax.set_xlabel("1/3-octave centre (Hz)")
        ax.set_ylabel("band level (dB, periodogram units)")
        cov = banks["easy"]["coverage"][rig]["above_300hz"]
        ax.set_title(f"{rig}: easy cloud brackets {100 * cov:.1f} % of bands > 300 Hz", fontsize=10)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=7, loc="lower left")

    ax = axes[2]
    ladder = st["coverage_ladder"]
    strengths = [float(s) for s in ladder["strengths"]]
    for preset, ls in (("easy", "-"), ("hard", "--")):
        for rig in ("dregon", "michaels"):
            vals = [
                ladder["rows"][f"{s:g}"][preset]["coverage"][rig]["above_300hz"] for s in strengths
            ]
            ax.plot(
                strengths,
                vals,
                ls,
                marker="o",
                ms=4,
                color=RIG_COLOUR[rig],
                label=f"{preset} {rig}",
            )
    ax.axhline(RS.COVERAGE_TARGET, color="k", lw=1.2, ls=":")
    ax.axvline(ladder["chosen_strength"], color="k", lw=1.4, ls="--")
    acc = ax.twinx()
    acc.plot(
        strengths,
        [1.0 - ladder["rows"][f"{s:g}"]["easy"]["rejection_rate"] for s in strengths],
        color="0.5",
        lw=1.2,
        marker="s",
        ms=3,
    )
    acc.set_ylabel("acceptance (grey)", color="0.4", fontsize=9)
    ax.set_xlabel("strength")
    ax.set_ylabel("coverage above 300 Hz")
    ax.set_title(
        f"ladder: chosen {ladder['chosen_strength']:g} ({ladder['n_per_strength']} draws/cell)",
        fontsize=10,
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")
    return _save(fig, "coverage.png")


def profiles_figure(banks: dict[str, Any], anchors: dict[str, Any]) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0))
    for col, rig in enumerate(("dregon", "michaels")):
        ax = axes[0, col]
        anchor = np.asarray(anchors[rig]["params"]["profile"]["profile_db"], dtype=np.float64)[0]
        k = np.arange(1, anchor.size + 1)
        cloud = banks["easy"]["profiles"][rig]
        for row in cloud[:120]:
            ax.plot(k, row[: anchor.size], color=RIG_COLOUR[rig], lw=0.35, alpha=0.18)
        ax.plot(k, anchor, color="k", lw=1.8, label="anchor, rotor 0")
        ax.set_xscale("log")
        ax.set_xlabel("rotor order k")
        ax.set_ylabel("profile (dB)")
        ax.set_title(f"{rig}: 120 of 1024 sampled profiles, rotor 0", fontsize=10)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)

        ax = axes[1, col]
        gains = np.asarray(banks["easy"]["coords"][rig]["gain"])
        slopes = np.asarray(banks["easy"]["coords"][rig]["slope"])
        ax.scatter(gains, slopes, s=5, alpha=0.3, color=RIG_COLOUR[rig], label="sampled rigs")
        a = RS.decompose(anchor)
        ax.scatter([a.gain], [a.slope], s=90, marker="*", color="k", zorder=4, label="anchor")
        ax.set_xlabel("rotor-0 profile gain (dB)")
        ax.set_ylabel("trend slope (dB / decade)")
        ax.set_title(
            f"realised spread: gain sd {gains.std(ddof=1):.2f} dB, slope sd "
            f"{slopes.std(ddof=1):.2f} dB/dec",
            fontsize=10,
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("easy bank: the neighbourhood around each rig's own fit", fontsize=11)
    fig.tight_layout()
    return _save(fig, "profiles.png")


def path_figure(banks: dict[str, Any], st: dict[str, Any]) -> Path:
    t = np.asarray(banks["hard"]["t"], dtype=np.float64)
    ks = RS.ks_uniform(t)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    ax = axes[0]
    ax.hist(t, bins=20, range=(0, 1), color="0.6", edgecolor="k", lw=0.5)
    ax.axhline(t.size / 20, color="k", ls="--", lw=1.2)
    ax.set_xlabel("mixing coordinate t")
    ax.set_ylabel("entries")
    ax.set_title(
        f"realised t: mean {ks['mean']:.3f}, KS {ks['ks']:.4f} vs threshold "
        f"{ks['threshold_95']:.4f}",
        fontsize=10,
    )
    ax.grid(alpha=0.25)

    ax = axes[1]
    gains = np.asarray(banks["hard"]["coords"]["gain"])
    ax.scatter(t, gains, s=5, alpha=0.3, color="#555")
    for rig in ("dregon", "michaels"):
        anchor = RS.decompose(
            np.asarray(
                RS.pin_speed_laws(RS.load_fit(RS.ANCHORS[rig]["cruise"]))["params"]["profile"][
                    "profile_db"
                ],
                dtype=np.float64,
            )[0][: RS.PATH_K_MAX]
        )
        ax.axhline(anchor.gain, color=RIG_COLOUR[rig], ls="--", lw=1.3, label=f"{rig} anchor")
    ax.set_xlabel("mixing coordinate t")
    ax.set_ylabel("rotor-0 profile gain (dB)")
    ax.set_title("the cloud walks from DREGON to FLY125", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[2]
    carried = np.asarray(banks["hard"]["standby"], dtype=bool)
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(t, edges) - 1, 0, 9)
    frac = [float(carried[idx == i].mean()) if (idx == i).any() else np.nan for i in range(10)]
    ax.bar(0.5 * (edges[:-1] + edges[1:]), frac, width=0.09, color="#7aa", edgecolor="k", lw=0.5)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="the policy: P(standby) = t")
    ax.set_xlabel("mixing coordinate t")
    ax.set_ylabel("fraction carrying the standby slot")
    ax.set_title(f"standby is a regime POLICY: {100 * carried.mean():.1f} % overall", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle(
        f"hard bank: {t.size} draws on the cruise-to-cruise path, K = 1..{RS.PATH_K_MAX}",
        fontsize=11,
    )
    fig.tight_layout()
    return _save(fig, "path_cloud.png")


def read_bank(path: Path, structure: dict[str, Any], probe_bands: bool = False) -> dict[str, Any]:
    """The bank's per-entry coordinates, from the file alone."""
    bank = json.loads(Path(path).read_text())
    entries = bank["entries"]
    out: dict[str, Any] = {"n": len(entries), "provenance": bank["provenance"]}
    if bank["provenance"]["preset"] == "easy":
        out["profiles"] = {}
        out["coords"] = {}
        for rig in ("dregon", "michaels"):
            rows = [e for e in entries if e["provenance"]["anchor"] == rig]
            prof = np.stack(
                [np.asarray(e["cruise"]["params"]["profile"]["profile_db"])[0] for e in rows]
            )
            parts = [RS.decompose(p) for p in prof]
            out["profiles"][rig] = prof
            out["coords"][rig] = {
                "gain": [q.gain for q in parts],
                "slope": [q.slope for q in parts],
            }
    else:
        out["t"] = [float(e["provenance"]["t"]) for e in entries]
        out["standby"] = [e["standby"] is not None for e in entries]
        prof = np.stack(
            [np.asarray(e["cruise"]["params"]["profile"]["profile_db"])[0] for e in entries]
        )
        parts = [RS.decompose(p) for p in prof]
        out["coords"] = {"gain": [q.gain for q in parts], "slope": [q.slope for q in parts]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--easy", default="data/rig_banks/noise_v2_easy_n2048.json")
    ap.add_argument("--hard", default="data/rig_banks/noise_v2_hard_n2048.json")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    st = RS.load_fit(RS.STRUCTURE_PATH)

    banks = {
        "easy": read_bank(Path(args.easy), st),
        "hard": read_bank(Path(args.hard), st),
    }
    # the band curves and the coverage come from the build reports, which
    # already hold what the guards measured
    reports = {
        preset: json.loads((OUT / f"build_{preset}.json").read_text())
        for preset in ("easy", "hard")
    }
    banks["easy"]["coverage"] = reports["easy"]["coverage"]
    banks["hard"]["coverage"] = reports["hard"]["coverage"]

    probe = RS.ModelProbe()
    anchors = RS.pinned_anchors()
    # band curves: re-probe a SUBSAMPLE (the build's own curves are not kept in
    # the bank, and 200 probes is 40 s)
    bands_easy: dict[str, Any] = {}
    for rig in ("dregon", "michaels"):
        rows = [
            e
            for e in json.loads(Path(args.easy).read_text())["entries"]
            if e["provenance"]["anchor"] == rig
        ][:100]
        bands_easy[rig] = np.stack([probe.bands(e["cruise"], RS.PROBE_CRUISE_RPS)[1] for e in rows])
    banks["easy"]["bands"] = bands_easy
    hard_rows = json.loads(Path(args.hard).read_text())["entries"][:100]
    banks["hard"]["bands"] = np.stack(
        [probe.bands(e["cruise"], RS.PROBE_CRUISE_RPS)[1] for e in hard_rows]
    )

    widths_ladder(st)
    coverage_figure(st, banks)
    profiles_figure(banks, anchors)
    path_figure(banks, st)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
