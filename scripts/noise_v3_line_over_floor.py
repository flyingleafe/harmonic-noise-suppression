"""Method B per rotor to order 74: line over local floor on the REAL fit pools.

The Fig E estimator of ``scripts/noise_v3_latent_runaway_figs.py``
(``pool_batch``, ``frame_carrier``, ``order_aligned``, ``line_floor_db``): on the
fit's own STFT pool, mic-mean power; per frame the bins at
``round(k f_r / df) + (-ALIGN_HALF..ALIGN_HALF)`` for that frame's label
carrier; MEAN over frames; line = the centre +-1 bin. Here per rotor (no rotor
mean), orders :data:`ORDERS`, on the real pool and on the round-3b model's
expected periodogram (latents zero) through the identical estimator.

Two floors:

* ``ring``: Fig E's, the median of the cells ``FLOOR_OFFS`` = 5..10 bins away.
  The order spacing is ``f_r / df`` ~ 10 bins, so this ring reaches the
  neighbouring orders ``k +- 1`` of the same rotor.
* ``mid``: spacing-aware, per frame the cells at ``+-(h - 1, h, h + 1)`` with
  ``h = round(f_r / (2 df))`` (mid-way between order ``k`` and its neighbours),
  each of the six MEAN over frames, then their median.

Plus the peak offset: the argmax (bins, ``-ALIGN_HALF..ALIGN_HALF``) of the
frame-mean aligned cells, so a systematic label bias shows up.

    python scripts/noise_v3_line_over_floor.py --run    # uni-cpu
    python scripts/noise_v3_line_over_floor.py --plot   # laptop

``--run`` writes ``results/noise_v3/line_over_floor/{dregon,michaels_cruise}.json``;
``--plot`` writes the explainer's ``lof74_{rig}.png`` and ``lof74_table.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/noise_v3/line_over_floor"
DOCS = ROOT / "docs/explainers/noise-model-v3-latent-runaway"
TABLE = DOCS / "lof74_table.json"

ORDERS = tuple(range(1, 75))
TABLE_ORDERS = (1, 2, 3, 5, 10, 20, 35, 50, 70)
OVER_DB = 3.0
#: rig key -> (round-3b fit, the support set its pool was built from, label kind)
RIGS = {
    "dregon": (
        "results/noise_v3/fits_r3b/dregon_room2_floor__flight_v3.json",
        "dregon-floor",
        "motors_command",
    ),
    "michaels_cruise": (
        "results/noise_v3/fits_r3b/michaels_fly125_cruise__flight_v3.json",
        "michaels-cruise",
        "rps_refined",
    ),
}
NAMES = {"dregon": "DREGON", "michaels_cruise": "Michael's cruise"}


def _figs() -> Any:
    path = ROOT / "scripts/noise_v3_latent_runaway_figs.py"
    spec = importlib.util.spec_from_file_location("noise_v3_latent_runaway_figs", str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


FG = _figs()


def mid_floor(power: np.ndarray, carrier: np.ndarray, df: float) -> tuple[np.ndarray, np.ndarray]:
    """``(R, K)`` spacing-aware floor power and ``(R,)`` the frame-mean ``h``:
    per frame the cells ``+-(h-1, h, h+1)`` around ``round(k f_r / df)``,
    ``h = round(f_r / (2 df))``; each cell MEAN over frames, then the median."""
    pm = power.mean(axis=0)  # (N, F)
    rows = np.arange(pm.shape[0])[:, None]
    out = np.empty((carrier.shape[0], len(ORDERS)))
    h_mean = np.empty(carrier.shape[0])
    for r in range(carrier.shape[0]):
        h = np.rint(carrier[r] / (2.0 * df)).astype(np.int64)
        h_mean[r] = h.mean()
        offs = np.stack([-(h + 1), -h, -(h - 1), h - 1, h, h + 1], axis=1)  # (N, 6)
        for i, k in enumerate(ORDERS):
            j = np.rint(k * carrier[r] / df).astype(np.int64)
            out[r, i] = np.median(pm[rows, j[:, None] + offs].mean(axis=0))
    return out, h_mean


def estimate(power: np.ndarray, carrier: np.ndarray, df: float) -> dict[str, Any]:
    """Every per-rotor, per-order number of one spectrum ``(M, N, F)``."""
    aligned = FG.order_aligned(power, carrier, df, ORDERS)  # (R, K, 2H+1)
    ring = FG.line_floor_db(aligned)
    mid, h_mean = mid_floor(power, carrier, df)
    mid_db = FG._db(mid)
    return dict(
        line_db=ring["line_db"],
        floor_ring_db=ring["floor_db"],
        over_ring_db=ring["over_db"],
        floor_mid_db=mid_db,
        over_mid_db=ring["line_db"] - mid_db,
        peak_offset_bins=aligned.argmax(axis=-1) - FG.ALIGN_HALF,
        mid_h_mean_bins=h_mean,
    )


def run(threads: int) -> None:
    import torch

    from experiments.noise_model import model as MD

    torch.set_num_threads(threads)
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    OUT.mkdir(parents=True, exist_ok=True)
    for key, (path, set_name, label_kind) in RIGS.items():
        t0 = time.time()
        print(f"{key}: building the pool ({set_name})", flush=True)
        fit = FG.load(ROOT / path)
        batch = FG.pool_batch(fit, v3=True, set_name=set_name)
        df = float(batch.grid.diagnostics["sr"]) / float(batch.grid.diagnostics["n_fft"])
        data = batch.power.numpy()
        carrier = FG.frame_carrier(batch)
        print(f"{key}: r3b expected periodogram, latents zero", flush=True)
        m0, whittle = FG.evaluate(batch, MD.params_from_dict(fit["params"]), None)
        payload = dict(
            rig=key,
            fit=path,
            support_set=set_name,
            supports=list(fit["supports"]),
            label_kind=label_kind,
            git=sha,
            orders=list(ORDERS),
            estimator=dict(
                align_half=FG.ALIGN_HALF,
                line="mean of cells |o| <= 1",
                floor_ring=f"median of cells |o| in {FG.FLOOR_OFFS[0]}..{FG.FLOOR_OFFS[1]}",
                floor_mid="per frame cells +-(h-1, h, h+1), h = round(f_r / (2 df)); "
                "each mean over frames; median of the six",
                peak_offset="argmax over o in -H..H of the frame-mean aligned cells",
                power="mic mean of the channel-normalised pool, frame mean per cell",
            ),
            pool=dict(
                n_mics=int(data.shape[0]),
                n_frames=int(data.shape[1]),
                df_hz=df,
                carrier_mean_rev_s=carrier.mean(axis=1),
                carrier_min_rev_s=carrier.min(axis=1),
                carrier_max_rev_s=carrier.max(axis=1),
                spacing_mean_bins=carrier.mean(axis=1) / df,
            ),
            r3b_whittle_zero_latents_nats=whittle,
            real=estimate(data, carrier, df),
            r3b=estimate(m0, carrier, df),
        )
        dst = OUT / f"{key}.json"
        dst.write_text(json.dumps(FG._r(payload), indent=1))
        print(f"{key}: wrote {dst} ({time.time() - t0:.0f} s)", flush=True)


def fig(d: dict[str, Any], dst: Path) -> None:
    plt = FG.plt
    k = np.asarray(d["orders"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4), sharex=True, sharey=True)
    for r, ax in enumerate(axes.ravel()):
        for src, color, name in (("real", FG.C_DATA, "real"), ("r3b", FG.C_R2, "r3b model")):
            e = d[src]
            ax.plot(k, e["over_mid_db"][r], color=color, lw=1.4, label=f"{name}, mid floor")
            ax.plot(
                k,
                e["over_ring_db"][r],
                color=color,
                lw=1.0,
                ls="--",
                alpha=0.8,
                label=f"{name}, ring floor (5–10 bins)",
            )
        ax.axhline(OVER_DB, color="#888888", lw=0.8, ls=":")
        ax.axhline(0.0, color="#888888", lw=0.5)
        ax.axvline(70, color="#2ca02c", lw=0.8, alpha=0.5)
        ax.set_title(f"rotor {r + 1} (mean {d['pool']['carrier_mean_rev_s'][r]:.1f} rev/s)")
        if r >= 2:
            ax.set_xlabel("order k")
        if r % 2 == 0:
            ax.set_ylabel("line over floor (dB)")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"{NAMES[d['rig']]}: Method B per rotor, labels {d['label_kind']}, "
        f"{d['pool']['n_frames']} frames"
    )
    fig.tight_layout()
    fig.savefig(dst)
    plt.close(fig)
    print(f"wrote {dst}")


def plot() -> None:
    table: dict[str, Any] = dict(orders=list(TABLE_ORDERS))
    for key in RIGS:
        d = FG.load(OUT / f"{key}.json")
        fig(d, DOCS / f"lof74_{key}.png")
        idx = [d["orders"].index(k) for k in TABLE_ORDERS]
        e = d["real"]
        i70 = d["orders"].index(70)
        table[key] = dict(
            label_kind=d["label_kind"],
            git=d["git"],
            n_frames=d["pool"]["n_frames"],
            carrier_mean_rev_s=d["pool"]["carrier_mean_rev_s"],
            **{
                name: [[row[i] for i in idx] for row in e[name]]
                for name in ("over_mid_db", "over_ring_db", "peak_offset_bins")
            },
            k70=dict(
                real_over_mid_db=[row[i70] for row in e["over_mid_db"]],
                real_over_ring_db=[row[i70] for row in e["over_ring_db"]],
                real_peak_offset_bins=[row[i70] for row in e["peak_offset_bins"]],
                r3b_over_mid_db=[row[i70] for row in d["r3b"]["over_mid_db"]],
            ),
        )
        for r in range(len(e["over_mid_db"])):
            print(
                f"{key} rotor {r + 1}: over_mid "
                + " ".join(f"k{k}={e['over_mid_db'][r][i]:.1f}" for k, i in zip(TABLE_ORDERS, idx))
                + " | offs "
                + " ".join(f"{e['peak_offset_bins'][r][i]}" for i in idx)
            )
    TABLE.write_text(json.dumps(table, indent=1))
    print(f"wrote {TABLE}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run", action="store_true", help="the pools and the estimator (uni-cpu)")
    ap.add_argument("--plot", action="store_true", help="figures + table JSON (laptop)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args(argv)
    if not (args.run or args.plot):
        ap.error("name --run and/or --plot")
    if args.run:
        run(args.threads)
    if args.plot:
        plot()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
