#!/usr/bin/env python
"""Figures of ``docs/explainers/noise-model-v3-latent-runaway.qmd``.

The explainer answers one question about the noise-model v3 campaign: the
alternation (rig <-> per-window latents) converges, yet every moment update
(sigma <- the fitted latents' spread) makes the latents wider. What do the
latents take up? Every number here is read from the committed fits and wander
records or evaluated with the fits' own forward model on the fit's own pool;
nothing is fitted.

Inputs:

* ``results/noise_v3/fits{,_r2}/dregon_room2_floor__flight_v3.json`` (round 1
  and round 2 selected fits) and their ``restarts/`` siblings;
* ``results/noise_v3/fits_r2/michaels_fly125_{cruise,standby}__flight_v3.json``
  and ``results/noise_v3/fits_r2/restarts/`` (the alternation moves);
* ``results/noise_v3/wander/{dregon,michaels}{,_mm1,_mm2}.json`` (the sigma
  ladder);
* ``results/noise_v2/rounds/round5/fits/dregon_room2_floor__flight_profile.json``
  (the v2 fit both rounds were warm-started from);
* the DREGON pool's five windows (``supports.support_set("dregon-floor")``),
  loaded, thinned and channel-normalised exactly as ``noise_v2_fit.py flight
  --mode flight_v3`` built them;
* Figs G-I3: the parameter view of ``notebooks/noise_lab.py`` (``param_view`` /
  ``draw_param_view``, the notebook's comb-over-floor cell) on four draws of
  the round-3b DREGON fit's v3 prior (``noise_v3_checks.prior_draw``, seeds
  0-3), the three round-3b fits (the page's v3; round 2 again for § 8), the v2
  fits of ``noise_lab.FIT_PATHS``, and the LEGACY rigs: the two stage-2 anchor
  fits the ``rig_easy`` / ``rig_hard`` banks were drawn around
  (``noise_lab.LegacyFit``) and two entries of the hard bank
  (``noise_lab.LegacyBank``). The legacy model has no expected-periodogram
  function, so its view is the mean periodogram of a render
  (:func:`legacy_view`). Every figure is written once per rotor (only that
  rotor's comb on) and once with all four (``_{all,rotor1..4}.png``);
* Figs J-K (§ Listen): one real window per rig (:data:`LISTEN_WINDOWS`, held
  out of every fit's pool) through the notebook's own primitives —
  ``trajectory("real")``, ``real_clip``, ``render_all`` of ``V2Fit(rig)`` and
  ``V3Fit(rig, round=V3_ROUND)`` under the notebook's level rule, ``line_stats``,
  ``spectrogram_figure`` — written as 16-bit WAVs to ``audio/``; the LTAS is
  ``revised_eval.absolute_ltas_bands`` on the bands inside
  :data:`LISTEN_LTAS_HZ`. ``--round TAG`` renders the same windows with the
  legacy anchor fits (:data:`LISTEN_LEGACY`, ``LegacyFit(...,
  dynamics="donor")``), v2, round 2 and round TAG; ``--listen-add KEY`` adds
  one source to an existing ``figdata_TAG.json`` without re-rendering the rest.

    PYTHONPATH=src python scripts/noise_v3_latent_runaway_figs.py            # evaluate + plot
    PYTHONPATH=src python scripts/noise_v3_latent_runaway_figs.py --rigs-only # Figs G-I3 only
    PYTHONPATH=src python scripts/noise_v3_latent_runaway_figs.py --listen-only # Figs J-K + WAVs
    PYTHONPATH=src python scripts/noise_v3_latent_runaway_figs.py --plot-only
    PYTHONPATH=src python scripts/noise_v3_latent_runaway_figs.py --round r3a  # r3a beside r2
    PYTHONPATH=src python scripts/noise_v3_latent_runaway_figs.py --round r3b --listen-add legacy

Writes ``docs/explainers/noise-model-v3-latent-runaway/`` (``fig_*.png``,
``figdata.json``, every plotted number, and ``audio/*.wav``).
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("docs/explainers/noise-model-v3-latent-runaway")
DATA = OUT / "figdata.json"

POOL = "dregon_room2_floor"
FITS = {
    "r1": Path(f"results/noise_v3/fits/{POOL}__flight_v3.json"),
    "r2": Path(f"results/noise_v3/fits_r2/{POOL}__flight_v3.json"),
}
RESTART_DIRS = {
    "r1": Path("results/noise_v3/fits/restarts"),
    "r2": Path("results/noise_v3/fits_r2/restarts"),
}
V2_FIT = Path(f"results/noise_v2/rounds/round5/fits/{POOL}__flight_profile.json")
WANDER_DIR = Path("results/noise_v3/wander")
CHANNEL_GAINS = "results/noise_v2/mic_gains/mic_gains.json"
#: the fit CLI's pool build (scripts/noise_v2_fit.py: K_CAP, --frame-stride default,
#: V3_CHUNK_FRAMES; the payload's front_end.sr_work)
K_CAP = 130
FRAME_STRIDE = 4
CHUNK_FRAMES = 16
V3_SR_WORK = 32000
V2_SR_WORK = 64000
TOL = 1e-4

#: wander families in ladder order: (label, sigma key or v group index)
FAMILIES = (
    ("d", "sigma_d_db"),
    ("u", "sigma_u_db"),
    ("u_j", "sigma_uj_db"),
    ("v k1–2", 0),
    ("v k3–8", 1),
    ("v k9–24", 2),
    ("v k25–60", 3),
    ("v k≥61", 4),
)
#: v order groups as 0-based slices of the K axis (k_edges 1, 3, 9, 25, 61, 110)
V_GROUPS = {"v k9–24": (8, 24), "v k25–60": (24, 60), "v k≥61": (60, None)}
#: floor control points shown in the u_j track panel (Hz, nearest knot)
UJ_SHOW_HZ = (71.0, 607.0, 3387.0)
#: orders of the profile figure
E_ORDERS = tuple(range(1, 25))
#: order-aligned cells either side of k f_r; line = |o| <= 1, local floor = |o| in 5..10
ALIGN_HALF = 10
FLOOR_OFFS = (5, 10)
#: Fig F bands: floor control points from this frequency up, inside the fit band
F_MIN_HZ = 500.0
F_MAX_HZ = 7900.0
#: Figs G-I3, the parameter view (``notebooks/noise_lab.py`` ``param_view``):
#: every rotor at one speed, the carrier span midpoint when it is outside the
#: fit's span; the prior draws' seeds; the floor readouts; the visibility bar
RIG_RPS = 80.0
PRIOR_SEEDS = (0, 1, 2, 3)
RIG_FLOOR_AT_HZ = (100.0, 1000.0, 4000.0)
RIG_OVER_DB = 3.0
RIG_FMIN_HZ = 20.0
RIG_NAMES = {"dregon": "DREGON", "cruise": "Michael's cruise", "standby": "Michael's standby"}
#: the pooled v3 fit of each rig key, by file stem, inside a round's fit directory
V3_POOLS = {
    "dregon": POOL,
    "cruise": "michaels_fly125_cruise",
    "standby": "michaels_fly125_standby",
}
#: the page's v3: its rig views (Figs G-H), § Listen and the prior draws; round 2
#: stays in § 8 as the runaway's contrast (Fig H2)
V3_ROUND = "r3b"
#: one PNG per entry: every rotor's comb on, then one rotor's comb alone
ROTOR_VIEWS = ("all", "rotor1", "rotor2", "rotor3", "rotor4")
#: Fig I2: the stage-2 anchor fits both legacy banks were drawn around (the
#: banks' ``provenance.anchors``: ``cruise_8clip.json:fly125_cruise_00`` and
#: ``dregon_room2_cruise_refined.json:free-flight_nosource_room2_cruise_00``,
#: ``noise_lab.LEGACY_FIT_CLIPS``), built as ``noise_lab.LegacyFit`` builds them
#: (``dynamics="donor"``: the zero-perturbation bank entry); Fig I3: the first
#: two entries of the bank the ``rig_hard_5050`` arm trained on
LEGACY_FITS = {"dregon": "dregon_cruise_refined", "michael": "michael_cruise"}
LEGACY_BANK = "hard"
LEGACY_BANK_ENTRIES = (0, 1)
LEGACY_NAMES = {"dregon": "DREGON", "michael": "Michael's cruise"}
#: the legacy view's render: seconds of constant speed, 8 mics (the bank's), the
#: seed, and the periodogram of ``experiments.stochastic_fit.data`` (2048-point
#: Hann, hop 512, the param view's frame)
LEGACY_VIEW_S = 12.0
LEGACY_VIEW_MICS = 8
LEGACY_VIEW_SEED = 0
#: § Listen: one 10 s real window per rig, held out of every fit's pool (the
#: DREGON pool reads free-flight 18-26 s; Michael's fits read FLY125 only),
#: every rotor inside the fits' carrier spans; the notebook's render cell
#: (``render_all(..., seed=0, n_mics=1, level=("window", 0.1))``): mic 0, a
#: per-clip RMS of 0.1; the clip order of the figure rows
LISTEN_WINDOWS = {
    "dregon": dict(dataset="DREGON-frames", recording="free-flight_nosource_room2", offset_s=50.0),
    "michaels": dict(dataset="michaels-frames", recording="FLY124", offset_s=40.0),
}
LISTEN_S = 10.0
LISTEN_SEED = 0
LISTEN_LEVEL = ("window", 0.1)
LISTEN_LTAS_HZ = (100.0, 7000.0)
LISTEN_LABELS = {
    "real": "real recording",
    "legacy": "legacy anchor fit (donor dynamics)",
    "v2": "v2 fit",
    "v3": "v3 round-2 fit",
}
#: § 8 Listen again: the anchor fit of :data:`LEGACY_FITS` each listen rig was
#: drawn around, rendered as the legacy training streams render a bank entry
#: (``noise_lab.LegacyFit``'s default ``dynamics="donor"``, line mode ``fm``)
LISTEN_LEGACY = {"dregon": LEGACY_FITS["dregon"], "michaels": LEGACY_FITS["michael"]}
LISTEN_LEGACY_DYNAMICS = "donor"
LISTEN_DYN_DB = 45.0
AUDIO = OUT / "audio"

C_R1, C_R2, C_V2, C_DATA = "#1f77b4", "#d62728", "#7f7f7f", "#000000"
WIN_LABELS = ("free-flight", "hovering", "updown", "rectangle", "spinning")

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)


# ── pure reads ─────────────────────────────────────────────────────────────


def load(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _db(x: Any) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-300))


def _r(x: Any, nd: int = 4) -> Any:
    """JSON-safe rounded copy (numpy to lists, non-finite to None): ``nd``
    decimals at ``|v| >= 1``, ``nd`` significant digits below (a move of 8.8e-5
    nats per cell must not round to 1e-4)."""
    if isinstance(x, dict):
        return {str(k): _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return _r(x.tolist(), nd)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        v = float(x)
        if not math.isfinite(v):
            return None
        return round(v, nd) if abs(v) >= 1.0 else float(f"{v:.{nd}g}")
    if isinstance(x, np.integer):
        return int(x)
    return x


def moves() -> dict[str, Any]:
    """Fig A: the Whittle move per cell of every alternation round, per restart."""
    out: dict[str, Any] = {}
    for rnd, d in RESTART_DIRS.items():
        pools = (
            (POOL,) if rnd == "r1" else (POOL, "michaels_fly125_cruise", "michaels_fly125_standby")
        )
        for pool in pools:
            for p in sorted(d.glob(f"{pool}__flight_v3__s*.json")):
                opt = load(p)["optimiser"]
                mv = [h.get("whittle_move_per_cell") for h in opt["history"][1:]]
                out[f"{rnd}/{pool}/{p.stem.rsplit('__', 1)[1]}"] = dict(
                    path=str(p),
                    move=mv,
                    rounds_run=int(opt["rounds_run"]),
                    rounds_allowed=int(opt["rounds"]),
                    converged=bool(opt.get("alternation_converged")),
                )
    return out


def ladder() -> dict[str, Any]:
    """Fig B: sigma per family, measured (mm0) -> mm1 -> mm2, per rig."""
    out: dict[str, Any] = {}
    for rig, label in (("dregon", "DREGON"), ("michaels", "Michael's cruise")):
        steps = {}
        for tag, suffix in (("mm0", ""), ("mm1", "_mm1"), ("mm2", "_mm2")):
            path = WANDER_DIR / f"{rig}{suffix}.json"
            w = load(path)
            vs = w["sigma_v_db_by_order"]["sigma_db"]
            steps[tag] = dict(
                path=str(path),
                sigma_db=[
                    float(w[key]) if isinstance(key, str) else float(vs[key]) for _, key in FAMILIES
                ],
            )
        out[rig] = dict(label=label, families=[f for f, _ in FAMILIES], steps=steps)
    return out


def latent_arrays(fit: dict[str, Any]) -> dict[str, np.ndarray]:
    """``{name: (W, ..., B)}`` of the fit's recorded latent tracks (dB)."""
    wins = sorted(fit["latents"]["windows"], key=lambda w: int(w["window"]))
    return {
        n: np.stack([np.asarray(w[n], dtype=np.float64) for w in wins])
        for n in ("d", "v", "u", "uj")
    }


def decompose(x: np.ndarray) -> dict[str, float]:
    """Mean square of ``(W, ..., B)`` tracks split into the part SHARED by every
    window and block (the pool mean per track), the window means' deviation
    from it, and the within-window part. The three add up to the mean square."""
    pool = x.mean(axis=(0, -1), keepdims=True)
    win = x.mean(axis=-1, keepdims=True)
    ms = float(np.mean(x**2))
    static = float(np.mean(np.broadcast_to(pool, x.shape) ** 2))
    between = float(np.mean(np.broadcast_to(win - pool, x.shape) ** 2))
    within = float(np.mean((x - win) ** 2))
    return dict(
        rms_db=math.sqrt(ms),
        static_rms_db=math.sqrt(static),
        between_rms_db=math.sqrt(between),
        within_rms_db=math.sqrt(within),
        static_share=static / ms,
        between_share=between / ms,
        within_share=within / ms,
        rms_without_static_db=math.sqrt(max(ms - static, 0.0)),
    )


def family_tracks(lat: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {"d": lat["d"], "u": lat["u"], "u_j": lat["uj"]}
    for g, (k0, k1) in V_GROUPS.items():
        out[g] = lat["v"][:, :, k0:k1]
    return out


def prior_sigma(wander: dict[str, Any]) -> dict[str, float]:
    vs = wander["sigma_v_db_by_order"]["sigma_db"]
    return {
        "d": float(wander["sigma_d_db"]),
        "u": float(wander["sigma_u_db"]),
        "u_j": float(wander["sigma_uj_db"]),
        "v k9–24": float(vs[2]),
        "v k25–60": float(vs[3]),
        "v k≥61": float(vs[4]),
    }


def tracks(fits: dict[str, dict[str, Any]], ctrl_hz: np.ndarray) -> dict[str, Any]:
    """Fig C: the recorded tracks (d, u, three u_j knots, v k9-24 quantiles)."""
    out: dict[str, Any] = dict(block_s=float(fits["r2"]["latents"]["block_s"]), ctrl_hz=ctrl_hz)
    js = [int(np.argmin(np.abs(np.log(ctrl_hz / f)))) for f in UJ_SHOW_HZ]
    out["uj_knots"] = js
    for rnd, fit in fits.items():
        lat = latent_arrays(fit)
        vg = lat["v"][:, :, 8:24]  # (W, R, 16, B)
        flat = vg.reshape(vg.shape[0], -1, vg.shape[-1])
        out[rnd] = dict(
            prior_sigma_db=prior_sigma(fit["params"]["wander"]),
            wander_path=fit["priors"]["wander"].get("path", "results/noise_v3/wander/dregon.json"),
            d=lat["d"],
            u=lat["u"],
            uj=lat["uj"][:, js],
            uj_pool_mean_db=lat["uj"].mean(axis=(0, 2)),
            u_pool_mean_db=float(lat["u"].mean()),
            d_pool_mean_db=lat["d"].mean(axis=(0, 2)),
            v924_q=np.quantile(flat, (0.1, 0.5, 0.9), axis=1),
            decomposition={k: decompose(v) for k, v in family_tracks(lat).items()},
        )
    return out


# ── forward-model evaluations ──────────────────────────────────────────────


def pool_batch(fit: dict[str, Any], *, v3: bool, set_name: str = "dregon-floor") -> Any:
    """The fit's pool exactly as the CLI built it from support set ``set_name``
    (``v3``: channel-normalised on the fit's rig, the v3 work rate; else the v2
    fit's unnormalised build)."""
    from experiments.noise_model import fit as FT
    from experiments.noise_model import model as MD
    from experiments.noise_model import supports as SUP

    name = str(fit["support"])
    rig = "dregon" if name.startswith("dregon") else "michaels"
    specs = {s.name: s for s in SUP.support_set(set_name)}
    members = []
    sups = [SUP.load_support(specs[n]) for n in fit["supports"]]
    for s in sups:
        members.append(
            (
                s.name,
                np.asarray(s.power, dtype=np.float64),
                np.asarray(s.carrier_rev_s_audio, dtype=np.float64),
                np.asarray(s.frame_starts, dtype=np.int64),
            )
        )
    batch = MD.flight_batch(
        name=name,
        members=members,
        sr=int(sups[0].sr),
        n_fft=int(sups[0].n_fft),
        hop=int(sups[0].hop),
        k_cap=K_CAP,
        frame_stride=FRAME_STRIDE,
        channel_gains=FT.load_channel_gains(CHANNEL_GAINS, rig=rig) if v3 else None,
        device="cpu",
        chunk_frames=CHUNK_FRAMES,
        sr_work=V3_SR_WORK if v3 else V2_SR_WORK,
    )
    return MD.with_blocks(batch, float(fit["latents"]["block_s"])) if v3 else batch


def v3_fits(tag: str) -> dict[str, Path]:
    """``{rig key: fit path}`` of v3 round ``tag`` (a ``noise_lab.V3_FIT_DIRS`` key)."""
    d = Path(_noise_lab().V3_FIT_DIRS[tag])
    return {key: d / f"{stem}__flight_v3.json" for key, stem in V3_POOLS.items()}


def static_shares(rounds: tuple[str, ...] = ("r1", "r2")) -> dict[str, Any]:
    """:func:`decompose` of every family of every pool's selected fit, per round."""
    out: dict[str, Any] = {}
    for rnd in rounds:
        for key, path in v3_fits(rnd).items():
            fams = family_tracks(latent_arrays(load(path)))
            out[f"{rnd}/{V3_POOLS[key]}"] = dict(
                path=str(path), families={k: decompose(v) for k, v in fams.items()}
            )
    return out


def frame_carrier(batch: Any) -> np.ndarray:
    """``(R, N)`` each frame's mean label carrier (rev/s), from the work grid."""
    return batch.rate_work.mean(dim=-1).numpy()


def window_latents(lat: dict[str, np.ndarray]) -> dict[int, Any]:
    import torch

    from experiments.noise_model import model as MD

    def t(a: np.ndarray) -> Any:
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float64)

    return {
        w: MD.WindowLatents(
            d=t(lat["d"][w]), v=t(lat["v"][w]), u=t(lat["u"][w]), uj=t(lat["uj"][w])
        )
        for w in range(lat["d"].shape[0])
    }


def evaluate(batch: Any, params: Any, latents: dict[int, Any] | None) -> tuple[np.ndarray, float]:
    """``(M (M, N, F), Whittle nats)`` of ``params`` (+ ``latents``) on ``batch``."""
    import torch

    from experiments.noise_model import model as MD

    t0 = time.time()
    b = batch if latents is None else replace(batch, latents=latents)
    with torch.no_grad():
        m = MD.forward(b, params, unit_autocorr=True)
        nats = float(MD.whittle_risk(batch, m))
    print(f"    forward {time.time() - t0:.0f} s, whittle {nats:.1f}", flush=True)
    return m.numpy(), nats


def floor_only(params: Any) -> Any:
    """``params`` with every line switched off (profile at -300 dB)."""
    import torch

    return replace(params, profile_db=torch.full_like(params.profile_db, -300.0))


def order_aligned(
    power: np.ndarray, carrier: np.ndarray, df: float, orders: tuple[int, ...]
) -> np.ndarray:
    """``(R, K, 2H+1)`` mic- and frame-mean power on the cells ``-H..H`` bins
    from each frame's own ``k f_r`` (``carrier`` ``(R, N)`` rev/s)."""
    pm = power.mean(axis=0)  # (N, F)
    n = pm.shape[0]
    offs = np.arange(-ALIGN_HALF, ALIGN_HALF + 1)
    out = np.empty((carrier.shape[0], len(orders), offs.size))
    rows = np.arange(n)[:, None]
    for r in range(carrier.shape[0]):
        for i, k in enumerate(orders):
            j = np.rint(k * carrier[r] / df).astype(np.int64)
            out[r, i] = pm[rows, j[:, None] + offs[None, :]].mean(axis=0)
    return out


def line_floor_db(aligned: np.ndarray) -> dict[str, np.ndarray]:
    """Line level (|o| <= 1 mean), local floor (median of |o| in 5..10) and
    their ratio, dB, per rotor and order."""
    h = ALIGN_HALF
    line = aligned[..., h - 1 : h + 2].mean(axis=-1)
    offs = np.abs(np.arange(-h, h + 1))
    sel = (offs >= FLOOR_OFFS[0]) & (offs <= FLOOR_OFFS[1])
    floor = np.median(aligned[..., sel], axis=-1)
    return dict(line_db=_db(line), floor_db=_db(floor), over_db=_db(line) - _db(floor))


def ridge_test(
    batch: Any, params: Any, fit: dict[str, Any], lat: dict[str, np.ndarray], whittle_fit: float
) -> dict[str, Any]:
    """Move the part of the latents SHARED by every window and block (the pool
    mean per track) into the rig, where a static offset belongs, and price it.

    Lines: ``profile_db[r, k] += mean d_r + mean v_rk`` (exact: the forward adds
    ``d + v`` to ``profile_db``). Floor: the control values ``sigma_B L z`` gain
    ``mean u + mean u_j`` through an EXACT solve for ``z`` (the rig spline and
    the latents share the basis ``shape_psd``). The Whittle term is unchanged
    by construction (checked by one evaluation); what changes is the OU prior
    of the latents and the rig priors of ``profile_db`` (Normal, sd 10 dB,
    about the measured centre) and ``z`` (N(0, I)). Each family is priced alone
    and all together."""
    import torch

    from data_processing.noise_model.v3 import Wander
    from experiments.noise_model import model as MD

    wander = Wander.from_mapping(load(fit["priors"]["wander"]["path"]))
    p_hat = np.asarray(fit["diagnostics"]["measured"]["profile_centre_db"], dtype=np.float64)
    p_sd = float(fit["priors"]["profile_db"]["sd"])
    prof = params.profile_db.numpy()
    sig_b = float(params.floor.shape_sd_db)
    chol = batch.grid.floor.shape_chol.numpy()
    z = params.floor.shape_z.numpy()
    ctrl = sig_b * chol @ z

    static = {n: lat[n].mean(axis=(0, -1)) for n in ("d", "v", "u", "uj")}
    ou_fit = MD.ou_prior_nats(window_latents(lat), wander)

    def price(names: tuple[str, ...]) -> dict[str, float]:
        moved = {n: lat[n] - (static[n][None, ..., None] if n in names else 0.0) for n in lat}
        ou = MD.ou_prior_nats(window_latents(moved), wander)
        delta_p = np.zeros_like(prof)
        if "d" in names:
            delta_p += static["d"][:, None]
        if "v" in names:
            delta_p += static["v"]
        d_ctrl = np.zeros_like(ctrl)
        if "u" in names:
            d_ctrl += float(static["u"])
        if "uj" in names:
            d_ctrl += static["uj"]
        z_new = np.linalg.solve(sig_b * chol, ctrl + d_ctrl)
        d_prof = float(
            (((prof + delta_p - p_hat) ** 2 - (prof - p_hat) ** 2) / (2.0 * p_sd**2)).sum()
        )
        d_z = 0.5 * float(z_new @ z_new - z @ z)
        return dict(
            ou_prior_change_nats=ou - ou_fit,
            profile_prior_change_nats=d_prof,
            z_prior_change_nats=d_z,
            total_change_nats=ou - ou_fit + d_prof + d_z,
            z_norm_new=float(np.linalg.norm(z_new)),
        )

    out: dict[str, Any] = dict(
        ou_prior_fit_nats=ou_fit,
        ou_prior_recorded_nats=float(fit["objective"]["ou_neg_log_prior_nats"]),
        z_norm_fit=float(np.linalg.norm(z)),
        n_cells=int(batch.n_cells),
        tol_nats=TOL * int(batch.n_cells),
        families={n: price((n,)) for n in ("d", "v", "u", "uj")},
    )
    out["all"] = price(("d", "v", "u", "uj"))
    # the Whittle check of the full move
    moved = {n: lat[n] - static[n][None, ..., None] for n in lat}
    d_ctrl = float(static["u"]) + static["uj"]
    params_moved = replace(
        params,
        profile_db=params.profile_db + torch.as_tensor(static["d"][:, None] + static["v"]),
        floor=replace(
            params.floor, shape_z=torch.as_tensor(np.linalg.solve(sig_b * chol, ctrl + d_ctrl))
        ),
    )
    _, w_moved = evaluate(batch, params_moved, window_latents(moved))
    out["whittle_fit_nats"] = whittle_fit
    out["whittle_moved_nats"] = w_moved
    return out


def window_spectra(
    batch: Any,
    w: int,
    data: np.ndarray,
    m0: np.ndarray,
    m1: np.ndarray,
    f0: np.ndarray,
    f1: np.ndarray,
    carrier: np.ndarray,
    df: float,
) -> dict[str, Any]:
    """Fig D: window ``w``'s frame-mean spectra (dB) on the fit band, and the
    smooth-vs-lines split of its data / zero-latent-model ratio."""
    band = batch.band.numpy()
    freqs = np.arange(data.shape[-1]) * df
    sel = np.asarray(batch.frame_window) == w
    mean_t = lambda a: a[:, sel].mean(axis=1)[:, band]  # noqa: E731  (M, Fb)
    d_m, m0_m, m1_m = mean_t(data), mean_t(m0), mean_t(m1)
    f0_mm, f1_mm = mean_t(f0).mean(axis=0), mean_t(f1).mean(axis=0)
    fb = freqs[band]
    ratio = _db(d_m.mean(axis=0)) - _db(m0_m.mean(axis=0))
    latent = _db(m1_m.mean(axis=0)) - _db(m0_m.mean(axis=0))
    per_mic = _db(d_m) - _db(m0_m)
    # a 1/3-octave running mean in log f: the smooth part of the gap
    lf = np.log2(fb)
    smooth = np.array([ratio[np.abs(lf - x) <= 1.0 / 6.0].mean() for x in lf])
    # line bins: within 1 bin of k f_r (window-mean carrier) for k <= 24
    cw = carrier[:, sel].mean(axis=1)
    line = np.zeros(fb.size, dtype=bool)
    for f_r in cw:
        for k in range(1, 25):
            line |= np.abs(fb - k * f_r) <= 1.01 * df
    rough = ratio - smooth
    bands = {}
    edges = (30.0, 150.0, 300.0, 500.0, 1000.0, 2000.0, 3000.0, 4500.0, 7900.0)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        s = (fb >= lo) & (fb < hi)
        bands[f"{lo:.0f}-{hi:.0f}"] = dict(
            gap_db=float(ratio[s].mean()),
            smooth_db=float(smooth[s].mean()),
            latent_fill_db=float(latent[s].mean()),
            floor_shift_db=float((_db(f1_mm) - _db(f0_mm))[s].mean()),
            mic_spread_db=float((per_mic[:, s].mean(axis=1)).std()),
        )
    return dict(
        window=int(w),
        name=str(batch.members[w]) if batch.members else "",
        freqs_hz=fb,
        data_db=_db(d_m),
        data_mean_db=_db(d_m.mean(axis=0)),
        model_zero_db=_db(m0_m.mean(axis=0)),
        model_fit_db=_db(m1_m.mean(axis=0)),
        floor_zero_db=_db(f0_mm),
        floor_fit_db=_db(f1_mm),
        ratio_db=ratio,
        ratio_smooth_db=smooth,
        latent_fill_db=latent,
        per_mic_minus_mean_db=per_mic - ratio[None, :],
        line_bins=line,
        carrier_rev_s=cw,
        verdict=dict(
            smooth_rms_db=float(np.sqrt(np.mean(smooth**2))),
            rough_rms_line_bins_db=float(np.sqrt(np.mean(rough[line] ** 2))),
            rough_rms_other_bins_db=float(np.sqrt(np.mean(rough[~line] ** 2))),
            corr_latent_vs_smooth_gap=float(np.corrcoef(latent, smooth)[0, 1]),
            bands=bands,
        ),
    )


def mic_residuals(
    batch: Any,
    data: np.ndarray,
    m0: np.ndarray,
    lat: dict[str, np.ndarray],
    ctrl_hz: np.ndarray,
    df: float,
) -> dict[str, Any]:
    """Fig F: per (window, block, knot >= 500 Hz, mic) the level residual
    ``data - zero-latent model`` (dB, the knot's band, the block's frames) and
    the floor latent at that knot, ``u + u_j``."""
    freqs = np.arange(data.shape[-1]) * df
    band = batch.band.numpy()
    lc = np.log(ctrl_hz)
    js = [j for j, f in enumerate(ctrl_hz) if F_MIN_HZ <= f <= F_MAX_HZ]
    fw = np.asarray(batch.frame_window)
    fbk = np.asarray(batch.frame_block)
    n_w, n_b = lat["u"].shape
    y = np.full((n_w, n_b, len(js), data.shape[0]), np.nan)
    x = np.full((n_w, n_b, len(js)), np.nan)
    for i, j in enumerate(js):
        lo = math.exp(0.5 * (lc[j - 1] + lc[j]))
        hi = math.exp(0.5 * (lc[j] + lc[j + 1])) if j + 1 < lc.size else F_MAX_HZ
        cols = band & (freqs >= lo) & (freqs < min(hi, F_MAX_HZ))
        for w in range(n_w):
            for b in range(n_b):
                rows = (fw == w) & (fbk == b)
                if not rows.any():
                    continue
                num = data[:, rows][:, :, cols].mean(axis=(1, 2))
                den = m0[:, rows][:, :, cols].mean(axis=(1, 2))
                y[w, b, i] = _db(num) - _db(den)
                x[w, b, i] = lat["u"][w, b] + lat["uj"][w, j, b]
    ok = np.isfinite(x)
    ybar = np.nanmean(y, axis=-1)
    dev = y - ybar[..., None]

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1])

    # within-knot versions: each knot's pool mean removed from x and y
    xk = x - np.nanmean(x, axis=(0, 1), keepdims=True)
    ybk = ybar - np.nanmean(ybar, axis=(0, 1), keepdims=True)
    # the per-mic part of every mic sums to zero over the mics, so its POOLED
    # correlation with a mic-shared x is zero by construction; per mic it is not
    devk = dev - np.nanmean(dev, axis=(0, 1), keepdims=True)
    # the per-mic part's static share: its pool mean per (knot, mic), the same
    # in every window and block
    dev_static = np.nanmean(dev, axis=(0, 1))  # (J, M)
    # per-window static per-mic deviation (block mean, then knot mean)
    dev_win = np.nanmean(dev, axis=(1, 2))  # (W, M)
    return dict(
        knots_hz=[float(ctrl_hz[j]) for j in js],
        x_db=x,
        y_db=y,
        n_points=int(ok.sum()),
        r_common=corr(x, ybar),
        r_common_within_knot=corr(xk, ybk),
        r_per_mic=[corr(x, y[..., m]) for m in range(y.shape[-1])],
        r_per_mic_within_knot=[
            corr(xk, y[..., m] - np.nanmean(y[..., m], axis=(0, 1), keepdims=True))
            for m in range(y.shape[-1])
        ],
        r_mic_part_per_mic_within_knot=[corr(xk, devk[..., m]) for m in range(y.shape[-1])],
        mic_part_static_share=float(np.nanmean(dev_static**2) / np.nanmean(dev**2)),
        mic_part_static_db=dev_static,
        sd_common_db=float(np.nanstd(ybar)),
        sd_common_within_knot_db=float(np.nanstd(ybk)),
        sd_mic_deviation_db=float(np.nanstd(dev)),
        sd_x_db=float(np.nanstd(x)),
        sd_x_within_knot_db=float(np.nanstd(xk)),
        window_mic_deviation_db=dev_win,
        window_mic_deviation_sd_db=float(np.nanstd(dev_win)),
    )


def _noise_lab() -> Any:
    """``notebooks/noise_lab.py``, whose ``param_view`` / ``draw_param_view``
    are the notebook's parameter-view cell."""
    root = Path(__file__).resolve().parent.parent
    if str(root / "notebooks") not in sys.path:
        sys.path.insert(0, str(root / "notebooks"))
    import noise_lab

    return noise_lab


def _sibling(name: str) -> Any:
    """A sibling script as a module."""
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rig_span(fit: dict[str, Any]) -> tuple[float, float]:
    """The fit's carrier span (``diagnostics.batch.carrier_{min,max}_rev_s``)."""
    b = fit["diagnostics"]["batch"]
    return float(b["carrier_min_rev_s"]), float(b["carrier_max_rev_s"])


def rig_rps(fit: dict[str, Any]) -> float:
    """:data:`RIG_RPS`, or the midpoint of the fit's carrier span when the span
    does not hold it."""
    lo, hi = rig_span(fit)
    return RIG_RPS if lo <= RIG_RPS <= hi else 0.5 * (lo + hi)


def view_numbers(view: dict[str, Any], rotor: int | None = None) -> dict[str, Any]:
    """The numbers of one parameter view: line over floor per order (k = 1, 2
    singled out), the orders over :data:`RIG_OVER_DB`, and the widest gamma of
    the orders shown, over rotor ``rotor`` (``None``: every rotor)."""
    f, floor, keep = view["f"], view["floor_db"], view["keep"]
    over = np.asarray(view["line_db"], dtype=np.float64) - np.interp(view["freq_hz"], f, floor)
    gamma = np.atleast_2d(np.asarray(view["gamma_hz"], dtype=np.float64))
    if rotor is not None:
        gamma = gamma[rotor : rotor + 1]
    return dict(
        over_db=over,
        over_k1_db=float(over[0]),
        over_k2_db=float(over[1]),
        n_orders_shown=int(keep.sum()),
        n_orders_over=int((over[keep] > RIG_OVER_DB).sum()),
        gamma_max_hz=float(gamma[:, keep].max()),
    )


def panel(
    view: dict[str, Any], rotor_full_db: list[np.ndarray], span: tuple[float, float] | None
) -> dict[str, Any]:
    """One row of Figs G-I3 from a parameter view with every rotor's comb on and
    ``rotor_full_db``, one expected periodogram per rotor with only that rotor's
    comb on: :func:`view_numbers` of both, the floor at :data:`RIG_FLOOR_AT_HZ`,
    and the expected periodogram over the floor halfway between the two lines
    around 1 kHz, i.e. what the comb puts BETWEEN its lines."""
    f, floor, rps = view["f"], view["floor_db"], float(view["rps"])
    mid_hz = (math.floor(1000.0 / rps) + 0.5) * rps
    rotors = []
    for r, full in enumerate(rotor_full_db):
        solo = {**view, "full_db": full, "line_db": np.interp(view["freq_hz"], f, full)}
        rotors.append(dict(full_db=full, line_db=solo["line_db"], **view_numbers(solo, r)))
    return dict(
        span_rev_s=span,
        view=view,
        **view_numbers(view),
        floor_at_db={f"{h:.0f}": float(np.interp(h, f, floor)) for h in RIG_FLOOR_AT_HZ},
        between_hz=mid_hz,
        between_over_floor_db=float(np.interp(mid_hz, f, view["full_db"] - floor)),
        rotors=rotors,
    )


def solo_fit(fit: dict[str, Any], rotor: int) -> dict[str, Any]:
    """A deep copy of a v2/v3 fit with every comb but rotor ``rotor``'s switched
    off (``profile_db`` -> -300 dB, the param view's own switch)."""
    out = copy.deepcopy(fit)
    prof = np.atleast_2d(np.asarray(out["params"]["profile"]["profile_db"], dtype=np.float64))
    alone = np.arange(prof.shape[0])[:, None] == rotor
    out["params"]["profile"]["profile_db"] = np.where(alone, prof, -300.0).tolist()
    return out


def rig_panel(nl: Any, fit: dict[str, Any], rps: float) -> dict[str, Any]:
    """:func:`panel` of ``noise_lab.param_view`` at ``rps``, of the whole rig
    and of each rotor alone (:func:`solo_fit`)."""
    v = nl.param_view(fit, rps)
    n_rotors = np.atleast_2d(v["gamma_hz"]).shape[0]
    solo = [nl.param_view(solo_fit(fit, r), rps)["full_db"] for r in range(n_rotors)]
    return panel(v, solo, rig_span(fit))


def v3_rig_panels(nl: Any, tag: str) -> dict[str, Any]:
    """One Figs G-I3 row per pooled v3 fit of round ``tag``."""
    out: dict[str, Any] = {}
    for key, path in v3_fits(tag).items():
        fit = load(path)
        rps = rig_rps(fit)
        out[key] = dict(path=str(path), rps=rps, **rig_panel(nl, fit, rps))
        print(f"v3 {tag} {key}: {rps:.1f} rev/s", flush=True)
    return out


def legacy_quiet(params: Any, rps: float) -> Any:
    """A legacy parameter set as the stream renders it in a flight that hovers
    at ``rps`` (``StochasticNoisePool.render``: ``amp_rps_ref`` becomes the
    hover, the widths and the shaft jitter scale by hover / ``amp_rps_ref``),
    with every wander process at its mean (the harmonic, floor-level,
    floor-tilt and per-mic modulation processes at 0 dB, as the v3 view holds
    its latents at zero) and no label error, so every shaft turns at ``rps``.
    The shaft jitter and the phase diffusion stay: they ARE the line width."""
    size = float(rps) / float(params.amp_rps_ref)
    return params.with_(
        amp_rps_ref=float(rps),
        gamma0=np.asarray(params.gamma0, dtype=np.float64) * size,
        gamma_slope=np.asarray(params.gamma_slope, dtype=np.float64) * size,
        shaft_jitter_rps=np.asarray(params.shaft_jitter_rps, dtype=np.float64) * size,
        harm_gp_std_db=0.0,
        floor_gp_std_db=0.0,
        floor_tilt_gp_std=0.0,
        umod_std_db=0.0,
        shaft_offset_rps=0.0,
    )


def legacy_mean_periodogram(params: Any, rps: float, *, line_mode: str, n_fft: int) -> np.ndarray:
    """``(F,)`` mic- and frame-mean periodogram (``stochastic_fit.data``'s, the
    fits' own units) of a :data:`LEGACY_VIEW_S` render at a constant ``rps``,
    ``normalize_rms=None`` (the rig's own level), :data:`LEGACY_VIEW_SEED`."""
    from data_processing import stochastic_rotor_noise as srn
    from experiments.stochastic_fit import data as SD

    track = np.full((int(params.n_rotors), int(round(LEGACY_VIEW_S * SD.SR))), float(rps))
    audio, _ = srn.synthesize(
        params,
        track,
        rng=np.random.default_rng(LEGACY_VIEW_SEED),
        n_mics=LEGACY_VIEW_MICS,
        n_fft=int(n_fft),
        normalize_rms=None,
        line_mode=line_mode,
    )
    pg = SD.periodogram(SD.Clip("legacy_view", "synthetic", audio, track))
    return pg.power.astype(np.float64).mean(axis=(0, 1))


def legacy_view(src: Any, rps: float) -> tuple[dict[str, Any], list[np.ndarray]]:
    """The parameter view of one legacy source (``noise_lab.LegacyFit`` /
    ``LegacyBank``) at ``rps``, in ``param_view``'s keys, and one expected
    periodogram per rotor with only that rotor's comb on.

    The legacy model has no expected-periodogram function, so ``full_db`` is
    :func:`legacy_mean_periodogram` of :func:`legacy_quiet` and ``floor_db`` the
    same with every ``profile_db`` at -300 dB (one seed, so the two share the
    floor's noise draw). ``gamma_hz`` is the Lorentzian half width
    ``max(gamma0 + gamma_slope k, gamma_min_bins df)``: the width of the
    incoherent share ``1 - w_k``; the coherent share ``w_k = exp(-(k /
    coherence_k_half)^2)`` is a tone on the jittering shaft
    (``stochastic_rotor_noise.synthesize``, ``line_mode="fm"``). ``ctrl_db``:
    ``floor_mean_db + floor_shape_db`` (shape + tilt) at the control points,
    plus the floor's speed factor at the hover (1 + ``floor_static_rel``) and
    the mic-mean floor gain (``fixed_mic_floor_db`` + ``fixed_mic_gain_all_db``).
    """
    from data_processing import stochastic_rotor_noise as srn

    p = legacy_quiet(src.params, rps)
    prof = np.atleast_2d(np.asarray(p.profile_db, dtype=np.float64))
    n_rotors, k_max = prof.shape

    def mean_db(profile: np.ndarray) -> np.ndarray:
        return _db(
            legacy_mean_periodogram(
                p.with_(profile_db=profile), rps, line_mode=src.line_mode, n_fft=src.n_fft
            )
        )

    full = mean_db(prof)
    floor = mean_db(np.full_like(prof, -300.0))
    solo = [
        mean_db(np.where(np.arange(n_rotors)[:, None] == r, prof, -300.0)) for r in range(n_rotors)
    ]
    sr = float(p.sample_rate)
    f = np.fft.rfftfreq(int(src.n_fft), 1.0 / sr)
    k = np.arange(1, k_max + 1, dtype=np.float64)
    freq = k * float(rps)
    gamma = np.maximum(
        np.asarray(p.gamma0, dtype=np.float64)[:, None]
        + np.asarray(p.gamma_slope, dtype=np.float64)[:, None] * k[None, :],
        float(p.gamma_min_bins) * sr / float(src.n_fft),
    )
    mic_db = np.zeros(LEGACY_VIEW_MICS)
    for fixed in (p.fixed_mic_floor_db, p.fixed_mic_gain_all_db):
        if fixed is not None:
            mic_db = mic_db + np.asarray(fixed, dtype=np.float64)
    gain = (1.0 + max(float(p.floor_static_rel), 0.0)) * float(np.mean(10.0 ** (mic_db / 10.0)))
    ctrl_hz = np.asarray(p.floor_ctrl_hz, dtype=np.float64)
    half = float(p.coherence_k_half)
    view = dict(
        rps=float(rps),
        fmax=sr / 2.0,
        v3=False,
        wind=False,
        legacy=True,
        f=f,
        full_db=full,
        floor_db=floor,
        ctrl_hz=ctrl_hz,
        ctrl_db=float(p.floor_mean_db) + srn.floor_shape_db(p, ctrl_hz) + 10.0 * np.log10(gain),
        gamma_hz=gamma,
        k=k,
        freq_hz=freq,
        line_db=np.interp(freq, f, full),
        keep=freq < sr / 2.0,
        coherent_share=np.exp(-((k / half) ** 2)) if half > 0.0 else np.zeros_like(k),
    )
    return view, solo


def legacy_rig_panels(nl: Any) -> dict[str, dict[str, Any]]:
    """Figs I2-I3: :func:`panel` of :func:`legacy_view` at :data:`RIG_RPS` for
    the anchors of :data:`LEGACY_FITS` and the entries of :data:`LEGACY_BANK_ENTRIES`."""
    sources = {
        "legacy_fit": {key: nl.LegacyFit(name) for key, name in LEGACY_FITS.items()},
        "legacy_bank": {f"e{i}": nl.LegacyBank(LEGACY_BANK, i) for i in LEGACY_BANK_ENTRIES},
    }
    out: dict[str, dict[str, Any]] = {}
    for group, srcs in sources.items():
        out[group] = {}
        for key, src in srcs.items():
            view, solo = legacy_view(src, RIG_RPS)
            out[group][key] = dict(
                name=src.name,
                entry=src.entry,
                line_mode=src.line_mode,
                rps=RIG_RPS,
                **panel(view, solo, None),
            )
            print(f"{group} {key}: {src.entry}", flush=True)
    return out


def rig_views() -> dict[str, Any]:
    """Figs G-I3: the parameter view of four v3 prior draws, the v3 fits of
    :data:`V3_ROUND` and of round 2, the v2 fits and the legacy rigs, every
    rotor at one speed, latents (legacy: wander) at zero."""
    nl = _noise_lab()
    checks = _sibling("noise_v3_checks")
    prior_fit = v3_fits(V3_ROUND)["dregon"]
    fit_d = load(prior_fit)
    out: dict[str, Any] = dict(
        prior_fit=str(prior_fit),
        over_bar_db=RIG_OVER_DB,
        floor_at_hz=RIG_FLOOR_AT_HZ,
        fmin_hz=RIG_FMIN_HZ,
        legacy_render=dict(
            seconds=LEGACY_VIEW_S,
            n_mics=LEGACY_VIEW_MICS,
            seed=LEGACY_VIEW_SEED,
            rule="mean periodogram of a constant-speed render, wander at 0 dB, no label error",
        ),
        prior={},
        **{f"v3_{V3_ROUND}": v3_rig_panels(nl, V3_ROUND)},
        v3_r2=v3_rig_panels(nl, "r2"),
        v2={},
    )
    for seed in PRIOR_SEEDS:
        draw, stats = checks.prior_draw(fit_d, np.random.default_rng(seed))
        rps = rig_rps(fit_d)
        out["prior"][f"s{seed}"] = dict(rps=rps, stats=stats, **rig_panel(nl, draw, rps))
        print(f"prior draw seed {seed}: done", flush=True)
    v2_paths = {
        "dregon": nl.FIT_PATHS["dregon"]["single"],
        "cruise": nl.FIT_PATHS["michaels"]["cruise"],
        "standby": nl.FIT_PATHS["michaels"]["standby"],
    }
    for key, path in v2_paths.items():
        fit = load(path)
        rps = rig_rps(fit)
        out["v2"][key] = dict(path=path, rps=rps, **rig_panel(nl, fit, rps))
        print(f"v2 {key}: {rps:.1f} rev/s", flush=True)
    out.update(legacy_rig_panels(nl))
    return out


def listen_key(tag: str) -> str:
    """The § Listen clip key of v3 round ``tag`` (round 2 is the original ``"v3"``)."""
    return "v3" if tag == "r2" else f"v3_{tag}"


def listen_label(key: str) -> str:
    if key in LISTEN_LABELS:
        return LISTEN_LABELS[key]
    return f"v3 round-{key.removeprefix('v3_r')} fit"


def listen_source(nl: Any, rig: str, key: str) -> Any:
    """The noise source of clip ``key`` for listen rig ``rig``: ``"legacy"``
    (:data:`LISTEN_LEGACY`), ``"v2"``, or a v3 round's :func:`listen_key`."""
    if key == "legacy":
        return nl.LegacyFit(LISTEN_LEGACY[rig], dynamics=LISTEN_LEGACY_DYNAMICS)
    if key == "v2":
        return nl.V2Fit(rig)
    return nl.V3Fit(rig, round="r2" if key == "v3" else key.removeprefix("v3_"))


def listen(keys: tuple[str, ...] = ("v2", listen_key(V3_ROUND))) -> dict[str, Any]:
    """Figs J-K: per rig, the real window of :data:`LISTEN_WINDOWS` and the
    sources ``keys`` (:func:`listen_source`) rendered on its rotor track, all
    under the notebook's level rule, as 16-bit WAVs, with
    ``noise_lab.line_stats`` (k = 1 to 8) and the band LTAS inside
    :data:`LISTEN_LTAS_HZ`."""
    import soundfile as sf

    from experiments.stochastic_fit import accept_stats
    from experiments.stochastic_fit import revised_eval as RE

    nl = _noise_lab()
    lo, hi = LISTEN_LTAS_HZ
    bands = [i for i, (b0, b1) in enumerate(accept_stats.BANDS) if b0 >= lo and b1 <= hi]
    AUDIO.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = dict(
        seconds=LISTEN_S,
        seed=LISTEN_SEED,
        level=list(LISTEN_LEVEL),
        mic=0,
        sr=nl.SR,
        dyn_range_db=LISTEN_DYN_DB,
        ltas_bands_hz=[list(accept_stats.BANDS[i]) for i in bands],
        labels={k: listen_label(k) for k in ("real", *keys)},
        rigs={},
    )
    for rig, window in LISTEN_WINDOWS.items():
        traj = nl.trajectory("real", duration_s=LISTEN_S, **window)
        sources = {key: listen_source(nl, rig, key) for key in keys}
        renders = nl.render_all(
            list(sources.values()), traj, seed=LISTEN_SEED, n_mics=1, level=LISTEN_LEVEL
        )
        clips = {
            "real": nl.real_clip(traj, n_mics=1, level=LISTEN_LEVEL),
            **{key: renders[src.name] for key, src in sources.items()},
        }
        stats = nl.line_stats(clips)
        m = dict(traj["meta"].items())
        row: dict[str, Any] = dict(
            **window,
            seconds=LISTEN_S,
            labels=m["labels"],
            slice_label=m["slice_label"],
            rps_min=m["rps_min"],
            rps_max=m["rps_max"],
            rps_mean=m["rps_mean"],
            spans_rev_s={key: src.span for key, src in sources.items()},
            rps=np.round(np.asarray(traj["rps"].data, dtype=np.float64), 2),
            clips={},
        )
        ltas: dict[str, np.ndarray] = {}
        for (key, clip), (_, st) in zip(clips.items(), stats.iterrows(), strict=True):
            x = np.asarray(clip["audio"].data, dtype=np.float64)[0]
            if np.max(np.abs(x)) >= 1.0:
                raise SystemExit(f"{rig} {key}: peak {np.max(np.abs(x)):.3f} would clip a WAV")
            wav = AUDIO / f"{rig}_{key}.wav"
            sf.write(wav, x, nl.SR, subtype="PCM_16")
            ltas[key] = RE.absolute_ltas_bands(x)[bands]
            cm = dict(clip["meta"].items())
            row["clips"][key] = dict(
                source=cm["source"],
                entry=cm["entry"],
                wav=str(wav.relative_to(OUT)),
                rms=cm["rms"],
                peak=cm["peak"],
                level_gain=cm["level_gain"],
                line_over_floor_db=[float(st[f"k={k}"]) for k in range(1, 9)],
                ltas_db=ltas[key],
            )
        for key in sources:
            dev = ltas[key] - ltas["real"]
            row["clips"][key]["ltas_dev_db"] = dev
            row["clips"][key]["ltas_mean_abs_db"] = float(np.mean(np.abs(dev)))
        out["rigs"][rig] = row
        print(
            f"listen {rig}: {m['slice_label']}, {m['rps_min']:.1f}-{m['rps_max']:.1f} rev/s; "
            + ", ".join(
                f"{key} k1 {c['line_over_floor_db'][0]:+.2f} k2 {c['line_over_floor_db'][1]:+.2f}"
                + (f" LTAS {c['ltas_mean_abs_db']:.2f}" if "ltas_mean_abs_db" in c else "")
                for key, c in row["clips"].items()
            ),
            flush=True,
        )
    return out


def compute() -> dict[str, Any]:
    import torch

    from experiments.noise_model import model as MD

    torch.set_num_threads(4)
    fits = {k: load(p) for k, p in FITS.items()}
    v2 = load(V2_FIT)
    out: dict[str, Any] = dict(
        sources=dict(
            fits={k: str(p) for k, p in FITS.items()},
            restarts={k: str(p) for k, p in RESTART_DIRS.items()},
            v2_fit=str(V2_FIT),
            wander_dir=str(WANDER_DIR),
            channel_gains=CHANNEL_GAINS,
            selected_seed={k: f["restarts"]["selected_seed"] for k, f in fits.items()},
            rounds_run={k: f["optimiser"]["rounds_run"] for k, f in fits.items()},
            whittle_nats={k: f["objective"]["whittle_nats"] for k, f in fits.items()},
            floor_shape_sd_db={
                k: f["params"]["floor"]["floor_shape_sd_db"] for k, f in fits.items()
            },
            floor_shape_z_norm={
                k: float(np.linalg.norm(f["params"]["floor"]["floor_shape_z"]))
                for k, f in fits.items()
            },
        )
    )
    out["moves"] = moves()
    out["ladder"] = ladder()
    out["static_shares"] = static_shares()

    print("building the DREGON pool (v3 build)", flush=True)
    batch = pool_batch(fits["r2"], v3=True)
    ctrl_hz = np.asarray(batch.grid.floor.ctrl_hz, dtype=np.float64)
    out["tracks"] = tracks(fits, ctrl_hz)
    df = float(batch.grid.diagnostics["sr"]) / float(batch.grid.diagnostics["n_fft"])
    data = batch.power.numpy()
    carrier = frame_carrier(batch)
    out["pool"] = dict(
        n_frames=int(data.shape[1]),
        n_mics=int(data.shape[0]),
        n_cells=int(batch.n_cells),
        members=list(batch.members),
        df_hz=df,
        carrier_mean_rev_s=float(carrier.mean()),
    )

    ev: dict[str, Any] = {}
    models: dict[str, dict[str, np.ndarray]] = {}
    for rnd, fit in fits.items():
        print(f"{rnd}: evaluating", flush=True)
        params = MD.params_from_dict(fit["params"])
        lat = latent_arrays(fit)
        m0, w0 = evaluate(batch, params, None)
        m1, w1 = evaluate(batch, params, window_latents(lat))
        models[rnd] = dict(m0=m0, m1=m1)
        ev[rnd] = dict(
            whittle_zero_latents_nats=w0,
            whittle_fitted_latents_nats=w1,
            whittle_recorded_nats=float(fit["objective"]["whittle_nats"]),
            whittle_round0_nats=float(fit["optimiser"]["history"][0]["whittle_nats"]),
            rig_lbfgs_iters=[h.get("rig_lbfgs_iters") for h in fit["optimiser"]["history"]],
            latent_evals=[h.get("latent_evals") for h in fit["optimiser"]["history"]],
        )
        if rnd == "r2":
            f0, _ = evaluate(batch, floor_only(params), None)
            f1, _ = evaluate(batch, floor_only(params), window_latents(lat))
            models[rnd].update(f0=f0, f1=f1)
            print("r2: ridge test", flush=True)
            ev["ridge_r2"] = ridge_test(batch, params, fit, lat, w1)
    out["whittle"] = ev

    # Fig D: the window whose u_j tracks are widest in round 2
    lat2 = latent_arrays(fits["r2"])
    uj_rms = np.sqrt((lat2["uj"] ** 2).mean(axis=(1, 2)))
    w_star = int(np.argmax(uj_rms))
    print(f"Fig D window {w_star} (u_j rms per window {np.round(uj_rms, 2)})", flush=True)
    m2 = models["r2"]
    out["window"] = window_spectra(
        batch, w_star, data, m2["m0"], m2["m1"], m2["f0"], m2["f1"], carrier, df
    )
    out["window"]["uj_rms_per_window_db"] = uj_rms

    # Fig F
    out["mics"] = mic_residuals(batch, data, m2["m0"], lat2, ctrl_hz, df)

    # Fig E: line over local floor per order, same estimator on every spectrum
    print("v2: evaluating on its own build", flush=True)
    batch_v2 = pool_batch(v2, v3=False)
    mv2, wv2 = evaluate(batch_v2, MD.params_from_dict(v2["params"]), None)
    carrier_v2 = frame_carrier(batch_v2)
    prof: dict[str, Any] = dict(orders=E_ORDERS, v2_whittle_nats=wv2)
    for key, arr, car in (
        ("real", data, carrier),
        ("r1", models["r1"]["m0"], carrier),
        ("r2", models["r2"]["m0"], carrier),
        ("r2_fitted_latents", models["r2"]["m1"], carrier),
        ("v2", mv2, carrier_v2),
    ):
        lf = line_floor_db(order_aligned(arr, car, df, E_ORDERS))
        prof[key] = {k: v.mean(axis=0) for k, v in lf.items()}  # rotor mean of dB
    out["profile"] = prof
    out["rigs"] = rig_views()
    out["listen"] = listen()
    return out


# ── figures ────────────────────────────────────────────────────────────────


def _save(fig: Any, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"wrote {OUT / name}")


def fig_a(d: dict[str, Any]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    # round 2 first, so round 1's hollow markers sit on top: the two coincide
    # over rounds 1-5
    for want in ("r2", "r1"):
        for key, rec in d["moves"].items():
            rnd, pool, _seed = key.split("/")
            if pool != POOL or rnd != want:
                continue
            mv = np.asarray([np.nan if v is None else v for v in rec["move"]], dtype=np.float64)
            rounds = np.arange(1, mv.size + 1)
            if rnd == "r1":
                ax.semilogy(rounds, mv, "o-", color=C_R1, ms=9, mfc="none", mew=1.6, lw=1.2)
            else:
                ax.semilogy(rounds, mv, "s-", color=C_R2, ms=4.5, lw=1.4, alpha=0.85)
    ax.axhline(TOL, color="k", ls="--", lw=1.2)
    ax.text(1.0, TOL * 0.9, "stop: 1e-4 nats per cell", ha="left", va="top", fontsize=11)
    ax.plot(
        [],
        [],
        "o-",
        color=C_R1,
        ms=9,
        mfc="none",
        mew=1.6,
        label="round 1: 4 restarts, 5 rounds allowed, none stop",
    )
    ax.plot([], [], "s-", color=C_R2, label="round 2: 4 restarts, 20 rounds allowed, all stop")
    ax.set_xlabel("alternation round (one latent step + one rig step)")
    ax.set_ylabel("Whittle move (nats per cell)")
    ax.set_xticks(range(1, 21))
    ax.set_xlim(0.5, 20.5)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right")
    ax.set_title("DREGON pool: the inner loop (rig ↔ latents, σ fixed)")
    _save(fig, "fig_a_inner_loop.png")


def fig_b(d: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8.4), sharex=True)
    steps = (
        ("mm0", "measured (mm0)", "#bbbbbb"),
        ("mm1", "update 1 (mm1)", "#f4a582"),
        ("mm2", "update 2 (mm2)", "#b2182b"),
    )
    for ax, rig in zip(axes, ("dregon", "michaels"), strict=True):
        rec = d["ladder"][rig]
        fams = rec["families"]
        x = np.arange(len(fams))
        wdt = 0.26
        for i, (tag, lab, col) in enumerate(steps):
            s = np.asarray(rec["steps"][tag]["sigma_db"], dtype=np.float64)
            bars = ax.bar(
                x + (i - 1) * wdt,
                s,
                wdt,
                color=col,
                edgecolor="k",
                lw=0.6,
                hatch="//" if tag == "mm0" else None,
                label=lab,
            )
            for bx, sv in zip(bars, s, strict=True):
                if sv == 0.0:
                    ax.text(
                        bx.get_x() + bx.get_width() / 2,
                        0.15,
                        "0",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                    )
        if rig == "dregon":
            ax.text(3.5, 1.3, "pinned at 0\n(σ measured 0)", ha="center", va="bottom", fontsize=11)
        ax.set_ylabel("prior σ (dB)")
        ax.set_title(rec["label"], loc="left")
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(0, 8.2)
    axes[0].legend(loc="upper left", ncol=3)
    axes[1].set_xticks(np.arange(len(d["ladder"]["dregon"]["families"])))
    axes[1].set_xticklabels(d["ladder"]["dregon"]["families"])
    axes[1].set_xlabel("latent family")
    fig.tight_layout()
    _save(fig, "fig_b_ladder.png")


def _window_axes(ax: Any, n_w: int, n_b: int, block_s: float, top: bool) -> None:
    span = n_b * block_s
    for w in range(1, n_w):
        ax.axvline(w * span, color="k", lw=0.8, alpha=0.5)
    if top:
        for w in range(n_w):
            ax.text(
                (w + 0.5) * span,
                1.02,
                WIN_LABELS[w],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=11,
            )
    ax.set_xlim(0, n_w * span)
    ax.grid(alpha=0.25)


def fig_c(d: dict[str, Any]) -> None:
    tr = d["tracks"]
    bs = float(tr["block_s"])
    r1, r2 = tr["r1"], tr["r2"]
    dd1, dd2 = np.asarray(r1["d"]), np.asarray(r2["d"])
    n_w, n_r, n_b = dd2.shape
    t = (np.arange(n_w * n_b) + 0.5) * bs

    def flat(a: np.ndarray) -> np.ndarray:  # (W, ..., B) -> (..., W B)
        a = np.moveaxis(np.asarray(a, dtype=np.float64), 0, -2)
        return a.reshape(a.shape[:-2] + (-1,))

    fig, axes = plt.subplots(4, 1, figsize=(12, 13.5), sharex=True)

    def bands(ax: Any, fam: str) -> None:
        for rec, c in ((r1, C_R1), (r2, C_R2)):
            s = rec["prior_sigma_db"][fam]
            ax.axhspan(-s, s, color=c, alpha=0.10, lw=0)
            ax.axhline(s, color=c, lw=0.8, ls=":")
            ax.axhline(-s, color=c, lw=0.8, ls=":")

    # d
    ax = axes[0]
    bands(ax, "d")
    rot_ls = ("-", "--", "-.", ":")
    for r in range(n_r):
        ax.plot(t, flat(dd1)[r], color=C_R1, lw=1.1, ls=rot_ls[r], alpha=0.8)
        ax.plot(t, flat(dd2)[r], color=C_R2, lw=1.5, ls=rot_ls[r])
    ax.set_ylabel("d (dB)")
    ax.set_title("d: per-rotor level (line style = rotor 1–4)", loc="left", pad=22)
    _window_axes(ax, n_w, n_b, bs, top=True)
    # u
    ax = axes[1]
    bands(ax, "u")
    ax.plot(t, flat(r1["u"]), color=C_R1, lw=1.5)
    ax.plot(t, flat(r2["u"]), color=C_R2, lw=1.8)
    ax.set_ylabel("u (dB)")
    ax.set_title("u: floor level, all frequencies", loc="left")
    _window_axes(ax, n_w, n_b, bs, top=False)
    # u_j
    ax = axes[2]
    bands(ax, "u_j")
    ctrl = np.asarray(tr["ctrl_hz"])
    kn_ls = ("-", "--", "-.")
    for i, j in enumerate(tr["uj_knots"]):
        ax.plot(t, flat(r1["uj"])[i], color=C_R1, lw=1.3, ls=kn_ls[i])
        ax.plot(t, flat(r2["uj"])[i], color=C_R2, lw=1.8, ls=kn_ls[i], label=f"{ctrl[j]:.0f} Hz")
    ax.set_ylabel("u_j (dB)")
    ax.set_title("u_j: floor colour at three control points", loc="left")
    leg = ax.legend(loc="lower left", ncol=3, title="control point (line style)")
    for h in leg.legend_handles:
        h.set_color("k")
    _window_axes(ax, n_w, n_b, bs, top=False)
    # v k9-24
    ax = axes[3]
    bands(ax, "v k9–24")
    for rec, c in ((r1, C_R1), (r2, C_R2)):
        q = np.asarray(rec["v924_q"], dtype=np.float64)  # (3, W, B)
        qf = np.stack([flat(qq) for qq in q])
        ax.fill_between(t, qf[0], qf[2], color=c, alpha=0.25, lw=0)
        ax.plot(t, qf[1], color=c, lw=1.6)
    ax.set_ylabel("v (dB)")
    ax.set_title("v, orders 9–24: median and 10–90 % over the 64 lines", loc="left")
    ax.set_xlabel("time (s), five 8 s windows side by side")
    _window_axes(ax, n_w, n_b, bs, top=False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Line2D([], [], color=C_R1, lw=1.5, label="round 1 track"),
        Line2D([], [], color=C_R2, lw=1.8, label="round 2 track"),
        Patch(color=C_R1, alpha=0.25, label="round 1 prior ±σ"),
        Patch(color=C_R2, alpha=0.25, label="round 2 prior ±σ"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    _save(fig, "fig_c_tracks.png")


def fig_d(d: dict[str, Any]) -> None:
    w = d["window"]
    f = np.asarray(w["freqs_hz"])
    fig, axes = plt.subplots(
        3, 1, figsize=(12, 13), sharex=True, gridspec_kw=dict(height_ratios=(2.2, 1.3, 1.0))
    )
    ax = axes[0]
    for m, row in enumerate(np.asarray(w["data_db"])):
        ax.plot(
            f, row, color="#999999", lw=0.5, alpha=0.6, label="data, each mic" if m == 0 else None
        )
    ax.plot(f, w["data_mean_db"], color=C_DATA, lw=1.0, label="data, mic mean")
    ax.plot(f, w["model_zero_db"], color="#2166ac", lw=1.2, label="model, latents at zero")
    ax.plot(f, w["model_fit_db"], color=C_R2, lw=1.2, label="model, fitted latents")
    ax.fill_between(
        f,
        w["model_zero_db"],
        w["model_fit_db"],
        color=C_R2,
        alpha=0.25,
        lw=0,
        label="what the latents add or remove",
    )
    ax.plot(
        f, w["floor_zero_db"], color="#2166ac", lw=1.3, ls="--", label="floor part, latents at zero"
    )
    ax.plot(f, w["floor_fit_db"], color=C_R2, lw=1.3, ls="--", label="floor part, fitted latents")
    ax.set_xscale("log")
    ax.set_ylabel("power (dB, fit units)")
    # low enough to show the whole dip of the fitted floor part
    lo = (
        min(float(np.percentile(np.asarray(w["data_db"]), 0.5)), float(np.min(w["floor_fit_db"])))
        - 4.0
    )
    hi = float(np.max(w["data_mean_db"])) + 4.0
    ax.set_ylim(lo, hi)
    j = int(np.argmin(w["floor_fit_db"]))
    ax.annotate(
        f"fitted floor part: {w['floor_fit_db'][j]:.0f} dB at {f[j]:.0f} Hz",
        xy=(f[j], w["floor_fit_db"][j]),
        xytext=(f[j] / 9.0, w["floor_fit_db"][j] + 3.0),
        arrowprops=dict(arrowstyle="->", color=C_R2),
        color=C_R2,
        fontsize=11,
    )
    ax.legend(loc="upper right", ncol=2)
    ax.set_title(
        f"DREGON round 2, window {w['window'] + 1} ({WIN_LABELS[w['window']]}), 8 s frame mean",
        loc="left",
    )
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ratio = np.asarray(w["ratio_db"])
    ax.plot(f, ratio, color="#999999", lw=0.6, label="data / model at zero latents, per bin")
    ax.plot(f, w["ratio_smooth_db"], color=C_DATA, lw=2.0, label="same, 1/3-octave mean")
    ax.plot(f, w["latent_fill_db"], color=C_R2, lw=1.6, label="fitted / zero-latent model")
    line = np.asarray(w["line_bins"], dtype=bool)
    ax.plot(f[line], ratio[line], "o", color="#ff7f00", ms=2.2, label="bins on lines k ≤ 24")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("ratio (dB)")
    ax.legend(loc="lower left", ncol=2)
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(np.percentile(ratio, 0.5) - 3, np.percentile(ratio, 99.5) + 3)

    ax = axes[2]
    pm = np.asarray(w["per_mic_minus_mean_db"])
    lf = np.log2(f)
    for m, row in enumerate(pm):
        sm = np.array([row[np.abs(lf - x) <= 1.0 / 6.0].mean() for x in lf])
        ax.plot(f, sm, lw=1.2, label=f"mic {m + 1}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mic − mic mean (dB)")
    ax.set_xlabel("frequency (Hz)")
    ax.legend(loc="upper left", ncol=8, fontsize=11, columnspacing=0.8, handlelength=1.2)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("per-mic ratio minus the mic-mean ratio, 1/3-octave mean", loc="left")
    fig.tight_layout()
    _save(fig, "fig_d_window.png")


def fig_e(d: dict[str, Any]) -> None:
    p = d["profile"]
    k = np.asarray(p["orders"])
    series = (
        ("real", "real data", C_DATA, "o-", 2.0),
        ("v2", "v2 fit", C_V2, "^-", 1.4),
        ("r1", "v3 round 1, latents at zero", C_R1, "s-", 1.4),
        ("r2", "v3 round 2, latents at zero", C_R2, "D-", 1.6),
        ("r2_fitted_latents", "v3 round 2, fitted latents", C_R2, "D--", 1.2),
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    for key, lab, c, mk, lw in series:
        rec = p[key]
        axes[0].plot(
            k,
            rec["over_db"],
            mk,
            color=c,
            lw=lw,
            ms=4.5,
            label=lab,
            mfc="none" if "fitted" in key else c,
        )
        axes[1].plot(
            k, rec["line_db"], mk, color=c, lw=lw, ms=4.5, mfc="none" if "fitted" in key else c
        )
        axes[2].plot(
            k, rec["floor_db"], mk, color=c, lw=lw, ms=4.5, mfc="none" if "fitted" in key else c
        )
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("line over local floor (dB)")
    axes[0].set_title("line over floor", loc="left")
    axes[1].set_ylabel("power (dB, fit units)")
    axes[1].set_title("line level", loc="left")
    axes[2].set_ylabel("power (dB, fit units)")
    axes[2].set_title("local floor level", loc="left")
    for ax in axes:
        ax.set_xlabel("order k")
        ax.set_xticks([1, 2, 4, 8, 12, 16, 20, 24])
        ax.grid(alpha=0.3)
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.02)
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, "fig_e_profile.png")


def fig_f(d: dict[str, Any]) -> None:
    mc = d["mics"]
    x = np.asarray(mc["x_db"], dtype=np.float64)  # (W, B, J)
    y = np.asarray(mc["y_db"], dtype=np.float64)  # (W, B, J, M)
    ybar = np.nanmean(y, axis=-1)
    xk = x - np.nanmean(x, axis=(0, 1), keepdims=True)
    ybk = ybar - np.nanmean(ybar, axis=(0, 1), keepdims=True)
    knots = mc["knots_hz"]
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw=dict(width_ratios=(1.15, 1.0, 0.9)))
    for i, fk in enumerate(knots):
        c = cmap(i / max(1, len(knots) - 1))
        axes[0].plot(
            x[..., i].ravel(),
            ybar[..., i].ravel(),
            "o",
            color=c,
            ms=3.5,
            alpha=0.7,
            label=f"{fk:.0f} Hz",
        )
        axes[1].plot(xk[..., i].ravel(), ybk[..., i].ravel(), "o", color=c, ms=3.5, alpha=0.7)
    lim = (float(np.nanmin(x)) - 2, float(np.nanmax(x)) + 2)
    axes[0].plot(lim, lim, "k--", lw=1, label="y = x")
    axes[0].set_xlabel("floor latent u + u_j (dB)")
    axes[0].set_ylabel("data − zero-latent model, mic mean (dB)")
    axes[0].set_title(f"(a) as fitted: r = {mc['r_common']:.2f}", loc="left")
    axes[0].legend(title="control point", loc="lower right", fontsize=11, ncol=2)
    axes[0].set_ylim(-12, 12)
    limk = (float(np.nanmin(xk)) - 1, float(np.nanmax(xk)) + 1)
    axes[1].plot(limk, limk, "k--", lw=1)
    axes[1].set_xlabel("u + u_j minus its mean per control point (dB)")
    axes[1].set_ylabel("mic-mean residual minus its mean (dB)")
    axes[1].set_title(f"(b) static part removed: r = {mc['r_common_within_knot']:.2f}", loc="left")
    mics = np.arange(1, y.shape[-1] + 1)
    wdt = 0.38
    axes[2].bar(
        mics - wdt / 2,
        mc["r_per_mic_within_knot"],
        wdt,
        color="#4393c3",
        label="each mic's residual",
    )
    axes[2].bar(
        mics + wdt / 2,
        mc["r_mic_part_per_mic_within_knot"],
        wdt,
        color="#f4a582",
        label="each mic − mic mean",
    )
    axes[2].axhline(0, color="k", lw=0.8)
    axes[2].set_xticks(mics)
    axes[2].set_xlabel("mic")
    axes[2].set_ylabel("r with u + u_j (static part removed)")
    axes[2].set_ylim(-0.6, 0.8)
    axes[2].set_title("(c) per mic", loc="left")
    axes[2].legend(loc="upper left", fontsize=11)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig_f_mics.png")


def _rig_rows(d: dict[str, Any]) -> list[dict[str, Any]]:
    """Every parameter-view row of ``d["rigs"]`` (prior draws, v3 rounds, v2)."""
    return [
        row
        for group in d["rigs"].values()
        if isinstance(group, dict)
        for row in group.values()
        if isinstance(row, dict) and "view" in row
    ]


def rig_ylim(d: dict[str, Any]) -> tuple[float, float]:
    """One y range for Figs G-I3: every panel's floor and expected periodogram
    between :data:`RIG_FMIN_HZ` and :data:`F_MAX_HZ` (the anti-alias roll-off
    at Nyquist left out), to the 10 dB."""
    lo, hi = np.inf, -np.inf
    for row in _rig_rows(d):
        v = row["view"]
        f = np.asarray(v["f"], dtype=np.float64)
        band = (f >= RIG_FMIN_HZ) & (f <= F_MAX_HZ)
        lo = min(lo, float(np.nanmin(np.asarray(v["floor_db"], dtype=np.float64)[band])))
        hi = max(hi, float(np.nanmax(np.asarray(v["full_db"], dtype=np.float64)[band])))
    return 10.0 * math.floor(lo / 10.0 - 0.5), 10.0 * math.ceil(hi / 10.0 + 0.5)


def fig_rigs(d: dict[str, Any], rows: list[tuple[dict[str, Any], str]], stem: str) -> None:
    """One figure per entry of :data:`ROTOR_VIEWS`, one row per rig:
    ``noise_lab.draw_param_view`` with every rotor's comb and gamma caps
    (``_all``; every rotor runs at one speed, so the stems are the rig's) or one
    rotor's comb and caps alone (``_rotorN``)."""
    nl = _noise_lab()
    ylim = rig_ylim(d)
    for key in ROTOR_VIEWS:
        r = None if key == "all" else int(key.removeprefix("rotor")) - 1
        fig, axes = plt.subplots(len(rows), 1, figsize=(11, 2.55 * len(rows) + 0.9), sharex=True)
        for ax, (row, label) in zip(np.atleast_1d(axes), rows, strict=True):
            if r is None:
                view, nums, who = row["view"], row, "all rotors"
            else:
                nums = row["rotors"][r]
                view = {**row["view"], "full_db": nums["full_db"], "line_db": nums["line_db"]}
                who = f"rotor {r + 1} alone"
            nl.draw_param_view(ax, view, r, spectrum=True)
            ax.set_xscale("log")
            ax.set_xlim(RIG_FMIN_HZ, float(view["fmax"]))
            ax.set_ylim(*ylim)
            ax.set_ylabel("dB (render units)" if view.get("legacy") else "dB (fit units)")
            ax.set_title(f"{label}, {row['rps']:.1f} rev/s", loc="left")
            ax.set_title(
                f"k=1 {nums['over_k1_db']:+.1f} dB, k=2 {nums['over_k2_db']:+.1f} dB, "
                f"{nums['n_orders_over']}/{nums['n_orders_shown']} orders > {RIG_OVER_DB:g} dB",
                loc="right",
            )
            ax.grid(alpha=0.3, which="both")
        last = np.atleast_1d(axes)[-1]
        ticks = [20, 50, 100, 200, 500, 1000, 2000, 4000, 8000]
        last.set_xticks(ticks)
        last.set_xticklabels([f"{t / 1000:g}k" if t >= 1000 else str(t) for t in ticks])
        last.set_xlabel("frequency (Hz)")
        handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
        fig.text(0.01, 0.995, who, ha="left", va="top", fontsize=15, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 1.0 - 0.75 / fig.get_figheight()))
        _save(fig, f"{stem}_{key}.png")


def fig_g(d: dict[str, Any]) -> None:
    rows = [(row, f"prior draw, seed {k[1:]}") for k, row in d["rigs"]["prior"].items()]
    fig_rigs(d, rows, "fig_g_prior_rigs")


def fig_h(d: dict[str, Any]) -> None:
    rows = [
        (row, f"v3 round {V3_ROUND[1:]}, {RIG_NAMES[k]}")
        for k, row in d["rigs"][f"v3_{V3_ROUND}"].items()
    ]
    fig_rigs(d, rows, "fig_h_v3_rigs")


def fig_h2(d: dict[str, Any]) -> None:
    """§ 8's contrast: the round-2 fits the runaway came from."""
    rows = [(row, f"v3 round 2, {RIG_NAMES[k]}") for k, row in d["rigs"]["v3_r2"].items()]
    fig_rigs(d, rows, "fig_h2_v3_r2_rigs")


def fig_i(d: dict[str, Any]) -> None:
    rows = [(row, f"v2, {RIG_NAMES[k]}") for k, row in d["rigs"]["v2"].items()]
    fig_rigs(d, rows, "fig_i_v2_rigs")


def fig_i2(d: dict[str, Any]) -> None:
    rows = [
        (row, f"legacy anchor, {LEGACY_NAMES[k]}") for k, row in d["rigs"]["legacy_fit"].items()
    ]
    fig_rigs(d, rows, "fig_i2_legacy_fits")


def fig_i3(d: dict[str, Any]) -> None:
    rows = [
        (row, f"legacy {LEGACY_BANK} bank, entry {k[1:]}")
        for k, row in d["rigs"]["legacy_bank"].items()
    ]
    fig_rigs(d, rows, "fig_i3_legacy_bank")


def fig_listen(d: dict[str, Any], rig: str, name: str) -> None:
    """One rig of § Listen: ``noise_lab.spectrogram_figure`` of the WAVs as
    written (real, then the sources in row order), one colour scale, the rotor track once."""
    import soundfile as sf
    import tdseries as td

    nl = _noise_lab()
    ls = d["listen"]
    row = ls["rigs"][rig]
    rps = td.uniform(
        np.asarray(row["rps"], dtype=np.float64),
        nl.RPS_PLOT_SR,
        dims=("rotor", "time"),
        t_start=0.0,
    )
    labels = ls.get("labels") or {}
    frames: dict[str, Any] = {}
    for key, clip in row["clips"].items():
        x, sr = sf.read(OUT / clip["wav"], dtype="float32")
        frames[labels.get(key) or listen_label(key)] = td.Frame(
            {"audio": td.uniform(x[None], sr, dims=("mic", "time"), t_start=0.0), "rps": rps}
        )
    fig = nl.spectrogram_figure(
        frames,
        dyn_range=float(ls["dyn_range_db"]),
        figsize=(11, 2.5 * len(frames) + 2.5),
        shared_scale=True,
        rps="once",
    )
    rps_ax = fig.axes[-1]
    lines = rps_ax.get_lines()
    rps_ax.set_title("")
    rps_ax.set_title("rotor speed, telemetry", loc="left")
    rps_ax.set_ylabel("rev/s")
    rps_ax.legend(
        lines,
        [f"rotor {r + 1}" for r in range(len(lines))],
        ncol=len(lines),
        loc="lower right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        fontsize=11,
    )
    _save(fig, name)


def plot(d: dict[str, Any]) -> None:
    fig_a(d)
    fig_b(d)
    fig_c(d)
    fig_d(d)
    fig_e(d)
    fig_f(d)
    fig_g(d)
    fig_h(d)
    fig_h2(d)
    fig_i(d)
    fig_i2(d)
    fig_i3(d)
    fig_listen(d, "dregon", "fig_j_listen_dregon.png")
    fig_listen(d, "michaels", "fig_k_listen_michaels.png")


def round_views(tag: str) -> dict[str, Any]:
    """``--round``: round ``tag`` beside round 2 — the static shares of every
    family, the parameter-view numbers of the three pooled fits (§ 8's table;
    the rig figures are Figs H / H2 of :func:`rig_views`), § Listen with the
    legacy anchor fit, v2 and both rounds rendered on the same real windows
    and seed."""
    nl = _noise_lab()
    return dict(
        tag=tag,
        static_shares=static_shares(("r2", tag)),
        rigs=dict(
            over_bar_db=RIG_OVER_DB,
            floor_at_hz=RIG_FLOOR_AT_HZ,
            fmin_hz=RIG_FMIN_HZ,
            v3_r2=v3_rig_panels(nl, "r2"),
            **{f"v3_{tag}": v3_rig_panels(nl, tag)},
        ),
        listen=listen(("legacy", "v2", listen_key("r2"), listen_key(tag))),
    )


def listen_merge(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """``--listen-add``: the clips of ``new`` (a :func:`listen` of more keys on
    the same windows) into ``old``, right after the real clip, in the row order
    a full :func:`round_views` writes. The real clip must come out the same."""
    for k in ("seconds", "seed", "level", "mic", "sr", "dyn_range_db", "ltas_bands_hz"):
        if old[k] != new[k]:
            raise SystemExit(f"--listen-add: {k} {new[k]!r} differs from {old[k]!r}")

    def first(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
        """``a`` then the keys of ``b`` not in ``a``."""
        return {**a, **{k: v for k, v in b.items() if k not in a}}

    old["labels"] = first(new["labels"], old["labels"])
    for rig, row in old["rigs"].items():
        add = new["rigs"][rig]
        if add["clips"]["real"] != row["clips"]["real"]:
            raise SystemExit(f"--listen-add: the {rig} real clip moved")
        row["clips"] = first(add["clips"], row["clips"])
        row["spans_rev_s"] = first(add["spans_rev_s"], row["spans_rev_s"])
    return old


def fig_static_shares(d: dict[str, Any], name: str) -> None:
    """Share of each family's mean square that is static (the pool mean per
    track), round 2 against round ``d["tag"]``, one panel per pool."""
    tag = str(d["tag"])
    ss = d["static_shares"]
    fams = list(next(iter(ss.values()))["families"])
    fig, axes = plt.subplots(1, len(V3_POOLS), figsize=(13, 4.6), sharey=True)
    x = np.arange(len(fams))
    for ax, (key, stem) in zip(np.atleast_1d(axes), V3_POOLS.items(), strict=True):
        for i, (rnd, colour) in enumerate((("r2", C_R2), (tag, C_R1))):
            fam = ss[f"{rnd}/{stem}"]["families"]
            share = [100.0 * float(fam[f]["static_share"]) for f in fams]
            ax.bar(x + (i - 0.5) * 0.38, share, 0.38, color=colour, label=f"round {rnd[1:]}")
        ax.set_xticks(x)
        ax.set_xticklabels(fams, rotation=35, ha="right")
        ax.set_title(RIG_NAMES[key], loc="left")
        ax.grid(alpha=0.3, axis="y")
    np.atleast_1d(axes)[0].set_ylabel("static share of the mean square (%)")
    np.atleast_1d(axes)[0].legend(loc="upper right")
    fig.tight_layout()
    _save(fig, name)


def plot_round(d: dict[str, Any]) -> None:
    tag = str(d["tag"])
    fig_static_shares(d, f"fig_l_static_shares_{tag}.png")
    fig_listen(d, "dregon", f"fig_j_listen_dregon_{tag}.png")
    fig_listen(d, "michaels", f"fig_k_listen_michaels_{tag}.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plot-only", action="store_true", help=f"redraw from {DATA}")
    mode.add_argument(
        "--rigs-only",
        action="store_true",
        help=f"recompute Figs G-I3 (no pool build) into {DATA}, then redraw",
    )
    mode.add_argument(
        "--listen-only",
        action="store_true",
        help=f"recompute Figs J-K and the WAVs (no pool build) into {DATA}, then redraw",
    )
    ap.add_argument(
        "--round",
        metavar="TAG",
        help="a later v3 round (a noise_lab.V3_FIT_DIRS key, e.g. r3a) beside round 2: static "
        "shares, parameter views and § Listen into figdata_TAG.json and fig_*_TAG.png (with "
        "--plot-only: redraw them)",
    )
    ap.add_argument(
        "--listen-add",
        metavar="KEY",
        help="with --round: render one more § Listen source (legacy, v2 or a v3 key) into "
        "figdata_TAG.json beside the clips already there, then redraw",
    )
    args = ap.parse_args(argv)
    if args.round:
        if args.rigs_only or args.listen_only:
            ap.error("--round takes --plot-only only")
        data = OUT / f"figdata_{args.round}.json"
        if args.plot_only:
            d = load(data)
        else:
            if args.listen_add:
                d = load(data)
                new = json.loads(json.dumps(_r(listen((str(args.listen_add),)), 4)))
                d["listen"] = listen_merge(d["listen"], new)
            else:
                d = _r(round_views(str(args.round)), 4)
            OUT.mkdir(parents=True, exist_ok=True)
            data.write_text(json.dumps(d, ensure_ascii=False) + "\n")
            print(f"wrote {data}")
        plot_round(d)
        return 0
    if args.listen_add:
        ap.error("--listen-add takes --round")
    if args.plot_only:
        d = load(DATA)
    else:
        if args.rigs_only:
            d = load(DATA)
            d["rigs"] = _r(rig_views(), 4)
        elif args.listen_only:
            d = load(DATA)
            d["listen"] = _r(listen(), 4)
        else:
            d = _r(compute(), 4)
        OUT.mkdir(parents=True, exist_ok=True)
        DATA.write_text(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"wrote {DATA}")
    plot(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
