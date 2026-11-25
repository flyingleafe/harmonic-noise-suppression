#!/usr/bin/env python
"""Fresh per-recording noise-model-v4 flight fit of a November M100 recording,
and its two matched held-out renders.

The model and the optimiser are the production v4 flight ones
(:mod:`experiments.noise_model.flight_v4`, ``docs/explainers/noise-model-v4.qmd``
§ Procedure B): the MAP Whittle fit of one pool with ``fit_flight`` (Adam, then
strong-Wolfe L-BFGS on a stratified subset, the all-frames polish and
fresh-memory restart passes until the production convergence test passes), the
phase pedestal free per order as in the paper's round-6 MD2 regime fits, and the
``noise-v4-fit/3`` payload rendered by :func:`data_processing.noise_model.render.render_noise`.
What this CLI replaces for the November rigs, and nothing else:

* NO MD2 GEOMETRY. Two calibrated stereo capsules of unknown position: the
  rotor-to-capsule term ``Delta G_rm`` is zero (every rotor reaches both
  channels alike). The L/R difference is an EMPIRICAL channel response per
  third-octave band, each channel's mean deviation from the two-channel mean
  over the fit windows (the ``band="transfer"`` construction of
  :func:`experiments.noise_model.fit.load_channel_gains`); the data are divided
  by it and the renderer applies it back, so the absolute level stays the
  model's.
* NO MD2 WANDER FILE. The held amplitude pedestal comes from the block wander
  measured on the fit windows only (:mod:`experiments.noise_model.wander`:
  order-tracked 0.5 s block line levels of the resolvable, rotor-dominant,
  low-noise lines, window-centred, lag-1 OU method of moments per order group),
  falling back to the pooled value per group and to the v4 bench prior medians
  only where nothing is measured (recorded in the report).
* THE ORDERS. FLY013 (bare motors): a generic shaft-order line at every order
  ``k`` (no blade count asserted: ``blades = 1``, no blade-pass prior) plus the
  motor family at the explicit electromagnetic orders 84 and 98. FLY018/022
  (2-blade M100 props): MD2's production rule, ``blades = 2`` and the motor
  family at ``k = 3 p n`` with 7 pole pairs.
* THE WORK RATE is the smallest of 32 / 64 kHz whose Nyquist clears the
  highest modelled order of the fastest carrier (a pool joining FLY013's ~35
  and ~132 rev/s plateaus needs 64 kHz for the orders the slow plateau shows).

Speeds are the native ``rps`` of the inputs, unmodified (the per-flight VK
scoring gains are not applied). One seed, 2026: the fit's restart seed and the
root of the two render seeds; no RMS normalisation anywhere.

Inputs (``--input DIR``): ``DIR/<recording>/fit.npz`` and ``holdout.npz``, each
``audio`` float32 ``(N, 2, T)`` at 16 kHz, ``rps`` float32 ``(N, 4, T)`` rev/s on
the audio grid, ``sr`` (16000) and ``start_s`` ``(N,)``; ``holdout.npz`` holds
the two examples; optional ``DIR/<recording>/selection.json`` is copied into the
report. Outputs (``--out ROOT``): ``ROOT/<recording>/`` ``fit.json`` (the
portable payload), ``reference_{1,2}.wav``, ``generated_{1,2}.wav`` (16 kHz
float WAV) and ``report.json``.

    # the real fit (GPU job)
    PYTHONPATH=src python scripts/november_cruise_fit.py --input <DIR> --recording FLY018 \\
        --out results/november_cruise_generator --device cuda --deadline-min 50 --progress 100
    # smoke (a reduced schedule is refused under results/november_cruise_generator)
    PYTHONPATH=src python scripts/november_cruise_fit.py --input <DIR> --recording FLY018 \\
        --out /tmp/ncg_smoke --device cpu --adam-steps 5 --lbfgs-iters 3 --polish-iters 2 \\
        --max-restart-passes 1 --k-cap 12
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

RECORDINGS = ("FLY013", "FLY018", "FLY022")
DATASET = "new_motor_data-frames@e766b6a9306b6687ebcc8e3a08fcb54a225b488eba962e6a9cbac15174ab4e0b"
#: The v4 flight front end's rate (2048 / 512 periodic Hann); its likelihood band
#: is 30-7900 Hz and the renderer's anti-alias passes 7900 / stops 8000 Hz.
MODEL_SR = 16000
SEED = 2026
#: ``Motor:Speed:{RFront,LFront,LBack,RBack}``, the catalog's rotor order
#: (docs/data-catalog.md § November 2026 motor recordings).
ROTOR_ORDER = ("RFront", "LFront", "LBack", "RBack")
RESULTS_ROOT = Path("results/november_cruise_generator")
WORK_RATES = (32000, 64000)
K_CAP = 130  # production MD2 flight cap; includes the bare-motor orders 84 and 98
#: DJI M100 motors: 7 pole pairs (MD2's flight_v4 constant).
POLE_PAIRS = 7
RIGS: dict[str, dict[str, Any]] = {
    "FLY013": dict(
        regime="bare-motor running (no propellers)",
        blades=1,
        motor_orders=(84, 98),
        orders_rule=(
            "phenomenological shaft-order branch: one line per rotor at EVERY order k (the "
            "payload's profile_aero_db), no blade count asserted (blades = 1: no blade-pass "
            "prior penalty); the motor family at the electromagnetic orders 84 and 98 only"
        ),
    ),
    "FLY018": dict(
        regime="airborne cruise",
        blades=2,
        motor_orders=None,
        orders_rule="MD2 production: 2-blade props, aerodynamic line at every k, motor at k = 3 p n",
    ),
    "FLY022": dict(
        regime="airborne cruise",
        blades=2,
        motor_orders=None,
        orders_rule="MD2 production: 2-blade props, aerodynamic line at every k, motor at k = 3 p n",
    ),
}
#: The production schedule (``scripts/noise_v4_flight.py fit`` / ``OptimV4``
#: defaults, ``fit_flight``'s restart-pass cap). Less than any of these is a
#: smoke, which may not write under :data:`RESULTS_ROOT`.
PRODUCTION = dict(
    adam_steps=1500,
    lbfgs_iters=600,
    lbfgs_frames=64,
    polish_iters=150,
    max_restart_passes=20,
    k_cap=K_CAP,
)
FRAME_STRIDE = 4
#: Block wander measurement (``scripts/noise_v3_measure_wander.py``'s choices):
#: block length, the per-track median measured block sd limit and the per-block
#: noise limit (dB), the OU lags; a group needs this many tracks for its own fit.
WANDER_BLOCK_S = 0.5
MAX_TRACK_NOISE_DB = 2.0
MAX_BLOCK_NOISE_DB = 4.0
WANDER_LAGS = (1,)
MIN_GROUP_TRACKS = 3
#: Seconds of ``--deadline-min`` kept for the payload, the held-out check and the renders.
OUTPUT_RESERVE_S = 300.0


def _finite_json(o: Any) -> Any:
    """``o`` with every non-finite float as ``null`` (strict JSON for any reader)."""
    if isinstance(o, dict):
        return {k: _finite_json(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_finite_json(v) for v in o]
    if isinstance(o, float | np.floating):
        return float(o) if math.isfinite(float(o)) else None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _finite_json(o.tolist())
    return o


# ── inputs ──────────────────────────────────────────────────────────────────


def load_windows(path: Path) -> dict[str, Any]:
    """One ``fit.npz`` / ``holdout.npz``, its contract checked."""
    with np.load(path) as z:
        audio = np.asarray(z["audio"], dtype=np.float32)
        rps = np.asarray(z["rps"], dtype=np.float64)
        sr = int(np.asarray(z["sr"]))
        start = np.asarray(z["start_s"], dtype=np.float64).reshape(-1)
    if sr != MODEL_SR:
        raise SystemExit(f"{path}: sr {sr}, the v4 flight front end needs {MODEL_SR} Hz audio")
    if audio.ndim != 3 or audio.shape[1] != 2:
        raise SystemExit(f"{path}: audio must be (N, 2, T), got {audio.shape}")
    if rps.shape != (audio.shape[0], 4, audio.shape[2]):
        raise SystemExit(f"{path}: rps must be (N, 4, T) = {(audio.shape[0], 4, audio.shape[2])}")
    if start.size != audio.shape[0]:
        raise SystemExit(f"{path}: {start.size} start_s for {audio.shape[0]} windows")
    if not (np.all(np.isfinite(audio)) and np.all(np.isfinite(rps)) and np.all(rps > 0.0)):
        raise SystemExit(f"{path}: non-finite audio or non-positive rps")
    return dict(
        audio=audio,
        rps=rps,
        start_s=start,
        duration_s=np.full(start.size, audio.shape[2] / float(sr)),
        path=str(path),
    )


def check_disjoint(fit: dict[str, Any], hold: dict[str, Any]) -> None:
    for s0, d0 in zip(hold["start_s"], hold["duration_s"], strict=True):
        for s1, d1 in zip(fit["start_s"], fit["duration_s"], strict=True):
            if s0 < s1 + d1 and s1 < s0 + d0:
                raise SystemExit(f"held-out window {s0:.3f}+{d0:g} s overlaps fit window {s1:.3f}")


def supports(win: dict[str, Any], rec: str, tag: str) -> list[Any]:
    """The windows on the v4 flight front end (``supports._support_from_clip``)."""
    from experiments.noise_model import spectrum as SP
    from experiments.noise_model import supports as SU
    from experiments.stochastic_fit.data import Clip

    out = []
    for i, (s, d) in enumerate(zip(win["start_s"], win["duration_s"], strict=True)):
        name = f"{rec}_{tag}{i + 1}@{s:.3f}+{d:g}"
        clip = Clip(name, rec, win["audio"][i], win["rps"][i], MODEL_SR)
        out.append(
            SU._support_from_clip(
                clip,
                kind="flight",
                name=name,
                n_fft=SP.FLIGHT_N_FFT,
                hop=SP.FLIGHT_HOP,
                segment=(float(s), float(s + d)),
                meta=dict(recording_id=rec, set=tag),
            )
        )
    return out


# ── constants measured on the fit windows ───────────────────────────────────


def channel_response(sups: list[Any], rec: str) -> Any:
    """The stereo pair's empirical response: per third-octave band (30-7900 Hz)
    each channel's band level (mean periodogram, dB) minus the two-channel mean,
    averaged over the fit windows."""
    from experiments.noise_model import model as MD
    from experiments.noise_model import wander as W

    f = np.asarray(sups[0].freqs_hz)
    edges = W.third_octave_edges()
    centres, bins = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        idx = np.nonzero((f >= lo) & (f < hi))[0]
        if idx.size:
            centres.append(math.sqrt(lo * hi))
            bins.append(idx)
    lev = np.stack(
        [
            np.stack(
                [10.0 * np.log10(np.asarray(s.power)[:, :, b].mean(axis=(1, 2))) for b in bins],
                axis=1,
            )
            for s in sups
        ]
    )  # (W, M, B)
    dev = lev - lev.mean(axis=1, keepdims=True)
    transfer = dev.mean(axis=0)
    return MD.ChannelGains(
        gains_db=transfer.mean(axis=1),
        source=f"{rec} fit windows ({len(sups)})",
        rig=f"november_m100_{rec}",
        rule=(
            f"per-channel transfer: each channel's deviation from the two-channel mean on the "
            f"{len(centres)} third-octave bands {centres[0]:.0f}-{centres[-1]:.0f} Hz (mean "
            "periodogram per window, dB), mean over the fit windows, interpolated in log f and "
            "held flat beyond the outer bands; window sd per band in the report"
        ),
        band_hz=np.asarray(centres),
        transfer_db=transfer,
    ), dev.std(axis=0)


def _median_block_sd(lb: Any) -> np.ndarray:
    """``(R, K)`` median measured mic-mean block sd, dB (inf with no measured block)."""
    ok = lb.measured() & np.isfinite(lb.s2_emp_micmean)
    out = np.full(ok.shape[:2], np.inf)
    for r, k in zip(*np.nonzero(ok.any(axis=-1)), strict=True):
        out[r, k] = float(np.sqrt(np.median(lb.s2_emp_micmean[r, k][ok[r, k]])))
    return out


def measure_wander(sups: list[Any]) -> dict[str, Any]:
    """``sigma_v`` / ``tau_v`` of the per-line block wander, per order group, on the
    fit windows only (the input :func:`flight_v4.flight_pedestal` reads)."""
    from experiments.noise_model import spectrum as SP
    from experiments.noise_model import v4_priors as V4
    from experiments.noise_model import wander as W

    edges = list(W.ORDER_GROUP_EDGES)
    n_g = len(edges)
    moms = [W.LagMoments() for _ in range(n_g)]
    pooled = W.LagMoments()
    counts = np.zeros(n_g, dtype=np.int64)
    windows = []
    alpha = V4.ALPHA_AERO[0]
    for s in sups:
        blocks = W.frame_blocks(s.frame_centres_s, WANDER_BLOCK_S)
        lb = W.measure_lines(s.power, s.freqs_hz, s.carrier_rev_s, blocks, block_s=WANDER_BLOCK_S)
        sel = lb.resolvable() & lb.dominant() & (_median_block_sd(lb) <= MAX_TRACK_NOISE_DB)
        y, s2 = lb.micmean_db(), lb.s2_emp_micmean
        # the v4 line speed law at its prior centre, removed per block
        adj = 10.0 * alpha * np.log10(np.maximum(lb.carrier_rev_s, 1e-9) / SP.AMP_RPS_REF)
        used = 0
        for r, k in zip(*np.nonzero(sel), strict=True):
            yy = np.where(s2[r, k] <= MAX_BLOCK_NOISE_DB**2, y[r, k], np.nan) - adj[r]
            g = int(W.order_group(lb.orders[k]))
            if moms[g].add(yy, None, s2[r, k], n_blocks=lb.n_blocks):
                counts[g] += 1
                used += 1
                pooled.add(yy, None, s2[r, k], n_blocks=lb.n_blocks)
        windows.append(
            dict(name=s.name, blocks=lb.n_blocks, candidate_tracks=int(sel.sum()), tracks=used)
        )

    def fit(mom: Any) -> Any:
        return W.fit_ou(mom, WANDER_BLOCK_S, lags=WANDER_LAGS)

    def tau_ok(f: Any) -> bool:
        return bool(np.isfinite(f.tau_s) and not f.clipped and not f.edge)

    rule = (
        f"{WANDER_BLOCK_S} s blocks of the order-tracked mic-mean line level "
        "(experiments.noise_model.wander.measure_lines) on the resolvable, rotor-dominant lines "
        f"with median measured block sd <= {MAX_TRACK_NOISE_DB} dB (blocks noisier than "
        f"{MAX_BLOCK_NOISE_DB} dB missing), the v4 line speed law at alpha = {alpha} removed, "
        "each track centred on its own window, OU by the exact lag-1 method of moments "
        "(wander.fit_ou) per order group; the rotor-common part is not split off. A group with "
        f"fewer than {MIN_GROUP_TRACKS} tracks takes the pooled fit; a clipped fit (no variance "
        "above the block noise) is sigma = 0; a tau on the grid edge or unfitted takes the "
        "pooled tau, else the v4 prior median rate"
    )
    base = dict(
        schema="november-wander/1",
        block_s=WANDER_BLOCK_S,
        rule=rule,
        windows=windows,
        n_tracks_by_group=counts.tolist(),
    )
    if pooled.n_pairs == 0:
        return base | dict(
            source="prior",
            note="no measurable line track in the fit windows: amplitude pedestal at the v4 "
            "bench prior medians (v4_priors.AMP_GROUPS, gamma_amp_log_median)",
        )
    pf = fit(pooled)
    p_sig = 0.0 if pf.clipped else pf.sigma_db
    p_tau = pf.tau_s if tau_ok(pf) else None
    sig, tau, sig_src, tau_src, fits = [], [], [], [], []
    for g in range(n_g):
        gf = fit(moms[g]) if counts[g] >= MIN_GROUP_TRACKS else None
        fits.append(None if gf is None else gf.as_dict())
        if gf is not None:
            sig.append(0.0 if gf.clipped else gf.sigma_db)
            sig_src.append("group_clipped" if gf.clipped else "group")
        else:
            sig.append(p_sig)
            sig_src.append("pooled")
        if gf is not None and tau_ok(gf):
            tau.append(gf.tau_s)
            tau_src.append("group")
        elif p_tau is not None:
            tau.append(p_tau)
            tau_src.append("pooled")
        else:
            rate_ln = float(V4.gamma_amp_log_median(np.asarray(edges[g])))
            tau.append(1.0 / (2.0 * math.pi * math.exp(rate_ln)))
            tau_src.append("prior")
    k_edges = edges + [10**6]
    return base | dict(
        source="measured",
        sigma_v_db=p_sig,
        tau_v_s=p_tau,
        pooled_fit=pf.as_dict(),
        group_fits=fits,
        sigma_v_db_by_order=dict(k_edges=k_edges, sigma_db=sig, source=sig_src),
        tau_v_s_by_order=dict(k_edges=k_edges, tau_s=tau, tau_source=tau_src),
    )


def amplitude_pedestal(k_max: int, wander: dict[str, Any]) -> dict[str, np.ndarray]:
    """:func:`flight_v4.flight_pedestal` on the measured wander, or the v4 prior
    medians (capped as the flight fit caps) when nothing was measured."""
    from experiments.noise_model import flight_v4 as FV
    from experiments.noise_model import v4_priors as V4

    if wander["source"] == "measured":
        return FV.flight_pedestal(k_max, wander=wander)
    k = np.arange(1, k_max + 1)
    ped = FV.flight_pedestal(k_max, wander=dict(sigma_v_db=0.0, tau_v_s=1.0))
    s2 = np.exp(V4.amp_log_median(k))
    ped.update(
        sigma2_amp=np.minimum(s2, V4.amp_cap(k)),
        sigma2_amp_uncapped=s2,
        gamma_amp_hz=np.exp(V4.gamma_amp_log_median(k)),
    )
    return ped


# ── the pool ────────────────────────────────────────────────────────────────


def _members(sups: list[Any]) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    return [
        (s.name, np.asarray(s.power), np.asarray(s.carrier_rev_s_audio), np.asarray(s.frame_starts))
        for s in sups
    ]


def build(
    args: argparse.Namespace, rec: str, sups: list[Any], gains: Any, wander: dict[str, Any]
) -> Any:
    """The pool at the smallest work rate of :data:`WORK_RATES` whose Nyquist clears
    the highest modelled order on the fastest carrier (:func:`flight_v4.joint_k_max`
    is the test), with the November constants."""
    from experiments.noise_model import flight_v4 as FV
    from experiments.noise_model import model as MD
    from experiments.noise_model import spectrum as SP

    rig = RIGS[rec]
    batch, k = None, 0
    for sr_work in WORK_RATES:
        batch = MD.flight_batch(
            name=f"november_{rec}",
            members=_members(sups),
            sr=MODEL_SR,
            n_fft=SP.FLIGHT_N_FFT,
            hop=SP.FLIGHT_HOP,
            k_cap=int(args.k_cap),
            frame_stride=FRAME_STRIDE,
            channel_gains=gains,
            device=args.device,
            chunk_frames=int(args.chunk_frames),
            sr_work=sr_work,
        )
        try:
            k = FV.joint_k_max(batch, k_cap=int(args.k_cap))
            break
        except ValueError as err:
            if sr_work == WORK_RATES[-1]:
                raise SystemExit(f"{err} (work rates tried: {list(WORK_RATES)})") from err
    assert batch is not None
    if rig["motor_orders"] is None:
        step = 3 * POLE_PAIRS
        motor = np.arange(step, k + 1, step, dtype=np.int64)
    else:
        motor = np.asarray([m for m in rig["motor_orders"] if m <= k], dtype=np.int64)
    return FV.FlightData(
        batch=replace(batch, k_max=k),
        k_max=k,
        motor_orders=motor,
        geometry_db=np.zeros((batch.n_rotors, batch.n_mics)),
        pedestal=amplitude_pedestal(k, wander),
        wander_path=f"measured on the {rec} fit windows (fit.json november.wander)",
        mics=list(range(batch.n_mics)),
        blades=int(rig["blades"]),
        pole_pairs=POLE_PAIRS,
    )


def heldout_whittle(model: Any, sups: list[Any], gains: Any, args: argparse.Namespace) -> Any:
    """The fitted model's Whittle risk on the held-out windows (same front end,
    same channel response), or why it was not evaluated."""
    from experiments.noise_model import model as MD
    from experiments.noise_model import spectrum as SP

    grid = model.grid
    hb = MD.flight_batch(
        name="heldout",
        members=_members(sups),
        sr=MODEL_SR,
        n_fft=SP.FLIGHT_N_FFT,
        hop=SP.FLIGHT_HOP,
        k_cap=int(model.data.k_max),
        frame_stride=FRAME_STRIDE,
        channel_gains=gains,
        device=args.device,
        chunk_frames=int(args.chunk_frames),
        sr_work=int(grid.sr_work),
    )
    assert hb.rate_work is not None
    top = float(hb.rate_work.max()) * int(model.data.k_max)
    if top >= 0.5 * float(grid.sr_work):
        return dict(skipped=f"order {model.data.k_max} reaches {top:.0f} Hz on a held-out carrier")
    return dict(whittle_nats=model.whittle(hb), weighted_cells=weighted_cells(hb))


def weighted_cells(batch: Any) -> float:
    """The exposure-weighted cell count ``sum_n w_n x M x band bins`` (the Whittle
    risk's own weighting), the per-cell normaliser of fit and held-out alike."""
    return float(batch.weights.sum()) * int(batch.n_mics) * int(batch.band.sum())


# ── the CLI ─────────────────────────────────────────────────────────────────


def _reduced(args: argparse.Namespace) -> list[str]:
    return [k for k, v in PRODUCTION.items() if int(getattr(args, k)) < int(v)]


def _under(path: Path, root: Path) -> bool:
    p, r = path.resolve(), root.resolve()
    return p == r or r in p.parents


def _rms_dbfs(x: np.ndarray) -> list[float]:
    return [
        float(10.0 * np.log10(max(float(np.mean(c.astype(np.float64) ** 2)), 1e-30))) for c in x
    ]


def _compact_optimiser(opt: dict[str, Any]) -> dict[str, Any]:
    """The optimiser record without the per-iteration L-BFGS traces (fit.json keeps them)."""
    out = dict(opt)
    out["stages"] = {
        k: (
            [dict(r) for r in v]
            if isinstance(v, list)
            else {kk: vv for kk, vv in v.items() if kk != "trace"}
        )
        for k, v in opt["stages"].items()
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--input", required=True, type=Path, help="DIR holding <recording>/fit.npz")
    ap.add_argument("--recording", required=True, choices=RECORDINGS)
    ap.add_argument("--out", type=Path, default=RESULTS_ROOT, help="writes OUT/<recording>/")
    ap.add_argument("--device", choices=("cuda", "cpu"), default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--k-cap", type=int, default=K_CAP)
    ap.add_argument("--chunk-frames", type=int, default=16)
    ap.add_argument("--kappa", type=float, default=0.0)
    ap.add_argument("--adam-steps", type=int, default=PRODUCTION["adam_steps"])
    ap.add_argument("--adam-lr", type=float, default=0.02)
    ap.add_argument("--adam-batch", type=int, default=8)
    ap.add_argument("--lbfgs-iters", type=int, default=PRODUCTION["lbfgs_iters"])
    ap.add_argument("--lbfgs-frames", type=int, default=PRODUCTION["lbfgs_frames"])
    ap.add_argument("--polish-iters", type=int, default=PRODUCTION["polish_iters"])
    ap.add_argument("--max-restart-passes", type=int, default=PRODUCTION["max_restart_passes"])
    ap.add_argument(
        "--deadline-min",
        type=float,
        default=None,
        help="stop the restart passes in time to write the outputs (converged=false then)",
    )
    ap.add_argument("--progress", type=int, default=100, help="print every N Adam steps")
    args = ap.parse_args(argv)

    import soundfile as sf
    import torch

    from data_processing.noise_model.render import render_noise
    from data_processing.noise_model.resample import AA_PASS_HZ, AA_STOP_HZ
    from experiments.noise_model import flight_v4 as FV
    from experiments.noise_model import spectrum as SP
    from experiments.noise_model import v4_priors as V4

    reduced = _reduced(args)
    if reduced and _under(args.out, RESULTS_ROOT):
        raise SystemExit(
            f"reduced schedule ({', '.join(reduced)}) is a smoke: refused under {RESULTS_ROOT}"
        )
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, int(args.threads)))
    rec = str(args.recording)
    rig = RIGS[rec]
    src = args.input / rec
    out = args.out / rec
    t0 = time.time()

    fit_w = load_windows(src / "fit.npz")
    hold_w = load_windows(src / "holdout.npz")
    if hold_w["audio"].shape[0] != 2:
        raise SystemExit(
            f"{src / 'holdout.npz'}: expected the two examples, got {hold_w['audio'].shape[0]}"
        )
    check_disjoint(fit_w, hold_w)
    sel_path = src / "selection.json"
    selection = json.loads(sel_path.read_text()) if sel_path.exists() else None
    rotor_order = list((selection or {}).get("rotor_order") or ROTOR_ORDER)

    fit_s = supports(fit_w, rec, "fit")
    hold_s = supports(hold_w, rec, "holdout")
    gains, gains_sd = channel_response(fit_s, rec)
    wander = measure_wander(fit_s)
    print(
        f"{rec}: {len(fit_s)} fit windows ({fit_w['duration_s'][0]:g} s), channel response "
        f"{np.round(gains.gains_db, 2).tolist()} dB band-mean, wander {wander['source']} "
        f"(tracks by group {wander['n_tracks_by_group']}), {time.time() - t0:.1f}s",
        flush=True,
    )

    data = build(args, rec, fit_s, gains, wander)
    b = data.batch
    print(
        f"pool: {int(b.power.shape[1])} frames x {b.n_mics} ch, K {data.k_max}, motor orders "
        f"{data.motor_orders.tolist()}, work rate {b.grid.sr_work} Hz, device {b.power.device}",
        flush=True,
    )
    t1 = time.time()
    measured = FV.measure(data)
    print(
        f"measured: mu {measured.floor_mean_db:.2f} dB, sigma_B {measured.floor_shape_sd_db:.2f} dB "
        f"({time.time() - t1:.1f}s)",
        flush=True,
    )
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    optim = FV.OptimV4(
        adam_steps=int(args.adam_steps),
        adam_lr=float(args.adam_lr),
        adam_batch=int(args.adam_batch),
        lbfgs_iters=int(args.lbfgs_iters),
        lbfgs_frames=int(args.lbfgs_frames),
        polish_iters=int(args.polish_iters),
        seed=SEED,
    )
    deadline_s = None
    if args.deadline_min:
        # counted from the fit's own start: the set-up already spent and the output
        # reserve are charged against the budget
        deadline_s = 60.0 * float(args.deadline_min) - (time.time() - t0) - OUTPUT_RESERVE_S
        if deadline_s <= 0:
            raise SystemExit(f"--deadline-min {args.deadline_min} leaves no time for the fit")
    model, record = FV.fit_flight(
        data,
        measured,
        kappa=float(args.kappa),
        optim=optim,
        progress=int(args.progress),
        deadline_s=deadline_s,
        max_restart_passes=int(args.max_restart_passes),
        free_phase_pedestal=True,
    )
    gpu_peak = float(torch.cuda.max_memory_allocated()) / 2**30 if args.device == "cuda" else None
    conv = record["optimiser"]["convergence"]
    print(
        f"fit: total {record['objective']['total_nats']:.6g} nats (start "
        f"{record['objective']['start_total_nats']:.6g}), converged {conv['converged']} "
        f"({conv['stop']}), wall {record['optimiser']['wall_s']:.0f}s, GPU peak {gpu_peak} GiB",
        flush=True,
    )

    pay = FV.payload(model, record, channel_gains=gains)
    pay["rig"] = f"november_m100_{rec}"
    pay["rotor_order"] = rotor_order
    pay["params"]["mic_response"]["geometry_rule"] = (
        "none asserted: Delta G_rm = 0 (no rotor-to-capsule geometry of the stereo recorder is "
        "known); the per-band L/R difference is array_response, measured on the fit windows"
    )
    pay["params"]["pedestal"]["source"]["amplitude"] = (
        "block wander measured on the fit windows (november.wander): sigma2 = (sigma_v ln10/20)^2 "
        "capped at v4_priors.amp_cap, gamma = 1/(2 pi tau_v), per order group"
        if wander["source"] == "measured"
        else "v4 bench prior medians (v4_priors.AMP_GROUPS, gamma_amp_log_median), capped: "
        "nothing measurable in the fit windows"
    )
    pay["pool"]["specs"] = [s.name for s in fit_s]
    fit_whittle_cells = weighted_cells(b)
    held = heldout_whittle(model, hold_s, gains, args)
    pay["november"] = dict(
        recording=rec,
        regime=rig["regime"],
        dataset=DATASET,
        rps_source="native rps of fit.npz on the audio grid, unmodified (no VK scoring gain)",
        orders_rule=rig["orders_rule"],
        motor_orders_requested=None if rig["motor_orders"] is None else list(rig["motor_orders"]),
        wander=wander,
        channel_response_window_sd_db=gains_sd.tolist(),
        fit_windows=dict(
            start_s=fit_w["start_s"].tolist(), duration_s=fit_w["duration_s"].tolist()
        ),
        selection=selection,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "fit.json").write_text(json.dumps(_finite_json(pay), indent=1))
    # the renders read the payload back from disk: what ships is what renders
    portable = json.loads((out / "fit.json").read_text())

    examples = []
    for i in range(2):
        idx = i + 1
        seed_i = int(np.random.SeedSequence([SEED, idx]).generate_state(1)[0])
        t2 = time.time()
        gen, diag = render_noise(
            portable,
            hold_w["rps"][i],
            sr=MODEL_SR,
            n_mics=2,
            seed=seed_i,
            return_diagnostics=True,
        )
        ref = hold_w["audio"][i]
        gen32 = np.asarray(gen, dtype=np.float32)
        if not np.all(np.isfinite(gen32)):
            raise SystemExit(f"example {idx}: the render is not finite")
        sf.write(
            out / f"reference_{idx}.wav", np.ascontiguousarray(ref.T), MODEL_SR, subtype="FLOAT"
        )
        sf.write(
            out / f"generated_{idx}.wav", np.ascontiguousarray(gen32.T), MODEL_SR, subtype="FLOAT"
        )
        examples.append(
            dict(
                index=idx,
                start_s=float(hold_w["start_s"][i]),
                duration_s=float(hold_w["duration_s"][i]),
                reference=f"reference_{idx}.wav",
                generated=f"generated_{idx}.wav",
                render_seed=seed_i,
                render_work_rate_hz=int(diag.get("sr_work", 0)),
                rps_mean=hold_w["rps"][i].mean(axis=1).tolist(),
                rms_dbfs=dict(reference=_rms_dbfs(ref), generated=_rms_dbfs(gen32)),
                peak_abs=dict(
                    reference=float(np.max(np.abs(ref))), generated=float(np.max(np.abs(gen32)))
                ),
                render_s=time.time() - t2,
            )
        )
        print(
            f"example {idx}: {examples[-1]['rms_dbfs']} dBFS ({time.time() - t2:.1f}s)", flush=True
        )

    obj = dict(record["objective"])
    obj["whittle_per_weighted_cell"] = obj["whittle_nats"] / fit_whittle_cells
    obj["weighted_cells"] = fit_whittle_cells
    if "whittle_nats" in held:
        held["whittle_per_weighted_cell"] = held["whittle_nats"] / held["weighted_cells"]
    obj["heldout"] = held
    stages = record["optimiser"]["stages"]
    trail = [
        ("start", obj["start_total_nats"]),
        ("after_adam", record["optimiser"]["total_after_adam"]),
    ]
    trail += [("polish", stages["polish"]["loss_after"])]
    trail += [
        (f"restart_pass_{i + 1}", r["loss_after"]) for i, r in enumerate(stages["restart_passes"])
    ]
    best_stage, best = min(trail, key=lambda t: float(t[1]))
    obj["all_frames_trail_nats"] = [dict(stage=s, total_nats=float(v)) for s, v in trail]
    obj["best_total_nats"] = float(best)
    obj["best_stage"] = best_stage
    obj["payload_is_best"] = float(trail[-1][1]) <= float(best) + 1e-9 * abs(float(best))
    report = dict(
        recording=rec,
        sr=MODEL_SR,
        seed=SEED,
        fit_start_s=fit_w["start_s"].tolist(),
        fit_duration_s=fit_w["duration_s"].tolist(),
        examples=examples,
        model=(
            "noise model v4, flight procedure B (experiments.noise_model.flight_v4): fresh "
            f"per-recording MAP Whittle fit on {rec} ({rig['regime']}), phase pedestal free per "
            "order; rendered by data_processing.noise_model.render.render_noise"
        ),
        fit=dict(
            optimizer=dict(
                name="Adam (minibatch) -> strong-Wolfe L-BFGS: stratified subset, all-frames "
                "polish, fresh-memory restart passes (experiments.noise_model.flight_v4.fit_flight)",
                **_compact_optimiser(record["optimiser"]),
            ),
            objective=obj,
            config=dict(
                schedule="production" if not reduced else f"REDUCED (smoke): {reduced}",
                front_end=dict(
                    sr=MODEL_SR,
                    n_fft=SP.FLIGHT_N_FFT,
                    hop=SP.FLIGHT_HOP,
                    frame_stride=FRAME_STRIDE,
                    sr_work=int(b.grid.sr_work),
                    likelihood_band_hz=[SP.BAND_F_MIN, SP.BAND_F_MAX],
                    renderer_antialias_hz=dict(pass_hz=AA_PASS_HZ, stop_hz=AA_STOP_HZ),
                ),
                pool=dict(
                    windows=len(fit_s),
                    frames=int(b.power.shape[1]),
                    channels=int(b.n_mics),
                    n_cells=int(b.n_cells),
                ),
                k_max=int(data.k_max),
                k_cap=int(args.k_cap),
                motor_orders=data.motor_orders.tolist(),
                blades=int(data.blades),
                pole_pairs=POLE_PAIRS,
                orders_rule=rig["orders_rule"],
                free_phase_pedestal=True,
                kappa=float(args.kappa),
                shaft_lam_pinned_per_s=V4.FLIGHT_LAM,
                geometry="none: Delta G_rm = 0 for every rotor and channel",
                channel_response=gains.as_dict(),
                amplitude_wander=wander,
                rps="native, unmodified",
                device=str(b.power.device),
                gpu_peak_gib=gpu_peak,
                deadline_min=args.deadline_min,
                wall_s_total=time.time() - t0,
                command=" ".join(sys.argv),
            ),
            convergence=conv,
        ),
        selection=selection,
        limitations=[
            f"Bandwidth: the v4 flight front end is {MODEL_SR} Hz (likelihood band "
            f"{SP.BAND_F_MIN:.0f}-{SP.BAND_F_MAX:.0f} Hz) and the renderer's anti-alias passes "
            f"{AA_PASS_HZ:.0f} / stops {AA_STOP_HZ:.0f} Hz: the generated audio carries nothing "
            "above 7.9 kHz, the 16 kHz references end at 8 kHz, and nothing of the 48 kHz "
            "originals above that is modelled.",
            "Geometry: none. Delta G_rm = 0, so every rotor reaches both channels alike; the only "
            "L/R difference is one empirical per-band channel response measured on the fit "
            "windows, shared by all rotors and the floor.",
            "Speeds: native rps, unmodified; label error is absorbed by the shaft wander "
            f"(lambda pinned at {V4.FLIGHT_LAM:.1f} 1/s, MD2's label band edge, not recalibrated "
            "for these logs).",
            "Amplitude pedestal: held from the fit windows' block wander "
            f"({WANDER_BLOCK_S} s blocks inside {fit_w['duration_s'][0]:g} s windows, each track "
            "centred on its window: wander slower than a few seconds is not seen; rotor-common "
            "part not split off), capped at the v4 caps; the Whittle likelihood does not "
            "identify it.",
            "No wind block: any outdoor wind enters only through the fitted broadband floor.",
            "One seed (2026): one fit restart and one render draw per example; no restart "
            "spread is available and no draw was selected.",
            f"Convergence: production test (gain/cell < {FV.TOL_NATS_PER_CELL} and group "
            f"movements under {FV.MOVE_TOL}); this fit stopped with '{conv['stop']}', "
            f"converged = {conv['converged']}.",
        ]
        + (
            [
                "FLY013 orders: the per-order line family is a phenomenological shaft-order "
                "branch (no blades asserted); motor family at 84 and 98 only. Those orders are "
                "inside the band only on the ~35 rev/s plateau (84 x 132 rev/s = 11 kHz).",
                "FLY013 joins the ~35 and ~132 rev/s plateaus in ONE fit with one set of speed "
                "laws (a ~3.8x span); the paper's MD2 model uses per-regime fits composed in the "
                "renderer instead.",
            ]
            if rec == "FLY013"
            else []
        )
        + ([f"REDUCED SCHEDULE (smoke): {reduced}; not a result."] if reduced else []),
    )
    (out / "report.json").write_text(json.dumps(_finite_json(report), indent=1))
    print(f"wrote {out}: wall {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
