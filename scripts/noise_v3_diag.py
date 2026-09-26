"""Noise v3 diagnosis: why the latents eat the rig (round-3 design inputs).

Light subcommands (JSON + one expected frame per fit; laptop under the cap):

``ltas-bias``  1(e): does a render reproduce the LTAS the fit explains? Per
    fit, per LTAS band: the fit's own block-mean spectrum (its FITTED latents,
    pooled over every window and block) against the render's expectation
    (FRESH zero-mean OU tracks in dB, so every family multiplies its power by
    ``E[10^{x/10}] = exp((c s)^2 / 2)``, ``c = ln 10 / 10``, ``s`` the track's
    sd after linear interpolation between block knots) and against the
    latents-at-zero model (:func:`experiments.noise_model.render.expected_periodogram`'s
    convention). The difference splits into the static part of the fitted
    latents (the render never draws it) and the Jensen term.

Every expected spectrum is one frame, every rotor at a constant carrier (the
middle of the pool's recorded span), mic mean; lines through the fit's own
kernel (:func:`spectrum.flight_line_spectra`), the floor through
:func:`spectrum._floor_frames` at latent zero, its latent moves applied as a
dB gain ``u + sum_j B_j(f) u_j`` on the analysis bins (the render's WOLA gain).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

C_DB = math.log(10.0) / 10.0
LTAS_BANDS_HZ = ((100, 300), (300, 700), (700, 1500), (1500, 3000), (3000, 5000), (5000, 7000))
POOLS = ("dregon_room2_floor", "michaels_fly125_cruise", "michaels_fly125_standby")


def _db(x: np.ndarray | float) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-300))


def load_fit(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def fit_components(fit: dict[str, Any], rps: float | None = None) -> dict[str, Any]:
    """``(R, K, F)`` line spectra, ``(F,)`` floor at latent zero, ``(F,)``
    wind, ``(J, F)`` floor-wander basis, bin freqs: one frame, mic mean,
    transfer applied."""
    import torch

    from data_processing.noise_model import spectrum as DSP
    from data_processing.noise_model.floor import floor_geometry
    from experiments.noise_model import model as MD
    from experiments.noise_model import render as RD
    from experiments.noise_model import spectrum as SP

    p = fit["params"]
    batch = fit["diagnostics"]["batch"]
    if rps is None:
        rps = 0.5 * (float(batch["carrier_min_rev_s"]) + float(batch["carrier_max_rev_s"]))
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    n_r = prof.shape[0]
    sr = int(fit["front_end"]["sr"])
    n_fft = int(fit["front_end"]["n_fft"])
    grid = SP.flight_grid(
        sr=sr, n_fft=n_fft, hop=int(fit["front_end"]["hop"]), sr_work=RD.fit_work_rate(fit)
    )
    params = MD.params_from_dict(p, n_mics=int(fit["n_mics"]))
    k_max = min(prof.shape[1], DSP.k_max_for_carrier(np.array([rps]), sr, k_cap=prof.shape[1]))
    carrier = np.full((n_r, n_fft), float(rps))
    with torch.no_grad():
        rate = SP.flight_rate_work(grid, carrier, np.array([0]))
        lines = (
            SP.flight_line_spectra(grid, params, rate_work=rate, k_max=k_max)[:, 0].cpu().numpy()
        )
        floor = SP._floor_frames(grid, rate, params.floor)[:, 0].cpu().numpy().mean(axis=0)
        wind = (
            SP.wind_frames(grid, params).cpu().numpy().mean(axis=0)
            if params.wind_db is not None
            else np.zeros_like(floor)
        )
        transfer = grid.transfer_power.cpu().numpy()
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    basis = floor_geometry(freqs, DSP.floor_ctrl_hz(sr))[0]  # (F, J)
    return dict(
        rps=float(rps),
        k_max=int(k_max),
        freqs=freqs,
        lines=lines * transfer[None, None, :],
        floor=floor * transfer,
        wind=wind * transfer,
        basis=basis.T,
    )


def _interp_var_factor(rho: np.ndarray | float) -> np.ndarray:
    """Time-mean variance of a linearly interpolated OU between two knots of
    correlation ``rho``, in units of the stationary variance: 2/3 + rho/3."""
    return 2.0 / 3.0 + np.asarray(rho, dtype=np.float64) / 3.0


def band_power(freqs: np.ndarray, spec: np.ndarray) -> np.ndarray:
    return np.array(
        [spec[..., (freqs >= lo) & (freqs < hi)].sum(axis=-1) for lo, hi in LTAS_BANDS_HZ]
    ).T


def jensen_factors(wander: Any, k_max: int, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``((K,) line, (F,) floor)`` render-mean power factors ``E[10^{x/10}]``
    of fresh zero-mean OU tracks (sd after linear interpolation in time)."""
    sd_d = float(wander.track_sigma("d", k_max))
    sd_v = np.broadcast_to(np.asarray(wander.track_sigma("v", k_max), dtype=np.float64), (k_max,))
    rho_d = float(wander.track_rho("d", k_max))
    rho_v = np.broadcast_to(np.asarray(wander.track_rho("v", k_max), dtype=np.float64), (k_max,))
    var_line = sd_d**2 * _interp_var_factor(rho_d) + sd_v**2 * _interp_var_factor(rho_v)
    var_floor = wander.sigma("u") ** 2 * _interp_var_factor(wander.rho("u")) + wander.sigma(
        "uj"
    ) ** 2 * _interp_var_factor(wander.rho("uj")) * (basis**2).sum(axis=0)
    return np.exp(0.5 * C_DB**2 * var_line), np.exp(0.5 * C_DB**2 * var_floor)


def ltas_bias(
    fit: dict[str, Any], comp: dict[str, Any] | None = None, measured: dict[str, Any] | None = None
) -> dict[str, Any]:
    """The band LTAS of: the fit's block mean with its fitted latents (``fit``),
    that with the static part (pool mean per track) removed (``fit_nostatic``),
    the render expectation (``render``), latents at zero (``zero``), the
    render expectation plus the static part (``render_static``) and that
    under the ``measured`` wander record (``render_static_measured``)."""
    from data_processing.noise_model.v3 import Wander

    comp = comp or fit_components(fit)
    k_max = comp["k_max"]
    lines, floor, wind, basis, freqs = (
        comp[k] for k in ("lines", "floor", "wind", "basis", "freqs")
    )
    wins = fit["latents"]["windows"]
    d = np.concatenate([np.asarray(w["d"], dtype=np.float64) for w in wins], axis=1)  # (R, B)
    v = np.concatenate([np.asarray(w["v"], dtype=np.float64)[:, :k_max] for w in wins], axis=2)
    u = np.concatenate([np.asarray(w["u"], dtype=np.float64) for w in wins])
    uj = np.concatenate([np.asarray(w["uj"], dtype=np.float64) for w in wins], axis=1)  # (J, B)
    n_b = u.size
    # fitted: sum_b over pooled blocks (each block weighs alike; frames per
    # block are near equal)
    line_gain = 10.0 ** ((d[:, None, :] + v) / 10.0)  # (R, K, B)
    floor_gain = 10.0 ** ((u[None, :] + basis.T @ uj) / 10.0)  # (F, B)
    fit_spec = (
        np.einsum("rkf,rkb->f", lines, line_gain) / n_b + floor * floor_gain.mean(axis=1) + wind
    )
    ds, vs, us, ujs = d.mean(axis=1), v.mean(axis=2), u.mean(), uj.mean(axis=1)
    line_gain0 = 10.0 ** (((d - ds[:, None])[:, None, :] + (v - vs[..., None])) / 10.0)
    floor_gain0 = 10.0 ** (((u - us)[None, :] + basis.T @ (uj - ujs[:, None])) / 10.0)
    fit_nostatic = (
        np.einsum("rkf,rkb->f", lines, line_gain0) / n_b + floor * floor_gain0.mean(axis=1) + wind
    )
    wander = Wander.from_mapping(fit["params"]["wander"])
    jl, jf = jensen_factors(wander, k_max, basis)
    render = np.einsum("rkf,k->f", lines, jl) + floor * jf + wind
    zero = lines.sum(axis=(0, 1)) + floor + wind
    static_line = 10.0 ** ((ds[:, None] + vs) / 10.0)
    static_floor = 10.0 ** ((us + basis.T @ ujs) / 10.0)
    render_static = (
        np.einsum("rkf,rk,k->f", lines, static_line, jl) + floor * static_floor * jf + wind
    )
    extra: dict[str, np.ndarray] = {}
    if measured is not None:
        jlm, jfm = jensen_factors(Wander.from_mapping(measured), k_max, basis)
        extra["render_static_measured"] = (
            np.einsum("rkf,rk,k->f", lines, static_line, jlm) + floor * static_floor * jfm + wind
        )
    # the same, split by component
    comp_lines = dict(
        fit=np.einsum("rkf,rkb->f", lines, line_gain) / n_b,
        render=np.einsum("rkf,k->f", lines, jl),
        zero=lines.sum(axis=(0, 1)),
    )
    comp_floor = dict(fit=floor * floor_gain.mean(axis=1), render=floor * jf, zero=floor)
    specs = dict(
        fit=fit_spec,
        fit_nostatic=fit_nostatic,
        render=render,
        zero=zero,
        render_static=render_static,
        **extra,
    )
    bands = {k: band_power(freqs, s) for k, s in specs.items()}
    tot = {k: float(s[(freqs >= 100) & (freqs < 7000)].sum()) for k, s in specs.items()}

    def dev(a: str, b: str) -> dict[str, Any]:
        raw = _db(bands[a]) - _db(bands[b])
        shift = float(_db(tot[a]) - _db(tot[b]))
        return dict(abs_db=raw.tolist(), rms_matched_db=(raw - shift).tolist(), total_db=shift)

    return dict(
        rps=comp["rps"],
        k_max=k_max,
        bands_hz=[list(b) for b in LTAS_BANDS_HZ],
        band_db={k: _db(b).tolist() for k, b in bands.items()},
        lines_band_db={k: _db(band_power(freqs, s)).tolist() for k, s in comp_lines.items()},
        floor_band_db={k: _db(band_power(freqs, s)).tolist() for k, s in comp_floor.items()},
        render_minus_fit=dev("render", "fit"),
        zero_minus_fit=dev("zero", "fit"),
        render_static_minus_fit=dev("render_static", "fit"),
        fit_nostatic_minus_fit=dev("fit_nostatic", "fit"),
        **(
            {"render_static_measured_minus_fit": dev("render_static_measured", "fit")}
            if "render_static_measured" in specs
            else {}
        ),
        jensen_line_db_by_group={
            f"{int(lo)}-{int(hi) - 1}": float(_db(jl[int(lo) - 1 : min(int(hi) - 1, k_max)]).mean())
            for lo, hi in zip((1, 3, 9, 25, 61), (3, 9, 25, 61, k_max + 1), strict=True)
            if int(lo) <= k_max
        },
        jensen_floor_db_range=[float(_db(jf).min()), float(_db(jf).max())],
        static_mean_db=dict(
            d=ds.tolist(),
            v_rotor_mean=vs.mean(axis=1).tolist(),
            u=float(us),
            uj=ujs.tolist(),
        ),
    )


def fold_static(fit: dict[str, Any]) -> dict[str, Any]:
    """A DEEP COPY of a ``noise-v3-fit/1`` payload with the static part of its
    fitted latents (the pool mean of every track over every window's blocks)
    moved into the rig, exactly: ``profile_db[r, k] += mean(d_r) + mean(v_rk)``,
    ``floor_mean_db += mean(u)``, ``floor_shape_z += L^-1 mean(u_j) / sigma_B``
    (the wander's ``u_j`` and the floor spline share the interpolation basis),
    and every window's tracks minus the same means. The expected spectrum of
    every block is unchanged; a render (fresh zero-mean tracks) now carries
    the static part. Priors are NOT re-priced here (``objective`` is stale and
    dropped)."""
    import copy

    from data_processing.noise_model import spectrum as DSP

    out = copy.deepcopy(fit)
    p = out["params"]
    wins = out["latents"]["windows"]
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    k_fit = min(prof.shape[1], min(np.asarray(w["v"]).shape[1] for w in wins))
    d = np.concatenate([np.asarray(w["d"], dtype=np.float64) for w in wins], axis=1)
    v = np.concatenate([np.asarray(w["v"], dtype=np.float64)[:, :k_fit] for w in wins], axis=2)
    u = np.concatenate([np.asarray(w["u"], dtype=np.float64) for w in wins])
    uj = np.concatenate([np.asarray(w["uj"], dtype=np.float64) for w in wins], axis=1)
    ds, vs, us, ujs = d.mean(axis=1), v.mean(axis=2), float(u.mean()), uj.mean(axis=1)
    prof[:, :k_fit] += ds[:, None] + vs
    p["profile"]["profile_db"] = prof.tolist()
    p["floor"]["floor_mean_db"] = float(p["floor"]["floor_mean_db"]) + us
    chol = DSP.floor_shape_chol(DSP.floor_ctrl_hz(int(out["sr"])))
    sd_b = float(p["floor"]["floor_shape_sd_db"])
    z = np.asarray(p["floor"]["floor_shape_z"], dtype=np.float64)
    p["floor"]["floor_shape_z"] = (z + np.linalg.solve(chol, ujs) / sd_b).tolist()
    for w in wins:
        w["d"] = (np.asarray(w["d"]) - ds[:, None]).tolist()
        wv = np.asarray(w["v"], dtype=np.float64)
        wv[:, :k_fit] -= vs[..., None]
        w["v"] = wv.tolist()
        w["u"] = (np.asarray(w["u"]) - us).tolist()
        w["uj"] = (np.asarray(w["uj"]) - ujs[:, None]).tolist()
    out["objective"] = dict(note="dropped by fold_static: the priors were not re-priced")
    out["fold_static"] = dict(
        source_git=fit.get("git"),
        rule="profile += mean(d)+mean(v); floor_mean += mean(u); z += L^-1 mean(u_j)/sigma_B; "
        "latents minus the same pool means",
        static_d_db=ds.tolist(),
        static_v_db=vs.tolist(),
        static_u_db=us,
        static_uj_db=ujs.tolist(),
        floor_shape_z_before=z.tolist(),
    )
    return out


def cmd_fold(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pool in POOLS:
        path = Path(args.fits) / f"{pool}__flight_v3.json"
        if not path.exists():
            continue
        fit = load_fit(path)
        folded = fold_static(fit)
        folded["fold_static"]["source"] = str(path)
        # check: the fitted block spectra are unchanged and the render carries
        # the static part
        comp_a, comp_b = fit_components(fit), fit_components(folded)
        a, b = ltas_bias(fit, comp_a), ltas_bias(folded, comp_b)
        fit_move = float(np.max(np.abs(np.subtract(a["band_db"]["fit"], b["band_db"]["fit"]))))
        render_vs = np.round(b["render_minus_fit"]["abs_db"], 2).tolist()
        folded["fold_static"]["check"] = dict(
            fit_block_mean_band_move_db=fit_move,
            render_minus_fit_abs_db_before=a["render_minus_fit"]["abs_db"],
            render_minus_fit_abs_db_after=b["render_minus_fit"]["abs_db"],
        )
        dst = out_dir / path.name
        dst.write_text(json.dumps(folded))
        print(
            f"{pool}: fit block-mean band move {fit_move:.2e} dB; render-fit after fold {render_vs}"
        )
        print(f"  wrote {dst}")


# ── 1(a)-(d): model sanity, tilt test, prior against posterior ─────────────

ORDER_GROUPS = ((1, 2), (3, 8), (9, 24), (25, 60), (61, 999))


def pooled_latents(fit: dict[str, Any]) -> dict[str, Any]:
    wins = fit["latents"]["windows"]
    return dict(
        windows=wins,
        d=np.concatenate([np.asarray(w["d"], dtype=np.float64) for w in wins], axis=1),
        v=np.concatenate([np.asarray(w["v"], dtype=np.float64) for w in wins], axis=2),
        u=np.concatenate([np.asarray(w["u"], dtype=np.float64) for w in wins]),
        uj=np.concatenate([np.asarray(w["uj"], dtype=np.float64) for w in wins], axis=1),
    )


def ou_quad(x: np.ndarray, sigma: Any, rho: Any) -> np.ndarray:
    """Per-track quadratic part of the stationary OU ``-log p`` (nats), last
    axis = blocks; ``sigma`` 0 tracks (pinned) give 0."""
    x = np.asarray(x, dtype=np.float64)
    s = np.broadcast_to(np.asarray(sigma, dtype=np.float64), x.shape[:-1])
    r = np.broadcast_to(np.asarray(rho, dtype=np.float64), x.shape[:-1])
    innov = x[..., 1:] - r[..., None] * x[..., :-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 0.5 * (x[..., 0] ** 2 / s**2 + (innov**2).sum(axis=-1) / (s**2 * (1.0 - r**2)))
    return np.where(s > 0.0, q, 0.0)


def ou_energy(fit: dict[str, Any], *, remove_static: bool = False) -> dict[str, float]:
    """Quadratic OU ``-log p`` per family, summed over windows, at the fit's
    own wander; ``remove_static``: every track minus its POOL mean first."""
    from data_processing.noise_model.v3 import Wander

    wander = Wander.from_mapping(fit["params"]["wander"])
    lat = pooled_latents(fit)
    k_fit = lat["v"].shape[1]
    means = {n: lat[n].mean(axis=-1, keepdims=True) for n in ("d", "v", "u", "uj")}
    out = {n: 0.0 for n in ("d", "v", "u", "uj")}
    for w in lat["windows"]:
        for n in out:
            x = np.asarray(w[n], dtype=np.float64)
            if remove_static:
                x = x - means[n]
            sig: Any = wander.track_sigma(n, k_fit)
            rho: Any = wander.track_rho(n, k_fit) if wander.active(n) else 0.0
            if n == "v":
                sig = np.broadcast_to(np.asarray(sig), (k_fit,))[None, :]
                rho = np.broadcast_to(np.asarray(rho), (k_fit,))[None, :]
            out[n] += float(ou_quad(x, sig, rho).sum())
    return out


def _linfit(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return dict(slope=float("nan"), intercept=float("nan"), r2=float("nan"), n=int(x.size))
    a, b = np.polyfit(x, y, 1)
    res = y - (a * x + b)
    tot = float(((y - y.mean()) ** 2).sum())
    return dict(
        slope=float(a),
        intercept=float(b),
        r2=float(1.0 - (res**2).sum() / tot) if tot > 0 else float("nan"),
        n=int(x.size),
        resid_sd=float(res.std()),
    )


def _ctrl_db(z: np.ndarray, fit: dict[str, Any], p: dict[str, Any]) -> np.ndarray:
    from data_processing.noise_model import spectrum as DSP

    return DSP.floor_shape_db(z, sr=int(fit["sr"]), scale_db=float(p["floor"]["floor_shape_sd_db"]))


def _by_group(a: np.ndarray, kk: int) -> dict[str, list[float]]:
    """Per rotor, the mean of ``a[:, k]`` over each order group up to ``kk``."""
    out: dict[str, list[float]] = {}
    for lo, hi in ORDER_GROUPS:
        if lo > kk:
            continue
        out[f"{lo}-{min(hi, kk)}"] = a[:, lo - 1 : min(hi, kk)].mean(axis=1).tolist()
    return out


def _ptp(g: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(np.ptp(v)) for k, v in g.items()}


def sanity(fit: dict[str, Any], fit_r1: dict[str, Any] | None) -> dict[str, Any]:
    """1(a)-(d) for one fit: shapes, per-rotor levels, tilts, prior table."""
    from data_processing.noise_model import spectrum as DSP
    from data_processing.noise_model.v3 import Wander

    p = fit["params"]
    comp = fit_components(fit)
    k_max, rps, freqs = comp["k_max"], comp["rps"], comp["freqs"]
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    n_r, k_all = prof.shape
    p_hat = np.asarray(fit["diagnostics"]["measured"]["profile_centre_db"], dtype=np.float64)
    lat = pooled_latents(fit)
    k_fit = min(k_all, lat["v"].shape[1], k_max)
    ds, vs = lat["d"].mean(axis=1), lat["v"].mean(axis=2)[:, :k_fit]
    us, ujs = float(lat["u"].mean()), lat["uj"].mean(axis=1)
    prof_fold = prof[:, :k_fit] + ds[:, None] + vs
    orders = np.arange(1, k_all + 1)
    # the floor at each line's frequency in PROFILE units: floor / unit line peak
    peak = comp["lines"].max(axis=-1) / 10.0 ** (prof[:, :k_max] / 10.0)
    floor_at = np.interp(orders[:k_max] * rps, freqs, comp["floor"] + comp["wind"])
    floor_prof_db = _db(floor_at[None, :] / peak)
    ctrl_hz = DSP.floor_ctrl_hz(int(fit["sr"]))
    wander = Wander.from_mapping(p["wander"])
    sig_v = np.broadcast_to(np.asarray(wander.track_sigma("v", k_fit), dtype=np.float64), (k_fit,))
    live = sig_v > 0.0

    per_rotor = dict(
        profile=_by_group(prof, k_max),
        profile_folded=_by_group(prof_fold, k_fit),
        p_hat=_by_group(p_hat, k_max),
        static_d_db=ds.tolist(),
        rotor_spread_db=dict(
            profile=_ptp(_by_group(prof, k_max)),
            profile_folded=_ptp(_by_group(prof_fold, k_fit)),
            p_hat=_ptp(_by_group(p_hat, k_max)),
        ),
    )
    logf = np.log2(orders[:k_max] * rps)
    tilt: dict[str, Any] = {}
    for name, arr in (("p_hat", p_hat), ("profile", prof), ("profile_folded", prof_fold)):
        kk = min(arr.shape[1], k_max)
        tilt[name] = dict(
            pooled=_linfit(np.tile(logf[:kk], n_r), arr[:, :kk].ravel()),
            per_rotor=[_linfit(logf[:kk], arr[r, :kk]) for r in range(n_r)],
        )
    tilt["static_v"] = dict(
        pooled_vs_log2f=_linfit(np.tile(logf[:k_fit][live], n_r), vs[:, live].ravel()),
        pooled_vs_k=_linfit(np.tile(orders[:k_fit][live], n_r), vs[:, live].ravel()),
        per_rotor_vs_log2f=[_linfit(logf[:k_fit][live], vs[r, live]) for r in range(n_r)],
        by_group=_by_group(vs, k_fit),
        rms_db=float(np.sqrt((vs[:, live] ** 2).mean())) if live.any() else 0.0,
    )
    tilt["static_d_plus_v"] = dict(
        pooled_vs_log2f=_linfit(np.tile(logf[:k_fit], n_r), (ds[:, None] + vs).ravel())
    )
    tilt["static_uj"] = dict(
        vs_log2f=_linfit(np.log2(ctrl_hz), ujs),
        values_db=ujs.tolist(),
        ctrl_hz=ctrl_hz.tolist(),
        static_u_db=us,
    )
    prom = prof[:, :k_max] - floor_prof_db
    prom_fold = prof_fold - floor_prof_db[:, :k_fit]
    lines_rig = dict(
        prominence_by_group=_by_group(prom, k_max),
        prominence_folded_by_group=_by_group(prom_fold, k_fit),
        share_over_floor=float((prom > 0).mean()),
        share_over_floor_folded=float((prom_fold > 0).mean()),
        p_hat_minus_floor_by_group=_by_group(p_hat[:, :k_max] - floor_prof_db, k_max),
    )

    # (d) prior against posterior, per rig block
    pri = fit["priors"]
    meas = fit["diagnostics"]["measured"]
    g0, gc = float(pri["gamma_hz"]["gamma0_hz"]), float(pri["gamma_hz"]["gamma_c"])
    gam = np.asarray(p["gamma_hz"], dtype=np.float64)
    gz = gam / (g0 * orders[None, : gam.shape[1]])
    prof_sd = float(pri["profile_db"]["sd"])
    dev = prof[:, :k_max] - p_hat[:, :k_max]
    z = np.asarray(p["floor"]["floor_shape_z"], dtype=np.float64)
    sig_nu, nu_scale = float(p["sigma_nu"]), float(pri["sigma_nu"]["scale_rad_s"])
    rows: dict[str, Any] = {}
    rows["gamma/(0.01k)"] = dict(
        prior=f"HalfNormal({gc:g})",
        prior_scale=gc,
        fitted_median=float(np.median(gz)),
        fitted_p90=float(np.quantile(gz, 0.9)),
        fitted_max=float(gz.max()),
        share_over_5=float((gz > 5).mean()),
        rms_over_scale=float(np.sqrt((gz**2).mean()) / gc),
        quad_nats=float((gz**2).sum() / (2 * gc**2)),
        n=int(gz.size),
    )
    rows["sigma_nu"] = dict(
        prior=f"HalfNormal({nu_scale:g} rad/s)",
        prior_scale=nu_scale,
        fitted=sig_nu,
        rms_over_scale=sig_nu / nu_scale,
        quad_nats=sig_nu**2 / (2 * nu_scale**2),
    )
    rows["profile - p_hat"] = dict(
        prior=f"Normal(p_hat, {prof_sd:g} dB)",
        prior_scale=prof_sd,
        fitted_mean=float(dev.mean()),
        fitted_sd=float(dev.std()),
        fitted_min=float(dev.min()),
        fitted_max=float(dev.max()),
        rms_over_scale=float(np.sqrt((dev**2).mean()) / prof_sd),
        by_group_mean=_by_group(dev, k_max),
        quad_nats=float((dev**2).sum() / (2 * prof_sd**2)),
        n=int(dev.size),
    )
    rows["floor z"] = dict(
        prior="Normal(0, I), c = mu + sigma_B L z",
        prior_scale=1.0,
        sigma_b_db=float(p["floor"]["floor_shape_sd_db"]),
        fitted=z.tolist(),
        norm=float(np.linalg.norm(z)),
        rms_over_scale=float(np.sqrt((z**2).mean())),
        ctrl_minus_measured_db=(
            _ctrl_db(z, fit, p) - np.asarray(meas["floor_ctrl_db"], dtype=np.float64)
        ).tolist(),
        quad_nats=float((z**2).sum() / 2),
    )
    if p.get("wind") is not None:
        w = np.asarray(p["wind"]["wind_db"], dtype=np.float64)
        w0 = np.asarray(meas["wind_db"], dtype=np.float64)
        wsd = float(pri["wind"]["sd"])
        rows["wind - measured"] = dict(
            prior=f"Normal(measured, {wsd:g} dB)",
            prior_scale=wsd,
            fitted=(w - w0).tolist(),
            rms_over_scale=float(np.sqrt(((w - w0) ** 2).mean()) / wsd),
            quad_nats=float(((w - w0) ** 2).sum() / (2 * wsd**2)),
        )
    pinned = fit["diagnostics"].get("span_pins", {}).get("pinned", [])
    for name, key, prior_key, log in (
        ("amp_exp", ("profile", "amp_exp"), "amp_exp", False),
        ("floor_exp", ("floor", "floor_exp"), "log_floor_exp", True),
        ("floor_static_rel", ("floor", "floor_static_rel"), "log_floor_static", True),
    ):
        val = float(p[key[0]][key[1]])
        loc, sc = (float(x) for x in pri[prior_key])
        x = math.log(val) if log else val
        rows[name] = dict(
            prior=f"{'LogNormal' if log else 'Normal'}({loc:.3g}, {sc:g})",
            prior_scale=sc,
            fitted=val,
            z=(x - loc) / sc,
            pinned=name in pinned,
            quad_nats=0.0 if name in pinned else (x - loc) ** 2 / (2 * sc**2),
        )
    ou_full, ou_nostatic = ou_energy(fit), ou_energy(fit, remove_static=True)
    ou_rows = {
        n: dict(
            quad_nats=ou_full[n],
            quad_nats_without_static=ou_nostatic[n],
            static_share_of_nats=(1.0 - ou_nostatic[n] / ou_full[n]) if ou_full[n] > 0 else 0.0,
        )
        for n in ou_full
    }
    moves: dict[str, Any] | None = None
    if fit_r1 is not None:
        p1 = fit_r1["params"]
        prof1 = np.asarray(p1["profile"]["profile_db"], dtype=np.float64)
        gam1 = np.asarray(p1["gamma_hz"], dtype=np.float64)
        z1 = np.asarray(p1["floor"]["floor_shape_z"], dtype=np.float64)
        moves = dict(
            r1_source=fit_r1.get("_path"),
            profile_rms_db=float(np.sqrt(((prof - prof1) ** 2).mean())),
            profile_max_abs_db=float(np.abs(prof - prof1).max()),
            log_gamma_rms=float(np.sqrt((np.log(gam / gam1) ** 2).mean())),
            sigma_nu=[float(p1["sigma_nu"]), sig_nu],
            floor_z_rms=float(np.sqrt(((z - z1) ** 2).mean())),
            floor_ctrl_rms_db=float(
                np.sqrt(((_ctrl_db(z, fit, p) - _ctrl_db(z1, fit_r1, p1)) ** 2).mean())
            ),
        )
        if p.get("wind") is not None and p1.get("wind") is not None:
            dw = np.asarray(p["wind"]["wind_db"]) - np.asarray(p1["wind"]["wind_db"])
            moves["wind_rms_db"] = float(np.sqrt((dw**2).mean()))
    return dict(
        rps=rps,
        k_max=k_max,
        shapes=dict(
            profile_db=list(prof.shape),
            gamma_hz=list(gam.shape),
            sigma_nu="scalar",
            floor_shape_z=list(z.shape),
            wind_db=None if p.get("wind") is None else [len(p["wind"]["wind_db"])],
            latents={
                n: list(np.asarray(lat["windows"][0][n]).shape) for n in ("d", "v", "u", "uj")
            },
        ),
        per_rotor=per_rotor,
        tilt=tilt,
        lines_in_rig=lines_rig,
        prior_table=rows,
        ou_prior=ou_rows,
        moves_r1_r2=moves,
        rig_neg_log_prior_nats=fit["objective"].get("rig_neg_log_prior_nats"),
        ou_neg_log_prior_nats=fit["objective"].get("ou_neg_log_prior_nats"),
        arrays=dict(
            profile_db=prof.tolist(),
            p_hat_db=p_hat.tolist(),
            profile_folded_db=prof_fold.tolist(),
            static_v_db=vs.tolist(),
            floor_prof_db=floor_prof_db.tolist(),
            gamma_over_g0k=gz.tolist(),
            log2_freq=logf.tolist(),
        ),
    )


def cmd_sanity(args: argparse.Namespace) -> None:
    out: dict[str, Any] = {}
    for pool in POOLS:
        path = Path(args.fits) / f"{pool}__flight_v3.json"
        if not path.exists():
            continue
        fit = load_fit(path)
        seed = fit.get("restarts", {}).get("selected_seed")
        r1_path = Path(args.fits_r1) / "restarts" / f"{pool}__flight_v3__s{seed}.json"
        fit_r1 = load_fit(r1_path) if r1_path.exists() else None
        if fit_r1 is not None:
            fit_r1["_path"] = str(r1_path)
        rec = sanity(fit, fit_r1)
        rec["fit"] = str(path)
        out[pool] = rec
        print(f"== {pool}")
        print(json.dumps(rec["per_rotor"]))
        for k2, v2 in rec["tilt"].items():
            print(" tilt", k2, json.dumps(v2)[:600])
        print(" lines", json.dumps(rec["lines_in_rig"]))
        for k2, v2 in rec["prior_table"].items():
            print(" prior", k2, json.dumps(v2)[:700])
        print(" ou", json.dumps(rec["ou_prior"]))
        print(" moves", json.dumps(rec["moves_r1_r2"]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


def cmd_figs(args: argparse.Namespace) -> None:
    """Fig 1b (p_hat, the fitted and the folded profile, the rig floor, per
    rotor, against log frequency) and Fig 1c (static v and static u_j against
    log frequency, with their regressions) from ``sanity_r2.json``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = json.loads(Path(args.sanity).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    names = {
        "dregon_room2_floor": "DREGON",
        "michaels_fly125_cruise": "Michael's cruise",
        "michaels_fly125_standby": "Michael's standby",
    }
    pools = [p for p in POOLS if p in s]
    fig, axes = plt.subplots(
        len(pools), 4, figsize=(17, 3.6 * len(pools)), sharex="row", squeeze=False
    )
    for i, pool in enumerate(pools):
        rec = s[pool]
        a = rec["arrays"]
        f = 2.0 ** np.asarray(a["log2_freq"])
        kk = f.size
        for r in range(4):
            ax = axes[i, r]
            ph = np.asarray(a["p_hat_db"])[r, :kk]
            pr = np.asarray(a["profile_db"])[r, :kk]
            pf = np.asarray(a["profile_folded_db"])[r]
            fl = np.asarray(a["floor_prof_db"])[r, :kk]
            ax.plot(f, ph, color="0.6", lw=1.2, label="prior centre p̂ (line + floor)")
            ax.plot(
                f, pr, color="C0", lw=1.0, marker=".", ms=3, label="fitted profile (latents at 0)"
            )
            ax.plot(f[: pf.size], pf, color="C3", lw=1.0, ls="--", label="profile + static d + v")
            ax.plot(f, fl, color="k", lw=1.0, ls=":", label="rig floor at k f_r")
            tl = rec["tilt"]["p_hat"]["per_rotor"][r]
            tp = rec["tilt"]["profile"]["per_rotor"][r]
            ax.set_title(
                f"{names[pool]}, rotor {r + 1}\np̂ {tl['slope']:+.2f} dB/oct (r² {tl['r2']:.2f}); "
                f"fit {tp['slope']:+.2f} (r² {tp['r2']:.2f})",
                fontsize=9,
            )
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            if r == 0:
                ax.set_ylabel("dB (profile units)")
            if i == len(pools) - 1:
                ax.set_xlabel(f"k f_r, Hz (f_r = {rec['rps']:.1f} rev/s)")
        axes[i, 0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "fig_1b_profiles.png", dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, len(pools), figsize=(5.2 * len(pools), 7.5), squeeze=False)
    for i, pool in enumerate(pools):
        rec = s[pool]
        a = rec["arrays"]
        sv = np.asarray(a["static_v_db"])
        lf = np.asarray(a["log2_freq"])[: sv.shape[1]]
        ax = axes[0, i]
        for r in range(sv.shape[0]):
            ax.plot(2.0**lf, sv[r], ".", ms=4, label=f"rotor {r + 1}")
        fitl = rec["tilt"]["static_v"]["pooled_vs_log2f"]
        if np.isfinite(fitl["slope"]):
            xx = np.linspace(lf.min(), lf.max(), 20)
            ax.plot(2.0**xx, fitl["slope"] * xx + fitl["intercept"], "k-", lw=1.5)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(
            f"{names[pool]}: static v (pool mean per line)\n"
            f"{fitl['slope']:+.2f} dB/oct, r² {fitl['r2']:.2f}, rms {rec['tilt']['static_v']['rms_db']:.2f} dB",
            fontsize=9,
        )
        ax.set_xlabel("k f_r, Hz")
        ax.set_ylabel("dB")
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
        ax = axes[1, i]
        uj = rec["tilt"]["static_uj"]
        c = np.asarray(uj["ctrl_hz"])
        ax.plot(c, uj["values_db"], "o-", color="C2", label="static u_j")
        ax.axhline(
            uj["static_u_db"], color="C1", ls="--", label=f"static u = {uj['static_u_db']:.2f} dB"
        )
        ft = uj["vs_log2f"]
        xx = np.log2(c)
        ax.plot(c, ft["slope"] * xx + ft["intercept"], "k-", lw=1.2)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(
            f"{names[pool]}: static u_j per control point\n{ft['slope']:+.2f} dB/oct, r² {ft['r2']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("control point, Hz")
        ax.set_ylabel("dB")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_1c_static_tilt.png", dpi=110)
    plt.close(fig)
    print(f"wrote {out}/fig_1b_profiles.png, fig_1c_static_tilt.png")


RIG_OF_POOL = {
    "dregon_room2_floor": "dregon",
    "michaels_fly125_cruise": "michaels",
    "michaels_fly125_standby": "michaels",
}


def cmd_ltas_bias(args: argparse.Namespace) -> None:
    out: dict[str, Any] = {}
    for pool in POOLS:
        path = Path(args.fits) / f"{pool}__flight_v3.json"
        if not path.exists():
            continue
        fit = load_fit(path)
        measured = json.loads((Path(args.wander_dir) / f"{RIG_OF_POOL[pool]}.json").read_text())
        rec = ltas_bias(fit, measured=measured)
        rec["fit"] = str(path)
        out[pool] = rec
        rm = rec["render_minus_fit"]
        print(
            f"{pool}: rps {rec['rps']:.1f}  render-fit abs {np.round(rm['abs_db'], 2).tolist()}"
            f"  rms-matched {np.round(rm['rms_matched_db'], 2).tolist()}"
        )
        for key in (
            "zero_minus_fit",
            "render_static_minus_fit",
            "render_static_measured_minus_fit",
        ):
            r = rec[key]
            print(
                f"    {key}: abs {np.round(r['abs_db'], 2).tolist()}"
                f"  rms-matched {np.round(r['rms_matched_db'], 2).tolist()}"
            )
        print(
            f"    jensen line {rec['jensen_line_db_by_group']} floor {rec['jensen_floor_db_range']}"
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ltas-bias")
    a.add_argument("--fits", default="results/noise_v3/fits_r2")
    a.add_argument("--out", default="results/noise_v3/diag/ltas_bias_r2.json")
    a.add_argument("--wander-dir", default="results/noise_v3/wander")
    a.set_defaults(func=cmd_ltas_bias)
    f = sub.add_parser("fold")
    f.add_argument("--fits", default="results/noise_v3/fits_r2")
    f.add_argument("--out", default="results/noise_v3/diag/folded_r2")
    f.set_defaults(func=cmd_fold)
    s = sub.add_parser("sanity")
    s.add_argument("--fits", default="results/noise_v3/fits_r2")
    s.add_argument("--fits-r1", default="results/noise_v3/fits")
    s.add_argument("--out", default="results/noise_v3/diag/sanity_r2.json")
    s.set_defaults(func=cmd_sanity)
    g = sub.add_parser("figs")
    g.add_argument("--sanity", default="results/noise_v3/diag/sanity_r2.json")
    g.add_argument("--out", default="results/noise_v3/diag")
    g.set_defaults(func=cmd_figs)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
