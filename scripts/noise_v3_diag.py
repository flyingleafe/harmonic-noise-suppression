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
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
