#!/usr/bin/env python
"""Why the v3 DREGON renders sit 4-6 dB under the real § Listen window in
300 Hz - 5 kHz after the Listen level match, while v2 matches it to 0.5 dB.

Every number is the fits' own forward model against the data it describes, per
band, per mic; nothing is fitted:

* **own windows**: the five 8 s windows of the DREGON pool (``fit["supports"]``,
  loaded as ``noise_v2_fit.py flight --mode flight_v3`` loads them), all 8 mics.
  v3 against the CHANNEL-NORMALISED data (``FT.load_channel_gains``, as the
  pool was built), v2 against the raw data it was fitted on. The data read
  every frame; the model's frame-mean expectation is evaluated on every
  :data:`STRIDE`-th frame's carrier;
* **Listen window**: ``free-flight_nosource_room2`` 50-60 s through the same
  flight-support path (:func:`supports.flight_dregon`), the Listen clip of
  ``scripts/noise_v3_latent_runaway_figs.py --round r3b``;
* v3 three ways: latents at zero (the rig), the RENDER expectation (fresh OU
  wander: ``noise_v3_diag.jensen_factors`` on every line and the floor) and,
  on its own windows, the fit's own FITTED latents (the Whittle optimum);
  v2 as fitted (mic ``m`` of the 8-mic model) and as ``render_noise(n_mics=1)``
  renders mic 0 (gains mean-pinned over one mic, i.e. unity);
* the model split into comb (narrow lines, ``gamma_hz`` < :data:`WIDE_HZ`, and
  wide ones), floor and v3's static per-mic wind;
* the per-window wind level the data asks for: the 30-300 Hz excess of the
  data over r3b's comb + floor, divided by the fixed wind shape;
* **realised**: the Listen WAVs of that run (``docs/explainers/
  noise-model-v3-latent-runaway/audio``, RMS 0.1) with their level gains divided
  back out, i.e. the renders and the real clip in absolute (fit) units;
* Michael's r3b (standby/cruise composition) through the same machinery on its
  Listen window (FLY124 40-50 s) and on its eight FLY125 cruise windows.

Band level = ``10 log10`` of the MEAN power over the band's bins and the frames.
The § Listen LTAS is the mean of dB over 1.95 Hz Welch bins; ``realised``
carries both conventions on the WAVs.

    PYTHONPATH=src python scripts/noise_v3_ltas_gap.py            # compute + write
    PYTHONPATH=src python scripts/noise_v3_ltas_gap.py --plot-only

Writes ``results/noise_v3/diag/ltas_gap_dregon.{json,md,png}``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/noise_v3/diag")
STEM = "ltas_gap_dregon"
FITS = {
    "r3b": "results/noise_v3/fits_r3b/dregon_room2_floor__flight_v3.json",
    "r2": "results/noise_v3/fits_r2/dregon_room2_floor__flight_v3.json",
    "v2": "results/noise_v2/rounds/round5/fits/dregon_room2_floor__flight_profile.json",
}
MICHAELS_R3B = {
    "standby": "results/noise_v3/fits_r3b/michaels_fly125_standby__flight_v3.json",
    "cruise": "results/noise_v3/fits_r3b/michaels_fly125_cruise__flight_v3.json",
}
CHANNEL_GAINS = "results/noise_v2/mic_gains/mic_gains.json"
LISTEN_DIR = Path("docs/explainers/noise-model-v3-latent-runaway")
LISTEN_WAVS = {"real": "real", "v2": "v2", "r2": "v3", "r3b": "v3_r3b"}
#: § Listen windows (``noise_v3_latent_runaway_figs.LISTEN_WINDOWS``), offsets
#: from each recording's first audio sample
LISTEN: dict[str, tuple[str, float]] = {
    "dregon": ("free-flight_nosource_room2", 50.0),
    "michaels": ("FLY124", 40.0),
}
LISTEN_S = 10.0
SR, N_FFT, HOP = 16000, 2048, 512
#: every STRIDE-th frame of a window (hop 512: one frame per 0.38 s)
STRIDE = 12
#: a line wider than one analysis bin (7.8 Hz) is counted as "wide"
WIDE_HZ = 8.0
#: the six § Listen LTAS bands plus the 30-100 Hz part of the fit band below them
BANDS = ((30, 100), (100, 300), (300, 700), (700, 1500), (1500, 3000), (3000, 5000), (5000, 7000))
BAND_LABELS = tuple(f"{lo}-{hi}" for lo, hi in BANDS)
MID = (300.0, 5000.0)
LOW = (100.0, 300.0)
WIND_FIT_HZ = (30.0, 300.0)
C_REAL, C_R3B, C_R2, C_V2 = "#000000", "#d62728", "#ff9896", "#7f7f7f"


def _db(x: Any) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-300))


def _r(x: Any, nd: int = 3) -> Any:
    if isinstance(x, dict):
        return {str(k): _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return _r(x.tolist(), nd)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), nd)
    if isinstance(x, np.integer):
        return int(x)
    return x


FREQS = np.fft.rfftfreq(N_FFT, 1.0 / SR)


def band_db(p: np.ndarray, bands: tuple[tuple[float, float], ...] = BANDS) -> np.ndarray:
    """``(M, n_bands)``: dB of the mean power over frames and bins, ``p`` ``(M, N, F)``."""
    return np.stack(
        [_db(p[..., (lo <= FREQS) & (hi > FREQS)].mean(axis=(-1, -2))) for lo, hi in bands], -1
    )


def span_db(p: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return band_db(p, ((lo, hi),))[..., 0]


# ── data ───────────────────────────────────────────────────────────────────


def window_data(spec: Any) -> dict[str, Any]:
    """One flight window: ``(M, N, F)`` power on every :data:`STRIDE`-th frame,
    its frame starts and the ``(R, T)`` carrier."""
    from experiments.noise_model import supports as SUP

    s = SUP.load_support(spec)
    starts = np.asarray(s.frame_starts, dtype=np.int64)
    sel = np.arange(0, starts.size, STRIDE)
    return dict(
        name=s.name,
        power=np.asarray(s.power, dtype=np.float64)[:, sel, :],
        power_all=np.asarray(s.power, dtype=np.float64),
        starts=starts[sel],
        carrier=np.asarray(s.carrier_rev_s_audio, dtype=np.float64),
    )


def pool_windows(fit: dict[str, Any], set_name: str) -> list[Any]:
    from experiments.noise_model import supports as SUP

    specs = {s.name: s for s in SUP.support_set(set_name)}
    return [specs[n] for n in fit["supports"]]


def listen_spec(rig: str) -> Any:
    """The § Listen window as a flight support spec on the recording clock."""
    from experiments.noise_model import supports as SUP
    from experiments.stochastic_fit import clips as C

    recording, offset_s = LISTEN[rig]
    if rig == "dregon":
        rec = C.load_recording(SUP.DREGON_DATASET, recording, None, SUP.DREGON_RPS_KEY)
        return SUP.flight_dregon(recording, rec.t_start + offset_s, LISTEN_S)
    return SUP.flight_michaels(recording, offset_s, LISTEN_S)


# ── model ──────────────────────────────────────────────────────────────────


def _diag() -> Any:
    """``scripts/noise_v3_diag.py`` (its :func:`jensen_factors`)."""
    import importlib.util
    import sys

    path = Path(__file__).resolve().parent / "noise_v3_diag.py"
    spec = importlib.util.spec_from_file_location("noise_v3_diag", str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["noise_v3_diag"] = mod
    spec.loader.exec_module(mod)
    return mod


def components(
    fit: dict[str, Any],
    carrier: np.ndarray,
    starts: np.ndarray,
    variants: dict[str, dict[str, Any]],
    *,
    n_mics: int = 8,
) -> dict[str, dict[str, np.ndarray]]:
    """``{variant: {part: (M, N, F)}}``: expected periodogram parts of ``fit``
    on the frames at ``starts`` — ``comb_narrow``, ``comb_wide``, ``floor``,
    ``wind`` — with the transfer and the per-mic gains the forward model
    applies, so at latents zero their sum is :func:`spectrum.flight_model`.

    A variant's options: ``latents`` (one window's fitted tracks ``{d, v, u,
    uj}`` indexed by each frame's ``block``, as :func:`model.forward_v3` applies
    them); ``jensen`` (the RENDER expectation: every line times ``E[10^{x/10}]``
    of fresh OU tracks of the fit's wander and the floor times its own,
    ``noise_v3_diag.jensen_factors``, as ``noise_v3_diag.ltas_bias`` does);
    ``single_mic`` (a v2 payload as ``render_noise(n_mics=1)`` renders mic 0:
    line and overall gains mean-pinned over that ONE mic, i.e. unity, the floor
    gain ``mic_floor_db`` absolute)."""
    import torch

    from data_processing.noise_model import spectrum as DSP
    from data_processing.noise_model.floor import floor_geometry
    from data_processing.noise_model.v3 import Wander
    from experiments.noise_model import model as MD
    from experiments.noise_model import render as RD
    from experiments.noise_model import spectrum as SP

    p = fit["params"]
    grid = SP.flight_grid(sr=SR, n_fft=N_FFT, hop=HOP, sr_work=RD.fit_work_rate(fit))
    params = MD.params_from_dict(p, n_mics=n_mics)
    k_prof = int(np.asarray(p["profile"]["profile_db"]).shape[1])
    kk = min(k_prof, SP.k_max_for_carrier(carrier.max(axis=1), SR, k_cap=k_prof))
    rate = SP.flight_rate_work(grid, carrier, starts)
    out: dict[str, dict[str, np.ndarray]] = {}
    with torch.no_grad():
        cache = SP.flight_cache(grid, params, rate_work=rate, k_max=kk)
        lines = cache.lines  # (R, N, K, F)
        n = lines.shape[1]
        wide = torch.as_tensor(np.asarray(p["gamma_hz"], dtype=np.float64)[:, :kk] >= WIDE_HZ)
        transfer = grid.transfer_power.numpy()[None, None, :]
        for vname, opt in variants.items():
            mult = torch.ones(lines.shape[:3], dtype=lines.dtype)
            floor = cache.floor0  # (N, F)
            lat = opt.get("latents")
            if lat is not None:
                b = lat["block"]
                ld = lat["d"][:, b][:, :, None] + np.transpose(lat["v"][:, :kk, b], (0, 2, 1))
                mult = torch.as_tensor(10.0 ** (ld / 10.0), dtype=lines.dtype)
                fdb = lat["u"][b][:, None] + lat["uj"][:, b].T @ grid.floor.shape_psd.numpy().T
                c = grid.floor.autocovariance_db(cache.floor_db[None, :] + torch.as_tensor(fdb))
                floor = SP.floor_frames_from_autocov(grid, cache.floor_g1, c)
            if opt.get("jensen"):
                ctrl = DSP.floor_ctrl_hz(SR)
                basis = floor_geometry(FREQS, ctrl)[0].T  # (J, F)
                jl, jf = _diag().jensen_factors(Wander.from_mapping(p["wander"]), kk, basis, ctrl)
                mult = mult * torch.as_tensor(jl, dtype=mult.dtype)[None, None, :]
                floor = floor * torch.as_tensor(jf, dtype=floor.dtype)[None, :]
            line_gain, all_gain = cache.line_gain, cache.all_gain
            if opt.get("single_mic"):
                line_gain, all_gain = torch.ones_like(line_gain), torch.ones_like(all_gain)
            tg = transfer * all_gain.numpy()[:, None, None]
            parts: dict[str, np.ndarray] = {}
            for name, mask in (("comb_narrow", ~wide), ("comb_wide", wide)):
                m = (mult * mask[:, None, :].to(mult.dtype)).unsqueeze(-2)  # (R, N, 1, K)
                shapes = torch.matmul(m, lines).squeeze(-2)  # (R, N, F)
                parts[name] = torch.einsum("mr,rnf->mnf", line_gain, shapes).numpy() * tg
            parts["floor"] = (floor[None] * cache.mic_floor[:, None, None]).numpy() * tg
            parts["wind"] = (
                np.broadcast_to(cache.wind[:, None, :].numpy(), (n_mics, n, FREQS.size)) * tg
                if cache.wind is not None
                else np.zeros((n_mics, n, FREQS.size))
            )
            parts["comb"] = parts["comb_narrow"] + parts["comb_wide"]
            parts["total"] = parts["comb"] + parts["floor"] + parts["wind"]
            if p.get("wind"):
                w_db = np.asarray(p["wind"]["wind_db"], dtype=np.float64)[:n_mics, None, None]
                parts["wind_unit"] = parts["wind"] / 10.0 ** (w_db / 10.0)
            out[vname] = parts
    return out


def window_latents(fit: dict[str, Any], name: str, starts: np.ndarray) -> dict[str, np.ndarray]:
    lat = next(w for w in fit["latents"]["windows"] if w["name"] == name)
    block_s = float(fit["latents"]["block_s"])
    nb = int(lat["n_blocks"])
    block = np.minimum(((starts + 0.5 * N_FFT) / SR / block_s).astype(np.int64), nb - 1)
    return dict(
        block=block, **{k: np.asarray(lat[k], dtype=np.float64) for k in ("d", "v", "u", "uj")}
    )


def blend(parts: dict[str, dict[str, Any]], weight: np.ndarray) -> dict[str, np.ndarray]:
    """Michael's standby/cruise composition: ``(1 - w) standby + w cruise`` per frame."""
    w = weight[None, :, None]
    keys = ("comb_narrow", "comb_wide", "comb", "floor", "wind", "total")
    return {k: (1.0 - w) * parts["standby"][k] + w * parts["cruise"][k] for k in keys}


# ── per-window summaries ───────────────────────────────────────────────────


def levels(parts: dict[str, np.ndarray]) -> dict[str, Any]:
    """Band levels (dB) of every part, per mic and mic-mean."""
    out: dict[str, Any] = {}
    for k in ("total", "comb", "comb_narrow", "comb_wide", "floor", "wind"):
        if k in parts:
            out[k] = dict(mic=band_db(parts[k]), mean=band_db(parts[k].mean(0, keepdims=True))[0])
    return out


def contrast(p: np.ndarray) -> np.ndarray:
    """``(M,)`` 100-300 Hz level minus 300 Hz - 5 kHz level (dB)."""
    return span_db(p, *LOW) - span_db(p, *MID)


def wind_hat(data: np.ndarray, parts: dict[str, np.ndarray]) -> np.ndarray:
    """``(M,)`` dB: the flat wind level that makes the fit's comb + floor + wind
    match the data's 30-300 Hz POWER, per mic (frame means, ratio of band sums;
    NaN where comb + floor alone already exceed the data)."""
    s = (WIND_FIT_HZ[0] <= FREQS) & (WIND_FIT_HZ[1] > FREQS)
    fm = lambda a: a.mean(axis=-2)[..., s].sum(axis=-1)  # noqa: E731
    excess = fm(data) - fm(parts["comb"] + parts["floor"])
    return np.where(excess > 0, _db(np.maximum(excess, 1e-300) / fm(parts["wind_unit"])), np.nan)


def summarise(
    data_raw: np.ndarray, gains_db: np.ndarray, models: dict[str, dict[str, np.ndarray]]
) -> dict[str, Any]:
    """Data levels and model-minus-data deviations of one window. v3 models
    are compared with the channel-normalised data, v2 with the raw data."""
    norm = data_raw / 10.0 ** (gains_db[:, None, None] / 10.0)
    data = dict(
        raw=dict(mic=band_db(data_raw), mean=band_db(data_raw.mean(0, keepdims=True))[0]),
        norm=dict(mic=band_db(norm), mean=band_db(norm.mean(0, keepdims=True))[0]),
    )
    out: dict[str, Any] = dict(
        data=data,
        mic_minus_mean_norm_db=data["norm"]["mic"] - data["norm"]["mean"][None, :],
        contrast_db=dict(data_norm=contrast(norm), data_raw=contrast(data_raw)),
        models={},
        dev_db={},
    )
    for key, parts in models.items():
        ref = data["raw"] if key.startswith("v2") else data["norm"]
        lv = levels(parts)
        out["models"][key] = lv
        out["dev_db"][key] = dict(
            mic=lv["total"]["mic"] - ref["mic"], mean=lv["total"]["mean"] - ref["mean"]
        )
        out["contrast_db"][key] = contrast(parts["total"])
        if parts.get("wind_unit") is not None:
            out.setdefault("wind_hat_db", {})[key] = wind_hat(norm, parts)
    return out


def level_match(model: np.ndarray, data: np.ndarray, lo: float, hi: float) -> float:
    """dB the model sits above the data in frame-mean power over ``[lo, hi)``."""
    s = (lo <= FREQS) & (hi > FREQS)
    return float(_db(model.mean(axis=-2)[..., s].sum()) - _db(data.mean(axis=-2)[..., s].sum()))


# ── realised: the § Listen WAVs ────────────────────────────────────────────


def realised(gain0_db: float, support_real_raw_db: np.ndarray) -> dict[str, Any]:
    """The Listen WAVs (mic 0) in absolute units: each clip's level gain divided
    out, the real clip also divided by mic 0's channel gain so every v3 row is in
    the fit's normalised units (v2's are raw: compare it with ``real_raw``).
    ``support_real_raw_db``: the same window's mic 0 through the fit's
    flight-support path, to check the two routes agree."""
    import soundfile as sf

    from experiments.stochastic_fit import accept_stats
    from experiments.stochastic_fit import revised_eval as RE
    from experiments.stochastic_fit.data import Clip

    six = [i for i, (lo, hi) in enumerate(accept_stats.BANDS) if lo >= 100 and hi <= 7000]

    fd = json.loads((LISTEN_DIR / "figdata_r3b.json").read_text())["listen"]["rigs"]["dregon"]
    clips = fd["clips"]
    out: dict[str, Any] = dict(
        level_gain={}, band_db={}, ge300_db={}, ge500_db={}, total_db={}, ltas_meandb={}
    )
    for key, name in LISTEN_WAVS.items():
        x, sr = sf.read(LISTEN_DIR / "audio" / f"dregon_{name}.wav")
        g = float(clips[name]["level_gain"])
        x = np.asarray(x, dtype=np.float64) / g
        out["level_gain"][key] = g
        rps = np.full((4, x.size), 80.0)
        pg = RE.window_periodogram(
            Clip("wav", "wav", x[None].astype(np.float32), rps, int(sr)),
            n_fft=N_FFT,
            hop=HOP,
        )
        p = np.asarray(pg.power, dtype=np.float64)
        out["band_db"][key] = band_db(p)[0]
        out["ge300_db"][key] = float(_db(p.mean(1)[..., (FREQS >= 300) & (FREQS < 7900)].sum()))
        out["ge500_db"][key] = float(_db(p.mean(1)[..., (FREQS >= 500) & (FREQS < 7900)].sum()))
        out["total_db"][key] = float(_db(p.mean(1).sum()))
        # the § Listen convention: mean of dB over 1.95 Hz Welch bins, six bands
        out["ltas_meandb"][key] = RE.absolute_ltas_bands(x)[six]
        if key == "real":
            for k in ("band_db", "ltas_meandb"):
                out[k]["real_raw"] = out[k]["real"]
                out[k]["real"] = out[k]["real"] - gain0_db
            for k in ("ge300_db", "ge500_db", "total_db"):
                out[k]["real_raw"] = out[k]["real"]
                out[k]["real"] = out[k]["real"] - gain0_db
    out["support_minus_wav_real_raw_db"] = support_real_raw_db - out["band_db"]["real_raw"]
    for k in (
        "dev_abs_db",
        "dev_full_rms_db",
        "dev_ge300_db",
        "dev_ge500_db",
        "ltas_meandb_dev_abs_db",
    ):
        out[k] = {}
    for k in LISTEN_WAVS:
        if k == "real":
            continue
        ref = "real_raw" if k == "v2" else "real"
        dev = out["band_db"][k] - out["band_db"][ref]
        out["dev_abs_db"][k] = dev
        out["dev_full_rms_db"][k] = dev - (out["total_db"][k] - out["total_db"][ref])
        out["dev_ge300_db"][k] = dev - (out["ge300_db"][k] - out["ge300_db"][ref])
        out["dev_ge500_db"][k] = dev - (out["ge500_db"][k] - out["ge500_db"][ref])
        out["ltas_meandb_dev_abs_db"][k] = out["ltas_meandb"][k] - out["ltas_meandb"][ref]
    out["figdata_ltas_dev_db"] = {
        k: clips[n].get("ltas_dev_db") for k, n in LISTEN_WAVS.items() if k != "real"
    }
    return out


# ── compute ────────────────────────────────────────────────────────────────


def listen_mic0(
    models: dict[str, dict[str, np.ndarray]], raw0: np.ndarray, gain0_db: float
) -> dict[str, Any]:
    """Mic 0 of the Listen window: every model's parts and its deviation under
    three level rules — none (absolute), the Listen's full-band RMS match, and
    a match on 300 Hz - 7.9 kHz power. v2 rows against the raw data."""
    norm0 = raw0 / 10.0 ** (gain0_db / 10.0)

    def low(a: np.ndarray) -> float:
        return float(a.mean(-2)[..., FREQS < 300].sum() / a.mean(-2).sum())

    out: dict[str, Any] = {}
    for key, parts in models.items():
        d0 = raw0 if key.startswith("v2") else norm0
        m0 = parts["total"][:1]
        dev = band_db(m0)[0] - band_db(d0)[0]
        full = level_match(m0, d0, 0.0, SR / 2.0)
        ge300 = level_match(m0, d0, 300.0, 7900.0)
        ge500 = level_match(m0, d0, 500.0, 7900.0)
        out[key] = dict(
            dev_abs_db=dev,
            shift_full_db=full,
            dev_full_rms_db=dev - full,
            shift_ge300_db=ge300,
            dev_ge300_db=dev - ge300,
            shift_ge500_db=ge500,
            dev_ge500_db=dev - ge500,
            parts_minus_data_db={
                k: band_db(parts[k][:1])[0] - band_db(d0)[0]
                for k in ("comb", "comb_narrow", "comb_wide", "floor", "wind")
                if k in parts and float(np.max(parts[k][:1])) > 0.0
            },
            power_share_below_300=dict(model=low(m0), data=low(d0)),
        )
    return out


def compute_dregon() -> dict[str, Any]:
    from experiments.noise_model import fit as FT

    fits = {k: json.loads(Path(v).read_text()) for k, v in FITS.items()}
    gains = np.asarray(
        FT.load_channel_gains(CHANNEL_GAINS, rig="dregon").gains_db, dtype=np.float64
    )
    specs = [(s, "pool") for s in pool_windows(fits["r3b"], "dregon-floor")]
    specs.append((listen_spec("dregon"), "listen"))
    v2p = fits["v2"]["params"]
    out: dict[str, Any] = dict(
        fits=FITS,
        channel_gains_db=gains,
        bands_hz=[list(b) for b in BANDS],
        stride=STRIDE,
        wide_hz=WIDE_HZ,
        wind_db={
            k: f["params"]["wind"]["wind_db"] for k, f in fits.items() if f["params"].get("wind")
        },
        wind_db_measured_centre=fits["r3b"]["diagnostics"]["measured"]["wind_db"],
        floor_exp={k: f["params"]["floor"]["floor_exp"] for k, f in fits.items()},
        amp_exp={k: f["params"]["profile"]["amp_exp"] for k, f in fits.items()},
        v2_mic_db=dict(
            mic_gains_db=v2p["mic_gains_db"],
            mic_floor_db=v2p["floor"]["mic_floor_db"],
            mic_line_gain_db=v2p["profile"]["mic_line_gain_db"],
            floor_tilt_db_oct=v2p["floor"]["floor_tilt_db_oct"],
        ),
        windows={},
    )
    for spec, role in specs:
        t0 = time.time()
        w = window_data(spec)
        models: dict[str, dict[str, np.ndarray]] = {}
        for key in ("r3b", "r2"):
            var: dict[str, dict[str, Any]] = {"zero": {}, "render": {"jensen": True}}
            if key == "r3b" and role == "pool":
                var["latents"] = {"latents": window_latents(fits[key], w["name"], w["starts"])}
            got = components(fits[key], w["carrier"], w["starts"], var)
            models[key], models[f"{key}_render"] = got["zero"], got["render"]
            if "latents" in got:
                models[f"{key}_latents"] = got["latents"]
        got = components(
            fits["v2"], w["carrier"], w["starts"], {"zero": {}, "n1": {"single_mic": True}}
        )
        models["v2"], models["v2_n1"] = got["zero"], got["n1"]
        # data: EVERY frame of the window; the model's frame-mean expectation is
        # read on every STRIDE-th frame's carrier
        rec = summarise(w["power_all"], gains, models)
        rec.update(
            role=role,
            n_frames_model=int(w["starts"].size),
            n_frames_data=int(w["power_all"].shape[1]),
            carrier_mean_rev_s=float(w["carrier"].mean()),
            carrier_range_rev_s=[float(w["carrier"].min()), float(w["carrier"].max())],
            data_all_frames_minus_subsample_db=band_db(w["power_all"].mean(0, keepdims=True))[0]
            - band_db(w["power"].mean(0, keepdims=True))[0],
        )
        if role == "listen":
            raw0 = w["power_all"][:1]
            norm0 = raw0 / 10.0 ** (gains[0] / 10.0)
            r3 = models["r3b_render"]
            asks = float(rec["wind_hat_db"]["r3b_render"][0])
            # the wind each OWN window asks for at mic 0, power-averaged (a window
            # whose comb + floor already exceed its data asks for none)
            own = [
                r["wind_hat_db"]["r3b_render"][0]
                for r in out["windows"].values()
                if r["role"] == "pool"
            ]
            pool_mean = float(
                _db(np.mean([10.0 ** (a / 10.0) if np.isfinite(a) else 0.0 for a in own]))
            )
            rec["wind_pool_power_mean_mic0_db"] = pool_mean
            # counterfactuals: r3b's render expectation with mic 0's wind at the
            # own-window power mean, at the level THIS window asks for, and none
            cf = {
                "r3b_render_wind_pool_mean": dict(
                    total=r3["comb"] + r3["floor"] + r3["wind_unit"] * 10.0 ** (pool_mean / 10.0)
                ),
                "r3b_render_wind_at_window": dict(
                    total=r3["comb"] + r3["floor"] + r3["wind_unit"] * 10.0 ** (asks / 10.0)
                ),
                "r3b_render_no_wind": dict(total=r3["comb"] + r3["floor"]),
            }
            lm = listen_mic0({**models, **cf}, raw0, float(gains[0]))
            # the channel-normalised MIC MEAN, for the mic-convention question
            norm_mean = (w["power_all"] / 10.0 ** (gains[:, None, None] / 10.0)).mean(
                0, keepdims=True
            )
            for key in ("r3b_render", "v2"):
                m = models[key]["total"].mean(0, keepdims=True)
                d = w["power_all"].mean(0, keepdims=True) if key == "v2" else norm_mean
                dev = band_db(m)[0] - band_db(d)[0]
                lm[f"{key}_mic_mean"] = dict(
                    dev_abs_db=dev,
                    dev_full_rms_db=dev - level_match(m, d, 0.0, SR / 2.0),
                    dev_ge300_db=dev - level_match(m, d, 300.0, 7900.0),
                    dev_ge500_db=dev - level_match(m, d, 500.0, 7900.0),
                )
            rec["listen_mic0"] = lm
            rec["wind_asks_mic0_db"] = asks
            rec["spectra_mic0_db"] = dict(
                freqs_hz=FREQS,
                data_norm=_db(norm0[0].mean(0)),
                data_raw=_db(raw0[0].mean(0)),
                **{
                    f"{key}_{k}": _db(models[key][k][0].mean(0))
                    for key in ("r3b_render", "v2", "v2_n1")
                    for k in ("total", "comb", "floor", "wind")
                    if not (key.startswith("v2") and k == "wind")
                },
            )
            rz = realised(float(gains[0]), band_db(raw0)[0])
            expected = {"r3b": "r3b_render", "r2": "r2_render", "v2": "v2_n1"}
            rz["expected_minus_realised_db"] = {
                k: np.asarray(lm[m]["dev_abs_db"]) - np.asarray(rz["dev_abs_db"][k])
                for k, m in expected.items()
            }
            out["realised"] = rz
        out["windows"][w["name"]] = rec
        print(f"{role} {w['name']}: {time.time() - t0:.0f} s", flush=True)
    return out


def compute_michaels() -> dict[str, Any]:
    """Michael's r3b composition on its Listen window and its cruise pool."""
    from experiments.noise_model import fit as FT
    from experiments.noise_model import render as RD

    fits = {k: json.loads(Path(v).read_text()) for k, v in MICHAELS_R3B.items()}
    gains = np.asarray(
        FT.load_channel_gains(CHANNEL_GAINS, rig="michaels").gains_db, dtype=np.float64
    )
    specs = [(s, "pool") for s in pool_windows(fits["cruise"], "michaels-cruise")]
    specs.append((listen_spec("michaels"), "listen"))
    out: dict[str, Any] = dict(fits=MICHAELS_R3B, channel_gains_db=gains, windows={})
    variants: dict[str, dict[str, Any]] = {"zero": {}, "render": {"jensen": True}}
    zeros = {k: 0.0 for k in ("comb_narrow", "comb_wide", "comb", "floor", "wind", "total")}
    for spec, role in specs:
        w = window_data(spec)
        wt = RD.regime_blend_weight(w["carrier"], sr=SR)[
            np.clip(w["starts"] + N_FFT // 2, 0, w["carrier"].shape[1] - 1)
        ]
        per = {
            k: (
                components(f, w["carrier"], w["starts"], variants)
                if bool(np.any((wt if k == "cruise" else 1.0 - wt) > 0.0))
                else {v: zeros for v in variants}
            )
            for k, f in fits.items()
        }
        models = {
            ("r3b" if v == "zero" else "r3b_render"): blend({k: per[k][v] for k in per}, wt)
            for v in variants
        }
        rec = summarise(w["power_all"], gains, models)
        rec.update(
            role=role,
            cruise_weight_mean=float(wt.mean()),
            carrier_mean_rev_s=float(w["carrier"].mean()),
        )
        if role == "listen":
            rec["listen_mic0"] = listen_mic0(models, w["power_all"][:1], float(gains[0]))
        out["windows"][w["name"]] = rec
        print(f"michaels {role} {w['name']}", flush=True)
    return out


# ── report ─────────────────────────────────────────────────────────────────


def _row(vals: Any, nd: int = 1) -> str:
    return " | ".join(
        "—" if v is None or not np.isfinite(v) else f"{v:+.{nd}f}".replace("-", "−") for v in vals
    )


def _hdr(first: str, cols: tuple[str, ...] = BAND_LABELS) -> str:
    return f"| {first} | " + " | ".join(cols) + " |\n|---" + "|---:" * len(cols) + "|\n"


def short(name: str) -> str:
    return (
        name.split("_nosource")[0].removeprefix("flight_dregon_").removeprefix("flight_michaels_")
    )


def _pool_avg(wins: dict[str, Any], pool: list[str], key: str, mic: str | int) -> np.ndarray:
    return np.mean(
        [
            wins[n]["dev_db"][key]["mean"]
            if mic == "mean"
            else np.asarray(wins[n]["dev_db"][key]["mic"])[mic]
            for n in pool
        ],
        axis=0,
    )


def headline(d: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    """The key per-band rows: fit's own windows vs the Listen window, r3b vs v2."""
    wins = d["dregon"]["windows"]
    pool = [n for n, r in wins.items() if r["role"] == "pool"]
    lis = next(n for n, r in wins.items() if r["role"] == "listen")
    L, lm = wins[lis], wins[lis]["listen_mic0"]
    rz = d["dregon"]["realised"]
    return [
        (
            "r3b render-exp., own windows (avg of 5), mic mean",
            _pool_avg(wins, pool, "r3b_render", "mean"),
        ),
        ("r3b render-exp., own windows, mic 0", _pool_avg(wins, pool, "r3b_render", 0)),
        ("r3b render-exp., Listen, mic mean", np.asarray(L["dev_db"]["r3b_render"]["mean"])),
        ("r3b render-exp., Listen, mic 0 (absolute)", np.asarray(lm["r3b_render"]["dev_abs_db"])),
        (
            "r3b render-exp., Listen, mic 0, full-band RMS match",
            np.asarray(lm["r3b_render"]["dev_full_rms_db"]),
        ),
        (
            "r3b realised WAV, Listen, mic 0, full-band RMS match",
            np.asarray(rz["dev_full_rms_db"]["r3b"]),
        ),
        (
            "r3b render-exp., Listen, mic 0, wind = own-window power mean, RMS match",
            np.asarray(lm["r3b_render_wind_pool_mean"]["dev_full_rms_db"]),
        ),
        (
            "r3b render-exp., Listen, mic 0, wind at this window's level, RMS match",
            np.asarray(lm["r3b_render_wind_at_window"]["dev_full_rms_db"]),
        ),
        (
            "r3b render-exp., Listen, mic 0, ≥300 Hz match",
            np.asarray(lm["r3b_render"]["dev_ge300_db"]),
        ),
        (
            "r3b render-exp., Listen, mic 0, ≥500 Hz match (above the wind)",
            np.asarray(lm["r3b_render"]["dev_ge500_db"]),
        ),
        ("r3b realised WAV, Listen, mic 0, ≥500 Hz match", np.asarray(rz["dev_ge500_db"]["r3b"])),
        (
            "r3b render-exp., Listen, mic mean, full-band RMS match",
            np.asarray(lm["r3b_render_mic_mean"]["dev_full_rms_db"]),
        ),
        ("v2 fit, own windows (avg of 5), mic mean", _pool_avg(wins, pool, "v2", "mean")),
        ("v2 fit, own windows, mic 0", _pool_avg(wins, pool, "v2", 0)),
        ("v2 fit, Listen, mic mean", np.asarray(L["dev_db"]["v2"]["mean"])),
        ("v2 fit, Listen, mic 0 (absolute)", np.asarray(lm["v2"]["dev_abs_db"])),
        ("v2 fit, Listen, mic 0, full-band RMS match", np.asarray(lm["v2"]["dev_full_rms_db"])),
        (
            "v2 fit, Listen, mic mean, full-band RMS match",
            np.asarray(lm["v2_mic_mean"]["dev_full_rms_db"]),
        ),
        (
            "v2 as rendered (n_mics=1), Listen, mic 0 (absolute)",
            np.asarray(lm["v2_n1"]["dev_abs_db"]),
        ),
        (
            "v2 as rendered (n_mics=1), Listen, mic 0, full-band RMS match",
            np.asarray(lm["v2_n1"]["dev_full_rms_db"]),
        ),
        (
            "v2 realised WAV, Listen, mic 0, full-band RMS match",
            np.asarray(rz["dev_full_rms_db"]["v2"]),
        ),
    ]


def report(d: dict[str, Any]) -> str:
    dr = d["dregon"]
    wins = dr["windows"]
    pool = [n for n, r in wins.items() if r["role"] == "pool"]
    lis = next(n for n, r in wins.items() if r["role"] == "listen")
    L = wins[lis]
    lm = L["listen_mic0"]
    rz = dr["realised"]
    s: list[str] = [FINDINGS.strip(), "\n\n## Tables\n\n"]
    s.append(
        "Band level = dB of the mean power over the band's bins and frames (2048/512 at "
        "16 kHz; data on every frame, the model's expectation on every "
        f"{STRIDE}th frame's carrier). Model − data. v3 against the channel-normalised data, "
        "v2 against the raw data it was fitted on. `zero` = latents at zero (the rig), "
        "`render` = the render expectation (fresh OU wander, `noise_v3_diag.jensen_factors`), "
        "`latents` = the fit's own fitted latents. `v2_n1` = the v2 payload as "
        "`render_noise(n_mics=1)` renders mic 0 (the § Listen clip). Bands in Hz.\n\n"
    )
    s.append("### T0. Headline (model − data, dB)\n\n")
    s.append(_hdr("what"))
    for lab, v in headline(d):
        s.append(f"| {lab} | {_row(v)} |\n")
    s.append("\n### T1. Every window, mic mean and mic 0 (model − data, dB)\n\n")
    s.append(_hdr("window / model / mic"))
    for key in ("r3b_render", "r3b", "r3b_latents", "r2_render", "v2"):
        for mic in ("mean", 0):
            for n in pool + [lis]:
                dv = wins[n]["dev_db"].get(key)
                if dv is None:
                    continue
                v = dv["mean"] if mic == "mean" else np.asarray(dv["mic"])[0]
                s.append(f"| {'Listen' if n == lis else short(n)} / {key} / {mic} | {_row(v)} |\n")
            s.append(f"| **own avg / {key} / {mic}** | {_row(_pool_avg(wins, pool, key, mic))} |\n")
    s.append("\n### T2. Per mic, own-window average vs Listen (model − data, dB)\n\n")
    s.append(_hdr("model / mic / set"))
    for key in ("r3b_render", "v2"):
        for m in range(8):
            s.append(f"| {key} / {m} / own avg | {_row(_pool_avg(wins, pool, key, m))} |\n")
            s.append(f"| {key} / {m} / Listen | {_row(np.asarray(L['dev_db'][key]['mic'])[m])} |\n")
    s.append("\n### T3. Wind: fitted per-mic level vs the level each window asks for (dB)\n\n")
    s.append(
        '"asks for" = 30–300 Hz frame-mean power of the data minus r3b\'s render-expected '
        "comb + floor, over the wind shape (— : comb + floor already exceed the data).\n\n"
    )
    s.append(_hdr("", tuple(f"mic {m}" for m in range(8))))
    s.append(f"| r3b fitted `wind_db` | {_row(dr['wind_db']['r3b'])} |\n")
    s.append(f"| r2 fitted `wind_db` | {_row(dr['wind_db']['r2'])} |\n")
    s.append(f"| measured prior centre | {_row(dr['wind_db_measured_centre'])} |\n")
    asks = []
    for n in pool + [lis]:
        a = np.asarray(wins[n]["wind_hat_db"]["r3b_render"], dtype=float)
        if n != lis:
            asks.append(a)
        s.append(f"| asks for: {'Listen 50–60 s' if n == lis else short(n)} | {_row(a)} |\n")
    # a window whose comb + floor already exceed the data asks for no wind (zero power)
    lin = np.nan_to_num(10.0 ** (np.asarray(asks) / 10.0), nan=0.0).mean(axis=0)
    pm = np.where(lin > 0.0, _db(lin), np.nan)
    s.append(f"| asks for: own windows, power mean (— = none) | {_row(pm)} |\n")
    s.append("\n### T4. 100–300 Hz level minus 300 Hz–5 kHz level (dB, per mic)\n\n")
    s.append(_hdr("window / what", tuple(f"mic {m}" for m in range(8))))
    for n in pool + [lis]:
        c = wins[n]["contrast_db"]
        for k in ("data_norm", "r3b_render", "v2"):
            s.append(f"| {'Listen' if n == lis else short(n)} / {k} | {_row(c[k])} |\n")
    s.append("\n### T5. Listen window, mic 0: parts and level-match rules (dB)\n\n")
    s.append(_hdr("what"))
    s.append(f"| data, normalised, absolute level | {_row(L['data']['norm']['mic'][0])} |\n")
    s.append(f"| data, raw, absolute level | {_row(L['data']['raw']['mic'][0])} |\n")
    for key in ("r3b_render", "r2_render", "v2", "v2_n1"):
        for part, v in lm[key]["parts_minus_data_db"].items():
            v = np.asarray(v, dtype=float)
            s.append(f"| {key} {part} − data | {_row(np.where(v < -100.0, np.nan, v))} |\n")
    for key in (
        "r3b_render",
        "r3b",
        "r2_render",
        "v2",
        "v2_n1",
        "r3b_render_wind_pool_mean",
        "r3b_render_wind_at_window",
        "r3b_render_no_wind",
        "r3b_render_mic_mean",
        "v2_mic_mean",
    ):
        for rule in ("dev_abs_db", "dev_full_rms_db", "dev_ge300_db", "dev_ge500_db"):
            s.append(f"| {key}: {rule.removesuffix('_db')} | {_row(lm[key][rule])} |\n")
    s.append(
        "\nLevel-match shifts (model above data in the rule's band) and the power share below 300 Hz (mic 0):\n\n"
    )
    s.append(
        "| model | full-band shift dB | ≥300 Hz shift dB | ≥500 Hz shift dB | share < 300 Hz model | data |\n"
        "|---|---:|---:|---:|---:|---:|\n"
    )
    for key in (
        "r3b_render",
        "r3b",
        "r2_render",
        "v2",
        "v2_n1",
        "r3b_render_wind_pool_mean",
        "r3b_render_wind_at_window",
        "r3b_render_no_wind",
    ):
        sh = lm[key]
        s.append(
            f"| {key} | {sh['shift_full_db']:+.2f} | {sh['shift_ge300_db']:+.2f} | {sh['shift_ge500_db']:+.2f} | "
            f"{sh['power_share_below_300']['model']:.2f} | {sh['power_share_below_300']['data']:.2f} |\n"
        )
    s.append("\n### T6. Realised Listen WAVs, mic 0, level gains divided out (clip − real, dB)\n\n")
    s.append(_hdr("clip / rule"))
    for rule in ("dev_abs_db", "dev_full_rms_db", "dev_ge300_db", "dev_ge500_db"):
        for k, v in rz[rule].items():
            s.append(f"| {k} / {rule.removesuffix('_db')} | {_row(v)} |\n")
    for k, v in rz["expected_minus_realised_db"].items():
        s.append(f"| {k} / render-expectation − realised (absolute) | {_row(v)} |\n")
    for k, v in rz["ltas_meandb_dev_abs_db"].items():
        s.append(f"| {k} / § Listen LTAS (mean dB, 8192), absolute | — | {_row(v)} |\n")
    for k, v in rz["figdata_ltas_dev_db"].items():
        s.append(f"| {k} / figdata `ltas_dev_db` (RMS-matched, mean dB) | — | {_row(v)} |\n")
    s.append(
        f"| real: support path − WAV path (mic 0 raw) | {_row(rz['support_minus_wav_real_raw_db'], 2)} |\n"
    )
    s.append("\n### T7. Mic minus channel-normalised mic mean, data (dB)\n\n")
    s.append(_hdr("mic / set"))
    for m in range(8):
        avg = np.mean([np.asarray(wins[n]["mic_minus_mean_norm_db"])[m] for n in pool], axis=0)
        s.append(f"| {m} / own avg | {_row(avg)} |\n")
        s.append(f"| {m} / Listen | {_row(np.asarray(L['mic_minus_mean_norm_db'])[m])} |\n")
    s.append("\n### T8. Data: Listen window minus own-window average (normalised, dB)\n\n")
    s.append(_hdr("mic"))
    own = np.mean([np.asarray(wins[n]["data"]["norm"]["mean"]) for n in pool], axis=0)
    s.append(f"| mean | {_row(np.asarray(L['data']['norm']['mean']) - own)} |\n")
    own0 = np.mean([np.asarray(wins[n]["data"]["norm"]["mic"])[0] for n in pool], axis=0)
    s.append(f"| 0 | {_row(np.asarray(L['data']['norm']['mic'])[0] - own0)} |\n")
    mi = d["michaels"]["windows"]
    s.append("\n### T9. Michael's r3b (standby/cruise blend) (model − normalised data, dB)\n\n")
    s.append(_hdr("window / model / mic"))
    mpool = [n for n, r in mi.items() if r["role"] == "pool"]
    for key in ("r3b_render", "r3b"):
        for mic in ("mean", 0):
            s.append(
                f"| own avg (8 FLY125 cruise) / {key} / {mic} | {_row(_pool_avg(mi, mpool, key, mic))} |\n"
            )
    ml = next(r for r in mi.values() if r["role"] == "listen")
    for key in ("r3b_render", "r3b"):
        s.append(f"| Listen FLY124 40–50 s / {key} / mean | {_row(ml['dev_db'][key]['mean'])} |\n")
        for rule in ("dev_abs_db", "dev_full_rms_db", "dev_ge300_db", "dev_ge500_db"):
            s.append(
                f"| Listen / {key} / 0 / {rule.removesuffix('_db')} | {_row(ml['listen_mic0'][key][rule])} |\n"
            )
    sh = ml["listen_mic0"]["r3b_render"]
    s.append(
        f"\nMichael's Listen mic 0, r3b render-exp.: full-band shift {sh['shift_full_db']:+.2f} dB, "
        f"share < 300 Hz model {sh['power_share_below_300']['model']:.2f} / data "
        f"{sh['power_share_below_300']['data']:.2f}.\n"
    )
    s.append(
        "\nData check, all frames − every "
        f"{STRIDE}th (mic mean, DREGON windows): max |Δ| "
        f"{max(float(np.max(np.abs(r['data_all_frames_minus_subsample_db']))) for r in wins.values()):.2f} dB "
        "(why the data side reads every frame).\n"
    )
    for key in ("r3b_render", "v2"):
        s.append(
            f"\n### A{1 if key == 'r3b_render' else 2}. Appendix: every window × mic, {key} − data (dB)\n\n"
        )
        s.append(_hdr("window / mic"))
        for n in pool + [lis]:
            for m in range(8):
                v = np.asarray(wins[n]["dev_db"][key]["mic"])[m]
                s.append(f"| {'Listen' if n == lis else short(n)} / {m} | {_row(v)} |\n")
    return "".join(s)


FINDINGS = """
# LTAS gap: v3 r3b DREGON vs the § Listen window

Why the r3b DREGON render sits −3.8 / −5.9 / −4.4 / −4.9 dB under the real
§ Listen clip at 0.3–5 kHz (`figdata_r3b.json`, RMS-matched, mic 0) while v2 is
within 0.5 dB. Everything below is the fits' own forward model against the data
(`scripts/noise_v3_ltas_gap.py`; tables regenerated by it). "render-exp." is the
r3b render expectation: the rig with every line and the floor times the Jensen
factor of fresh OU wander, which reproduces the realised Listen WAV to
≤ 0.6 dB in every band except 0.7–1.5 kHz, where this seed drew 1.4 dB low (T6).

## Verdict

The r3b fit is right on its own data, and the Listen comparison turns a
< 300 Hz error of one term on one mic into a 4–6 dB "mid deficit". On its five
windows the render expectation matches the channel-normalised mic mean within
0.7 dB in every band below 5 kHz (−0.1 / +0.4 / +0.3 / −0.1 / −0.6 / −0.7 dB at
30 Hz–5 kHz; +1.9 at 5–7 kHz, the known item), and within 0.4 dB with its
fitted latents: comb and floor are not too quiet. On the Listen window's mic 0
the render expectation is +3.6 / +3.0 dB too loud below 300 Hz and only
−0.5…−1.5 dB low at 0.7–5 kHz in absolute level. 78 % of the real clip's
power is below 300 Hz (89 % in the model), so the notebook's full-band RMS
match takes 3.5 dB off every band, and that turns −1 dB into −4…−6 dB. The
excess below 300 Hz is mic 0's static wind, fitted at −5.3 dB. The five fit
windows ask for −4.8…−7.9 dB (power mean −6.7 dB) and this window for
−9.3 dB. With the wind at this window's level the RMS-matched mids come back to
−1.4 / −1.5 / −0.9 / −1.8 dB. The rest is the window itself: 50–60 s is
1.3–1.8 dB brighter above 700 Hz than the pool, and v2's fit sees the same.
So the fit is wrong in one term, the wind: static by construction, 1.4 dB
above its own windows' power mean on mic 0, and unable to follow a capsule
whose wind moves by 4.5 dB between windows. The comparison is also unfair to
v3 twice over: the level rule is dominated by that term, and v2's clip is not
v2's mic 0. `render_noise(n_mics=1)` mean-pins v2's per-mic overall and line
gains to unity over the one rendered mic but keeps mic 0's absolute floor gain.
That moves the floor −2.9 dB and the comb +2.4…+3.0 dB against the fit's own
mic 0. The v2 fit's actual mic 0, under the same match, is −1.6 / −2.9 / −3.8 dB
at 0.7–5 kHz.

Fixes that add no model parameter:
(1) level-match the comparison on ≥ 500 Hz, above the wind's support. r3b mic 0
is then −0.5 / +0.1 / −0.8 dB at 0.7–5 kHz (realised WAV −0.9 / +0.9 / −0.1),
and the whole gap sits below 700 Hz, where it belongs to the wind.
(2) centre the wind on the per-window measurements, power-averaged: mic 0
goes −5.3 → −6.7 dB and the RMS-matched mids improve by 1.1 dB. It cannot
reach this window, whose wind is 2.6 dB under the pool mean.
(3) compare the 8-mic mean, or every mic, instead of mic 0: RMS-matched
−1.0 / −2.6 / −3.4 / −3.4 dB at 0.3–5 kHz, or −0.3 / −1.1 / −1.1 dB at 0.7–5 kHz
with the ≥ 500 Hz match.
Channel normalisation that includes < 500 Hz does not help. A scalar per-mic
gain cancels under any level match, and mic 0's low-band excess over the
normalised mean is not a channel constant: +3.2 / +4.0 dB at 100–700 Hz on the
pool, +1.6 / +1.2 dB on this window.

Michael's r3b shows no analogous deficit. Its own cruise windows are within
0.3 dB below 5 kHz. On its Listen window mic 0 is −0.3 / +0.3 / −0.1 / +0.4 dB
at 0.1–3 kHz after the RMS match. There the model is 6.5 dB short at
30–100 Hz (no wind term), so the match lifts the mids by 1.7 dB rather than
lowering them.

## 1. The fit's own windows (T0, T1, T2, T4, A1, A2)

* **r3b, mic mean, render-exp.**: window average −0.1 / +0.4 / +0.3 / −0.1 /
  −0.6 / −0.7 / +1.9 dB (30–100 … 5–7k). Per window every band below 5 kHz is
  within −1.7…+1.3 dB. With the FITTED latents (the Whittle optimum) the average is
  −0.2 / +0.4 / +0.1 / 0.0 / +0.1 / −0.1 / −0.2 dB. At latents zero it is
  −0.2 / +0.3 / −0.1 / −1.1 / −1.8 / −2.4 / −2.9 dB: the rig is the median
  level, and the render's wander restores the mean.
* **r3b, mic 0**: +1.7 / +0.9 / −0.9 / −1.0 / +0.8 / +0.3 / +2.9 dB. Mic 0 is
  1.9 dB too steep between 100–300 Hz and 0.3–1.5 kHz even on its own
  windows. After the ≥ 500 Hz rank-one normalisation the data's mic 0 still sits
  +1.7 / +3.2 / +4.0 / +0.9 / −1.4 / −1.0 / −1.0 dB against the mic mean (T7).
  v3's only per-mic freedom is the wind below 500 Hz, so mic 0's wind soaks
  up its low band and nothing carries its 0.3–1.5 kHz excess.
* **100–300 Hz minus 0.3–5 kHz, mic 0 (T4)**: data 16.5–17.9 dB on the five
  windows (16.5 on Listen), r3b 19.0–19.1, v2 17.2–17.3.
* **v2**: mic mean −1.3 / −0.8 / −0.5 / +0.7 / +0.6 / +0.1 / 0.0 dB, mic 0
  +0.3 / −1.1 / −1.5 / +0.4 / +0.3 / −0.2 / −0.1 dB. v2 is about 2 dB flatter
  than r3b on the same data (lows −0.8…−1.3 against r3b's −0.1…+0.4, mids
  +0.6…+0.7 against −0.1…−0.6). The data sit between the two.
* **r2, for the record**: render-exp. mic mean +2.1 / +1.9 / +2.3 dB above
  1.5 kHz on its own windows (the static part the fold removed). That confirms
  the campaign doc's [inference]: r2's closer Listen clip was a render bias
  that pointed the right way.

## 2. The Listen window, 50–60 s (T0, T3, T5, T6, T8)

* **Parts, mic 0, render-exp. − data**: wind +3.0 / +2.5 / +0.1 dB at
  30–100 / 100–300 / 300–700 Hz. Floor −10.0 / −13.8 / −5.5 / −4.1 / −10.1 /
  −16.7 / −13.0 dB. Comb −7.0 / −8.0 / −10.0 / −4.1 / −1.0 / −1.6 / +1.2 dB.
  Below 300 Hz the r3b model on mic 0 is nearly all wind.
* **Wind per mic (T3)**: the fitted `wind_db` for mic 0 is −5.3 dB. The own
  windows ask for −6.3 / −7.7 / −7.4 / −4.8 / −7.9 dB (power mean −6.7), and
  this window for −9.3 dB (the measured prior centre was −9.0). The asks barely
  depend on which comb + floor are subtracted (latents zero, render-exp. or
  fitted latents agree within 0.3 dB on mic 0), so they are data.
* **Counterfactuals, mic 0, RMS-matched**: as fitted −2.0 / −4.6 / −4.0 /
  −4.9 dB at 0.3–5 kHz. With the wind at the own-window power mean −1.8 / −3.5 /
  −2.9 / −3.8. With the wind at this window's level −1.4 / −1.5 / −0.9 / −1.8.
  The level-match shift goes +3.48 → +2.36 → +0.36 dB.
* **The window**: Listen minus own-window average, normalised data: mic mean
  −1.0 / −0.5 / +0.3 / +1.6 / +1.8 / +1.6 / +1.3 dB, mic 0 −2.0 / −2.1 / −2.4 /
  +0.1 / +1.3 / +1.7 / +1.5 dB. Both fits lose about 1.6 dB at 0.7–5 kHz to it
  (Listen minus own-window mic-mean deviation: r3b −1.6 / −1.8 / −1.7, v2 −1.5 /
  −1.7 / −1.6 dB).
* **Mic 0 against the normalised mean**: mic 0's low-band excess drops from
  +3.2 / +4.0 dB (100–300 / 300–700 Hz, pool) to +1.6 / +1.2 dB on this window.
  So the static per-mic wind, which on the pool carries that excess, overshoots
  here.
* **Checks**: the flight-support path and the notebook's `real_clip` give the
  same mic 0 to 0.01 dB. The v2 `n_mics=1` expectation equals the v2 WAV to
  0.1 dB. The data read every frame, because a 1-in-12 frame subsample moves
  band levels by up to 1.0 dB.

## 3. v2 on the same two sets (T0, T5, T6)

* The v2 Listen clip is `render_noise(n_mics=1)`, i.e. `v2_n1`: comb and
  overall gains at unity, and mic 0's absolute `mic_floor_db` of +3.8 dB. The v2
  fit's own mic 0 carries the pinned overall gain (+2.93 dB) and mic 0's line
  gains (−4.1 / −7.1 / −5.0 / −5.3 dB per rotor after pinning). Against that
  mic 0, the clip's floor is −2.9 dB and its comb +2.4…+3.0 dB. The v2 FIT's
  mic 0 on this window, RMS-matched: +0.4 / −0.8 / −0.9 / −1.6 / −2.9 / −3.8 /
  −3.4 dB (shift +1.98 dB, < 300 Hz share 0.85 against 0.78). The clip:
  +0.3 / −1.0 / −1.3 / 0.0 / +0.8 / −0.5 / −0.2 dB (shift −0.19 dB, share 0.81).
* What carries v2's match is its per-mic gains, through the render convention
  rather than the fit. Dropping them (`n_mics=1`) shifts the clip's low/mid
  balance by about 5 dB against the fit's own mic 0, towards the real mic 0.
  v2's low band is the tilted floor (−5.55 dB/oct) with a broadband per-mic
  floor gain, not a separate wind term. Below 300 Hz the clip's floor is
  −0.8 / −2.0 dB and its total +0.1 / −1.2 dB against the data, where r3b's
  wind alone is +3.0 / +2.5. The wide-line floor carries v2's band above 1.5 kHz: lines
  with γ ≥ 8 Hz sit −2.8 / −2.9 / −1.2 dB against the data at 1.5–3 / 3–5 /
  5–7 kHz, against −3.3 / −6.8 / −17.2 dB for the narrow lines and −8.0 /
  −8.7 / −8.6 dB for the floor. r3b carries that band with narrow lines lifted
  by the wander's Jensen factor (−2.5 / −1.6 / +1.2 dB). Both reproduce their
  own windows there. The wide-line floor is not what separates them on this
  window below 5 kHz.

## 4. Michael's r3b (T9)

Standby/cruise composition, same machinery. On its eight FLY125 cruise
windows, mic mean, render-exp.: −0.3 / −0.1 / −0.1 / −0.3 / −0.1 / +0.2 /
+1.4 dB. On its Listen window (FLY124 40–50 s), mic 0: absolute −6.5 / −2.0 /
−1.4 / −1.8 / −1.3 / +0.2 / +5.3 dB. RMS-matched: −4.8 / −0.3 / +0.3 / −0.1 /
+0.4 / +1.9 / +7.0 dB, consistent with the realised −1.0 / −1.0 / −0.4 / +0.2 /
+2.2 / +7.6 in the six LTAS bands. The shift is −1.70 dB, with a < 300 Hz share
of 0.64 against 0.75. There is no mid-band deficit: 0.1–3 kHz is within
0.4 dB. The 5–7 kHz excess is the separate known item.
"""


# ── figure ─────────────────────────────────────────────────────────────────


def smooth(db: np.ndarray, f: np.ndarray, frac: float = 1.0 / 6.0) -> np.ndarray:
    p = 10.0 ** (np.asarray(db) / 10.0)
    lf = np.log2(np.maximum(f, 1e-9))
    return _db(np.array([p[np.abs(lf - x) <= frac / 2].mean() for x in lf]))


def plot(d: dict[str, Any]) -> None:
    dr = d["dregon"]
    wins = dr["windows"]
    pool = [n for n, r in wins.items() if r["role"] == "pool"]
    lis = next(n for n, r in wins.items() if r["role"] == "listen")
    L = wins[lis]
    sp = L["spectra_mic0_db"]
    f = np.asarray(sp["freqs_hz"])
    keep = (f >= 30) & (f <= 7900)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes[0, 0]
    data_s = smooth(sp["data_norm"], f)[keep]
    ax.plot(f[keep], data_s, color=C_REAL, lw=1.8, label="real, mic 0 (normalised)")
    ax.plot(
        f[keep],
        smooth(sp["r3b_render_total"], f)[keep],
        color=C_R3B,
        lw=1.5,
        label="r3b render-exp. total",
    )
    wk = f[keep] < 520.0  # the wind shape is exactly zero above 500 Hz
    wind_s = smooth(sp["r3b_render_wind"], f)[keep]
    ax.plot(f[keep][wk], wind_s[wk], color=C_R3B, lw=1.1, ls=":", label="r3b wind (mic 0, −5.3 dB)")
    ax.plot(
        f[keep],
        smooth(sp["r3b_render_floor"], f)[keep],
        color=C_R3B,
        lw=1.1,
        ls="--",
        label="r3b floor",
    )
    ax.plot(
        f[keep],
        smooth(sp["r3b_render_comb"], f)[keep],
        color=C_R3B,
        lw=0.9,
        ls="-.",
        label="r3b comb",
    )
    g0 = float(dr["channel_gains_db"][0])
    ax.plot(
        f[keep],
        smooth(np.asarray(sp["v2_n1_total"]) - g0, f)[keep],
        color=C_V2,
        lw=1.5,
        label="v2 as rendered, total",
    )
    ax.plot(
        f[keep],
        smooth(np.asarray(sp["v2_n1_floor"]) - g0, f)[keep],
        color=C_V2,
        lw=1.1,
        ls="--",
        label="v2 as rendered, floor",
    )
    ax.set_xscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("dB (normalised fit units, 1/6-oct smoothed)")
    ax.set_title("Listen window 50–60 s, mic 0: render expectation vs data")
    ax.grid(alpha=0.25, which="both")
    ax.set_ylim(float(data_s.min()) - 15.0, float(data_s.max()) + 5.0)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    for lo, _hi in BANDS:
        ax.axvline(lo, color="#bbbbbb", lw=0.5)

    ax = axes[0, 1]
    x = np.arange(len(BANDS))
    lm = L["listen_mic0"]
    rows = [
        ("r3b, own windows, mic mean", _pool_avg(wins, pool, "r3b_render", "mean"), C_R3B, "//"),
        ("r3b, own windows, mic 0", _pool_avg(wins, pool, "r3b_render", 0), C_R3B, ".."),
        ("r3b, Listen, mic 0", lm["r3b_render"]["dev_abs_db"], C_R3B, ""),
        ("v2 fit, own windows, mic 0", _pool_avg(wins, pool, "v2", 0), C_V2, ".."),
        ("v2 as rendered, Listen, mic 0", lm["v2_n1"]["dev_abs_db"], C_V2, ""),
    ]
    wdt = 0.16
    for i, (lab, v, c, h) in enumerate(rows):
        ax.bar(
            x + (i - 2) * wdt,
            np.asarray(v, dtype=float),
            wdt,
            color=c,
            alpha=0.55 if h else 0.95,
            hatch=h,
            label=lab,
            edgecolor="k",
            lw=0.3,
        )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, BAND_LABELS, rotation=20)
    ax.set_xlabel("band (Hz)")
    ax.set_ylabel("model − data, NO level match (dB)")
    ax.set_title("Absolute level: own windows (avg of 5) vs Listen (r3b = render expectation)")
    ax.set_ylim(-2.5, 5.0)
    ax.legend(fontsize=8, ncol=2, loc="upper center")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[1, 0]
    m = np.arange(8)
    ax.plot(m, dr["wind_db"]["r3b"], "o-", color=C_R3B, lw=2.2, label="r3b fitted wind_db")
    ax.plot(m, dr["wind_db_measured_centre"], "s--", color="#9467bd", label="measured prior centre")
    cmap = plt.get_cmap("Blues")
    for i, n in enumerate(pool):
        ax.plot(
            m,
            np.asarray(wins[n]["wind_hat_db"]["r3b_render"], dtype=float),
            "x-",
            color=cmap(0.4 + 0.12 * i),
            lw=0.9,
            label=f"asks: {short(n)}",
        )
    ax.plot(
        m,
        np.asarray(L["wind_hat_db"]["r3b_render"], dtype=float),
        "D-",
        color="k",
        lw=1.8,
        label="asks: Listen 50–60 s",
    )
    ax.set_xlabel("mic")
    ax.set_ylabel("flat wind level (dB, normalised units)")
    ax.set_title("Wind: fitted level vs the 30–300 Hz excess over r3b comb + floor")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    rz = dr["realised"]
    rows2 = [
        ("r3b render-exp., full-band RMS match", lm["r3b_render"]["dev_full_rms_db"], C_R3B, ""),
        ("r3b realised WAV, full-band RMS match", rz["dev_full_rms_db"]["r3b"], C_R3B, "xx"),
        (
            "r3b, wind at this window's level, RMS match",
            lm["r3b_render_wind_at_window"]["dev_full_rms_db"],
            "#ff7f0e",
            "",
        ),
        ("r3b render-exp., ≥300 Hz match", lm["r3b_render"]["dev_ge300_db"], C_R3B, "//"),
        ("r3b render-exp., ≥500 Hz match", lm["r3b_render"]["dev_ge500_db"], C_R3B, ".."),
        ("v2 as rendered, full-band RMS match", lm["v2_n1"]["dev_full_rms_db"], C_V2, ""),
        ("v2 realised WAV, full-band RMS match", rz["dev_full_rms_db"]["v2"], C_V2, "xx"),
    ]
    wdt = 0.12
    for i, (lab, v, c, h) in enumerate(rows2):
        off = (i - 0.5 * (len(rows2) - 1)) * wdt
        ax.bar(
            x + off,
            np.asarray(v, dtype=float),
            wdt,
            color=c,
            alpha=0.8,
            hatch=h,
            label=lab,
            edgecolor="k",
            lw=0.3,
        )
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, BAND_LABELS, rotation=20)
    ax.set_xlabel("band (Hz)")
    ax.set_ylabel("model − data after level match (dB)")
    ax.set_title("Listen mic 0 under level-match rules")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{STEM}.png", dpi=110)
    plt.close(fig)
    print(f"wrote {OUT / STEM}.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args(argv)
    path = OUT / f"{STEM}.json"
    if args.plot_only:
        d = json.loads(path.read_text())
    else:
        d = _r(dict(dregon=compute_dregon(), michaels=compute_michaels()), 4)
        path.write_text(json.dumps(d))
        print(f"wrote {path}")
    (OUT / f"{STEM}.md").write_text(report(d))
    print(f"wrote {OUT / STEM}.md")
    plot(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
