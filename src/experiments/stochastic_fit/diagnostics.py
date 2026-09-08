"""Where the excess lives: read the residual field of a fitted clip.

Under a correct model the residual ``R = I / M`` is exponential with mean 1
and no structure. Each function below asks one structured question of it —
along the fitted lines, over time, between the lines, per microphone — and
returns plain arrays a notebook or the report can plot. Everything is
computed from one result file (``run.fit`` output) and is cheap.

Order bands follow the paper's split: 1–8, 9–24, 25–64, 65+.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fit import LOO_OFFSETS, loo_reference

ORDER_BANDS: tuple[tuple[int, int], ...] = ((1, 8), (9, 24), (25, 64), (65, 10_000))
BAND_NAMES = ("k1-8", "k9-24", "k25-64", "k65+")


CLIP_CACHE = Path.home() / ".cache" / "stochastic_fit_clips"


def load_clip_cached(clip_id: str):
    """The bundle clip, from the local cache or R2."""
    from .run import BUCKET, PREFIX, clip_from_bytes, r2_client

    CLIP_CACHE.mkdir(parents=True, exist_ok=True)
    local = CLIP_CACHE / f"{clip_id}.npz"
    if not local.exists():
        body = r2_client().get_object(Bucket=BUCKET, Key=f"{PREFIX}/{clip_id}.npz")["Body"].read()
        local.write_bytes(body)
    return clip_from_bytes(local.read_bytes())


@dataclass
class FitResult:
    clip_id: str
    group: str
    variant: str
    power: np.ndarray  # (M, N, F)
    spectrum: np.ndarray  # (M, N, F) fitted expected periodogram
    loo: np.ndarray  # (M, N, F) leave-one-out smoother
    freqs: np.ndarray
    times: np.ndarray
    rps: np.ndarray  # (R, N) reference carriers
    scores: dict[str, Any]
    params: dict[str, Any]
    spec: dict[str, Any]
    meta: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> FitResult:
        """Load a result file. Slim files (``spectrum_db`` only) get their
        periodogram and LOO smoother rebuilt from the clip bundle (cached
        under ``CLIP_CACHE``, fetched from R2 on first use)."""
        with np.load(path, allow_pickle=True) as z:
            meta = json.loads(str(z["meta"]))
            scores = json.loads(str(z["scores"]))
            common = (
                z["freqs"],
                z["times"],
                z["rps"],
                scores,
                json.loads(str(z["params"])),
                json.loads(str(z["spec"])),
                meta,
            )
            if "power" in z.files:  # full-size file: rebuild the smoother with the current offsets
                power, spectrum = z["power"], z["spectrum"]
                _, m_hat = loo_reference(
                    power.astype(np.float32) / scores["power_scale"], z["freqs"] >= 30.0
                )
                return cls(
                    meta["clip_id"],
                    meta["group"],
                    meta["variant"],
                    power,
                    spectrum,
                    (m_hat * scores["power_scale"]).astype(np.float32),
                    *common,
                )
            spectrum = (10.0 ** (z["spectrum_db"].astype(np.float32) / 10.0)).astype(np.float32)
        from .data import periodogram

        clip = load_clip_cached(meta["clip_id"])
        pg = periodogram(clip)
        assert pg.power.shape == spectrum.shape, (pg.power.shape, spectrum.shape)
        band = pg.freqs >= 30.0
        _, m_hat = loo_reference(pg.power.astype(np.float32) / scores["power_scale"], band)
        return cls(
            meta["clip_id"],
            meta["group"],
            meta["variant"],
            pg.power,
            spectrum,
            (m_hat * scores["power_scale"]).astype(np.float32),
            *common,
        )

    @property
    def residual(self) -> np.ndarray:
        return self.power / np.maximum(self.spectrum, 1e-30)

    @property
    def carrier(self) -> np.ndarray:
        return np.asarray(self.params["carrier"])  # (R, N), includes any fitted offset

    @property
    def gamma(self) -> np.ndarray:
        return np.asarray(self.params["gamma"])  # (R, K)

    @property
    def df(self) -> float:
        return float(self.freqs[1] - self.freqs[0])


def _band_index(k: np.ndarray) -> np.ndarray:
    idx = np.full(k.shape, -1)
    for b, (lo, hi) in enumerate(ORDER_BANDS):
        idx[(k >= lo) & (k <= hi)] = b
    return idx


def line_cells(res: FitResult, f_max: float | None = None) -> dict[str, np.ndarray]:
    """Every in-band (rotor, order, frame) line with its centre bin and width.

    Lines closer than ``2*gamma`` to any other rotor's line at the same frame
    are flagged ``isolated=False`` so shape diagnostics can exclude overlaps.
    """
    K = res.gamma.shape[1]
    nyq = f_max or float(res.freqs[-1])
    k = np.arange(1, K + 1)
    carrier = res.carrier  # (R, N)
    centres = k[None, :, None] * carrier[:, None, :]  # (R, K, N)
    live = (centres > 30.0) & (centres < nyq)
    r_i, k_i, n_i = np.nonzero(live)
    c = centres[r_i, k_i, n_i]
    g = res.gamma[r_i, k_i]
    # isolation: nearest other-rotor line at the same frame
    isolated = np.ones(c.size, dtype=bool)
    for n in np.unique(n_i):
        sel = n_i == n
        cc, gg = c[sel], g[sel]
        order = np.argsort(cc)
        cs, gs = cc[order], gg[order]
        # neighbours in sorted order (any rotor, including own adjacent orders)
        gap_prev = np.r_[np.inf, np.diff(cs)]
        gap_next = np.r_[np.diff(cs), np.inf]
        g_prev = np.r_[0.0, gs[:-1]]
        g_next = np.r_[gs[1:], 0.0]
        iso = (gap_prev > gs + g_prev + res.df) & (gap_next > gs + g_next + res.df)
        out = np.empty_like(iso)
        out[order] = iso
        isolated[sel] = out
    return dict(
        rotor=r_i,
        order=k_i + 1,
        frame=n_i,
        centre=c,
        gamma=g,
        band=_band_index(k_i + 1),
        isolated=isolated,
    )


def line_profile(
    res: FitResult, offsets: np.ndarray | None = None, only_isolated: bool = True
) -> dict[str, Any]:
    """Mean residual at signed offsets from each line centre, in units of the
    line's fitted half width, per order band. Flat at 1 = the shape is right.

    Also returns the same profile of the *periodogram* and of the *model*
    (both normalized by the model's peak) so a plot can show what the data
    look like against what was fitted, not only their ratio.
    """
    offsets = np.linspace(-4.0, 4.0, 33) if offsets is None else offsets
    cells = line_cells(res)
    keep = cells["isolated"] if only_isolated else np.ones(cells["centre"].size, dtype=bool)
    M, N, F = res.power.shape
    resid = res.residual
    prof_r = np.zeros((len(ORDER_BANDS), offsets.size))
    prof_i = np.zeros_like(prof_r)
    prof_m = np.zeros_like(prof_r)
    count = np.zeros(len(ORDER_BANDS))
    for b in range(len(ORDER_BANDS)):
        sel = keep & (cells["band"] == b)
        if not sel.any():
            continue
        c, g, n = cells["centre"][sel], cells["gamma"][sel], cells["frame"][sel]
        # sample at f = c + o*gamma by linear interpolation along frequency
        f_at = c[:, None] + offsets[None, :] * g[:, None]  # (L, O)
        pos = f_at / res.df
        i0 = np.clip(np.floor(pos).astype(int), 0, F - 2)
        w = np.clip(pos - i0, 0.0, 1.0)
        for m in range(M):
            rr = resid[m, n[:, None], i0] * (1 - w) + resid[m, n[:, None], i0 + 1] * w
            ii = res.power[m, n[:, None], i0] * (1 - w) + res.power[m, n[:, None], i0 + 1] * w
            mm = res.spectrum[m, n[:, None], i0] * (1 - w) + res.spectrum[m, n[:, None], i0 + 1] * w
            peak = mm[:, offsets.size // 2][:, None]
            prof_r[b] += rr.sum(axis=0)
            prof_i[b] += (ii / peak).sum(axis=0)
            prof_m[b] += (mm / peak).sum(axis=0)
            count[b] += rr.shape[0]
    ok = count > 0
    prof_r[ok] /= count[ok, None]
    prof_i[ok] /= count[ok, None]
    prof_m[ok] /= count[ok, None]
    return dict(
        offsets=offsets, residual=prof_r, data=prof_i, model=prof_m, count=count, bands=BAND_NAMES
    )


def centre_residual_by_order(res: FitResult) -> dict[str, np.ndarray]:
    """Mean residual at each order's centre bin over frames and mics, per rotor:
    ``(R, K)`` — a static level error the free profile should not leave, so
    structure here is time-varying misfit that the drift model could not follow."""
    cells = line_cells(res)
    R, K = res.gamma.shape
    acc = np.zeros((R, K))
    cnt = np.zeros((R, K))
    bins = np.clip(np.rint(cells["centre"] / res.df).astype(int), 0, res.freqs.size - 1)
    resid = res.residual.mean(axis=0)  # over mics
    np.add.at(acc, (cells["rotor"], cells["order"] - 1), resid[cells["frame"], bins])
    np.add.at(cnt, (cells["rotor"], cells["order"] - 1), 1.0)
    return dict(mean=np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan), count=cnt)


def half_order_residual(res: FitResult) -> dict[str, np.ndarray]:
    """Mean residual at half-integer orders ``(k + 1/2) r`` per band — lines the
    family does not have (sub-harmonics, blade-passing sidebands)."""
    K = res.gamma.shape[1]
    k = np.arange(1, K) + 0.5
    carrier = res.carrier
    centres = k[None, :, None] * carrier[:, None, :]
    live = (centres > 30.0) & (centres < res.freqs[-1])
    bins = np.clip(np.rint(centres / res.df).astype(int), 0, res.freqs.size - 1)
    resid = res.residual.mean(axis=0)
    out = np.zeros(len(ORDER_BANDS))
    cnt = np.zeros(len(ORDER_BANDS))
    band = _band_index(np.floor(k).astype(int))
    for b in range(len(ORDER_BANDS)):
        sel = live & (band[None, :, None] == b)
        if sel.any():
            r_i, k_i, n_i = np.nonzero(sel)
            out[b] = resid[n_i, bins[r_i, k_i, n_i]].mean()
            cnt[b] = sel.sum()
    return dict(mean=out, count=cnt, bands=np.array(BAND_NAMES))


def temporal_acf(res: FitResult, max_lag: int = 30) -> dict[str, np.ndarray]:
    """Autocorrelation of ``log R`` at line centres over frames, per band,
    averaged over lines and mics. White = the drift model is adequate."""
    cells = line_cells(res)
    R, K = res.gamma.shape
    logr = np.log(np.maximum(res.residual, 1e-12))  # (M, N, F)
    bins = np.clip(np.rint(cells["centre"] / res.df).astype(int), 0, res.freqs.size - 1)
    acf = np.zeros((len(ORDER_BANDS), max_lag + 1))
    cnt = np.zeros(len(ORDER_BANDS))
    # per (rotor, order): the series over frames
    for r in range(R):
        for k in range(K):
            sel = (cells["rotor"] == r) & (cells["order"] == k + 1)
            if sel.sum() < max_lag + 5:
                continue
            n, bb = cells["frame"][sel], bins[sel]
            b = int(cells["band"][sel][0])
            for m in range(logr.shape[0]):
                x = logr[m, n, bb]
                x = x - x.mean()
                v = float(np.dot(x, x))
                if v <= 0:
                    continue
                for lag in range(max_lag + 1):
                    acf[b, lag] += float(np.dot(x[: x.size - lag], x[lag:])) / v
                cnt[b] += 1
    ok = cnt > 0
    acf[ok] /= cnt[ok, None]
    return dict(acf=acf, count=cnt, lags=np.arange(max_lag + 1), bands=np.array(BAND_NAMES))


def coherence_by_band(res: FitResult) -> dict[str, np.ndarray]:
    """Normalized variance ``Var(R)/E[R]^2`` of the residual at line centres per
    band: 1 for exponential (narrowband noise), lower for a tone in noise. The
    same statistic on the synthetic controls is the exponential reference."""
    cells = line_cells(res)
    bins = np.clip(np.rint(cells["centre"] / res.df).astype(int), 0, res.freqs.size - 1)
    out = np.full(len(ORDER_BANDS), np.nan)
    cnt = np.zeros(len(ORDER_BANDS))
    for b in range(len(ORDER_BANDS)):
        sel = cells["band"] == b
        if sel.sum() < 10:
            continue
        vals = res.residual[:, cells["frame"][sel], bins[sel]].ravel()
        out[b] = float(vals.var() / max(vals.mean() ** 2, 1e-30))
        cnt[b] = vals.size
    return dict(norm_var=out, count=cnt, bands=np.array(BAND_NAMES))


def between_lines(res: FitResult, n_bins: int = 24) -> dict[str, Any]:
    """Mean residual on floor cells (further than ``3*gamma`` from every line)
    against frequency (log-spaced bins) and per microphone — floor shape and
    the common-floor assumption."""
    M, N, F = res.power.shape
    cells = line_cells(res)
    near = np.zeros((N, F), dtype=bool)
    for c, g, n in zip(cells["centre"], cells["gamma"], cells["frame"], strict=True):
        w = max(1.5 * g, res.df)
        lo = max(int(np.floor((c - w) / res.df)), 0)
        hi = min(int(np.ceil((c + w) / res.df)) + 1, F)
        near[n, lo:hi] = True
    floor = ~near
    floor[:, res.freqs < 30.0] = False
    resid = res.residual
    edges = np.geomspace(30.0, float(res.freqs[-1]), n_bins + 1)
    prof = np.full(n_bins, np.nan)
    for i in range(n_bins):
        fsel = (res.freqs >= edges[i]) & (res.freqs < edges[i + 1])
        mask = floor & fsel[None, :]
        if mask.sum() > 20:
            prof[i] = float(resid[:, mask].mean())
    per_mic_floor = np.array([float(resid[m][floor].mean()) for m in range(M)])
    per_mic_line = np.array([float(resid[m][near].mean()) for m in range(M)])
    return dict(
        freq_centres=np.sqrt(edges[:-1] * edges[1:]),
        residual=prof,
        per_mic_floor=per_mic_floor,
        per_mic_line=per_mic_line,
        floor_fraction=float(floor.mean()),
    )


def excess_by_region(res: FitResult) -> dict[str, float]:
    """Excess over the LOO reference split into line cells and floor cells
    (nats/cell, inner frames only), so the misfit can be attributed."""
    M, N, F = res.power.shape
    half = max(abs(s) for s in LOO_OFFSETS)
    inner = slice(half, N - half)
    cells = line_cells(res)
    near = np.zeros((N, F), dtype=bool)
    for c, g, n in zip(cells["centre"], cells["gamma"], cells["frame"], strict=True):
        w = max(g, res.df)
        lo = max(int(np.floor((c - w) / res.df)), 0)
        hi = min(int(np.ceil((c + w) / res.df)) + 1, F)
        near[n, lo:hi] = True
    band = res.freqs >= 30.0
    p = res.power[:, inner][..., band].astype(np.float64)
    m_fit = np.maximum(res.spectrum[:, inner][..., band], 1e-30)
    m_loo = np.maximum(res.loo[:, inner][..., band], 1e-30)
    cell_fit = p / m_fit + np.log(m_fit)
    cell_loo = p / m_loo + np.log(m_loo)
    from scipy.special import digamma

    n_nb = len(LOO_OFFSETS)
    bias = n_nb / (n_nb - 1.0) - 1.0 + (float(digamma(n_nb)) - np.log(n_nb))
    ex = cell_fit - (cell_loo - bias)
    lines = near[inner][:, band][None].repeat(M, axis=0)
    return dict(
        excess_all=float(ex.mean()),
        excess_lines=float(ex[lines].mean()) if lines.any() else float("nan"),
        excess_floor=float(ex[~lines].mean()) if (~lines).any() else float("nan"),
        line_fraction=float(lines.mean()),
    )


def fitted_parameter_summary(res: FitResult) -> dict[str, Any]:
    """The fitted parameters in the sampler's coordinates, for the range check."""
    p = res.params
    prof = np.asarray(p["profile_db"])  # (R, K)
    R, K = prof.shape
    k = np.arange(1, K + 1)
    cells = line_cells(res)
    live = np.zeros((R, K), dtype=bool)
    live[cells["rotor"], cells["order"] - 1] = True
    # only lines that stand out of the floor are constrained by the data: an
    # invisible line's level is arbitrary and would swamp every statistic
    gamma_all = np.asarray(p["gamma"])
    fl_shape = np.asarray(p["floor_shape_db"])
    ctrl = np.asarray(p["floor_ctrl_hz"])
    mean_rps = res.carrier.mean(axis=1)
    speed_db = 10 * 2.5 * np.log10(np.maximum(mean_rps, 1e-3) / 80.0)
    peak_db = prof - 10 * np.log10(np.pi * gamma_all) + speed_db[:, None]
    f_line = np.maximum(k[None, :] * mean_rps[:, None], 30.0)
    floor_db = (
        p["floor_mean_db"]
        + np.interp(np.log2(f_line), np.log2(ctrl), fl_shape)
        + p["floor_tilt_db_oct"] * np.log2(f_line / 500.0)
        + speed_db.mean()
    )
    visible = live & (peak_db > floor_db - 6.0)
    live = visible
    rolloff, jitter = [], []
    for r in range(R):
        sel = live[r] & np.isfinite(prof[r])
        if sel.sum() > 5:
            x = -10.0 * np.log10(k[sel])
            a, b = np.polyfit(x, prof[r][sel], 1)
            rolloff.append(float(a))
            jitter.append(float(np.std(prof[r][sel] - (a * x + b))))
    h = np.asarray(p["h_db"])  # (R, K, Tk)
    h_live = h[live]
    h_std = float(np.mean(h_live.std(axis=1))) if h_live.size else float("nan")
    # correlation time: lag where the mean knot-ACF drops below 1/e
    knots = np.asarray(p["knots_s"])
    dt = float(knots[1] - knots[0]) if knots.size > 1 else float("nan")
    tau = float("nan")
    if h_live.size and h_live.shape[1] > 4:
        x = h_live - h_live.mean(axis=1, keepdims=True)
        v = (x * x).sum(axis=1)
        ok = v > 1e-9
        if ok.any():
            acf = np.array(
                [
                    (x[ok, : x.shape[1] - lag] * x[ok, lag:]).sum(axis=1) / v[ok]
                    for lag in range(x.shape[1])
                ]
            ).mean(axis=1)
            below = np.nonzero(acf < np.exp(-0.5))[0]  # SE kernel at lag = tau
            tau = float(below[0] * dt) if below.size else float("nan")
    # common/private split: variance of the per-rotor mean drift over total
    common = (
        float(
            np.mean([h[r][live[r]].mean(axis=0).var() for r in range(R) if live[r].sum() > 1])
            / max(h_live.var(), 1e-12)
        )
        if h_live.size
        else float("nan")
    )
    # floor under the median line peak, at the reference speed
    fl = fl_shape
    floor_rel = float(np.median(peak_db[live] - floor_db[live])) if live.any() else float("nan")
    return dict(
        rolloff_p=rolloff,
        harm_jitter_db=jitter,
        gamma0_hz=[float(x) for x in p["gamma0"]],
        gamma_slope_hz=[float(x) for x in p["gamma_slope"]],
        harm_gp_std_db=h_std,
        harm_gp_tau_s=tau,
        harm_coherence=common,
        floor_rel_db=-floor_rel,
        visible_lines=int(live.sum()),
        visible_fraction_by_band=[
            float(live[:, lo - 1 : hi].mean()) for lo, hi in ORDER_BANDS if lo <= K
        ],
        floor_tilt_db_oct=float(p["floor_tilt_db_oct"]),
        floor_shape_std_db=float(np.std(fl)),
        mic_gain_spread_db=float(np.ptp(np.asarray(p["mic_gain_db"]), axis=0).mean()),
        mic_floor_spread_db=float(np.ptp(np.asarray(p["mic_floor_db"]))),
        rps_offset_rms=float(np.sqrt(np.mean(np.square(np.asarray(p["rps_offset"]))))),
        amp_exp=float(p["amp_exp"]),
        floor_exp=float(p["floor_exp"]),
        floor_static_rel=float(p["floor_static_rel"]),
    )


def phase_diagnostics(
    res: FitResult, lags: tuple[int, ...] = (1, 2, 3, 4, 6, 8, 12, 16)
) -> dict[str, Any]:
    """Lag coherence along isolated lines per order band (tested estimator,
    :mod:`.phase_stats`), the Lorentzian prediction at the band's median
    fitted width, and the centre regressions on speed deviation / sub-bin
    offset. Needs the clip's audio (bundle cache)."""
    from .phase_stats import centre_regressions, lag_coherence, lorentzian_lag_prediction

    clip = load_clip_cached(res.clip_id)
    cells = line_cells(res)
    bins = np.clip(np.rint(cells["centre"] / res.df).astype(int), 0, res.freqs.size - 1)
    carrier = res.carrier
    dev = 100.0 * (
        carrier[cells["rotor"], cells["frame"]] / carrier.mean(axis=1)[cells["rotor"]] - 1.0
    )
    sub_bin = np.abs(cells["centre"] / res.df - np.rint(cells["centre"] / res.df))
    log_r = np.log(np.maximum(res.residual[:, cells["frame"], bins], 1e-9)).mean(axis=0)
    out: dict[str, Any] = dict(lags=list(lags), bands={})
    R = carrier.shape[0]
    for b, name in enumerate(BAND_NAMES):
        sel = (cells["band"] == b) & cells["isolated"]
        if sel.sum() < 50:
            continue
        tracks = []
        for r in range(R):
            for k in np.unique(cells["order"][sel & (cells["rotor"] == r)]):
                s2 = sel & (cells["rotor"] == r) & (cells["order"] == k)
                tracks.append((cells["frame"][s2], cells["centre"][s2]))
        lc = lag_coherence(clip.audio, tracks, lags)
        gamma_med = float(np.median(cells["gamma"][sel]))
        out["bands"][name] = dict(
            coherence=lc.coherence.tolist(),
            null=lc.null.tolist(),
            n_pairs=lc.n_pairs.tolist(),
            gamma_median_hz=gamma_med,
            lorentzian_prediction=lorentzian_lag_prediction(gamma_med, np.asarray(lags)).tolist(),
            regressions=centre_regressions(log_r[sel], dev[sel], sub_bin[sel]),
        )
    return out


def analyse(path: Path) -> dict[str, Any]:
    """Every diagnostic of one result file, JSON-serializable."""
    res = FitResult.load(path)

    def plain(d: dict[str, Any]) -> dict[str, Any]:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}

    return dict(
        clip_id=res.clip_id,
        group=res.group,
        variant=res.variant,
        scores=res.scores,
        excess=excess_by_region(res),
        profile=plain(line_profile(res)),
        half_order=plain(half_order_residual(res)),
        acf=plain(temporal_acf(res)),
        coherence=plain(coherence_by_band(res)),
        floor=plain(between_lines(res)),
        params=fitted_parameter_summary(res),
        phase=phase_diagnostics(res),
        planted=res.meta.get("planted"),
    )
