"""A GENERIC drone prior: rig payloads drawn from laws, not from fits
(:mod:`.rig_prior` is the other thing: redraws from the prior a FIT recorded).

Every number is grounded in the campaign measurements (DREGON bench singles,
the per-drone gallery, Michael's and DREGON flight fits, the six-rig
telemetry) — see ``docs/experiments/static-rig-profiles.md`` § Round 3 and
``docs/single-rotor-bench-fit-design.md``. A draw is a ``noise-v3-fit/1``
payload :func:`data_processing.noise_model.render.render_noise` accepts,
carrying the three prior-rig blocks the renderer reads on top of a v3 fit:
``am`` (per-order log-OU amplitude envelopes shared by the mics — the single
term that absorbs amplitude modulation AND the fast bounded phase wobble),
``profile.mic_dev_sd_db`` (the per-(mic, line) near-field scatter, redrawn
per clip) and ``wind_sc`` (SC wind noise on some capsules, from none up to
the DREGON free-flight level). No per-line Wiener width (``gamma_hz`` = 0),
no block wander, no pair EQ.

Profile law per rotor, written as LINE OVER FLOOR (dB, the model's expected
periodogram of one rotor at ``cal_rps``, 2048-point Hann at 16 kHz — the
frame of ``noise_lab.param_view``, which stacks the rotors: + 6 dB for four):
the blade-pass family (even orders, 2 blades) starts ``k2_over_floor_db``
over the floor at ``k = 2`` (Michael's flight fits 36 / 29 dB stacked) and
decays as a power law in ``k`` until it reaches a faint TAIL that then rides
the floor to the last order (the flight fits: 3-8 dB stacked from ``k`` ~ 15
to 75-100) — a comb sinks into its floor and never ends on a cliff; odd
orders sit below the even law by a penalty growing with ``log10 k``;
multiples of the motor's commutation period ``3 p`` (``p`` pole pairs) get a
boost; every rotor of a drone scatters about the drone's profile. The
conversion to ``profile_db`` (window gain, the line's width at order ``k``,
the floor's shape) is read off the model itself (:func:`line_contrast_db`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from data_processing.noise_model import FIT_SCHEMA_V3
from data_processing.noise_model import spectrum as SP
from data_processing.noise_model.constants import FLOOR_SHAPE_N_CTRL
from data_processing.noise_model.spectrum import floor_ctrl_hz, floor_shape_chol

SR_WORK = 32000
SR = SP.FLIGHT_SR


@dataclass(frozen=True)
class PriorSpec:
    """The laws. ``(mu, sd)`` pairs are Normal in dB unless the name says log."""

    name: str = "prior_v1"
    n_rotors: int = 4
    k_max: int = 115
    blades: int = 2
    # aero comb (even orders), line over floor: k2 - slope log10(k/2), floored at the tail
    k2_over_floor_db: tuple[float, float] = (26.0, 5.0)  # per rotor, at cal_rps
    slope_db_dec: tuple[float, float] = (25.0, 5.0)
    tail_over_floor_db: tuple[float, float] = (0.0, 2.0)  # per rotor
    cal_rps: float = 60.0  # the whole comb in band
    bpf_boost_db: tuple[float, float] = (8.0, 4.0)  # k = 2 over the law
    # odd orders: law - (a + b log10 k)
    odd_a_db: tuple[float, float] = (5.0, 2.0)
    odd_b_db: tuple[float, float] = (10.0, 4.0)
    # motor family: multiples of 3 p
    pole_pairs: tuple[int, ...] = (6, 7, 7, 8, 11)
    motor_boost_db: tuple[float, float] = (9.0, 3.0)
    motor_decay_per_multiple: float = 0.8
    # per-rotor scatter about the drone profile
    rotor_line_sd_db: float = 2.5
    rotor_level_sd_db: float = 1.5
    # per-(mic, line) scatter, redrawn per clip (per-run near-field + the pair EQ it absorbs)
    mic_dev_sd_db: float = 3.4
    # speed laws (power)
    amp_exp: tuple[float, float] = (5.5, 1.5)
    log_floor_exp: tuple[float, float] = (np.log(5.0), 0.3)
    log_floor_static_rel: tuple[float, float] = (np.log(2.5e-3), 1.0)
    # shaft OU (bench law): rad/s and 1/s, log-normal
    log_sigma_nu: tuple[float, float] = (np.log(0.4), 0.4)
    log_lam: tuple[float, float] = (np.log(12.0), 0.4)
    # per-order pedestal (AM + fast phase wobble), shared by mics
    log_am_s2_low: tuple[float, float] = (np.log(0.05), 1.0)  # k <= 8
    log_am_s2_mid: tuple[float, float] = (np.log(0.12), 0.7)  # 9..30
    am_s2_high_base: float = 0.07  # + k^2 sigma_psi^2 above 30
    am_s2_cap: float = 0.6  # the coherent core keeps >= 55 % of a line
    wobble_gamma_hz: float = (
        25.0  # the pedestal's rate where the fast wobble dominates (sidebands beyond +-16 Hz)
    )
    # rad, per drone: the six-rig telemetry spans 0.005-0.08; kept to the quiet half
    log_sigma_psi: tuple[float, float] = (np.log(0.012), 0.7)
    log_am_gamma_low: tuple[float, float] = (np.log(0.4), 0.6)  # Hz, k <= 8
    log_am_gamma_high: tuple[float, float] = (np.log(1.0), 0.6)
    am_parity_rho: float = 0.25  # AR(1) across same-parity neighbours (log sigma2)
    # floor: level and shape
    floor_mean_db: tuple[float, float] = (-36.0, 4.0)
    floor_shape_sd_db: tuple[float, float] = (4.0, 5.0)  # uniform
    floor_template_p_hump: float = 0.5  # else tilt
    # wind
    wind_sc: dict[str, Any] = field(
        default_factory=lambda: dict(
            p_wind=0.7, max_over_floor_db=33.0, shield_mean_db=8.0, speed_range=[2.0, 10.0]
        )
    )


def _n(rng: np.random.Generator, law: tuple[float, float], size: Any = None) -> Any:
    return rng.normal(law[0], law[1], size)


def _ln(rng: np.random.Generator, law: tuple[float, float], size: Any = None) -> Any:
    return np.exp(rng.normal(law[0], law[1], size))


def _template_ctrl_db(
    rng: np.random.Generator, spec: PriorSpec, ctrl_hz: np.ndarray
) -> tuple[np.ndarray, str]:
    """A mean floor shape at the control points (dB about the level): a bench
    1-5 kHz hump or a flight tilt."""
    if rng.uniform() < spec.floor_template_p_hump:
        c = 8.0 * np.exp(-0.5 * ((np.log2(ctrl_hz / 2200.0)) / 1.1) ** 2)
        return c - c.mean(), "hump"
    c = -6.0 * np.log2(np.maximum(ctrl_hz, 300.0) / 300.0)
    return c - c.mean(), "tilt"


def _payload(
    spec: PriorSpec,
    *,
    profile: np.ndarray,
    sigma_nu: float,
    lam: float,
    amp_exp: float,
    floor: dict[str, Any],
    am: dict[str, Any] | None,
) -> dict[str, Any]:
    """A ``noise-v3-fit/1`` payload around ``profile`` ``(R', K)`` (``R'`` rotors)."""
    R, K = profile.shape
    return {
        "schema": FIT_SCHEMA_V3,
        "kind": "prior",
        "mode": spec.name,
        "n_rotors": R,
        "n_mics": 8,
        "k_max": K,
        "sr": SR,
        "front_end": {
            "sr": SR,
            "n_fft": SP.FLIGHT_N_FFT,
            "hop": SP.FLIGHT_HOP,
            "sr_work": SR_WORK,
            "n_fft_work": SP.FLIGHT_N_FFT * SR_WORK // SR,
        },
        "params": {
            "sigma_nu": sigma_nu,
            "lam": lam,
            "gamma_hz": np.zeros((R, K)).tolist(),
            "carrier_rev_s": None,
            "profile": {
                "profile_db": profile.tolist(),
                "amp_exp": amp_exp,
                "mic_dev_sd_db": spec.mic_dev_sd_db,
            },
            "floor": dict(floor),
            "wind": None,
            "wander": dict(
                sigma_d_db=0.0,
                tau_d_s=1.0,
                sigma_v_db=0.0,
                tau_v_s=1.0,
                sigma_u_db=0.0,
                tau_u_s=1.0,
                sigma_uj_db=0.0,
                tau_uj_s=1.0,
                block_s=0.5,
            ),
            "am": am,
            "wind_sc": dict(spec.wind_sc),
            "array_response": None,
        },
    }


CAL_N_FFT = 2048


def line_contrast_db(payload: dict[str, Any], rps: float, orders: np.ndarray) -> np.ndarray:
    """``10 log10((L + B) / B)`` of the lines ``orders`` of a ONE-rotor payload at
    constant ``rps``: the model's expected periodogram (mic mean, 2048-point
    Hann at 16 kHz, the parameter view's frame) at the bin nearest ``k rps``,
    with the comb on (``L + B``) and off (``B``)."""
    from experiments.noise_model.render import expected_periodogram

    def one(fit: dict[str, Any]) -> np.ndarray:
        m = expected_periodogram(
            fit, np.full((1, CAL_N_FFT), float(rps)), n_fft=CAL_N_FFT, hop=CAL_N_FFT, sr=SR
        )
        return m[:, 0, :].mean(axis=0)

    off = {**payload, "params": {**payload["params"], "profile": {**payload["params"]["profile"]}}}
    off["params"]["profile"]["profile_db"] = (
        np.asarray(payload["params"]["profile"]["profile_db"]) * 0.0 - 300.0
    ).tolist()
    full, floor = one(payload), one(off)
    bins = np.clip(
        np.rint(np.asarray(orders) * rps / (SR / CAL_N_FFT)).astype(int), 0, full.size - 1
    )
    return 10.0 * np.log10(full[bins] / floor[bins])


def sample_prior(rng: np.random.Generator, spec: PriorSpec = PriorSpec()) -> dict[str, Any]:
    """One drone: a ``noise-v3-fit/1`` payload (plus ``_prior`` provenance)."""
    R, K = spec.n_rotors, spec.k_max
    k = np.arange(1, K + 1, dtype=np.float64)
    # --- comb law, line over floor (dB) ---
    slope = float(_n(rng, spec.slope_db_dec))
    k2, tail = float(_n(rng, spec.k2_over_floor_db)), float(_n(rng, spec.tail_over_floor_db))
    law = np.maximum(k2 - slope * np.maximum(np.log10(k / 2.0), 0.0), tail)
    law[1] += _n(rng, spec.bpf_boost_db)
    even = (k % spec.blades) == 0
    odd_pen = _n(rng, spec.odd_a_db) + _n(rng, spec.odd_b_db) * np.log10(k)
    law = np.where(even, law, law - odd_pen)
    p = int(rng.choice(spec.pole_pairs))
    period = 3 * p
    boost = _n(rng, spec.motor_boost_db)
    mult = (k % period == 0) & (k > 0)
    law[mult] += boost * spec.motor_decay_per_multiple ** (k[mult] / period - 1)
    # --- floor ---
    ctrl_hz = floor_ctrl_hz(SR)
    floor_mean = float(_n(rng, spec.floor_mean_db))
    sd_b = float(rng.uniform(*spec.floor_shape_sd_db))
    template, template_name = _template_ctrl_db(rng, spec, ctrl_hz)
    chol = floor_shape_chol(ctrl_hz)
    z = rng.standard_normal(FLOOR_SHAPE_N_CTRL) * 0.7 + np.linalg.solve(chol, template / sd_b)
    ctrl_db = floor_mean + sd_b * (chol @ z)
    floor = {
        "floor_mean_db": floor_mean,
        "floor_shape_z": z.tolist(),
        "floor_shape_sd_db": sd_b,
        "floor_exp": float(_ln(rng, spec.log_floor_exp)),
        "floor_static_rel": float(_ln(rng, spec.log_floor_static_rel)),
    }
    sigma_nu, lam = float(_ln(rng, spec.log_sigma_nu)), float(_ln(rng, spec.log_lam))
    amp_exp = float(_n(rng, spec.amp_exp))
    # --- to profile_db: the contrast of a flat profile at the floor's level gives the
    # per-order conversion (the floor is fixed, so one shift per order is exact) ---
    flat = _payload(
        spec,
        profile=np.full((1, K), floor_mean),
        sigma_nu=sigma_nu,
        lam=lam,
        amp_exp=amp_exp,
        floor=floor,
        am=None,
    )
    c_flat = line_contrast_db(flat, spec.cal_rps, np.arange(1, K + 1))
    over_flat = 10.0 * np.log10(np.maximum(10.0 ** (c_flat / 10.0) - 1.0, 1e-12))
    drone = floor_mean + law - over_flat
    # --- per-rotor scatter ---
    profile = (
        drone[None, :]
        + rng.normal(0.0, spec.rotor_line_sd_db, (R, K))
        + rng.normal(0.0, spec.rotor_level_sd_db, (R, 1))
    )
    # --- pedestal per order (shared by rotors up to a per-rotor jitter) ---
    sigma_psi = float(_ln(rng, spec.log_sigma_psi))
    # random part: AR(1) along same-parity neighbours (two chains), unit variance
    eps = rng.standard_normal(K)
    for j in range(2, K):
        eps[j] = spec.am_parity_rho * eps[j - 2] + np.sqrt(1 - spec.am_parity_rho**2) * eps[j]
    log_s2 = np.where(
        k <= 8,
        spec.log_am_s2_low[0] + spec.log_am_s2_low[1] * eps,
        np.where(
            k <= 30,
            spec.log_am_s2_mid[0] + spec.log_am_s2_mid[1] * eps,
            np.log(np.minimum(spec.am_s2_high_base + k**2 * sigma_psi**2, spec.am_s2_cap))
            + 0.3 * eps,
        ),
    )
    am_s2 = np.exp(log_s2)[None, :].repeat(R, 0) * np.exp(rng.normal(0.0, 0.2, (R, K)))
    am_g = np.where(
        k <= 8, _ln(rng, spec.log_am_gamma_low, K), _ln(rng, spec.log_am_gamma_high, K)
    )[None, :].repeat(R, 0)
    payload = _payload(
        spec,
        profile=profile,
        sigma_nu=sigma_nu,
        lam=lam,
        amp_exp=amp_exp,
        floor=floor,
        am={"sigma2": am_s2.tolist(), "gamma_hz": am_g.tolist()},
    )
    payload["_prior"] = {
        "spec": {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in asdict(spec).items()},
        "drawn": dict(
            slope_db_dec=slope,
            k2_over_floor_db=k2,
            tail_over_floor_db=tail,
            over_floor_db=law.tolist(),
            pole_pairs=p,
            motor_period=period,
            motor_boost_db=float(boost),
            floor_template=template_name,
            sigma_psi=sigma_psi,
            floor_ctrl_db=ctrl_db.tolist(),
        ),
    }
    return payload


TRAJ_FITS = "dload:rps-traj-fits"


def trajectory_windows(
    rng: np.random.Generator,
    n_windows: int,
    duration_s: float,
    *,
    sr: int = SR,
    hover_range: tuple[float, float] = (40.0, 95.0),
    rotor_sep: tuple[float, float] = (2.0, 15.0),
    full_flight: bool = False,
) -> list[np.ndarray]:
    """``n_windows`` airborne trajectories ``(R, T)`` rev/s at ``sr`` from the
    trajectory hyperprior (a fresh drone per window: the reserved rig name
    ``"posterior"`` of :mod:`data_processing.trajectory_model.source`)."""
    from data_processing.trajectory_model.source import FittedTrajectorySource, load_bundle

    src = FittedTrajectorySource(
        load_bundle(TRAJ_FITS),
        {"posterior": 1.0},
        measurement_noise=False,
        hover_range=hover_range,
        rotor_sep=rotor_sep,
    )
    fs_low = 100.0
    n_low = int(round(duration_s * fs_low))
    out = []
    while len(out) < n_windows:
        if full_flight:
            low = src.flight(rng, fs_low, duration_s=duration_s)
            clip = None
        else:
            draw = src.draw(rng)
            low = draw.params.sample_airborne(n_low, rng, fs=fs_low, clip=draw.clip)
            clip = draw.clip
        low = np.atleast_2d(np.asarray(low, dtype=np.float64))
        if clip is not None:
            # a hyperprior draw whose airborne process saturates at the ESC clamp for a
            # fifth of the window is a pinned, physically empty trajectory: redraw
            at_clamp = (low <= float(clip[0]) + 1e-6) | (low >= float(clip[1]) - 1e-6)
            if at_clamp.mean(axis=1).max() > 0.2:
                continue
        t_low = np.arange(low.shape[1]) / fs_low
        t_hi = np.arange(int(round(low.shape[1] / fs_low * sr))) / sr
        out.append(np.stack([np.interp(t_hi, t_low, row) for row in low]))
    return out


def gated_draws(
    rng: np.random.Generator,
    n: int,
    *,
    spec: PriorSpec = PriorSpec(),
    pool: Any,
    threshold_db: float = 1.0,
    min_orders: int = 1,
    pass_frac: float = 0.8,
    max_attempts: int = 200,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``n`` prior drones passing the identifiability gate
    (:mod:`experiments.noise_model.identifiability`, every rotor with at least
    ``min_orders`` harmonics ``threshold_db`` over everything else in
    ``pass_frac`` of its eligible frames of ``pool``), plus draw statistics."""
    from experiments.noise_model.identifiability import GateConfig, gate

    cfg = GateConfig(
        threshold_db=threshold_db, min_orders=min_orders, pass_frac=pass_frac, min_eligible_frames=5
    )
    out: list[dict[str, Any]] = []
    tried, reasons = 0, []
    while len(out) < n and tried < max_attempts:
        tried += 1
        payload = sample_prior(rng, spec)
        res = gate(payload, pool, cfg)
        if res.passed:
            payload["_prior"]["gate"] = dict(
                frames_ok=res.frames_ok.tolist(),
                eligible=res.eligible.tolist(),
                threshold_db=threshold_db,
            )
            out.append(payload)
        else:
            reasons.append(res.reason)
    return out, dict(tried=tried, accepted=len(out), reasons=reasons)
