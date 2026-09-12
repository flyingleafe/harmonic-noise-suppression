"""Which of the fitted per-order amplitudes are DATA, and which are PRIOR.

The rig fit exports a profile value for every order it carries, and nothing in
that array says how much of each number the recording actually paid for. An
order whose line sits far under the broadband floor contributes almost no
likelihood, so its exported value is the hierarchical prior's mean plus
whatever the optimizer's noise left behind — and the flat hold above DREGON's
order 91 (§4.1 of the explainer) is the visible end of a gradient that starts
much lower.

This module puts a number on that. For one line of unit-area power ``P`` and
half width ``gamma`` sitting on a local floor density ``F``, the Whittle
likelihood's information about the line's LEVEL (in dB) is available in closed
form. Write the line's density as ``p L(f)`` with ``L`` a unit-peak Lorentzian
and ``p = P / (pi gamma)`` its peak, let ``rho = p / F`` be the peak-to-floor
ratio, and let the observation supply ``M`` microphones over ``T`` seconds. A
level change ``d`` dB scales the line's density by ``10**(d/10)``, so

    I = (ln10/10)^2 * sum_cells (line / (line + floor))^2
      = (ln10/10)^2 * M * T * gamma * pi * rho^2 / (2 (1 + rho)^(3/2))

using that ``T`` seconds of record carry ``T`` independent spectral cells per
hertz however the analysis is blocked (§13.3), and that the frequency integral
``int rho^2 / (rho + 1 + u^2)^2 du`` is ``pi rho^2 / (2 (1 + rho)^(3/2))``.

The posterior standard deviation is then ``1/sqrt(I)`` dB, to be read against
the prior it would otherwise sit at. Two regimes matter:

* ``rho >> 1`` — ``I`` grows only as ``sqrt(rho)``: a loud line is measured to a
  fraction of a dB and there is nothing to argue about.
* ``rho << 1`` — ``I`` falls as ``rho^2``, so the error grows as ``1/rho``. A
  line 10 dB under the floor in peak density is still worth about a prior
  standard deviation on a four-second eight-microphone clip; one 20 dB under is
  worth nothing at all. Visibility is not a threshold, it is this curve.

Nothing here is fitted. It reads an accepted population summary and reports
what that fit's own numbers imply about their own credibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

#: dB per natural log unit of power, squared — the (ln10/10)^2 above.
_DB = (np.log(10.0) / 10.0) ** 2


@dataclass(frozen=True)
class OrderInformation:
    """Per ``(clip, rotor, order)`` credibility of one fitted profile value."""

    margin_db: np.ndarray  # (C,R,K) peak line density over local floor
    post_std_db: np.ndarray  # (C,R,K) posterior std of the level, dB
    #: ``(K,)`` hierarchical residual std the order falls back to when the clip
    #: says nothing, where the population model carries one. DREGON's profile
    #: model has no per-order residual term, so this is optional and the
    #: absolute posterior std is the criterion that works on both rigs.
    prior_std_db: np.ndarray | None
    orders: np.ndarray  # (K,) order index
    clip_ids: tuple[str, ...]

    @property
    def contraction(self) -> np.ndarray | None:
        """``(C,R,K)`` posterior std over prior std, where a prior exists. 1
        means "we learned nothing about this order and returned the prior".

        This is the strictest reading and it has one trap: an order whose
        population prior is ALREADY sharper than one clip's likelihood scores
        badly here while being perfectly well measured. Michael's order 2 is
        the case — 0.08 dB of posterior width against a residual prior of
        0.001 dB, because every clip agrees about the blade-passing line.
        Absolute width is therefore the primary criterion below.
        """
        if self.prior_std_db is None:
            return None
        return np.minimum(self.post_std_db / self.prior_std_db[None, None, :], 1.0)

    def measured(self, max_std_db: float = 3.0) -> np.ndarray:
        """``(C,R,K)`` orders this clip pins to better than ``max_std_db``."""
        return self.post_std_db <= max_std_db

    def last_measured_order(self, max_std_db: float = 3.0) -> np.ndarray:
        """``(C,R)`` highest order that is measured, 0 if none is."""
        ok = self.measured(max_std_db)
        idx = np.where(ok.any(axis=2), ok.shape[2] - np.argmax(ok[:, :, ::-1], axis=2), 0)
        return idx


def line_information(
    *,
    power: np.ndarray,
    gamma_hz: np.ndarray,
    floor_density: np.ndarray,
    n_mics: int,
    seconds: float,
    line_shape: str = "lorentz",
) -> tuple[np.ndarray, np.ndarray]:
    """``(margin_db, post_std_db)`` for lines given as unit-area powers.

    ``power``, ``gamma_hz`` and ``floor_density`` broadcast against each other
    in linear units; ``power`` is the line's integrated power and
    ``floor_density`` the floor's power spectral density at the line.
    """
    if line_shape == "gauss":
        # unit-area Gaussian of half width at half maximum gamma
        peak = power * np.sqrt(np.log(2.0) / np.pi) / np.maximum(gamma_hz, 1e-12)
    else:
        peak = power / (np.pi * np.maximum(gamma_hz, 1e-12))
    rho = peak / np.maximum(floor_density, 1e-300)
    info = _DB * n_mics * seconds * gamma_hz * np.pi * rho**2 / (2.0 * (1.0 + rho) ** 1.5)
    with np.errstate(divide="ignore"):
        std = 1.0 / np.sqrt(np.maximum(info, 1e-300))
    return 10.0 * np.log10(np.maximum(rho, 1e-300)), std


def _floor_density_db(
    params: dict[str, Any], order_hz: np.ndarray, speed: np.ndarray
) -> np.ndarray:
    """``(R,K)`` floor PSD in dB at each rotor's own line frequencies."""
    from .model import AMP_RPS_REF

    shape = np.interp(
        np.log2(np.maximum(order_hz, 1.0)),
        np.log2(np.asarray(params["floor_ctrl_hz"], dtype=np.float64)),
        np.asarray(params["floor_shape_db"], dtype=np.float64),
    )
    tilt = float(params["floor_tilt_db_oct"]) * np.log2(np.maximum(order_hz, 1.0) / 500.0)
    db = float(params["floor_mean_db"]) + shape + tilt
    gain = np.mean((np.maximum(speed, 0.0) / AMP_RPS_REF) ** float(params["floor_exp"])) + float(
        params["floor_static_rel"]
    )
    return db + 10.0 * np.log10(max(gain, 1e-30))


def order_information(
    summary: dict[str, Any],
    split: str,
    *,
    speeds: dict[str, np.ndarray],
    line_shape: str = "lorentz",
) -> OrderInformation:
    """Read a population summary and score every fitted profile value.

    ``speeds`` maps clip id to that clip's ``(R,)`` median rev/s — the fit's
    own carrier, which decides both where each line sits and how the speed law
    scales it. A clip missing from ``speeds`` is skipped.
    """
    from .model import AMP_RPS_REF

    clips = summary.get(split) or {}
    prior_raw = summary.get("profile_residual_std_db")
    prior = None if prior_raw is None else np.atleast_1d(np.asarray(prior_raw, dtype=np.float64))
    margins: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    ids: list[str] = []
    for clip_id, clip in clips.items():
        speed = speeds.get(clip_id)
        if speed is None:
            continue
        params = clip["params"]
        profile = np.asarray(params["profile_db"], dtype=np.float64)  # (R,K)
        gamma = np.asarray(params["gamma"], dtype=np.float64)  # (R,K)
        knots = np.asarray(params["knots_s"], dtype=np.float64)
        seconds = float(knots[-1] - knots[0]) or 1.0
        n_mics = int(np.asarray(params["mic_gain_db"]).shape[0])
        k = np.arange(1, profile.shape[1] + 1, dtype=np.float64)
        order_hz = np.asarray(speed, dtype=np.float64)[:, None] * k[None, :]
        power = (
            10.0 ** (profile / 10.0)
            * ((np.maximum(np.asarray(speed), 0.0) / AMP_RPS_REF) ** float(params["amp_exp"]))[
                :, None
            ]
        )
        floor = 10.0 ** (_floor_density_db(params, order_hz, np.asarray(speed)) / 10.0)
        margin, std = line_information(
            power=power,
            gamma_hz=gamma,
            floor_density=floor,
            n_mics=n_mics,
            seconds=seconds,
            line_shape=line_shape,
        )
        margins.append(margin)
        stds.append(std)
        ids.append(clip_id)
    if not margins:
        raise ValueError("no clip of the summary had a speed supplied")
    k_min = min(m.shape[1] for m in margins)
    return OrderInformation(
        margin_db=np.stack([m[:, :k_min] for m in margins]),
        post_std_db=np.stack([s[:, :k_min] for s in stds]),
        prior_std_db=(
            None
            if prior is None
            else (np.broadcast_to(prior, (k_min,)) if prior.size == 1 else prior[:k_min])
        ),
        orders=np.arange(1, k_min + 1),
        clip_ids=tuple(ids),
    )


def information_summary(info: OrderInformation, *, max_std_db: float = 3.0) -> dict[str, Any]:
    """The reportable digest: where the claim dies, per rotor and per order."""
    measured = info.measured(max_std_db)
    last = info.last_measured_order(max_std_db)
    per_order = measured.reshape(-1, measured.shape[2]).mean(axis=0)
    contraction = info.contraction
    return {
        "prior_std_db_median": (
            None if info.prior_std_db is None else float(np.median(info.prior_std_db))
        ),
        "beats_prior_fraction": (
            None if contraction is None else float((contraction <= 0.5).mean())
        ),
        "max_std_db": max_std_db,
        "n_clips": int(measured.shape[0]),
        "n_rotors": int(measured.shape[1]),
        "n_orders": int(measured.shape[2]),
        "measured_fraction": float(measured.mean()),
        "measured_orders_per_rotor": {
            "median": float(np.median(measured.sum(axis=2))),
            "min": int(measured.sum(axis=2).min()),
            "max": int(measured.sum(axis=2).max()),
        },
        "last_measured_order": {
            "median": float(np.median(last)),
            "min": int(last.min()),
            "max": int(last.max()),
        },
        "rotors_with_no_measured_order": int((last == 0).sum()),
        "fraction_measured_by_order": [float(v) for v in per_order],
        "median_margin_db_by_order": [
            float(v) for v in np.median(info.margin_db.reshape(-1, measured.shape[2]), axis=0)
        ],
        "median_post_std_db_by_order": [
            float(v) for v in np.median(info.post_std_db.reshape(-1, measured.shape[2]), axis=0)
        ],
    }


# ── the marginal version ─────────────────────────────────────────────────────
#
# Everything above treats one line as if the floor under it were KNOWN and no
# other line existed. Neither holds, and both failures push the same way:
#
# * the floor's level, its 14-knot shape, its tilt, its slow drift and its
#   per-microphone pattern are all fitted, so a line's excess competes with a
#   floor free to rise and explain it;
# * order k of rotor r sits ``k * (s_r - s_r')`` away from the same order of
#   its neighbour, and the rotors of a real flight differ by ~13.7 rev/s only
#   on average — a pair can be much closer. Once a width reaches that spacing
#   the two lines are ONE feature and no record length separates them. This is
#   what decides whether a PER-ROTOR profile is identifiable, and the isolated
#   calculation above cannot see it at all.
#
# The marginal information is the Schur complement: assemble the Fisher matrix
# over every line level AND every floor latent, invert, read the diagonal. For
# the Whittle likelihood of an exponential periodogram of mean ``M``,
#
#     J_ij = sum_cells (dlog M / dtheta_i)(dlog M / dtheta_j)
#
# and for a level in dB ``dlog M / da`` is ``ln10/10`` times that component's
# SHARE of ``M`` — small exactly where the line is buried, and nearly parallel
# between two lines that overlap, which is what makes the matrix singular.


@dataclass(frozen=True)
class MarginalInformation:
    """Per ``(rotor, order)`` marginal posterior width for one clip."""

    clip_id: str
    std_db: np.ndarray  # (R,K) marginal std of the line level, dB
    isolated_std_db: np.ndarray  # (R,K) the same with everything else known
    prior_std_db: float
    n_floor_params: int

    @property
    def cost_of_sharing_db(self) -> np.ndarray:
        """The width that comes from sharing the spectrum, not from the noise."""
        return self.std_db - self.isolated_std_db

    def identifiable(self, max_std_db: float = 3.0) -> np.ndarray:
        return self.std_db <= max_std_db


def _unit_line(offset_hz: np.ndarray, gamma_hz: float, *, line_shape: str) -> np.ndarray:
    """Unit-area line density at signed offsets from the centre."""
    if line_shape == "gauss":
        sigma = gamma_hz / np.sqrt(2.0 * np.log(2.0))
        return np.exp(-0.5 * (offset_hz / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
    return gamma_hz / (np.pi * (offset_hz**2 + gamma_hz**2))


def marginal_information(
    params: dict[str, Any],
    *,
    freqs: np.ndarray,
    times: np.ndarray,
    rps: np.ndarray,
    n_mics: int,
    clip_id: str = "",
    line_shape: str = "lorentz",
    prior_std_db: float = 10.0,
    support_widths: float = 12.0,
    f_min: float = 30.0,
) -> MarginalInformation:
    """Marginal information about every line level of one fitted clip.

    ``prior_std_db`` is a deliberately weak prior on the line levels, present
    only to keep the matrix invertible where the likelihood is singular — and
    that is the finding rather than a nuisance: a direction the data cannot see
    comes back at the prior width, so a reported std at the prior means the
    recording said nothing about that order.
    """
    from .model import AMP_RPS_REF

    freqs = np.asarray(freqs, dtype=np.float64)
    rps = np.asarray(rps, dtype=np.float64)
    profile_db = np.asarray(params["profile_db"], dtype=np.float64)
    gamma = np.asarray(params["gamma"], dtype=np.float64)
    n_rotors, n_harm = profile_db.shape
    n_freq = freqs.size
    df = float(freqs[1] - freqs[0])
    nyquist = float(freqs[-1])

    # ── the fitted mean spectrum, rebuilt in numpy ───────────────────────────
    ctrl_hz = np.asarray(params["floor_ctrl_hz"], dtype=np.float64)
    shape_db = np.asarray(params["floor_shape_db"], dtype=np.float64)
    f_oct = np.log2(np.maximum(freqs, f_min) / ctrl_hz[0])
    ctrl_oct = np.log2(ctrl_hz / ctrl_hz[0])
    weights = np.zeros((n_freq, ctrl_hz.size))
    idx = np.clip(np.searchsorted(ctrl_oct, f_oct) - 1, 0, ctrl_oct.size - 2)
    frac = np.clip((f_oct - ctrl_oct[idx]) / np.maximum(np.diff(ctrl_oct)[idx], 1e-12), 0.0, 1.0)
    weights[np.arange(n_freq), idx] = 1.0 - frac
    weights[np.arange(n_freq), idx + 1] = frac
    tilt_oct = np.log2(np.maximum(freqs, 1.0) / 500.0)

    knots = np.asarray(params["knots_s"], dtype=np.float64)
    t_rel = np.asarray(times, dtype=np.float64) - float(times[0])
    level_frame = np.interp(t_rel, knots, np.asarray(params["floor_level_db"], dtype=np.float64))
    tilt_frame = float(params["floor_tilt_db_oct"]) + np.interp(
        t_rel, knots, np.asarray(params["floor_tilt_gp"], dtype=np.float64)
    )
    speed = np.maximum(rps, 0.0) / AMP_RPS_REF
    floor_gain = (speed ** float(params["floor_exp"])).mean(axis=0) + float(
        params["floor_static_rel"]
    )
    mic_floor_db = np.asarray(params["mic_floor_db"], dtype=np.float64)
    mic_gain_db = np.asarray(params["mic_gain_db"], dtype=np.float64)  # (M,R)
    gain_raw = params.get("gain_all_db")
    gain_all_db = np.zeros(n_mics) if gain_raw is None else np.asarray(gain_raw, dtype=np.float64)

    floor_db = (
        float(params["floor_mean_db"])
        + (weights @ shape_db)[None, :]
        + level_frame[:, None]
        + tilt_frame[:, None] * tilt_oct[None, :]
    )
    floor = 10.0 ** (floor_db / 10.0) * floor_gain[:, None]
    floor = floor[None] * 10.0 ** ((mic_floor_db + gain_all_db)[:, None, None] / 10.0)

    line_power = (
        10.0 ** (profile_db / 10.0)[:, :, None] * (speed ** float(params["amp_exp"]))[:, None, :]
    )  # (R,K,N)
    mic_line = 10.0 ** ((mic_gain_db + gain_all_db[:, None]) / 10.0)  # (M,R)

    # ── design rows ──────────────────────────────────────────────────────────
    # A line's row is nonzero only over its own support, so rows are kept as
    # (frequency slice, values) and the matrix is filled block by block.
    k = np.arange(1, n_harm + 1, dtype=np.float64)
    centers = k[None, :, None] * rps[:, None, :]  # (R,K,N)
    total = floor.copy()
    rows: list[tuple[int, int, np.ndarray]] = []
    for r in range(n_rotors):
        for i in range(n_harm):
            g = max(float(gamma[r, i]), df)
            half = max(support_widths * g, 3.0 * df)
            lo = int(np.clip(np.floor((centers[r, i].min() - half) / df), 0, n_freq - 1))
            hi = int(np.clip(np.ceil((centers[r, i].max() + half) / df) + 1, lo + 1, n_freq))
            off = freqs[None, lo:hi] - centers[r, i][:, None]  # (N,dF)
            live = (centers[r, i] > df) & (centers[r, i] < nyquist)
            contrib = (
                mic_line[:, r][:, None, None]
                * (line_power[r, i] * live)[None, :, None]
                * _unit_line(off, g, line_shape=line_shape)[None, :, :]
            )  # (M,N,dF)
            total[:, :, lo:hi] += contrib
            rows.append((lo, hi, contrib))

    inv_total = 1.0 / np.maximum(total, 1e-300)
    c = np.log(10.0) / 10.0
    n_lines = n_rotors * n_harm
    scaled = [(lo, hi, v * inv_total[:, :, lo:hi]) for lo, hi, v in rows]

    floor_rows: list[np.ndarray] = [floor * inv_total, floor * tilt_oct * inv_total]
    floor_rows += [floor * weights[:, j] * inv_total for j in range(ctrl_hz.size)]
    for m in range(n_mics):
        row = np.zeros_like(floor)
        row[m] = floor[m] * inv_total[m]
        floor_rows.append(row)
    n_floor = len(floor_rows)

    n_par = n_lines + n_floor
    fisher = np.zeros((n_par, n_par))
    for a in range(n_lines):
        lo_a, hi_a, va = scaled[a]
        for b in range(a, n_lines):
            lo_b, hi_b, vb = scaled[b]
            lo, hi = max(lo_a, lo_b), min(hi_a, hi_b)
            if hi <= lo:
                continue
            x = va[:, :, lo - lo_a : hi - lo_a]
            y = vb[:, :, lo - lo_b : hi - lo_b]
            fisher[a, b] = fisher[b, a] = float(np.einsum("mnf,mnf->", x, y))
    for j, fr in enumerate(floor_rows):
        for a in range(n_lines):
            lo_a, hi_a, va = scaled[a]
            fisher[a, n_lines + j] = fisher[n_lines + j, a] = float(
                np.einsum("mnf,mnf->", va, fr[:, :, lo_a:hi_a])
            )
        for j2 in range(j, n_floor):
            fisher[n_lines + j, n_lines + j2] = fisher[n_lines + j2, n_lines + j] = float(
                np.einsum("mnf,mnf->", fr, floor_rows[j2])
            )
    fisher *= c**2

    # An order out of band has exactly zero information, so its isolated width
    # is an infinity that only clutters a report. The prior is the ceiling of
    # anything the model would ever state, so both widths are capped there.
    isolated = np.minimum(
        np.sqrt(1.0 / np.maximum(np.diag(fisher)[:n_lines], 1e-300)), prior_std_db
    )
    # The prior is put on the LINE levels only, so the reported width is the
    # cost of sharing the spectrum with a free floor, not a prior's opinion
    # about that floor.
    ridge = np.zeros(n_par)
    ridge[:n_lines] = 1.0 / prior_std_db**2
    cov = np.linalg.inv(fisher + np.diag(ridge))
    std = np.sqrt(np.maximum(np.diag(cov)[:n_lines], 0.0))
    return MarginalInformation(
        clip_id=clip_id,
        std_db=std.reshape(n_rotors, n_harm),
        isolated_std_db=isolated.reshape(n_rotors, n_harm),
        prior_std_db=prior_std_db,
        n_floor_params=n_floor,
    )
