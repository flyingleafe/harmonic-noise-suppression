"""The exact discrete-time renderer of the v2 model, plus its expected ``M``.

THE RENDERER ITSELF MOVED to :mod:`data_processing.noise_model.render` — a
training stream synthesises fitted rotor noise through
:class:`data_processing.noise_v2_pool.NoiseV2Pool`, and the import-linter
contract "nothing imports experiments" forbids reaching up from
``data_processing``. Every public name is re-exported here unchanged, so
``from experiments.noise_model import render as RD`` and ``RD.render_noise``,
``RD.render_noise_regimes``, ``RD.regime_blend_weight``, ``RD.regime_seeds``,
``RD.REGIME_ORDER``, ``RD.READABLE_SCHEMAS`` and ``RD._check_schema`` keep
meaning exactly what they meant. Read that module's docstring for what is
drawn and why.

WHAT STAYS HERE: the two EXPECTED-periodogram functions. They evaluate the v2
forward model (:mod:`.spectrum`'s torch flight model, via
:func:`.model.params_from_dict`), which is the fit's machinery and no part of
a data stream.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from data_processing.noise_model.params import check_schema as _check_schema
from data_processing.noise_model.render import *  # noqa: F403
from data_processing.noise_model.render import (
    READABLE_SCHEMAS,
    REGIME_ORDER,
    _regime_fits,
    fit_work_rate,
    regime_blend_weight,
    regime_seeds,
    render_noise,
    render_noise_regimes,
)

from . import model as MD
from . import spectrum as SP

__all__ = [
    "REGIME_ORDER",
    "READABLE_SCHEMAS",
    "expected_periodogram",
    "expected_periodogram_regimes",
    "fit_work_rate",
    "regime_blend_weight",
    "regime_seeds",
    "render_noise",
    "render_noise_regimes",
]


def expected_periodogram(
    fit: dict[str, Any],
    rps_rev_s: np.ndarray,
    *,
    n_fft: int = SP.FLIGHT_N_FFT,
    hop: int = SP.FLIGHT_HOP,
    sr: int = SP.FLIGHT_SR,
    n_mics: int = 8,
    frame_chunk: int = 16,
    sr_work: int | None = None,
) -> np.ndarray:
    """``(n_mics, n_frames, n_fft // 2 + 1)`` predicted ``M`` on a moving carrier.

    The v2 analogue of :func:`revised_phase.predict_spectrum`: the fitted comb
    and floor evaluated through the FLIGHT forward model on the real carrier,
    in ``data.periodogram`` units, with the frame grid
    ``arange(1 + (T - n_fft) // hop) * hop`` — ``data.periodogram``'s own
    framing, so bin ``j`` is ``j sr / n_fft`` and frame centre ``i`` is
    ``(start_i + n_fft / 2) / sr``, exactly the support's. ``sr_work`` is the
    kernel's work rate and its render-chain transfer; ``None``: the fit's own
    (:func:`fit_work_rate`), so the ``M`` is the one its likelihood read.
    """
    import torch

    p = _check_schema(fit)
    rps = np.atleast_2d(np.asarray(rps_rev_s, dtype=np.float64))
    n = int(rps.shape[1])
    if n < n_fft:
        raise ValueError(f"{n} samples is shorter than n_fft {n_fft}")
    starts = np.arange(1 + (n - int(n_fft)) // int(hop)) * int(hop)
    grid = SP.flight_grid(
        sr=sr, n_fft=n_fft, hop=hop, sr_work=fit_work_rate(fit) if sr_work is None else sr_work
    )
    params = MD.params_from_dict(p, n_mics=int(n_mics))
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    k_max = min(
        int(prof.shape[1]),
        SP.k_max_for_carrier(rps.max(axis=1), sr, k_cap=int(prof.shape[1])),
    )
    out = np.empty((int(n_mics), starts.size, int(n_fft) // 2 + 1), dtype=np.float64)
    with torch.no_grad():
        for c0 in range(0, starts.size, int(frame_chunk)):
            sl = slice(c0, min(c0 + int(frame_chunk), starts.size))
            rate = SP.flight_rate_work(grid, rps, starts[sl])
            m = SP.flight_model(grid, params, rate_work=rate, k_max=k_max)
            got = m.cpu().numpy()
            if got.shape[0] < n_mics:
                raise ValueError(f"fit carries {got.shape[0]} microphones, asked for {n_mics}")
            out[:, sl, :] = got[:n_mics]
    return out


def expected_periodogram_regimes(
    fits_by_regime: Mapping[str, dict[str, Any]],
    rps_rev_s: np.ndarray,
    *,
    n_fft: int = SP.FLIGHT_N_FFT,
    hop: int = SP.FLIGHT_HOP,
    sr: int = SP.FLIGHT_SR,
    n_mics: int = 8,
    frame_chunk: int = 16,
    sr_work: int | None = None,
) -> np.ndarray:
    """The composition's expected ``M``: the same power blend, per FRAME.

    ``(1 - w_i) M_standby + w_i M_cruise`` with ``w_i`` the composed weight at
    frame ``i``'s CENTRE sample, which is the expectation of what
    :func:`render_noise_regimes` draws (the two streams are independent, so
    their powers add) and therefore the predicted periodogram the likelihood
    gate must read for this candidate.
    """
    fits = _regime_fits(fits_by_regime)
    rps = np.atleast_2d(np.asarray(rps_rev_s, dtype=np.float64))
    w = regime_blend_weight(rps, sr=sr)
    starts = np.arange(1 + (int(rps.shape[1]) - int(n_fft)) // int(hop)) * int(hop)
    centres = np.clip(starts + int(n_fft) // 2, 0, w.size - 1)
    w_frame = w[centres]
    out: np.ndarray | None = None
    for regime, weight in (("standby", 1.0 - w_frame), ("cruise", w_frame)):
        if not bool(np.any(weight > 0.0)):
            continue
        m = expected_periodogram(
            fits[regime],
            rps,
            n_fft=n_fft,
            hop=hop,
            sr=sr,
            n_mics=n_mics,
            frame_chunk=frame_chunk,
            sr_work=sr_work,
        )
        term = np.asarray(m, dtype=np.float64) * weight[None, :, None]
        out = term if out is None else out + term
    if out is None:  # pragma: no cover - the weights sum to one by construction
        raise ValueError("no regime carries any weight on this carrier track")
    return out
