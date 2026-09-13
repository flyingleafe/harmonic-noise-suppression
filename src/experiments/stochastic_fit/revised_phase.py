"""Candidate 1 of the revised rotor-phase model: ONE shared shaft process per
rotor plus ONE independent acoustic phase diffusion, fitted by ordinary Adam on
a single plug-in composite MAP objective.

THE MODEL. For rig rotors ``r``, clips ``c``, orders ``k``, mics ``m``::

    f0_r(t)  = raw_r(t - delay_r) + b_{c,r}                        [Hz]
    dnu_r    = -lam nu_r dt + sqrt(2 lam sigma^2) dW_r             [nu in rad/s]
    dtheta_r = nu_r dt                                     [rad, theta_r(0) = 0]
    deps_rk  = sqrt(2 D) dW_rk              [independent per (r,k), source-side]
    Phi_r(t) = 2 pi Int_0^t f0_r + theta_r(t)
    x_m(t)   = sum_{r,k} A_{mrk}(t) cos(k Phi_r(t) + eps_rk(t) + alpha_mrk)
               + u_m(t)

``lam`` and ``sigma`` are rig-shared and FROZEN (moment-estimated,
:func:`estimate_shaft_dynamics`); ``D`` is ONE scalar per rig, initialized from
the residual moments and then fitted in the spectral MAP. ``theta_r`` is shared
across all orders and all mics; ``eps_rk`` is shared across mics (it is a source
property) and independent across ``(r, k)``. The initial residual and source
phases are independent uniform per harmonic, which is why every component adds
in POWER and why the fixed mic response phase ``alpha_mrk`` never enters the
expected periodogram. ``eps`` never enters the physical shaft speed.

``A_{mrk}`` and ``u_m`` are the EXISTING harmonic power profile, speed law, mic
gains and floor of :mod:`experiments.stochastic_fit.model` (same functional
forms, constants imported from there, GP drift terms off), with one recording
gain shared across regimes and no free per-clip level — a per-clip level
absorbs the speed law.

THE AMPLITUDE CONSTANT, VERIFIED NOT REMEMBERED. The atom amplitude is
``A_rk(t) = sqrt(2 P_rk(t))`` with ``P`` the line power of
:meth:`model.CombSpectrum.line_power`'s functional form, so a stationary tone's
mean-square power is ``A^2 / 2 = P``. Through the periodic Hann window of
``data.periodogram`` that makes an on-bin tone's peak periodogram exactly
``A^2 n_fft / 6`` (``(A/2)^2 (sum w)^2 / sum w^2`` with ``sum w = n/2`` and
``sum w^2 = 3n/8``), which is the identity the level initialization inverts.
Checked numerically against the kernel and against a Monte-Carlo mean
periodogram over random initial phases: the kernel reproduces the brute-force
double sum to 6.1e-16 relative, matches the Monte-Carlo mean of a
fractional-bin tone to 5.8e-6 (40k draws, 2 s.e. = 0.13) and of a planted chirp
with a diffusing phase to 1.2e-2 (2 s.e. = 8.1e-2), and the planted tone's
measured mean square is 0.7062 against ``P = 0.7``. NOTE that ``P`` here is a
MEAN-SQUARE power, not :mod:`model`'s per-hertz density weight, so
``profile_db`` is not numerically interchangeable with the legacy exports'.

THE LEGACY CONVERSION, EXPLICITLY. The two conventions are related by a fixed
factor, so nobody has to re-derive it::

    P_candidate = 2 * P_legacy_physical / fs_analysis       (because A = sqrt(2 P))

with ``fs_analysis = 16000 Hz`` the analysis SAMPLE RATE (``SR``), i.e. the rate
``data.periodogram`` reads at. It is the sample rate and NOT a band edge: not
7900 Hz and not 8000 Hz — using a band edge here produces a factor-two error. A
candidate export must NEVER be seeded from a legacy export without applying
this conversion explicitly.

THE FLOOR'S CONVENTION. ALL physical floor parameters (``floor_mean_db``,
``floor_shape_db``, ``floor_tilt_db_oct``, ``floor_exp``, ``floor_static_rel``,
``mic_floor_db``) are defined at the 16 kHz ANALYSIS convention, i.e. as read
by ``data.periodogram`` at ``sr = SR``. ONE function,
:func:`floor_power_spectrum`, turns them into a power spectrum on the work
grid — the fit reads it to build the floor's covariance and the renderer shapes
its white noise with its square root, so the two cannot drift — and it carries
the ``sample_rate_work / sample_rate`` RATE factor, because broadband noise
shaped on the work grid reads that much lower once decimated. That scalar alone
was never enough: the frequency-dependent part of the chain is the known
transfer below.

THE FLOOR IS OBSERVED THROUGH THE SAME KERNEL AS THE LINES. The coloured
Gaussian floor is treated as the real part of a PROPER COMPLEX PROCESS: a
complex process whose covariance is ``2 R_floor`` has a real part of covariance
``R_floor``. So
:func:`~experiments.stochastic_fit.phase_kernel.expected_periodogram_from_atoms`
is called with the REAL atom ``a_t = w(t) A_floor(t)`` on the work grid and
``r_tau = 2 R_floor(tau)`` on the work lag grid, at the same oversampling and
the same transfer normalization as the lines. THE DERIVATION, in one line: with
a real atom the kernel's image fold makes ``P_Z(-f) = P_Z(f)``, so
``M = 0.5 P_Z / sum(w^2)``, and the factor two turns that into
``sum_{t,s} w_t w_s A_t A_s R_floor(t-s) e^{-2 pi i f (t-s)} / sum(w^2)`` —
the EXACT expected Hann periodogram of the floor. It carries the window's
response on a coloured spectrum AND the within-window variation of the
amplitude envelope (the nonlinear ``floor_exp`` inside a ramp window is
evaluated sample by sample, exactly as the renderer applies it, never as an
exponent of a rectangular mean rate), with no dense ``(T, T)`` operator, no
extra stochastic state and no three-tap approximation. ``R_floor`` is the
inverse FFT of that same floor power spectrum, recomputed inside the graph at
every optimization step because it depends on the fitted shape; the LEVEL lives
in ``A_floor`` (which carries ``sqrt(c(0))``) and the SHAPE in
``r_tau = 2 c(tau) / c(0)``.

THE OBSERVATION LAW: ONE WORK GRID, ONE KNOWN TRANSFER. Atoms are synthesized
on a DECLARED integer-oversampled grid ``sample_rate_work`` (default
64000 = 4 x 16000). The word "native" is retired as ambiguous: this is a MODEL
grid, and real-data sampling and ``clips.decimate`` are untouched by it. The
window there is the periodic Hann of length
``n_fft_work = n_fft * sample_rate_work / sr``, so
``Fs_work / n_fft_work == Fs / n_fft``: the kernel's bins ARE the analysis bins
and the analysis band is the HEAD slice (the first ``n_fft // 2 + 1`` bins),
never every fourth bin. The work-grid periodogram is converted to analysis
units by ``Fs / Fs_work``, because the kernel divides by ``sum(w_work^2)`` and
a tone's kernel output scales with the window length.

The synthetic arm passes through the render anti-alias FIR designed at
``sample_rate_work`` and then the unchanged ``resample_poly`` decimator, so the
expected spectrum is multiplied — EXACTLY ONCE, on one shared code path
(:meth:`_RevisedModel.frame_model`, which :func:`predict_spectrum` also uses) —
by the known power transfer :func:`stage2.render_transfer_power`, evaluated on
the analysis grid and cached per fit. It multiplies the line shapes AND the
floor, because both pass through the same chain. Attribution: the AA is flat to
0.000 dB through 7900 Hz within its 0.01 dB ripple spec and -100.4 dB at
8000 Hz; the ~4.9 dB at 7900 Hz is the ``resample_poly`` decimator's rolloff,
NOT the AA. The composite is their product.

That multiplier is a WINDOW/FILTER-COMMUTATION APPROXIMATION, not an exact
moving-filtered periodogram: in the real renderer the filtering PRECEDES the
Hann window, while here it scales an already-windowed expected spectrum. The
finite-window kernel itself remains exact. No dense covariance operator and no
"exact filtered periodogram" machinery is added here; the campaign's accuracy
gate decides whether the approximation stands.

WHY APPLYING IT INSIDE THE FIT IS SELF-CONSISTENT. The fit sees real 16 kHz
data ``D`` and models ``M = T * S(params)``, so it learns ``S`` with
``T * S ~ D``. The renderer synthesizes ``S`` on the work grid and then
PHYSICALLY applies ``T``, so the synthetic arm reproduces ``D`` in the analysis
band whatever the real clip's own 44.1 kHz history was. The consequence, stated
plainly rather than left for a reader to trip over: ``S`` — and therefore
``profile_db`` — is NOT a calibrated absolute source spectrum. It is the
source-side quantity of THIS declared observation chain, comparable only across
exports that declare the same chain.

NO ORDER IS EVER DROPPED. Every order ``k = 1..k_cap`` contributes its actual
within-window time-varying atom in every frame. Out-of-band content is removed
by the work-grid synthesis and the transfer — the physical mechanism — never by
a whole-frame mask on a frame-mean carrier, which was discontinuous exactly at
a band-edge crossing. A carrier whose order reaches the WORK Nyquist is outside
the model's domain and RAISES, in the fit and in the renderer alike; it is
never silently truncated.

DELAY. Candidate 1 runs with ``delay_s = [0, 0, 0, 0]``, because the published
frames already apply the documented audio/telemetry alignment; no free delay is
to be introduced unless a later diagnostic motivates one. ``FitConfig.delay_s``
stays REQUIRED with no invented default — the manifest supplies the zeros
explicitly, so the choice stays visible.

WHAT IS FITTED, AND WHAT THE OBJECTIVE IS. One objective, one prior::

    L = (1 / temperature) * sum_i a_i * sum_{m, f in band} (I_i / M_i + log M_i)
        + 0.5 * sum e^2                              (exact integrated-OU state)
        + bias prior + parameter priors

This is a MARGINAL COMPOSITE RISK / plug-in composite MAP. It is NOT an exact
joint NLL, not an evidence, not a posterior; overlapping windows reuse the same
samples, so frame counts are NOT evidence counts and no uncertainty may be read
off cellwise Hessians or the nominal frame count. ``temperature`` DIVIDES the
composite risk. Its frozen scalar value is ``T = J / H`` with
``U = sum_i a_i (1 - I / M)`` (the untempered FIT-risk log-gain score),
``H = sum_i a_i`` (the total exposure) and ``J = Var(U)`` over fixed baseline
predictive draws, so the objective is ``L / T + one prior``. The calibration is
matched to the actual UNNORMALIZED risk returned by :func:`composite_risk` —
never to a reported per-second or per-frame score. It is calibrated on
planted/baseline data and then frozen — this code reads it, it never tunes it.
Weights ``a_i = hop / n_fft`` are normalized per unique audio duration and
identical duplicated windows SPLIT their exposure (:func:`composite_weights`).

COST ARITHMETIC (so the next reader can check feasibility without rerunning
it). One call of :func:`~experiments.stochastic_fit.phase_kernel.expected_periodogram_from_atoms`
per (rotor, order chunk, frame) — NOT per mic, because the mic gain is a
time-invariant scalar multiplying the per-(r, k, frame) shape — on the WORK
grid, so every tensor is ``sample_rate_work / sr`` = 4x longer than it was on
the analysis grid. With ``frame_chunk = 1``, ``harmonic_chunk = 32``, R = 4,
``n_fft = 16384``, ``n_fft_work = 65536`` and float32 atoms, ONE chunk holds:
wrapped phase float64 ``4*32*65536*8 B = 67 MB`` (freed after the cast),
amplitude/atoms complex64 ``67 MB``, and inside the kernel four complex64
buffers of ``128 x 131072`` entries, ``134 MB`` each — ~0.6-0.8 GB live,
INDEPENDENT of ``k_cap``, because ``torch.utils.checkpoint`` frees each chunk's
interior and recomputes it in the backward pass (the pattern
:meth:`model.CombSpectrum.lines` already uses, at the same chunk of 32).
Unchunked (``harmonic_chunk = None``) the same tensors scale by ``k_cap / 32``:
~5-6 GB at ``k_cap = 230``, which is why the default is chunked. NO ORDER IS
EVER DROPPED to fit memory and ``k_cap`` is never lowered for it: memory is an
implementation problem. The floor costs ONE more kernel call per frame chunk (a
real atom, one ``(n_c, 131072)`` FFT pair) plus one inverse FFT of
``FLOOR_PSD_OVERSAMPLE * n_fft_work`` per step for ``R_floor``: negligible
beside the lines. Work per frame is ``2 R K`` FFTs of length ``2 n_fft_work``
(``1040 x 131072 log 131072`` ~ 2.3e9 flops at ``k_cap = 130``), so the fit is
FFT-bound and linear in frames x orders; checkpointing recomputes the forward
once, about a third more time.

NOT IN THIS MODULE, deliberately: no EM, no Kalman smoother, no particle
filter, no Laplace approximation, no crossing graph, no per-harmonic latent
inference, no second carrier spline, no free ``D`` per order, no acceptance
scoring (that is the evaluator's).
"""

from __future__ import annotations

import dataclasses
import math
import subprocess
import time
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.checkpoint
from torch import Tensor

from .data import Clip, periodogram
from .model import (
    AMP_RPS_REF,
    FLOOR_SHAPE_F_MIN,
    FLOOR_SHAPE_N_CTRL,
    FLOOR_TILT_REF_HZ,
    Spec,
    interp_matrix,
    se_cholesky,
)
from .phase_kernel import BAND_HZ, HOP, N_FFT, SR, expected_periodogram_from_atoms, hann_window
from .phase_stats import _runs as _frame_runs
from .phase_stats import stft_at
from .stage2 import RENDER_TRANSFER_SPEC, antialias, render_transfer_power

SCHEMA_VERSION = 1
MODEL_FAMILY = "shared_shaft_ou"

#: The same guard :meth:`model.CombSpectrum.whittle` uses, so a cell of the
#: revised risk and a cell of the legacy Whittle term are floored identically.
MODEL_FLOOR = 1e-12

#: The DECLARED integer-oversampled synthesis/analysis grid: 4 x the 16 kHz
#: analysis rate, exactly (never 65536, which is not an integer multiple).
#: Atoms are synthesized here and the expected spectrum is read back on the
#: analysis grid; real-data sampling and ``clips.decimate`` are untouched.
SAMPLE_RATE_WORK = 64000

#: ONE positive speed floor, shared by the prediction and the renderer. The
#: speed laws carry FITTED exponents, so a stopped rotor at exactly 0 raises 0
#: to an unconstrained power: non-finite audio on one path, a finite prediction
#: on the other. Both paths clamp here instead. It is a constant, not a knob.
SPEED_FLOOR_RPS = 1e-6

#: The CALIBRATED speed domain of the fitted speed laws: rotors at or above
#: 20 rev/s, i.e. standby, ramp and cruise. Below it the fitted exponents are
#: an extrapolation nothing in this campaign measured, and a stopped rotor is
#: NOT modelled. :func:`off_regime_extrapolation` REPORTS a trajectory that
#: goes there; :data:`SPEED_FLOOR_RPS` exists to prevent infinities, not to
#: pretend the model covers a motors-off rotor.
CALIBRATED_MIN_RPS = 20.0

#: The floor's autocovariance is read from its power spectrum on a frequency
#: grid this many times finer than the work window, so the inverse FFT's
#: periodic images sit at lags of at least three window lengths and the lags
#: ``0..n_fft_work-1`` the kernel reads carry none of them. At the campaign's
#: front end that lag grid is 4 s against a floor whose slowest structure is
#: the 30 Hz shape-control floor (~33 ms).
FLOOR_PSD_OVERSAMPLE = 4

_SPEC_DEFAULTS = {f.name: f.default for f in dataclasses.fields(Spec)}
#: Floor-shape prior scale and correlation length, and the speed-law exponent
#: initialization: read from :class:`model.Spec`'s own defaults rather than
#: copied, so the revised floor is the existing floor.
FLOOR_SHAPE_STD_DB = float(_SPEC_DEFAULTS["floor_shape_std_db"])
FLOOR_SHAPE_OCT = float(_SPEC_DEFAULTS["floor_shape_oct"])
AMP_EXP_INIT = float(_SPEC_DEFAULTS["amp_rps_exponent"])

# ── preregistered moment gate ────────────────────────────────────────────────
# PREREGISTERED: these five numbers were fixed on 2026-09-12, BEFORE any moment
# estimate was computed on any clip, and they are not to be tuned against an
# estimate. A line that fails the gate is EXCLUDED from the moment stage — in
# particular an unresolved overlap is excluded, never allocated. The NUMBERS
# are unchanged; where they are applied was corrected: the SNR threshold is
# tested per MICROPHONE and per FRAME (:func:`line_snr_db`), never on a
# mic-average or on the track's median, and lag pairs are formed only inside a
# continuous run of valid frames (:func:`valid_lag_pairs`).
#: minimum line SNR over the LOCAL floor, in dB, required of EACH (mic, frame)
GATE_MIN_LINE_SNR_DB = 10.0
#: minimum distance to the nearest competing line of any rotor/order, in bins
#: of the MOMENT front end (its own window, not the frozen spectral one)
GATE_MIN_ISOLATION_BINS = 4.0
#: orders considered at all (inclusive): low enough that k^2 sigma^2 broadening
#: does not swamp the line, high enough that the k^2 scaling is visible
GATE_ORDER_RANGE = (2, 40)
#: minimum number of USABLE samples in a (clip, rotor, order) track: gated
#: (mic, frame) cells for the track, and surviving pairs for a (k, lag) cell —
#: never the clip's total frame count
GATE_MIN_FRAMES = 32
#: a (k, lag) cell whose wrapped phase-increment residuals spread wider than
#: this is CENSORED by wrapping, so its variance is not an estimate of anything
GATE_MAX_WRAP_SPREAD_RAD = 1.2

#: Broad proper prior on the per-rotor population mean bias. 5 Hz is wide
#: against both rigs' measured telemetry-offset scales (0.41 rev/s on Michael's
#: crops, 1.38 on DREGON's, quoted in :mod:`model`'s BASE_VARIANT comment).
BIAS_MEAN_PRIOR_STD_HZ = 5.0


@dataclass(frozen=True)
class MomentGate:
    """The preregistered moment-stage gate (see the module constants)."""

    min_line_snr_db: float = GATE_MIN_LINE_SNR_DB
    min_isolation_bins: float = GATE_MIN_ISOLATION_BINS
    order_range: tuple[int, int] = GATE_ORDER_RANGE
    min_frames: int = GATE_MIN_FRAMES
    max_wrap_spread_rad: float = GATE_MAX_WRAP_SPREAD_RAD

    def as_dict(self) -> dict[str, Any]:
        return dict(
            min_line_snr_db=self.min_line_snr_db,
            min_isolation_bins=self.min_isolation_bins,
            order_range=list(self.order_range),
            min_frames=self.min_frames,
            max_wrap_spread_rad=self.max_wrap_spread_rad,
            preregistered=True,
        )


#: The gate every moment estimate in this campaign uses.
MOMENT_GATE = MomentGate()


@dataclass(frozen=True)
class MomentConfig:
    """The moment stage's OWN front end, deliberately shorter than the frozen
    spectral one: ``lam`` is separated from ``sigma`` by the LAG SHAPE of the
    phase-increment variance, which needs many short lags inside one
    correlation time, not one 1.02 s window."""

    window: int = 2048
    hop: int = 512
    lags: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

    def as_dict(self) -> dict[str, Any]:
        return dict(window=self.window, hop=self.hop, lags=list(self.lags))


@dataclass(frozen=True)
class ShaftDynamics:
    """Frozen rig dynamics: ``lam``/``sigma`` are constants in stage 2, ``D`` is
    only INITIALIZED here and then fitted by the spectral MAP."""

    lam: float  # 1/s
    sigma: float  # rad/s, stationary std of nu
    d_init: float  # rad^2/s
    identified: bool
    diagnostics: dict[str, Any]


@dataclass
class FitConfig:
    """Everything the fit needs that is not data. The manifest supplies every
    field the CLI passes; nothing here is guessed at fit time.

    ``state_rate_hz`` is NOT chosen from ``lam`` or from the hop. The cubic
    Hermite interpolation of the state onto the audio grid omits the
    intra-interval integrated-OU bridge variance, whose one-interval size is
    ``(2/3) sigma^2 lam dt^3`` in ``theta`` and therefore ``k^2`` times that in
    the atom's phase. For the campaign's reference dynamics (``sigma`` = 6
    rad/s, ``lam`` = 6 /s) at ``k`` = 100 that omitted phase variance is
    ``5.9 rad^2`` at a 16 ms grid, ``1.15e-2 rad^2`` at 2 ms (500 Hz, this
    default) and ``1.4e-3 rad^2`` at 1 ms. 500 Hz is the default because 2 ms
    already puts the omission two orders of magnitude below one radian squared
    at the highest order the rigs carry; the planted TWO-SIDED grid check
    (``tests/experiments/test_stochastic_fit_revised_phase.py``, which compares
    predictions built on a 1000 Hz and a 500 Hz grid against the reference
    path) is what justifies it, and refining it is a configuration change. The
    export's ``coarsening_sensitivity`` block is NOT that check and must never
    be read as one.
    """

    n_fft: int = N_FFT
    hop: int = HOP
    sr: int = SR
    #: the DECLARED work grid the atoms are synthesized on: an INTEGER multiple
    #: of ``sr`` (default 4x). It is a model grid, not anybody's recording rate,
    #: and it is recorded in the export as ``model_grid_hz``.
    sample_rate_work: int = SAMPLE_RATE_WORK
    band_hz: tuple[float, float] = BAND_HZ
    state_rate_hz: float = 500.0
    k_cap: int = 130
    #: one telemetry delay per rotor, in seconds; REQUIRED (no invented default)
    delay_s: tuple[float, ...] | None = None
    iters: int = 400
    lr: float = 0.05
    seed: int = 0
    frame_chunk: int = 1
    #: orders per kernel call. ``None`` computes every order in one call; an
    #: int splits them and wraps each block in ``torch.utils.checkpoint`` — the
    #: same memory device :meth:`model.CombSpectrum.lines` uses, at the same
    #: block of 32. It is a NUMERICAL implementation detail: no order is
    #: dropped, ``k_cap`` is untouched and the prediction is unchanged; only
    #: the peak memory of computing it changes. 32 keeps a 16 GB card inside
    #: its budget at ``k_cap = 230`` on the 4x work grid.
    harmonic_chunk: int | None = 32
    frames_per_step: int | None = None
    #: the frozen composite temperature ``T = J / H``; it DIVIDES the composite
    #: risk (``L / T``). Calibrated on planted/baseline data against the
    #: UNNORMALIZED risk of :func:`composite_risk` and then FROZEN; read, never
    #: tuned. Must be finite and strictly positive.
    temperature: float = 1.0
    bias_std_hz: float = 0.5
    bias_mean_std_hz: float = BIAS_MEAN_PRIOR_STD_HZ
    log_d_mean: float = 0.0
    log_d_std: float = 2.0
    #: real dtype of the atoms and the spectral kernel ("float32" or "float64").
    #: The carrier phase is ALWAYS accumulated in float64 and wrapped into
    #: ``[0, 2 pi)`` before the cast, so float32 atoms cost 1e-7 rad, not the
    #: 0.1 rad a float32 phase accumulation over a 16 s clip would cost.
    atom_dtype: str = "float32"
    moments: MomentConfig = field(default_factory=MomentConfig)
    gate: MomentGate = field(default_factory=MomentGate)
    #: manifest-derived provenance the export copies verbatim
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        t = float(self.temperature)
        if not math.isfinite(t) or t <= 0.0:
            raise ValueError(
                "composite.temperature must be finite and strictly positive "
                f"(it divides the composite risk), got {self.temperature!r}"
            )
        sr, srw = int(self.sr), int(self.sample_rate_work)
        if srw <= 0 or sr <= 0 or srw % sr != 0:
            raise ValueError(
                "front_end.sample_rate_work must be a positive INTEGER multiple of the analysis "
                f"rate sr={self.sr!r} (the default is {SAMPLE_RATE_WORK} = 4 x 16000), got "
                f"{self.sample_rate_work!r}"
            )
        self.n_fft_work  # noqa: B018 — validate the derived window here, not mid-fit

    @property
    def n_fft_work(self) -> int:
        """The work-grid window: the same DURATION and the same bin spacing as
        ``n_fft`` at ``sr``, because ``Fs_work / n_fft_work == Fs / n_fft``.

        A property rather than a stored field: ``n_fft`` is overridden after
        construction (``predict_spectrum`` rebuilds a config with the caller's
        front end), and the derived window must never go stale.
        """
        num = int(self.n_fft) * int(self.sample_rate_work)
        if num % int(self.sr) != 0:
            raise ValueError(
                f"n_fft {self.n_fft} at sample_rate_work {self.sample_rate_work} does not divide "
                f"exactly by sr {self.sr}: the work window would not share the analysis bin grid"
            )
        return num // int(self.sr)

    def rotor_delays(self, n_rotors: int) -> np.ndarray:
        if self.delay_s is None:
            raise ValueError(
                "FitConfig.delay_s is required: name one telemetry delay per rotor "
                f"(expected {n_rotors}); the manifest key is rigs.<rig>.delay_s"
            )
        d = np.atleast_1d(np.asarray(self.delay_s, dtype=np.float64))
        if d.size == 1:
            d = np.repeat(d, n_rotors)
        if d.size != n_rotors:
            raise ValueError(f"delay_s has {d.size} entries for {n_rotors} rotors")
        return d

    def front_end(self) -> dict[str, Any]:
        return dict(
            n_fft=self.n_fft,
            hop=self.hop,
            sr=self.sr,
            band_hz=list(self.band_hz),
            sample_rate_work=self.sample_rate_work,
        )


@dataclass(frozen=True)
class CarrierTrack:
    """The DIAGNOSTIC telemetry-conditioned MAP carrier track of one clip."""

    time_s: np.ndarray  # (S,)
    nu_hz: np.ndarray  # (R, S) = nu / 2 pi
    theta_rad: np.ndarray  # (R, S)
    bias_hz: np.ndarray  # (R,)
    total_hz: np.ndarray  # (R, S) = raw(t - delay) + b + nu / 2 pi
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RevisedRender:
    audio: np.ndarray  # (M, T) float32 at sample_rate
    physical_rps: np.ndarray  # (R, T) realized interval-average shaft rev/s
    reference_rps: np.ndarray  # (R, T) the raw telemetry reference the arms share
    sample_rate: int
    diagnostics: dict[str, Any]


# ── exact integrated-OU algebra ─────────────────────────────────────────────


def _xp(*arrays: Any) -> Any:
    """``torch`` if any argument is a tensor, else ``numpy`` (single dispatch)."""
    for a in arrays:
        if isinstance(a, Tensor):
            return torch
    return np


def ou_transition(lam: float, sigma: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """EXACT discrete transition ``F`` and covariance ``Q`` of the integrated OU
    on a grid of step ``dt``, for the state ``z = (theta, nu)``::

        e = exp(-lam dt)
        F = [[1, (1 - e) / lam], [0, e]]
        Var[nu]        = sigma^2 (1 - e^2)
        Cov[theta, nu] = (sigma^2 / lam) (1 - e)^2
        Var[theta]     = (2 sigma^2 / lam^2) [x - 2(1 - e) + (1 - e^2)/2],  x = lam dt

    Every entry is evaluated through ``expm1``, and ``Var[theta]`` — whose
    bracket cancels to ``x^3/3 - x^4/4 + 7 x^5/60`` — switches to that series
    below ``x = 1e-2`` instead of subtracting two nearly equal numbers. The
    small-step limits are ``Var[nu] -> 2 sigma^2 lam dt``,
    ``Cov -> sigma^2 lam dt^2`` and ``Var[theta] -> (2/3) sigma^2 lam dt^3``.
    """
    lam, sigma, dt = float(lam), float(sigma), float(dt)
    if lam <= 0.0:
        raise ValueError(f"lam must be positive, got {lam!r}")
    if sigma < 0.0:
        raise ValueError(f"sigma must be non-negative, got {sigma!r}")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt!r}")
    x = lam * dt
    e = math.exp(-x)
    one_m_e = -math.expm1(-x)
    one_m_e2 = -math.expm1(-2.0 * x)
    f_mat = np.array([[1.0, one_m_e / lam], [0.0, e]], dtype=np.float64)
    var_nu = sigma**2 * one_m_e2
    cov = (sigma**2 / lam) * one_m_e**2
    bracket = (
        x**3 / 3.0 - x**4 / 4.0 + 7.0 * x**5 / 60.0
        if x < 1e-2
        else x - 2.0 * one_m_e + 0.5 * one_m_e2
    )
    var_theta = 2.0 * sigma**2 / lam**2 * bracket
    q_mat = np.array([[var_theta, cov], [cov, var_nu]], dtype=np.float64)
    return f_mat, q_mat


def ou_cholesky(lam: float, sigma: float, dt: float) -> tuple[float, float, float]:
    """Lower Cholesky ``[[a, 0], [b, c]]`` of :func:`ou_transition`'s ``Q``."""
    _, q = ou_transition(lam, sigma, dt)
    a = math.sqrt(max(float(q[0, 0]), 0.0))
    if a == 0.0:
        return 0.0, 0.0, math.sqrt(max(float(q[1, 1]), 0.0))
    b = float(q[0, 1]) / a
    c = math.sqrt(max(float(q[1, 1]) - b * b, 0.0))
    return a, b, c


def integrated_ou_increment_var(tau_s: Any, *, lam: float, sigma: float) -> Any:
    """``Var[theta(t + tau) - theta(t)]`` of a STATIONARILY initialized
    integrated OU: ``2 sigma^2 [ |tau|/lam - (1 - exp(-lam|tau|))/lam^2 ]``.

    In ``x = lam |tau|`` the bracket is ``(x - 1 + exp(-x)) / lam^2``, which is
    ``x^2/2 - x^3/6 + x^4/24`` for small ``x`` (so the increment variance is
    ``sigma^2 tau^2`` in the frozen-rate limit — a rate held at its stationary
    scale). Below ``x = 1e-3`` the series is used instead of the cancelling
    difference. NOTE this is a different bracket from the one-step TRANSITION
    ``Var[theta]`` of :func:`ou_transition`, which starts from ``nu = 0`` and is
    ``O(lam dt^3)``; the two must not be interchanged.
    """
    xp = _xp(tau_s)
    tau = xp.abs(tau_s)
    x = lam * tau
    series = x**2 / 2.0 - x**3 / 6.0 + x**4 / 24.0
    direct = x + xp.expm1(-x)
    bracket = xp.where(x < 1e-3, series, direct)
    return 2.0 * sigma**2 / lam**2 * bracket


def conditional_r_tau(tau_s: Any, *, d: Any) -> Any:
    """TRAINING / conditional residual autocorrelation ``exp(-D |tau|)``.

    The atom already carries ``k theta_MAP(t)``, so the shaft term is spent;
    multiplying the OU term in again here is the double-counting the contract
    forbids.
    """
    if isinstance(d, Tensor) and not isinstance(tau_s, Tensor):
        tau_s = torch.as_tensor(tau_s, dtype=d.dtype, device=d.device)
    xp = _xp(tau_s, d)
    return xp.exp(-d * xp.abs(tau_s))


def prior_r_tau(tau_s: Any, k: Any, *, lam: float, sigma: float, d: Any) -> Any:
    """PRIOR / held-out predictive residual autocorrelation::

        R_k(tau) = exp( -D|tau| - k^2 sigma^2 [ |tau|/lam
                                                - (1 - exp(-lam|tau|))/lam^2 ] )

    i.e. ``exp(-0.5 Var[k dtheta] - 2 D |tau| / 2)`` with ``dtheta`` the
    stationarily initialized integrated-OU increment
    (:func:`integrated_ou_increment_var`) and ``eps`` a Wiener process of
    diffusion ``D``. ``k`` broadcasts against the leading dims of ``tau_s``.

    Backend dispatch reads ALL THREE of ``tau_s``, ``k`` and ``d``: a torch
    ``k`` with numpy lags is a torch call, and dispatching on the lags alone
    would either raise on a CUDA ``k`` or silently drop its autograd. The first
    tensor argument owns the device, and its dtype owns the precision unless it
    is an integer (an order count often is), in which case float64 is used.
    """
    ref = next((v for v in (tau_s, k, d) if isinstance(v, Tensor)), None)
    if ref is None:
        tau = np.abs(np.asarray(tau_s, dtype=np.float64))
        var = integrated_ou_increment_var(tau, lam=lam, sigma=sigma)
        k_np = np.asarray(k, dtype=np.float64)
        return np.exp(-np.asarray(d, dtype=np.float64) * tau - 0.5 * k_np**2 * var)
    dtype = ref.dtype if ref.is_floating_point() else torch.float64
    cast = lambda v: (  # noqa: E731
        v.to(dtype=dtype, device=ref.device)
        if isinstance(v, Tensor)
        else torch.as_tensor(v, dtype=dtype, device=ref.device)
    )
    tau = torch.abs(cast(tau_s))
    var = integrated_ou_increment_var(tau, lam=lam, sigma=sigma)
    return torch.exp(-cast(d) * tau - 0.5 * cast(k) ** 2 * var)


def _ar1_causal(drive: Any, e: float) -> Any:
    """``y_j = e y_{j-1} + drive_j`` along the last axis, vectorized.

    ``lam`` is frozen and ``dt`` uniform, so the recursion is linear
    time-invariant: it is a causal convolution with ``e^n``. numpy uses
    ``scipy.signal.lfilter``; torch uses an FFT convolution (differentiable,
    ``O(S log S)``, batched over (clip, rotor), GPU-friendly) — a per-state
    autograd loop over the 8000-16000 states of a 16 s clip is not acceptable.
    The kernel DECAYS, so it is truncated where ``e^n`` falls below 1e-12
    (float32 atoms cannot see anything smaller); when ``e`` is within 1e-12 of 1
    no truncation is possible and the full length is used.
    """
    s = int(drive.shape[-1])
    klen = s if e >= 1.0 - 1e-12 else min(s, int(math.ceil(math.log(1e-12) / math.log(e))) + 1)
    if not isinstance(drive, Tensor):
        from scipy.signal import lfilter

        return lfilter(np.array([1.0]), np.array([1.0, -e]), np.asarray(drive), axis=-1)
    n = 1 << int(s + klen - 1).bit_length()
    kern = e ** torch.arange(klen, dtype=drive.dtype, device=drive.device)
    y = torch.fft.irfft(torch.fft.rfft(drive, n=n) * torch.fft.rfft(kern, n=n), n=n)
    return y[..., :s]


def simulate_state(innov: Any, *, lam: float, sigma: float, dt: float) -> tuple[Any, Any]:
    """``(theta, nu)`` on the state grid from WHITENED innovations.

    ``innov`` is ``(..., S, 2)`` standard-normal coordinates::

        nu_0    = sigma * innov[..., 0, 0]        (stationary initial law)
        theta_0 = 0                               (gauge)
        nu_j    = e nu_{j-1} + b innov[..., j, 0] + c innov[..., j, 1]
        theta_j = theta_{j-1} + ((1 - e)/lam) nu_{j-1} + a innov[..., j, 0]

    with ``e = exp(-lam dt)`` and ``[[a, 0], [b, c]]`` the Cholesky of the exact
    ``Q(dt)``. In these coordinates the state log-prior is EXACTLY
    ``0.5 sum e^2`` — one exact Gaussian integrated-OU prior, isotropic and well
    conditioned — and the MAP is identical to the MAP in ``z`` coordinates
    because the map is linear with constant Jacobian.

    ``innov[..., 0, 1]`` is unused (the ``theta_0 = 0`` gauge has no innovation);
    the prior must exclude it, see :meth:`_RevisedModel.prior`.
    """
    xp = _xp(innov)
    a, b, c = ou_cholesky(lam, sigma, dt)
    e = math.exp(-lam * dt)
    f01 = -math.expm1(-lam * dt) / lam
    e1 = innov[..., 0]
    e2 = innov[..., 1]
    drive = b * e1 + c * e2
    if xp is torch:
        drive = torch.cat((sigma * e1[..., :1], drive[..., 1:]), dim=-1)
    else:
        drive = np.concatenate((sigma * e1[..., :1], drive[..., 1:]), axis=-1)
    nu = _ar1_causal(drive, e)
    inc = f01 * nu[..., :-1] + a * e1[..., 1:]
    # the theta_0 = 0 gauge is seeded from a length-one slice of the STATE
    # grid, not of the increments: with S == 1 there is no increment at all,
    # and slicing the empty increment array returned a length-0 theta against
    # a length-1 nu, breaking the same-grid contract.
    if xp is torch:
        theta = torch.cat((torch.zeros_like(nu[..., :1]), torch.cumsum(inc, dim=-1)), dim=-1)
    else:
        theta = np.concatenate((np.zeros_like(nu[..., :1]), np.cumsum(inc, axis=-1)), axis=-1)
    return theta, nu


def _hermite(theta: Any, nu: Any, pos: Any, dt: float) -> Any:
    """Cubic Hermite value of ``theta`` at fractional grid positions ``pos``.

    ``nu = dtheta/dt`` is the exact derivative of the state, so
    ``(theta_j, nu_j, theta_{j+1}, nu_{j+1})`` determines a cubic that is
    consistent by construction. This OMITS the integrated-OU bridge variance
    inside a grid interval: it is a GRID APPROXIMATION justified by the planted
    two-sided check in the tests (see :class:`FitConfig.state_rate_hz`), NOT by
    the export's ``coarsening_sensitivity`` block, which only coarsens an
    already-fitted path. No bridge kernel is added.
    """
    xp = _xp(theta)
    n_states = int(theta.shape[-1])
    if xp is torch:
        j = torch.clamp(torch.floor(pos).to(torch.int64), 0, n_states - 2)
        s = pos - j.to(pos.dtype)
    else:
        j = np.clip(np.floor(pos).astype(np.int64), 0, n_states - 2)
        s = pos - j
    s2, s3 = s * s, s * s * s
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = s3 - 2.0 * s2 + s
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = s3 - s2
    t0, t1 = theta[..., j], theta[..., j + 1]
    n0, n1 = nu[..., j], nu[..., j + 1]
    return h00 * t0 + h10 * dt * n0 + h01 * t1 + h11 * dt * n1


def hermite_theta(theta: Any, nu: Any, *, dt_state: float, n_out: int, sr: int = SR) -> Any:
    """``theta`` on an audio grid of ``n_out`` samples at ``sr``, from the state
    grid of step ``dt_state`` (see :func:`_hermite` for the approximation)."""
    xp = _xp(theta)
    if xp is torch:
        pos = torch.arange(int(n_out), dtype=theta.dtype, device=theta.device) / (sr * dt_state)
    else:
        pos = np.arange(int(n_out), dtype=np.float64) / (sr * dt_state)
    return _hermite(theta, nu, pos, dt_state)


def n_states_for(n_samples: int, *, sr: int, state_rate_hz: float) -> int:
    """States needed to cover ``n_samples`` audio samples (one spare knot so the
    last sample sits inside an interval, never in an extrapolation)."""
    return int(math.floor((n_samples - 1) / sr * state_rate_hz)) + 2


# ── the composite risk and its exposure weights ─────────────────────────────


def composite_weights(frame_keys: Sequence[tuple[Any, int]], *, hop: int, n_fft: int) -> np.ndarray:
    """Positive composite weights ``a_i``, normalized per UNIQUE audio duration.

    A frame carries ``hop / n_fft`` of exposure — the fraction of its own window
    length that is new audio — and identical duplicated windows SPLIT it: if the
    same ``(clip_id, start_sample)`` key appears ``d`` times, each copy gets
    ``a_i / d``. Duplicating an observation must not create exposure or
    information, and this is the only place that is enforced.
    """
    keys = list(frame_keys)
    counts: dict[tuple[Any, int], int] = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    base = float(hop) / float(n_fft)
    return np.array([base / counts[key] for key in keys], dtype=np.float64)


def composite_risk(power: Any, model: Any, weights: Any, *, band: Any = None) -> Any:
    """``sum_i a_i sum_{m, f in band} (I_i / M_i + log M_i)``.

    The marginal composite risk of the module docstring: proper for the mean
    prediction under correlated overlapping windows, and NOT a joint NLL.
    ``power``/``model`` are ``(M, N, F)``, ``weights`` is ``(N,)``, ``band`` is a
    boolean mask or index over the last axis.
    """
    xp = _xp(power, model, weights)
    if band is not None:
        power, model = power[..., band], model[..., band]
    if xp is torch:
        m = torch.clamp(model, min=MODEL_FLOOR)
        cell = power / m + torch.log(m)
        w = torch.as_tensor(weights, dtype=cell.dtype, device=cell.device)
        return torch.einsum("n,mnf->", w, cell)
    m = np.clip(np.asarray(model, dtype=np.float64), MODEL_FLOOR, None)
    cell = np.asarray(power, dtype=np.float64) / m + np.log(m)
    return float(np.einsum("n,mnf->", np.asarray(weights, dtype=np.float64), cell))


# ── what may be fitted at all: raw tracks, the frozen cohort, disjointness ──

#: The RAW rotor tracks candidate 1 may be fitted on. ``rps_refined`` is an
#: INFERRED label and ``clips.RPS_KEYS`` resolves ``auto`` to it FIRST, so both
#: are refused: this candidate is fitted on telemetry, never on labels the
#: project itself inferred.
RAW_RPS_KEYS: tuple[str, ...] = ("rps", "motors_measured", "motors_command")

#: The FROZEN per-rig training cohort. It is PROVENANCE, not a preference: a
#: manifest naming a different support is a different experiment, and it is
#: refused rather than quietly fitted into an ordinary scoreable export.
TRAINING_COHORT: dict[str, dict[str, Any]] = {
    "michaels": dict(
        recording_tokens=("FLY125",),
        regimes=("standby", "ramp", "cruise"),
        n_recordings=None,
        seconds=None,
        description="FLY125 only, the JOINT standby + ramp + cruise training support",
    ),
    "dregon": dict(
        recording_tokens=("room2", "nosource"),
        regimes=("cruise",),
        n_recordings=5,
        seconds=16.0,
        description="the five room2 pure-noise recordings, cruise only, 16 s windows",
    ),
    "bench": dict(
        recording_tokens=(),
        regimes=None,
        n_recordings=None,
        seconds=None,
        description="planted/physical diagnostics only; never a scored arm",
    ),
}


def check_rotor_track_key(rps_key: str, *, rig: str, where: str = "manifest") -> str:
    """Refuse an INFERRED rotor track, before a single clip is loaded.

    The entry point promises never to fit on inferred labels, and
    :func:`clips.load_clip` takes the key verbatim, so this is where the
    promise is kept.
    """
    key = str(rps_key)
    if key in RAW_RPS_KEYS:
        return key
    raise ValueError(
        f"{where}: rigs.{rig}.rps_key is {key!r}. Candidate 1 is fitted on RAW telemetry and "
        f"accepts only {list(RAW_RPS_KEYS)}: 'rps_refined' is an INFERRED label and 'auto' "
        "resolves to it first (clips.RPS_KEYS), so either would fit the candidate on labels "
        "this project inferred. Name the raw track explicitly — no default is substituted."
    )


def check_training_cohort(
    rig: str, clips: Sequence[dict[str, Any]], *, where: str = "manifest"
) -> None:
    """Enforce the frozen cohort of :data:`TRAINING_COHORT` on a manifest's clip
    rows, BEFORE the moment stage, so nothing expensive starts on a manifest
    that is not the preregistered support."""
    spec = TRAINING_COHORT.get(str(rig))
    if spec is None:
        raise ValueError(
            f"{where}: no frozen training cohort for rig {rig!r} "
            f"(cohorts are declared for {sorted(TRAINING_COHORT)})"
        )
    rows = list(clips)
    if not rows:
        raise ValueError(f"{where}: rigs.{rig}.clips is empty")
    recordings = [str(r.get("recording")) for r in rows]
    regimes = [str(r.get("regime")).lower() for r in rows]
    seconds = [float(r.get("seconds", float("nan"))) for r in rows]
    detail = (
        f"expected {spec['description']}; the manifest names {sorted(set(recordings))} with "
        f"regimes {sorted(set(regimes))} and window lengths "
        f"{sorted({round(s, 6) for s in seconds})} s over {len(rows)} row(s)"
    )
    for token in spec["recording_tokens"]:
        bad = sorted({r for r in recordings if str(token).lower() not in r.lower()})
        if bad:
            raise ValueError(
                f"{where}: rigs.{rig}.clips names {bad}, which do not carry {token!r} — {detail}"
            )
    want_regimes = spec["regimes"]
    if want_regimes is not None and set(regimes) != {str(w).lower() for w in want_regimes}:
        raise ValueError(
            f"{where}: rigs.{rig}.clips covers regimes {sorted(set(regimes))}, not exactly "
            f"{sorted(want_regimes)} — {detail}"
        )
    n_rec = spec["n_recordings"]
    if n_rec is not None and (len(rows) != int(n_rec) or len(set(recordings)) != int(n_rec)):
        raise ValueError(
            f"{where}: rigs.{rig}.clips has {len(rows)} row(s) over {len(set(recordings))} "
            f"distinct recording(s), not {int(n_rec)} — {detail}"
        )
    want_s = spec["seconds"]
    if want_s is not None:
        bad_len = sorted(
            {round(s, 6) for s in seconds if not math.isclose(s, float(want_s), abs_tol=1e-6)}
        )
        if bad_len:
            raise ValueError(
                f"{where}: rigs.{rig}.clips carries window length(s) {bad_len} s, not "
                f"{float(want_s)} s — {detail}"
            )


def check_manifest_supports(
    rig: str, clips: Sequence[dict[str, Any]], *, where: str = "manifest"
) -> None:
    """Refuse duplicated or overlapping TRAINING windows in a manifest, naming
    BOTH rows (see :func:`check_training_supports` for why)."""
    spans = []
    for i, row in enumerate(clips):
        start = float(row.get("start_s", 0.0))
        spans.append((i, str(row.get("recording")), start, start + float(row.get("seconds", 0.0))))
    for pos, (i, rec_a, lo_a, hi_a) in enumerate(spans):
        for j, rec_b, lo_b, hi_b in spans[pos + 1 :]:
            if rec_a != rec_b:
                continue
            if (lo_a, hi_a) == (lo_b, hi_b) or (lo_a < hi_b and lo_b < hi_a):
                raise ValueError(
                    f"{where}: rigs.{rig}.clips[{i}] ({rec_a} [{lo_a:g}, {hi_a:g}) s) and "
                    f"rigs.{rig}.clips[{j}] ({rec_b} [{lo_b:g}, {hi_b:g}) s) are the same or "
                    "overlapping TRAINING support. Cut disjoint windows: every row gets its "
                    "own state block and its own bias prior."
                )


def check_training_supports(rows: Sequence[tuple[str, Clip]]) -> None:
    """REJECT duplicated or overlapping TRAINING supports — the fit's own
    backstop, run before any state or bias block is allocated.

    Splitting the composite weight of a duplicated WINDOW
    (:func:`composite_weights`) stops a repeated observation from creating
    exposure, but it cannot stop a repeated CLIP from creating PARAMETERS: each
    row carries its own ``state_innov`` block and its own bias, each with a full
    prior, so fitting the same audio twice buys two half-losses and TWO priors —
    extra shrinkage, i.e. information smuggled in by duplication. Two rows whose
    windows merely intersect share latent shaft phase that two independent state
    blocks cannot represent either.

    The answer is a refusal, NOT a shared-state framework keyed by recording
    time. Held-out and predictive use of overlapping windows is unaffected: this
    is a property of the fit cohort, and the predictive paths build one clip at
    a time.
    """
    spans: list[tuple[str, str, float, float]] = []
    for clip_id, clip in rows:
        meta = dict(getattr(clip, "meta", {}) or {})
        recording = meta.get("recording_id")
        # recording-absolute when the clip knows where it came from; a planted
        # or synthetic clip has only its own identity to be duplicated by
        key = (
            f"{meta.get('dataset', '')}:{recording}" if recording is not None else f"clip:{clip_id}"
        )
        start = float(meta.get("start_s", 0.0))
        spans.append((str(clip_id), key, start, start + float(meta.get("duration_s", 0.0))))
    for pos, (id_a, key_a, lo_a, hi_a) in enumerate(spans):
        for id_b, key_b, lo_b, hi_b in spans[pos + 1 :]:
            if key_a != key_b:
                continue
            if (lo_a, hi_a) == (lo_b, hi_b) or (lo_a < hi_b and lo_b < hi_a):
                raise ValueError(
                    f"duplicate or overlapping TRAINING support: {id_a!r} covers "
                    f"[{lo_a:g}, {hi_a:g}) s and {id_b!r} covers [{lo_b:g}, {hi_b:g}) s of "
                    f"{key_a}. Each row gets its own state block and its own bias prior, so the "
                    "shared audio would enter the objective twice and its priors twice; cut "
                    "disjoint windows. (Held-out and predictive use of overlapping windows is "
                    "unaffected.)"
                )


def unidentified_diagnostic(
    rig_id: str, dynamics: ShaftDynamics, config: FitConfig
) -> dict[str, Any]:
    """The auditable record of a moment stage that did NOT identify the dynamics.

    NOT an export: it carries no parameters, because no MAP was run. Freezing
    unidentifiable ``lam``/``sigma`` into a full fit produces an artifact with
    no valid dynamics estimate, which the evaluator must reject anyway — after
    the fit has been paid for.
    """
    return dict(
        schema_version=SCHEMA_VERSION,
        model_family=MODEL_FAMILY,
        rig_id=str(rig_id),
        status="moment_stage_unidentified",
        is_export=False,
        note="the moment stage did not identify the shaft dynamics, so NO MAP fit was run and "
        "this file is NOT a candidate export; it records what was found so the failure is "
        "auditable",
        unidentifiable_reason=str(dynamics.diagnostics.get("unidentifiable_reason", "unrecorded")),
        moment_estimate=dict(lam=dynamics.lam, sigma=dynamics.sigma, d_init=dynamics.d_init),
        moments=dynamics.diagnostics,
        identified=False,
        training_provenance=dict(config.provenance),
        code_version=_git_head(),
    )


# ── the floor's power spectrum: ONE source of truth for fit and render ──────


def floor_geometry(freqs_hz: Any, ctrl_hz: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(shape_matrix, tilt_oct)`` of the floor at ``freqs_hz``.

    Pure geometry — no fitted parameter — so a caller builds it once per grid
    and multiplies the fitted control points into it every step.
    :func:`model.interp_matrix` clamps to the end knots, which is what carries
    the shape past the last control point: the work grid runs to the work
    Nyquist, the control points only to the analysis Nyquist.
    """
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    ctrl = np.asarray(ctrl_hz, dtype=np.float64)
    ctrl_oct = np.log2(ctrl / ctrl[0])
    f_oct = np.log2(np.maximum(freqs, FLOOR_SHAPE_F_MIN) / ctrl[0])
    tilt_oct = np.log2(np.maximum(freqs, FLOOR_SHAPE_F_MIN) / FLOOR_TILT_REF_HZ)
    return interp_matrix(f_oct, ctrl_oct), tilt_oct


def floor_power_spectrum(
    shape_matrix: Any,
    tilt_oct: Any,
    *,
    mean_db: Any,
    ctrl_db: Any,
    tilt_db_oct: Any,
    rate_factor: float,
) -> Any:
    """The floor's power spectrum on the grid ``shape_matrix``/``tilt_oct``
    describe, in WORK-grid periodogram units.

    THE one source of truth: :meth:`_RevisedModel.floor_psd` reads it to build
    ``R_floor`` inside the fit's graph and :func:`render_revised` shapes its
    white noise with its square root, so the fitted floor and the synthesized
    floor cannot drift apart.

    ``rate_factor = sample_rate_work / sample_rate`` because the parameters are
    at the ANALYSIS convention: noise shaped on the work grid reads
    ``sample_rate / sample_rate_work`` times lower once decimated, which is the
    very factor the fit's ``grid_power_factor`` divides back out.

    numpy in, numpy out; torch in, torch out and differentiable — the body is
    one matrix product, one broadcast and one power.
    """
    db = mean_db + shape_matrix @ ctrl_db + tilt_db_oct * tilt_oct
    return float(rate_factor) * 10.0 ** (db / 10.0)


def off_regime_extrapolation(rate_rps: Any, *, where: str) -> dict[str, Any]:
    """Report a trajectory that leaves the CALIBRATED speed domain.

    The speed laws were fitted on rotors at or above :data:`CALIBRATED_MIN_RPS`
    (standby, ramp and cruise). Below that the fitted exponents are
    extrapolation — for a negative exponent an arbitrarily large one — and
    :data:`SPEED_FLOOR_RPS` only keeps the arithmetic finite. This is a NAMED
    diagnostic, never a silent repair: no motors-off physics is invented here.
    """
    rate = np.asarray(rate_rps, dtype=np.float64)
    below = rate < CALIBRATED_MIN_RPS
    return dict(
        where=str(where),
        calibrated_min_rps=CALIBRATED_MIN_RPS,
        speed_floor_rps=SPEED_FLOOR_RPS,
        min_rps=float(rate.min()) if rate.size else float("nan"),
        fraction_below_calibrated=float(below.mean()) if rate.size else 0.0,
        off_regime=bool(below.any()),
        note="rotors below the calibrated domain are UNSUPPORTED extrapolation, not a modelled "
        "motors-off regime; the speed floor prevents infinities, it does not extend the model",
    )


# ── stage 1: moment initialization ──────────────────────────────────────────


def _wrap(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _delayed_rps(rps: np.ndarray, delays: np.ndarray, sr: int) -> np.ndarray:
    """``raw_r(t - delay_r)`` on the clip's own audio grid (edge-held)."""
    t = np.arange(rps.shape[1], dtype=np.float64) / sr
    return np.stack([np.interp(t - float(d), t, r) for d, r in zip(delays, rps, strict=True)])


def _frame_means(track: np.ndarray, starts: np.ndarray, width: int) -> np.ndarray:
    """``(R, N)`` means of ``track`` over the frames — the same frame-mean speed
    convention as :func:`data.periodogram`."""
    cs = np.concatenate((np.zeros((track.shape[0], 1)), np.cumsum(track, axis=1)), axis=1)
    return (cs[:, starts + width] - cs[:, starts]) / float(width)


def line_snr_db(power: np.ndarray, centres: np.ndarray, df: float) -> np.ndarray:
    """``(M, N)`` line-to-local-floor ratio in dB, PER MICROPHONE and PER FRAME.

    Line level is the largest of the three bins around the centre; the local
    floor is the median of an annulus 8-32 bins away on both sides, which skips
    the line's own window skirts.

    Deliberately NOT mic-averaged and not reduced over frames: an order strong
    in some channels and noise-only in others passes a mic-averaged gate while
    the weak channels contribute nearly uniform phase, and a median over the
    track admits every weak frame of a half-strong track. The preregistered
    threshold is applied to THIS (mic, frame) mask instead.
    """
    n_bins = power.shape[2]
    centre_bin = np.clip(np.round(centres / df).astype(np.int64), 0, n_bins - 1)
    near = np.clip(centre_bin[:, None] + np.arange(-1, 2)[None, :], 0, n_bins - 1)
    annulus = np.concatenate((np.arange(-32, -7), np.arange(8, 33)))
    far = np.clip(centre_bin[:, None] + annulus[None, :], 0, n_bins - 1)
    rows = np.arange(centres.size)[:, None]
    line = power[:, rows, near].max(axis=2)  # (M, N)
    floor = np.median(power[:, rows, far], axis=2)  # (M, N)
    return 10.0 * np.log10(np.maximum(line, 1e-30) / np.maximum(floor, 1e-30))


def valid_lag_pairs(valid: np.ndarray, lag: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """``(mic, lo_frames, hi_frames)`` for one frame lag, per microphone.

    A pair enters ONLY if both endpoints lie inside the same continuous run of
    valid frames of that microphone: an increment straddling a frame the SNR
    gate rejected is not an increment of the process being measured, and it
    inflates the variance the moments are read from. ``valid`` is the boolean
    ``(M, N)`` gate mask; the run split is :func:`phase_stats._runs`, the same
    maximal-consecutive-run helper ``lag_coherence`` uses.
    """
    lag = int(lag)
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    for m in range(int(valid.shape[0])):
        idx = np.flatnonzero(valid[m])
        for run in _frame_runs(idx):
            frames = idx[run]
            if frames.size > lag:
                out.append((m, frames[:-lag], frames[lag:]))
    return out


def _raw_phase_in_frame(
    raw: np.ndarray, starts: np.ndarray, width: int, sr: int
) -> np.ndarray:
    """Integrated raw telemetry phase sampled inside each analysis frame.

    ``Phi_raw(t_start + n)`` for ``n = 0..width-1``.  Demodulating by this
    GLOBAL moving telemetry removes the carrier exactly, so the coefficient
    phase is the residual ``k theta(t) + epsilon(t)`` and lagged increments
    are residual increments directly.  A constant phase reference would cancel
    in the increment; using the actual integrated phase avoids subtracting a
    nominal advance separately.
    """
    raw = np.asarray(raw, dtype=np.float64)
    phi = 2.0 * np.pi * np.cumsum(raw, axis=-1) / float(sr)
    n = np.arange(width)
    idx = starts[:, None] + n[None, :]
    return phi[idx]


def _stft_demod_integrated(
    audio: np.ndarray,
    raw: np.ndarray,
    frames: np.ndarray,
    k: float,
    window: np.ndarray,
    hop: int,
    sr: int,
) -> np.ndarray:
    """Hann-windowed coefficient after demodulating by ``exp(-1j k Phi_raw(t))``
    sampled inside the frame.

    Returns ``(M, N)`` complex coefficients, one per microphone and frame.
    The nominal carrier advance is removed by the within-window integration, so
    the coefficient phase is the residual ``k theta(t) + epsilon(t)`` (up to
    the finite-window average).  Lagged increments are therefore residual
    increments directly; no further subtraction of a nominal advance is needed.
    """
    audio = np.asarray(audio, dtype=np.float64)
    window = np.asarray(window, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.int64)
    starts = frames * int(hop)
    width = int(window.size)
    phase = _raw_phase_in_frame(raw, starts, width, sr)  # (N, width)
    n = np.arange(width)
    seg = (
        audio[:, starts[:, None] + n[None, :]]
        * window[None, None, :]
    )  # (M, N, width)
    return (seg * np.exp(-1j * float(k) * phase)[None, :, :]).sum(axis=-1)


def _pair_id(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Compact identifier of a (lo_frame, hi_frame) pair for set intersection."""
    return (
        np.asarray(lo, dtype=np.int64) << 32
    ) | np.asarray(hi, dtype=np.int64)


def _aligned_covariance(
    res_a: np.ndarray, pid_a: np.ndarray, res_b: np.ndarray, pid_b: np.ndarray
) -> tuple[float, int]:
    """Mean product of ``res_a`` and ``res_b`` over their common pair ids.

    The inputs are already circular-mean-removed.  The returned covariance is
    a wrapped-phase sample covariance, not an exact posterior quantity.
    """
    if pid_a.size == 0 or pid_b.size == 0:
        return 0.0, 0
    common, ia, ib = np.intersect1d(
        pid_a, pid_b, assume_unique=True, return_indices=True
    )
    if common.size == 0:
        return 0.0, 0
    a = np.asarray(res_a, dtype=np.float64)[ia]
    b = np.asarray(res_b, dtype=np.float64)[ib]
    return float(np.mean(a * b)), int(common.size)


def estimate_shaft_dynamics(
    rows: Sequence[tuple[str, Clip]],
    *,
    gate: MomentGate = MOMENT_GATE,
    config: FitConfig,
) -> ShaftDynamics:
    """Moment initialization of ``(lam, sigma, D)`` from complex line phases.

    The SHAFT term is identified from the OFF-DIAGONAL structure function of
    same-rotor order pairs:

        ``Cov[dpsi_k(tau), dpsi_l(tau)] = k l 2 sigma^2
            [tau/lam - (1 - exp(-lam tau))/lam^2]``,

    which is zero for independent per-harmonic motions even when their
    marginal variances are tuned to reproduce the same diagonals.  The
    independent per-harmonic diffusion ``D`` is initialized from the DIAGONAL
    residual after removing the shared OU term.

    Phase increments are formed after demodulating by the integrated raw
    telemetry phase ``exp(-1j k Phi_raw(t))`` inside each finite window, so the
    lagged coefficient difference is a residual increment directly; no nominal
    advance is subtracted twice.  This is the same convention the model uses for
    its atoms.

    These moment data are initializers and diagnostics ONLY: they never re-enter
    the stage-2 objective as a second likelihood factor. ``lam`` and ``sigma``
    are frozen constants in stage 2; ``D`` is only initialized here and then
    refined by the spectral MAP.

    The reported covariance is a DIAGNOSTIC (the residuals are correlated across
    lags, orders and mics, and the wrapped-phase covariance is not an exact
    posterior), reported alongside the empirical block-to-block variation across
    clips.  If the lag range does not bracket ``1/lam`` or the off-diagonal
    Jacobian is ill-conditioned, ``identified`` is ``False`` and
    ``diagnostics['unidentifiable_reason']`` says why; no invented "measured"
    value is ever substituted.
    """
    from collections import defaultdict
    from scipy.optimize import least_squares

    mc = config.moments
    k_lo, k_hi = gate.order_range
    band_lo, band_hi = config.band_hz
    rejected: dict[str, int] = {}
    cross_failures: dict[str, int] = {}
    n_valid_cells_total = 0
    n_lag_pairs_total = 0
    diag_cells: list[dict[str, Any]] = []
    offdiag_cells: list[dict[str, Any]] = []
    passed: list[dict[str, Any]] = []
    order_residuals: dict[
        tuple[str, int, float, float, int], tuple[np.ndarray, np.ndarray]
    ] = {}

    def reject(reason: str) -> None:
        rejected[reason] = rejected.get(reason, 0) + 1

    def cross_failure(reason: str) -> None:
        cross_failures[reason] = cross_failures.get(reason, 0) + 1

    for clip_id, clip in rows:
        if clip.sr != config.sr:
            raise ValueError(f"{clip_id}: sample rate {clip.sr} != config.sr {config.sr}")
        delays = config.rotor_delays(clip.rps.shape[0])
        raw = _delayed_rps(np.asarray(clip.rps, dtype=np.float64), delays, config.sr)
        pg = periodogram(clip, mc.window, mc.hop)
        n_frames = pg.power.shape[1]
        if n_frames <= max(mc.lags):
            reject("clip shorter than the longest lag")
            continue
        starts = np.arange(n_frames) * mc.hop
        rate = _frame_means(raw, starts, mc.window)  # (R, N) frame-mean rev/s
        df = pg.df
        # every line of every rotor/order, for the per-frame isolation test
        all_k = np.arange(1, config.k_cap + 1, dtype=np.float64)
        all_lines = rate[:, None, :] * all_k[None, :, None]  # (R, K, N)
        frames = np.arange(n_frames)
        window = np.hanning(mc.window + 1)[:mc.window]
        for r in range(rate.shape[0]):
            for k in range(max(1, k_lo), k_hi + 1):
                centres = k * rate[r]
                if centres.min() < band_lo or centres.max() > band_hi:
                    reject("out of band")
                    continue
                sep = np.abs(all_lines - centres[None, None, :])
                sep[r, k - 1, :] = np.inf
                # PER-FRAME isolation: a frame is usable only where THIS order is
                # separated from every other rotor/order at THAT frame.  The old
                # global ``sep.min()`` rejected an order for the whole clip if a
                # crossing occurred anywhere.
                min_sep = np.min(sep, axis=(0, 1))
                isolated = min_sep >= gate.min_isolation_bins * df
                if not isolated.any():
                    reject("isolation")
                    continue
                snr = line_snr_db(pg.power, centres, df)  # (M, N)
                valid = (snr >= gate.min_line_snr_db) & isolated[None, :]
                n_valid = int(valid.sum())
                if n_valid < gate.min_frames:
                    reject("too few usable (mic, frame) cells")
                    continue
                # Demodulate by the exact within-window integrated raw phase; the
                # coefficient phase is the residual directly.
                z = _stft_demod_integrated(
                    clip.audio, raw[r], frames, float(k), window, mc.hop, config.sr
                )
                used_lags = 0
                n_pairs_track = 0
                for lag in mc.lags:
                    tau = lag * mc.hop / config.sr
                    per_mic_resids: list[np.ndarray] = []
                    found_pair = False
                    for m, lo_i, hi_i in valid_lag_pairs(valid, lag):
                        dpsi = np.angle(z[m, hi_i] * np.conj(z[m, lo_i]))
                        # per-clip bias and the OU mean show up as a constant
                        # offset at fixed lag: remove the CIRCULAR mean, per mic
                        res = _wrap(dpsi - np.angle(np.mean(np.exp(1j * dpsi))))
                        pid = _pair_id(lo_i, hi_i)
                        order_residuals[(str(clip_id), int(r), float(k), float(tau), int(m))] = (
                            pid,
                            res,
                        )
                        per_mic_resids.append(res)
                        found_pair = True
                    if not found_pair:
                        continue
                    resid = np.concatenate(per_mic_resids)
                    if resid.size < gate.min_frames:
                        continue
                    spread = float(np.sqrt(np.mean(resid**2)))
                    if spread > gate.max_wrap_spread_rad:
                        reject("wrap-censored")
                        continue
                    diag_cells.append(
                        dict(
                            clip_id=clip_id,
                            rotor=r,
                            k=float(k),
                            tau=float(tau),
                            var=float(np.mean(resid**2)),
                            n=int(resid.size),
                        )
                    )
                    used_lags += 1
                    n_pairs_track += int(resid.size)
                if used_lags:
                    n_valid_cells_total += n_valid
                    n_lag_pairs_total += n_pairs_track
                    passed.append(
                        dict(
                            clip_id=clip_id,
                            rotor=r,
                            k=k,
                            lags=used_lags,
                            valid_mic_frame_cells=n_valid,
                            valid_fraction=float(valid.mean()),
                            lag_pairs=n_pairs_track,
                            median_valid_snr_db=float(np.median(snr[valid])),
                        )
                    )

    if not diag_cells:
        raise ValueError(
            "no (clip, rotor, order, lag) diagonal cell passed the PREREGISTERED moment gate "
            f"{gate.as_dict()}; rejections: {rejected}. The gate is not to be relaxed to "
            "manufacture an estimate — report it and pick a different support."
        )

    # Cross-order covariances: intersect the exact (frame, time, mic) supports
    # of each same-rotor order pair.  No independence is assumed across mics or
    # orders in the bootstrap/diagnostics; the intersection is per mic.
    grouped: dict[
        tuple[str, int, float, int], list[tuple[float, np.ndarray, np.ndarray]]
    ] = defaultdict(list)
    for key, (pid, res) in order_residuals.items():
        clip_id_k, r, k, tau, m = key
        grouped[(clip_id_k, r, tau, m)].append((k, pid, res))

    for (clip_id_k, r, tau, m), items in grouped.items():
        items_sorted = sorted(items, key=lambda x: x[0])
        for i in range(len(items_sorted)):
            k_i, pid_i, res_i = items_sorted[i]
            for j in range(i + 1, len(items_sorted)):
                k_j, pid_j, res_j = items_sorted[j]
                cov, n_common = _aligned_covariance(res_i, pid_i, res_j, pid_j)
                if n_common >= gate.min_frames:
                    offdiag_cells.append(
                        dict(
                            clip_id=clip_id_k,
                            rotor=r,
                            k=float(k_i),
                            l=float(k_j),
                            tau=float(tau),
                            cov=float(cov),
                            n=int(n_common),
                        )
                    )
                else:
                    cross_failure("cross-order pair too few common pairs")

    def offdiag_residual(
        log_p: np.ndarray, kl: np.ndarray, tau: np.ndarray, cov: np.ndarray
    ) -> np.ndarray:
        lam, sigma = np.exp(log_p)
        return kl * integrated_ou_increment_var(tau, lam=lam, sigma=sigma) - cov

    def fit_offdiag(
        kl: np.ndarray, tau: np.ndarray, cov: np.ndarray
    ) -> Any:
        lam0 = 1.0 / max(float(np.mean(tau)), 1e-6)
        basis = kl * integrated_ou_increment_var(tau, lam=lam0, sigma=1.0)
        coef, *_ = np.linalg.lstsq(basis[:, None], cov, rcond=None)
        sigma0 = math.sqrt(max(float(coef[0]), 1e-6))
        x0 = np.log([lam0, sigma0])
        return least_squares(offdiag_residual, x0, args=(kl, tau, cov), method="trf")

    def fit_d(
        k: np.ndarray, tau: np.ndarray, var: np.ndarray, lam: float, sigma: float
    ) -> tuple[float, Any]:
        shaft = k**2 * integrated_ou_increment_var(tau, lam=lam, sigma=sigma)
        resid = var - shaft
        tau2 = 2.0 * tau

        def d_residual(d: np.ndarray) -> np.ndarray:
            return tau2 * d[0] - resid

        d0 = max(float(np.median(resid / tau2)), 1e-12)
        r = least_squares(d_residual, [d0], bounds=(0.0, np.inf), method="trf")
        return float(r.x[0]), r

    offdiag_fit: Any | None = None
    d_res: Any | None = None
    lam = float("nan")
    sigma = float("nan")
    d_init = float("nan")

    if len(offdiag_cells) >= 2:
        kl_arr = np.array([c["k"] * c["l"] for c in offdiag_cells])
        tau_off = np.array([c["tau"] for c in offdiag_cells])
        cov_arr = np.array([c["cov"] for c in offdiag_cells])
        try:
            offdiag_fit = fit_offdiag(kl_arr, tau_off, cov_arr)
            lam, sigma = (float(v) for v in np.exp(offdiag_fit.x))
        except (ValueError, np.linalg.LinAlgError):
            offdiag_fit = None

    if offdiag_fit is not None and diag_cells:
        k_arr = np.array([c["k"] for c in diag_cells])
        tau_arr = np.array([c["tau"] for c in diag_cells])
        var_arr = np.array([c["var"] for c in diag_cells])
        d_init, d_res = fit_d(k_arr, tau_arr, var_arr, lam, sigma)

    cond = (
        float(np.linalg.cond(offdiag_fit.jac))
        if (offdiag_fit is not None and offdiag_fit.jac.size)
        else float("inf")
    )
    cov: list[list[float]] | None = None
    if offdiag_fit is not None:
        try:
            dof = max(offdiag_fit.fun.size - 2, 1)
            s2 = 2.0 * float(offdiag_fit.cost) / dof
            cov = (s2 * np.linalg.inv(offdiag_fit.jac.T @ offdiag_fit.jac)).tolist()
        except np.linalg.LinAlgError:
            cov = None

    per_clip: dict[str, list[float]] = {}
    clip_ids = sorted({str(c["clip_id"]) for c in diag_cells})
    if len(clip_ids) > 1:
        for cid in clip_ids:
            off_sel = [c for c in offdiag_cells if str(c["clip_id"]) == cid]
            diag_sel = [c for c in diag_cells if str(c["clip_id"]) == cid]
            if len(off_sel) < 2 or len(diag_sel) < 2:
                continue
            try:
                kl_c = np.array([c["k"] * c["l"] for c in off_sel])
                tau_c = np.array([c["tau"] for c in off_sel])
                cov_c = np.array([c["cov"] for c in off_sel])
                r_c = fit_offdiag(kl_c, tau_c, cov_c)
                lam_c, sigma_c = (float(v) for v in np.exp(r_c.x))
                k_d = np.array([c["k"] for c in diag_sel])
                tau_d = np.array([c["tau"] for c in diag_sel])
                var_d = np.array([c["var"] for c in diag_sel])
                d_c, _ = fit_d(k_d, tau_d, var_d, lam_c, sigma_c)
                per_clip[cid] = [lam_c, sigma_c, float(d_c)]
            except (ValueError, np.linalg.LinAlgError):
                continue
    block = (
        np.std(np.log(np.array(list(per_clip.values()))), axis=0).tolist()
        if len(per_clip) > 1
        else None
    )

    reasons: list[str] = []
    if offdiag_fit is None:
        reasons.append(
            "no identifiable shared-shaft structure function from cross-order covariances"
        )
        tau_min = tau_max = float("nan")
        n_offdiag_orders = n_offdiag_lags = 0
        n_offdiag_pairs = 0
    else:
        tau_min = float(tau_off.min())
        tau_max = float(tau_off.max())
        if not (lam * tau_min <= 0.5):
            reasons.append(
                f"shortest lag {tau_min:.4f} s is not short against 1/lam = {1.0 / lam:.4f} s"
            )
        if not (lam * tau_max >= 1.0):
            reasons.append(
                f"longest lag {tau_max:.4f} s does not reach 1/lam = {1.0 / lam:.4f} s"
            )
        if cond > 1e4:
            reasons.append(f"off-diagonal Jacobian condition number {cond:.3g} > 1e4")
        n_offdiag_orders = len({c["k"] for c in offdiag_cells} | {c["l"] for c in offdiag_cells})
        n_offdiag_lags = int(np.unique(tau_off).size)
        n_offdiag_pairs = len(
            {(c["clip_id"], c["rotor"], c["k"], c["l"]) for c in offdiag_cells}
        )
        if n_offdiag_orders < 2:
            reasons.append("fewer than two distinct orders in cross-order cells")
        if n_offdiag_lags < 3:
            reasons.append("fewer than three distinct lags in cross-order cells")

    k_arr = np.array([c["k"] for c in diag_cells])
    tau_arr = np.array([c["tau"] for c in diag_cells])
    n_diag_orders = int(np.unique(k_arr).size)
    n_diag_lags = int(np.unique(tau_arr).size)

    diagnostics: dict[str, Any] = dict(
        gate=gate.as_dict(),
        front_end=dict(
            **mc.as_dict(),
            sr=config.sr,
            note="the moment stage uses its OWN shorter window/hop, deliberately: lam is "
            "separated from sigma by the lag shape, not by spectral resolution",
        ),
        orders_passed=passed,
        rejections=rejected,
        crossorder_failures=cross_failures,
        gate_counts=dict(
            valid_mic_frame_cells=n_valid_cells_total,
            lag_pairs=n_lag_pairs_total,
            note="the preregistered SNR threshold is applied per MICROPHONE and per FRAME, "
            "isolation is a PER-FRAME mask, and lag pairs are formed only inside one "
            "continuous run of valid frames, so no pair straddles a gated frame",
        ),
        diagonal_cells=dict(
            n_cells=len(diag_cells),
            n_orders=n_diag_orders,
            n_lags=n_diag_lags,
            lag_range_s=[float(tau_arr.min()), float(tau_arr.max())],
            cells=diag_cells,
        ),
        crossorder_cells=dict(
            n_cells=len(offdiag_cells),
            n_distinct_order_pairs=n_offdiag_pairs,
            n_orders=n_offdiag_orders,
            n_lags=n_offdiag_lags,
            lag_range_s=[tau_min, tau_max],
            cells=offdiag_cells,
        ),
        n_cells=len(diag_cells),
        n_orders=n_diag_orders,
        n_lags=n_diag_lags,
        lag_range_s=[float(tau_arr.min()), float(tau_arr.max())],
        estimate=dict(lam=lam, sigma=sigma, d_init=d_init),
        d_initialization_note=(
            "D is initialized from diagonal residuals after subtracting the shared OU term; "
            "it is then refined by the spectral MAP"
        ),
        diagnostic_covariance_log_params=cov,
        diagnostic_covariance_note=(
            "DIAGNOSTIC only: the off-diagonal Jacobian gives the (lam, sigma) covariance; "
            "wrapped-phase residuals are correlated across lags, orders and mics, so this is "
            "not an exact sampling covariance and no GLS diagonal approximation is exact"
        ),
        per_clip_estimates=per_clip,
        block_variation_log_std=block,
        jacobian_cond=cond,
        residual_rms=(
            float(np.sqrt(np.mean(offdiag_fit.fun**2)))
            if offdiag_fit is not None
            else None
        ),
        diagonal_residual_rms=(
            float(np.sqrt(np.mean(d_res.fun**2)))
            if d_res is not None
            else None
        ),
        cells=diag_cells,
    )
    if reasons:
        diagnostics["unidentifiable_reason"] = "; ".join(reasons)
    return ShaftDynamics(
        lam=lam,
        sigma=sigma,
        d_init=d_init,
        identified=not reasons,
        diagnostics=diagnostics,
    )


# ── stage 2: the plug-in composite MAP ──────────────────────────────────────


@dataclass
class _ClipData:
    clip_id: str
    group: str
    n_samples: int
    starts: np.ndarray  # (N,) frame start samples
    weights: np.ndarray  # (N,) composite weights
    power: Tensor | None  # (M, N, F) observed periodogram, CPU float32
    raw_hz: Tensor  # (R, T) delayed telemetry, device float64
    raw_hz_work: Tensor  # (R, T * q) delayed telemetry on the work grid, float64
    frame_rate: Tensor  # (R, N) frame-mean delayed telemetry, device float64
    n_states: int
    meta: dict[str, Any]


def _frame_key(clip_id: str, meta: dict[str, Any], start: int, sr: int) -> tuple[str, int]:
    """The identity of an observation WINDOW, for :func:`composite_weights`.

    Recording-absolute, not clip-relative: two manifest clips cut from the same
    recording can name the very same audio window, and a clip-local key would
    let that window be scored twice. A clip with no recording provenance (a
    planted or synthetic clip) falls back to its own id.
    """
    recording = meta.get("recording_id")
    if recording is None:
        return (clip_id, int(start))
    offset = int(round(float(meta.get("start_s", 0.0)) * sr))
    return (f"{meta.get('dataset', '')}:{recording}", offset + int(start))


def _check_order_domain(
    max_carrier_hz: Sequence[float] | np.ndarray, *, k_cap: int, sample_rate_work: int, where: str
) -> None:
    """Raise if ANY modelled order of any rotor reaches the work Nyquist.

    ``max_carrier_hz`` is the per-rotor maximum of the REALIZED carrier over
    the span being synthesized — telemetry plus bias plus the state excursion
    (``nu``), not the raw track alone.

    This replaces the old whole-frame order mask. An order is never dropped and
    never truncated: at ``k_cap = 230`` and carriers below ~110 rev/s the
    highest modelled line is ~25.3 kHz against the 32 kHz work Nyquist, so a
    real clip cannot trip this; anything that does is outside the declared
    domain and must stop, loudly, rather than be silently discarded.
    """
    nyq = float(sample_rate_work) / 2.0
    for r, value in enumerate(np.atleast_1d(np.asarray(max_carrier_hz, dtype=np.float64))):
        f_max = float(value)
        if int(k_cap) * f_max < nyq:
            continue
        k_bad = int(math.ceil(nyq / f_max)) if f_max > 0.0 else int(k_cap)
        raise ValueError(
            f"{where}: rotor {r} order {k_bad} reaches {k_bad * f_max:.1f} Hz, at or above the "
            f"{nyq:.1f} Hz Nyquist of the {int(sample_rate_work)} Hz work grid (realized carrier "
            f"up to {f_max:.4f} rev/s, k_cap {int(k_cap)}). Out of the model's declared domain: "
            "raise sample_rate_work or lower k_cap — no order is ever silently dropped."
        )


class _RevisedModel(torch.nn.Module):
    """The candidate's expected periodogram and its parameters.

    One module for the whole rig: the physical parameters are shared by every
    clip, each clip owns its whitened state innovations and its bias.
    """

    def __init__(
        self,
        rows: Sequence[tuple[str, Clip]],
        *,
        dynamics: ShaftDynamics,
        config: FitConfig,
        device: str | torch.device = "cpu",
        observe: bool = True,
    ):
        super().__init__()
        if not rows:
            raise ValueError("at least one (clip_id, Clip) row is required")
        # BEFORE any state or bias block is allocated: a duplicated or
        # overlapping training window would buy a second state block and a
        # second bias prior out of the same audio.
        check_training_supports(rows)
        self.config = config
        self.dynamics = dynamics
        self.lam, self.sigma = float(dynamics.lam), float(dynamics.sigma)
        self.dt_state = 1.0 / float(config.state_rate_hz)
        dev = torch.device(device)
        self._dev = dev
        self.atom_dtype = torch.float32 if config.atom_dtype == "float32" else torch.float64

        n_fft, hop, sr = config.n_fft, config.hop, config.sr
        sr_work, n_fft_work = int(config.sample_rate_work), config.n_fft_work
        #: work samples per analysis sample; the frame starts map by this exact
        #: integer, which is why the work rate must be an integer multiple
        self.oversample = sr_work // sr
        self.n_rotors = int(rows[0][1].rps.shape[0])
        self.n_mics = int(rows[0][1].audio.shape[0])
        self.K = int(config.k_cap)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
        self.freqs_np = freqs
        band = (freqs >= config.band_hz[0]) & (freqs <= config.band_hz[1])
        if not band.any():
            raise ValueError(f"band {config.band_hz} selects no bin at n_fft {n_fft}, sr {sr}")
        window_work = hann_window(n_fft_work)
        self.window_work_sumsq = float(np.sum(window_work**2))
        #: work-grid periodogram -> analysis-grid periodogram units. The kernel
        #: divides by ``sum(w_work^2)`` and a tone's output scales with the
        #: window length, so a 4x longer window reads 4x high.
        self.grid_power_factor = float(sr) / float(sr_work)

        self.register_buffer("band", torch.as_tensor(band, device=dev))
        #: the work window in FLOAT64: the floor's amplitude envelope is formed
        #: in float64 and cast once, and the lines cast it to the atom dtype
        self.register_buffer(
            "window_work", torch.as_tensor(window_work, dtype=torch.float64, device=dev)
        )
        self.register_buffer(
            "k_orders", torch.arange(1, self.K + 1, dtype=torch.float64, device=dev)
        )
        self.register_buffer(
            "win_idx_work", torch.arange(n_fft_work, dtype=torch.int64, device=dev)
        )
        self.register_buffer(
            "tau_s_work",
            torch.arange(n_fft_work, dtype=torch.float64, device=dev) / float(sr_work),
        )
        # THE KNOWN TRANSFER, evaluated ONCE per fit: it depends only on the two
        # grids, never on a frame, a rotor or an order. Owned by the renderer's
        # own module so the fit and the render cannot drift apart.
        transfer = np.asarray(
            render_transfer_power(freqs, sample_rate_work=sr_work, sample_rate_out=sr),
            dtype=np.float64,
        )
        if transfer.shape != freqs.shape:
            raise ValueError(
                f"render_transfer_power returned {transfer.shape} for {freqs.shape} analysis "
                "frequencies: the power gain must be one value per analysis bin"
            )
        self.register_buffer(
            "transfer_power", torch.as_tensor(transfer, dtype=torch.float64, device=dev)
        )

        # FLOOR GEOMETRY, on the floor's own PSD/lag grid: the existing smooth
        # log-frequency shape of model.py, evaluated FLOOR_PSD_OVERSAMPLE times
        # finer than the work window so the autocovariance the kernel reads
        # carries none of the inverse FFT's periodic images. The control points
        # stay where the export declares them (analysis Nyquist); the shape is
        # clamped past the last one, which is exactly what the renderer does.
        ctrl_hz = np.geomspace(FLOOR_SHAPE_F_MIN, float(freqs[-1]), FLOOR_SHAPE_N_CTRL)
        ctrl_oct = np.log2(ctrl_hz / ctrl_hz[0])
        self.ctrl_hz = ctrl_hz
        self.floor_psd_len = int(FLOOR_PSD_OVERSAMPLE) * int(n_fft_work)
        #: the parameters are at the ANALYSIS convention; the floor is
        #: synthesized and read on the WORK grid (see floor_power_spectrum)
        self.floor_rate_factor = float(sr_work) / float(sr)
        shape_psd, tilt_psd = floor_geometry(
            np.fft.rfftfreq(self.floor_psd_len, d=1.0 / sr_work), ctrl_hz
        )
        self.register_buffer(
            "floor_shape_psd", torch.as_tensor(shape_psd, dtype=torch.float64, device=dev)
        )
        self.register_buffer(
            "floor_tilt_oct_psd", torch.as_tensor(tilt_psd, dtype=torch.float64, device=dev)
        )
        self.register_buffer(
            "shape_chol",
            torch.as_tensor(
                se_cholesky(FLOOR_SHAPE_N_CTRL, float(ctrl_oct[1] - ctrl_oct[0]), FLOOR_SHAPE_OCT),
                dtype=torch.float64,
                device=dev,
            ),
        )

        self.clips: list[_ClipData] = []
        frame_keys: list[tuple[str, int]] = []
        for clip_id, clip in rows:
            if int(clip.sr) != sr:
                raise ValueError(f"{clip_id}: sample rate {clip.sr} != front end {sr}")
            if int(clip.rps.shape[0]) != self.n_rotors:
                raise ValueError(f"{clip_id}: {clip.rps.shape[0]} rotors, expected {self.n_rotors}")
            if int(clip.audio.shape[0]) != self.n_mics:
                raise ValueError(f"{clip_id}: {clip.audio.shape[0]} mics, expected {self.n_mics}")
            n = int(clip.audio.shape[1])
            if n < n_fft:
                raise ValueError(f"{clip_id}: {n} samples is shorter than n_fft {n_fft}")
            starts = np.arange(1 + (n - n_fft) // hop) * hop
            frame_keys.extend(_frame_key(clip_id, clip.meta, int(s), sr) for s in starts)
            self.clips.append(
                _ClipData(
                    clip_id=clip_id,
                    group=clip.group,
                    n_samples=n,
                    starts=starts,
                    weights=np.zeros(starts.size),  # filled once all keys are known
                    power=None,
                    raw_hz=torch.zeros(1),
                    raw_hz_work=torch.zeros(1),
                    frame_rate=torch.zeros(1),
                    n_states=n_states_for(n, sr=sr, state_rate_hz=config.state_rate_hz),
                    meta=dict(clip.meta),
                )
            )
        weights = composite_weights(frame_keys, hop=hop, n_fft=n_fft)
        cut = 0
        delays = config.rotor_delays(self.n_rotors)
        for cd, (_, clip) in zip(self.clips, rows, strict=True):
            cd.weights = weights[cut : cut + cd.starts.size]
            cut += cd.starts.size
            raw = _delayed_rps(np.asarray(clip.rps, dtype=np.float64), delays, sr)
            cd.raw_hz = torch.as_tensor(raw, dtype=torch.float64, device=dev)
            # the same delayed telemetry, resampled onto the DECLARED work grid
            # the atoms are synthesized on (edge-held past the last sample, the
            # convention the renderer's own np.interp uses)
            t_src = np.arange(raw.shape[1]) / float(sr)
            t_work = np.arange(raw.shape[1] * self.oversample) / float(sr_work)
            cd.raw_hz_work = torch.as_tensor(
                np.stack([np.interp(t_work, t_src, r) for r in raw]),
                dtype=torch.float64,
                device=dev,
            )
            cd.frame_rate = torch.as_tensor(
                _frame_means(raw, cd.starts, n_fft), dtype=torch.float64, device=dev
            )
            if observe:
                cd.power = torch.as_tensor(
                    periodogram(clip, n_fft, hop).power, dtype=torch.float32
                )  # kept on the CPU; one chunk at a time is moved to the device

        f64 = dict(dtype=torch.float64, device=dev)
        zeros = lambda *s: torch.nn.Parameter(torch.zeros(*s, **f64))  # noqa: E731
        self.profile_db = zeros(self.n_rotors, self.K)
        self.amp_exp = torch.nn.Parameter(torch.tensor(AMP_EXP_INIT, **f64))
        self.floor_mean_db = zeros(1)
        self.floor_shape_z = zeros(FLOOR_SHAPE_N_CTRL)
        self.floor_tilt_db_oct = zeros(1)
        self.floor_exp = torch.nn.Parameter(torch.tensor(AMP_EXP_INIT, **f64))
        self.floor_static_raw = torch.nn.Parameter(torch.tensor(-6.0, **f64))
        self.mic_floor_db = zeros(self.n_mics)
        self.mic_gain_db = zeros(self.n_mics, self.n_rotors)
        self.gain_all_db = zeros(self.n_mics)
        self.log_d = torch.nn.Parameter(torch.tensor(math.log(max(dynamics.d_init, 1e-9)), **f64))
        self.bias_hz = zeros(len(self.clips), self.n_rotors)
        self.bias_mean_hz = zeros(self.n_rotors)
        self.state_innov = torch.nn.ParameterList(
            [zeros(self.n_rotors, cd.n_states, 2) for cd in self.clips]
        )
        if observe:
            self._init_levels()

    # ── initialization ──────────────────────────────────────────────────

    def _init_levels(self) -> None:
        """Crude but real initialization of the levels from the observed data.

        The floor mean is the in-band median periodogram; each order's power is
        the excess of its nearest bin over that floor, converted with the
        finite-window identity of a Hann-windowed tone (peak periodogram
        ``A^2 n_fft / 6``, so ``P = A^2/2 = 3 I_peak / n_fft``) and divided by
        the speed law at the clip's own rate.
        """
        band = self.band.detach().cpu().numpy()
        obs = [cd for cd in self.clips if cd.power is not None]
        if not obs:
            return
        med = float(np.median(np.concatenate([cd.power.numpy()[..., band].ravel() for cd in obs])))
        with torch.no_grad():
            self.floor_mean_db.fill_(10.0 * math.log10(max(med, 1e-30)))
            df = float(self.freqs_np[1] - self.freqs_np[0])
            prof = np.zeros((self.n_rotors, self.K))
            hits = np.zeros((self.n_rotors, self.K))
            for cd in obs:
                assert cd.power is not None
                power = cd.power.numpy().mean(axis=0)  # (N, F)
                rate = cd.frame_rate.detach().cpu().numpy()
                for r in range(self.n_rotors):
                    for k in range(1, self.K + 1):
                        bins = np.round(k * rate[r] / df).astype(np.int64)
                        ok = (bins >= 1) & (bins < power.shape[1] - 1)
                        if not ok.any():
                            continue
                        rows_i = np.arange(power.shape[0])[ok]
                        peak = power[rows_i, bins[ok]]
                        excess = max(float(np.median(peak)) - med, med * 1e-3, 1e-30)
                        p_ms = max(3.0 * excess / self.config.n_fft, 1e-30)
                        speed = max(float(np.median(rate[r][ok])) / AMP_RPS_REF, 1e-6)
                        prof[r, k - 1] += 10.0 * math.log10(p_ms) - AMP_EXP_INIT * 10.0 * (
                            math.log10(speed)
                        )
                        hits[r, k - 1] += 1.0
            filled = hits > 0
            init = np.where(filled, prof / np.maximum(hits, 1.0), -120.0)
            self.profile_db.copy_(torch.as_tensor(init, dtype=torch.float64, device=self._dev))

    # ── parameter views ─────────────────────────────────────────────────

    def fitted_parameters(self) -> dict[str, torch.nn.Parameter]:
        out: dict[str, torch.nn.Parameter] = {
            "profile_db": self.profile_db,
            "amp_exp": self.amp_exp,
            "floor_mean_db": self.floor_mean_db,
            "floor_shape_z": self.floor_shape_z,
            "floor_tilt_db_oct": self.floor_tilt_db_oct,
            "floor_exp": self.floor_exp,
            "floor_static_raw": self.floor_static_raw,
            "mic_floor_db": self.mic_floor_db,
            "mic_gain_db": self.mic_gain_db,
            "gain_all_db": self.gain_all_db,
            "log_d": self.log_d,
            "bias_hz": self.bias_hz,
            "bias_mean_hz": self.bias_mean_hz,
        }
        for i, p in enumerate(self.state_innov):
            out[f"state_innov[{self.clips[i].clip_id}]"] = p
        return out

    def mic_line_gain(self) -> Tensor:
        """``(M, R)`` linear gain on the line power; the mean over mics is pinned."""
        mg = self.mic_gain_db - self.mic_gain_db.mean(dim=0, keepdim=True)
        return 10.0 ** (mg / 10.0)

    def all_gain(self) -> Tensor:
        """``(M,)`` one recording-level gain on everything, mean removed."""
        return 10.0 ** ((self.gain_all_db - self.gain_all_db.mean()) / 10.0)

    def floor_shape_db(self) -> Tensor:
        return FLOOR_SHAPE_STD_DB * (self.shape_chol @ self.floor_shape_z)

    def theta_nu(self, ci: int) -> tuple[Tensor, Tensor]:
        return simulate_state(
            self.state_innov[ci], lam=self.lam, sigma=self.sigma, dt=self.dt_state
        )

    # ── forward ─────────────────────────────────────────────────────────

    def floor_psd(self) -> Tensor:
        """``(P // 2 + 1,)`` floor power spectrum on the work grid, in
        work-periodogram units, from the SHARED :func:`floor_power_spectrum`.

        Recomputed every step INSIDE the graph: it depends on the fitted mean,
        shape and tilt, so a detached numpy call would silently freeze the
        floor's colour at its initialization.
        """
        return floor_power_spectrum(
            self.floor_shape_psd,
            self.floor_tilt_oct_psd,
            mean_db=self.floor_mean_db,
            ctrl_db=self.floor_shape_db(),
            tilt_db_oct=self.floor_tilt_db_oct,
            rate_factor=self.floor_rate_factor,
        )

    def floor_covariance(self) -> tuple[Tensor, Tensor]:
        """``(c0, r_tau)``: the floor's variance and ``2 c(tau) / c(0)`` on the
        work lag grid.

        ``c = irfft(PSD)`` is the autocovariance of exactly the process
        :func:`render_revised` synthesizes — same spectrum, same units. The
        factor two is what makes the REAL part of a complex process of
        covariance ``2 R`` have covariance ``R`` (module docstring); the level
        is folded into the atom as ``sqrt(c0)`` so ``r_tau`` carries only shape.
        """
        c = torch.fft.irfft(self.floor_psd().to(torch.complex128), n=self.floor_psd_len)
        c0 = c[0].clamp_min(MODEL_FLOOR)
        return c0, 2.0 * c[: self.config.n_fft_work] / c0

    def _floor_frames(self, rate: Tensor) -> Tensor:
        """``(M, n_c, F)`` broadband floor of a chunk's frames, AS OBSERVED.

        ``rate`` is the chunk's ``(R, n_c, n_fft_work)`` WORK-grid carrier, the
        very samples the atoms use. The floor goes through the SAME
        finite-window kernel as the lines, with the real atom
        ``w(t) A_floor(t)`` and ``r_tau = 2 R_floor(tau)``, so it carries the
        window's response on a coloured spectrum and the within-window variation
        of its amplitude envelope EXACTLY: the renderer applies
        ``mean_r (rate_r(t)/ref)^floor_exp + static`` sample by sample on this
        grid, and this evaluates the same envelope on the same grid. No
        three-tap smoothing, no rectangular frame-mean rate.
        """
        # SPEED_FLOOR_RPS rather than 0: the speed law has a FITTED exponent,
        # and 0 ** negative is not finite. ONE constant, shared with
        # render_revised; a trajectory below the CALIBRATED domain is reported
        # by off_regime_extrapolation, never repaired here.
        gain_t = ((rate.clamp_min(SPEED_FLOOR_RPS) / AMP_RPS_REF) ** self.floor_exp).mean(
            dim=0
        ) + torch.nn.functional.softplus(self.floor_static_raw)  # (n_c, n_fft_work)
        c0, r_tau = self.floor_covariance()
        # FLOAT64 here regardless of ``atom_dtype``: the floor spans tens of dB
        # across the band, and a deep bin is a near-cancellation of terms set by
        # the loud end. This is ONE atom per frame — a couple of MB of FFT
        # buffers beside the lines' hundreds — so the precision is free.
        # ``c0`` cancels exactly between the atom and ``r_tau``; it is carried
        # only to keep both O(1).
        atom = self.window_work[None, :] * torch.sqrt(c0 * gain_t)
        shape_work = expected_periodogram_from_atoms(
            atom, r_tau, n_fft=self.config.n_fft_work, window_sumsq=self.window_work_sumsq
        )  # (n_c, n_fft_work // 2 + 1)
        floor = shape_work[..., : self.config.n_fft // 2 + 1] * self.grid_power_factor
        return floor[None] * 10.0 ** (self.mic_floor_db[:, None, None] / 10.0)

    def frame_model(
        self,
        ci: int,
        fi: np.ndarray,
        *,
        state: tuple[Tensor, Tensor] | None,
        kernel: str = "conditional",
        state_dt: float | None = None,
        bias: Tensor | None = None,
    ) -> Tensor:
        """``(M, n_c, F)`` expected periodogram of the chunk ``fi``.

        Atoms carry the ACTUAL within-window integrated carrier: the phase is
        ``2 pi cumsum((raw(t - delay) + b)/Fs_work) + theta(t)`` sampled at
        every WORK-grid sample of the window, never a frame-mean speed, never a
        chirp-width heuristic and never ``exp(i theta_centre)`` factored out of
        the window. Every component is summed in POWER — the cross terms vanish
        exactly under the independent uniform initial phases — so crossing
        orders remain shared continuous states and are never assigned to bins.

        EVERY order ``k = 1..k_cap`` contributes in EVERY frame: there is no
        whole-frame order gate. Out-of-band content is removed by the declared
        observation chain (the work grid plus the known transfer), which is the
        mechanism that physically removes it; an order that reaches the work
        Nyquist is out of domain and raises (:func:`_check_order_domain`).
        ``config.harmonic_chunk`` only splits the order axis into checkpointed
        blocks: a MEMORY device, never a change to what is modelled.

        This is the ONE code path that applies the transfer, so
        :func:`predict_spectrum` and the fit cannot disagree and the multiplier
        cannot be applied twice.
        """
        if kernel not in ("conditional", "prior"):
            raise ValueError(f"kernel must be 'conditional' or 'prior', got {kernel!r}")
        cd = self.clips[ci]
        cfg = self.config
        sr_work = float(cfg.sample_rate_work)
        n_fft_work = cfg.n_fft_work
        dt_state = self.dt_state if state_dt is None else float(state_dt)
        b = self.bias_hz[ci] if bias is None else bias
        starts = torch.as_tensor(
            cd.starts[fi] * self.oversample, dtype=torch.int64, device=self._dev
        )
        win = starts[:, None] + self.win_idx_work[None, :]  # (n_c, n_fft_work)
        rate = cd.raw_hz_work[:, win] + b[:, None, None]  # (R, n_c, n_fft_work)
        # the within-window integrated carrier. A LOCAL cumsum, not a slice of a
        # clip-long one: only phase DIFFERENCES inside the window reach the
        # kernel (a constant cancels under the uniform initial phase), and a
        # local sum is the better conditioned of the two.
        phase = (2.0 * np.pi / sr_work) * torch.cumsum(rate, dim=-1)
        if state is not None:
            theta, nu = state
            pos = (win.to(torch.float64) / sr_work) / dt_state
            phase = phase + _hermite(theta, nu, pos, dt_state)
        # the REALIZED instantaneous carrier, state excursion included: what the
        # domain check has to see, rather than the raw telemetry alone
        f_inst = torch.diff(phase, dim=-1) * (sr_work / (2.0 * np.pi))
        _check_order_domain(
            torch.amax(f_inst, dim=(1, 2)).detach().cpu().numpy(),
            k_cap=self.K,
            sample_rate_work=cfg.sample_rate_work,
            where=f"{cd.clip_id}: frame_model",
        )
        del f_inst
        # PHASE CENTERING, per window, in float64, BEFORE the multiplication by
        # k. A constant phase offset per (window, order) cancels IDENTICALLY in
        # |X(f)|^2 and in the atom's autocorrelation, so the expected-power
        # gauge permits it EXACTLY — it adds no stochastic structure and the
        # absolute phase must never be "restored". Centering on the window's
        # own midpoint halves the largest angle k multiplies, which is what
        # keeps the float32 cast below at 1e-7 rad instead of rounding a huge
        # accumulated angle.
        phase = phase - phase[..., n_fft_work // 2 : n_fft_work // 2 + 1]

        prof_amp = torch.sqrt(2.0 * 10.0 ** (self.profile_db / 10.0))  # (R, K)
        speed_amp = torch.sqrt((rate.clamp_min(SPEED_FLOOR_RPS) / AMP_RPS_REF) ** self.amp_exp)
        # the window and the speed law are order-INDEPENDENT: form the envelope
        # once, so an order block carries only its own phase and its own level
        env = speed_amp.to(self.atom_dtype) * self.window_work.to(self.atom_dtype)
        d_scalar = torch.exp(self.log_d)

        def order_block(
            prof_c: Tensor, k_c: Tensor, phase_c: Tensor, env_c: Tensor, d_c: Tensor
        ) -> Tensor:
            # k * (Phi - Phi_ref) and the wrap are BOTH in float64; only the
            # wrapped angle is cast
            kph = torch.remainder(k_c[None, :, None, None] * phase_c[:, None], 2.0 * np.pi)
            atoms = torch.polar(
                prof_c.to(self.atom_dtype)[:, :, None, None] * env_c[:, None],
                kph.to(self.atom_dtype),
            )
            if kernel == "conditional":
                r_tau = conditional_r_tau(self.tau_s_work, d=d_c)[None, None, None, :]
            else:
                r_tau = prior_r_tau(
                    self.tau_s_work[None, :],
                    k_c[:, None],
                    lam=self.lam,
                    sigma=self.sigma,
                    d=d_c,
                )[None, :, None, :]
            shapes_work = expected_periodogram_from_atoms(
                atoms, r_tau, n_fft=n_fft_work, window_sumsq=self.window_work_sumsq
            )  # (R, k_c, n_c, n_fft_work // 2 + 1)
            # Fs_work / n_fft_work == Fs / n_fft, so work bin j IS analysis bin
            # j: the analysis band is the HEAD of the work spectrum. It is NOT
            # every q-th bin — a stride here would read q times the intended
            # frequency and produce a perfectly plausible, completely wrong
            # spectrum.
            return shapes_work[..., : cfg.n_fft // 2 + 1].sum(dim=1)  # (R, n_c, F)

        k_chunk = self.K if cfg.harmonic_chunk is None else max(1, int(cfg.harmonic_chunk))
        shapes = torch.zeros(
            self.n_rotors,
            int(np.asarray(fi).size),
            cfg.n_fft // 2 + 1,
            dtype=self.atom_dtype,
            device=self._dev,
        )
        for k0 in range(0, self.K, k_chunk):
            args = (
                prof_amp[:, k0 : min(k0 + k_chunk, self.K)],
                self.k_orders[k0 : min(k0 + k_chunk, self.K)],
                phase,
                env,
                d_scalar,
            )
            if cfg.harmonic_chunk is not None and torch.is_grad_enabled():
                # a block's (R, k, n_c, 2 n_fft_work) interior IS the peak;
                # recomputing it in the backward pass costs about a third more
                # time and removes it (model.CombSpectrum.lines, same chunk)
                piece: Tensor = torch.utils.checkpoint.checkpoint(  # type: ignore[assignment]
                    order_block, *args, use_reentrant=False
                )
            else:
                piece = order_block(*args)
            shapes = shapes + piece
        shapes = shapes * self.grid_power_factor
        lines = torch.einsum("mr,rnf->mnf", self.mic_line_gain().to(shapes.dtype), shapes)
        floor = self._floor_frames(rate)  # float64
        # THE KNOWN TRANSFER, APPLIED EXACTLY ONCE, here and nowhere else: the
        # render AA at Fs_work composed with the resample_poly decimator to
        # Fs, cached per fit. Lines and floor pass through the same chain, so
        # the multiplier sits on their SUM. This is the window/filter
        # COMMUTATION APPROXIMATION (the real renderer filters before the Hann
        # window; here a filter power gain scales an already-windowed expected
        # spectrum) — the finite-window kernel itself stays exact. The sum is
        # formed in float64: the risk reads it in float64 anyway, and a float32
        # floor under a loud line is needless rounding.
        observed = (floor + lines.to(torch.float64)) * self.transfer_power[None, None, :]
        return observed * self.all_gain()[:, None, None]

    def chunk_risk(self, ci: int, fi: np.ndarray, *, scale: float = 1.0) -> Tensor:
        cd = self.clips[ci]
        if cd.power is None:
            raise ValueError(f"{cd.clip_id}: no observed periodogram (built with observe=False)")
        model = self.frame_model(ci, fi, state=self.theta_nu(ci), kernel="conditional")
        obs = cd.power.index_select(1, torch.as_tensor(fi, dtype=torch.int64)).to(
            device=self._dev, dtype=torch.float64
        )
        w = torch.as_tensor(cd.weights[fi] * scale, dtype=torch.float64, device=self._dev)
        return composite_risk(obs, model.to(torch.float64), w, band=self.band)

    def prior(self) -> Tensor:
        """``-log p`` of the state, the bias hierarchy and the parameters.

        ONE state prior: ``0.5 sum e^2`` over the USED innovations (the
        ``theta_0 = 0`` gauge slot ``innov[..., 0, 1]`` carries no innovation and
        is excluded). The ``log D`` prior is a broad Gaussian on a FITTED
        parameter — it is explicitly NOT conjugate to the noisy moment-derived
        ``D`` and must not be described as such.
        """
        cfg = self.config
        p = torch.zeros((), dtype=torch.float64, device=self._dev)
        for e in self.state_innov:
            p = p + 0.5 * (e[:, 0, 0].square().sum() + e[:, 1:, :].square().sum())
        p = p + 0.5 * self.floor_shape_z.square().sum()
        p = p + 0.5 * ((self.bias_hz - self.bias_mean_hz[None, :]) / cfg.bias_std_hz).square().sum()
        p = p + 0.5 * (self.bias_mean_hz / cfg.bias_mean_std_hz).square().sum()
        p = p + 0.5 * ((self.log_d - cfg.log_d_mean) / cfg.log_d_std) ** 2
        return p

    # ── export / load ───────────────────────────────────────────────────

    def parameter_export(self) -> dict[str, Any]:
        with torch.no_grad():
            mg = self.mic_gain_db - self.mic_gain_db.mean(dim=0, keepdim=True)
            return dict(
                lam=self.lam,
                sigma=self.sigma,
                d_scalar=float(torch.exp(self.log_d).item()),
                delay_s={
                    str(r): float(v) for r, v in enumerate(self.config.rotor_delays(self.n_rotors))
                },
                bias_hz={
                    cd.clip_id: self.bias_hz[i].cpu().numpy().tolist()
                    for i, cd in enumerate(self.clips)
                },
                bias_mean_hz=self.bias_mean_hz.cpu().numpy().tolist(),
                bias_prior_std_hz=float(self.config.bias_std_hz),
                profile_db=self.profile_db.cpu().numpy().tolist(),
                amp_exp=float(self.amp_exp.item()),
                floor_mean_db=float(self.floor_mean_db.item()),
                floor_shape_db=self.floor_shape_db().cpu().numpy().tolist(),
                floor_ctrl_hz=self.ctrl_hz.tolist(),
                floor_tilt_db_oct=float(self.floor_tilt_db_oct.item()),
                floor_exp=float(self.floor_exp.item()),
                floor_static_rel=float(torch.nn.functional.softplus(self.floor_static_raw).item()),
                mic_gain_db=mg.cpu().numpy().tolist(),
                gain_all_db=(self.gain_all_db - self.gain_all_db.mean()).cpu().numpy().tolist(),
                mic_floor_db=self.mic_floor_db.cpu().numpy().tolist(),
                k_cap=int(self.K),
                n_mics=int(self.n_mics),
                n_rotors=int(self.n_rotors),
            )

    def load_parameters(self, params: dict[str, Any]) -> None:
        """Load a :meth:`parameter_export` block (prediction and diagnostics)."""
        for key, have, name in (
            ("n_mics", self.n_mics, "microphones"),
            ("n_rotors", self.n_rotors, "rotors"),
            ("k_cap", self.K, "orders"),
        ):
            want = params.get(key)
            if want is not None and int(want) != int(have):
                raise ValueError(
                    f"export was fitted with {int(want)} {name}, this clip/config carries "
                    f"{int(have)}: the two are not comparable"
                )

        def put(p: torch.nn.Parameter, value: Any) -> None:
            p.copy_(torch.as_tensor(np.asarray(value, dtype=np.float64), device=self._dev))

        with torch.no_grad():
            put(self.profile_db, params["profile_db"])
            put(self.amp_exp, params["amp_exp"])
            put(self.floor_mean_db, np.atleast_1d(params["floor_mean_db"]))
            # the floor shape is exported as the dB CURVE; invert the whitening
            curve = torch.as_tensor(
                np.asarray(params["floor_shape_db"], dtype=np.float64), device=self._dev
            )
            z = torch.linalg.solve_triangular(
                self.shape_chol, (curve / FLOOR_SHAPE_STD_DB)[:, None], upper=False
            )[:, 0]
            self.floor_shape_z.copy_(z)
            put(self.floor_tilt_db_oct, np.atleast_1d(params["floor_tilt_db_oct"]))
            put(self.floor_exp, params["floor_exp"])
            self.floor_static_raw.copy_(
                torch.as_tensor(
                    float(np.log(np.expm1(max(float(params["floor_static_rel"]), 1e-12)))),
                    dtype=torch.float64,
                    device=self._dev,
                )
            )
            put(self.mic_floor_db, params["mic_floor_db"])
            put(self.mic_gain_db, params["mic_gain_db"])
            put(self.gain_all_db, params["gain_all_db"])
            self.log_d.copy_(
                torch.as_tensor(
                    math.log(max(float(params["d_scalar"]), 1e-30)),
                    dtype=torch.float64,
                    device=self._dev,
                )
            )
            put(self.bias_mean_hz, params["bias_mean_hz"])
            # a clip the export never saw has no bias of its own: the bias LAW
            # (the population mean) is what a new clip gets
            for i, cd in enumerate(self.clips):
                stored = params.get("bias_hz", {}).get(cd.clip_id)
                self.bias_hz[i].copy_(
                    torch.as_tensor(
                        np.asarray(
                            stored if stored is not None else params["bias_mean_hz"],
                            dtype=np.float64,
                        ),
                        device=self._dev,
                    )
                )


def _chunks(values: np.ndarray, size: int) -> list[np.ndarray]:
    size = max(1, int(size))
    return [values[i : i + size] for i in range(0, values.size, size)]


def _git_head() -> str | None:
    """The checkout's HEAD, read-only (``git rev-parse``), or ``None``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError:
        return None
    return out.stdout.strip() or None


#: The dynamics and grid the planted TWO-SIDED check validated the 500 Hz state
#: grid at (``tests/experiments/test_stochastic_fit_revised_phase.py``). A fit
#: outside this range has not been checked, and its export says so.
COARSENING_VALIDATED_AT: dict[str, float] = dict(lam=6.0, sigma=6.0, k=100.0, state_rate_hz=500.0)

COARSENING_SENSITIVITY_NOTE = (
    "SANITY INDICATOR ONLY — NOT A CONVERGENCE RESULT and never an acceptance criterion. It is "
    "the sensitivity of ONE frame's predicted spectrum to COARSENING the already-fitted MAP "
    "path by two. It never optimizes and never predicts from a REFINED grid, so a path that "
    "omitted recoverable subwindow motion can report an arbitrarily small change. The real "
    "check is a matched planted fit/prediction comparison at 500 vs 1000 Hz, run in the "
    "verification phase; if the fitted parameters fall outside the range validated there "
    "(COARSENING_VALIDATED_AT) this export MUST be flagged as needing real grid refinement "
    "before acceptance."
)


def _coarsening_sensitivity(model: _RevisedModel) -> dict[str, Any]:
    """Change in ONE frame's predicted spectrum when the fitted MAP path is
    SUBSAMPLED by two.

    This is NOT a convergence result and must not be used for acceptance: the
    "coarse" arm is a subsample of the path the fit already produced, not a fit
    or a prediction on a refined grid, so it can only say whether the
    prediction is sensitive to throwing half the fitted path away. The planted
    two-sided check in the tests is what justifies the grid; this block is a
    sanity indicator that travels with the artifact.
    """
    with torch.no_grad():
        ci = 0
        cd = model.clips[ci]
        fi = np.array([cd.starts.size // 2])
        theta, nu = model.theta_nu(ci)
        fine = model.frame_model(ci, fi, state=(theta, nu), kernel="conditional")
        coarse = model.frame_model(
            ci,
            fi,
            state=(theta[:, ::2].contiguous(), nu[:, ::2].contiguous()),
            state_dt=2.0 * model.dt_state,
        )
        band = model.band
        f, c = fine[..., band], coarse[..., band]
        rel = ((f - c).abs() / f.clamp_min(MODEL_FLOOR)).flatten()
        exceeded = [
            name
            for name, value in (
                ("lam", model.lam),
                ("sigma", model.sigma),
                ("k", float(model.K)),
            )
            if value > COARSENING_VALIDATED_AT[name]
        ]
        if float(model.config.state_rate_hz) < COARSENING_VALIDATED_AT["state_rate_hz"]:
            exceeded.append("state_rate_hz")
        return dict(
            clip_id=cd.clip_id,
            frame=int(fi[0]),
            rate_hz=float(model.config.state_rate_hz),
            coarse_rate_hz=float(model.config.state_rate_hz) / 2.0,
            max_rel_change=float(rel.max().item()),
            median_rel_change=float(rel.median().item()),
            band_l1_rel_change=float(
                ((f - c).abs().sum() / f.abs().sum().clamp_min(MODEL_FLOOR)).item()
            ),
            is_convergence_result=False,
            note=COARSENING_SENSITIVITY_NOTE,
            validated_at=dict(COARSENING_VALIDATED_AT),
            needs_real_grid_refinement=bool(exceeded),
            needs_real_grid_refinement_reason=(
                "outside the validated range in " + ", ".join(exceeded) if exceeded else ""
            ),
        )


def _bias_confounding(model: _RevisedModel) -> dict[str, Any]:
    """The finite-record confounding between ``b_{c,r}`` and the clip-mean of
    ``nu_r / 2 pi``: a slow OU excursion and a constant bias are the same
    observation over one 16 s clip. REPORTED, never "fixed" with a second free
    path."""
    with torch.no_grad():
        bias = model.bias_hz.detach().cpu().numpy()
        nu_mean = np.zeros_like(bias)
        for ci in range(len(model.clips)):
            _, nu = model.theta_nu(ci)
            nu_mean[ci] = nu.mean(dim=-1).cpu().numpy() / (2.0 * np.pi)
        out: dict[str, Any] = dict(
            bias_hz=bias.tolist(),
            clip_mean_nu_hz=nu_mean.tolist(),
            note="b_{c,r} and the clip mean of nu_r/2pi are confounded over a finite record; "
            "their SUM is what the carrier identifies",
        )
        if bias.shape[0] > 2:
            out["per_rotor_correlation"] = [
                float(np.corrcoef(bias[:, r], nu_mean[:, r])[0, 1])
                if np.std(bias[:, r]) > 0 and np.std(nu_mean[:, r]) > 0
                else float("nan")
                for r in range(bias.shape[1])
            ]
            out["sum_std_hz"] = np.std(bias + nu_mean, axis=0).tolist()
            out["bias_std_hz"] = np.std(bias, axis=0).tolist()
            out["nu_mean_std_hz"] = np.std(nu_mean, axis=0).tolist()
        return out


STATE_GRID_NOTE = (
    "cubic Hermite interpolation omits the OU bridge variance within a grid interval; the grid "
    "is justified by the planted TWO-SIDED check at 500 vs 1000 Hz, never by the export's "
    "coarsening_sensitivity block, which only coarsens an already-fitted path"
)


def fit_revised(
    rows: Sequence[tuple[str, Clip]],
    *,
    rig_id: str,
    dynamics: ShaftDynamics,
    config: FitConfig,
    device: str | torch.device = "cpu",
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fit the candidate by ordinary Adam on the plug-in composite MAP.

    ``rows`` are ``(clip_id, Clip)`` at ``config.sr``, whose ``rps`` track is the
    RAW telemetry — refined labels are never used to fit. The returned export is
    JSON-serializable and carries the physical parameters, the provenance needed
    to reproduce the fit, and TRAINING-ONLY diagnostics (the MAP state path among
    them; it is never a predictive input for a new clip).
    """
    t0 = time.time()
    torch.manual_seed(int(config.seed))
    model = _RevisedModel(rows, dynamics=dynamics, config=config, device=device, observe=True)
    named = model.fitted_parameters()
    opt = torch.optim.Adam(list(named.values()), lr=float(config.lr))
    rng = np.random.default_rng(int(config.seed))
    trace: list[float] = []
    grad_norms: dict[str, dict[str, float]] = {"first_step": {}, "last_step": {}}
    every = max(1, int(config.iters) // 20)

    for it in range(int(config.iters)):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for ci, cd in enumerate(model.clips):
            fi = np.arange(cd.starts.size)
            scale = 1.0
            if config.frames_per_step is not None and config.frames_per_step < fi.size:
                # UNIFORM sampling WITHOUT replacement; each selected frame keeps
                # its OWN composite weight and the sum is scaled by
                # N_full / N_sampled. That is UNBIASED for the full risk. The
                # self-normalized ``weights.sum() / weights[fi].sum()`` it
                # replaces was NOT: it handed every draw the clip's whole
                # exposure regardless of the drawn frames' weights, so with
                # unequal (duplicate-split) weights the duplicated windows
                # regained influence and the MAP target moved with
                # frames_per_step.
                n_full = int(fi.size)
                fi = np.sort(rng.choice(fi, size=int(config.frames_per_step), replace=False))
                scale = float(n_full) / float(fi.size)
            for chunk in _chunks(fi, config.frame_chunk):
                loss = model.chunk_risk(ci, chunk, scale=scale) / config.temperature
                loss.backward()
                total += float(loss.detach())
        prior = model.prior()
        prior.backward()
        total += float(prior.detach())
        step_norms = {
            name: (float(p.grad.norm().item()) if p.grad is not None else 0.0)
            for name, p in named.items()
        }
        if it == 0:
            grad_norms["first_step"] = step_norms
        grad_norms["last_step"] = step_norms
        opt.step()
        trace.append(total)
        if progress is not None and (it % every == 0 or it == int(config.iters) - 1):
            progress(f"iter {it:4d}  composite risk {total:.4f}")

    with torch.no_grad():
        map_state: dict[str, Any] = {}
        off_regime: dict[str, Any] = {}
        for ci, cd in enumerate(model.clips):
            theta, nu = model.theta_nu(ci)
            map_state[cd.clip_id] = dict(
                time_s=(np.arange(cd.n_states) * model.dt_state).tolist(),
                theta_rad=theta.cpu().numpy().tolist(),
                nu_hz=(nu.cpu().numpy() / (2.0 * np.pi)).tolist(),
            )
            off_regime[cd.clip_id] = off_regime_extrapolation(
                (cd.raw_hz + model.bias_hz[ci][:, None]).cpu().numpy(), where=cd.clip_id
            )

    provenance = dict(config.provenance)
    clip_rows = []
    for cd in model.clips:
        meta = cd.meta
        clip_rows.append(
            dict(
                clip_id=cd.clip_id,
                dataset=meta.get("dataset"),
                recording=meta.get("recording_id"),
                start_s=meta.get("start_s"),
                seconds=meta.get("duration_s"),
                channels=meta.get("channels"),
                rps_key=meta.get("rps_key"),
                regime=provenance.get("regimes", {}).get(cd.clip_id),
                group=cd.group,
                n_frames=int(cd.starts.size),
            )
        )
    training_provenance = dict(
        manifest_path=provenance.get("manifest_path"),
        manifest_sha256=provenance.get("manifest_sha256"),
        clips=clip_rows,
        front_end=config.front_end(),
        #: the two grids, named unambiguously: the atoms are synthesized on
        #: model_grid_hz and the periodogram is read on analysis_grid_hz.
        #: Nothing here is called "native".
        model_grid_hz=int(config.sample_rate_work),
        analysis_grid_hz=int(config.sr),
        render_transfer_spec=RENDER_TRANSFER_SPEC,
        observation_law=(
            "atoms on the model grid, expected periodogram read on the analysis grid, "
            "multiplied ONCE by the known render-AA + resample_poly power transfer "
            "(window/filter commutation approximation); profile_db is the source-side "
            "quantity of THIS chain, not a calibrated absolute spectrum"
        ),
        state_rate_hz=float(config.state_rate_hz),
        optimizer=dict(
            name="adam",
            iters=int(config.iters),
            lr=float(config.lr),
            frame_chunk=int(config.frame_chunk),
            harmonic_chunk=config.harmonic_chunk,
            frames_per_step=config.frames_per_step,
            minibatch_estimator="uniform without replacement, own weights, scaled N_full/N_"
            "sampled (unbiased for the full risk)",
            atom_dtype=config.atom_dtype,
            device=str(device),
        ),
        seed=int(config.seed),
        composite_temperature=float(config.temperature),
        composite_semantics=(
            "marginal composite risk / plug-in composite MAP: not an exact joint NLL, not "
            "an evidence, not a posterior; frame counts are not evidence counts"
        ),
        moment_gate=config.gate.as_dict(),
        moment_front_end=config.moments.as_dict(),
        priors=dict(
            bias_std_hz=float(config.bias_std_hz),
            bias_mean_std_hz=float(config.bias_mean_std_hz),
            log_d_mean=float(config.log_d_mean),
            log_d_std=float(config.log_d_std),
            log_d_note="a broad Gaussian prior on a FITTED parameter; NOT conjugate to the "
            "noisy moment-derived D",
        ),
        code_version=_git_head(),
    )
    # EVERY other key the caller put in config.provenance survives into the
    # serialized provenance — ``bench_diagnostic_only`` and ``scored_arm``
    # among them, so a downstream arm selector can enforce the
    # no-unreported-control-arm rule from the artifact itself instead of
    # trusting whoever ran it. Hand-listing the fields is what lost them.
    for key, value in provenance.items():
        training_provenance.setdefault(key, value)
    export = dict(
        schema_version=SCHEMA_VERSION,
        model_family=MODEL_FAMILY,
        rig_id=str(rig_id),
        parameters=model.parameter_export(),
        training_provenance=training_provenance,
        diagnostics=dict(
            map_state=map_state,
            map_state_note="TRAINING ONLY: a plug-in MAP path, never a predictive input for a "
            "new clip and never compared to the refined tracks as ground truth",
            moments=dynamics.diagnostics,
            identified=bool(dynamics.identified),
            loss_trace=trace,
            grad_norms=grad_norms,
            bias_vs_nu_mean_confounding=_bias_confounding(model),
            state_grid_note=STATE_GRID_NOTE,
            coarsening_sensitivity=_coarsening_sensitivity(model),
            off_regime_extrapolation=off_regime,
            complexity=dict(
                stochastic_mechanisms=2,
                rig_dynamic_params=3,
                rig_dynamic_params_detail="lam, sigma frozen (moment-estimated); D fitted",
                latent_state_blocks=len(model.clips) * model.n_rotors,
                latent_state_rate_hz=float(config.state_rate_hz),
                nuisance_blocks=dict(
                    bias_per_clip_rotor=int(model.bias_hz.numel()),
                    bias_population_mean=int(model.bias_mean_hz.numel()),
                    profile_db=int(model.profile_db.numel()),
                ),
                inference_stages=2,
                approximations=[
                    "cubic Hermite state interpolation omits the intra-interval OU bridge "
                    "variance (grid approximation)",
                    "plug-in composite risk over overlapping windows: proper for the mean "
                    "prediction, not a joint likelihood",
                    "lam, sigma frozen at their moment estimates",
                    "the known render/decimation transfer is applied to the WINDOWED expected "
                    "spectrum (window/filter commutation), not as an exact filtered periodogram",
                    "no order is dropped: out-of-band content is removed by the work grid and "
                    "the transfer, and an order at the work Nyquist raises",
                ],
            ),
            runtime_s=time.time() - t0,
        ),
    )
    return export


# ── prediction, rendering and the diagnostic carrier ────────────────────────


def _config_from_export(export: dict[str, Any], **overrides: Any) -> FitConfig:
    prov = export.get("training_provenance")
    if not isinstance(prov, dict):
        raise ValueError("export carries no training_provenance: cannot rebuild its front end")
    fe = prov.get("front_end")
    if not isinstance(fe, dict):
        raise ValueError("export.training_provenance.front_end is missing")
    params = export["parameters"]
    delays = params.get("delay_s")
    if delays is None:
        raise ValueError("export.parameters.delay_s is missing")
    if isinstance(delays, dict):
        delay_s = tuple(float(delays[k]) for k in sorted(delays, key=int))
    else:
        delay_s = tuple(float(v) for v in np.atleast_1d(delays))
    opt = prov.get("optimizer", {})
    cfg = FitConfig(
        n_fft=int(fe["n_fft"]),
        hop=int(fe["hop"]),
        sr=int(fe["sr"]),
        # the DECLARED model grid of that export, never guessed: a prediction on
        # a different grid is a different observation law
        sample_rate_work=int(fe["sample_rate_work"]),
        band_hz=tuple(float(v) for v in fe["band_hz"]),  # type: ignore[arg-type]
        state_rate_hz=float(prov["state_rate_hz"]),
        k_cap=int(params["k_cap"]),
        delay_s=delay_s,
        iters=int(opt.get("iters", 400)),
        lr=float(opt.get("lr", 0.05)),
        seed=int(prov.get("seed", 0)),
        frame_chunk=int(opt.get("frame_chunk", 1)),
        harmonic_chunk=opt.get("harmonic_chunk", 32),
        frames_per_step=opt.get("frames_per_step"),
        temperature=float(prov.get("composite_temperature", 1.0)),
        atom_dtype=str(opt.get("atom_dtype", "float32")),
    )
    priors = prov.get("priors", {})
    cfg.bias_std_hz = float(priors.get("bias_std_hz", params.get("bias_prior_std_hz", 0.5)))
    cfg.bias_mean_std_hz = float(priors.get("bias_mean_std_hz", BIAS_MEAN_PRIOR_STD_HZ))
    cfg.log_d_mean = float(priors.get("log_d_mean", 0.0))
    cfg.log_d_std = float(priors.get("log_d_std", 2.0))
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _dynamics_from_export(export: dict[str, Any]) -> ShaftDynamics:
    p = export["parameters"]
    diag = export.get("diagnostics", {})
    return ShaftDynamics(
        lam=float(p["lam"]),
        sigma=float(p["sigma"]),
        d_init=float(p["d_scalar"]),
        identified=bool(diag.get("identified", False)),
        diagnostics={"source": "export"},
    )


def predict_spectrum(
    export: dict[str, Any],
    clip: Clip,
    *,
    n_fft: int = N_FFT,
    hop: int = HOP,
    mode: str = "prior",
) -> np.ndarray:
    """``(M, N, F)`` expected periodogram, in ``data.periodogram(clip, n_fft,
    hop).power`` units and shape exactly.

    ``mode="prior"`` (the DEFAULT and the held-out path): the atoms carry the raw
    telemetry plus the learned bias LAW (the population mean ``mu_r``, because a
    new clip's ``b`` is unknown), ``m = 0``, and ``R_k`` is the FULL prior kernel
    (OU + ``D``) of :func:`prior_r_tau`. It never touches the clip's audio
    samples (only its length), never runs an optimization, never reads a refined
    label and never uses a stored training MAP path.

    ``mode="training_replay"``: diagnostic only. Requires ``clip.clip_id`` in the
    export's training diagnostics and RAISES otherwise; uses the stored MAP path
    in the atom with the conditional ``exp(-D|tau|)`` kernel. Stored MAP paths
    are diagnostics, never a default predictive input for a new clip.

    BOTH modes go through :meth:`_RevisedModel.frame_model`, so the prediction
    is the OBSERVED spectrum of the declared chain: atoms synthesized on the
    export's ``model_grid_hz``, read on the analysis grid, and multiplied ONCE
    by :func:`stage2.render_transfer_power`. There is no second application
    here and no unfiltered variant — a caller wanting the source-side spectrum
    divides by that same gain.
    """
    cfg = _config_from_export(export, n_fft=int(n_fft), hop=int(hop))
    model = _RevisedModel(
        [(clip.clip_id, clip)],
        dynamics=_dynamics_from_export(export),
        config=cfg,
        device="cpu",
        observe=False,
    )
    model.load_parameters(export["parameters"])
    cd = model.clips[0]
    if mode == "prior":
        with torch.no_grad():
            bias = model.bias_mean_hz.clone()
            out = [
                model.frame_model(0, chunk, state=None, kernel="prior", bias=bias)
                for chunk in _chunks(np.arange(cd.starts.size), cfg.frame_chunk)
            ]
    elif mode == "training_replay":
        stored = export.get("diagnostics", {}).get("map_state", {})
        if clip.clip_id not in stored:
            raise ValueError(
                f"mode='training_replay' needs the MAP state of {clip.clip_id!r}, and the "
                f"export's training diagnostics hold {sorted(stored)}"
            )
        entry = stored[clip.clip_id]
        theta = torch.as_tensor(np.asarray(entry["theta_rad"], dtype=np.float64))
        nu = torch.as_tensor(np.asarray(entry["nu_hz"], dtype=np.float64)) * 2.0 * np.pi
        with torch.no_grad():
            out = [
                model.frame_model(0, chunk, state=(theta, nu), kernel="conditional")
                for chunk in _chunks(np.arange(cd.starts.size), cfg.frame_chunk)
            ]
    else:
        raise ValueError(f"mode must be 'prior' or 'training_replay', got {mode!r}")
    return torch.cat(out, dim=1).cpu().numpy()


def infer_carrier(
    export: dict[str, Any], clip: Clip, *, config: FitConfig | None = None
) -> CarrierTrack:
    """The DIAGNOSTIC telemetry-conditioned MAP carrier track of one clip.

    A plug-in MAP track, NOT a posterior mean: the physical parameters are held
    at the export's values and only this clip's state innovations and bias are
    optimized, under the same composite risk and the same exact state prior.
    Never called implicitly inside :func:`predict_spectrum` or
    :func:`render_revised`.

    Comparison against the project's existing refined rotor-speed tracks is
    DIAGNOSTIC only: disagreement is not a failure, and the refined tracks are
    not ground truth.
    """
    cfg = config or _config_from_export(export)
    model = _RevisedModel(
        [(clip.clip_id, clip)],
        dynamics=_dynamics_from_export(export),
        config=cfg,
        device="cpu",
        observe=True,
    )
    model.load_parameters(export["parameters"])
    with torch.no_grad():
        model.bias_hz[0].copy_(model.bias_mean_hz)
    fitted = [model.state_innov[0], model.bias_hz]
    opt = torch.optim.Adam(fitted, lr=float(cfg.lr))
    cd = model.clips[0]
    trace: list[float] = []
    for _ in range(int(cfg.iters)):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for chunk in _chunks(np.arange(cd.starts.size), cfg.frame_chunk):
            loss = model.chunk_risk(0, chunk) / cfg.temperature
            loss.backward()
            total += float(loss.detach())
        prior = model.prior()
        prior.backward()
        total += float(prior.detach())
        opt.step()
        trace.append(total)
    with torch.no_grad():
        theta, nu = model.theta_nu(0)
        theta_np = theta.cpu().numpy()
        nu_hz = nu.cpu().numpy() / (2.0 * np.pi)
        bias = model.bias_hz[0].cpu().numpy()
        time_s = np.arange(cd.n_states) * model.dt_state
        raw_at = np.stack(
            [
                np.interp(
                    time_s,
                    np.arange(cd.n_samples) / cfg.sr,
                    cd.raw_hz[r].cpu().numpy(),
                )
                for r in range(model.n_rotors)
            ]
        )
    return CarrierTrack(
        time_s=time_s,
        nu_hz=nu_hz,
        theta_rad=theta_np,
        bias_hz=bias,
        total_hz=raw_at + bias[:, None] + nu_hz,
        diagnostics=dict(
            clip_id=clip.clip_id,
            loss_trace=trace,
            state_rate_hz=float(cfg.state_rate_hz),
            iters=int(cfg.iters),
            kind="plug-in MAP track (not a posterior mean)",
            refined_track_comparison="diagnostic only; the refined tracks are not ground truth",
            state_grid_note=STATE_GRID_NOTE,
        ),
    )


def render_revised(
    export: dict[str, Any],
    rps: np.ndarray,
    *,
    sample_rate: int = SR,
    sample_rate_work: int = SAMPLE_RATE_WORK,
    n_mics: int = 8,
    seed: int = 0,
) -> RevisedRender:
    """Synthesize the candidate on the declared WORK grid and decimate on the
    real clips' own path.

    ONE FRESH OU DRAW of ``(theta, nu)`` and fresh ``eps``, driven by the RAW
    telemetry ``rps`` (``(R, T)`` at ``sample_rate``) plus the learned bias law.
    No refined labels, no inferred corrections, no posterior path — this is the
    primary held-out predictive renderer.

    ``sample_rate_work`` (default 64000 = 4 x 16000) is the same declared grid
    the fit synthesizes its atoms on; "native" is retired as a name because
    this is a model grid and not any recording's sampling rate. The render
    low-passes ITSELF with :func:`stage2.antialias` (imported, not
    reimplemented) before the unchanged :func:`clips.decimate`, because the
    comb runs past the output Nyquist — and that AA plus that decimator are
    exactly the chain :func:`stage2.render_transfer_power` describes to the
    fit, which is why the synthetic arm and the fitted prediction land on the
    same observed spectrum.

    EVERY order ``1..k_cap`` is rendered. An order that reaches the work
    Nyquist raises (:func:`_check_order_domain`) instead of being skipped.

    ``physical_rps`` is the interval average of the EXACT phase, excluding
    ``eps``, on the output grid and with exactly the audio's length and
    timeline; ``reference_rps`` is the shared raw telemetry every arm is scored
    against (the MAE against it is primary, ``physical_rps`` is the
    diagnostic).
    """
    from . import clips as C

    params = export["parameters"]
    lam, sigma = float(params["lam"]), float(params["sigma"])
    d_scalar = float(params["d_scalar"])
    profile_db = np.asarray(params["profile_db"], dtype=np.float64)
    n_rotors, n_orders = profile_db.shape
    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    if rps.shape[0] != n_rotors:
        raise ValueError(f"rps has {rps.shape[0]} rotors, the export has {n_rotors}")
    delays = _config_from_export(export).rotor_delays(n_rotors)

    ss = np.random.SeedSequence(int(seed))
    rng_state, rng_eps, rng_alpha, rng_floor = (np.random.default_rng(s) for s in ss.spawn(4))

    n_in = rps.shape[1]
    n_work = int(round(n_in / sample_rate * sample_rate_work))
    t_src = np.arange(n_in) / float(sample_rate)
    t_work = np.arange(n_work) / float(sample_rate_work)
    raw_work = np.stack([np.interp(t_work, t_src, r) for r in rps])
    raw_work = np.stack(
        [np.interp(t_work - float(d), t_work, r) for d, r in zip(delays, raw_work, strict=True)]
    )
    bias_mean = np.asarray(params["bias_mean_hz"], dtype=np.float64)
    reference_carrier = raw_work + bias_mean[:, None]  # (R, T) Hz, unclamped
    # THE CALIBRATED DOMAIN IS 20 rev/s AND ABOVE. A trajectory below it is
    # REPORTED as unsupported extrapolation — never repaired into credible
    # motors-off physics by a clamp.
    off_regime = off_regime_extrapolation(reference_carrier, where="render_revised")
    if off_regime["off_regime"]:
        warnings.warn(
            f"render_revised: {off_regime['fraction_below_calibrated'] * 100:.1f}% of the input "
            f"trajectory is below the calibrated {CALIBRATED_MIN_RPS} rev/s domain (minimum "
            f"{off_regime['min_rps']:.4f} rev/s). The speed laws are UNSUPPORTED there; the "
            "render stays finite only because both paths clamp at SPEED_FLOOR_RPS.",
            RuntimeWarning,
            stacklevel=2,
        )
    # the SAME positive speed floor the fitted prediction uses, so a stopped
    # rotor cannot raise 0 to a negative fitted exponent here and stay finite
    # there (SPEED_FLOOR_RPS)
    f0 = np.maximum(reference_carrier, SPEED_FLOOR_RPS)  # (R, T) Hz
    dt = 1.0 / float(sample_rate_work)

    # fresh integrated-OU draw on the work grid, exact transition
    innov = rng_state.standard_normal((n_rotors, n_work, 2))
    theta, nu = simulate_state(innov, lam=lam, sigma=sigma, dt=dt)
    phase = 2.0 * np.pi * np.cumsum(f0, axis=1) * dt + theta  # (R, T) rad
    inst_rps = np.gradient(phase, dt, axis=1) / (2.0 * np.pi)
    # the domain check runs BEFORE any per-order draw, on the REALIZED carrier
    # (the fresh OU excursion included), so an out-of-domain export stops
    # instead of quietly losing an order
    _check_order_domain(
        inst_rps.max(axis=1),
        k_cap=n_orders,
        sample_rate_work=sample_rate_work,
        where="render_revised",
    )

    amp_exp = float(params["amp_exp"])
    mic_gain_db = np.asarray(params["mic_gain_db"], dtype=np.float64)
    gain_all_db = np.asarray(params["gain_all_db"], dtype=np.float64)
    mic_floor_db = np.asarray(params["mic_floor_db"], dtype=np.float64)
    if mic_gain_db.shape[0] < n_mics:
        raise ValueError(f"export carries {mic_gain_db.shape[0]} mics, asked for {n_mics}")
    mic_gain_db = mic_gain_db[:n_mics, :n_rotors]
    gain_all_db = gain_all_db[:n_mics]
    mic_floor_db = mic_floor_db[:n_mics]
    line_gain = 10.0 ** ((mic_gain_db - mic_gain_db.mean(axis=0, keepdims=True)) / 10.0)
    all_gain = 10.0 ** ((gain_all_db - gain_all_db.mean()) / 10.0)

    audio = np.zeros((n_mics, n_work), dtype=np.float64)
    speed = f0 / AMP_RPS_REF  # f0 already carries SPEED_FLOOR_RPS
    rendered_orders = 0
    for r in range(n_rotors):
        line_amp_r = np.sqrt(2.0 * 10.0 ** (profile_db[r] / 10.0))
        speed_amp = np.sqrt(speed[r] ** amp_exp)
        for k in range(1, n_orders + 1):
            eps = np.concatenate(
                (
                    [0.0],
                    np.cumsum(math.sqrt(2.0 * d_scalar * dt) * rng_eps.standard_normal(n_work - 1)),
                )
            )
            arg = k * phase[r] + eps
            env = line_amp_r[k - 1] * speed_amp
            # alpha_mrk: the fixed mic response phase, which absorbs the shared
            # uniform initial source phase. It is drawn ONCE per (rotor, order)
            # as an (n_mics,) vector and is CONSTANT IN TIME for the whole
            # render: a fixed per-render mic response, NEVER per-sample
            # randomness. It is not fitted because it is not identifiable from
            # an expected periodogram — the model's independent uniform initial
            # phase is exactly what makes every component add in power.
            # CONSEQUENCE, stated plainly: inter-channel phase structure at each
            # line is a fixed random response rather than a geometric one.
            # Adding a geometry model is deliberately NOT done here.
            alpha = rng_alpha.uniform(0.0, 2.0 * np.pi, size=n_mics)
            # Angle addition: cos(arg + alpha_m) = cos(arg) cos(alpha_m) -
            # sin(arg) sin(alpha_m). TWO full-length trig passes per (rotor,
            # order) instead of n_mics of them (8x fewer; ~2.9e9 cos calls on a
            # 16 s 4-rotor 130-order clip). Do NOT "simplify" this back to
            # np.cos(arg + alpha[m]) inside the mic loop. The per-mic loop is
            # kept deliberately: an (n_mics, n_work) outer product would
            # allocate ~45 MB per order.
            ec = env * np.cos(arg)
            es = env * np.sin(arg)
            for m in range(n_mics):
                g_m = math.sqrt(line_gain[m, r])
                audio[m] += (g_m * math.cos(alpha[m])) * ec
                audio[m] -= (g_m * math.sin(alpha[m])) * es
            rendered_orders += 1

    # broadband floor: white noise shaped to the fitted floor, with the floor's
    # own speed law as a slowly varying amplitude envelope — the SAME power
    # spectrum the fit reads (floor_power_spectrum) and the SAME envelope the
    # fit evaluates sample by sample, so the two cannot drift.
    #
    # THE RATE FACTOR, AND ONLY THE RATE FACTOR, lives in that spectrum. A
    # periodogram in ``data.periodogram`` units reads ``P_phys * sr/2`` (white
    # noise of variance s^2 reads s^2 at ANY rate), so broadband noise shaped
    # on the work grid reads ``sample_rate / sample_rate_work`` times LOWER
    # after decimation — 6.02 dB at 64 -> 16 kHz. Tones are unaffected
    # (decimation preserves amplitude), the floor is not, so it is
    # pre-compensated. That SCALAR was never enough on its own: that was
    # exactly the defect. The frequency-dependent part of the chain (the render
    # AA and the decimator's ~4.9 dB rolloff at 7900 Hz) is carried by the
    # transfer the FIT applies, and it reaches the render physically, below,
    # through the very same antialias + decimate calls.
    shape_mat, tilt_oct = floor_geometry(
        np.fft.rfftfreq(n_work, d=dt), np.asarray(params["floor_ctrl_hz"], dtype=np.float64)
    )
    floor_psd = floor_power_spectrum(
        shape_mat,
        tilt_oct,
        mean_db=float(params["floor_mean_db"]),
        ctrl_db=np.asarray(params["floor_shape_db"], dtype=np.float64),
        tilt_db_oct=float(params["floor_tilt_db_oct"]),
        rate_factor=float(sample_rate_work) / float(sample_rate),
    )
    floor_gain_t = (speed ** float(params["floor_exp"])).mean(axis=0) + float(
        params["floor_static_rel"]
    )
    for m in range(n_mics):
        white = rng_floor.standard_normal(n_work)
        shaped = np.fft.irfft(np.fft.rfft(white) * np.sqrt(floor_psd), n=n_work)
        audio[m] += shaped * np.sqrt(floor_gain_t) * math.sqrt(10.0 ** (mic_floor_db[m] / 10.0))
    audio *= np.sqrt(all_gain)[:, None]

    audio = antialias(audio, sample_rate_work)
    clip = Clip(
        "revised_render",
        "synthetic",
        np.asarray(audio, dtype=np.float32),
        inst_rps,
        int(sample_rate_work),
    )
    out = C.decimate(clip, int(sample_rate))
    n_out = int(out.audio.shape[1])
    dt_out = 1.0 / float(sample_rate)
    t_out = np.arange(n_out + 1) * dt_out
    # the exact phase on the OUTPUT grid. The one point past the work-grid span
    # (the last interval's right edge) is extrapolated with the final
    # instantaneous rate, never held flat, so the last interval average is the
    # rate the render actually ends on.
    phi_out = np.stack([np.interp(t_out, t_work, phase[r]) for r in range(n_rotors)])
    tail = t_out > t_work[-1]
    if tail.any():
        phi_out[:, tail] = (
            phase[:, -1:] + 2.0 * np.pi * inst_rps[:, -1:] * (t_out[tail] - t_work[-1])[None, :]
        )
    physical_rps = np.diff(phi_out, axis=1) / (2.0 * np.pi * dt_out)
    reference_rps = np.stack([np.interp(t_out[:n_out], t_src, r) for r in rps])
    return RevisedRender(
        audio=np.asarray(out.audio, dtype=np.float32),
        physical_rps=physical_rps,
        reference_rps=reference_rps,
        sample_rate=int(sample_rate),
        diagnostics=dict(
            seed=int(seed),
            sample_rate_work=int(sample_rate_work),
            n_work=n_work,
            model_grid_hz=int(sample_rate_work),
            analysis_grid_hz=int(sample_rate),
            lam=lam,
            sigma=sigma,
            d_scalar=d_scalar,
            rendered_orders=rendered_orders,
            rendered_orders_note="every order 1..k_cap of every rotor is rendered; an order "
            "reaching the work Nyquist raises instead of being skipped",
            bias_mean_hz=bias_mean.tolist(),
            off_regime_extrapolation=off_regime,
            eps_in_physical_rps=False,
            physical_rps_definition="(Phi(t_{i+1}) - Phi(t_i)) / (2 pi dt_out), eps excluded",
            reference_rps_definition="the raw telemetry every arm is scored against (primary)",
            mean_abs_physical_minus_reference_rps=float(
                np.mean(np.abs(physical_rps - reference_rps))
            ),
        ),
    )


__all__ = [
    "BIAS_MEAN_PRIOR_STD_HZ",
    "CALIBRATED_MIN_RPS",
    "COARSENING_SENSITIVITY_NOTE",
    "COARSENING_VALIDATED_AT",
    "FLOOR_PSD_OVERSAMPLE",
    "GATE_MAX_WRAP_SPREAD_RAD",
    "GATE_MIN_FRAMES",
    "GATE_MIN_ISOLATION_BINS",
    "GATE_MIN_LINE_SNR_DB",
    "GATE_ORDER_RANGE",
    "MODEL_FAMILY",
    "MOMENT_GATE",
    "RAW_RPS_KEYS",
    "SAMPLE_RATE_WORK",
    "SCHEMA_VERSION",
    "SPEED_FLOOR_RPS",
    "STATE_GRID_NOTE",
    "TRAINING_COHORT",
    "CarrierTrack",
    "FitConfig",
    "MomentConfig",
    "MomentGate",
    "RevisedRender",
    "ShaftDynamics",
    "check_manifest_supports",
    "check_rotor_track_key",
    "check_training_cohort",
    "check_training_supports",
    "composite_risk",
    "composite_weights",
    "conditional_r_tau",
    "estimate_shaft_dynamics",
    "fit_revised",
    "floor_geometry",
    "floor_power_spectrum",
    "hermite_theta",
    "infer_carrier",
    "integrated_ou_increment_var",
    "line_snr_db",
    "n_states_for",
    "off_regime_extrapolation",
    "ou_cholesky",
    "ou_transition",
    "predict_spectrum",
    "prior_r_tau",
    "render_revised",
    "simulate_state",
    "unidentified_diagnostic",
    "valid_lag_pairs",
]
