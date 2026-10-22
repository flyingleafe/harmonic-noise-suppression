"""Static rotor-spectral noise model — the *simplest* generator that forces
harmonic tracking.

Motivation (E8). A neural noise generator trained to match real drone noise
(``PositionalHarmonicNoiseGen``) still fails to teach an RPS predictor that
transfers to real data (E7: real val PIT MSE ~222, R^2 -10.5, both PIT). The
hypothesis: on generated data the predictor *reverse-engineers the amplitude
dynamics* (harmonic/noise amplitudes that co-vary with RPS) instead of tracking
the harmonic comb's *frequency* — and that amplitude->RPS shortcut does not
exist in real recordings.

This module removes the shortcut by construction:

* **Static harmonic comb.** Per clip, a fixed per-harmonic amplitude profile
  ``a_k`` (k = 1..K) that is **constant in time and independent of RPS**. The
  comb sits at ``k * rps(t)`` (matching the neural gen's ``f0 = rps``
  convention, ``harmonic_gen_new.py``: ``f0s = ms``), so the *only* cue for RPS
  is the comb's frequency spacing.
* **Static broadband floor.** A fixed pink-ish spectrum at a level such that
  **at least ``min_harm_above_floor`` (default 30%) of the in-band harmonics of
  every rotor clear the floor** — harmonics stay trackable, but (like real
  recordings) the high ones may wash out.
* **Wide profile variety.** Each clip samples a fresh profile (rolloff, blade
  emphasis, per-harmonic irregularity, floor level/tilt) drawn *widely* across
  ranges calibrated from real DREGON + Michael's noise, so the predictor cannot
  memorise one envelope — yet within a clip amplitudes carry zero RPS info.

The RPS trajectory (``data_processing.rps_synthesis``) drives the comb and is
the exact, noise-free label. ``StaticCombNoisePool`` exposes the same
``sample_timeframe(rng, duration_s) -> td.Frame`` interface as the other noise
pools (``kind: static_comb``); synthesis is cheap analytic numpy, so it runs
directly in the DataLoader workers — no GPU, no producer process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tdseries as td

from data_processing import rps_synthesis, trajectory_model
from data_processing.frames import make_recording_frame
from data_processing.noise_model.render import ENV_RATE_HZ, _render_lines


def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    """``cfg[key]`` for a mapping, ``cfg.key`` for an OmegaConf node or object."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ── Profile sampling ────────────────────────────────────────────────────────


@dataclass
class ProfileRanges:
    """Sampling ranges for the static rotor-spectral profile, wide enough to
    span (and slightly exceed) what is measured on real DREGON + Michael's.

    Defaults are lightly calibrated (see ``estimate_profile_stats`` +
    ``scripts``/diagnostics); override via config to widen/narrow the family.
    """

    # Harmonic amplitude rolloff a_k ~ k**(-p); larger p = faster high-harmonic
    # decay. Real single-rotor combs measure p ~ 0.6..1.6.
    rolloff_p: tuple[float, float] = (0.4, 1.9)
    # Per-harmonic static irregularity (dB, Gaussian) — spectral "texture".
    harm_jitter_db: tuple[float, float] = (2.0, 8.0)
    # Blade multiplicity: emphasise harmonics whose index is a multiple of b
    # (blade-pass dominance). b sampled from this set; emphasis strength in dB.
    blade_counts: tuple[int, ...] = (1, 2, 3)
    blade_emphasis_db: tuple[float, float] = (0.0, 10.0)
    # Broadband floor: PSD ~ f**(-tilt); level in dB relative to the comb's
    # median in-band harmonic amplitude (negative = floor below the comb).
    # Calibrated to real single-rotor DREGON/Michael's: floor sits only
    # -1.6..-11.6 dB below the comb (a high floor => realistic washout).
    floor_tilt: tuple[float, float] = (0.0, 1.5)
    floor_rel_db: tuple[float, float] = (-16.0, -1.0)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ProfileRanges:
        if not d:
            return cls()
        f = cls()
        for k, v in d.items():
            if hasattr(f, k):
                setattr(f, k, tuple(v) if isinstance(v, (list, tuple)) else v)
        return f


@dataclass
class RotorProfile:
    """A concrete static per-rotor spectral profile for one clip."""

    a_k: np.ndarray  # (K,) harmonic amplitudes (linear), a_1 normalised to 1
    floor_tilt: float  # PSD ~ f**(-tilt)
    floor_level: float  # broadband floor amplitude scale (linear, rel to comb)
    frac_above_floor: float = field(default=0.0)  # diagnostic


def sample_profile(
    rng: np.random.Generator,
    ranges: ProfileRanges,
    *,
    n_harmonics: int,
    ref_rps: float,
    sample_rate: int,
    min_harm_above_floor: float = 0.30,
) -> RotorProfile:
    """Draw one static rotor profile, enforcing the >=``min_harm_above_floor``
    fraction of in-band harmonics above the broadband floor.

    ``ref_rps`` (rev/s) sets which harmonics are in-band (k*ref_rps < Nyquist)
    for the floor-coverage constraint; the profile itself is RPS-independent.
    """
    K = int(n_harmonics)
    k = np.arange(1, K + 1, dtype=np.float64)

    p = rng.uniform(*ranges.rolloff_p)
    a = k ** (-p)
    # Blade-pass emphasis: boost every b-th harmonic.
    b = int(rng.choice(ranges.blade_counts))
    emph_db = rng.uniform(*ranges.blade_emphasis_db)
    if b > 1 and emph_db > 0:
        a[(np.arange(1, K + 1) % b) == 0] *= 10.0 ** (emph_db / 20.0)
    # Static per-harmonic irregularity (fixed for the clip).
    jit_db = rng.uniform(*ranges.harm_jitter_db)
    a *= 10.0 ** (rng.normal(0.0, jit_db, size=K) / 20.0)
    a /= a[0] if a[0] > 0 else 1.0  # normalise fundamental to 1

    tilt = rng.uniform(*ranges.floor_tilt)
    floor_rel_db = rng.uniform(*ranges.floor_rel_db)

    # In-band harmonic mask (k*ref_rps below Nyquist).
    nyq = sample_rate / 2.0
    in_band = (k * max(ref_rps, 1e-6)) < nyq
    n_band = int(in_band.sum()) or 1

    # The floor's amplitude at harmonic k scales as (k*ref_rps)**(-tilt/2)
    # (amplitude ~ sqrt(PSD)); normalise so floor_level multiplies a reference.
    fk = (k * max(ref_rps, 1e-6)) ** (-tilt / 2.0)
    fk /= fk[0] if fk[0] > 0 else 1.0
    comb_ref = float(np.median(a[in_band])) if n_band else 1.0

    def frac_above(level: float) -> float:
        floor_amp = level * comb_ref * fk
        return float(np.mean((a > floor_amp)[in_band]))

    level = 10.0 ** (floor_rel_db / 20.0)
    # If too few harmonics clear the floor, lower the floor until >= target.
    guard = 0
    while frac_above(level) < min_harm_above_floor and guard < 60:
        level *= 10.0 ** (-1.0 / 20.0)  # -1 dB
        guard += 1
    frac = frac_above(level)

    return RotorProfile(
        a_k=a.astype(np.float32),
        floor_tilt=float(tilt),
        floor_level=float(level * comb_ref),
        frac_above_floor=frac,
    )


# ── Synthesis ───────────────────────────────────────────────────────────────


def _comb_waveform(
    rps: np.ndarray,
    a_k: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    band_taper_frac: float = 0.0,
) -> np.ndarray:
    """Additive harmonic comb for one rotor: sum_k a_k sin(k*phase + phi_k),
    with ``phase`` the integral of ``2*pi*rps``.

    ``band_taper_frac`` controls what happens at the top of the band. At 0.0 a
    harmonic is zeroed the instant it crosses Nyquist — a brick wall, so the
    comb ends at ``K * rps`` whenever that is in band, giving every frame slower
    than ``nyquist / K`` a cutoff frequency exactly proportional to the rotor
    speed a model is asked to predict. Measured on the built stochastic stream,
    that left a +1.84 dB step at the cutoff with the spectrum's own tilt
    differenced out, against +0.50 dB in real DREGON audio, and 100% of ramp
    frames carried one.

    Above 0.0 the amplitudes instead fade to zero across the top
    ``band_taper_frac`` of the band on a raised cosine, so the comb dissolves
    into the floor rather than stopping dead. Pair it with a ``K`` large enough
    that ``K * rps`` clears Nyquist at the speeds of interest, or the brick wall
    simply returns below the taper.
    """
    T = rps.shape[-1]
    phase = 2.0 * np.pi * np.cumsum(rps) / sample_rate  # fundamental phase (T,)
    K = a_k.shape[0]
    out = np.zeros(T, dtype=np.float64)
    nyq = sample_rate / 2.0
    taper = float(np.clip(band_taper_frac, 0.0, 1.0))
    f_lo = nyq * (1.0 - taper)

    # Harmonic k is exp(i*k*phase), so each order is the previous one times
    # exp(i*phase) — one complex multiply instead of a fresh transcendental per
    # order. Covering the band at low RPM needs a few hundred orders, and at 300
    # orders the sine-per-order form costs about 3.5x what the recurrence does.
    # Relative drift after K multiplies is about K*eps, so at K in the hundreds
    # it stays far below the quantization floor.
    step = np.exp(1j * phase)
    cur = np.ones(T, dtype=np.complex128)
    rps_min, rps_max = float(np.min(rps)), float(np.max(rps))
    for i in range(K):
        k = i + 1
        # This order and every higher one stay above Nyquist: nothing is left to
        # add, and skipping them changes neither the output nor the rng stream.
        if k * rps_min >= nyq:
            break
        cur *= step  # now exp(i*(i+1)*phase)
        if taper > 0.0:
            # Three cases, cheapest first. An order that stays under the taper's
            # onset for the whole window has a flat amplitude and needs no
            # per-sample work at all; one that stays over Nyquist is silent.
            # Only an order that actually straddles the taper pays for a cosine
            # across the window, and in a slow window that is a small minority.
            if k * rps_max <= f_lo:
                amp = a_k[i]
            elif k * rps_min >= nyq:
                continue
            else:
                x = np.clip((k * rps - f_lo) / max(nyq - f_lo, 1e-9), 0.0, 1.0)
                amp = a_k[i] * 0.5 * (1.0 + np.cos(np.pi * x))
        else:
            amp = np.where(k * rps < nyq, a_k[i], 0.0)
            if not np.any(amp):
                continue
        phi = rng.uniform(0.0, 2.0 * np.pi)
        out += amp * (cur.real * np.sin(phi) + cur.imag * np.cos(phi))
    return out


def _floor_waveform(
    n: int, tilt: float, level: float, sample_rate: int, rng: np.random.Generator
) -> np.ndarray:
    """Broadband floor: white noise shaped to PSD ~ f**(-tilt), scaled so its
    RMS ~= ``level`` (matched to the comb reference in ``sample_profile``)."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
    shape = np.ones_like(freqs)
    nz = freqs > 0
    shape[nz] = freqs[nz] ** (-tilt / 2.0)
    shape[~nz] = 0.0  # drop DC
    shaped = np.fft.irfft(spec * shape, n=n)
    rms = float(np.sqrt(np.mean(shaped**2))) or 1.0
    return (shaped / rms * level).astype(np.float64)


# ── Pool ────────────────────────────────────────────────────────────────────


class StaticCombNoisePool:
    """Analytic static-comb + broadband-floor noise source (``kind:
    static_comb``). Same ``sample_timeframe`` interface as the other pools."""

    #: ``meta.recording_id`` of every rendered window.
    RECORDING_ID = "static_comb"

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        duration_s: float = 1.0,
        n_harmonics: int = 100,
        n_mics: int = 8,
        n_rotors: int = 4,
        min_harm_above_floor: float = 0.30,
        band_taper_frac: float = 0.0,
        aggressiveness: float = 1.0,
        amp_rps_exponent: float = 2.5,
        amp_rps_ref: float = 80.0,
        rps_kind: str = "synthetic_intermittent",
        flight_fs: float = 200.0,
        flight_reuse: int = 32,
        fitted_traj: Any = None,  # the whole ``rps`` block, mapping or OmegaConf node
        drone_profile_range: tuple[float, float] = (0.0, 1.0),
        mic_gain_db: tuple[float, float] = (-12.0, 0.0),
        rps_scale_range: tuple[float, float] = (1.0, 1.0),
        normalize_rms: float | tuple[float, float] = 0.1,
        render_reuse: int = 1,
        render_pool: int = 0,
        render_duration_s: float | None = None,
        ranges: ProfileRanges | dict[str, Any] | None = None,
        seed: int = 0,
    ):
        self.sample_rate = int(sample_rate)
        self.chunk_s = float(duration_s)
        self.n_harmonics = int(n_harmonics)
        self.n_mics = int(n_mics)
        self.n_rotors = int(n_rotors)
        self.min_harm_above_floor = float(min_harm_above_floor)
        self.band_taper_frac = float(band_taper_frac)
        self.aggressiveness = float(aggressiveness)
        # Physically-plausible amplitude scaling: rotor aeroacoustic sound power
        # ~ tip-speed^~5 => pressure amplitude ~ rps^~2.5; zero rps => silence.
        self.amp_rps_exponent = float(amp_rps_exponent)
        self.amp_rps_ref = float(amp_rps_ref)
        # RPS excitation: "synthetic_intermittent" (cruise-only, per-window) or
        # "full_flight" (ground->warm-up->takeoff->cruise->landing->ground; a low-
        # rate flight is generated once per `flight_reuse` windows and windowed, so
        # windows visit the low-/zero-RPS regimes). `flight_fs` is the internal
        # trajectory rate (RPS is slow; upsampled per window).
        self.rps_kind = str(rps_kind)
        self.flight_fs = float(flight_fs)
        self.flight_reuse = int(flight_reuse)
        # "fitted_traj" windows a full flight from the FITTED trajectory model
        # instead of the scaffold; the block is resolved here so a bad policy
        # fails when the pool is built (see trajectory_model.source).
        self._traj = (
            trajectory_model.build_from_config(fitted_traj or {})
            if self.rps_kind == trajectory_model.FITTED_KIND
            else None
        )
        self._flight: dict[str, Any] | None = None  # cached full-flight state
        self._flight_uses = 0
        self.drone_profile_range: tuple[float, float] = (
            float(drone_profile_range[0]),
            float(drone_profile_range[1]),
        )
        self.mic_gain_db: tuple[float, float] = (float(mic_gain_db[0]), float(mic_gain_db[1]))
        # Per-window multiplier on the whole trajectory. The audio is rendered
        # FROM the labels, so this moves every comb line with its own label and
        # leaves the floor where it is. Its purpose is to spread the speed prior
        # (docs/experiments/stochastic-transfer.md).
        self.rps_scale_range: tuple[float, float] = (
            float(rps_scale_range[0]),
            float(rps_scale_range[1]),
        )
        # Output level. A pair is a log-uniform range drawn per window. A fixed
        # level is what leaves a synthetic-only model level-fragile: it peaks at
        # the one level it trained at and collapses far below it, while a
        # real-trained model is flat across a hundredfold range.
        self.normalize_rms: float | tuple[float, float] = (
            (float(normalize_rms[0]), float(normalize_rms[1]))
            if isinstance(normalize_rms, (list, tuple))
            else float(normalize_rms)
        )
        # Render reuse, the noise_v2 pool's (NoiseV2Pool._pooled_render): one
        # slot of a rolling pool of ``render_pool`` renders (4 x render_reuse
        # by default) is refreshed every ``render_reuse`` draws; a slot holds
        # ``render_duration_s`` of audio (the window when unset) and every draw
        # cuts its own uniformly placed window out of it, rotor track alike.
        # ``render_reuse: 1`` without a duration renders every window afresh.
        self.render_reuse = max(int(render_reuse), 1)
        self.render_pool = int(render_pool) if render_pool else 4 * self.render_reuse
        self.render_duration_s = None if render_duration_s is None else float(render_duration_s)
        self._pool: list[tuple[float, np.ndarray, np.ndarray]] = []
        self._pool_draws = 0
        self.ranges = (
            ranges if isinstance(ranges, ProfileRanges) else ProfileRanges.from_dict(ranges)
        )
        self._base_seed = int(seed)
        # Placeholder geometry (analytic model: mic/rotor positions are not used
        # for synthesis, only carried in frame meta for interface parity).
        self.mic_pos = np.zeros((self.n_mics, 3), dtype=np.float64)
        self.rotor_pos = np.zeros((self.n_rotors, 3), dtype=np.float64)

    @classmethod
    def _config_kwargs(cls, cfg: Any, *, duration_s: float, sample_rate: int) -> dict[str, Any]:
        """The constructor keywords one ``sources.noise`` entry names."""

        def g(key, default=None):
            return _cfg_value(cfg, key, default)

        rps = g("rps", {}) or {}
        ranges = g("profile_ranges")
        if ranges is not None and not isinstance(ranges, dict):
            from data_processing.generated_noise import _to_plain

            ranges = _to_plain(ranges)

        def _pair(key: str, default: tuple[float, float]) -> tuple[float, float]:
            v = g(key, default)
            return (float(v[0]), float(v[1]))

        return dict(
            sample_rate=sample_rate,
            duration_s=duration_s,
            n_harmonics=int(g("n_harmonics", 100)),
            band_taper_frac=float(g("band_taper_frac", 0.0)),
            n_mics=int(g("n_mics", 8)),
            n_rotors=int(g("n_rotors", 4)),
            min_harm_above_floor=float(g("min_harm_above_floor", 0.30)),
            aggressiveness=float(rps.get("aggressiveness", 1.0)),
            amp_rps_exponent=float(g("amp_rps_exponent", 2.5)),
            amp_rps_ref=float(g("amp_rps_ref", 80.0)),
            rps_kind=str(rps.get("kind", "synthetic_intermittent")),
            flight_fs=float(rps.get("flight_fs", 200.0)),
            flight_reuse=int(rps.get("flight_reuse", 32)),
            fitted_traj=rps,
            drone_profile_range=_pair("drone_profile_range", (0.0, 1.0)),
            mic_gain_db=_pair("mic_gain_db", (-12.0, 0.0)),
            rps_scale_range=_pair("rps_scale_range", (1.0, 1.0)),
            normalize_rms=(
                _pair("normalize_rms_range", (0.0, 0.0))
                if g("normalize_rms_range") is not None
                else float(g("normalize_rms", 0.1))
            ),
            render_reuse=int(g("render_reuse", 1)),
            render_pool=int(g("render_pool", 0)),
            render_duration_s=(
                None if g("render_duration_s") is None else float(g("render_duration_s"))
            ),
            ranges=ranges,
            seed=int(g("seed", 0)),
        )

    @classmethod
    def from_config(cls, cfg: Any, *, duration_s: float, sample_rate: int) -> StaticCombNoisePool:
        return cls(**cls._config_kwargs(cfg, duration_s=duration_s, sample_rate=sample_rate))

    def close(self) -> None:  # interface parity with GeneratedNoisePool
        return None

    def _sample_rps_window(self, rng: np.random.Generator, T: int, duration_s: float) -> np.ndarray:
        """Return one ``(R, T)`` rps window at audio rate.

        ``synthetic_intermittent`` generates a fresh cruise-only window directly;
        ``full_flight`` (the scaffold) and ``fitted_traj`` (the fitted model)
        window a cached low-rate full flight (regenerated every
        ``flight_reuse`` calls), so successive windows visit warm-up / takeoff /
        cruise / landing / ground (zero RPS) in proportion to their durations.
        """
        scale = float(rng.uniform(*self.rps_scale_range))
        if self.rps_kind not in trajectory_model.FLIGHT_KINDS:
            blend = float(rng.uniform(*self.drone_profile_range))
            return (
                scale
                * rps_synthesis.generate_intermittent_batch(
                    1,
                    duration_s,
                    self.sample_rate,
                    drone_profile=blend,
                    aggressiveness=self.aggressiveness,
                    rng=rng,
                )[0]
            )

        if self._flight is None or self._flight_uses >= self.flight_reuse:
            if self._traj is not None:
                flight = self._traj.flight(rng, self.flight_fs)  # (R, Nlow)
            else:
                blend = float(rng.uniform(*self.drone_profile_range))
                # low-rate trajectory (RPS is slow; the per-sample motor low-pass is a
                # python loop, so audio-rate over a whole flight would be too slow).
                flight = rps_synthesis.generate_full_flight(
                    None,
                    self.flight_fs,
                    drone_profile=blend,
                    aggressiveness=self.aggressiveness,
                    rng=rng,
                )  # (R, Nlow)
            self._flight = {"rps": flight, "t_low": np.arange(flight.shape[1]) / self.flight_fs}
            self._flight_uses = 0
        self._flight_uses += 1
        flight = self._flight["rps"]
        t_low = self._flight["t_low"]
        total_s = float(t_low[-1])
        max_start = max(0.0, total_s - duration_s)
        start_s = float(rng.uniform(0.0, max_start)) if max_start > 0 else 0.0
        t_win = start_s + np.arange(T) / self.sample_rate
        window = np.stack([np.interp(t_win, t_low, flight[r]) for r in range(flight.shape[0])])
        return scale * window

    def render(
        self, rng: np.random.Generator, duration_s: float
    ) -> tuple[np.ndarray, np.ndarray, list[Any]]:
        """Render (audio (M,T), rps (R,T), per-rotor profiles) — factored out of
        ``sample_timeframe`` so diagnostics can inspect the profiles."""
        T = int(round(duration_s * self.sample_rate))
        rps = self._sample_rps_window(rng, T, duration_s)  # (R, T)
        R = rps.shape[0]

        combs = np.empty((R, T), dtype=np.float64)
        floors = np.empty((R, T), dtype=np.float64)
        profiles: list[RotorProfile] = []
        for r in range(R):
            # Floor in-band constraint is defined at (near-)hover, so low/zero-RPS
            # windows still get a sensible profile.
            ref_rps = max(float(np.median(rps[r])), 0.25 * self.amp_rps_ref)
            prof = sample_profile(
                rng,
                self.ranges,
                n_harmonics=self.n_harmonics,
                ref_rps=ref_rps,
                sample_rate=self.sample_rate,
                min_harm_above_floor=self.min_harm_above_floor,
            )
            profiles.append(prof)
            combs[r] = _comb_waveform(
                rps[r], prof.a_k, self.sample_rate, rng, band_taper_frac=self.band_taper_frac
            )
            floors[r] = _floor_waveform(T, prof.floor_tilt, prof.floor_level, self.sample_rate, rng)

        # Physically-plausible amplitude: scale each rotor's whole contribution
        # (comb AND floor) by (rps/ref)^p, so level rises with rps and is exactly
        # zero at zero rps (rotor off => silence) — the cue is monotonic and
        # consistent between synthetic and real, unlike the neural gen's arbitrary
        # amplitude dynamics. Within a steady (cruise) clip rps ~ const, so this is
        # ~flat and carries little RPS info; it only bites in the transition/ground
        # windows a full-flight trajectory now visits.
        amp = (np.maximum(rps, 0.0) / self.amp_rps_ref) ** self.amp_rps_exponent  # (R, T)

        # Per-mic linear mix of rotor combs+floors with sampled per-(mic,rotor) gains.
        lo, hi = self.mic_gain_db
        gains = 10.0 ** (rng.uniform(lo, hi, size=(self.n_mics, R)) / 20.0)
        audio = np.empty((self.n_mics, T), dtype=np.float32)
        per_rotor = (combs + floors) * amp  # (R, T)
        for m in range(self.n_mics):
            audio[m] = (gains[m][:, None] * per_rotor).sum(axis=0).astype(np.float32)
        audio = self._normalized(audio, rng)
        return audio, rps.astype(np.float32), profiles

    def _normalized(self, audio: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """``audio`` scaled to the clip's RMS target (log-uniform if a range)."""
        if isinstance(self.normalize_rms, tuple):
            lo, hi = self.normalize_rms
            level = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            level = float(self.normalize_rms)
        rms = float(np.sqrt(np.mean(audio**2))) or 1.0
        return (audio / rms * level).astype(np.float32)

    def _pooled_render(
        self, rng: np.random.Generator, duration_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(audio, rps)`` of one ``duration_s`` window, from the rolling
        render pool when reuse is on (the constructor's ``render_reuse``)."""
        span = max(float(duration_s), self.render_duration_s or 0.0)
        if self.render_reuse <= 1 and span == float(duration_s):
            audio, rps, _ = self.render(rng, duration_s)
            return audio, rps
        matching = [item for item in self._pool if item[0] == span]
        if not matching or self._pool_draws % self.render_reuse == 0:
            audio, rps, _ = self.render(rng, span)
            slot = (span, audio, rps)
            if len(self._pool) < self.render_pool:
                self._pool.append(slot)
            else:
                self._pool[int(rng.integers(len(self._pool)))] = slot
            matching = [item for item in self._pool if item[0] == span]
        self._pool_draws += 1
        _, audio, rps = matching[int(rng.integers(len(matching)))]
        n = int(round(float(duration_s) * self.sample_rate))
        if audio.shape[-1] > n:
            start = int(rng.integers(audio.shape[-1] - n + 1))
            audio, rps = audio[:, start : start + n], rps[:, start : start + n]
        return audio, rps

    def sample_timeframe(self, rng: np.random.Generator, duration_s: float) -> td.Frame:
        audio, rps = self._pooled_render(rng, duration_s)
        audio_us = td.uniform(
            np.ascontiguousarray(audio), self.sample_rate, dims=("mic", "time"), t_start=0.0
        )
        t = np.arange(audio.shape[-1], dtype=np.float64) / self.sample_rate
        rps_es = td.events(t, np.ascontiguousarray(rps), dims=("rotor", "time"), t_start=0.0)
        return make_recording_frame(
            {"audio": audio_us, "rps": rps_es},
            meta={"recording_id": self.RECORDING_ID},
            mic_pos=self.mic_pos,
            rotor_pos=self.rotor_pos,
        )


# ── Fixed-profile comb (the generator label-sensitivity probe) ──────────────


@dataclass(frozen=True)
class FixedCombSpec:
    """One comb profile, **frozen** across the whole dataset.

    :class:`StaticCombNoisePool` draws a fresh profile per clip, which is right
    for an RPS predictor (it must not memorize an envelope) and wrong for a
    *generator* probe: a per-clip random profile is unpredictable from the RPS
    conditioning, so it adds irreducible loss that would mask the per-harmonic
    effect being measured. Freezing the profile makes the target a
    deterministic function of the trajectory alone — every remaining
    per-harmonic amplitude deficit is then attributable to the objective or to
    the conditioning, which is exactly the question.

    The amplitude is also **RPS-independent** (no ``(rps/ref)^p`` scaling, no
    per-clip RMS normalization), so the only thing time-variation in the RPS
    does to the target is move the comb's lines in frequency.
    """

    n_harmonics: int = 80  # k*rps stays below Nyquist for rps <= 100 rev/s
    rolloff_p: float = 1.0  # a_k ~ k**(-p)
    blade_count: int = 2  # blade-pass emphasis on every b-th harmonic
    blade_emphasis_db: float = 6.0
    harm_jitter_db: float = 4.0  # frozen per-harmonic texture
    floor_tilt: float = 1.0
    floor_rel_db: float = -18.0  # broadband floor, well below every line
    profile_seed: int = 20260806
    target_rms: float = 0.1

    def profile(self, *, ref_rps: float, sample_rate: int) -> RotorProfile:
        """The frozen :class:`RotorProfile` this spec names."""
        ranges = ProfileRanges(
            rolloff_p=(self.rolloff_p, self.rolloff_p),
            harm_jitter_db=(self.harm_jitter_db, self.harm_jitter_db),
            blade_counts=(self.blade_count,),
            blade_emphasis_db=(self.blade_emphasis_db, self.blade_emphasis_db),
            floor_tilt=(self.floor_tilt, self.floor_tilt),
            floor_rel_db=(self.floor_rel_db, self.floor_rel_db),
        )
        return sample_profile(
            np.random.default_rng(self.profile_seed),
            ranges,
            n_harmonics=self.n_harmonics,
            ref_rps=ref_rps,
            sample_rate=sample_rate,
            min_harm_above_floor=0.0,  # the floor level is pinned, not searched
        )

    def gain(self, profile: RotorProfile) -> float:
        """Clip-independent gain putting the comb at ``target_rms``.

        Derived from the profile (``rms = sqrt(sum a_k^2 / 2)``) rather than
        measured per clip, so the level never leaks information about the
        window's trajectory.
        """
        comb_rms = float(np.sqrt(np.sum(np.asarray(profile.a_k, dtype=np.float64) ** 2) / 2.0))
        return float(self.target_rms / (comb_rms or 1.0))


def render_fixed_comb(
    rps: np.ndarray,
    profile: RotorProfile,
    *,
    sample_rate: int,
    gain: float,
    rng: np.random.Generator,
    with_floor: bool = True,
) -> np.ndarray:
    """Render one rotor's frozen-profile comb (+ floor) from a ``(T,)`` track.

    Harmonic phases are drawn per call (they are unobservable to a magnitude
    objective) but the amplitudes are the spec's, so two calls on the same
    trajectory differ only in phase.
    """
    rps = np.asarray(rps, dtype=np.float64)
    out = _comb_waveform(rps, np.asarray(profile.a_k, dtype=np.float64), sample_rate, rng)
    if with_floor:
        out = out + _floor_waveform(
            rps.shape[-1], profile.floor_tilt, profile.floor_level, sample_rate, rng
        )
    return (out * float(gain)).astype(np.float64)


# ── The diverse static comb (``kind: diverse_comb``) ────────────────────────
#
# The static comb above draws a profile from one rolloff, a Gaussian texture
# and a blade emphasis, over a per-rotor pink floor kept under 70 % of the
# lines. This family keeps the comb static within a clip and makes the
# population wide: separate power laws for the even (blade-passing) and odd
# (imbalance) orders, the odd one always under the even one; heavy-tailed
# per-line departures, so a line is occasionally far above its law; random line
# drop-outs; random rotor levels; and ONE white floor per microphone whose level
# is drawn as the share of a rotor's lines it buries.


@dataclass
class DiverseCombRanges:
    """Per-clip sampling ranges of :class:`DiverseCombNoisePool`, all uniform.

    Levels are line POWER in dB (``20 log10`` of a line amplitude). The
    defaults widen the measured rig profiles
    (``docs/experiments/static-rig-profiles.md``: even-order slope 25 +/- 5
    dB/decade, odd orders 8-9 +/- 9 dB under the even law, per-line scatter
    3.5-5.6 dB, rotor levels 2.3-5.2 dB apart).
    """

    #: Even-order law ``-even_slope_db * log10(k)``.
    even_slope_db: tuple[float, float] = (10.0, 30.0)
    #: The odd-order law sits this far under the even law at ``k = 1`` ...
    odd_penalty_db: tuple[float, float] = (3.0, 20.0)
    #: ... and falls this much faster (dB/decade), so it is under it everywhere.
    odd_extra_slope_db: tuple[float, float] = (0.0, 5.0)
    #: Scale (dB) of the per-line Student-t departure from the law.
    line_scatter_db: tuple[float, float] = (2.0, 5.0)
    line_scatter_dof: float = 3.0
    #: Departures are clipped to +/- this (dB).
    line_scatter_max_db: float = 30.0
    #: Probability that a line is dropped, drawn per rotor.
    drop_prob: tuple[float, float] = (0.0, 0.3)
    #: Level of each rotor's whole comb (dB), drawn per rotor.
    rotor_gain_db: tuple[float, float] = (-4.0, 4.0)
    #: Share of its counted lines the white floor buries on the MOST buried
    #: rotor of a microphone (:func:`white_floor_std`).
    floor_subsumed: tuple[float, float] = (0.0, 0.9)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> DiverseCombRanges:
        if not d:
            return cls()
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown comb_ranges key(s) {unknown}; known: {sorted(known)}")
        kwargs: dict[str, Any] = {
            k: tuple(v) if isinstance(v, (list, tuple)) else v for k, v in d.items()
        }
        return cls(**kwargs)


@dataclass
class DiverseRotorProfile:
    """One rotor's static comb for one clip, and the draws that made it."""

    level_db: np.ndarray  # (K,) line power (dB) incl. the rotor gain; dropped lines too
    kept: np.ndarray  # (K,) False where the line is dropped
    even_slope_db: float
    odd_penalty_db: float
    odd_slope_db: float
    scatter_db: float
    drop_prob: float
    gain_db: float

    @property
    def amplitudes(self) -> np.ndarray:
        """``(K,)`` line amplitudes, zero where dropped."""
        return np.where(self.kept, 10.0 ** (self.level_db / 20.0), 0.0)


def sample_diverse_rotor(
    rng: np.random.Generator, ranges: DiverseCombRanges, *, n_harmonics: int
) -> DiverseRotorProfile:
    """Draw one rotor's static comb: two power laws, Student-t departures,
    drop-outs and a level."""
    k = np.arange(1, int(n_harmonics) + 1, dtype=np.float64)
    lk = np.log10(k)
    even_slope = float(rng.uniform(*ranges.even_slope_db))
    penalty = float(rng.uniform(*ranges.odd_penalty_db))
    odd_slope = even_slope + float(rng.uniform(*ranges.odd_extra_slope_db))
    law = np.where(k % 2 == 0, -even_slope * lk, -penalty - odd_slope * lk)
    scatter = float(rng.uniform(*ranges.line_scatter_db))
    cap = float(ranges.line_scatter_max_db)
    departure = np.clip(scatter * rng.standard_t(ranges.line_scatter_dof, size=k.size), -cap, cap)
    drop = float(rng.uniform(*ranges.drop_prob))
    kept = rng.random(k.size) >= drop
    gain = float(rng.uniform(*ranges.rotor_gain_db))
    return DiverseRotorProfile(
        level_db=law + departure + gain,
        kept=kept,
        even_slope_db=even_slope,
        odd_penalty_db=penalty,
        odd_slope_db=odd_slope,
        scatter_db=scatter,
        drop_prob=drop,
        gain_db=gain,
    )


def white_floor_std(
    line_level: list[np.ndarray] | tuple[np.ndarray, ...], subsumed: float
) -> tuple[float, np.ndarray]:
    """``(sigma, buried)``: the white floor's std on one microphone and the
    share of each rotor's counted lines it buries.

    ``line_level[r]`` holds rotor ``r``'s counted lines at this microphone on
    the floor's scale: a line is BURIED when its value is under ``sigma``
    (:func:`render_diverse_comb` passes the line's power per spectrogram bin,
    as an RMS). ``sigma`` is the largest value at which no rotor loses more
    than ``subsumed`` of its lines, so the most buried rotor loses
    ``floor(subsumed * n) / n`` of them. A rotor with no counted line does not
    constrain it.
    """
    bounds = []
    for v in line_level:
        if v.size:
            ordered = np.sort(v)
            bounds.append(float(ordered[min(int(np.floor(subsumed * v.size)), v.size - 1)]))
    sigma = min(bounds) if bounds else 0.0
    buried = np.array([float(np.mean(v < sigma)) if v.size else 0.0 for v in line_level])
    return sigma, buried


def render_diverse_comb(
    rps: np.ndarray,
    profiles: list[DiverseRotorProfile],
    mic_gains: np.ndarray,
    subsumed: float,
    rng: np.random.Generator,
    *,
    sample_rate: int,
    amp_rps_exponent: float,
    amp_rps_ref: float,
    band_taper_frac: float,
    floor_ref_nfft: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(audio (M, T) float32, sigma (M,), buried (M, R))`` from ``(R, T)`` rev/s.

    Rotor ``r`` reaches microphone ``m`` with amplitude gain
    ``mic_gains[m, r]`` and its whole comb scales with
    ``(rps / amp_rps_ref) ** amp_rps_exponent`` (silent when stopped). Each
    microphone adds independent white noise at :func:`white_floor_std` of the
    lines it hears, read at each rotor's window-median speed (at least a
    quarter of ``amp_rps_ref``, so a stopped window still gets a floor); the
    counted lines are the kept ones below Nyquist at that speed.

    The lines are the noise-model renderer's (:mod:`.noise_model.render`): one
    complex envelope per line, ``a_k e^{i psi_k}`` (``psi_k`` uniform) times
    the speed law and the band-top taper, on its 1 kHz envelope grid, carried
    by ``cos / sin(k phi)`` of the rotor's phase at the output rate. The
    static comb is that model with every per-line wander, width and per-mic
    phase set to zero. Orders that are dropped, or over Nyquist for the whole
    window, are not rendered.

    "Buried" is read in the spectrogram: line ``a cos(...)`` stands at
    ``(a^2 / 2) * N / 3`` in a ``N = floor_ref_nfft`` Hann periodogram
    normalised by ``sum(w^2)``, where white noise of variance ``sigma^2``
    stands at ``sigma^2``; equivalently the line's power is compared with the
    floor's power inside one Hann bin (ENBW ``1.5 fs / N``). Against the
    floor's TOTAL variance instead, a "buried" line would still stand
    ``10 log10(N / 3)`` dB (28 dB at N = 2048) over the floor on screen.
    """
    rps = np.asarray(rps, dtype=np.float64)
    n_rotors, n = rps.shape
    nyq = sample_rate / 2.0
    f_lo = nyq * (1.0 - float(np.clip(band_taper_frac, 0.0, 1.0)))
    ref = np.maximum(np.median(rps, axis=1), 0.25 * amp_rps_ref)
    speed_gain = (ref / amp_rps_ref) ** amp_rps_exponent
    in_bin = np.sqrt(float(floor_ref_nfft) / 3.0)
    hop_env = sample_rate // ENV_RATE_HZ
    if hop_env * ENV_RATE_HZ != sample_rate:
        raise ValueError(f"sample_rate {sample_rate} is not a multiple of {ENV_RATE_HZ} Hz")
    knots = np.minimum(np.arange(n // hop_env + 2) * hop_env, n - 1)
    n_mics = mic_gains.shape[0]
    audio = np.zeros((n_mics, n), dtype=np.float32)
    counted: list[np.ndarray] = []
    for r, prof in enumerate(profiles):
        a = prof.amplitudes
        k = np.arange(1, a.size + 1)
        counted.append(a[prof.kept & (k * ref[r] < nyq)] * speed_gain[r] * in_bin / np.sqrt(2.0))
        live = (a > 0.0) & (k * rps[r].min() < nyq)
        k, a = k[live], a[live]
        rk = rps[r, knots]
        law = (np.maximum(rk, 0.0) / amp_rps_ref) ** amp_rps_exponent
        env = (a * np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=k.size)))[:, None] * law
        # raised-cosine fade over the top band_taper_frac of the band (a brick
        # wall at Nyquist when 0), only on the orders that ever reach it
        top = k * rps[r].max() > f_lo
        if top.any():
            x = np.clip((k[top, None] * rk - f_lo) / max(nyq - f_lo, 1e-9), 0.0, 1.0)
            env[top] *= 0.5 * (1.0 + np.cos(np.pi * x))
        phase = 2.0 * np.pi * np.cumsum(rps[r]) / sample_rate
        _render_lines(
            audio,
            np.cos(phase).astype(np.float32),
            np.sin(phase).astype(np.float32),
            env,
            hop_env,
            np.repeat(mic_gains[:, r : r + 1], k.size, axis=1),
            k,
        )
    sigma = np.empty(n_mics)
    buried = np.empty((n_mics, n_rotors))
    for m in range(n_mics):
        sigma[m], buried[m] = white_floor_std(
            [c * mic_gains[m, r] for r, c in enumerate(counted)], subsumed
        )
        audio[m] += sigma[m] * rng.standard_normal(n, dtype=np.float32)
    return audio, sigma, buried


class DiverseCombNoisePool(StaticCombNoisePool):
    """``kind: diverse_comb``: the static comb with a wide, realistic-shaped
    profile population and one white floor per microphone
    (:func:`sample_diverse_rotor`, :func:`render_diverse_comb`).

    Everything else is the static comb's: the trajectory (``rps`` block, so
    ``fitted_traj`` and its flight cache), ``n_harmonics``,
    ``band_taper_frac``, the speed law, ``mic_gain_db`` (per microphone and
    rotor), ``rps_scale_range``, the output level and the render reuse.
    :attr:`last_render` keeps the draws behind the most recent render for
    diagnostics.
    """

    RECORDING_ID = "diverse_comb"

    def __init__(
        self,
        *,
        comb_ranges: DiverseCombRanges | dict[str, Any] | None = None,
        floor_ref_nfft: int = 2048,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.comb_ranges = (
            comb_ranges
            if isinstance(comb_ranges, DiverseCombRanges)
            else DiverseCombRanges.from_dict(comb_ranges)
        )
        #: The spectrogram resolution "buried" is read at
        #: (:func:`render_diverse_comb`): the campaign's 128 ms analysis window.
        self.floor_ref_nfft = int(floor_ref_nfft)
        self.last_render: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, cfg: Any, *, duration_s: float, sample_rate: int) -> DiverseCombNoisePool:
        for legacy in ("profile_ranges", "min_harm_above_floor"):
            if _cfg_value(cfg, legacy) is not None:
                raise ValueError(
                    f"kind: diverse_comb has no {legacy!r} (a kind: static_comb key); "
                    "its profile lives in comb_ranges"
                )
        kwargs = cls._config_kwargs(cfg, duration_s=duration_s, sample_rate=sample_rate)
        del kwargs["ranges"], kwargs["min_harm_above_floor"]
        ranges = _cfg_value(cfg, "comb_ranges")
        if ranges is not None and not isinstance(ranges, dict):
            from data_processing.generated_noise import _to_plain

            ranges = _to_plain(ranges)
        return cls(
            comb_ranges=ranges,
            floor_ref_nfft=int(_cfg_value(cfg, "floor_ref_nfft", 2048)),
            **kwargs,
        )

    def render(
        self, rng: np.random.Generator, duration_s: float
    ) -> tuple[np.ndarray, np.ndarray, list[Any]]:
        """``(audio (M, T), rps (R, T), per-rotor profiles)``."""
        n = int(round(duration_s * self.sample_rate))
        rps = self._sample_rps_window(rng, n, duration_s)
        n_rotors = rps.shape[0]
        profiles = [
            sample_diverse_rotor(rng, self.comb_ranges, n_harmonics=self.n_harmonics)
            for _ in range(n_rotors)
        ]
        lo, hi = self.mic_gain_db
        mic_gains = 10.0 ** (rng.uniform(lo, hi, size=(self.n_mics, n_rotors)) / 20.0)
        subsumed = float(rng.uniform(*self.comb_ranges.floor_subsumed))
        audio, sigma, buried = render_diverse_comb(
            rps,
            profiles,
            mic_gains,
            subsumed,
            rng,
            sample_rate=self.sample_rate,
            amp_rps_exponent=self.amp_rps_exponent,
            amp_rps_ref=self.amp_rps_ref,
            band_taper_frac=self.band_taper_frac,
            floor_ref_nfft=self.floor_ref_nfft,
        )
        self.last_render = {
            "profiles": profiles,
            "mic_gains": mic_gains,
            "subsumed": subsumed,
            "floor_std": sigma,
            "buried": buried,
        }
        return self._normalized(audio, rng), rps.astype(np.float32), profiles
