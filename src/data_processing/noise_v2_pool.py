"""Fitted rotor noise as an online-mixing source (``kind: noise_v2``).

The noise-model-v2 campaign fitted the two rigs of this project — DREGON's
single-motor bench and Michael's FLY125 in flight — to a generative model with
an ABSOLUTE level: a shared integrated-OU shaft per rotor, one free Lorentzian
half width per rotor and order, and a coloured floor with its own speed
envelope (``docs/explainers/noise-model-v2-plan.qmd``). This pool renders that
model on a synthetic flight trajectory, so a training stream can mix speech
against noise whose comb, linewidths, floor colour and speed laws are the
MEASURED ones rather than a hand-written family's.

Against ``kind: stochastic`` (:mod:`data_processing.stochastic_rotor_noise`),
which draws a fresh invented rig per window, this pool renders a FINITE set of
fitted rigs. That is the whole point — it is the transfer end of the
curriculum, not a variety generator — and it is also its limitation: two fits
is two rigs. A bank (below) is how that set grows.

WHAT THE POLICY LOOKS LIKE::

    kind: noise_v2
    weight: 1.0
    n_mics: 8
    n_rotors: 4
    fits:
      - name: dregon_room2_floor
        cruise: results/noise_v2/rounds/round5/fits/dregon_room2_floor__flight_profile.json
      - name: michaels_fly125
        cruise: results/noise_v2/rounds/round3/fits/michaels_fly125_cruise__flight.json
        standby: results/noise_v2/rounds/round3/fits/michaels_fly125_standby__flight.json
    rps:
      kind: fitted_traj          # or full_flight
      fits: dload:rps-traj-fits
      rigs: {michaels: 1.0}
      flight_fs: 200
      flight_reuse: 32
    rps_scale_range: [1.0, 1.0]
    render_reuse: 48
    comb_offset_db: 0.0

``fits`` and ``preset_bank`` are alternatives and exactly one is required.

THE ENTRY. One entry is one rig. ``cruise`` is its fit; ``standby`` is the
same rig's standby fit when it has one, and then the two are composed per
sample by :func:`~data_processing.noise_model.render.render_noise_regimes`
(smoothstep on the slowest rotor, 0 at 45 rev/s and 1 at 65, so the ramp
interpolates the two regimes' LEVELS instead of switching). Without a
``standby`` the single fit is rendered on the whole track by
:func:`~data_processing.noise_model.render.render_noise`. One entry is drawn
UNIFORMLY per rendered window.

THE PRESET BANK (``preset_bank: <path>.json``) is the same entry list with the
fit payloads INLINE instead of on disk, so a sampler can write hundreds of
rigs — a neighbourhood around the two fitted ones, say — into one file that a
policy names in one line::

    {
      "format": "noise-v2-bank/1",
      "entries": [
        {
          "name": "dregon_room2_floor",
          "cruise": {"schema": "noise-v2-fit/2", "params": {...}, ...},
          "standby": null,
          "traj_rig": "dregon",
          "provenance": {"source": "results/.../fits/....json", "note": "..."}
        }
      ],
      "provenance": {"builder": "...", "built_at": "...", "base_fits": [...]}
    }

``cruise`` and ``standby`` carry a COMPLETE fit payload each — whatever
``check_schema`` reads, i.e. ``{"schema": "noise-v2-fit/2", "params": {...}}``
plus whatever else the writer kept — so a bank is self-contained and a fit
that moves on disk does not silently change a bank's meaning. ``standby`` is
``null`` for a single-regime rig. ``traj_rig`` is advisory: the name of the
trajectory rig the entry was built around, for a later sampler that pairs a
noise rig with its own flight statistics; this pool does not read it (the
``rps`` block owns the trajectory). ``provenance`` is free-form on both the
bank and each entry, and nothing here validates it beyond requiring the
``format`` tag.

THE LEVEL. ``render_noise`` is ABSOLUTE: the rendered root-mean-square is the
fitted one (DREGON's round-5 fit lands near 0.03-0.05 at 8 mics, Michael's
cruise near 0.1), and by default this pool keeps it, because that absolute
level is the thing the fit measured. ``normalize_rms`` / ``normalize_rms_range``
and ``level_mode`` are the stochastic pool's keys with the stochastic pool's
semantics, for an arm that wants the base policy's level draw instead: a
scalar or a log-uniform range, applied per window (``level_mode: window``) or
as the level AT THE REFERENCE SPEED with the window's own speed envelope kept
(``level_mode: flight``).

``comb_offset_db`` adds a constant to every rotor's ``profile_db`` on a DEEP
COPY of each loaded fit, raising or lowering the comb against the floor
without touching the floor; the loaded payloads are never mutated. It is the
same knob ``notebooks/noise_v2_sampler.py`` exposes, and it exists because the
comb-to-floor ratio is the one thing the round-5 DREGON fit is known to be
uncertain about.

EVERYTHING IS VALIDATED WHEN THE POOL IS BUILT — the fit files are read, the
schemas checked, the rotor and microphone counts compared against the policy's
— so a bad policy fails where it is loaded and not in a DataLoader worker on
some later window.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

from data_processing import rps_synthesis, trajectory_model
from data_processing.frames import make_recording_frame
from data_processing.noise_model.constants import AMP_RPS_REF
from data_processing.noise_model.params import check_schema
from data_processing.noise_model.render import render_noise, render_noise_regimes
from data_processing.noise_model.spectrum import FLIGHT_SR, SAMPLE_RATE_WORK

#: The bank-file tag :func:`load_preset_bank` accepts, documented above.
PRESET_BANK_FORMAT = "noise-v2-bank/1"

__all__ = ["NoiseV2Pool", "NoiseV2Entry", "PRESET_BANK_FORMAT", "load_preset_bank"]


@dataclass(frozen=True)
class NoiseV2Entry:
    """One fitted rig: a cruise fit, optionally a standby fit of the same rig."""

    name: str
    cruise: dict[str, Any]
    standby: dict[str, Any] | None = None
    #: Advisory: the trajectory rig this entry was built around. The ``rps``
    #: block owns the trajectory, so nothing here reads it; a later sampler
    #: that pairs a noise rig with its own flight statistics will.
    traj_rig: str | None = None
    provenance: dict[str, Any] | None = None

    @property
    def per_regime(self) -> bool:
        return self.standby is not None


def _read_fit(path: str | Path, *, where: str) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"{where}: no such fit file: {p}")
    try:
        fit = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{where}: {p} is not readable JSON: {exc}") from exc
    if not isinstance(fit, dict):
        raise ValueError(f"{where}: {p} does not hold a fit object")
    return fit


def load_preset_bank(path: str | Path) -> tuple[NoiseV2Entry, ...]:
    """Every entry of a ``noise-v2-bank/1`` file, unvalidated against a pool.

    The payloads are returned as they were written; :class:`NoiseV2Pool`
    checks their schemas and shapes against its own ``n_mics`` / ``n_rotors``.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"preset_bank: no such bank file: {p}")
    bank = json.loads(p.read_text())
    fmt = bank.get("format") if isinstance(bank, dict) else None
    if fmt != PRESET_BANK_FORMAT:
        raise ValueError(f"{p}: expected format {PRESET_BANK_FORMAT!r}, got {fmt!r}")
    raw = bank.get("entries")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{p}: a bank needs a non-empty 'entries' list")
    out: list[NoiseV2Entry] = []
    for index, item in enumerate(raw):
        where = f"{p}: entry {index}"
        if not isinstance(item, dict):
            raise ValueError(f"{where}: not an object")
        cruise = item.get("cruise")
        if not isinstance(cruise, dict):
            raise ValueError(f"{where}: 'cruise' must carry a whole fit payload")
        standby = item.get("standby")
        if standby is not None and not isinstance(standby, dict):
            raise ValueError(f"{where}: 'standby' must be null or a whole fit payload")
        out.append(
            NoiseV2Entry(
                name=str(item.get("name", f"entry{index}")),
                cruise=cruise,
                standby=standby,
                traj_rig=None if item.get("traj_rig") is None else str(item["traj_rig"]),
                provenance=item.get("provenance"),
            )
        )
    return tuple(out)


def _offset_comb(fit: dict[str, Any], comb_offset_db: float) -> dict[str, Any]:
    """``fit`` with ``comb_offset_db`` on every rotor's ``profile_db``.

    A DEEP COPY: the loaded payload is never mutated, so the same file read by
    two arms at two offsets cannot leak one arm's comb into the other.
    """
    if not comb_offset_db:
        return fit
    out = copy.deepcopy(fit)
    prof = np.asarray(out["params"]["profile"]["profile_db"], dtype=np.float64)
    out["params"]["profile"]["profile_db"] = (prof + float(comb_offset_db)).tolist()
    return out


class NoiseV2Pool:
    """Fitted rotor-noise source (``kind: noise_v2``) — see the module docstring."""

    def __init__(
        self,
        *,
        sample_rate: int = FLIGHT_SR,
        duration_s: float = 1.0,
        n_mics: int = 8,
        n_rotors: int = 4,
        entries: tuple[NoiseV2Entry, ...] = (),
        rps_kind: str = "full_flight",
        flight_fs: float = 200.0,
        flight_reuse: int = 32,
        fitted_traj: Any = None,  # the whole ``rps`` block, mapping or OmegaConf node
        drone_profile_range: tuple[float, float] = (0.0, 1.0),
        aggressiveness: float | tuple[float, float] = 1.0,
        mode_scales: dict[str, float] | None = None,
        rotor_trim_rel: tuple[float, float] | None = None,
        flight_phases: dict[str, Any] | None = None,
        rps_scale_range: tuple[float, float] = (1.0, 1.0),
        render_reuse: int = 1,
        render_pool: int = 0,
        normalize_rms: float | tuple[float, float] | None = None,
        level_mode: str = "window",
        comb_offset_db: float = 0.0,
        sr_work: int = SAMPLE_RATE_WORK,
        seed: int = 0,
    ):
        self.sample_rate = int(sample_rate)
        self.chunk_s = float(duration_s)
        self.n_mics = int(n_mics)
        self.n_rotors = int(n_rotors)
        self.rps_kind = str(rps_kind)
        self.flight_fs = float(flight_fs)
        self.flight_reuse = int(flight_reuse)
        # Rendering the v2 model costs ~0.4 s of CPU per second of 8-mic audio,
        # far more than the stochastic family's filtered noise, so the rolling
        # render pool is not an optimisation here but the thing that makes the
        # source usable at all. Same mechanism and same keys as
        # StochasticNoisePool: refresh one slot every ``render_reuse`` draws,
        # take a random slot otherwise, so the render rate is 1 / render_reuse
        # from the first sample while the number of distinct clips in
        # circulation climbs to ``render_pool``. Speech, SNR and every
        # augmentation are still drawn per sample downstream.
        self.render_reuse = max(int(render_reuse), 1)
        self.render_pool = int(render_pool) if render_pool else 4 * self.render_reuse
        self._pool: list[tuple[float, np.ndarray, np.ndarray]] = []
        self._pool_draws = 0
        self.drone_profile_range = (float(drone_profile_range[0]), float(drone_profile_range[1]))
        self.aggressiveness: float | tuple[float, float] = (
            (float(aggressiveness[0]), float(aggressiveness[1]))
            if isinstance(aggressiveness, (list, tuple))
            else float(aggressiveness)
        )
        self.mode_scales = dict(mode_scales) if mode_scales else None
        self.rotor_trim_rel = (
            (float(rotor_trim_rel[0]), float(rotor_trim_rel[1])) if rotor_trim_rel else None
        )
        self.flight_phases = dict(flight_phases) if flight_phases else None
        self.rps_scale_range = (float(rps_scale_range[0]), float(rps_scale_range[1]))
        self.normalize_rms: float | tuple[float, float] | None = (
            None
            if normalize_rms is None
            else (
                (float(normalize_rms[0]), float(normalize_rms[1]))
                if isinstance(normalize_rms, (list, tuple))
                else float(normalize_rms)
            )
        )
        self.level_mode = str(level_mode)
        if self.level_mode not in ("window", "flight"):
            raise ValueError(f"level_mode must be 'window' or 'flight', got {self.level_mode!r}")
        self.comb_offset_db = float(comb_offset_db)
        self.sr_work = int(sr_work)
        self._base_seed = int(seed)
        if self.rps_kind not in trajectory_model.FLIGHT_KINDS:
            raise ValueError(
                f"noise_v2 renders on a whole flight; rps.kind must be one of "
                f"{list(trajectory_model.FLIGHT_KINDS)}, got {self.rps_kind!r}"
            )
        # ``rps.kind: fitted_traj``: resolve the fitted-model source here so a
        # bad policy fails when the pool is built and not in a DataLoader
        # worker on some later window.
        self._traj = (
            trajectory_model.build_from_config(fitted_traj or {})
            if self.rps_kind == trajectory_model.FITTED_KIND
            else None
        )
        self._flight: trajectory_model.FlightCache | None = None
        self.entries = tuple(entries)
        if not self.entries:
            raise ValueError("noise_v2 needs at least one entry: give it 'fits' or 'preset_bank'")
        self.entries = tuple(
            NoiseV2Entry(
                name=e.name,
                cruise=_offset_comb(e.cruise, self.comb_offset_db),
                standby=(
                    None if e.standby is None else _offset_comb(e.standby, self.comb_offset_db)
                ),
                traj_rig=e.traj_rig,
                provenance=e.provenance,
            )
            for e in self.entries
        )
        self._check_entries()
        # Interface parity with the other pools: the fitted model has no
        # geometry, and the frame carries placeholders.
        self.mic_pos = np.zeros((self.n_mics, 3), dtype=np.float64)
        self.rotor_pos = np.zeros((self.n_rotors, 3), dtype=np.float64)

    # ── validation ──────────────────────────────────────────────────────────

    def _check_entries(self) -> None:
        """Refuse a fit this pool cannot render, naming the mismatch.

        Everything checked here would otherwise surface as a ``ValueError``
        from inside :func:`render_noise`, in a DataLoader worker, on some later
        window — or, for a ``/1`` payload with no ``gamma_hz``, as a ``KeyError``
        three frames deeper.
        """
        for entry in self.entries:
            for regime, fit in (("cruise", entry.cruise), ("standby", entry.standby)):
                if fit is None:
                    continue
                where = f"noise_v2 entry {entry.name!r} ({regime})"
                try:
                    params = check_schema(fit)
                except (ValueError, KeyError) as exc:
                    raise ValueError(f"{where}: {exc}") from exc
                if not isinstance(params, dict):
                    raise ValueError(f"{where}: 'params' is not an object")
                self._check_params(params, where=where)

    def _check_params(self, params: dict[str, Any], *, where: str) -> None:
        for key in ("sigma_nu", "lam", "profile", "floor", "mic_gains_db"):
            if key not in params:
                raise ValueError(f"{where}: the fit carries no {key!r}")
        prof = np.atleast_2d(np.asarray(params["profile"]["profile_db"], dtype=np.float64))
        if prof.shape[0] not in (1, self.n_rotors):
            raise ValueError(
                f"{where}: the fit carries {prof.shape[0]} rotor profiles, which is neither 1 "
                f"(broadcast) nor this pool's n_rotors={self.n_rotors}"
            )
        for name, arr in (
            ("profile.mic_line_gain_db", params["profile"]["mic_line_gain_db"]),
            ("floor.mic_floor_db", params["floor"]["mic_floor_db"]),
            ("mic_gains_db", params["mic_gains_db"]),
        ):
            got = int(np.asarray(arr, dtype=np.float64).shape[0])
            if got < self.n_mics:
                raise ValueError(
                    f"{where}: the fit carries {got} microphones in {name}, "
                    f"this pool renders n_mics={self.n_mics}"
                )

    # ── construction from a policy ──────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: Any, *, duration_s: float, sample_rate: int) -> NoiseV2Pool:
        def g(key: str, default: Any = None) -> Any:
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            return getattr(cfg, key, default)

        rps = g("rps", {}) or {}
        fits = g("fits")
        bank = g("preset_bank")
        if (fits is None) == (bank is None):
            raise ValueError("noise_v2 needs exactly one of 'fits' or 'preset_bank'")
        if bank is not None:
            entries = load_preset_bank(str(bank))
        else:
            entries = tuple(
                NoiseV2Entry(
                    name=str(item.get("name", f"fit{index}")),
                    cruise=_read_fit(item["cruise"], where=f"noise_v2 fits[{index}].cruise"),
                    standby=(
                        None
                        if item.get("standby") is None
                        else _read_fit(item["standby"], where=f"noise_v2 fits[{index}].standby")
                    ),
                    traj_rig=None if item.get("traj_rig") is None else str(item["traj_rig"]),
                )
                for index, item in enumerate(list(fits))
            )

        def pair(key: str, default: tuple[float, float]) -> tuple[float, float]:
            v = g(key, default)
            return (float(v[0]), float(v[1]))

        return cls(
            sample_rate=sample_rate,
            duration_s=duration_s,
            n_mics=int(g("n_mics", 8)),
            n_rotors=int(g("n_rotors", 4)),
            entries=entries,
            rps_kind=str(rps.get("kind", "full_flight")),
            flight_fs=float(rps.get("flight_fs", 200.0)),
            flight_reuse=int(rps.get("flight_reuse", 32)),
            fitted_traj=rps,
            drone_profile_range=pair("drone_profile_range", (0.0, 1.0)),
            aggressiveness=(
                (float(rps["aggressiveness_range"][0]), float(rps["aggressiveness_range"][1]))
                if rps.get("aggressiveness_range") is not None
                else float(rps.get("aggressiveness", 1.0))
            ),
            mode_scales=(dict(rps["mode_scales"]) if rps.get("mode_scales") else None),
            rotor_trim_rel=(tuple(rps["rotor_trim_rel"]) if rps.get("rotor_trim_rel") else None),
            flight_phases=(
                {k: tuple(v) for k, v in dict(rps["phases"]).items()}
                if rps.get("phases") is not None
                else None
            ),
            rps_scale_range=pair("rps_scale_range", (1.0, 1.0)),
            render_reuse=int(g("render_reuse", 1)),
            render_pool=int(g("render_pool", 0)),
            normalize_rms=(
                pair("normalize_rms_range", (0.0, 0.0))
                if g("normalize_rms_range") is not None
                else (None if g("normalize_rms") is None else float(g("normalize_rms")))
            ),
            level_mode=str(g("level_mode", "window")),
            comb_offset_db=float(g("comb_offset_db", 0.0)),
            seed=int(g("seed", 0)),
        )

    def close(self) -> None:  # interface parity with the other pools
        return None

    # ── trajectory ──────────────────────────────────────────────────────────

    def sample_rps(self, rng: np.random.Generator, duration_s: float) -> np.ndarray:
        """``(R, T)`` rotor speeds at the audio rate for one window.

        A whole flight is generated at ``flight_fs`` and held for
        ``flight_reuse`` windows; each window is a uniformly placed slice of it
        (:func:`data_processing.trajectory_model.window_flight`), so successive
        windows visit the ground, warm-up, takeoff, cruise and landing phases
        in proportion to their durations — and the standby fit of a per-regime
        entry actually gets used. The window is multiplied by one log-uniform
        draw from ``rps_scale_range``, which is exact: a stopped rotor stays
        stopped and every other speed moves with its own comb.
        """
        lo_s, hi_s = self.rps_scale_range
        scale = (
            float(np.exp(rng.uniform(np.log(lo_s), np.log(hi_s))))
            if lo_s > 0.0 and hi_s > lo_s
            else float(lo_s)
        )
        if self._flight is None or self._flight.uses >= self.flight_reuse:
            self._flight = trajectory_model.make_flight_cache(self._new_flight(rng), self.flight_fs)
        self._flight.uses += 1
        window = trajectory_model.window_flight(self._flight, rng, duration_s, self.sample_rate)
        return scale * window

    def _new_flight(self, rng: np.random.Generator) -> np.ndarray:
        if self._traj is not None:
            flight = self._traj.flight(rng, self.flight_fs)
        else:
            aggressiveness = (
                float(rng.uniform(*self.aggressiveness))
                if isinstance(self.aggressiveness, tuple)
                else self.aggressiveness
            )
            blend = float(rng.uniform(*self.drone_profile_range))
            phases = (
                rps_synthesis.FlightPhaseRanges(**self.flight_phases)
                if self.flight_phases
                else None
            )
            flight = rps_synthesis.generate_full_flight(
                None,
                self.flight_fs,
                drone_profile=blend,
                aggressiveness=aggressiveness,
                phases=phases,
                mode_scales=self.mode_scales,
                rotor_trim_rel=self.rotor_trim_rel,
                rng=rng,
            )
        flight = np.atleast_2d(np.asarray(flight, dtype=np.float64))
        if flight.shape[0] != self.n_rotors:
            raise ValueError(
                f"the trajectory source produced {flight.shape[0]} rotors, "
                f"this pool renders n_rotors={self.n_rotors}"
            )
        return flight

    # ── level ───────────────────────────────────────────────────────────────

    def _draw_level(self, rng: np.random.Generator) -> float | None:
        """One reference level, log-uniform over the configured range."""
        if self.normalize_rms is None:
            return None
        if isinstance(self.normalize_rms, tuple):
            lo, hi = self.normalize_rms
            return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        return float(self.normalize_rms)

    def _apply_level(
        self, audio: np.ndarray, rps: np.ndarray, entry: NoiseV2Entry, level: float | None
    ) -> np.ndarray:
        """Rescale to ``level``, or leave the fit's ABSOLUTE level alone.

        ``level_mode`` has the stochastic pool's meaning. ``"window"``
        normalises the window itself, so every window leaves at the same level
        whatever the rotors are doing. ``"flight"`` treats the number as the
        level the window would have AT THE REFERENCE SPEED and puts the
        window's own speed envelope back — here the fitted floor envelope
        ``mean_r(speed_r ** floor_exp) + floor_static_rel``, which is the very
        factor :func:`render_noise` shaped the floor with, as a power, hence
        the square root.
        """
        if level is None:
            return audio
        rms = float(np.sqrt(np.mean(np.square(audio)))) or 1.0
        gain = float(level) / rms
        if self.level_mode == "flight":
            floor = check_schema(entry.cruise)["floor"]
            speed = np.maximum(rps, 0.0) / AMP_RPS_REF
            envelope = (speed ** float(floor["floor_exp"])).mean(axis=0) + float(
                floor["floor_static_rel"]
            )
            gain *= float(np.sqrt(max(float(np.mean(envelope)), 0.0)))
        return audio * gain

    # ── render ──────────────────────────────────────────────────────────────

    def render(
        self, rng: np.random.Generator, duration_s: float
    ) -> tuple[np.ndarray, np.ndarray, NoiseV2Entry]:
        """``(audio (M, T) float32, rps (R, T) float32, entry)`` for one window."""
        rps = self.sample_rps(rng, duration_s)
        entry = self.entries[int(rng.integers(len(self.entries)))]
        level = self._draw_level(rng)
        # The render's own seed: folded from the pool's configured seed and one
        # draw of the stream's generator, so two arms that differ only in
        # ``seed`` draw independent shaft and line phases on the same window.
        seed = int(
            np.random.SeedSequence([self._base_seed, int(rng.integers(1 << 31))]).generate_state(
                1, dtype=np.uint32
            )[0]
        )
        if entry.per_regime:
            assert entry.standby is not None
            audio = render_noise_regimes(
                {"standby": entry.standby, "cruise": entry.cruise},
                rps,
                sr=self.sample_rate,
                n_mics=self.n_mics,
                seed=seed,
                sr_work=self.sr_work,
            )
        else:
            audio = render_noise(
                entry.cruise,
                rps,
                sr=self.sample_rate,
                n_mics=self.n_mics,
                seed=seed,
                sr_work=self.sr_work,
            )
        audio = self._apply_level(np.asarray(audio, dtype=np.float64), rps, entry, level)
        return audio.astype(np.float32), rps.astype(np.float32), entry

    def _pooled_render(
        self, rng: np.random.Generator, duration_s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """One rendered clip, from the rolling pool when reuse is enabled.

        Reuse starts immediately and the pool GROWS: one slot is refreshed
        every ``render_reuse`` draws, appended until the pool is full and
        replacing a random slot afterwards.
        """
        if self.render_reuse <= 1:
            audio, rps, _ = self.render(rng, duration_s)
            return audio, rps
        matching = [entry for entry in self._pool if entry[0] == duration_s]
        if not matching or self._pool_draws % self.render_reuse == 0:
            audio, rps, _ = self.render(rng, duration_s)
            entry = (float(duration_s), audio, rps)
            if len(self._pool) < self.render_pool:
                self._pool.append(entry)
            else:
                self._pool[int(rng.integers(len(self._pool)))] = entry
            matching = [item for item in self._pool if item[0] == duration_s]
        self._pool_draws += 1
        _, audio, rps = matching[int(rng.integers(len(matching)))]
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
            meta={"recording_id": "noise_v2"},
            mic_pos=self.mic_pos,
            rotor_pos=self.rotor_pos,
        )
