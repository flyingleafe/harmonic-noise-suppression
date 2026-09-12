"""The central fit: one Whittle MAP over a clip set, selected by REGIME.

Every fitted campaign the project has run is this one call with a different
regime. A regime fixes four things and nothing else:

* **where the clips come from** — a published frames dataset plus a window
  rule (flight: windows where every rotor stays inside a speed band; bench:
  the steady span of each single-motor recording);
* **the forward model's variant** — what the clip can physically have (a
  clamped bench motor has no rate trajectory, so no chirp width and no
  telemetry-offset latent; a flown rig has eight microphones and a moving
  shaft);
* **the harmonic ladder's cap** — set by the slowest rate the regime admits;
* **which parameters are tied across clips** — one shaft's profile shared over
  setpoints on the bench, one rig's profile shared over windows in flight.

The bench regime is what Stage 1 fitted (four motors x five setpoints of
DREGON's single-motor rig, one channel, the rate itself fitted because those
recordings have no telemetry) and it stays a first-class option here: the
two-component line model was identified on it, and it is the only regime where
one rotor is the whole signal.

Run it with ``scripts/stochastic_fit.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit import rig
from experiments.stochastic_fit import stage1_bayes as SB
from experiments.stochastic_fit import stage2 as S2
from experiments.stochastic_fit.data import periodogram


@dataclass(frozen=True)
class Regime:
    """Everything a regime fixes; see the module docstring."""

    name: str
    #: published frames dataset the clips come from by default
    dataset: str
    #: forward-model variant (``model.Spec`` fields)
    variant: dict[str, Any]
    #: harmonic-ladder cap, sized by the slowest rate the regime admits
    k_cap: int
    seconds: float
    #: channels fitted by default — ``None`` is every published microphone
    channels: tuple[int, ...] | None
    #: rig-level ties (``rig.RigSpec`` fields)
    ties: dict[str, Any] = field(default_factory=dict)
    #: flight window selection; ignored by a bench regime
    min_rps: float | None = None
    max_rps: float | None = None
    stride_s: float | None = None
    #: static single-motor cells instead of flight windows
    bench: bool = False


REGIMES: dict[str, Regime] = {
    # Michael's FLY125 in cruise: the accepted flight fit (8 clips x 16 s).
    "cruise": Regime(
        name="cruise",
        dataset=S2.FIT_DATASET,
        variant=S2.S2_VARIANT,
        k_cap=int(S2.REGIMES["cruise"]["k_cap"]),
        seconds=S2.CRUISE_SECONDS,
        channels=None,
        ties=dict(rotor_delta=True),
        min_rps=float(S2.REGIMES["cruise"]["min_rps"]),
        max_rps=S2.REGIMES["cruise"]["max_rps"],
        stride_s=S2.REGIMES["cruise"]["stride_s"],
    ),
    # Every rotor spinning but slow. A slower shaft puts MORE orders under
    # Nyquist (7900 / 35 = 226 against 121 in cruise), so standby needs a
    # taller comb, not a shorter one.
    "standby": Regime(
        name="standby",
        dataset=S2.FIT_DATASET,
        variant=S2.S2_VARIANT,
        k_cap=int(S2.REGIMES["standby"]["k_cap"]),
        seconds=S2.STANDBY_SECONDS,
        channels=None,
        ties=dict(rotor_delta=True),
        min_rps=float(S2.REGIMES["standby"]["min_rps"]),
        max_rps=S2.REGIMES["standby"]["max_rps"],
        stride_s=S2.REGIMES["standby"]["stride_s"],
    ),
    # DREGON's single-motor bench: ONE rotor is the whole signal, the rate is a
    # fitted parameter (no telemetry is published for these recordings), and
    # every dynamic is pinned off because a clamped motor at a fixed setpoint
    # has none. The profile, the width law, the floor shape and the speed law
    # are tied across cells: one shaft measured at five setpoints.
    "bench": Regime(
        name="bench",
        dataset=SB.BENCH_DATASET,
        variant=SB.BENCH_VARIANT,
        k_cap=SB.K_CAP,
        seconds=SB.BENCH_SECONDS,
        channels=(SB.CHANNEL,),
        ties=dict(tie_profile=True, tie_width=True, tie_floor_shape=True, tie_speed_law=True),
        bench=True,
    ),
}


def rig_family(group: str) -> str:
    """Which physical rig a clip's group belongs to.

    ``dregon_room1``/``dregon_room2`` are the same airframe in two rooms;
    ``fly124``/``fly125`` are Michael's two flights of one rig; ``motorN`` are
    the single-motor bench cells.
    """
    g = str(group)
    if g.startswith("motor"):
        return "bench"
    if g.startswith("dregon"):
        return "dregon"
    if g.startswith("fly"):
        return "michaels"
    return g


def assert_one_rig(staged_rows: list[tuple[str, str, Any, Any]]) -> None:
    """Refuse a clip set that spans two rigs, or two channel/rotor counts.

    `rig.fit_rig` takes the rotor count and the `Spec` of the FIRST clip and
    ties the rig-level parameters (profile, width law, floor shape, microphone
    gains) across all of them. Pooling two rigs therefore does not fit two
    rigs: it ties one rig's parameters to the other's clips, silently.
    """
    rigs = {rig_family(group) for _cid, group, _clip, _pg in staged_rows}
    if len(rigs) > 1:
        raise ValueError(
            f"a rig fit cannot pool {sorted(rigs)}: fit_rig ties the rig-level "
            "parameters across every clip. Fit each rig separately."
        )
    shapes = {
        (int(clip.audio.shape[0]), int(clip.rps.shape[0])) for _cid, _g, clip, _pg in staged_rows
    }
    if len(shapes) > 1:
        raise ValueError(
            f"clips disagree on (channels, rotors): {sorted(shapes)} — one rig fit needs one "
            "geometry; select channels explicitly or fit the recordings separately"
        )


def rows(
    regime: str = "cruise",
    *,
    recordings: tuple[str, ...] = (S2.FIT_RECORDING,),
    dataset: str | None = None,
    version: str | None = None,
    rps_key: str = C.DEFAULT_RPS_KEY,
    channels: str | tuple[int, ...] | None = None,
    max_clips: int = 8,
    seconds: float | None = None,
    min_rps: float | None = None,
    motors: tuple[int, ...] = SB.FIT_MOTORS,
    speeds: tuple[int, ...] = SB.SPEEDS,
    log: Any = print,
) -> list[tuple[str, str, Any, Any]]:
    """``[(clip_id, group, clip_16k, periodogram)]`` for one regime.

    A flight recording is named ``[dataset[@version]:]RECORDING_ID``, so one
    rig fit can pool several recordings **of the same rig** (DREGON's five
    room2 training flights, or room1 with room2); ``max_clips`` is per
    recording. A bare id takes ``dataset``. Pooling two DIFFERENT rigs is
    refused by :func:`assert_one_rig`: `fit_rig` builds one `RigParams` from
    the first clip's spec and rotor count and ties the rig-level parameters
    across every clip, so a mixed set would tie one rig's profile, widths and
    microphone pattern to another's.
    """
    r = REGIMES[regime]
    dataset = dataset or r.dataset
    chans = channels if channels is not None else r.channels
    seconds = float(r.seconds if seconds is None else seconds)

    if r.bench:
        cells = SB.bench_cells(
            tuple(motors), tuple(speeds), dataset=dataset, version=version, channels=chans
        )
        out = []
        for (motor, speed), clip in cells.items():
            pg = periodogram(clip, n_fft=SB.N_FFT, hop=SB.HOP)
            out.append((clip.clip_id, f"motor{motor}", clip, pg))
            log(
                f"  Motor{motor}_{speed}: {clip.audio.shape[0]} mic, {pg.power.shape[1]} frames, "
                f"rate seed {clip.meta['rate_seed']:.1f} rev/s"
            )
        return out

    out = []
    for spec in recordings:
        source, _, rid = str(spec).rpartition(":")
        found = S2.cruise_clips(
            rid,
            dataset=source or dataset,
            version=version,
            rps_key=rps_key,
            channels=chans,
            tag=regime,
            seconds=seconds,
            max_clips=max_clips,
            min_rps=float(min_rps if min_rps is not None else (r.min_rps or 0.0)),
            max_rps=r.max_rps,
            stride_s=r.stride_s,
        )
        if not found:
            log(f"  {rid}: no {regime} window of {seconds:g}s")
        for cid, clip, pg in found:
            out.append((cid, clip.group, clip, pg))
            log(
                f"  {cid}: {clip.audio.shape[0]} mics, {pg.power.shape[1]} frames, "
                f"rotors {np.round(np.asarray(pg.rps).mean(axis=1), 1).tolist()} rev/s"
            )
    return out


def fit(
    regime: str = "cruise",
    *,
    recordings: tuple[str, ...] = (S2.FIT_RECORDING,),
    dataset: str | None = None,
    version: str | None = None,
    rps_key: str = C.DEFAULT_RPS_KEY,
    channels: str | tuple[int, ...] | None = None,
    max_clips: int = 8,
    seconds: float | None = None,
    k_cap: int | None = None,
    min_rps: float | None = None,
    motors: tuple[int, ...] = SB.FIT_MOTORS,
    speeds: tuple[int, ...] = SB.SPEEDS,
    rotor_delta: bool | None = None,
    floor_dynamics: bool = False,
    iters: tuple[int, int, int] = (120, 120, 300),
    ladder: tuple[int, ...] = (16, 48),
    device: str = "cpu",
    log: Any = print,
) -> dict[str, Any]:
    """Joint MAP over the clips of one regime, with the rig-level ties tied.

    ``rotor_delta`` gives each rotor (flight) or each motor (bench) its own
    profile offset on top of the shared shape; it defaults to the regime's own
    setting. ``floor_dynamics`` lets the broadband floor's level and tilt drift
    — the only dynamic the identifiability argument permits, because a floor GP
    cannot steal per-order line level (line drift absorbed +4.97 dB of static
    profile on the bench, which is why it stays off).

    The summary records the dataset, version, label track and channels it was
    fitted on: a fit on the refined label is not comparable with one on raw
    telemetry, and a claim about a campaign has to name its inputs.
    """
    r = REGIMES[regime]
    staged_rows = rows(
        regime,
        recordings=tuple(recordings),
        dataset=dataset,
        version=version,
        rps_key=rps_key,
        channels=channels,
        max_clips=max_clips,
        seconds=seconds,
        min_rps=min_rps,
        motors=motors,
        speeds=speeds,
        log=log,
    )
    if not staged_rows:
        raise ValueError(
            f"no bench cell for motors {motors} speeds {speeds}"
            if r.bench
            else f"{list(recordings)}: no {regime} window of {seconds or r.seconds} s found"
        )
    assert_one_rig(staged_rows)
    variant = {**r.variant, **(S2.FLOOR_DYNAMICS if floor_dynamics else {})}
    staged = rig.stage_clips(
        staged_rows,
        variant=variant,
        f_max=S2.F_MAX,
        k_cap=int(r.k_cap if k_cap is None else k_cap),
    )
    ties = dict(r.ties)
    if rotor_delta is not None:
        ties["rotor_delta"] = bool(rotor_delta)
    out = rig.fit_rig(
        staged,
        rig.RigSpec(**ties),
        device=device,
        ladder=ladder,
        iters=iters,
        lr=0.1,
        log=log,
    )
    first = staged_rows[0][2]
    out["regime"] = regime
    out["variant"] = variant
    out["rig_ties"] = ties
    provenance: dict[str, Any] = dict(
        regime=regime,
        dataset=first.meta.get("dataset"),
        dataset_version=first.meta.get("dataset_version"),
        rps_key=first.meta.get("rps_key"),
        channels=list(first.meta.get("channels", [])),
        n_fft=SB.N_FFT,
        clips=[cid for cid, _g, _c, _pg in staged_rows],
        starts_s=[float(clip.meta.get("start_s", 0.0)) for _c, _g, clip, _pg in staged_rows],
        seconds=float(seconds or r.seconds),
    )
    if r.bench:
        provenance.update(
            motors=list(motors),
            speeds=list(speeds),
            rate_seeds={cid: float(clip.meta["rate_seed"]) for cid, _g, clip, _pg in staged_rows},
        )
    else:
        provenance.update(recordings=list(recordings))
    out["data"] = provenance
    return out


#: Writing a summary is the same job for every regime.
save = S2.save

__all__ = ["REGIMES", "Regime", "assert_one_rig", "fit", "rig_family", "rows", "save"]
