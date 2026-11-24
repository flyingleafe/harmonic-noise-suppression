"""Multiscale path energy score and its real-data sensitivity gate.

The ensemble axis is independent flights, never timestamps. Every transformed
window stays a joint vector of ordered time samples and all four rotor modes.
The score is conditional on the schedule, not a score of phase durations.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.rps_traj import phase_dynamics as pdyn
from experiments.rps_traj import regime_campaign as rc
from experiments.rps_traj.phases import Landmarks, landmarks
from experiments.rps_traj.scores import energy_score
from tracking.rotors import MIXER

FS = pdyn.FS
ORIGINS = 8
DRAWS = 32
CONTROL_PHASE = {
    "jitter": "cruise",
    "drift": "cruise",
    "reverse_rotor": "cruise",
    "clip_spike": "startup",
}


def features(paths: np.ndarray) -> dict[str, np.ndarray]:
    """(paths,rotors,time) -> (paths,modes,ordered coefficients) by half-width.

    One octave per key; shifts of half a box width reduce dependence on the
    alignment of an event with the analysis grid. No wraparound or padding.
    """
    x = (MIXER.T / 4) @ paths
    out = {"raw": x, "level": x.mean(axis=-1, keepdims=True)}
    prefix = np.pad(x.cumsum(axis=-1), ((0, 0), (0, 0), (1, 0)))
    width = 2  # 80 ms, not the repeated samples of the slowest logger
    while 2 * width <= x.shape[-1]:
        start = np.arange(0, x.shape[-1] - 2 * width + 1, max(1, width // 2))
        out[str(width)] = (
            prefix[..., start + 2 * width] - 2 * prefix[..., start + width] + prefix[..., start]
        ) / width
        width *= 2
    return out


@dataclass
class WindowScore:
    scale: dict[str, np.ndarray]
    observed: dict[str, np.ndarray]

    @classmethod
    def prepare(cls, real: np.ndarray, reference: np.ndarray, floor: float) -> WindowScore:
        train = features(reference)
        scale = {}
        for key, a in train.items():
            if key in ("raw", "level"):
                size = a.std(axis=(0, 2))
            else:
                size = np.sqrt((a**2).mean(axis=(0, 2)))
            scale[key] = np.maximum(size[:, None], floor)
        return cls(scale, features(real[None]))

    def profile(self, ensemble: np.ndarray) -> dict[str, float]:
        out = {}
        for key, a in features(ensemble).items():
            scale = self.scale[key]
            x = (a / scale).reshape(len(a), -1)
            y = (self.observed[key] / scale).ravel()
            out[key] = energy_score(x, y) / np.sqrt(y.size)
        out["score"] = float(np.mean(list(out.values())))
        return out


@dataclass
class Trace:
    name: str
    speed: np.ndarray
    marks: Landmarks

    def spans(self) -> dict[str, tuple[int, int]]:
        m = self.marks
        out = {}
        for phase, (event, lo, hi) in pdyn.windows(m).items():
            i = getattr(m, event)
            out[phase] = (i + lo, i + hi)
        if m.arrival is not None:
            out["cruise"] = (m.arrival + round(3 * FS), (m.landing or m.n) - round(2.5 * FS))
        return out


def extract(x: np.ndarray, span: tuple[int, int], n: int, u: float, phase: str) -> np.ndarray:
    lo, hi = span
    if phase == "standby":
        lo = hi - n  # aligned to departure, not plateau onset
    if phase == "cruise":
        lo += round(u * (hi - lo - n))
    return x[:, lo : lo + n]


def distort(
    name: str, refs: np.ndarray, donors: list[Trace], rng: np.random.Generator
) -> np.ndarray:
    out = refs.copy()
    for x, flight in zip(out, donors, strict=True):
        hover = flight.marks.hover
        if name == "jitter":
            x += rng.normal(0, 0.02 * hover, x.shape)
        elif name == "drift":
            x += rng.choice([-1, 1]) * 0.08 * hover * np.linspace(-1, 1, x.shape[-1])
        elif name == "reverse_rotor":
            x[0] = x[0, ::-1].copy()
        elif name == "clip_spike":
            assert flight.marks.standby_level is not None
            x -= np.maximum(0, x.mean(axis=0) - flight.marks.standby_level.mean())
        else:
            raise ValueError(name)
    return out


def mean_profile(profiles: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([p[key] for p in profiles])) for key in profiles[0]}


def evaluate(*, models: bool, out_dir: Path) -> list[dict]:
    """Run the same windowing and normalization in both passes. Real donors are
    other flights from the training split, one path per donor per replicate.
    Model paths are independent draws on the held-out schedule.
    """
    rows = []
    for rig in rc.FLIGHT_RIGS:
        data = rc.rig_data(rig)
        traces = []
        for f in data.flights:
            x = rc._to_flight_rate(f.rps, f.fs)
            m = landmarks(x, FS)
            if m is None:
                raise ValueError(f"no landmarks: {f.flight}")
            traces.append(Trace(f.flight, x, m))
        for fold, held in enumerate(data.held_out):
            train = [f for f in traces if f.name not in held]
            floor = 0.005 * float(np.median([f.marks.hover for f in train]))
            generators = (
                {
                    arm: pdyn.fit_arm(arm, rig, str(fold), str(out_dir), same_clock=True)
                    for arm in pdyn.EXAMPLE_ARMS
                }
                if models
                else {}
            )
            for i, target in enumerate(f for f in traces if f.name in held):
                rng = np.random.default_rng(2026 + 10 * fold + i)
                generated = {}
                for arm, generate in generators.items():
                    paths = generate(target.marks, DRAWS, np.random.default_rng(40 + fold))
                    if paths is None or len(paths) != DRAWS:
                        raise ValueError(f"incomplete draws: {rig} {arm} {target.name}")
                    generated[arm] = paths
                for phase, span in target.spans().items():
                    donors = [f for f in train if phase in f.spans()]
                    if len(donors) < 2:
                        raise ValueError(f"no independent reference: {rig}, {phase}")
                    spans = [f.spans()[phase] for f in donors]
                    n = min(1024, *(b - a for a, b in [span, *spans]))
                    if n < 4:
                        raise ValueError(f"too short: {target.name} {phase}")
                    per: dict[str, list] = defaultdict(list)
                    reps = ORIGINS if phase == "cruise" else 1
                    for rep in range(reps):
                        u = (rep + 0.5) / reps
                        obs = extract(target.speed, span, n, u, phase)
                        refs = np.stack(
                            [
                                extract(f.speed, s, n, rng.random(), phase)
                                for f, s in zip(donors, spans, strict=True)
                            ]
                        )
                        scorer = WindowScore.prepare(obs, refs, floor)
                        per["real"].append(scorer.profile(refs))
                        for name, region in CONTROL_PHASE.items():
                            if region == phase and (name != "clip_spike" or rig == "michaels"):
                                per[name].append(scorer.profile(distort(name, refs, donors, rng)))
                        for arm, paths in generated.items():
                            # Map the common command clock to the generator's clock;
                            # never re-detect or warp a generated transient.
                            samples = []
                            for path, events in paths:
                                shift = (events["motors_on"] or 0) - (target.marks.motors_on or 0)
                                lo = span[0] + shift
                                if phase == "standby":
                                    lo = span[1] + shift - n
                                if phase == "cruise":
                                    lo += round(u * (span[1] - span[0] - n))
                                w = pdyn.cut(path, lo, 0, n)
                                if w is None:
                                    raise ValueError("generated path starts after window")
                                samples.append(w)
                            per[arm].append(scorer.profile(np.stack(samples)))
                    rows.append(
                        {
                            "rig": rig,
                            "fold": fold,
                            "flight": target.name,
                            "phase": phase,
                            "samples": n,
                            "donors": len(donors),
                            "origins": reps,
                            "profiles": {k: mean_profile(v) for k, v in per.items()},
                        }
                    )
    return rows


def gate(rows: list[dict]) -> dict:
    """Lock before scoring models: median paired penalty > clean flight MAD,
    majority of flights worse, and correct octave localization for jitter / drift.
    Each control must pass on a strict majority of its applicable rigs.
    """
    summary = {}
    for control in CONTROL_PHASE:
        rigs = {}
        for rig in rc.FLIGHT_RIGS:
            group = [r["profiles"] for r in rows if r["rig"] == rig and control in r["profiles"]]
            if not group:
                continue
            clean = np.array([p["real"]["score"] for p in group])
            mad = float(np.median(np.abs(clean - np.median(clean))))
            delta = np.array([p[control]["score"] - p["real"]["score"] for p in group])
            keys = set.intersection(*(set(p[control]) for p in group))
            profile = {
                k: float(np.median([p[control][k] - p["real"][k] for p in group])) for k in keys
            }
            fine = profile["2"]
            coarse = profile[max((k for k in keys if k.isdigit()), key=int)]
            localized = (control != "jitter" or fine > max(coarse, 0)) and (
                control != "drift" or coarse > max(fine, 0)
            )
            rigs[rig] = {
                "clean_median": float(np.median(clean)),
                "clean_mad": mad,
                "median_delta": float(np.median(delta)),
                "worse": int((delta > 0).sum()),
                "n": len(group),
                "profile_delta": profile,
                "passed": bool(np.median(delta) > mad and (delta > 0).mean() > 0.5 and localized),
            }
        summary[control] = {
            "rigs": rigs,
            "passed": sum(x["passed"] for x in rigs.values()) > len(rigs) / 2,
        }
    return summary
