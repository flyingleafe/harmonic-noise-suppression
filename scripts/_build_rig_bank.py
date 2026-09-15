"""Build a PRESET BANK of render-ready rig parameters for a training stream.

PENDING USER DECISION (2026-09-15) — READ BEFORE REBUILDING A BANK. The banks
in `data/rig_banks/` have two known defects: the anchors' LAST comb order was
never constrained by the fit (it sat outside the 30-7900 Hz band at the fit's
own speeds) and renders as a loud sweeping line at stream speeds, and the two
rigs' anchors disagree on label provenance (Michael raw, DREGON `rps_refined`).
The fix is a user decision, stated with evidence and options in
`docs/experiments/rig-sampler-transfer-pair.md` § "Open decision: rebuild the
banks". Rebuilding a bank before that is answered reproduces the defects.

The bank is what lets an online-mix policy train on the rig-neighbourhood
sampler's family: `experiments.stochastic_fit.rig_sampler` draws exports, this
script turns each one into the renderer's own coordinates and writes them as one
JSON file that `data_processing.stochastic_rotor_noise.StochasticNoisePool`
reads through `preset_bank:`. The conversion happens HERE, offline, because
`src/data_processing/` may not import `src/experiments/` (import-linter), and
because a guarded redraw loop has no business running in a DataLoader worker.

Two modes, which are the two transfer arms:

    # EASY: close neighbourhoods of the two real rigs' cruise fits
    python scripts/_build_rig_bank.py --mode neighbourhood --name rig_easy_n512 \
        --anchor results/S2/cruise_8clip.json:fly125_cruise_00 \
        --dynamics conf/online_mix/rig_fitted_5050.yaml:1 \
        --anchor results/S2/dregon_room2_cruise_refined.json:free-flight_nosource_room2_cruise_00 \
        --dynamics conf/online_mix/rig_fitted_5050.yaml:0 \
        --strength 2.5

    # HARD: a wide cloud along the path between them
    python scripts/_build_rig_bank.py --mode path --name rig_hard_n512 \
        --anchor results/S2/cruise_8clip.json:fly125_cruise_00 \
        --dynamics conf/online_mix/rig_fitted_5050.yaml:1 \
        --anchor results/S2/dregon_room2_cruise_refined.json:free-flight_nosource_room2_cruise_00 \
        --dynamics conf/online_mix/rig_fitted_5050.yaml:0 \
        --spread 2.0

Writes `data/rig_banks/<name>.json` (gitignored, like every other fit artifact)
with the entries and a `provenance` block: the anchors and their SHA-256, the
dynamics donors, the mode and widths, the seed, the min-speed vector the order
ladder was sized from, the resulting `k_use`, the guard statistics and git HEAD.

WHAT ONE ENTRY IS MADE OF. A stage-2 export describes ONE FITTED CLIP: its rig
identity (timbre, floor shape, line widths, channel pattern, speed law) and
nothing that varies inside a window — `stage2.params_from_export` deliberately
zeroes every drift process, because the fit carries those as per-clip latents.
A training window needs both, so each entry is

    the stream's own per-clip draw (`srn.sample_params` on the DONOR policy's
    `ranges`: the measured amplitude OU process, the floor drift, the per-mic
    modulation, the shaft-jitter time constant and spread, the label error)
    + the sampled rig's identity on top.

The line WIDTH crosses over through the same seam the fitted policy uses: in
`line_mode: fm` a line's width comes from the shaft's speed jitter, not from
`gamma`, and a Gaussian line with shaft-rate std sigma has HWHM
`sqrt(2 log 2) * k * sigma`, so the entry's `gamma_slope` is handed to
`sample_params` as `fixed_shaft_jitter_rps = gamma_slope / sqrt(2 log 2)` —
`raw_predictive.population_ranges`' declared conversion, verbatim. Without this
a bank would render a comb of dead-steady tones, which is not the fitted rig and
not the stream the arms are compared against.

WHY A MIN-SPEED VECTOR. `params_from_export` sizes the order ladder from
`rates.min()` so the comb reaches Nyquist for the SLOWEST rotor; a bank entry is
rendered at every speed the stream produces, so the ladder is built once at the
slowest CRUISE speed the policy can ask for — the policy's `rps_scale_range`
lower bound times the slowest blended cruise trim of `rps_synthesis` (Michael's
72 rev/s against DREGON's 80). Lines past a faster window's Nyquist carry no
power and are masked by `fm_lines`, so an over-long ladder is safe while a short
one would cut the comb.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

import experiments.stochastic_fit.rig_sampler as rig_sampler
from data_processing import rps_synthesis
from data_processing import stochastic_rotor_noise as srn
from experiments.stochastic_fit import stage2

#: Where banks live. Gitignored (`/data/*`), like the fitted summaries in
#: `omnirun-outputs/`; a remote training job needs the file shipped with the
#: checkout.
BANK_DIR = Path("data/rig_banks")

#: The rate the training stream renders at, i.e. the pool's `sample_rate`. An
#: entry's `sample_rate` must match its pool's, and the pool refuses the bank
#: loudly when it does not.
WORK_RATE = 16000

#: Microphones the stream renders. The fitted per-(mic, rotor) pattern is copied
#: into the entry, so this must match the policy's `n_mics`.
N_MICS = 8

#: Lower end of the policy's `rps_scale_range`, the per-window multiplier on the
#: whole trajectory (`conf/online_mix/rig_fitted_5050.yaml`).
DEFAULT_RPS_SCALE_MIN = 0.45

#: Upper end of the same range, used by the fold check below.
DEFAULT_RPS_SCALE_MAX = 1.2

#: The band grid the Nyquist fold check reports on (Hz), and the ceiling it
#: enforces below 7.5 kHz.
FOLD_BANDS = ((1500, 3000), (3000, 5000), (5000, 6500), (6500, 7500), (7500, 7900), (7900, 8000))
FOLD_TOLERANCE_DB = 0.05

#: STATED BOUND on the speed exponents, and the only place in this pipeline
#: where a sampled value is bounded by anything other than a measured width.
#:
#: WHY. `rig_sampler` draws `amp_exp` and `floor_exp` with the measured
#: between-refit sigma of 2.733, which is a legitimate WIDTH but has a tail no
#: fit supports: an unbounded bank reached 24.7 and 31.9 dB per dB of rotor
#: speed, and at that exponent an idle or ramp window carries a comb ~20 dB
#: less prominent than the same rig at cruise — the low-speed part of every
#: training clip would be a fiction. The bound is the range the six measured
#: fits actually occupy (`results/rig_sampler/structure.json`,
#: `between_rig.per_fit`): `amp_exp` 4.398 (michael_standby) to 14.111
#: (dregon_flight), `floor_exp` -3.711 (dregon_cruise_refined) to 6.792
#: (michael_cruise).
#:
#: The `floor_exp` floor is raised from the measured -3.711 to 0 for a reason
#: that is not statistical: a negative exponent is `0 ** negative` at the exact
#: zero of a full flight's ground phase, i.e. infinite line power and NaN
#: audio, which is why `rig_sampler.check_sample` refuses it outright. A
#: clipped draw is re-checked against the sampler's guards and redrawn if the
#: clip has made it implausible, so the bank stays an honest draw from the
#: bounded family rather than a pile-up on the boundary.
AMP_EXP_BOUND = (4.398, 14.111)
FLOOR_EXP_BOUND = (0.0, 6.792)
EXPONENT_BOUND_SOURCE = "results/rig_sampler/structure.json:between_rig.per_fit (six fits)"

#: HWHM of a unit-std Gaussian, i.e. the gamma-to-shaft-jitter conversion.
GAUSS_HWHM = float(np.sqrt(2.0 * np.log(2.0)))

#: The fields an entry takes from the FIT rather than from the donor policy's
#: per-clip draw. Everything not listed here (and not a `fixed_*` range below)
#: is the stream's own draw, so the arms keep the base policy's dynamics.
IDENTITY_FIELDS = (
    "profile_db",
    "harm_mean_db",
    "floor_ctrl_hz",
    "floor_ctrl_db",
    "floor_tilt_db_oct",
    "floor_mean_db",
    "floor_static_rel",
    "amp_rps_exponent",
    "amp_rps_exponent_floor",
    "amp_rps_ref",
    "coherence_k_half",
    "gamma_min_bins",
)


def anchor_spec(text: str) -> tuple[Path, str]:
    """``<fit.json>:<clip-id>`` -> ``(path, selector)``."""
    path_text, _, selector = text.rpartition(":")
    if not path_text or not selector:
        raise argparse.ArgumentTypeError(f"--anchor wants <fit.json>:<clip_id>, got {text!r}")
    return Path(path_text), selector


def dynamics_spec(text: str) -> tuple[Path, int]:
    """``<policy.yaml>:<source-index>`` -> ``(path, index)``."""
    path_text, _, index = text.rpartition(":")
    if not path_text or not index.isdigit():
        raise argparse.ArgumentTypeError(
            f"--dynamics wants <policy.yaml>:<source-index>, got {text!r}"
        )
    return Path(path_text), int(index)


def donor_ranges(path: Path, index: int) -> srn.StochasticRanges:
    """The per-clip draw ranges of one stochastic source of one policy."""
    policy = yaml.safe_load(path.read_text())
    source = policy["sources"]["noise"][index]
    if source.get("kind") != "stochastic":
        raise SystemExit(f"{path}: source {index} is {source.get('kind')!r}, not stochastic")
    return srn.StochasticRanges.from_dict(source.get("ranges"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - not a git checkout
        return "unknown"


def canonical_digest(payload: Any) -> str:
    """SHA-256 of a JSON payload written canonically — the content digest."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def inputs_digest(args: argparse.Namespace, ranges: list[srn.StochasticRanges]) -> str:
    """Digest of everything that determines the bank's CONTENT.

    Anchors and donor ranges by value, every width and count, the stated
    exponent bound, and the source of this script and of the sampler — so a
    code change invalidates the digest and the rebuild is not skipped.
    """
    return canonical_digest(
        {
            "mode": args.mode,
            "n": int(args.n),
            "seed": int(args.seed),
            "strength": float(args.strength),
            "spread": float(args.spread),
            "max_attempts": int(args.max_attempts),
            "rps_scale_min": float(args.rps_scale_min),
            "rps_scale_max": float(args.rps_scale_max),
            "work_rate": WORK_RATE,
            "n_mics": N_MICS,
            "amp_exp_bound": list(AMP_EXP_BOUND),
            "floor_exp_bound": list(FLOOR_EXP_BOUND),
            "anchors": [
                {"fit": str(path), "clip": selector, "sha256": sha256(path)}
                for path, selector in args.anchor
            ],
            "dynamics": [
                {"policy": str(path), "index": index, "ranges": canonical_digest(asdict(r))}
                for (path, index), r in zip(args.dynamics, ranges)
            ],
            "builder": sha256(Path(__file__)),
            "sampler": sha256(Path(rig_sampler.__file__)),
            "renderer": sha256(Path(srn.__file__)),
        }
    )


def slowest_cruise_rps() -> float:
    """The slowest cruise trim any blended drone profile holds, rev/s.

    `drone_profile_range: [0, 1]` blends DREGON's profile into Michael's and the
    common-mode trim is linear in the blend, so the slowest cruise the stream
    holds is the smaller of the two trims.
    """
    return float(
        min(rps_synthesis.DREGON_PROFILE.common.trim, rps_synthesis.MICHAELS_PROFILE.common.trim)
    )


def fastest_cruise_rps() -> float:
    """The fastest cruise trim any blended drone profile holds, rev/s."""
    return float(
        max(rps_synthesis.DREGON_PROFILE.common.trim, rps_synthesis.MICHAELS_PROFILE.common.trim)
    )


def _band_power_db(x: np.ndarray, n: int = 8192) -> np.ndarray:
    """Per-band power in dB on :data:`FOLD_BANDS`, mic- and frame-averaged."""
    xx = np.atleast_2d(np.asarray(x, dtype=np.float64))
    w = np.hanning(n + 1)[:n]
    power = np.zeros(n // 2 + 1)
    for ch in xx:
        frames = np.stack([ch[s : s + n] * w for s in range(0, ch.size - n + 1, n // 2)])
        power += (np.abs(np.fft.rfft(frames, axis=-1)) ** 2).mean(axis=0)
    power /= xx.shape[0]
    freqs = np.fft.rfftfreq(n, 1.0 / WORK_RATE)
    return np.asarray(
        [
            10.0 * np.log10(max(power[(freqs >= lo) & (freqs < hi)].mean(), 1e-300))
            for lo, hi in FOLD_BANDS
        ]
    )


def fold_check(
    entries: list[srn.StochasticParams], *, rps_scale_max: float, count: int = 4
) -> dict[str, Any]:
    """Does this bank emit anything that FOLDED from above the output Nyquist?

    The ladder is sized for the stream's slowest cruise, so at its FASTEST the
    top orders sit above 8 kHz. There is no antialias filter in the training path
    — `synthesize` renders straight at 16 kHz — so the question is whether the
    renderer emits those orders at all. It does not: `build_psd` zeroes the power
    of any order whose centre reaches `nyquist - gamma`
    (`stochastic_rotor_noise.py:1164`, `:1185`) and the `fm` tone bank mutes each
    tone sample-by-sample while its label frequency is past Nyquist (`:1505`,
    `:1518-1519`). This asserts it on the built bank instead of trusting the
    reading: each entry is rendered at a CONSTANT fastest-cruise speed — where
    "ever above Nyquist" and "always above Nyquist" are the same set of orders —
    against a reference whose out-of-band orders are removed from the profile.
    Any difference is content that folded down.
    """
    speed = float(rps_scale_max) * fastest_cruise_rps()
    n = int(2.0 * WORK_RATE)
    kwargs: dict[str, Any] = dict(
        n_mics=N_MICS, n_fft=srn.DEFAULT_N_FFT, normalize_rms=None, line_mode="fm"
    )
    deltas, removed_orders = [], []
    for params in entries[: int(count)]:
        rps = np.full((params.n_rotors, n), speed)
        k = np.arange(1, params.n_harmonics + 1, dtype=np.float64)
        over = k * speed >= WORK_RATE / 2.0
        profile = np.array(params.profile_db, dtype=np.float64)
        profile[:, over] = -400.0
        removed_orders.append(int(over.sum()))
        params = params.with_(amp_rps_ref=speed)
        full, _ = srn.synthesize(params, rps, rng=np.random.default_rng(3), **kwargs)
        trunc, _ = srn.synthesize(
            params.with_(profile_db=profile), rps, rng=np.random.default_rng(3), **kwargs
        )
        deltas.append(_band_power_db(full) - _band_power_db(trunc))
    worst = np.abs(np.stack(deltas)).max(axis=0)
    out = {
        "speed_rps": speed,
        "orders_above_nyquist": removed_orders,
        "bands_hz": [list(b) for b in FOLD_BANDS],
        "worst_abs_db": [float(v) for v in worst],
        "tolerance_db": FOLD_TOLERANCE_DB,
        "entries_checked": len(deltas),
    }
    below = [
        (band, float(value))
        for band, value in zip(FOLD_BANDS, worst)
        if band[1] <= 7500 and value > FOLD_TOLERANCE_DB
    ]
    if below:
        raise SystemExit(
            "fold check FAILED: the bank's ladder leaks folded content below 7.5 kHz — "
            + ", ".join(f"{lo}-{hi} Hz: {v:+.3f} dB" for (lo, hi), v in below)
        )
    return out


def silent_when_stopped(export: dict[str, Any]) -> bool:
    """Does this rig go QUIET as its rotors stop, instead of diverging?

    A stream with `rps: {kind: full_flight}` renders the ground phase at EXACTLY
    zero rotor speed, and the amplitude law is `(rps / ref) ** amp_exp`, so a
    negative exponent is `0 ** negative` — infinite line power, and NaN audio
    once the tone bank scales it. `rig_sampler.check_sample`'s zero-speed
    finiteness guard misses it: it works on the model LTAS, where every line of a
    stopped rotor collapses onto DC and the infinity does not survive the
    scatter. Measured: an accepted spread-3 path draw with `amp_exp = -7.27`
    renders `max |x| = nan` on a trajectory with a stopped section.

    So this is a build-time guard, not a policy: "a stopped rotor is silent" is
    the renderer's own invariant (`build_psd`), and a draw that breaks it is
    redrawn rather than clipped.
    """
    return float(export["amp_exp"]) >= 0.0 and float(export.get("floor_exp", 0.0)) >= 0.0


def clip_exponents(export: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Bound the speed exponents to the fitted range; name what was clipped."""
    out = dict(export)
    clipped: list[str] = []
    for key, (lo, hi) in (("amp_exp", AMP_EXP_BOUND), ("floor_exp", FLOOR_EXP_BOUND)):
        value = float(out.get(key, out["amp_exp"]))
        bounded = float(np.clip(value, lo, hi))
        if bounded != value:
            clipped.append(key)
        out[key] = bounded
    return out, clipped


def path_reference(
    anchors: list[dict[str, Any]], levels: tuple[float, float], t: float
) -> dict[str, Any]:
    """The interpolated point `sample_path` guarded a draw against, rebuilt.

    `sample_path` keeps the mixing coordinate in `_sampler.path.t` but not the
    interpolated export itself, and a clipped draw has to be re-checked against
    the SAME reference the sampler used, not against an endpoint. The rebuild is
    deterministic (no random draw in `interpolate_exports`), so it reproduces
    that reference exactly.
    """
    mid = rig_sampler.interpolate_exports(anchors[0], anchors[1], t, levels=levels)
    mid.pop("_path", None)
    return mid


def entry_params(
    export: dict[str, Any],
    ranges: srn.StochasticRanges,
    rng: np.random.Generator,
    *,
    rates: np.ndarray,
) -> srn.StochasticParams:
    """One bank entry: the stream's per-clip draw with the fitted rig on top."""
    fitted = stage2.params_from_export(export, rates, sample_rate=WORK_RATE, n_mics=N_MICS)
    # The widths and the channel pattern reach the draw through the very
    # `fixed_*` seams the base policy uses, so the per-clip width SPREAD
    # (`shaft_jitter_log_std`) still applies on top, exactly as it does there.
    ranges = replace(
        ranges,
        fixed_gamma0_hz=tuple(np.asarray(fitted.gamma0, dtype=np.float64).tolist()),
        fixed_gamma_slope_hz=tuple(np.asarray(fitted.gamma_slope, dtype=np.float64).tolist()),
        fixed_shaft_jitter_rps=tuple(
            (np.asarray(fitted.gamma_slope, dtype=np.float64) / GAUSS_HWHM).tolist()
        ),
        fixed_mic_gain_db=_as_nested(fitted.fixed_mic_gain_db),
        fixed_mic_floor_db=_as_tuple(fitted.fixed_mic_floor_db),
        fixed_mic_gain_all_db=_as_tuple(fitted.fixed_mic_gain_all_db),
    )
    draw = srn.sample_params(
        rng,
        ranges,
        n_rotors=fitted.n_rotors,
        n_harmonics=fitted.n_harmonics,
        sample_rate=WORK_RATE,
        line_bin_integrate=fitted.line_bin_integrate,
    )
    return draw.with_(**{name: getattr(fitted, name) for name in IDENTITY_FIELDS})


def _as_tuple(value: np.ndarray | None) -> tuple[float, ...] | None:
    return None if value is None else tuple(np.asarray(value, dtype=np.float64).tolist())


def _as_nested(value: np.ndarray | None) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    return tuple(tuple(row) for row in np.asarray(value, dtype=np.float64).tolist())


def guard_stats(samplers: list[dict[str, Any]]) -> dict[str, Any]:
    """Acceptance and guard-firing statistics over every draw's `_sampler`."""
    attempts = [int(s["attempts"]) for s in samplers]
    fired: dict[str, int] = {}
    for s in samplers:
        for failed in s["rejected"]:
            for guard in failed:
                fired[guard] = fired.get(guard, 0) + 1
    measured = {
        key: [float(s["guards"][key]) for s in samplers]
        for key in ("ltas_rms_db", "ltas_shape_rms_db", "ltas_band_max_db", "ltas_level_db")
    }
    return {
        "draws": len(samplers),
        "attempts_total": int(sum(attempts)),
        "attempts_mean": float(np.mean(attempts)),
        "attempts_max": int(max(attempts)),
        "first_try_fraction": float(np.mean([a == 1 for a in attempts])),
        "rejections_by_guard": dict(sorted(fired.items())),
        "accepted": {
            key: {
                "mean": float(np.mean(values)),
                "p95": float(np.percentile(values, 95.0)),
                "max": float(np.max(values)),
            }
            for key, values in measured.items()
        },
        "trend_drop_db_min": float(min(min(s["guards"]["trend_drop_db"]) for s in samplers)),
        "parity_low_db_min": float(min(min(s["guards"]["parity_low_db"]) for s in samplers)),
    }


def build(
    args: argparse.Namespace,
    anchors: list[dict[str, Any]],
    ranges: list[srn.StochasticRanges],
    rates: np.ndarray,
) -> tuple[list[dict[str, Any]], list[srn.StochasticParams], list[int], dict[str, int]]:
    """``(exports, entries, donor index per entry, builder rejection counts)``.

    Each entry owns substream ``i`` of the seed, as `rig_sampler.sample_batch`
    does, so any one entry is reproducible on its own, and the SAME substream
    draws the rig and its per-clip dynamics. A draw the builder refuses advances
    the substream index instead of being patched, which keeps the bank an honest
    draw from the bounded family.
    """
    if args.mode == "neighbourhood":
        # Split evenly; the remainder goes to the first anchors, so a bank is
        # exactly `--n` entries whatever the anchor count.
        share = [args.n // len(anchors)] * len(anchors)
        for i in range(args.n - sum(share)):
            share[i] += 1
    else:
        share = [args.n]
    levels = (
        (rig_sampler.model_level_db(anchors[0]), rig_sampler.model_level_db(anchors[1]))
        if args.mode == "path"
        else (0.0, 0.0)
    )
    exports: list[dict[str, Any]] = []
    entries: list[srn.StochasticParams] = []
    donors: list[int] = []
    rejected = {
        "diverges_when_stopped": 0,
        "clipped_amp_exp": 0,
        "clipped_floor_exp": 0,
        "clipped_then_rejected": 0,
    }
    for arm, count in enumerate(share):
        accepted, substream = 0, 0
        while accepted < count:
            rng = np.random.default_rng([args.seed + arm, substream])
            substream += 1
            if args.mode == "neighbourhood":
                export = rig_sampler.sample_rig(
                    anchors[arm], rng, strength=args.strength, max_attempts=args.max_attempts
                )
                reference, donor = anchors[arm], arm
            else:
                # `t` is left to `sample_path`, which draws it uniform on
                # [0, 1]: the cloud must be as likely to sit at either end as in
                # the middle, so neither real rig is the privileged one. The
                # per-clip DYNAMICS have no path convention — they are not
                # renderable export fields — so a draw takes them from the
                # nearer anchor's rig, which is also the rig it most resembles.
                export = rig_sampler.sample_path(
                    anchors[0],
                    anchors[1],
                    rng,
                    spread=args.spread,
                    max_attempts=args.max_attempts,
                )
                t = float(export["_sampler"]["path"]["t"])
                reference, donor = path_reference(anchors, levels, t), (0 if t < 0.5 else 1)
            if not silent_when_stopped(export):
                rejected["diverges_when_stopped"] += 1
                continue
            export, clipped = clip_exponents(export)
            if clipped:
                for key in clipped:
                    rejected[f"clipped_{key}"] += 1
                # The guards were evaluated on the UNCLIPPED draw, and the
                # exponents move the model level at the anchor's speed, so a
                # clipped draw has to face them again.
                guards = rig_sampler.check_sample(export, reference)
                export["_sampler"]["guards"] = guards
                export["_sampler"]["exponents_clipped"] = clipped
                if not guards["ok"]:
                    rejected["clipped_then_rejected"] += 1
                    continue
            exports.append(export)
            entries.append(entry_params(export, ranges[donor], rng, rates=rates))
            donors.append(donor)
            accepted += 1
    return exports, entries, donors, rejected


def spread_echo(exports: list[dict[str, Any]], entries: list[srn.StochasticParams]) -> None:
    """Print the realised spread of trend slope, rotor gain, gamma and the rest."""
    rows: dict[str, list[float]] = {name: [] for name in SPREAD_ROWS}
    for export, params in zip(exports, entries):
        profile = rig_sampler.effective_profile(export)
        for rotor in range(profile.shape[0]):
            parts = rig_sampler.decompose_profile(profile[rotor])
            rows["rotor gain (dB)"].append(parts.gain)
            rows["trend slope (dB/decade)"].append(parts.slope)
        rows["gamma0 (Hz)"].append(float(np.mean(params.gamma0)))
        rows["gamma_slope (Hz/order)"].append(float(np.mean(params.gamma_slope)))
        rows["shaft jitter (rev/s)"].append(float(np.mean(params.shaft_jitter_rps)))
        rows["amp_exp (dB/dB rps)"].append(float(params.amp_rps_exponent))
        rows["floor_exp (dB/dB rps)"].append(float(params.amp_rps_exponent_floor or 0.0))
        rows["harm_gp std (dB)"].append(float(params.harm_gp_std_db))
        rows["label error (rev/s)"].append(float(np.max(np.abs(params.shaft_offset_rps))))
        rows["k_use (orders)"].append(float(params.n_harmonics))
    print("realised spread over the bank:")
    print(f"  {'quantity':24s} {'mean':>9s} {'std':>8s} {'min':>9s} {'max':>9s}")
    for name, values in rows.items():
        v = np.asarray(values, dtype=np.float64)
        print(f"  {name:24s} {v.mean():9.3f} {v.std():8.3f} {v.min():9.3f} {v.max():9.3f}")


def path_coordinate_stats(exports: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The REALISED distribution of the path's mixing coordinate.

    ``sample_path`` draws ``t`` uniform on [0, 1], but a draw can be refused by
    the guards, and rejection is not uniform in ``t`` — a point halfway between
    two rigs is a different distance from both anchors than an endpoint is. So
    "neither rig is especially likely" is a claim about the ACCEPTED draws, and
    this is the measurement of it: a ten-bin histogram plus a uniformity check
    (the Kolmogorov-Smirnov distance to U(0, 1), which needs no tables to read —
    0.05 at n = 2048 is a 1-in-20 level deviation).
    """
    ts = [float(e["_sampler"]["path"]["t"]) for e in exports if "path" in (e.get("_sampler") or {})]
    if not ts:
        return None
    t = np.sort(np.asarray(ts, dtype=np.float64))
    counts, edges = np.histogram(t, bins=10, range=(0.0, 1.0))
    ecdf = (np.arange(1, t.size + 1)) / t.size
    ks = float(np.max(np.abs(ecdf - t)))
    return {
        "n": int(t.size),
        "bins": [round(float(v), 2) for v in edges],
        "counts": [int(c) for c in counts],
        "mean": float(t.mean()),
        "ks_distance_to_uniform": ks,
        "ks_95pct_threshold": float(1.36 / np.sqrt(t.size)),
    }


SPREAD_ROWS = (
    "rotor gain (dB)",
    "trend slope (dB/decade)",
    "gamma0 (Hz)",
    "gamma_slope (Hz/order)",
    "shaft jitter (rev/s)",
    "amp_exp (dB/dB rps)",
    "floor_exp (dB/dB rps)",
    "harm_gp std (dB)",
    "label error (rev/s)",
    "k_use (orders)",
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("neighbourhood", "path"), required=True)
    ap.add_argument(
        "--anchor",
        type=anchor_spec,
        action="append",
        required=True,
        metavar="FIT.json:CLIP_ID",
        help="one anchor per flag; neighbourhood mode splits the draws evenly over them",
    )
    ap.add_argument(
        "--dynamics",
        type=dynamics_spec,
        action="append",
        required=True,
        metavar="POLICY.yaml:INDEX",
        help=(
            "per-clip draw ranges for each anchor, in anchor order: the stochastic source of "
            "the policy whose dynamics regime this arm must reproduce. Pass one per anchor, "
            "or one for all of them"
        ),
    )
    ap.add_argument("--name", required=True, help="bank file stem under data/rig_banks/")
    ap.add_argument("--n", type=int, default=512, help="entries in the bank (default 512)")
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument(
        "--strength", type=float, default=1.0, help="neighbourhood width (--mode neighbourhood)"
    )
    ap.add_argument(
        "--spread", type=float, default=1.0, help="cloud width around the path (--mode path)"
    )
    ap.add_argument("--max-attempts", type=int, default=16)
    ap.add_argument(
        "--rps-scale-min",
        type=float,
        default=DEFAULT_RPS_SCALE_MIN,
        help="the policy's rps_scale_range lower bound; sizes the order ladder",
    )
    ap.add_argument(
        "--rps-scale-max",
        type=float,
        default=DEFAULT_RPS_SCALE_MAX,
        help="the policy's rps_scale_range upper bound; sets the fold check's speed",
    )
    ap.add_argument("--out", type=Path, default=None, help="override the output path")
    ap.add_argument(
        "--force",
        action="store_true",
        help="rebuild even when the target already matches these inputs",
    )
    args = ap.parse_args(argv)

    if args.mode == "path" and len(args.anchor) != 2:
        raise SystemExit(f"--mode path wants exactly two anchors, got {len(args.anchor)}")
    if len(args.dynamics) == 1:
        args.dynamics = args.dynamics * len(args.anchor)
    if len(args.dynamics) != len(args.anchor):
        raise SystemExit(
            f"{len(args.dynamics)} --dynamics donors for {len(args.anchor)} anchors; "
            "pass one per anchor or exactly one for all"
        )

    anchors = [rig_sampler.load_anchor(path, selector) for path, selector in args.anchor]
    ranges = [donor_ranges(path, index) for path, index in args.dynamics]
    n_rotors = int(np.atleast_2d(np.asarray(anchors[0]["profile_db"])).shape[0])
    min_rps = float(args.rps_scale_min) * slowest_cruise_rps()
    rates = np.full(n_rotors, min_rps, dtype=np.float64)
    digest = inputs_digest(args, ranges)

    out = args.out or BANK_DIR / f"{args.name}.json"
    # IDEMPOTENT BY CONTENT, not by mtime: the banks are gitignored build
    # products and a training job rebuilds them before it trains, so a job that
    # restarts must not pay for a rebuild it does not need. The digest covers
    # every input INCLUDING the source of this script, the sampler and the
    # renderer, so a code change is not silently skipped.
    if out.exists() and not args.force:
        try:
            existing = json.loads(out.read_text()).get("provenance", {})
        except (OSError, ValueError):
            existing = {}
        if existing.get("inputs_digest") == digest:
            print(
                f"{out} already matches these inputs (digest {digest[:16]}), skipping the "
                "rebuild; pass --force to rebuild anyway"
            )
            return 0

    started = time.time()
    exports, entries, donors, rejected = build(args, anchors, ranges, rates)
    for index, params in enumerate(entries):
        # Round-trip every entry HERE: a bank a pool cannot read is a build
        # failure, not a training failure twenty minutes into a job.
        srn.params_from_entry(srn.params_to_entry(params), where=f"entry {index}")
    fold = fold_check(entries, rps_scale_max=args.rps_scale_max)
    serialized = [srn.params_to_entry(params) for params in entries]

    bank = {
        "format": srn.PRESET_BANK_FORMAT,
        "provenance": {
            "built_by": "scripts/_build_rig_bank.py",
            "git_head": git_head(),
            # No build TIMESTAMP on purpose: with the volatile fields out, the
            # file is byte-reproducible from its inputs, so its SHA-256 is
            # itself a check that a rebuild produced the same bank. The wall
            # time is printed instead.
            "inputs_digest": digest,
            "entries_sha256": canonical_digest(serialized),
            "mode": args.mode,
            "n": int(args.n),
            "seed": int(args.seed),
            "strength": float(args.strength) if args.mode == "neighbourhood" else None,
            "spread": float(args.spread) if args.mode == "path" else None,
            "max_attempts": int(args.max_attempts),
            "exponent_bound": {
                "amp_exp": list(AMP_EXP_BOUND),
                "floor_exp": list(FLOOR_EXP_BOUND),
                "measured_from": EXPONENT_BOUND_SOURCE,
                "floor_exp_measured_min": -3.711,
                "note": (
                    "the fitted range over the six measured fits; floor_exp's lower bound is "
                    "raised from the measured -3.711 to 0 because a negative floor exponent "
                    "diverges at zero rotor speed. A clipped draw is re-checked against the "
                    "sampler's guards and redrawn if the clip made it implausible"
                ),
            },
            "sampler_widths": rig_sampler.WIDTHS.as_dict(),
            "anchors": [
                {
                    "fit": str(path),
                    "clip": selector,
                    "sha256": sha256(path),
                    "physical_level": True,
                    "power_scale_folded_db": float(anchor.get("power_scale_folded_db", 0.0)),
                    "dynamics_from": f"{dpath}:{dindex}",
                    "entries": int(sum(1 for d in donors if d == arm)),
                }
                for arm, ((path, selector), anchor, (dpath, dindex)) in enumerate(
                    zip(args.anchor, anchors, args.dynamics)
                )
            ],
            "identity_from_fit": list(IDENTITY_FIELDS)
            + [
                "gamma0 (via fixed_gamma0_hz)",
                "gamma_slope (via fixed_gamma_slope_hz)",
                "shaft_jitter_rps (gamma_slope / sqrt(2 log 2), via fixed_shaft_jitter_rps)",
                "fixed_mic_gain_db",
                "fixed_mic_floor_db",
                "fixed_mic_gain_all_db",
            ],
            "dynamics_from_donor": (
                "every other field of srn.sample_params: the amplitude OU process, the floor "
                "level/tilt drift, the per-mic modulation, the shaft-jitter time constant and "
                "its per-clip log spread, the phase diffusion and the label error"
            ),
            "order_ladder": {
                "rps_scale_min": float(args.rps_scale_min),
                "slowest_cruise_rps": slowest_cruise_rps(),
                "min_rps_vector": rates.tolist(),
                "k_use": sorted({int(p.n_harmonics) for p in entries}),
                "k_max_from_min_rps": int(np.floor((WORK_RATE / 2) / max(min_rps, 1.0))),
                "note": (
                    "k_use = min(k_max_from_min_rps, the fit's own profile length); the fitted "
                    "ladder is the binding cap whenever it is the shorter of the two"
                ),
            },
            "render_units": {
                "sample_rate": WORK_RATE,
                "n_mics": N_MICS,
                "converted_by": "experiments.stochastic_fit.stage2.params_from_export",
            },
            "guards": guard_stats([e["_sampler"] for e in exports]),
            "rejected_by_builder": rejected,
            "nyquist_fold": fold,
            "path_coordinate": path_coordinate_stats(exports),
        },
        "entries": serialized,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bank))
    print(
        f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(entries)} entries, "
        f"{time.time() - started:.1f} s)"
    )
    print(
        f"digests: inputs {digest}\n"
        f"         entries {bank['provenance']['entries_sha256']}\n"
        f"         file    {sha256(out)}"
    )
    guards = bank["provenance"]["guards"]
    print(
        f"guards: {guards['attempts_mean']:.2f} attempts/draw (max {guards['attempts_max']}), "
        f"{100 * guards['first_try_fraction']:.1f}% first try, "
        f"rejections {guards['rejections_by_guard'] or 'none'}"
    )
    print(
        f"exponent bound amp_exp {AMP_EXP_BOUND} / floor_exp {FLOOR_EXP_BOUND}: clipped "
        f"{rejected['clipped_amp_exp']} amp_exp and {rejected['clipped_floor_exp']} floor_exp "
        f"draws, of which {rejected['clipped_then_rejected']} were then refused by the guards "
        f"and redrawn; {rejected['diverges_when_stopped']} redrawn for a divergent speed law"
    )
    print(
        "entries per arm: "
        + ", ".join(
            f"{anchor['clip']} <- {anchor['dynamics_from']}: {anchor['entries']}"
            for anchor in bank["provenance"]["anchors"]
        )
    )
    print(
        f"Nyquist fold at {fold['speed_rps']:.0f} rev/s "
        f"({fold['orders_above_nyquist']} orders above 8 kHz of {entries[0].n_harmonics}), "
        f"worst |delta| per band (dB): "
        + ", ".join(
            f"{lo // 1000 if lo % 1000 == 0 else lo / 1000:g}-{hi / 1000:g}k {v:.3f}"
            for (lo, hi), v in zip(FOLD_BANDS, fold["worst_abs_db"])
        )
    )
    coords = bank["provenance"]["path_coordinate"]
    if coords is not None:
        print(
            f"realised path coordinate t: mean {coords['mean']:.3f}, ten-bin counts "
            f"{coords['counts']}, KS distance to uniform "
            f"{coords['ks_distance_to_uniform']:.4f} against a 95% threshold of "
            f"{coords['ks_95pct_threshold']:.4f}"
        )
    spread_echo(exports, entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
