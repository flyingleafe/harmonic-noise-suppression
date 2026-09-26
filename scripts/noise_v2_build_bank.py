"""Build a PRESET BANK of noise-model-v2 rigs for a training stream.

WHAT IT MAKES. A ``noise-v2-bank/1`` file
(:func:`data_processing.noise_v2_pool.load_preset_bank`) — one entry per
sampled rig, each carrying a COMPLETE inline ``noise-v2-fit/2`` payload for its
cruise regime, either Michael's standby payload or ``null`` for its standby
slot, the trajectory rig it is flown on, and its own draw provenance. A policy
names the file in one line::

    kind: noise_v2
    preset_bank: data/rig_banks/noise_v2_easy_n2048.json

TWO PRESETS, both from :mod:`experiments.noise_model.rig_sampler`:

``easy``
    ``n / 2`` neighbourhood draws around EACH rig's own fit (DREGON room-2
    round 5, Michael's FLY125 round 3), every entry carrying ``traj_rig`` =
    its own rig, so the pool flies each comb on its own rig's fitted flight
    envelope. Michael's entries carry a perturbed standby payload; DREGON has
    no standby fit and its entries carry ``null``.
``hard``
    ``n`` draws from the CLOUD along the CRUISE-to-CRUISE path between the two
    rigs, mixing coordinate ``t ~ U[0, 1]`` on the common order range
    ``k = 1..81`` (DREGON's orders 82-88 are dropped, recorded in the bank's
    provenance). ``traj_rig`` is ``null`` on every entry, so the policy's own
    ``rps`` block — the fitted-trajectory rig hyperprior for this arm — flies
    them. The standby SLOT is a regime policy carried with probability ``t``.

NOISE MODEL V3 (``--generation v3``). The same two presets, the same seed,
strength, widths, guards and entry count, drawn around the round-2 v3 fits
with their static latent part folded into the rig
(:data:`experiments.noise_model.rig_sampler.ANCHORS_V3`). Each entry carries
``noise-v3-fit/1`` payloads in the same ``noise-v2-bank/1`` container; the
file is ``noise_v3_<preset>_n<n>.json`` and the report
``results/noise_v3/rig_sampler/build_<preset>.json``. How every v2 coordinate
maps onto v3 (floor spline, wind, wander carried unperturbed, the trend guard
around a flat-trend anchor) is in the sampler's module docstring.
``--generation v3r3`` is the same v3 construction around the folded round-3b
fits (:data:`experiments.noise_model.rig_sampler.ANCHORS_V3R3`): file
``noise_v3r3_<preset>_n<n>.json``, report
``results/noise_v3r3/rig_sampler/build_<preset>.json``.

REPRODUCIBILITY. One seed (:data:`SEED`), one substream per entry index
(``default_rng([seed, i])``), so a bank is bit-identical at any worker count
and any build order, and a single entry can be re-derived on its own. The
build is IDEMPOTENT: the content digest of everything that determines the bank
(anchor SHA-256s, widths, guards, seed, strength, count) is written into the
file, and a rebuild whose digest matches an existing bank skips the work and
exits 0. ``--force`` rebuilds anyway.

SELF-CHECK. Every bank is re-read from disk through
:class:`data_processing.noise_v2_pool.NoiseV2Pool` and eight of its entries are
rendered for 2 s at 8 microphones; the build FAILS if any render is non-finite
or silent. A bank that cannot be streamed is not written as if it could.

BANKS ARE BUILD PRODUCTS. ``data/rig_banks/`` is gitignored; a training job
rebuilds the bank in-place before ``train.py`` (the submit wrapper calls this
script), which is cheaper and safer than shipping 40 MB of JSON.

USAGE::

    PYTHONPATH=src python scripts/noise_v2_build_bank.py --preset easy
    PYTHONPATH=src python scripts/noise_v2_build_bank.py --preset hard --force
    PYTHONPATH=src python scripts/noise_v2_build_bank.py --preset easy --n 64 \\
        --out /tmp/nv2_easy_64.json
    PYTHONPATH=src python scripts/noise_v2_build_bank.py --preset hard --generation v3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.noise_model.render import regime_blend_weight
from data_processing.noise_v2_pool import NoiseV2Pool, load_preset_bank
from experiments.noise_model import rig_sampler as RS

#: The single seed of every bank of this campaign.
SEED = 20260921

#: Where banks live. Gitignored, like every other build product under ``data/``.
BANK_DIR = Path("data/rig_banks")

#: Draws per bank, per the transfer-pair design: 1024 per rig (easy) and 2048
#: along the path (hard).
DEFAULT_N = 2048

#: Entries the self-check renders, and for how long each window is.
SELF_CHECK_S = 2.0
#: A self-check standby window must keep every rotor at or under this, the
#: upper edge of the regime gate's standby band (the cruise blend weight is 0
#: at 45 rev/s and 1 at 65).
STANDBY_MAX_RPS = 40.0


def default_out(preset: str, n: int, generation: str = "v2") -> Path:
    return BANK_DIR / f"noise_{generation}_{preset}_n{n}.json"


def report_dir(generation: str) -> Path:
    """Where a generation's sidecar build reports live."""
    return Path(f"results/noise_{generation}/rig_sampler")


def chosen_strength(structure: dict[str, Any]) -> tuple[float, str]:
    """The strength the coverage ladder picked, and why."""
    ladder = structure.get("coverage_ladder")
    if not ladder:
        return 2.0, "structure.json carries no coverage ladder; the default 2.0 is used"
    return float(ladder["chosen_strength"]), str(ladder["reason"])


def existing_digest(path: Path) -> str | None:
    """The content digest of a bank already on disk, or ``None``."""
    if not path.is_file():
        return None
    try:
        with path.open() as fh:
            head = fh.read(4096)
        # the provenance block sits at the END of the file; a full parse is the
        # only correct read, and a bank is tens of MB, so only do it when the
        # file at least looks like one
        if '"format"' not in head:
            return None
        return str(json.loads(path.read_text())["provenance"]["digest"])
    except (json.JSONDecodeError, KeyError, OSError):
        return None


def _check_indices(entries: Any, preset: str) -> list[tuple[int, str]]:
    """Which entries the self-check renders, and why each one is in the list.

    NOT ``entries[:8]`` through one pool: a uniform draw can repeat an entry
    and an easy bank's first eight are all one rig, so such a check would
    exercise half the bank's structure at best. The selection spans both rigs
    and both regime policies for an easy bank, and the whole mixing coordinate
    for a hard one.
    """
    if preset == "easy":
        dregon = [i for i, e in enumerate(entries) if e.traj_rig == "dregon"]
        michaels = [i for i, e in enumerate(entries) if e.traj_rig == "michaels"]
        if len(dregon) < 4 or len(michaels) < 4:
            raise SystemExit(f"self-check FAILED: {len(dregon)} DREGON / {len(michaels)} Michael's")
        picks = [dregon[i] for i in _spread(len(dregon), 4)]
        picks += [michaels[i] for i in _spread(len(michaels), 4)]
        return [(i, "cruise-only" if entries[i].standby is None else "per-regime") for i in picks]
    order = sorted(range(len(entries)), key=lambda i: _t_of(entries[i]))
    picks = [order[i] for i in _spread(len(order), 8)]
    with_standby = [i for i in order if entries[i].standby is not None]
    for extra in with_standby[:2]:
        if extra not in picks:
            picks[-1] = extra
            picks = picks[:-1] + [extra]
    return [(i, "per-regime" if entries[i].standby is not None else "cruise-only") for i in picks]


def _t_of(entry: Any) -> float:
    """One entry's recorded mixing coordinate."""
    return float((entry.provenance or {})["t"])


def _spread(n: int, k: int) -> list[int]:
    """``k`` indices spread evenly over ``range(n)``, endpoints included."""
    return [int(round(v)) for v in np.linspace(0, n - 1, k)]


def self_check(path: Path) -> dict[str, Any]:
    """Load the written bank as a stream source and render CHOSEN entries.

    Each entry is rendered through its OWN one-entry pool, so the entry that
    was checked is the entry that was named. A per-regime entry is rendered a
    second time on a STANDBY-speed window (every rotor at or under 40 rev/s),
    which is the only way the two-regime composition — the smoothstep blend of
    the standby and cruise payloads — is exercised at all.
    """
    entries = load_preset_bank(path)
    preset = "easy" if entries[0].traj_rig is not None else "hard"
    rows: list[dict[str, Any]] = []
    for index, kind in _check_indices(entries, preset):
        entry = entries[index]
        row: dict[str, Any] = {"index": index, "name": entry.name, "regime_policy": kind}
        if preset == "hard":
            row["t"] = round(_t_of(entry), 4)
        row["cruise_window"] = _render_one(entry, scale=(1.0, 1.0), tag="cruise")
        if entry.standby is not None:
            row["standby_window"] = _render_one(entry, scale=(0.35, 0.35), tag="standby")
        rows.append(row)
    rms = [r["cruise_window"]["rms"] for r in rows]
    return {
        "n_entries_total": len(entries),
        "n_rendered": len(rows),
        "duration_s": SELF_CHECK_S,
        "rendered": rows,
        "rms_min": float(np.min(rms)),
        "rms_max": float(np.max(rms)),
        "rms_median": float(np.median(rms)),
        "n_standby_windows": int(sum(1 for r in rows if "standby_window" in r)),
        "per_regime_entries": int(sum(1 for e in entries if e.standby is not None)),
        "traj_rigs": sorted({str(e.traj_rig) for e in entries}),
    }


def _render_one(entry: Any, *, scale: tuple[float, float], tag: str) -> dict[str, Any]:
    """One window of ONE entry through its own pool, with the guards asserted.

    ``scale`` is the pool's own ``rps_scale_range``, the per-window multiplier
    on the whole trajectory; 0.35 puts a cruise trim into the 20-45 rev/s
    standby band. Windows are drawn until one lands in the intended speed band
    (a full flight also visits the ground, where every rotor is stopped), and a
    standby check that never finds one FAILS rather than passing quietly.
    """
    pool = NoiseV2Pool(
        sample_rate=16000,
        duration_s=SELF_CHECK_S,
        n_mics=8,
        n_rotors=4,
        entries=(entry,),
        rps_kind="full_flight",
        rps_scale_range=scale,
        render_reuse=1,
        seed=SEED,
    )
    rng = np.random.default_rng(SEED)
    want_standby = tag == "standby"
    for _ in range(32):
        audio, rps, got = pool.render(rng, SELF_CHECK_S)
        peak = float(np.max(rps))
        in_band = (5.0 <= peak <= STANDBY_MAX_RPS) if want_standby else (peak >= 45.0)
        if not np.all(np.isfinite(audio)):
            raise SystemExit(f"self-check FAILED: {got.name} ({tag}) rendered non-finite audio")
        if audio.shape != (8, int(16000 * SELF_CHECK_S)) or rps.shape[0] != 4:
            raise SystemExit(
                f"self-check FAILED: {got.name} ({tag}) rendered {audio.shape} / "
                f"{rps.shape[0]} rotors"
            )
        value = float(np.sqrt(np.mean(np.square(audio.astype(np.float64)))))
        if in_band:
            if not (value > 0.0):
                raise SystemExit(f"self-check FAILED: {got.name} ({tag}) rendered silence")
            weight = float(
                np.mean(regime_blend_weight(np.asarray(rps, dtype=np.float64), sr=16000))
            )
            return {
                "rms": value,
                "rps_min": float(np.min(rps)),
                "rps_max": peak,
                "cruise_blend_weight_mean": weight,
            }
    raise SystemExit(
        f"self-check FAILED: {entry.name} ({tag}) found no window in the intended speed band "
        "in 32 draws"
    )


def realised_coverage(
    entries: list[dict[str, Any]], stats: dict[str, Any], structure: dict[str, Any]
) -> dict[str, Any]:
    """Does the BUILT bank's cloud bracket each rig's real windows?

    The strength ladder measures this on a small sample; the built bank has
    more draws, so its min-to-max envelope can only be wider. Measured here on
    the real thing, per rig, from the expected-periodogram band curves the
    guards already computed.
    """
    bands = np.asarray(stats["bands_db"], dtype=np.float64)
    out: dict[str, Any] = {}
    for rig in ("dregon", "michaels"):
        centres, levels = RS.real_bands_of(structure, rig)
        sel = (
            bands
            if entries[0]["traj_rig"] is None
            else bands[[i for i, e in enumerate(entries) if e["provenance"]["anchor"] == rig]]
        )
        cov = RS.coverage(sel, levels, centres)
        out[rig] = {k: v for k, v in cov.items() if k != "band_centres_hz"}
    return out


def report(
    payload: dict[str, Any],
    path: Path,
    check: dict[str, Any],
    cov: dict[str, Any],
    wall_s: float,
    n_workers: int,
) -> None:
    prov = payload["provenance"]
    stats = prov["statistics"]
    print(f"\n{path}  ({path.stat().st_size / 1e6:.1f} MB, sha256 {RS.sha256(path)[:16]})")
    print(
        f"  preset {prov['preset']} / mode {prov['mode']}, strength {prov['strength']:g}, "
        f"{prov['n']} entries, seed {prov['seed']}"
    )
    print(f"  digest {prov['digest'][:16]}")
    print(
        f"  acceptance: {stats['attempts_mean']:.3f} attempts per entry, "
        f"{100 * stats['first_try_frac']:.1f}% first try, "
        f"{stats['rejected_draws']} draws rejected "
        f"({100 * stats['rejection_rate']:.1f}%)"
    )
    print(f"  guards fired: {stats['guards_fired'] or 'none'}")
    if "ks_uniform" in stats:
        ks = stats["ks_uniform"]
        print(
            f"  mixing coordinate: mean {ks['mean']:.3f}, KS {ks['ks']:.4f} against the 95% "
            f"threshold {ks['threshold_95']:.4f} ({'passes' if ks['passes'] else 'FAILS'}); "
            f"deciles {stats['t_decile_counts']}"
        )
        print(f"  standby slot carried by {100 * stats['standby_carried_frac']:.1f}% of entries")
    else:
        print(f"  per rig: {stats['per_rig']}")
    if prov["orders_dropped"]:
        drop = prov["orders_dropped"]
        print(f"  orders dropped: dregon {drop['dregon']}, michaels {drop['michaels']}")
    print(
        "  coverage of the real windows: "
        + ", ".join(
            f"{rig} {100 * row['above_300hz']:.1f}% above 300 Hz "
            f"({100 * row['all_bands']:.1f}% all bands)"
            for rig, row in cov.items()
        )
    )
    print(
        f"  self-check: {check['n_rendered']} entries rendered one at a time for "
        f"{check['duration_s']:g} s, {check['n_standby_windows']} of them also on a "
        f"standby-speed window; cruise RMS {check['rms_min']:.4f}-{check['rms_max']:.4f}; "
        f"{check['per_regime_entries']} per-regime entries in the bank, traj rigs "
        f"{check['traj_rigs']}"
    )
    for row in check["rendered"]:
        tail = (
            ""
            if "standby_window" not in row
            else (
                f" | standby {row['standby_window']['rps_min']:.1f}-"
                f"{row['standby_window']['rps_max']:.1f} rev/s, blend "
                f"{row['standby_window']['cruise_blend_weight_mean']:.2f}, RMS "
                f"{row['standby_window']['rms']:.4f}"
            )
        )
        t = "" if "t" not in row else f" t={row['t']:.3f}"
        print(
            f"    [{row['index']:5d}] {row['name']}{t} {row['regime_policy']}: cruise "
            f"{row['cruise_window']['rps_min']:.1f}-{row['cruise_window']['rps_max']:.1f} rev/s, "
            f"RMS {row['cruise_window']['rms']:.4f}{tail}"
        )
    print(f"  wall {wall_s:.1f} s ({n_workers} workers)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--preset", required=True, choices=RS.PRESETS)
    ap.add_argument(
        "--generation",
        default="v2",
        choices=tuple(RS.GENERATIONS),
        help="model generation of the anchors (default v2; v3 / v3r3 = the folded round-2 / round-3b v3 fits)",
    )
    ap.add_argument("--n", type=int, default=DEFAULT_N, help=f"draws (default {DEFAULT_N})")
    ap.add_argument("--out", default=None, help="output path (default data/rig_banks/...)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--strength",
        type=float,
        default=None,
        help="neighbourhood width / path spread (default: structure.json's coverage ladder)",
    )
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-attempts", type=int, default=16)
    ap.add_argument("--force", action="store_true", help="rebuild even if the digest matches")
    args = ap.parse_args(argv)

    structure = RS.load_fit(RS.STRUCTURE_PATH)
    strength, why = chosen_strength(structure)
    if args.strength is not None:
        strength, why = float(args.strength), "given on the command line"
    spec = RS.BankSpec(
        preset=args.preset,
        n=int(args.n),
        seed=int(args.seed),
        strength=float(strength),
        ltas_tol_db=RS.default_tolerances(structure),
        widths=RS.WIDTHS.as_dict(),
        max_attempts=int(args.max_attempts),
        generation=args.generation,
    )
    out = Path(args.out) if args.out else default_out(args.preset, int(args.n), args.generation)
    out.parent.mkdir(parents=True, exist_ok=True)

    have = existing_digest(out)
    if have is not None and have == spec.digest and not args.force:
        print(f"{out} is already the bank this spec describes (digest {have[:16]}); skipping")
        # A skip still VERIFIES: the digest says the file would be rebuilt
        # identically, not that it is loadable and renderable today. The
        # check is a dozen renders and it is what a training job's pre-flight
        # actually needs.
        check = self_check(out)
        print(
            f"  self-check on the existing bank: {check['n_rendered']} entries rendered one at "
            f"a time, {check['n_standby_windows']} also on a standby-speed window, cruise RMS "
            f"{check['rms_min']:.4f}-{check['rms_max']:.4f}"
        )
        report_path = report_dir(args.generation) / f"build_{args.preset}.json"
        if report_path.is_file():
            row = json.loads(report_path.read_text())
            row["self_check"] = check
            row["self_checked_at"] = datetime.now(UTC).isoformat(timespec="seconds")
            report_path.write_text(json.dumps(row, indent=2) + "\n")
            print(f"  refreshed {report_path}")
        return 0
    if have is not None:
        print(f"rebuilding {out}: digest {have[:16]} -> {spec.digest[:16]}")

    print(
        f"building the {args.generation} {args.preset} bank: {spec.n} draws at strength "
        f"{strength:g} ({why}), seed {spec.seed}"
    )
    started = time.time()
    entries, stats = RS.build_entries(spec, workers=args.workers, log=print)
    payload = RS.bank_payload(entries, spec, stats)
    # written ONCE: the self-check reads the file back, and anything it learns
    # goes to the sidecar report, so the bank's bytes are a function of its
    # inputs and of nothing else.
    out.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    check = self_check(out)
    cov = realised_coverage(entries, stats, structure)
    wall = time.time() - started
    report(payload, out, check, cov, wall, stats["n_workers"])

    report_path = report_dir(args.generation) / f"build_{args.preset}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "bank": str(out),
                "sha256": RS.sha256(out),
                "size_bytes": int(out.stat().st_size),
                "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "git": RS.git_head(),
                "wall_s": wall,
                "n_workers": stats["n_workers"],
                "strength": float(strength),
                "strength_reason": why,
                "digest": spec.digest,
                "code": RS.code_digest(),
                "provenance": payload["provenance"],
                "statistics": {k: v for k, v in stats.items() if k != "bands_db"},
                "self_check": check,
                "coverage": cov,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"  report {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
