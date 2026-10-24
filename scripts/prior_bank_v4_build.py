"""Build a ``noise-prior-bank-v4/1`` bank: the prior_v1 rigs converted to noise model v4.

    PYTHONPATH=src python scripts/prior_bank_v4_build.py --workers 16

Reads the pinned v1 bank (``experiments.noise_model.prior_v4.SOURCE_BANK``,
through ``noise_v2_pool.load_preset_bank``), computes the bank's one SC wind
shape (:func:`prior_v4.wind_shape_block`), converts every rig
(:func:`prior_v4.convert_prior_v1`, per-rig stream ``SeedSequence([seed, i])``)
in ``--workers`` processes, reads every payload back through
``V4Params.from_payload``, and writes the bank with
:func:`data_processing.noise_model.prior_bank_v4.write_prior_bank_v4`
(default ``data/rig_banks/prior_v4_n<N>.npz``). The report — counts of motor
lines, the ``alpha_motor`` draw, the wind share and levels, the pedestal split,
floor and level ranges, a float16 round trip of a sample of rigs — goes to
``results/prior_rigs/<stem>.report.json`` and into the bank's ``meta.header``.
Publishing to dload is a separate step.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

_BANK: Any = None
_SHAPE: dict[str, Any] | None = None


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _init(source: str, shape: dict[str, Any]) -> None:
    global _BANK, _SHAPE
    from data_processing.noise_v2_pool import load_preset_bank

    _BANK, _SHAPE = load_preset_bank(source), shape


def _convert(job: tuple[str, int, int, int]) -> list[dict[str, Any]]:
    """Convert rigs ``lo .. hi - 1`` and read each back through the v4 reader."""
    source, seed, lo, hi = job
    from data_processing.noise_model.v4 import V4Params
    from experiments.noise_model.prior_v4 import convert_prior_v1

    assert _BANK is not None and _SHAPE is not None
    out = []
    for i in range(lo, hi):
        pl = convert_prior_v1(_BANK, i, shape=_SHAPE, seed=seed, source_bank=source)
        V4Params.from_payload(pl)
        out.append(pl)
    return out


def _q(x: Any) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64).ravel()
    if a.size == 0:
        return {"n": 0}
    qs = np.quantile(a, [0.0, 0.05, 0.5, 0.95, 1.0])
    return dict(
        n=int(a.size),
        mean=float(a.mean()),
        sd=float(a.std()),
        min=float(qs[0]),
        p05=float(qs[1]),
        p50=float(qs[2]),
        p95=float(qs[3]),
        max=float(qs[4]),
    )


def _stats(payloads: list[dict[str, Any]], bank: Any) -> dict[str, Any]:
    """The report's counts and distributions over the converted rigs."""
    from data_processing.noise_model.prior_bank_v4 import SILENT_LINE_DB
    from data_processing.noise_model.v4 import WIND_OFF_DB
    from experiments.noise_model import v4_priors as VP
    from experiments.noise_model.prior_v4 import floor_band_db

    P = [pl["params"] for pl in payloads]
    idx = np.asarray([pl["_prior_v4"]["source_index"] for pl in payloads])
    boost = np.asarray(bank.arrays["motor_boost_db"], dtype=np.float64)[idx]
    motor = [np.asarray(p["profile_motor_db"]) for p in P]
    live = np.asarray([bool(np.any(m > SILENT_LINE_DB)) for m in motor])
    n_orders = np.asarray([len(p["motor_orders"]) for p in P])
    alpha = np.asarray([p["alpha_motor"] for p in P])
    s2a = np.stack([p["pedestal"]["sigma2_amp"] for p in P])
    s2p = np.stack([p["pedestal"]["sigma2_phase"] for p in P])
    K = s2a.shape[1]
    k = np.arange(1, K + 1)
    groups = []
    for lo, hi, _med, cap in VP.AMP_GROUPS:
        sel = (k >= lo) & (k <= min(hi, K))
        if not np.any(sel):
            continue
        a, ph = s2a[:, sel], s2p[:, sel]
        groups.append(
            dict(
                orders=[int(lo), int(min(hi, K))],
                cap=float(cap),
                capped_frac=float(np.mean(ph > 0.0)),
                sigma2_total=_q(a + ph),
                sigma2_amp=_q(a),
                sigma2_phase=_q(ph),
            )
        )
    winds = [p["wind"] for p in P]
    has = np.asarray([w is not None for w in winds])
    levels = np.concatenate(
        [np.asarray(w["level_db"]) for w in winds if w is not None] or [np.zeros(0)]
    )
    on = levels > WIND_OFF_DB
    with_wind = [i for i, w in enumerate(winds) if w is not None]
    ref = np.asarray(
        [floor_band_db(payloads[i], P[i]["wind"]["shape"]["freqs_hz"]) for i in with_wind]
    )
    excess = np.concatenate(
        [np.asarray(P[i]["wind"]["level_db"]) - r for i, r in zip(with_wind, ref, strict=True)]
        or [np.zeros(0)]
    )
    aero = np.stack([np.asarray(p["profile_aero_db"]) for p in P])
    live_motor = np.concatenate([m[m > SILENT_LINE_DB] for m in motor] or [np.zeros(0)])
    return dict(
        n=len(payloads),
        motor_lines=dict(
            rigs_with_motor_lines=int(live.sum()),
            rigs_with_motor_lines_frac=float(live.mean()),
            rigs_silent_motor=int((~live).sum()),
            v1_boost_positive=int((boost > 0.0).sum()),
            orders_per_rig=_q(n_orders),
            live_lines=int(sum(int(np.sum(m > SILENT_LINE_DB)) for m in motor)),
            profile_motor_db=_q(live_motor),
            v1_motor_boost_db=_q(boost),
        ),
        alpha_motor=dict(
            **_q(alpha),
            abs_max=float(np.abs(alpha).max()),
            frac_abs_gt_2=float(np.mean(np.abs(alpha) > 2.0)),
            prior=dict(normal=list(VP.ALPHA_MOTOR), bound=VP.ALPHA_MOTOR_MAX),
        ),
        alpha_aero=_q([p["alpha_aero"] for p in P]),
        wind=dict(
            rigs_with_wind=int(has.sum()),
            rigs_with_wind_frac=float(has.mean()),
            rigs_with_wind_all_capsules_off=int(
                sum(
                    1
                    for w in winds
                    if w is not None and np.all(np.asarray(w["level_db"]) <= WIND_OFF_DB)
                )
            ),
            capsules_on_frac_of_windy=float(on.mean()) if on.size else 0.0,
            capsules_on_frac_of_all=float(on.sum() / (len(P) * int(payloads[0]["n_mics"]))),
            level_db_on=_q(levels[on]),
            excess_over_floor_db_on=_q(excess[excess > -100.0]),
            floor_20_100_db=_q(ref),
        ),
        pedestal=dict(
            capped_frac_all=float(np.mean(s2p > 0.0)),
            sigma2_phase_over_v4_max=int(np.sum(s2p > VP.PHASE_SIGMA2_MAX)),
            phase_sigma2_max_v4=VP.PHASE_SIGMA2_MAX,
            by_group=groups,
        ),
        shaft=dict(
            sigma_nu_rad_s=_q([2.0 * np.pi * 80.0 * p["shaft"]["eps"] for p in P]),
            lam_per_s=_q([p["shaft"]["lam_ref_per_s"] for p in P]),
        ),
        floor=dict(
            mean_db=_q([p["floor"]["mean_db"] for p in P]),
            shape_sd_db=_q([p["floor"]["shape_sd_db"] for p in P]),
            b0=_q([p["floor"]["b0"] for p in P]),
            static_rel=_q([p["floor"]["static_rel"] for p in P]),
        ),
        levels=dict(
            profile_aero_db=_q(aero),
            profile_aero_max_db_per_rig=_q(aero.max(axis=(1, 2))),
        ),
    )


def _round_trip(path: Path, payloads: list[dict[str, Any]], n: int = 64) -> dict[str, Any]:
    """Largest float16 deviations of a sample of rigs read back from the bank."""
    from data_processing.noise_model.prior_bank_v4 import PriorBankV4

    bank = PriorBankV4(path)
    pick = np.linspace(0, len(payloads) - 1, min(n, len(payloads))).astype(int)
    worst: dict[str, float] = {}

    def upd(key: str, a: Any, b: Any, rel: bool = False) -> None:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        d = np.abs(a - b) / (np.abs(b) + 1e-12) if rel else np.abs(a - b)
        worst[key] = max(worst.get(key, 0.0), float(d.max()) if d.size else 0.0)

    for i in pick:
        q, p = bank.payload(int(i))["params"], payloads[int(i)]["params"]
        upd("profile_aero_db", q["profile_aero_db"], p["profile_aero_db"])
        upd("profile_motor_db", q["profile_motor_db"], p["profile_motor_db"])
        for key in ("sigma2_amp", "sigma2_phase", "gamma_amp_hz"):
            upd(f"pedestal.{key}_rel", q["pedestal"][key], p["pedestal"][key], rel=True)
        if p["wind"] is not None:
            upd("wind.level_db", q["wind"]["level_db"], p["wind"]["level_db"])
        if q["motor_orders"] != p["motor_orders"]:
            raise RuntimeError(f"rig {i}: motor orders differ after the round trip")
    return dict(rigs=int(pick.size), max_abs_dev=worst)


def _args() -> argparse.Namespace:
    from experiments.noise_model.prior_v4 import SEED, SOURCE_BANK

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--source", default=SOURCE_BANK, help="the v1 bank (dload URI or path)")
    ap.add_argument("--n", type=int, default=None, help="convert the first N rigs only")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    a = _args()
    from data_processing.noise_model.prior_bank import PriorBank
    from data_processing.noise_model.prior_bank_v4 import write_prior_bank_v4
    from data_processing.noise_v2_pool import load_preset_bank
    from experiments.noise_model.prior_v4 import RULES, wind_shape_block

    t0 = time.time()
    bank = load_preset_bank(a.source)
    if not isinstance(bank, PriorBank):
        raise SystemExit(f"{a.source} is not a noise-prior-bank/1 file")
    n = len(bank) if a.n is None else min(int(a.n), len(bank))
    out = a.out or Path(f"data/rig_banks/prior_v4_n{n}.npz")
    t_shape = time.time()
    shape = wind_shape_block()
    t_shape = time.time() - t_shape
    jobs = [(a.source, a.seed, lo, min(lo + a.chunk, n)) for lo in range(0, n, a.chunk)]
    t_conv = time.time()
    with mp.get_context("spawn").Pool(
        a.workers, initializer=_init, initargs=(a.source, shape)
    ) as pool:
        payloads = [pl for part in pool.imap(_convert, jobs) for pl in part]
    t_conv = time.time() - t_conv
    print(f"{len(payloads)} rigs converted in {t_conv:.1f} s ({a.workers} workers)", flush=True)
    stats = _stats(payloads, bank)
    header: dict[str, Any] = dict(
        builder="scripts/prior_bank_v4_build.py",
        git=_git_sha(),
        host=platform.node(),
        built_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        source_bank=a.source,
        source_n=len(bank),
        seed=a.seed,
        workers=a.workers,
        n=len(payloads),
        rules=RULES,
        stats=stats,
    )
    t_write = time.time()
    path = write_prior_bank_v4(out, payloads, header)
    t_write = time.time() - t_write
    header["round_trip"] = _round_trip(path, payloads)
    header["seconds"] = dict(
        shape=round(t_shape, 1),
        convert=round(t_conv, 1),
        write=round(t_write, 1),
        total=round(time.time() - t0, 1),
    )
    header["bank"] = dict(path=str(path), mb=round(path.stat().st_size / 1e6, 2))
    report = Path("results/prior_rigs") / f"{path.stem}.report.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(header, indent=1, default=float))
    print(
        f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB) and {report} "
        f"in {header['seconds']['total']:.0f} s",
        flush=True,
    )


if __name__ == "__main__":
    main()
