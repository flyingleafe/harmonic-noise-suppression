"""A compact bank of NEIGHBOURHOOD rigs in noise model v4 (``noise-nbhd-bank-v4/1``): one ``.npz``.

The easy training setup (Dmitrii, 2026-10-26; ``docs/experiments/rp-synthetic.md``)
draws drones AROUND the fitted rigs: each draw is a perturbation of one base
fit (``experiments.noise_model.sectors_v4.perturb_fit``), a ``noise-v4-fit/3``
payload or a ``standby``/``cruise`` pair, flown on its base's own trajectory
law (``traj_rig``). A fitted payload is large (the DREGON bench fit is 0.7 MB of
JSON, most of it the mic response and the fit's own records) and a
perturbation moves only a few fields, so the bank stores every base ONCE (in
the JSON ``meta``) and, per draw, only :data:`PATCH_FIELDS` as float64 arrays:
a read draw is the written one exactly.

:class:`NbhdBankV4` is a lazy ``Sequence`` of ``NoiseV2Entry`` read by
``noise_v2_pool.load_preset_bank`` on the ``meta.format`` tag. Built by
``scripts/easy_bank_v4_build.py``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "DROPPED_FIELDS",
    "NBHD_BANK_V4_FORMAT",
    "PATCH_FIELDS",
    "NbhdBankV4",
    "write_nbhd_bank_v4",
]

NBHD_BANK_V4_FORMAT = "noise-nbhd-bank-v4/1"

#: Every ``params`` field a perturbation sets, as a path; a field a base lacks
#: (``profile_motor_db`` of a rig without motor lines) is not stored for it.
PATCH_FIELDS: tuple[tuple[str, ...], ...] = (
    ("profile_aero_db",),
    ("profile_motor_db",),
    ("alpha_aero",),
    ("alpha_motor",),
    ("floor", "shape_z"),
    ("floor", "b0"),
    ("floor", "static_rel"),
    ("pedestal", "sigma2_amp"),
    ("pedestal", "sigma2_phase"),
    ("pedestal", "width_phase_hz"),
    ("pedestal", "gamma_phase_hz"),
)
#: ``params`` fields a perturbation removes: derived ones it would leave stale.
DROPPED_FIELDS: tuple[tuple[str, ...], ...] = (("hierarchy",), ("floor", "ctrl_db"))
#: The comb tables a pool's ``comb_offset_db`` moves.
_COMB = ("profile_aero_db", "profile_motor_db")


def _get(params: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = params
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def _key(base: int, regime: str, path: tuple[str, ...]) -> str:
    return f"b{base}/{regime}/{'.'.join(path)}"


def write_nbhd_bank_v4(
    path: str | Path,
    bases: Sequence[Mapping[str, Any]],
    draws: Iterable[Mapping[str, Any]],
    header: dict[str, Any],
) -> Path:
    """Write a neighbourhood bank.

    ``bases``: ``{"name", "traj_rig", "regimes": {regime: payload}}`` with
    regimes ``{"single"}`` or ``{"standby", "cruise"}``. ``draws``:
    ``{"base": index, "name", "regimes": {regime: payload}}``, each payload a
    perturbation of its base that differs from it only in
    :data:`PATCH_FIELDS` and :data:`DROPPED_FIELDS` (anything else is refused,
    since the bank could not give it back), carrying its ``perturbation``
    provenance. ``draws`` is read once (a generator keeps one payload in
    memory at a time). ``header`` is free provenance."""
    for b in bases:
        if set(b["regimes"]) not in ({"single"}, {"standby", "cruise"}):
            raise ValueError(f"base {b['name']}: regimes must be single or standby+cruise")
    rows: dict[str, list[np.ndarray]] = {}
    count = [0] * len(bases)
    index: list[dict[str, Any]] = []
    for d in draws:
        b = int(d["base"])
        for regime, fit in bases[b]["regimes"].items():
            bp, dp = fit["params"], d["regimes"][regime]["params"]
            _check_patch_only(bp, dp, where=f"{d['name']}/{regime}")
            for field in PATCH_FIELDS:
                if _get(bp, field) is not None:
                    rows.setdefault(_key(b, regime, field), []).append(
                        np.asarray(_get(dp, field), np.float64)
                    )
        index.append(
            {
                "name": str(d["name"]),
                "base": b,
                "row": count[b],
                "perturbation": next(iter(d["regimes"].values())).get("perturbation"),
            }
        )
        count[b] += 1
    if not index:
        raise ValueError("a bank needs at least one draw")
    meta = {
        "format": NBHD_BANK_V4_FORMAT,
        "n": len(index),
        "bases": [
            {"name": str(b["name"]), "traj_rig": b.get("traj_rig"), "regimes": dict(b["regimes"])}
            for b in bases
        ],
        "draws": index,
        "header": header,
    }
    arrays = {k: np.stack(v) for k, v in rows.items()}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), meta=np.array(json.dumps(meta)), **arrays)  # type: ignore[arg-type]
    return out


def _check_patch_only(base: Mapping[str, Any], drawn: Mapping[str, Any], *, where: str) -> None:
    """Refuse a draw that moved anything outside the patch and dropped fields."""
    skip = {f for f in PATCH_FIELDS} | {f for f in DROPPED_FIELDS}

    def walk(a: Any, b: Any, path: tuple[str, ...]) -> None:
        if path in skip:
            return
        if isinstance(a, Mapping) and isinstance(b, Mapping):
            for k in set(a) | set(b):
                if (*path, k) in skip:
                    continue
                if k not in a or k not in b:
                    raise ValueError(f"{where}: field {'.'.join((*path, k))} added or removed")
                walk(a[k], b[k], (*path, k))
        elif a != b:
            raise ValueError(f"{where}: field {'.'.join(path)} moved outside the bank's patch")

    walk(base, drawn, ())


class NbhdBankV4(Sequence[Any]):
    """The draws of a ``noise-nbhd-bank-v4/1`` file as a lazy sequence of
    ``NoiseV2Entry`` (``cruise`` the payload, or the ``standby``/``cruise``
    pair; ``traj_rig`` the base's). ``comb_offset_db`` is added to every line,
    aerodynamic and motor, when a payload is assembled."""

    def __init__(self, path: str | Path, *, comb_offset_db: float = 0.0):
        self.path = Path(path)
        with np.load(self.path) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("format") != NBHD_BANK_V4_FORMAT:
                raise ValueError(
                    f"{self.path}: expected format {NBHD_BANK_V4_FORMAT!r}, got {meta.get('format')!r}"
                )
            self.arrays = {k: z[k] for k in z.files if k != "meta"}
        self.meta = meta
        self.bases: list[dict[str, Any]] = meta["bases"]
        self.draws: list[dict[str, Any]] = meta["draws"]
        self.header: dict[str, Any] = meta["header"]
        self.n = int(meta["n"])
        self.comb_offset_db = float(comb_offset_db)
        if len(self.draws) != self.n:
            raise ValueError(f"{self.path}: {len(self.draws)} draws, header says {self.n}")

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

    def with_offset(self, comb_offset_db: float) -> NbhdBankV4:
        other = NbhdBankV4.__new__(NbhdBankV4)
        other.__dict__.update(self.__dict__)
        other.comb_offset_db = float(comb_offset_db)
        return other

    @property
    def traj_rigs(self) -> tuple[str, ...]:
        return tuple(sorted({b["traj_rig"] for b in self.bases if b.get("traj_rig")}))

    def probe(self) -> list[Any]:
        """The first draw of every base: the entries a pool validates."""
        first: dict[int, int] = {}
        for i, d in enumerate(self.draws):
            first.setdefault(int(d["base"]), i)
        return [self[i] for i in sorted(first.values())]

    def regimes(self, i: int) -> dict[str, dict[str, Any]]:
        """Draw ``i``'s payloads by regime, ``perturbation`` provenance included."""
        i = int(i)
        if not 0 <= i < self.n:
            raise IndexError(i)
        d = self.draws[i]
        b, row = int(d["base"]), int(d["row"])
        out: dict[str, dict[str, Any]] = {}
        for regime, fit in self.bases[b]["regimes"].items():
            q = dict(fit)
            p = dict(fit["params"])
            p["floor"] = dict(p["floor"])
            p["pedestal"] = dict(p["pedestal"])
            for path in DROPPED_FIELDS:
                parent = p if len(path) == 1 else p[path[0]]
                parent.pop(path[-1], None)
            for field in PATCH_FIELDS:
                key = _key(b, regime, field)
                if key not in self.arrays:
                    continue
                value = self.arrays[key][row]
                if field[0] in _COMB:
                    value = value + self.comb_offset_db
                value = value.tolist() if value.ndim else float(value)
                if len(field) == 1:
                    p[field[0]] = value
                else:
                    p[field[0]][field[1]] = value
            q["params"] = p
            q["perturbation"] = d["perturbation"]
            out[regime] = q
        return out

    def __getitem__(self, i: Any) -> Any:
        from data_processing.noise_v2_pool import NoiseV2Entry

        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self.n))]
        i = int(i)
        if i < 0:
            i += self.n
        regs = self.regimes(i)
        d = self.draws[i]
        return NoiseV2Entry(
            name=str(d["name"]),
            cruise=regs.get("single") or regs["cruise"],
            standby=regs.get("standby"),
            traj_rig=self.bases[int(d["base"])].get("traj_rig"),
            provenance={
                "perturbation": d["perturbation"],
                "base": self.bases[int(d["base"])]["name"],
            },
        )
