"""A compact bank of PRIOR rigs in noise model v4 (``noise-prior-bank-v4/2``): one ``.npz``.

The rigs of the wide prior in noise model v4: ``noise-v4-fit/3`` FLIGHT
payloads drawn by ``experiments.noise_model.prior_v4.sample_prior_v4`` (the
prior_v1 laws of ``experiments.noise_model.drone_prior``, mapped by the rules
recorded in the bank's ``meta.header``, a phase pedestal drawn per order) and
kept by the identifiability gate. A rig's bulk is
the ``(R, K)`` aerodynamic profile, the ``(R, n_m)`` motor profile and the four
``(K,)`` pedestal rows; the rest is a dozen scalars, the floor spline's
weights and the draw ``(worker, attempt)`` that reproduces the rig. Everything
the rigs of a bank share — the geometry and rates, the floor control grid and
rule, the unit mic response, the prior, rules and seed — sits once in the
JSON ``meta.const`` inside the file.

:func:`rig_payload` is THE one assembly of a rig's payload from ``const`` and
the rig's own fields: the sampler builds its output through it and
:meth:`PriorBankV4.payload` rebuilds the same dict from the arrays, so a
written and a read rig cannot differ in layout, only by float16 (0.03 dB on
the dB tables, 5e-4 relative on the pedestal). The motor orders are
``3 p n``, ``n = 1 .. n_motor``; a rig without motor lines keeps its orders at
:data:`SILENT_LINE_DB`. A prior-v4 rig carries no wind (Dmitrii,
2026-10-25): ``params.wind`` is None, and wind is the ``noise_v2`` pool's
per-clip render option ``wind_sc``.

:class:`PriorBankV4` is a lazy ``Sequence`` of ``NoiseV2Entry``
(``prior_v4_<i>``, ``traj_rig`` None: the policy's own trajectory block flies
them), read by ``noise_v2_pool.load_preset_bank`` on the ``meta.format`` tag.
Built by ``scripts/prior_bank_v4_build.py``.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.noise_model import FIT_SCHEMA_V4

__all__ = [
    "PRIOR_BANK_V4_FORMAT",
    "SILENT_LINE_DB",
    "PriorBankV4",
    "bank_const",
    "rig_payload",
    "write_prior_bank_v4",
]

PRIOR_BANK_V4_FORMAT = "noise-prior-bank-v4/2"

#: The level (dB) of a line that is not there: the motor lines of a rig whose
#: v1 motor boost is not positive, and the padding of the motor table.
SILENT_LINE_DB = -300.0

#: The payload's top-level keys every rig of a bank shares.
TOP_CONST = (
    "kind",
    "rig",
    "n_rotors",
    "n_mics",
    "blades",
    "f_ref_rev_s",
    "sr",
    "band_hz",
    "front_end",
)
PEDESTAL = ("sigma2_amp", "gamma_amp_hz", "sigma2_phase", "gamma_phase_hz")
#: The rig's float32 scalars: ``(array name, params block, key)``.
SCALARS = (
    ("alpha_aero", None, "alpha_aero"),
    ("alpha_motor", None, "alpha_motor"),
    ("eps", "shaft", "eps"),
    ("lam_ref_per_s", "shaft", "lam_ref_per_s"),
    ("floor_mean_db", "floor", "mean_db"),
    ("floor_shape_sd_db", "floor", "shape_sd_db"),
    ("floor_b0", "floor", "b0"),
    ("floor_static_rel", "floor", "static_rel"),
)


def bank_const(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a prior-v4 ``payload`` every rig of its bank shares."""
    p = payload["params"]
    prov = payload["_prior_v4"]
    return {
        **{k: payload[k] for k in TOP_CONST},
        "shaft": {k: p["shaft"][k] for k in ("a_sigma", "b_lam")},
        "floor": {k: p["floor"][k] for k in ("floor_rule", "ctrl_hz", "b1")},
        "mic_response": p["mic_response"],
        "provenance": {k: prov[k] for k in ("prior", "rules", "seed")},
    }


def rig_payload(const: Mapping[str, Any], rig: Mapping[str, Any]) -> dict[str, Any]:
    """The ``noise-v4-fit/3`` flight payload of one prior rig.

    ``const`` is :func:`bank_const`'s block; ``rig`` holds the rig's own
    fields: ``profile_aero_db (R, K)``, ``profile_motor_db (R, n_m)``,
    ``pole_pairs``, the :data:`SCALARS` names, the :data:`PEDESTAL` rows
    ``(K,)``, ``floor_shape_z (14,)``, ``draw_worker``, ``draw_attempt``. Small
    containers are copied. The payload carries no wind (``params.wind``
    None)."""
    f64 = lambda x: np.asarray(x, dtype=np.float64).tolist()  # noqa: E731
    p = int(rig["pole_pairs"])
    motor = np.asarray(rig["profile_motor_db"], dtype=np.float64)
    orders = 3 * p * np.arange(1, motor.shape[1] + 1)
    out = {
        "schema": FIT_SCHEMA_V4,
        **{k: copy.deepcopy(const[k]) for k in TOP_CONST},
        "pole_pairs": p,
        "params": {
            "profile_aero_db": f64(rig["profile_aero_db"]),
            "motor_orders": orders.tolist(),
            "profile_motor_db": motor.tolist(),
            "alpha_aero": float(rig["alpha_aero"]),
            "alpha_motor": float(rig["alpha_motor"]),
            "pedestal": {k: f64(rig[k]) for k in PEDESTAL},
            "shaft": {
                "eps": float(rig["eps"]),
                "a_sigma": float(const["shaft"]["a_sigma"]),
                "lam_ref_per_s": float(rig["lam_ref_per_s"]),
                "b_lam": float(const["shaft"]["b_lam"]),
                "f_ref_rev_s": float(const["f_ref_rev_s"]),
            },
            "floor": {
                "floor_rule": str(const["floor"]["floor_rule"]),
                "mean_db": float(rig["floor_mean_db"]),
                "shape_z": f64(rig["floor_shape_z"]),
                "shape_sd_db": float(rig["floor_shape_sd_db"]),
                "ctrl_hz": f64(const["floor"]["ctrl_hz"]),
                "b0": float(rig["floor_b0"]),
                "b1": float(const["floor"]["b1"]),
                "static_rel": float(rig["floor_static_rel"]),
            },
            "mic_response": copy.deepcopy(const["mic_response"]),
            "wind": None,
        },
        "_prior_v4": {
            **copy.deepcopy(const["provenance"]),
            "draw_worker": int(rig["draw_worker"]),
            "draw_attempt": int(rig["draw_attempt"]),
        },
    }
    return out


def _rig_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The inverse of :func:`rig_payload`: a payload's own fields. A payload
    with a wind block is refused: a prior-v4 rig carries no wind."""
    p = payload["params"]
    pole_pairs = int(payload["pole_pairs"])
    motor = np.asarray(p["profile_motor_db"], dtype=np.float64)
    orders = np.asarray(p["motor_orders"], dtype=np.int64)
    if not np.array_equal(orders, 3 * pole_pairs * np.arange(1, orders.size + 1)):
        raise ValueError(f"motor orders {orders.tolist()} are not 3 p n for p = {pole_pairs}")
    if p.get("wind") is not None:
        raise ValueError(
            "a prior-v4 rig carries no wind; wind is the noise_v2 render option wind_sc"
        )
    out: dict[str, Any] = {
        "profile_aero_db": np.asarray(p["profile_aero_db"], dtype=np.float64),
        "profile_motor_db": motor.reshape(-1, orders.size),
        "pole_pairs": pole_pairs,
        "floor_shape_z": np.asarray(p["floor"]["shape_z"], dtype=np.float64),
        "draw_worker": int(payload["_prior_v4"]["draw_worker"]),
        "draw_attempt": int(payload["_prior_v4"]["draw_attempt"]),
    }
    for name in PEDESTAL:
        out[name] = np.asarray(p["pedestal"][name], dtype=np.float64)
    for name, block, key in SCALARS:
        out[name] = float(p[key] if block is None else p[block][key])
    return out


def write_prior_bank_v4(
    path: str | Path, payloads: Sequence[Mapping[str, Any]], header: dict[str, Any]
) -> Path:
    """Write prior rigs (each a :func:`rig_payload` dict) as one bank;
    ``header`` is free provenance (builder, rules, statistics). The shared part
    comes from the first rig; a rig whose shared part differs from it, or that
    carries a wind block, is refused, since the bank could not give it back."""
    if not payloads:
        raise ValueError("a bank needs at least one rig")
    const = bank_const(payloads[0])
    rigs = [_rig_fields(pl) for pl in payloads]
    for i, pl in enumerate(payloads):
        if bank_const(pl) != const:
            raise ValueError(f"rig {i} does not share the bank's constant part")
    n = len(rigs)
    R, K = rigs[0]["profile_aero_db"].shape
    n_max = max(r["profile_motor_db"].shape[1] for r in rigs)
    a: dict[str, np.ndarray] = {
        "profile_aero_db": np.empty((n, R, K), np.float16),
        "profile_motor_db": np.full((n, R, n_max), SILENT_LINE_DB, np.float16),
        "n_motor": np.empty(n, np.int16),
        "pole_pairs": np.empty(n, np.int16),
        "floor_shape_z": np.empty((n, rigs[0]["floor_shape_z"].size), np.float32),
        "draw_worker": np.empty(n, np.int64),
        "draw_attempt": np.empty(n, np.int64),
    }
    for name in PEDESTAL:
        a[name] = np.empty((n, K), np.float16)
    for name, _, _ in SCALARS:
        a[name] = np.empty(n, np.float32)
    for i, r in enumerate(rigs):
        n_m = r["profile_motor_db"].shape[1]
        a["profile_aero_db"][i] = r["profile_aero_db"]
        a["profile_motor_db"][i, :, :n_m] = r["profile_motor_db"]
        a["n_motor"][i] = n_m
        for name in ("pole_pairs", "floor_shape_z", "draw_worker", "draw_attempt", *PEDESTAL):
            a[name][i] = r[name]
        for name, _, _ in SCALARS:
            a[name][i] = r[name]
    meta = {"format": PRIOR_BANK_V4_FORMAT, "n": n, "const": const, "header": header}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    a["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(str(out), **a)  # type: ignore[arg-type]
    return out


class PriorBankV4(Sequence[Any]):
    """The rigs of a ``noise-prior-bank-v4/2`` file as a lazy sequence of
    ``NoiseV2Entry`` (``name`` ``prior_v4_<i>``, ``cruise`` the payload,
    ``traj_rig`` None). ``comb_offset_db`` is added to every line, aerodynamic
    and motor, when a payload is assembled."""

    def __init__(self, path: str | Path, *, comb_offset_db: float = 0.0):
        self.path = Path(path)
        with np.load(self.path) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("format") != PRIOR_BANK_V4_FORMAT:
                raise ValueError(
                    f"{self.path}: expected format {PRIOR_BANK_V4_FORMAT!r}, "
                    f"got {meta.get('format')!r}"
                )
            self.arrays = {k: z[k] for k in z.files if k != "meta"}
        self.meta = meta
        self.const: dict[str, Any] = meta["const"]
        self.header: dict[str, Any] = meta["header"]
        self.n = int(meta["n"])
        self.comb_offset_db = float(comb_offset_db)
        if self.arrays["profile_aero_db"].shape[0] != self.n:
            raise ValueError(
                f"{self.path}: {self.arrays['profile_aero_db'].shape[0]} rigs, header says {self.n}"
            )

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

    def with_offset(self, comb_offset_db: float) -> PriorBankV4:
        other = PriorBankV4.__new__(PriorBankV4)
        other.__dict__.update(self.__dict__)
        other.comb_offset_db = float(comb_offset_db)
        return other

    @property
    def traj_rigs(self) -> tuple[str, ...]:
        return ()

    def payload(self, i: int) -> dict[str, Any]:
        """Rig ``i`` as a ``noise-v4-fit/3`` payload, ``_prior_v4`` provenance included."""
        i = int(i)
        if not 0 <= i < self.n:
            raise IndexError(i)
        a = self.arrays
        n_m = int(a["n_motor"][i])
        rig: dict[str, Any] = {
            "profile_aero_db": a["profile_aero_db"][i].astype(np.float64) + self.comb_offset_db,
            "profile_motor_db": a["profile_motor_db"][i, :, :n_m].astype(np.float64)
            + self.comb_offset_db,
            "pole_pairs": int(a["pole_pairs"][i]),
            "floor_shape_z": a["floor_shape_z"][i],
            "draw_worker": int(a["draw_worker"][i]),
            "draw_attempt": int(a["draw_attempt"][i]),
        }
        for name in PEDESTAL:
            rig[name] = a[name][i]
        for name, _, _ in SCALARS:
            rig[name] = float(a[name][i])
        return rig_payload(self.const, rig)

    def __getitem__(self, i: Any) -> Any:
        from data_processing.noise_v2_pool import NoiseV2Entry

        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self.n))]
        i = int(i)
        if i < 0:
            i += self.n
        return NoiseV2Entry(
            name=f"prior_v4_{i:05d}", cruise=self.payload(i), standby=None, traj_rig=None
        )
