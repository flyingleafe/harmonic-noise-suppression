"""A compact bank of PRIOR rigs in noise model v4 (``noise-prior-bank-v4/1``): one ``.npz``.

The prior_v1 rigs (:mod:`.prior_bank`, ``noise-v3-fit/1`` payloads drawn by
``experiments.noise_model.drone_prior``) converted to ``noise-v4-fit/3``
FLIGHT payloads by ``experiments.noise_model.prior_v4`` (the conversion rules
decided 2026-10-24, recorded in the bank's ``meta.header``). A rig's bulk is
the ``(R, K)`` aerodynamic profile, the ``(R, n_m)`` motor profile and the four
``(K,)`` pedestal rows; the rest is a dozen scalars, the floor spline's
weights and the per-mic wind levels. Everything the rigs of a bank share — the
geometry and rates, the floor control grid and rule, the unit mic response,
the SC gust law and its measured mean shape, the source pin — sits once in
the JSON ``meta.const`` inside the file.

:func:`rig_payload` is THE one assembly of a rig's payload from ``const`` and
the rig's own fields: the converter builds its output through it and
:meth:`PriorBankV4.payload` rebuilds the same dict from the arrays, so a
written and a read rig cannot differ in layout, only by float16 (0.03 dB on
the dB tables, 5e-4 relative on the pedestal). The motor orders are
``3 p n``, ``n = 1 .. n_motor``; a rig without motor lines keeps its orders at
:data:`SILENT_LINE_DB`. A capsule without wind carries
:data:`.v4.WIND_OFF_DB`, which the renderer skips; a rig without wind has
``params.wind`` None.

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
from data_processing.noise_model.v4 import WIND_OFF_DB

__all__ = [
    "PRIOR_BANK_V4_FORMAT",
    "SILENT_LINE_DB",
    "PriorBankV4",
    "bank_const",
    "rig_payload",
    "write_prior_bank_v4",
]

PRIOR_BANK_V4_FORMAT = "noise-prior-bank-v4/1"

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
#: The wind block's keys every rig shares (the per-mic ``level_db`` is the rig's).
WIND_CONST = ("kind", "speed_offset_ms", "speed_scale_ms", "gust_rate_hz", "shape")
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
    wind = p.get("wind")
    prov = payload["_prior_v4"]
    return {
        **{k: payload[k] for k in TOP_CONST},
        "shaft": {k: p["shaft"][k] for k in ("a_sigma", "b_lam")},
        "floor": {k: p["floor"][k] for k in ("floor_rule", "ctrl_hz", "b1")},
        "mic_response": p["mic_response"],
        # a rig without wind still belongs to a bank whose wind law is fixed
        "wind": None if wind is None else {k: wind[k] for k in WIND_CONST},
        "provenance": {k: prov[k] for k in ("source_bank", "rules", "seed")},
    }


def rig_payload(const: Mapping[str, Any], rig: Mapping[str, Any]) -> dict[str, Any]:
    """The ``noise-v4-fit/3`` flight payload of one prior rig.

    ``const`` is :func:`bank_const`'s block (its wind law must be set when the
    rig has wind); ``rig`` holds the rig's own fields: ``profile_aero_db (R,
    K)``, ``profile_motor_db (R, n_m)``, ``pole_pairs``, the
    :data:`SCALARS` names, the :data:`PEDESTAL` rows ``(K,)``,
    ``floor_shape_z (14,)``, ``wind_level_db (M,)`` or None, ``source_index``,
    ``v1_seed``. Small containers are copied; the wind ``shape`` (thousands of
    floats, read-only to every reader) is shared."""
    f64 = lambda x: np.asarray(x, dtype=np.float64).tolist()  # noqa: E731
    p = int(rig["pole_pairs"])
    motor = np.asarray(rig["profile_motor_db"], dtype=np.float64)
    orders = 3 * p * np.arange(1, motor.shape[1] + 1)
    level = rig["wind_level_db"]
    wind = None
    if level is not None:
        law = const["wind"]
        if law is None:
            raise ValueError("a rig with wind needs the bank's wind law in const['wind']")
        wind = {k: (law[k] if k == "shape" else copy.deepcopy(law[k])) for k in WIND_CONST}
        wind["level_db"] = f64(level)
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
            "wind": wind,
        },
        "_prior_v4": {
            **copy.deepcopy(const["provenance"]),
            "source_index": int(rig["source_index"]),
            "v1_seed": int(rig["v1_seed"]),
        },
    }
    return out


def _rig_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The inverse of :func:`rig_payload`: a payload's own fields."""
    p = payload["params"]
    pole_pairs = int(payload["pole_pairs"])
    motor = np.asarray(p["profile_motor_db"], dtype=np.float64)
    orders = np.asarray(p["motor_orders"], dtype=np.int64)
    if not np.array_equal(orders, 3 * pole_pairs * np.arange(1, orders.size + 1)):
        raise ValueError(f"motor orders {orders.tolist()} are not 3 p n for p = {pole_pairs}")
    wind = p.get("wind")
    out: dict[str, Any] = {
        "profile_aero_db": np.asarray(p["profile_aero_db"], dtype=np.float64),
        "profile_motor_db": motor.reshape(-1, orders.size),
        "pole_pairs": pole_pairs,
        "floor_shape_z": np.asarray(p["floor"]["shape_z"], dtype=np.float64),
        "wind_level_db": None if wind is None else np.asarray(wind["level_db"], dtype=np.float64),
        "source_index": int(payload["_prior_v4"]["source_index"]),
        "v1_seed": int(payload["_prior_v4"]["v1_seed"]),
    }
    for name in PEDESTAL:
        out[name] = np.asarray(p["pedestal"][name], dtype=np.float64)
    for name, block, key in SCALARS:
        out[name] = float(p[key] if block is None else p[block][key])
    return out


def write_prior_bank_v4(
    path: str | Path, payloads: Sequence[Mapping[str, Any]], header: dict[str, Any]
) -> Path:
    """Write converted prior rigs (each a :func:`rig_payload` dict) as one bank;
    ``header`` is free provenance (builder, rules, statistics). The shared part
    comes from the first rig with wind (else the first); a rig whose shared
    part differs from it is refused, since the bank could not give it back."""
    if not payloads:
        raise ValueError("a bank needs at least one rig")
    first = next((pl for pl in payloads if pl["params"].get("wind") is not None), payloads[0])
    const = bank_const(first)
    rigs = [_rig_fields(pl) for pl in payloads]
    for i, pl in enumerate(payloads):
        c = bank_const(pl)
        if c["wind"] is None:
            c["wind"] = const["wind"]
        if c != const:
            raise ValueError(f"rig {i} does not share the bank's constant part")
    n = len(rigs)
    R, K = rigs[0]["profile_aero_db"].shape
    M = int(const["n_mics"])
    n_max = max(r["profile_motor_db"].shape[1] for r in rigs)
    a: dict[str, np.ndarray] = {
        "profile_aero_db": np.empty((n, R, K), np.float16),
        "profile_motor_db": np.full((n, R, n_max), SILENT_LINE_DB, np.float16),
        "n_motor": np.empty(n, np.int16),
        "pole_pairs": np.empty(n, np.int16),
        "floor_shape_z": np.empty((n, rigs[0]["floor_shape_z"].size), np.float32),
        "wind_level_db": np.full((n, M), WIND_OFF_DB, np.float16),
        "has_wind": np.zeros(n, bool),
        "source_index": np.empty(n, np.int64),
        "v1_seed": np.empty(n, np.int64),
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
        for name in ("pole_pairs", "floor_shape_z", "source_index", "v1_seed", *PEDESTAL):
            a[name][i] = r[name]
        for name, _, _ in SCALARS:
            a[name][i] = r[name]
        if r["wind_level_db"] is not None:
            a["wind_level_db"][i] = r["wind_level_db"]
            a["has_wind"][i] = True
    meta = {"format": PRIOR_BANK_V4_FORMAT, "n": n, "const": const, "header": header}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    a["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(str(out), **a)  # type: ignore[arg-type]
    return out


class PriorBankV4(Sequence[Any]):
    """The rigs of a ``noise-prior-bank-v4/1`` file as a lazy sequence of
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
            "wind_level_db": a["wind_level_db"][i] if bool(a["has_wind"][i]) else None,
            "source_index": int(a["source_index"][i]),
            "v1_seed": int(a["v1_seed"][i]),
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
