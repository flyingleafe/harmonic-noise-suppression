"""A compact bank of PRIOR-DRAWN rigs (``noise-prior-bank/1``): one ``.npz``.

A prior rig (``experiments.noise_model.drone_prior``) is a ``noise-v3-fit/1``
payload whose bulk is three ``(R, K)`` tables - ``profile_db``,
``am.sigma2``, ``am.gamma_hz`` - plus a few scalars and the floor spline's
weights; ``front_end``, the zero ``wander`` block, ``wind_sc`` and the
prior's spec are the same for every rig of a bank. 16 384 rigs as inlined
JSON are ~600 MB; here the tables are float16 arrays in one compressed
``.npz`` (~40 MB) with the constant parts in a JSON header inside it.
:class:`PriorBank` is a lazy sequence of ``NoiseV2Entry``: a payload is
assembled from the arrays when indexed, so a DataLoader worker holds the
arrays, not 16 k dicts. float16 keeps 0.03 dB on a -60..+60 dB profile
(line scatter 3.5 dB), 5e-4 on ``sigma2`` <= 0.7, 0.02 Hz on a 25 Hz rate.

Built by ``scripts/prior_bank_build.py``; consumed by ``kind: noise_v2``
with ``preset_bank: <file>.npz`` (``noise_v2_pool.load_preset_bank``).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.noise_model import FIT_SCHEMA_V3

PRIOR_BANK_FORMAT = "noise-prior-bank/1"
SCALARS = (
    "sigma_nu",
    "lam",
    "amp_exp",
    "floor_mean_db",
    "floor_shape_sd_db",
    "floor_exp",
    "floor_static_rel",
)
DRAWN = (
    "slope_db_dec",
    "k2_over_floor_db",
    "tail_over_floor_db",
    "odd_pen_k3_db",
    "odd_pen_k30_db",
    "motor_boost_db",
    "motor_period",
    "pole_pairs",
    "sigma_psi",
)
TEMPLATES = ("hump", "tilt")


def _scalar(p: dict[str, Any], name: str) -> float:
    if name in ("sigma_nu", "lam"):
        return float(p[name])
    if name == "amp_exp":
        return float(p["profile"]["amp_exp"])
    return float(p["floor"][name])


def _tables(n: int, R: int, K: int, J: int) -> dict[str, np.ndarray]:
    a: dict[str, np.ndarray] = {
        "profile_db": np.empty((n, R, K), np.float16),
        "am_sigma2": np.empty((n, R, K), np.float16),
        "am_gamma_hz": np.empty((n, K), np.float16),  # one rate per order, shared by rotors
        "floor_shape_z": np.empty((n, J), np.float32),
        "over_floor_db": np.empty((n, K), np.float16),
        "floor_ctrl_db": np.empty((n, J), np.float32),
        "floor_template": np.empty(n, np.uint8),
        "gate_frames_ok": np.zeros((n, R), np.int16),
        "gate_eligible": np.zeros((n, R), np.int16),
        "seed": np.zeros(n, np.int64),
    }
    for name in SCALARS + DRAWN:
        a[name] = np.empty(n, np.float32)
    return a


def _fill(a: dict[str, np.ndarray], i: int, pl: dict[str, Any]) -> None:
    p, d = pl["params"], pl["_prior"]["drawn"]
    a["profile_db"][i] = p["profile"]["profile_db"]
    a["am_sigma2"][i] = p["am"]["sigma2"]
    g = np.asarray(p["am"]["gamma_hz"], dtype=np.float64)
    if not np.allclose(g, g[:1]):
        raise ValueError("a prior rig's AM rates are shared by its rotors; this payload's are not")
    a["am_gamma_hz"][i] = g[0]
    a["floor_shape_z"][i] = p["floor"]["floor_shape_z"]
    a["over_floor_db"][i] = d["over_floor_db"]
    a["floor_ctrl_db"][i] = d["floor_ctrl_db"]
    a["floor_template"][i] = TEMPLATES.index(d["floor_template"])
    for name in SCALARS:
        a[name][i] = _scalar(p, name)
    for name in DRAWN:
        a[name][i] = d[name]
    g = pl["_prior"].get("gate")
    if g is not None:
        a["gate_frames_ok"][i] = g["frames_ok"]
        a["gate_eligible"][i] = g["eligible"]
    a["seed"][i] = int(pl["_prior"].get("seed", -1))


CONST_KEYS = ("schema", "kind", "mode", "n_rotors", "n_mics", "k_max", "sr", "front_end")


def write_prior_bank(
    path: str | Path, payloads: Sequence[dict[str, Any]], header: dict[str, Any]
) -> Path:
    """Write prior draws (each carrying ``_prior``) as one bank; ``header`` is
    free provenance (builder, seed, gate configuration, statistics). The
    constant payload parts come from the first payload."""
    if not payloads:
        raise ValueError("a bank needs at least one rig")
    first = payloads[0]
    p0 = first["params"]
    R, K = np.asarray(p0["profile"]["profile_db"]).shape
    a = _tables(len(payloads), int(R), int(K), len(p0["floor"]["floor_shape_z"]))
    for i, pl in enumerate(payloads):
        _fill(a, i, pl)
    const = {k: first[k] for k in CONST_KEYS}
    const["wander"] = p0["wander"]
    const["wind_sc"] = p0["wind_sc"]
    const["mic_dev_sd_db"] = p0["profile"].get("mic_dev_sd_db", 0.0)
    const["spec"] = first["_prior"]["spec"]
    meta = {"format": PRIOR_BANK_FORMAT, "n": len(payloads), "const": const, "header": header}
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    a["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(str(out), **a)  # type: ignore[arg-type]
    return out


class PriorBank(Sequence[Any]):
    """The rigs of a ``noise-prior-bank/1`` file as a lazy sequence of
    ``NoiseV2Entry`` (``name`` ``prior_<i>``, ``cruise`` the payload,
    ``traj_rig`` None: the policy's own trajectory block flies them)."""

    def __init__(self, path: str | Path, *, comb_offset_db: float = 0.0):
        self.path = Path(path)
        with np.load(self.path) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("format") != PRIOR_BANK_FORMAT:
                raise ValueError(
                    f"{self.path}: expected format {PRIOR_BANK_FORMAT!r}, got {meta.get('format')!r}"
                )
            self.arrays = {k: z[k] for k in z.files if k != "meta"}
        self.meta = meta
        self.const: dict[str, Any] = meta["const"]
        self.header: dict[str, Any] = meta["header"]
        self.n = int(meta["n"])
        self.comb_offset_db = float(comb_offset_db)
        if self.arrays["profile_db"].shape[0] != self.n:
            raise ValueError(
                f"{self.path}: {self.arrays['profile_db'].shape[0]} rigs, header says {self.n}"
            )

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

    def with_offset(self, comb_offset_db: float) -> PriorBank:
        other = PriorBank.__new__(PriorBank)
        other.__dict__.update(self.__dict__)
        other.comb_offset_db = float(comb_offset_db)
        return other

    @property
    def traj_rigs(self) -> tuple[str, ...]:
        return ()

    def _params(self, i: int) -> dict[str, Any]:
        a, c = self.arrays, self.const
        R, K = a["profile_db"].shape[1:]
        prof = a["profile_db"][i].astype(np.float64) + self.comb_offset_db
        f64 = lambda name: a[name][i].astype(np.float64).tolist()  # noqa: E731
        return {
            "sigma_nu": float(a["sigma_nu"][i]),
            "lam": float(a["lam"][i]),
            "gamma_hz": np.zeros((R, K)).tolist(),
            "carrier_rev_s": None,
            "profile": {
                "profile_db": prof.tolist(),
                "amp_exp": float(a["amp_exp"][i]),
                "mic_dev_sd_db": float(c["mic_dev_sd_db"]),
            },
            "floor": {
                "floor_mean_db": float(a["floor_mean_db"][i]),
                "floor_shape_z": f64("floor_shape_z"),
                "floor_shape_sd_db": float(a["floor_shape_sd_db"][i]),
                "floor_exp": float(a["floor_exp"][i]),
                "floor_static_rel": float(a["floor_static_rel"][i]),
            },
            "wind": None,
            "wander": dict(c["wander"]),
            "am": {
                "sigma2": f64("am_sigma2"),
                "gamma_hz": np.repeat(
                    a["am_gamma_hz"][i : i + 1].astype(np.float64), R, 0
                ).tolist(),
            },
            "wind_sc": dict(c["wind_sc"]),
            "array_response": None,
        }

    def payload(self, i: int) -> dict[str, Any]:
        """Rig ``i`` as a ``noise-v3-fit/1`` payload, ``_prior`` provenance included."""
        i = int(i)
        if not 0 <= i < self.n:
            raise IndexError(i)
        a, c = self.arrays, self.const
        f64 = lambda name: a[name][i].astype(np.float64).tolist()  # noqa: E731
        drawn: dict[str, Any] = {name: float(a[name][i]) for name in DRAWN}
        drawn["floor_template"] = TEMPLATES[int(a["floor_template"][i])]
        drawn["over_floor_db"] = f64("over_floor_db")
        drawn["floor_ctrl_db"] = f64("floor_ctrl_db")
        out: dict[str, Any] = {k: c[k] for k in CONST_KEYS}
        out["schema"] = FIT_SCHEMA_V3
        out["params"] = self._params(i)
        out["_prior"] = {
            "spec": c["spec"],
            "drawn": drawn,
            "seed": int(a["seed"][i]),
            "gate": {
                "frames_ok": a["gate_frames_ok"][i].tolist(),
                "eligible": a["gate_eligible"][i].tolist(),
            },
        }
        return out

    def __getitem__(self, i: Any) -> Any:
        from data_processing.noise_v2_pool import NoiseV2Entry

        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self.n))]
        i = int(i)
        if i < 0:
            i += self.n
        return NoiseV2Entry(
            name=f"prior_{i:05d}", cruise=self.payload(i), standby=None, traj_rig=None
        )
