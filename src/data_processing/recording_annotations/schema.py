"""Source and acoustic conditions of a recording. Unknown is never synonymous with no."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any

UNKNOWN = "unknown"
NA = "not_applicable"
_MISSING = (UNKNOWN, NA)

# Blade count is per active shaft assembly, including stacked propellers on that shaft.
FACETS: dict[str, tuple[str, ...]] = {
    "rig_mount": (
        "fixed_stand",
        "free_flight",
        "tethered",
        "handheld",
        "grounded",
        "mixed",
        *_MISSING,
    ),
    "speed_regime": ("constant", "variable", "mixed", "off", *_MISSING),
    "propulsion": ("propellers", "motor_only", "mixed", "none", *_MISSING),
    "rotor_activity": ("all", "single", "subset", "off", "mixed", *_MISSING),
    "relative_motion": ("fixed", "moving", "mixed", *_MISSING),
    "speed_supplied": ("yes", "no", *_MISSING),
    "sound_content": (
        "drone_only",
        "co_recorded",
        "digital_mix",
        "both",
        "no_drone",
        "mixed",
        *_MISSING,
    ),
    "environment": (
        "anechoic",
        "semi_anechoic",
        "indoors",
        "outdoors",
        "mixed",
        "simulated",
        *_MISSING,
    ),
    "blade_count": (*map(str, range(13)), "mixed", *_MISSING),
    "wind": ("significant", "not_significant", "mixed", *_MISSING),
}


def valid_value(facet: str, value: str) -> bool:
    """Allow any explicitly documented blade count; other facets have a closed vocabulary."""
    values = FACETS[facet]
    if value in values:
        return True
    return (
        facet == "blade_count"
        and value.isascii()
        and value.isdecimal()
        and not value.startswith("0")
    )


def normalize(record: Mapping[str, Any], *, has_audio: bool) -> dict[str, Any]:
    """Validate and complete one classification, preserving role-specific source evidence.

    Missing traits stay unknown on audio, not_applicable on non-audio records. A source can
    still describe the rig/room associated with a log. Missing *provided* RPM is not filled
    by a spectral estimate, even when all rotor rates are resolved acoustically.
    """
    out = dict(record)
    evidence = dict(out.get("evidence") or {})
    for name in FACETS:
        value = out.get(name)
        if value is None:
            value = UNKNOWN if has_audio else NA
        if name == "blade_count" and isinstance(value, Integral) and not isinstance(value, bool):
            value = str(value)
        if not isinstance(value, str) or not valid_value(name, value):
            raise ValueError(f"invalid {name}={value!r}")
        out[name] = value
        if name not in evidence:
            if value == UNKNOWN:
                evidence[name] = "Not established by the source or a validated measurement."
            elif value == NA:
                evidence[name] = "No applicable observation for this entry."
            else:
                raise ValueError(f"{name}={value!r} has no evidence")
        if not isinstance(evidence[name], str) or not evidence[name].strip():
            raise ValueError(f"{name}: evidence must identify the source or measurement")
    out["evidence"] = evidence
    for name in ("n_active", "n_rotors"):
        count = out.get(name)
        if count is not None and (
            isinstance(count, bool) or not isinstance(count, Integral) or count < 0
        ):
            raise ValueError(f"invalid {name}={count!r}")
        out[name] = None if count is None else int(count)
    active, installed, activity = out["n_active"], out["n_rotors"], out["rotor_activity"]
    if active is not None:
        if installed is not None and active > installed:
            raise ValueError("active rotor count exceeds installed rotor count")
        expected = {"off": 0, "single": 1, "all": installed}.get(activity)
        if expected is not None and active != expected:
            raise ValueError(f"rotor_activity={activity!r} is inconsistent with n_active={active}")
        if active and out["speed_regime"] == "off":
            raise ValueError("speed_regime='off' is inconsistent with a powered rotor")
    blades = out["blade_count"]
    if blades.isdecimal() and (
        (out["propulsion"] == "motor_only" and blades != "0")
        or (out["propulsion"] == "propellers" and blades == "0")
    ):
        raise ValueError(
            f"blade_count={blades} is inconsistent with propulsion={out['propulsion']!r}"
        )
    for field, facet in (("role_motion", "relative_motion"), ("role_wind", "wind")):
        if field in out:
            roles = out[field]
            if not isinstance(roles, Mapping) or any(
                not isinstance(k, str) or not k or v not in FACETS[facet] for k, v in roles.items()
            ):
                raise ValueError(f"invalid {field}: {roles!r}")
    if not has_audio:
        out["tach_candidate"] = False
    elif out["speed_regime"] == "constant" and out["sound_content"] != "no_drone":
        out["tach_candidate"] = True
    else:
        out.setdefault("tach_candidate", out["speed_regime"] == "mixed")
    if not isinstance(out["tach_candidate"], bool):
        raise ValueError("tach_candidate must be boolean")
    return out
