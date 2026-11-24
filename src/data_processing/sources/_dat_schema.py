"""Record schemas from the definitions embedded in DJI ``DJI_LOG_V3`` ``.DAT`` logs.

M100 flight-controller logs carry their own record layouts as type-65533
packets. Each packet payload is one NUL-terminated ASCII line; a section
header spans three ``\\n``-separated lines in one packet::

    Op.[config]\\nname\\tesc_data\\ntype\\t10090
    Op.int16_t\\tesc1_speed 0
    Op.uint16_t\\tesc1_voltage 0
    ...
    Op.[string]\\nname\\tfly_log\\ntype\\t32768
    Op.[config_dump]\\nname\\tSys_Cfg\\ntype\\t65535
    Op.expr\\tE_gps_snr1\\tbitand(gps_snr1,127)

``Op.[config]`` sections define a binary record: fields packed in order,
little-endian, no padding, so the record size is the sum of the field sizes
(DatCon, BudWalkerJava/DatCon ``src/DatConRecs/RecDef/OpConfig.java``).
Field types are ``uint8_t``/``int8_t``/``uint16_t``/``int16_t``/``uint32_t``/
``int32_t``/``fp32``/``float``/``double``/``fp64`` (case-insensitive); the
trailing number is the field's default value, not part of the layout. Field
names are kept verbatim (``compass.mx``, ``posFlg[0]``); arrays are written
element by element, so every field is a scalar. ``Op.[string]`` and
``Op.[config_dump]`` sections name text record types and ``Op.expr`` lines
are derived quantities over fields of any record (bit extraction with
``bitand``/``shift_r``, quaternion to Euler): neither describes payload bytes.

Layouts are keyed by ``(record type, byte size)`` and apply only to packets of
exactly that size; a shorter layout is never applied as a prefix. A definition
stream that lost a line yields a layout which is shorter but NOT a prefix of
the real one: FLY013 dropped a log chunk holding ``fdi_info``'s second float
(``gyroy_bias_raw_device``), so its embedded layout is 22 B against 26 B
packets, and prefix decoding would shift the five following fields.

Decoding a group: ``np.ascontiguousarray(m).view(dtype)[:, 0]`` on the
``(n_packets, dtype.itemsize)`` uint8 matrix gives a structured array that
``pandas.DataFrame`` takes directly, one scalar column per field.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Iterator

import numpy as np

DEFINITION_RECORD_TYPE = 65533

_FIELD_FORMATS = {
    "uint8_t": "u1",
    "int8_t": "i1",
    "uint16_t": "<u2",
    "int16_t": "<i2",
    "uint32_t": "<u4",
    "int32_t": "<i4",
    "fp32": "<f4",
    "float": "<f4",
    "double": "<f8",
    "fp64": "<f8",
}
_TEXT_KINDS = ("string", "config_dump")
_HEADER = re.compile(r"op\.\[(.*)\]", re.IGNORECASE)
_NUMBER = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")
_RECORD_TYPE = re.compile(r"\d{1,5}")

Schema = tuple[str, np.dtype]

# FLY013 lost this layout's Y-bias field declaration. Apply the full layout
# from the other two M100 flights only when the *specific* damaged definition
# matches, never just because an unrelated firmware also uses record type 26.
_FDI_FIELDS = [
    ("gyro_bias_raw_flag_device", "u1"),
    ("gyrox_bias_raw_device", "<f4"),
    ("gyroy_bias_raw_device", "<f4"),
    ("gyroz_bias_raw_device", "<f4"),
    ("gyro_static_flag", "u1"),
    ("wx_stdvar", "<f4"),
    ("wy_stdvar", "<f4"),
    ("wz_stdvar", "<f4"),
]
_FDI = ("fdi_info", np.dtype(_FDI_FIELDS))
_TRUNCATED_FDI = ("fdi_info", np.dtype(_FDI_FIELDS[:2] + _FDI_FIELDS[3:]))


class _MalformedSection(ValueError):
    pass


def _lines(messages: Iterable[bytes]) -> Iterator[list[str]]:
    """Whitespace-split lines; a message ends at its first NUL (DatCon)."""
    for message in messages:
        text = bytes(message).split(b"\x00", 1)[0].decode("ascii", errors="replace")
        for line in text.split("\n"):
            if tokens := line.split():
                yield tokens


def _field(tokens: list[str]) -> tuple[str, str] | None:
    """``(name, numpy format)`` of a field line, ``None`` for an ``Op.expr``."""
    op = tokens[0].lower()
    if op == "op.expr":
        return None
    fmt = _FIELD_FORMATS.get(op[3:]) if op.startswith("op.") else None
    if fmt is None:
        raise _MalformedSection(f"unsupported line {' '.join(tokens)!r}")
    words = tokens[1:-1] if len(tokens) > 2 and _NUMBER.fullmatch(tokens[-1]) else tokens[1:]
    if not words or any(
        _NUMBER.fullmatch(w) or not (w.isascii() and w.isprintable()) for w in words
    ):
        raise _MalformedSection(f"bad field name in {' '.join(tokens)!r}")
    return "_".join(words), fmt


def _dtype(fields: list[tuple[str, str]]) -> np.dtype:
    """Packed structured dtype; repeated names get ``_2``, ``_3``, ... suffixes."""
    seen: set[str] = set()
    named = []
    for name, fmt in fields:
        unique, k = name, 1
        while unique in seen:
            k += 1
            unique = f"{name}_{k}"
        seen.add(unique)
        named.append((unique, fmt))
    return np.dtype(named)


def _sections(messages: Iterable[bytes]) -> Iterator[tuple[str, str, int, np.dtype | None]]:
    """``(kind, name, record type, dtype or None)`` per section; warns on bad ones."""
    head: list[str] | None = None  # header tokens: kind, then name, then type
    name, rtype = "", -1
    fields: list[tuple[str, str]] = []
    error: str | None = None

    def close() -> Iterator[tuple[str, str, int, np.dtype | None]]:
        if head is None:
            return
        kind = head[0]
        problem = error
        if problem is None and len(head) < 3:
            problem = "header without name/type"
        if problem is None and kind == "config" and not fields:
            problem = "no fields"
        if problem is None and kind != "config" and kind not in _TEXT_KINDS:
            problem = f"unsupported section kind {kind!r}"
        if problem is not None:
            warnings.warn(
                f"DAT record definition Op.[{kind}] {name!r} (type {rtype}) ignored: {problem}",
                stacklevel=4,
            )
            return
        yield kind, name, rtype, _dtype(fields) if kind == "config" else None

    for tokens in _lines(messages):
        if header := _HEADER.fullmatch(tokens[0]):
            yield from close()
            head, name, rtype, fields, error = [header.group(1).lower()], "", -1, [], None
            if len(tokens) > 1:
                error = f"junk after header {' '.join(tokens)!r}"
            continue
        if head is None:
            warnings.warn(
                f"DAT record definition line outside a section: {' '.join(tokens)!r}", stacklevel=3
            )
            continue
        if error is not None:
            continue
        if len(head) == 1:
            if tokens[0].lower() == "name" and len(tokens) > 1:
                name = "_".join(tokens[1:])
                head.append(name)
            else:
                error = f"expected 'name', got {' '.join(tokens)!r}"
            continue
        if len(head) == 2:
            if (
                tokens[0].lower() == "type"
                and len(tokens) == 2
                and _RECORD_TYPE.fullmatch(tokens[1])
            ):
                rtype = int(tokens[1])
                head.append(tokens[1])
                if rtype > 0xFFFF:
                    error = f"record type {rtype} exceeds uint16"
            else:
                error = f"expected 'type <uint16>', got {' '.join(tokens)!r}"
            continue
        try:
            field = _field(tokens)
        except _MalformedSection as e:
            error = str(e)
            continue
        if field is not None:
            if head[0] != "config":
                error = f"field in non-binary section: {' '.join(tokens)!r}"
            else:
                fields.append(field)
    yield from close()


def parse_definitions(messages: Iterable[bytes]) -> dict[tuple[int, int], Schema]:
    """Map (record type, exact payload width) to the record name and packed dtype.

    Input: XOR-decoded RecDefs messages in file order. Unknown layouts are
    preferable to wrong values: conflicting or malformed definitions are not
    used for decoding. Text and calculated signals are not packed fields.
    """
    schemas: dict[tuple[int, int], Schema] = {}
    conflicted: set[tuple[int, int]] = set()
    for _, name, rtype, dtype in _sections(messages):
        if dtype is None:
            continue
        key = (rtype, dtype.itemsize)
        if key in conflicted:
            continue
        if key in schemas and schemas[key] != (name, dtype):
            warnings.warn(
                f"DAT record definitions for type {rtype}, {dtype.itemsize} B disagree "
                f"({schemas[key][0]!r} vs {name!r}); both ignored",
                stacklevel=3,
            )
            del schemas[key]
            conflicted.add(key)
            continue
        schemas[key] = (name, dtype)
    if schemas.get((26, 22)) == _TRUNCATED_FDI and (26, 26) not in conflicted:
        schemas.setdefault((26, 26), _FDI)
    return schemas
