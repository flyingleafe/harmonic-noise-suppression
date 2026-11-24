"""Embedded DJI_LOG_V3 record definitions (type 65533) -> packed structured dtypes.

Message texts are verbatim from the M100 FLY013/FLY018/FLY022 logs except where
a test needs syntax those logs lack (marked).
"""

from __future__ import annotations

import struct
import warnings

import numpy as np
import pandas as pd
import pytest

from data_processing.sources._dat_schema import parse_definitions

FDI_INFO = [
    "Op.[config]\nname\tfdi_info\ntype\t26",
    "Op.uint8_t\tgyro_bias_raw_flag_device 0",
    "Op.float\tgyrox_bias_raw_device 0",
    "Op.float\tgyroy_bias_raw_device 0",
    "Op.float\tgyroz_bias_raw_device 0",
    "Op.uint8_t\tgyro_static_flag 0",
    "Op.float\twx_stdvar 0",
    "Op.float\twy_stdvar 0",
    "Op.float\twz_stdvar 0",
]
SIM_SENSOR = [
    "Op.[config]\nname\tsimulator_sensor_data_25HZ\ntype\t1402",
    "Op.int16_t\tsimulator_compass_MagData[0] 0",
    "Op.int16_t\tsimulator_compass_MagData[1] 0",
    "Op.int16_t\tsimulator_compass_MagData[2] 0",
    "Op.uint16_t\tsimulator_compass_cnt 0",
]
MIXED = [
    "Op.[config]\nname\tmixed\ntype\t1006",  # synthetic: every scalar type
    "Op.uint8_t\tg_api_emergency_break_flag  0",
    "Op.int8_t\ti8 0",
    "Op.uint16_t\tu16 0",
    "Op.int16_t\tcompass.mx 0",
    "Op.uint32_t\tctrl_tick 0",
    "Op.int32_t\tpack_current 0",
    "Op.fp32\thdop 0",
    "Op.double\tlati 0",
    "Op.fp64\tlongti 0",
]
TRAILER = [
    "Op.[string]\nname\tfly_log\ntype\t32768",
    "Op.[config_dump]\nname\tSys_Cfg\ntype\t65535",
    "Op.[string]\nname\tcfg_log\ntype\t65533",
    "Op.expr\tE_gps_snr1\tbitand(gps_snr1,127)",
    "Op.expr\tE_component_gnss_hp\t\tbitand(shift_r(ns_data_component_all,0),1)",
]


def _messages(lines: list[str]) -> list[bytes]:
    return [line.encode() + b"\x00" for line in lines]


def _parse(lines: list[str]):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return parse_definitions(_messages(lines))


def _layout(dtype: np.dtype) -> list[tuple[str, str, int]]:
    return [(name, f[0].str, f[1]) for name, f in (dtype.fields or {}).items()]


def test_layouts_keyed_by_type_and_exact_width():
    schemas = _parse(FDI_INFO + SIM_SENSOR + MIXED + TRAILER)

    assert set(schemas) == {(26, 26), (1402, 8), (1006, 34)}
    name, dtype = schemas[(26, 26)]
    assert name == "fdi_info"
    assert _layout(dtype) == [
        ("gyro_bias_raw_flag_device", "|u1", 0),
        ("gyrox_bias_raw_device", "<f4", 1),
        ("gyroy_bias_raw_device", "<f4", 5),
        ("gyroz_bias_raw_device", "<f4", 9),
        ("gyro_static_flag", "|u1", 13),
        ("wx_stdvar", "<f4", 14),
        ("wy_stdvar", "<f4", 18),
        ("wz_stdvar", "<f4", 22),
    ]
    assert schemas[(1402, 8)][1].names == (
        "simulator_compass_MagData[0]",
        "simulator_compass_MagData[1]",
        "simulator_compass_MagData[2]",
        "simulator_compass_cnt",
    )
    assert _layout(schemas[(1006, 34)][1]) == [
        ("g_api_emergency_break_flag", "|u1", 0),
        ("i8", "|i1", 1),
        ("u16", "<u2", 2),
        ("compass.mx", "<i2", 4),
        ("ctrl_tick", "<u4", 6),
        ("pack_current", "<i4", 10),
        ("hdop", "<f4", 14),
        ("lati", "<f8", 18),
        ("longti", "<f8", 26),
    ]


def test_dtype_views_packet_matrix_as_dataframe():
    dtype = _parse(MIXED)[(1006, 34)][1]
    rows = [
        struct.pack("<BbHhIifdd", 1, -2, 60000, -300, 4_000_000_000, -5, 0.5, 51.25, -0.125),
        struct.pack("<BbHhIifdd", 0, 7, 1, 2, 3, 4, -1.5, 0.0, 179.5),
    ]
    matrix = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(2, 34)

    df = pd.DataFrame(matrix.view(dtype)[:, 0])

    assert tuple(df.columns) == dtype.names
    assert df["compass.mx"].tolist() == [-300, 2]
    assert df["ctrl_tick"].tolist() == [4_000_000_000, 3]
    assert df["longti"].tolist() == [-0.125, 179.5]


def test_message_ends_at_nul_and_header_may_span_messages():
    # synthetic: header split over packets, bytes after the NUL ignored
    messages = [
        b"Op.[config]\x00",
        b"name\tsplit\x00junk",
        b"type\t7\x00",
        b"Op.int16_t\ta 0\x00\xff",
    ]
    assert {k: (n, d.names) for k, (n, d) in parse_definitions(messages).items()} == {
        (7, 2): ("split", ("a",))
    }


def test_no_definitions_gives_empty_mapping():
    assert _parse([]) == {}
    assert _parse(TRAILER) == {}


def test_lost_field_never_matches_full_width():
    # FLY013 lost one definition packet mid-section: the remaining layout is
    # NOT a prefix of the real one, so it must not claim the real width.
    lost = SIM_SENSOR[:2] + SIM_SENSOR[3:]
    schemas = _parse(lost)
    assert set(schemas) == {(1402, 6)}
    assert dict(schemas[(1402, 6)][1].fields or {})["simulator_compass_cnt"][1] == 4


def test_lost_fdi_info_line_falls_back_to_known_m100_layout():
    truncated = FDI_INFO[:3] + FDI_INFO[4:]  # exactly FLY013's definition stream
    complete = _parse(FDI_INFO)[(26, 26)]

    schemas = _parse(truncated + SIM_SENSOR)

    assert schemas[(26, 22)][1].names == tuple(
        n for n in (complete[1].names or ()) if n != "gyroy_bias_raw_device"
    )
    assert schemas[(26, 26)] == complete
    assert (1402, 8) in schemas


def test_logged_layout_wins_over_builtin():
    own = ["Op.[config]\nname\tfdi_other\ntype\t26"] + ["Op.uint16_t\tw 0"] * 13  # synthetic
    schemas = _parse(own)
    assert set(schemas) == {(26, 26)}
    assert schemas[(26, 26)][0] == "fdi_other"
    assert (schemas[(26, 26)][1].names or ())[:3] == ("w", "w_2", "w_3")


def test_unsupported_field_type_drops_only_its_section():
    bad = SIM_SENSOR[:2] + ["Op.uint64_t\tstamp 0"] + SIM_SENSOR[2:]  # synthetic
    with pytest.warns(UserWarning, match="simulator_sensor_data_25HZ.*unsupported line"):
        schemas = parse_definitions(_messages(MIXED + bad + FDI_INFO))
    assert set(schemas) == {(1006, 34), (26, 26)}


@pytest.mark.parametrize(
    "header",
    [
        "Op.[config]\nname\tno_type\ntype\tx1",
        "Op.[config]\ntype\t5",
        "Op.[config] trailing\nname\tjunk\ntype\t5",
        "Op.[config]\nname\ttoo_big\ntype\t70000",
        "Op.[bitfield]\nname\tunknown_kind\ntype\t5",
    ],
)
def test_malformed_header_never_extends_previous_section(header):
    with pytest.warns(UserWarning, match="ignored"):
        schemas = parse_definitions(_messages(SIM_SENSOR + [header, "Op.int32_t\tx 0"]))
    assert set(schemas) == {(1402, 8)}


@pytest.mark.parametrize(
    "line",
    ["Op.float\t0", "Op.float\ta 1 b", "Op.uint8_t", "posFlg[0] 0", "Op.uint8_t\tx 0\x01"],
)
def test_malformed_field_line_drops_section(line):
    with pytest.warns(UserWarning, match="ignored"):
        schemas = parse_definitions(_messages(SIM_SENSOR[:2] + [line] + SIM_SENSOR[2:]))
    assert schemas == {}


def test_field_in_text_section_is_rejected_not_attached():
    with pytest.warns(UserWarning, match="non-binary"):
        schemas = parse_definitions(_messages(SIM_SENSOR + TRAILER[:1] + ["Op.uint8_t\tx 0"]))
    assert set(schemas) == {(1402, 8)}


def test_orphan_lines_warn():
    with pytest.warns(UserWarning, match="outside a section"):
        assert parse_definitions(_messages(["Op.uint8_t\tx 0"])) == {}


def test_repeated_definitions_dedupe_and_conflicts_drop():
    assert set(_parse(SIM_SENSOR + MIXED + SIM_SENSOR)) == {(1402, 8), (1006, 34)}

    renamed = [SIM_SENSOR[0], "Op.uint16_t\tother 0"] + SIM_SENSOR[2:]  # same width, new layout
    with pytest.warns(UserWarning, match="disagree"):
        schemas = parse_definitions(_messages(SIM_SENSOR + renamed + SIM_SENSOR))
    assert schemas == {}


def test_conflicting_fdi_info_gets_no_builtin():
    other = ["Op.[config]\nname\tfdi_info\ntype\t26"] + ["Op.uint16_t\tw 0"] * 13  # synthetic
    with pytest.warns(UserWarning, match="disagree"):
        assert parse_definitions(_messages(FDI_INFO + other)) == {}
