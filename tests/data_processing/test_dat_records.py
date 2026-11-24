"""Builtin M100 DAT payload schemas, checked on real FLY022 packets.

The hex fixtures are XOR-decoded payloads taken from
``15_Nov/FLY022.dat``. Expected values come from the DatCon 4.3.0 export of
the same flight after reversing DatCon's scaling. Coordinates and hardware
identifiers are replaced with synthetic values; real locations are not fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

from data_processing.sources._dat_records import BUILTIN, ROTORS, decode_builtin

# CSV row: Motor:Speed 4208/4178/4382/4128, EscTemp 42/40/42/40,
# Current 5.83/4.91/4.98/4.10 A, Volts 22.0/22.0/21.9/21.9.
MOTOR = (
    "0047027010dc002a002c06e6020280f3ffb10000eb015210dc0028002506db020280e5ffbb0000f2011e11db"
    "002a003806f90202801a00c600009a012010db0028001d06ce020280dfffdc00" + "00" * 76
)
# GPS:Time 231605, synthetic Lat 12.3456789, heightMSL 20.769, hDOP 0.9, pDOP 1.66,
# sAcc 7.20, numGPS 7, numGLNAS 6, numSV 13, velD -0.01.
GPS = (
    "fb283501b58803004040e1e415cd5b07215100000000000000000000000080bf0000b4420000264300004040"
    "00003444000030410000fb4307000000060000000d00b300"
)
# The SNR packet paired (same index) with this GPS packet; gps_used 7, gln_used 8.
GPS_3000 = (
    "fb283501a28b03004040e1e415cd5b07955b0000000080400000e0c0000030c100009c420000144300004040"
    "00404e440000304100c0064407000000080000000f00de00"
)
GPS_SNR_3000 = (
    "0000001a00000000001916aaaf0000000000000000000000b30000b1b100b2b2aeb2a70000000000001db4b0"
    "000000000000a7afaf0000000000000000000000e0"
)
# IMU_ATTI(0) row atti_cnt 25484: press:D 166.5665, alti:D 165.2817,
# accelX 0.078125, gyroX 0.2167439 deg/s, magX 641, temperature 54.96, numSats 14.
IMU = (
    "000000000000e0bf000000000000d03f059126430400a03dd81fe33bc94483bf6eea773b98b601bd5fd2f5bb"
    "1c482543b0bc7c3f4058b2bc4f631e3d09821c3e3e3d64bc0bae9bbe6a2fbcbdd33f273c32990e3e07ee0a3c"
    "d0fcf33a80e0863b37c0d33b8102cdfe06067815d710e60c000000000e008c63"
)
# Battery(0) row: cellVolts1 3.767, cellVolts6 3.784, totalVolts 22.7,
# current 11.566, Temp 30, RemainingCap 3013, battery% 70.
BATTERY = (
    "9411d110c50bac58d2d203461e000d000000b70ed30eb90ec70ed30ec80e7bd300000000000000000000000000"
)
# osd_data row: flightTime 314400 (raw 3144), relativeHeight 1.1, flyCState GPS_Atti (6).
OSD = (
    "000000000000e0bf000000000000d03f0b000200000000000a00efffb2ff06000dd094000f090000c500480c"
    "011e8a0a05000000"
)
TEMP_CALI = (
    "00000001000120030000000000000000000060c100005c42aaaa04041f437a3c6fc563bcb01be9ba3c0a113f"
    "5454563e5371963d18185c420000000022fe1ab8cfc981395a7f9b39c92320bb7ceda83b3ccf0fbb0f0f5c42"
    "000000000000000000000000"
)


def _rows(hex_payload: str) -> np.ndarray:
    return np.frombuffer(bytes.fromhex(hex_payload), dtype=np.uint8)[None, :].copy()


def _decode(rtype: int, hex_payload: str) -> np.void:
    rows = _rows(hex_payload)
    _, dtype = BUILTIN[(rtype, rows.shape[1])]
    return rows.view(dtype)[0, 0]


def test_motor_esc_fields_match_datcon_export():
    rec = _decode(10090, MOTOR)
    assert [int(rec[r]) for r in ROTORS] == [4208, 4178, 4382, 4128]
    assert [int(rec[f"temperature_{r}"]) for r in ROTORS] == [42, 40, 42, 40]
    assert [int(rec[f"status_{r}"]) for r in ROTORS] == [0, 0, 0, 0]
    assert int(rec["ppm_recv_RFront"]) == 1580 and int(rec["v_out_RFront"]) == 742
    assert all(int(rec[f"esc{i}_speed"]) == 0 for i in range(5, 9))
    extra = decode_builtin(10090, _rows(MOTOR))
    current = [extra[f"current_A_{r}"][0] for r in ROTORS]
    np.testing.assert_allclose(current, [5.83, 4.91, 4.98, 4.10])
    volts = [extra[f"voltage_V_{r}"][0] for r in ROTORS]
    np.testing.assert_allclose(volts, [22.0, 22.0, 21.9, 21.9])


def test_gps_fields_match_datcon_export():
    rec = _decode(2096, GPS)
    assert (int(rec["gps_date"]), int(rec["gps_time"])) == (20261115, 231605)
    assert (float(rec["hdop"]), float(rec["pdop"]), float(rec["acc_44"])) == (90.0, 166.0, 720.0)
    assert (int(rec["gps_used"]), int(rec["gln_used"]), int(rec["numsv"])) == (7, 6, 13)
    assert float(rec["gps_fix"]) == 3.0
    extra = decode_builtin(2096, _rows(GPS))
    assert extra["utc"][0] == np.datetime64("2026-11-15T23:16:05")
    assert extra["lat_deg"][0] == pytest.approx(12.3456789, abs=1e-7)
    assert extra["lon_deg"][0] == pytest.approx(-45.5, abs=1e-7)
    assert extra["hmsl_m"][0] == pytest.approx(20.769)
    assert extra["vel_d_ms"][0] == pytest.approx(-0.01)


def test_gps_utc_is_nat_before_fix():
    rows = _rows(GPS)
    rows[0, :8] = 0
    assert np.isnat(decode_builtin(2096, rows)["utc"][0])


def test_gps_snr_used_flags_count_used_satellites():
    gps, snr = _decode(2096, GPS_3000), _decode(2112, GPS_SNR_3000)
    gps_flags = [int(snr[f"gps_snr{i}"]) >= 128 for i in range(1, 33)]
    gln_flags = [int(snr[f"gln_snr{i}"]) >= 128 for i in range(1, 33)]
    used = (int(gps["gps_used"]), int(gps["gln_used"]))
    assert (sum(gps_flags), sum(gln_flags)) == used == (7, 8)


def test_imu_atti_press_and_alti_follow_the_data():
    rec = _decode(2048, IMU)
    assert float(rec["press"]) == pytest.approx(166.5665, abs=1e-4)
    assert float(rec["alti"]) == pytest.approx(165.2817, abs=1e-4)
    assert float(rec["ax"]) == pytest.approx(0.078125, abs=1e-6)
    assert np.degrees(float(rec["wx"])) == pytest.approx(0.2167439, abs=1e-6)
    ints = (int(rec["mx"]), int(rec["tx"]), int(rec["svn"]), int(rec["atti_cnt"]))
    assert ints == (641, 5496, 14, 25484)
    assert _decode(1, IMU)["atti_cnt"] == 25484  # type 1 shares the layout


def test_battery_info_matches_datcon_export():
    rec = _decode(17, BATTERY)
    assert int(rec["design_capacity"]) == 4500  # DatCon's "FullChargeCap"
    assert int(rec["remaining_capacity"]) == 3013
    assert (int(rec["pack_voltage"]), int(rec["current"])) == (22700, -11566)
    assert (int(rec["cell1"]), int(rec["cell6"])) == (3767, 3784)
    assert (int(rec["temperature"]), int(rec["capacity_percentage"])) == (30, 70)
    assert decode_builtin(17, _rows(BATTERY)) == {}


def test_osd_general_keeps_undocumented_tail_bytes():
    rec = _decode(12, OSD)
    assert (int(rec["relative_height"]), int(rec["motor_startup_time"])) == (11, 3144)
    assert int(rec["mode1"]) & 0x7F == 6
    assert (int(rec["_b50"]), int(rec["_b51"])) == (0, 0)


def test_temp_cali_documented_prefix_and_raw_tail():
    rec = _decode(2195, TEMP_CALI)
    assert (float(rec["temp_min"]), float(rec["temp_max"])) == (-14.0, 55.0)
    assert float(rec["temp_temp"]) == pytest.approx(55.0235, abs=1e-4)
    tail = bytes(int(rec[f"_b{i}"]) for i in range(56, 100))
    assert tail == bytes.fromhex(TEMP_CALI)[56:]
