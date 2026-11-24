"""Built-in payload schemas for M100 ``DJI_LOG_V3`` packets.

The firmware embeds field definitions (type 65533 RecDefs) for only part of
its packet types. :data:`BUILTIN` covers the native packets in the November
FLY013/FLY018/FLY022 logs whose layout is documented elsewhere. It maps
``(record_type, payload_length)`` to ``(group, dtype)``. ``dtype`` is a packed
little-endian structured dtype. Its ``itemsize`` equals ``payload_length``,
and every field is a scalar. One group's XOR-decoded payload matrix
``rows: uint8[n, payload_length]`` therefore views losslessly as
``rows.view(dtype)[:, 0]``.

Sources (layouts only; no code is copied):

- DatCon (https://github.com/BudWalkerJava/DatCon, ``src/DatConRecs``): ``RecIMU``,
  ``IMUEX40``, ``GpsGroup``, ``RecMotor152_10090``, ``MagGroup`` and the
  DJI-viewer classes in ``FromViewer`` and ``FromOtherV3Dats``.
- https://github.com/o-gs/dji-firmware-tools (``comm_dissector/wireshark/dji-{p3,mavic}-flyrec-proto.lua``).
- The embedded RecDef ``esc_data`` (type 10090, 152 B) found in all three logs.

Each layout was checked against real packets and against the paired DatCon
CSV exports (see ``tests/data_processing/test_dat_records.py``). Where the
references disagree, the data decides. In ``imu_atti``, the noisy baro channel
is at offset 16 (``press``) and the fused altitude at 44 (``alti``), matching
the Mavic dissector and DatCon's ``press:D``/``alti:D``. The P3 dissector has
them swapped. Field names come from the reference that matches the data, with
the dissector's group prefix and the DJI instance suffix (``0``/``_00``)
removed so instance groups share names. Obvious typos (``longtitude``,
``fixed_altitedue``, ``imu_init_fail_reson``) are corrected. A field named
``_b<offset>`` is a uint8 byte that no reference documents for this payload
size. Such fields occur only in groups that are otherwise documented.

Packet types without any documented layout are left out on purpose. These
are 161, 199, 300, 2128, 2144, 2160, 2172, 2224, 2240, 2275, 2305, 2307, 2314,
2336, 2337, 2340, 2341 and 2343. Readers keep their payloads raw.
"""

from __future__ import annotations

import numpy as np

#: Motor block order in the ESC payload (= DatCon ``Motor:*`` suffixes).
ROTORS = ("RFront", "LFront", "LBack", "RBack")


def _bytes(start: int, stop: int) -> list[tuple[str, str]]:
    return [(f"_b{i}", "u1") for i in range(start, stop)]


def _xyz(prefix: str, fmt: str, axes: str = "xyz") -> list[tuple[str, str]]:
    return [(f"{prefix}{a}", fmt) for a in axes]


# fmt: off
# DatCon RecIMU / dji-mavic imu_atti_0 (0x0800); same layout for type 1.
_IMU_ATTI = [
    ("long", "f8"),  # rad
    ("lati", "f8"),  # rad
    ("press", "f4"),  # baro altitude, m (raw)
    *_xyz("a", "f4"),  # accelerometer, g
    *_xyz("w", "f4"),  # gyro, rad/s
    ("alti", "f4"),  # fused altitude, m
    ("qw", "f4"), ("qx", "f4"), ("qy", "f4"), ("qz", "f4"),
    *_xyz("ag", "f4"),
    *_xyz("vg", "f4"),  # velocity N/E/D, m/s
    *_xyz("gb", "f4"),  # gyro bias
    *_xyz("m", "i2"),  # magnetometer
    *_xyz("t", "i2"),  # temperatures, 0.01 degC
    ("sensor_stat", "u2"),
    ("filter_stat", "u2"),
    ("svn", "u2"),  # satellites
    ("atti_cnt", "u2"),
]

# DatCon IMUEX40 / dji-p3 imu_ex_00 (0x0810); same layout for type 3.
_IMU_EX = [
    ("vo_vx", "f4"), ("vo_vy", "f4"), ("vo_vz", "f4"),
    ("vo_px", "f4"), ("vo_py", "f4"), ("vo_pz", "f4"),
    ("us_v", "f4"), ("us_p", "f4"),
    ("vo_flag_navi", "u2"), ("imu_err_flag", "u2"), ("vo_flag_rsv", "u2"), ("imu_ex_cnt", "u2"),
]

# DatCon FromViewer temp_cali_data_00_2195 / app_temp_cali_data_00_2198 (84 B).
# The logs carry 100 B. The documented 0..55 prefix fits the data
# (temp_min -14, temp_max 55, temp_temp ~55 degC). Bytes 56..99 do not
# follow the 84 B tail (base_* is shifted by one float), so they stay raw.
_TEMP_CALI = [
    ("start_flag", "u1"), ("state", "i1"), ("cali_cnt", "u1"), ("temp_ready", "u1"),
    ("step", "u1"), ("cali_type", "u1"), ("tick", "u2"),
    ("grav_acc_x", "i1"), ("grav_acc_y", "i1"), ("grav_acc_z", "i1"),
    ("dst_cali_temp", "i1"), ("dst_cali_temp_1", "f4"),
    ("temp_min", "f4"), ("temp_max", "f4"),
    ("temp_cali_status", "u1"), ("base_cali_status", "u1"),
    ("cfg_temp_cali_fw_version", "u1"), ("cur_temp_cali_fw_version", "u1"),
    *_xyz("temp_bw_", "f4"), *_xyz("temp_ba_", "f4"), ("temp_temp", "f4"),
    *_bytes(56, 100),
]


def _motor() -> list[tuple[str, str]]:
    # 19-byte ESC blocks. Offsets come from DatCon RecMotor152_10090. Types
    # follow the in-file esc_data RecDef. The speed (RPM) uses the bare rotor
    # name. Blocks 5..8 are zero on the 4-rotor M100 and keep RecDef names.
    names = [
        [f"status_{r}", f"current_{r}", r, f"voltage_{r}", f"temperature_{r}",
         f"ppm_recv_{r}", f"v_out_{r}", f"debug0_{r}", f"debug1_{r}", f"debug2_{r}"]
        for r in ROTORS
    ] + [
        [f"esc{i}_{f}" for f in ("status", "current", "speed", "voltage", "temperature",
                                 "ppm_recv", "v_out", "debug0", "debug1", "debug2")]
        for i in range(5, 9)
    ]
    fmts = ("u1", "i2", "i2", "u2", "i2", "u2", "u2", "u2", "i2", "i2")
    return [(n, f) for block in names for n, f in zip(block, fmts)]


_LAYOUTS: dict[tuple[int, int], tuple[str, list[tuple[str, str]]]] = {
    (1, 120): ("imu_atti", _IMU_ATTI),
    (2048, 120): ("imu_atti_00", _IMU_ATTI),
    (3, 40): ("imu_ex", _IMU_EX),
    (2064, 40): ("imu_ex_00", _IMU_EX),
    # DatCon FromViewer imu_init_6 = dji-p3 imu_init.
    (6, 54): ("imu_init", [
        *_xyz("imu_offset_", "f4"), *_xyz("gps_offset_", "f4"),
        ("imu_dir", "u2"), ("imu_key", "u1"), ("o_sw", "u1"),
        *_xyz("mag_bias_", "f4"), *_xyz("mag_scale_", "f4"),
        ("init_counter", "u2"),
    ]),
    # dji-p3 osd_general (50 B). The logs carry 52 B and bytes 50..51 are
    # undocumented. ``mode1`` bit 7 is the RC state; bits 0..6 are the flight mode.
    (12, 52): ("osd_general", [
        ("longitude", "f8"), ("latitude", "f8"),  # rad
        ("relative_height", "i2"),  # 0.1 m
        ("vgx", "i2"), ("vgy", "i2"), ("vgz", "i2"),
        ("pitch", "i2"), ("roll", "i2"), ("yaw", "i2"),  # 0.1 deg
        ("mode1", "u1"), ("latest_cmd", "u1"), ("controller_state", "u4"),
        ("gps_nums", "u1"), ("gohome_landing_reason", "u1"), ("start_fail_reason", "u1"),
        ("controller_state_ext", "u1"), ("batt_remain", "u1"), ("ultrasonic_height", "u1"),
        ("motor_startup_time", "u2"), ("motor_startup_times", "u1"),
        ("bat_alarm1", "u1"), ("bat_alarm2", "u1"), ("version_match", "u1"),
        ("product_type", "u1"), ("imu_init_fail_reason", "i1"),
        *_bytes(50, 52),
    ]),
    # DatCon FromViewer osd_home_13 (26 B). The logs carry 36 B; the
    # dissector only has placeholder names after byte 26.
    (13, 36): ("osd_home", [
        ("osd_lon", "f8"), ("osd_lat", "f8"),  # rad
        ("osd_alt", "f4"), ("osd_home_state", "u2"), ("fixed_altitude", "u2"),
        ("course_lock_torsion", "i2"),
        *_bytes(26, 36),
    ]),
    # DatCon FromViewer battery_info45_17 = dji-p3 battery_info (first 45 B).
    (17, 45): ("battery_info", [
        ("design_capacity", "u2"), ("full_charge_capacity", "u2"),
        ("remaining_capacity", "u2"),  # mAh
        ("pack_voltage", "u2"),  # mV
        ("current", "i2"),  # mA, negative = discharge
        ("life_percentage", "u1"), ("capacity_percentage", "u1"),
        ("temperature", "i2"),  # degC
        ("cycle_count", "u2"), ("serial_number", "u2"),
        *[(f"cell{i}", "u2") for i in range(1, 7)],  # mV
        ("average_current", "i2"), ("right", "u1"),
        ("error_count", "u4"), ("n_discharge_times", "u4"), ("current_status", "u4"),
    ]),
    # DatCon FromViewer smart_battery_info_18 / dji-p3 smart_battery_info,
    # truncated at the 58 B boundary (after g_filter_vol).
    (18, 58): ("smart_battery_info", [
        ("rest_time", "u2"), ("need_time_for_gohome", "u2"), ("need_time_for_land", "u2"),
        ("gohome_battery_level", "u2"), ("land_battery_level", "u2"),
        ("radius_for_gohome", "f4"), ("request_gohome", "u2"), ("bat_dec_speed", "f4"),
        ("smart_battery_state", "u4"),
        *[(n, "u1") for n in (
            "level1_over_current", "level2_over_current", "level1_over_temp",
            "level2_under_temp", "level1_low_temp", "level2_low_temp", "short_cir",
            "low_vol_cells", "damage_cells", "exchange_cells", "user_gohome_level",
            "user_land_level", "user_action_for_gohome", "user_action_for_land",
            "user_use_smart_bat", "flag_main_vol_low_gohome", "flag_main_vol_low_land",
            "flag_user_gohome", "flag_user_land", "flag_smart_bat_gohome",
            "flag_smart_bat_land", "flag_cell_err", "flag_communite_err")],
        ("real_desc_speed", "f4"),
        ("flag_vol_very_low", "u1"), ("flag_temp_and_vol_low", "u1"),
        ("flag_first_charge_not_full", "u1"),
        ("g_filter_vol", "f4"),
    ]),
    (19, 2): ("imu_data_status", [("start_fan", "u1"), ("led_status", "u1")]),
    # DatCon FromViewer PWM_OUTPUT_20 = dji-p3 pwm_output.
    (20, 16): ("pwm_output", [(f"M{i}", "u2") for i in range(1, 9)]),
    (40, 84): ("airport_limit_data", [
        ("area_state", "u1"), ("action_state", "u1"), ("work_point_num", "u1"),
        *[f for k in range(3) for f in (
            (f"lat_int_{k}", "i4"), (f"lon_int_{k}", "i4"), (f"point_from_where_{k}", "u1"),
            (f"flag_limit_radius_{k}", "u1"), (f"flag_limit_hi_{k}", "u1"),
            (f"d2limit_edge_{k}", "f4"), (f"d2limit_hi_{k}", "f4"),
            (f"directx_{k}", "f4"), (f"directy_{k}", "f4"))],
    ]),
    (115, 4): ("asr", [("lead", "u4")]),
    # dji-p3 gps_glns (type 5) layout; DatCon GpsGroup decodes 2096 as GPS 0.
    # Offset 40 is the fix type (3 = 3D), as the dissector says; DatCon's
    # "hacc" at 40 is contradicted by the data. The references disagree on
    # 44/48/52 (DatCon sAcc = 44/100; P3: gnss_flag, hacc, sacc), so those
    # floats get neutral names.
    (2096, 68): ("gps_glns_00", [
        ("gps_date", "u4"),  # yyyymmdd
        ("gps_time", "u4"),  # hhmmss UTC
        ("gps_lon", "i4"), ("gps_lat", "i4"),  # 1e-7 deg
        ("hmsl", "i4"),  # mm
        ("vel_n", "f4"), ("vel_e", "f4"), ("vel_d", "f4"),  # cm/s
        ("hdop", "f4"), ("pdop", "f4"),  # x100
        ("gps_fix", "f4"),
        ("acc_44", "f4"), ("acc_48", "f4"), ("acc_52", "f4"),
        ("gps_used", "u4"), ("gln_used", "u4"), ("numsv", "u2"), ("gpsglns_cnt", "u2"),
    ]),
    # dji-p3 gps_snr (type 11) layout. In the logs, the number of entries
    # with bit 7 set equals gps_glns_00 gps_used/gln_used packet for packet.
    # The low 7 bits are the SNR.
    (2112, 65): ("gps_snr_00", [
        *[(f"gps_snr{i}", "u1") for i in range(1, 33)],
        *[(f"gln_snr{i}", "u1") for i in range(1, 33)],
        ("gln_cnt", "u1"),
    ]),
    (2176, 44): ("temp_ctl_data_00", [
        ("dst_value", "i2"), ("cur_dst_temp", "i2"), ("p", "i4"), ("i", "i4"), ("d", "i4"),
        ("dl_err", "i2"), ("real_ctl_out_per", "u1"), ("slope_type", "u1"),
        ("temp_ctl_slope", "u1"), ("t_finish", "u1"), ("err_last", "i2"),
        ("ctl_out_value", "u4"), ("real_ctl_out_value", "u4"),
        ("i_small", "i4"), ("out", "i4"), ("cnt", "u4"),
    ]),
    (2195, 100): ("temp_cali_data_00", _TEMP_CALI),
    (2198, 100): ("app_temp_cali_data_00", _TEMP_CALI),
    # DatCon FromOtherV3Dats ATTI_MINI0_2208 = dji-mavic atti_mini0.
    (2208, 40): ("atti_mini0", [
        ("s_qw", "f4"), ("s_qx", "f4"), ("s_qy", "f4"), ("s_qz", "f4"),
        ("s_pgz", "f4"), ("s_vgz", "f4"), ("s_agz", "f4"),
        ("s_rsv0", "u4"), ("s_rsv1", "u4"), ("s_cnt", "u4"),
    ]),
    # dji-mavic cali_mag_00 (DatCon MagGroup reads the first three).
    (2256, 8): ("cali_mag_00", [("c_mx", "i2"), ("c_my", "i2"), ("c_mz", "i2"), ("c_m_cnt", "u2")]),
    (52720, 42): ("uc_monitor", [(n, "u2") for n in (
        "cpu_usage_pcnt", "cpu_usage_task_b_pcnt", "cpu_usage_task_a_pcnt", "uc_resv0",
        "max_time_workqueue", "max_time_task_b", "max_time_task_a", "max_time_task_c",
        "max_time_task_d", "uc_resv1", "stack_usage_irq_pcnt", "stack_usage_workqueue_pcnt",
        "stack_usage_b_pcnt", "stack_usage_a_pcnt", "stack_usage_c_pcnt",
        "stack_usage_d_pcnt", "uc_resv2", "pend_cnt_a", "pend_cnt_b", "pend_cnt_c",
        "pend_cnt_d")]),
    (10090, 152): ("Motor", _motor()),
}
# fmt: on


def _dtype(key: tuple[int, int], fields: list[tuple[str, str]]) -> np.dtype:
    dt = np.dtype([(name, "<" + fmt) for name, fmt in fields])
    if dt.itemsize != key[1]:
        raise AssertionError(f"builtin {key}: dtype covers {dt.itemsize} B")
    return dt


#: ``(record_type, payload_length) -> (group, packed little-endian dtype)``.
BUILTIN: dict[tuple[int, int], tuple[str, np.dtype]] = {
    key: (group, _dtype(key, fields)) for key, (group, fields) in _LAYOUTS.items()
}


def _utc(date: np.ndarray, time: np.ndarray) -> np.ndarray:
    """``yyyymmdd`` + ``hhmmss`` integers -> ``datetime64[s]``; NaT where unset."""
    d = date.astype(np.int64)
    t = time.astype(np.int64)
    out = (
        (d // 10000 - 1970).astype("M8[Y]").astype("M8[M]") + (d // 100 % 100 - 1).astype("m8[M]")
    ).astype("M8[s]")
    out += (d % 100 - 1).astype("m8[D]")
    out += (t // 10000 * 3600 + t // 100 % 100 * 60 + t % 100).astype("m8[s]")
    out[d == 0] = np.datetime64("NaT")
    return out


def decode_builtin(rtype: int, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Engineered columns for a builtin group, *in addition to* its raw fields.

    ``rows`` is the contiguous, XOR-decoded ``uint8[n, payload_length]``
    matrix of one group. Returns ``{}`` for groups without conversions. The
    scale factors are the documented ones. DatCon applies them to
    GPS/ESC/IMU output, and the dissectors state units for GPS.
    """
    group, dtype = BUILTIN[(rtype, rows.shape[1])]
    r = np.ascontiguousarray(rows).view(dtype)[:, 0]
    if group in ("imu_atti", "imu_atti_00"):
        return {"lon_deg": np.degrees(r["long"]), "lat_deg": np.degrees(r["lati"])}
    if group == "osd_general":
        return {"lon_deg": np.degrees(r["longitude"]), "lat_deg": np.degrees(r["latitude"])}
    if group == "osd_home":
        return {"lon_deg": np.degrees(r["osd_lon"]), "lat_deg": np.degrees(r["osd_lat"])}
    if group == "gps_glns_00":
        return {
            "utc": _utc(r["gps_date"], r["gps_time"]),
            "lon_deg": r["gps_lon"] / 1e7,
            "lat_deg": r["gps_lat"] / 1e7,
            "hmsl_m": r["hmsl"] / 1e3,
            "vel_n_ms": r["vel_n"].astype(np.float64) / 100.0,
            "vel_e_ms": r["vel_e"].astype(np.float64) / 100.0,
            "vel_d_ms": r["vel_d"].astype(np.float64) / 100.0,
        }
    if group == "Motor":
        out: dict[str, np.ndarray] = {}
        for name in ROTORS:
            out[f"current_A_{name}"] = r[f"current_{name}"] / 100.0
            out[f"voltage_V_{name}"] = r[f"voltage_{name}"] / 10.0
        return out
    return {}
