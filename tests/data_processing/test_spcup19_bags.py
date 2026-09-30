"""``spcup19_bags.read_bag_audio`` on tiny synthetic HARK bags (no real data).

The bags mimic KumamoTech's: overlapping ``hark_msgs/HarkWave`` frames
(512 samples, 160-sample advance, zero header stamps, contiguous ``count``)
plus the ``/position`` / ``/attitude`` / ``/sound_sources_position`` topics.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("rosbags")

from rosbags.rosbag1 import Writer  # noqa: E402
from rosbags.typesys import Stores, get_types_from_msg, get_typestore  # noqa: E402

from data_processing.sources.spcup19_bags import read_bag_audio  # noqa: E402

SR, LENGTH, ADVANCE, NCH = 16000, 512, 160, 3
T0_NS = 1_552_162_818_000_000_000
HARK_DEFS = {
    "hark_msgs/msg/HarkWaveVal": "float32[] wavedata",
    "hark_msgs/msg/HarkWave": (
        "std_msgs/Header header\nint32 count\nint32 nch\nint32 length\n"
        "int32 data_bytes\nHarkWaveVal[] src"
    ),
}


def _typestore():
    store = get_typestore(Stores.ROS1_NOETIC)
    for name, text in HARK_DEFS.items():
        store.register(get_types_from_msg(text, name))
    return store


def _write_bag(path, signal, counts, *, latency_s=0.004):
    """Frame ``signal`` (C, T) HARK-style; frame k ends at sample 512 + 160 k."""
    store = _typestore()
    wave_t = store.types["hark_msgs/msg/HarkWave"]
    val_t = store.types["hark_msgs/msg/HarkWaveVal"]
    header_t = store.types["std_msgs/msg/Header"]
    time_t = store.types["builtin_interfaces/msg/Time"]
    point_t = store.types["geometry_msgs/msg/Point"]
    quat_t = store.types["geometry_msgs/msg/Quaternion"]
    string_t = store.types["std_msgs/msg/String"]
    with Writer(path) as writer:
        wave = writer.add_connection("/HarkWave", "hark_msgs/msg/HarkWave", typestore=store)
        pos = writer.add_connection("/position", "geometry_msgs/msg/Point", typestore=store)
        att = writer.add_connection("/attitude", "geometry_msgs/msg/Quaternion", typestore=store)
        src = writer.add_connection(
            "/sound_sources_position", "std_msgs/msg/String", typestore=store
        )
        events = []
        for k, count in enumerate(counts):
            end = LENGTH + ADVANCE * k
            frame = signal[:, end - LENGTH : end]
            msg = wave_t(
                header=header_t(seq=k, stamp=time_t(sec=0, nanosec=0), frame_id=""),
                count=count,
                nch=NCH,
                length=LENGTH,
                data_bytes=NCH * LENGTH * 4,
                src=[val_t(wavedata=frame[c].astype(np.float32)) for c in range(NCH)],
            )
            t_ns = T0_NS + round((end / SR + latency_s) * 1e9)
            events.append((t_ns, wave, store.serialize_ros1(msg, wave.msgtype)))
        for i, t_s in enumerate((0.05, 0.25)):
            t_ns = T0_NS + round(t_s * 1e9)
            events.append(
                (t_ns, pos, store.serialize_ros1(point_t(x=float(i), y=2.0, z=0.5), pos.msgtype))
            )
            quat = quat_t(x=0.0, y=0.0, z=-0.667, w=-0.745)
            events.append((t_ns + 1, att, store.serialize_ros1(quat, att.msgtype)))
        text = string_t(data="sound_source1_(x,y)=(5.658837,3.437567)")
        events.append((T0_NS + 1000, src, store.serialize_ros1(text, src.msgtype)))
        for t_ns, conn, raw in sorted(events, key=lambda e: e[0]):
            writer.write(conn, t_ns, raw)


def _signal(n_frames, seed=0):
    rng = np.random.default_rng(seed)
    n = LENGTH + ADVANCE * (n_frames - 1)
    return np.round(rng.normal(0.0, 3000.0, size=(NCH, n))).astype(np.float32)


def test_overlapping_frames_rebuild_the_continuous_stream(tmp_path):
    n_frames = 40
    signal = _signal(n_frames)
    bag = tmp_path / "hark.bag"
    _write_bag(bag, signal, counts=range(1000, 1000 + n_frames))

    audio, sr, info = read_bag_audio(bag)

    assert sr == SR
    assert audio.dtype == np.float32
    np.testing.assert_array_equal(audio, signal / np.float32(32768.0))
    assert (info["frame_length"], info["advance"], info["n_frames"]) == (LENGTH, ADVANCE, n_frames)
    assert info["count_range"] == (1000, 1000 + n_frames - 1)
    # Audio sample 0 is dated by the lateness floor (minimum latency taken as
    # zero): 4 ms after its nominal time, so bag-time telemetry shifts by -4 ms.
    assert info["t0_ns"] == T0_NS + 4_000_000
    np.testing.assert_allclose(info["position"]["t"], [0.046, 0.246], atol=1e-6)
    np.testing.assert_array_equal(info["position"]["xyz"], [[0.0, 2.0, 0.5], [1.0, 2.0, 0.5]])
    assert info["attitude"]["xyzw"].shape == (2, 4)
    assert info["sound_sources"] == {"sound_source1": (5.658837, 3.437567)}


def test_dropped_frame_is_refused(tmp_path):
    n_frames = 12
    signal = _signal(n_frames, seed=1)
    counts = list(range(n_frames))
    counts[6:] = [c + 1 for c in counts[6:]]  # count jumps 5 -> 7
    bag = tmp_path / "gap.bag"
    _write_bag(bag, signal, counts=counts)

    with pytest.raises(ValueError, match="count jumps 5 -> 7"):
        read_bag_audio(bag)
