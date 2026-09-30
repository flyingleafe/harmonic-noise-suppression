"""ROS-bag readers for the SPCUP19 ego-noise packages that ship bags.

KumamoTech ships its two outdoor flights only as ROS1 bags (``data1.bag``,
``data2.bag``), recorded on the ground station by HARK + MAVROS from an
enRoute Zion PG560 streaming a 16-ch RASP-ZX array over Wi-Fi:

- ``/HarkWave`` (``hark_msgs/HarkWave``, 100 msg/s): the audio as HARK
  frames. Each message holds ``nch`` = 16 ``src[c].wavedata`` arrays of
  ``length`` = 512 samples; consecutive frames OVERLAP and advance by 160
  samples (frame ``k+1``'s first 352 samples are frame ``k``'s last 352), so
  the continuous signal is frame 0 followed by the last 160 samples of every
  later frame. Values are integer-valued float32 (peak 11750 on both bags),
  scaled here by the int16 full scale. ``header.stamp`` is zero on every
  message: only the bag receive time dates a frame. The HARK frame counter
  ``count`` and ``header.seq`` are contiguous on both bags.
- ``/position`` (``geometry_msgs/Point``, ~2 Hz): metres, x = East,
  y = North, z = up, origin = take-off point (report Figs. 2-3 axes).
- ``/attitude`` (``geometry_msgs/Quaternion``, ~4 Hz): UAV attitude,
  ``(x, y, z, w)`` as published by MAVROS.
- ``/sound_sources_position`` (``std_msgs/String``, 2 msgs):
  ``sound_source<i>_(x,y)=(x,y)``, the static sources in the same E/N frame.

The sample rate is not carried by the messages: RASP-ZX + HARK record at
16 kHz, and the frame arrival times confirm it (15996 Hz / 15983 Hz implied
over 87 s / 179 s). Telemetry times are mapped onto the audio clock through
the lower envelope of the frame arrival lateness, which absorbs the
receive-clock drift and the one 0.24 s arrival step of ``data2.bag``
(``info["arrival_lateness_s"]``).

Shout COOEE!'s ``2019-02-04-15-53-42.bag`` has no reader: it is the MAVROS/
OptiTrack log of an UNPUBLISHED dynamic flight (armed 20.8 s, airborne
24.8-232.3 s ≈ the 205 s ``file 3`` of ``extract_pos.m``), not of either
published ~80 s static-hover wav.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

import numpy as np

HARK_TOPIC = "/HarkWave"
POSITION_TOPIC = "/position"
ATTITUDE_TOPIC = "/attitude"
SOURCES_TOPIC = "/sound_sources_position"
#: RASP-ZX / HARK capture rate (not stored in ``HarkWave``; checked on arrival times).
HARK_SAMPLE_RATE = 16000
#: HarkWave values are integer-valued floats in the int16 range.
HARK_FULL_SCALE = 32768.0
#: Largest tolerated |arrival-implied rate / sample_rate - 1|.
MAX_RATE_MISMATCH = 0.01
#: Window [s] of the running minimum that estimates the arrival-lateness floor.
LATENESS_WINDOW_S = 2.0
_SOURCE_RE = re.compile(r"(\w+?)_\(x,y\)=\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)")


def _frame_advance(first: np.ndarray, second: np.ndarray) -> int:
    """Smallest hop ``h`` with ``second[:, :L-h] == first[:, h:]`` (``L`` if none)."""
    length = first.shape[1]
    for hop in range(1, length):
        if np.array_equal(second[:, : length - hop], first[:, hop:]):
            return hop
    return length


def _lateness_floor(recv_s: np.ndarray, lateness: np.ndarray, frames_per_window: int):
    """``(window centre times, window minima)`` of the per-frame arrival lateness."""
    starts = np.arange(0, len(lateness), frames_per_window)
    floor = np.minimum.reduceat(lateness, starts)
    ends = np.append(starts[1:], len(lateness)) - 1
    return 0.5 * (recv_s[starts] + recv_s[ends]), floor


def read_bag_audio(
    bag_path: str | Path, sample_rate: int = HARK_SAMPLE_RATE
) -> tuple[np.ndarray, int, dict[str, Any]]:
    """``((C, T) float32 audio, sr, info)`` from a HARK ``/HarkWave`` ROS1 bag.

    Frames must be contiguous (``count`` steps by exactly 1) and overlap
    consistently; anything else raises ``ValueError`` rather than splicing.
    ``info`` holds the stream facts (``n_frames``, ``frame_length``,
    ``advance``, ``count_range``, ``seq_range``, ``duration_s``,
    ``bag_duration_s``, ``arrival_rate_hz``, ``arrival_lateness_s``,
    ``t0_ns`` = bag time of audio sample 0, taking the minimum delivery
    latency as zero) and, when the bag has them,
    ``position`` (``t`` s on the audio clock, ``xyz`` m), ``attitude``
    (``t``, ``xyzw``) and ``sound_sources`` (``{name: (x, y)}`` m).

    ``rosbags`` is imported lazily: only this ingest path needs it.
    """
    from rosbags.highlevel import AnyReader

    chunks: list[np.ndarray] = []
    recv_ns: list[int] = []
    counts: list[int] = []
    seqs: list[int] = []
    prev: np.ndarray | None = None
    length = advance = 0
    tele: dict[str, list] = {POSITION_TOPIC: [], ATTITUDE_TOPIC: [], SOURCES_TOPIC: []}
    with AnyReader([Path(bag_path)]) as reader:
        bag_span = (int(reader.start_time), int(reader.end_time))
        audio_conns = [c for c in reader.connections if c.topic == HARK_TOPIC]
        if not audio_conns:
            raise ValueError(f"{bag_path}: no {HARK_TOPIC} connection")
        for conn, t_ns, raw in reader.messages(connections=audio_conns):
            msg = cast(Any, reader.deserialize(raw, conn.msgtype))
            frame = np.stack([np.asarray(s.wavedata, dtype=np.float32) for s in msg.src])
            if frame.shape != (msg.nch, msg.length):
                raise ValueError(
                    f"{bag_path}: frame {msg.count} is {frame.shape}, header says ({msg.nch}, {msg.length})"
                )
            if prev is None:
                length = frame.shape[1]
                chunks.append(frame)
            else:
                if msg.count != counts[-1] + 1:
                    raise ValueError(
                        f"{bag_path}: HarkWave count jumps {counts[-1]} -> {msg.count}"
                    )
                if frame.shape != prev.shape:
                    raise ValueError(f"{bag_path}: frame shape changed at count {msg.count}")
                if not advance:
                    advance = _frame_advance(prev, frame)
                overlap = length - advance
                if not np.array_equal(frame[:, :overlap], prev[:, advance:]):
                    raise ValueError(
                        f"{bag_path}: frame {msg.count} breaks the {advance}-sample overlap"
                    )
                chunks.append(frame[:, overlap:])
            prev = frame
            recv_ns.append(int(t_ns))
            counts.append(int(msg.count))
            seqs.append(int(msg.header.seq))
        tele_conns = [c for c in reader.connections if c.topic in tele]
        for conn, t_ns, raw in reader.messages(connections=tele_conns):
            tele[conn.topic].append((int(t_ns), reader.deserialize(raw, conn.msgtype)))

    audio = np.concatenate(chunks, axis=1)
    audio /= np.float32(HARK_FULL_SCALE)
    advance = advance or length

    # Arrival lateness of each frame's last sample relative to a nominal clock.
    recv = np.asarray(recv_ns, dtype=np.int64)
    recv_s = (recv - recv[0]) / 1e9
    end_sample = length + advance * np.arange(len(recv))
    lateness = recv_s - end_sample / sample_rate
    arrival_rate = sample_rate
    if len(recv) > 1 and recv_s[-1] > 0:
        arrival_rate = float(advance / np.polyfit(end_sample / advance, recv_s, 1)[0])
        if abs(arrival_rate / sample_rate - 1.0) > MAX_RATE_MISMATCH:
            raise ValueError(
                f"{bag_path}: frames arrive at {arrival_rate:.0f} Hz, not {sample_rate} Hz"
            )
    per_window = max(1, round(LATENESS_WINDOW_S * sample_rate / advance))
    floor_t, floor = _lateness_floor(recv_s, lateness, per_window)

    def audio_clock(t_ns: np.ndarray) -> np.ndarray:
        rel = (np.asarray(t_ns, dtype=np.int64) - recv[0]) / 1e9
        return rel - np.interp(rel, floor_t, floor)

    info: dict[str, Any] = {
        "topic": HARK_TOPIC,
        "n_channels": int(audio.shape[0]),
        "n_frames": len(recv),
        "frame_length": int(length),
        "advance": int(advance),
        "count_range": (counts[0], counts[-1]),
        "seq_range": (seqs[0], seqs[-1]),
        "duration_s": audio.shape[1] / sample_rate,
        "bag_duration_s": (bag_span[1] - bag_span[0]) / 1e9,
        "arrival_rate_hz": arrival_rate,
        "arrival_lateness_s": (float(floor.min() - floor[0]), float(floor.max() - floor[0])),
        "t0_ns": int(recv[0] + round(floor[0] * 1e9)),
        "full_scale": HARK_FULL_SCALE,
    }
    if tele[POSITION_TOPIC]:
        t_ns, msgs = zip(*tele[POSITION_TOPIC], strict=True)
        info["position"] = {
            "t": audio_clock(np.asarray(t_ns)),
            "xyz": np.asarray([(m.x, m.y, m.z) for m in msgs], dtype=np.float64),
        }
    if tele[ATTITUDE_TOPIC]:
        t_ns, msgs = zip(*tele[ATTITUDE_TOPIC], strict=True)
        info["attitude"] = {
            "t": audio_clock(np.asarray(t_ns)),
            "xyzw": np.asarray([(m.x, m.y, m.z, m.w) for m in msgs], dtype=np.float64),
        }
    if tele[SOURCES_TOPIC]:
        info["sound_sources"] = {
            name: (float(x), float(y))
            for _, msg in tele[SOURCES_TOPIC]
            for name, x, y in _SOURCE_RE.findall(msg.data)
        }
    return audio, sample_rate, info
