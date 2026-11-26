"""The external speech reference must never masquerade as an onboard noise microphone."""

import hashlib

import numpy as np
import soundfile as sf

from data_processing.sources import kuleuven_ego
from data_processing.streams import frame_to_sample, sample_to_frame


def test_rpm_take_keeps_reference_in_channel_17_with_role_aligned_geometry(tmp_path, monkeypatch):
    root = tmp_path / "rpm4000"
    root.mkdir()
    files = {}
    pcm = np.arange(17 * 40, dtype=np.int16).reshape(17, 40) - 250
    for row, samples in enumerate(pcm):
        stem = f"Channel_{row + 1:02d}" if row < 16 else "Speech_Ref"
        path = root / f"RPM4000_{stem}.wav"
        sf.write(path, samples, 44100, subtype="PCM_16")
        blob = path.read_bytes()
        files[path.relative_to(tmp_path).as_posix()] = (
            row,
            len(blob),
            hashlib.md5(blob).hexdigest(),
        )
    monkeypatch.setattr(kuleuven_ego, "FILES", files)
    monkeypatch.setattr(kuleuven_ego, "RPMS", (4000,))
    ((_, built),) = kuleuven_ego.build(tmp_path)
    frame = sample_to_frame(frame_to_sample(built))

    audio = frame["audio"]
    np.testing.assert_array_equal(audio.data, pcm.astype(np.float32) / 32768)
    assert "reference_audio" not in frame
    channel = audio.dim_index("mic").labels
    assert len(set(channel)) == 17
    for field in ("mic_pos", "mic_role", "mic_site_possible"):
        assert frame[field].dim_index("mic").labels == channel
    assert frame["mic_role"].data.tolist() == (
        ["under-body main array"] * 12
        + ["under-rotor ego-noise reference"] * 4
        + ["external speech reference above loudspeaker"]
    )
    assert frame["meta"]["original_files"][16] == "rpm4000/RPM4000_Speech_Ref.wav"
    assert frame["meta"]["channel_names"] == list(channel)
    possible = np.asarray(frame["mic_site_possible"].data)
    assert possible[:12, :12].any(axis=1).all() and not possible[:12, 12:].any()
    assert possible[12:16, 12:].any(axis=1).all() and not possible[12:16, :12].any()
    np.testing.assert_array_equal(possible[13], possible[15])  # mirror-ambiguous side pair
    assert not possible[16].any()  # the loudspeaker mic cannot occupy any onboard site
    xyz = np.asarray(frame["mic_pos"].data)
    assert xyz.shape == (17, 3) and np.isnan(xyz[:8]).all()  # unknown cube-site assignment
    assert xyz[9, 0] < 0 < xyz[10, 0] and xyz[14, 0] < 0 < xyz[12, 0]
    assert np.isnan(xyz[[13, 15], 1]).all()
    np.testing.assert_array_equal(xyz[12:16, 2], np.zeros(4))
    np.testing.assert_equal(xyz[16], [2.0, 0.0, np.nan])  # stand/room axes, not body axes
