"""KAIST drone fault (Zenodo 7779574) builder on synthetic TARs."""

from __future__ import annotations

import io
import tarfile

import numpy as np
import pytest
import soundfile as sf

from data_processing.sources import kaist_drone as kd


def _audio(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-0.5, 0.5, 8000)


def _encoded(x: np.ndarray, split: str) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, 16000, format="WAV" if split == "train" else "FLAC", subtype="PCM_24")
    return buf.getvalue()


def _tar(path, members: dict[str, np.ndarray]) -> None:
    with tarfile.open(path, "w") as tar:
        for name, x in members.items():
            raw = _encoded(x, name.split("/")[1])
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))


@pytest.fixture
def root(tmp_path, monkeypatch):
    same = _audio(3)
    tars = {
        "A": {
            "A/train/mic1/A_F_N_mic1_5_DuckPond_3_snr=11.1.wav": _audio(1),
            "A/train/mic2/A_F_N_mic2_5_DuckPond_3_snr=11.1.wav": _audio(2),
        },
        "B": {  # one name-conflicting draw, as in drone B "Right" valid/test
            "B/test/mic1/B_R_N_9_EoeunHill_2_snr=13.3.wav": _audio(4),
            "B/test/mic2/B_R_MF4_74_EoeunHill_2_snr=13.3.wav": _audio(5),
        },
        "C": {  # mic2 is a PCM copy of mic1
            "C/valid/mic1/C_R_PC4_7_WestDoor_1_snr=12.5.wav": same,
            "C/valid/mic2/C_R_PC4_7_WestDoor_1_snr=12.5.wav": same,
        },
    }
    sizes = {}
    for drone, members in tars.items():
        path = tmp_path / f"drone_{drone}.tar"
        _tar(path, members)
        sizes[path.name] = (path.stat().st_size, "")
    monkeypatch.setattr(kd, "ZENODO_FILES", sizes)
    monkeypatch.setattr(kd, "_DISPUTED_DRAWS", 1)
    return tmp_path


def test_one_mono_frame_per_released_waveform(root):
    frames = dict(kd.build(root))
    assert sorted(frames) == [
        "A/train/mic1/A_F_N_mic1_5_DuckPond_3_snr=11.1",
        "A/train/mic2/A_F_N_mic2_5_DuckPond_3_snr=11.1",
        "B/test/mic1/B_R_N_9_EoeunHill_2_snr=13.3",
        "B/test/mic2/B_R_MF4_74_EoeunHill_2_snr=13.3",
        "C/valid/mic1=mic2/C_R_PC4_7_WestDoor_1_snr=12.5",
    ]
    for frame in frames.values():
        assert frame["audio"].dims == ("time",)
        assert frame["audio"].shape == (8000,)

    a1 = frames["A/train/mic1/A_F_N_mic1_5_DuckPond_3_snr=11.1"]["meta"]
    assert list(a1["observation"]["mics"]) == ["mic1"]
    assert (
        a1["mixture_partner_file"]
        == "drone_A.tar:A/train/mic2/A_F_N_mic2_5_DuckPond_3_snr=11.1.wav"
    )
    assert a1["parent_window"] == "A_F_N_mic1_5"
    assert a1["window_offset_s_inferred"] == 1.25
    assert a1["label"]["health"] == "normal"

    merged = frames["C/valid/mic1=mic2/C_R_PC4_7_WestDoor_1_snr=12.5"]["meta"]
    assert list(merged["original_files"]) == [
        "drone_C.tar:C/valid/mic1/C_R_PC4_7_WestDoor_1_snr=12.5.wav",
        "drone_C.tar:C/valid/mic2/C_R_PC4_7_WestDoor_1_snr=12.5.wav",
    ]
    assert list(merged["observation"]["mics"]) == ["mic1", "mic2"]


def test_name_conflict_keeps_filename_label_and_flags_mic2(root):
    frames = dict(kd.build(root))
    mic1 = frames["B/test/mic1/B_R_N_9_EoeunHill_2_snr=13.3"]["meta"]
    mic2 = frames["B/test/mic2/B_R_MF4_74_EoeunHill_2_snr=13.3"]["meta"]
    assert mic1["label"]["health"] == "normal"
    assert "label_disputed" not in mic1["label"]
    assert mic2["label"]["fault_code"] == "MF4"
    assert mic2["label"]["label_disputed"]
    assert mic2["label"]["mixture_partner_fault_code"] == "N"
    assert mic2["label"]["mixture_partner_segment_index"] == 9
    assert "parent_window" not in mic2


def test_unexpected_name_conflict_raises(monkeypatch, root):
    path = root / "drone_A.tar"
    _tar(
        path,
        {
            "A/train/mic1/A_F_N_mic1_5_DuckPond_3_snr=11.1.wav": _audio(1),
            "A/train/mic2/A_F_MF1_mic2_6_DuckPond_3_snr=11.1.wav": _audio(2),
        },
    )
    monkeypatch.setitem(kd.ZENODO_FILES, path.name, (path.stat().st_size, ""))
    with pytest.raises(ValueError, match="unexpected name conflict"):
        dict(kd.build(root))
