"""KU Leuven ESAT-STADIUS UAV ego-noise + speech recordings (RDR doi:10.48804/PZAVUC).

Tengan et al. (ICA 2022, arXiv:2211.02690; thesis ch. 5-6): a MikroKopter MK
EASY Quadro V3 with a custom bottom frame, rigidly mounted on a stand in the
Siemens DISW semi-anechoic chamber (Leuven), throttle held at three fixed
levels (average rotor speed 4000 / 5000 / 6000 rpm, laser probe under one
blade). A loudspeaker 2 m from the stand base, near the floor, plays one male
VCTK utterance; an external microphone 0.2 m above it records the speech
reference. The UAV carries a 16-element electret array: channels 01-12 = the
main array, 13-16 = one microphone below each propeller (ego-noise
references). Each published WAV is one channel, 44.1 kHz PCM16 mono, the
17 files of a run equal length (``RPM<rpm>_Channel_<NN>.wav``,
``RPM<rpm>_Speech_Ref.wav``).

One Frame per RPM: all 17 equal-length files in acquisition order, including
the external speech reference in row 17 (zero-based index 16). Native PCM16
is stored exactly as normalized float32. ``audio``, ``mic_pos`` and ``mic_role``
have the same labeled mic axis. No per-sample RPM is published.

The figure identifies sets of microphone sites, not their within-group WAV
numbering. ``mic_pos`` gives only identifiable coordinate components; NaN
means unknown. ``mic_site_pos_body`` and ``mic_site_possible`` expose the
figure estimates and the admissible assignments, not an invented permutation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tdseries as td

DATASET = "KULeuven-UAV-egonoise"
DOI = "10.48804/PZAVUC"
RPMS = (4000, 5000, 6000)
CHANNEL_ROLES = (
    ("under-body main array",) * 12
    + ("under-rotor ego-noise reference",) * 4
    + ("external speech reference above loudspeaker",)
)
N_ONBOARD = 16
CHANNEL_NAMES = tuple(f"Ch {i:02d}: {role}" for i, role in enumerate(CHANNEL_ROLES, start=1))
N_CHANNELS = len(CHANNEL_NAMES)
#: Median onboard-minus-external speech lag (samples; early/middle/late 2 s
#: speech windows; GCC-PHAT, 300–4000 Hz, source diagnostics 37573572).
_SPEECH_LAG = {
    4000: [309.9, 311.1, 312.2],
    5000: [318.1, 314.5, 311.4],
    6000: [309.1, 311.7, 312.8],
}
_REFERENCE_CLOCK_NOTE = (
    "Co-recorded, but a shared hardware clock is unverified. Onboard speech arrives "
    "6.1–8.0 ms after the external reference; the median changes by 2–7 samples "
    "within a run. Native waveforms and acoustic/device delay are preserved."
)

#: RDR Dataverse dataset 689, version 1.1 (released 2022-08-29): relpath ->
#: (datafile id, bytes, md5). Datafile ids are immutable per published version.
FILES: dict[str, tuple[int, int, str]] = {
    "README.txt": (750, 1341, "d3b539fe31e5eff1bdd33fce77a1a4eb"),
    "rpm4000/RPM4000_Channel_01.wav": (706, 2867598, "dbc574b1ca04101e3d8f810bf58a966d"),
    "rpm4000/RPM4000_Channel_02.wav": (717, 2867598, "e1fc8508f83105f9ce878830a40687b8"),
    "rpm4000/RPM4000_Channel_03.wav": (696, 2867598, "31944be809f9be2a783bfd84104beec3"),
    "rpm4000/RPM4000_Channel_04.wav": (723, 2867598, "f18a6e729ca8906c1aa4d5e304a42898"),
    "rpm4000/RPM4000_Channel_05.wav": (714, 2867598, "23ff315eab0c28ee51955472e91a4b12"),
    "rpm4000/RPM4000_Channel_06.wav": (704, 2867598, "9e663d5889faba66e10fc1a7feffff10"),
    "rpm4000/RPM4000_Channel_07.wav": (726, 2867598, "f5cffd80598be8a139d4097934fe5d4e"),
    "rpm4000/RPM4000_Channel_08.wav": (707, 2867598, "abfd413a8c33c59c7594a37150c63be6"),
    "rpm4000/RPM4000_Channel_09.wav": (722, 2867598, "de738e7089e84cb2fba050dde02e6500"),
    "rpm4000/RPM4000_Channel_10.wav": (692, 2867598, "703685d70cf197f0191d399736c08aab"),
    "rpm4000/RPM4000_Channel_11.wav": (735, 2867598, "def0710430ca9a031ec3f0c37bd33cf4"),
    "rpm4000/RPM4000_Channel_12.wav": (733, 2867598, "d2e3bace07e2e0902dcad31299fe4ba9"),
    "rpm4000/RPM4000_Channel_13.wav": (725, 2867598, "f8c40753d60923de9599af755f52aa8a"),
    "rpm4000/RPM4000_Channel_14.wav": (731, 2867598, "ff9c506255e9dd126629314cc0c9e81e"),
    "rpm4000/RPM4000_Channel_15.wav": (713, 2867598, "cc0c4a775c961ddb03e92423cc610260"),
    "rpm4000/RPM4000_Channel_16.wav": (690, 2867598, "138adeab2563cf737d09e732a85b2cef"),
    "rpm4000/RPM4000_Speech_Ref.wav": (695, 2867598, "334d2d381643b4ab40a6bb888c04dd4a"),
    "rpm5000/RPM5000_Channel_01.wav": (702, 2848658, "82f2829ac728ea62a8ca1d7dd5b3adc4"),
    "rpm5000/RPM5000_Channel_02.wav": (739, 2848658, "3b2615665d8fbe781cd80bb69c882213"),
    "rpm5000/RPM5000_Channel_03.wav": (724, 2848658, "765ed0bba1d8dbf83a36416e20f3174f"),
    "rpm5000/RPM5000_Channel_04.wav": (734, 2848658, "cccac8c62dd488345224dc08f4600245"),
    "rpm5000/RPM5000_Channel_05.wav": (715, 2848658, "ce1707abfe281cd4cc01f33d7c5aec48"),
    "rpm5000/RPM5000_Channel_06.wav": (698, 2848658, "61267fc8f7b9363c0475e21370f0772c"),
    "rpm5000/RPM5000_Channel_07.wav": (709, 2848658, "c4fafdaff1c1c5cca8b581176d658f6e"),
    "rpm5000/RPM5000_Channel_08.wav": (719, 2848658, "70d6f416015a5246fcd89de5e21866ab"),
    "rpm5000/RPM5000_Channel_09.wav": (720, 2848658, "d3253d37e94a4d9b6186b11e015309ac"),
    "rpm5000/RPM5000_Channel_10.wav": (705, 2848658, "6c440a1351704a1d511aa2aa5a8c5d22"),
    "rpm5000/RPM5000_Channel_11.wav": (694, 2848658, "726ef6f3b845f0a6147a3e1c801950ef"),
    "rpm5000/RPM5000_Channel_12.wav": (727, 2848658, "6912d24d095e436bb15c7229dbc00861"),
    "rpm5000/RPM5000_Channel_13.wav": (691, 2848658, "7e48b15ec4c6058b3a351a7179c0ce50"),
    "rpm5000/RPM5000_Channel_14.wav": (738, 2848658, "250c6f928ecb39e5e40dcdac457f7d0c"),
    "rpm5000/RPM5000_Channel_15.wav": (728, 2848658, "878fdbc17dba5c5519b67a1d14217f2e"),
    "rpm5000/RPM5000_Channel_16.wav": (711, 2848658, "4224a4e55deecd4d07768367f85fd5e6"),
    "rpm5000/RPM5000_Speech_Ref.wav": (697, 2848658, "3acbd09ea46aac0972f14371bb8b0b44"),
    "rpm6000/RPM6000_Channel_01.wav": (699, 2866908, "91591d7bf4bd32881bf85180e3204b38"),
    "rpm6000/RPM6000_Channel_02.wav": (718, 2866908, "7d249864044b5028931599f13a15801c"),
    "rpm6000/RPM6000_Channel_03.wav": (737, 2866908, "2313126890fb23afe152070036dc0605"),
    "rpm6000/RPM6000_Channel_04.wav": (700, 2866908, "41c7655ad41c01c5e0e7c1c2e8b723bd"),
    "rpm6000/RPM6000_Channel_05.wav": (712, 2866908, "52056d37fa7e45a29b2ce1d3f3ef1d2c"),
    "rpm6000/RPM6000_Channel_06.wav": (730, 2866908, "fe186f7d682b1e99619e3000acf70a0d"),
    "rpm6000/RPM6000_Channel_07.wav": (716, 2866908, "c9d5f0d13d4617eb649afee97df1cb62"),
    "rpm6000/RPM6000_Channel_08.wav": (703, 2866908, "8a1e69bb12dfb188e8e089de9df6fcfc"),
    "rpm6000/RPM6000_Channel_09.wav": (732, 2866908, "b8d729d7f405ec7b69bf3f29f6c3048d"),
    "rpm6000/RPM6000_Channel_10.wav": (708, 2866908, "f295df22e0c94af463c2314dd5f1ac2e"),
    "rpm6000/RPM6000_Channel_11.wav": (710, 2866908, "c835fa8d23919d49a6dbd51e3b287cea"),
    "rpm6000/RPM6000_Channel_12.wav": (701, 2866908, "bdf52863d7ddf9fbca8984cba320b7de"),
    "rpm6000/RPM6000_Channel_13.wav": (729, 2866908, "109466a48a0126e6f6a404aacd45b779"),
    "rpm6000/RPM6000_Channel_14.wav": (740, 2866908, "c672bddb7e69286d5ee01bbe1e505e72"),
    "rpm6000/RPM6000_Channel_15.wav": (693, 2866908, "60cec51ce604441aa9108396631db6e8"),
    "rpm6000/RPM6000_Channel_16.wav": (721, 2866908, "1511e14d80158ceb1ff2bb6d2e899920"),
    "rpm6000/RPM6000_Speech_Ref.wav": (736, 2866908, "694faac827158d8afab752c6f206c545"),
}


def download(dest: Path) -> Path:
    """Fetch the pinned v1.1 files into ``dest`` (skips files already at size)."""
    from data_processing.downloaders import http_fetch

    dest = Path(dest)
    for rel, (fid, size, _) in FILES.items():
        http_fetch(
            f"https://rdr.kuleuven.be/api/access/datafile/{fid}", dest / rel, expected_size=size
        )
    return dest


#: Measured placement constraints and unordered estimates digitized from the paper.
GEOMETRY = {
    "mic_pos": "One row per audio channel in metres: stand axis at the figure arm-plane "
    "height; +x points from the stand towards the loudspeaker, +z up. NaN means "
    "unidentified. Assigned onboard components are figure and acoustic estimates, not a "
    "survey. Ch17's nominal horizontal (2, 0) follows from the stand-axis convention; "
    "it is 0.2 m above the loudspeaker, not 0.2 m above the floor or above the other mics.",
    "channel_roles": {
        "main_array": list(range(1, 13)),
        "propeller_reference": list(range(13, 17)),
        "external_speech_reference": [17],
        "numbering": "one-based source channel numbers (audio rows 0-16)",
        "evidence": "deposit README.txt",
    },
    "main_array_layout": {
        "description": "embedded in the UAV's custom bottom frame; Fig. 3 draws 8 main "
        "microphones on the corners of a box-shaped frame and 4 further out",
        "aligned_with_rotor_axes": "the 4 channels added from M_S+N=8 to 12 (channel ids "
        "not stated) lie on the rotors' rotational axes (paper Sec. 5)",
        "status": "qualitative (figure + text)",
    },
    "propeller_mics": {
        "description": "one microphone below each two-blade propeller; rotor-to-channel "
        "assignment not documented",
        "status": "stated",
    },
    "rotors": {
        "n": 4,
        "configuration": "+",
        "propeller_diameter_m": 0.2748,
        "drone_pitch_lambda": 2.18,
        "opposite_rotor_hub_distance_m": 0.599,
        "opposite_rotor_hub_distance_status": "derived (lambda x d), same UAV as thesis ch. 4",
    },
    "frame_bottom_height_above_floor_m": 1.15,
    "loudspeaker": {
        "horizontal_distance_from_stand_base_m": 2.0,
        "height": "close to ground level",
        "status": "stated",
    },
    "reference_mic_above_loudspeaker_m": 0.2,
    "coordinate_frame": "mic_pos uses stand axes. mic_site_pos_body uses the paper figure's "
    "axes: choose its +x,+y arm towards the loudspeaker; reflection about that arm remains "
    "unidentified. mic_site_possible records possible assignments, not measured positions.",
    "acoustic_assignment": {
        "method": "inferred from 300-4000 Hz speech GCC-PHAT in early/mid/late 2 s windows "
        "at the three RPMs, plus the 4/8/12-site figure reconstruction",
        "main_arm_pair": "Ch10 lags Ch11 by 71.6-72.3 samples in all nine windows, greater "
        "than the possible cube baselines: Ch11 is on the speaker-facing outer arm, Ch10 opposite",
        "rotor_arm_pairs": "Ch15 lags Ch13 by 52-54 samples in the usable 4000/5000 rpm "
        "windows. Ch13 is speaker-facing, Ch15 opposite. Ch14 and Ch16 are the side pair; "
        "one source direction cannot distinguish left from right.",
        "remaining_main": "No unique assignment to sites for the other ten main microphones.",
    },
    "figure_reconstruction": {
        "status": "estimated from the paper figure; point sets are UNORDERED (channel "
        "order within each group unknown). Site labels are diagram points, not WAV numbers.",
        "method": "orthographic inversion of plots/miccoords_Ma_{4,8,12}.eps (screen "
        "vectors of one X/Y/Z tick step, marker centroids, one geometric constraint per "
        "point: arm diagonals / faces of the drawn box); all 16 points fall on a tick "
        "lattice (residual <= 0.05 tick); scale from outer mics on the rotor axes, "
        "r_outer = D/2 = 0.2995 m => 1 tick = 0.04236 m; cross-checked on the setup photo",
        "frame": "origin at the arm crossing in the rotor/arm plane, z up, arms on the "
        "+-45 deg diagonals of x/y; right-handed assumed (mirror ambiguity); rotation "
        "about z relative to the drone front unknown",
        "units": "m",
        "propeller_mics_ch13_16": [
            [0.169, 0.169, 0.0],
            [-0.169, 0.169, 0.0],
            [-0.169, -0.169, 0.0],
            [0.169, -0.169, 0.0],
        ],
        "main_subset_Ma4_bottom_box": [
            [0.0635, 0.0424, -0.297],
            [-0.0424, 0.0635, -0.297],
            [0.0424, -0.0635, -0.297],
            [-0.0635, -0.0424, -0.297],
        ],
        "main_added_Ma8_top_box": [
            [0.0424, 0.0635, -0.169],
            [-0.0635, 0.0424, -0.169],
            [0.0635, -0.0424, -0.169],
            [-0.0424, -0.0635, -0.169],
        ],
        "main_added_Ma12_outer": [
            [0.212, 0.212, -0.381],
            [-0.212, 0.212, -0.381],
            [-0.212, -0.212, -0.381],
            [0.212, -0.212, -0.381],
        ],
        "uncertainty": "scale 1 tick in [0.040, 0.053] m; lattice quantisation +-0.02 m; "
        "outer-mic z +-0.10 m (photo suggests ~0.28 m drop vs 0.38 m in the plot)",
        "channel_inference": "[INFERENCE] if the figure subsets are channels 1..M (the "
        "reference is 'the first channel' in every configuration), ch01-04 = bottom box, "
        "ch05-08 = top box, ch09-12 = outer; order within each group unknown",
        "provenance": "figure reconstruction by project agent AiraKuGeometry, 2026-11-26",
    },
    "evidence": "Tengan et al., ICA 2022 (arXiv:2211.02690v2) Sec. 4.1, Figs. 1-3, Sec. 5; "
    "Tengan PhD thesis (KU Leuven 2023) Secs. 4.2, 5.4.1, 6.4.1; deposit README.txt",
}


def _channel_geometry(mics: td.LabelIndex) -> dict[str, td.Series]:
    groups = (
        ("main_subset_Ma4_bottom_box", "lower box"),
        ("main_added_Ma8_top_box", "upper box"),
        ("main_added_Ma12_outer", "outer array"),
        ("propeller_mics_ch13_16", "under rotor"),
    )
    figure = GEOMETRY["figure_reconstruction"]
    site_names = tuple(f"{label} site {i}" for _, label in groups for i in range(1, 5))
    sites = td.LabelIndex(site_names)
    xyz = td.LabelIndex(("x", "y", "z"))
    positions = np.full((N_CHANNELS, 3), np.nan)
    outer, rotor = (
        np.asarray(figure["main_added_Ma12_outer"]),
        np.asarray(figure["propeller_mics_ch13_16"]),
    )
    outer_r, rotor_r = np.linalg.norm(outer[0, :2]), np.linalg.norm(rotor[0, :2])
    positions[9] = (-outer_r, 0.0, outer[0, 2])  # Ch10: far outer arm
    positions[10] = (outer_r, 0.0, outer[0, 2])  # Ch11: near outer arm
    positions[12:16, 0] = (rotor_r, 0.0, -rotor_r, 0.0)
    positions[[12, 14], 1] = 0.0  # Ch14/16: sign of y unresolved
    positions[12:16, 2] = rotor[0, 2]
    positions[16, :2] = (GEOMETRY["loudspeaker"]["horizontal_distance_from_stand_base_m"], 0.0)
    possible = np.zeros((N_CHANNELS, len(site_names)), dtype=bool)
    possible[:12, :12] = True
    possible[:12, [8, 10]] = False  # these two outer sites are identified acoustically
    possible[[9, 10], :] = False
    possible[9, 10] = possible[10, 8] = True
    possible[12, 12] = possible[14, 14] = True
    possible[[13, 15], 13] = possible[[13, 15], 15] = True
    return {
        "mic_pos": td.wrap(positions, ("mic", "xyz"), {"mic": mics, "xyz": xyz}),
        "mic_role": td.wrap(np.asarray(CHANNEL_ROLES), ("mic",), {"mic": mics}),
        "mic_site_pos_body": td.wrap(
            np.concatenate([figure[key] for key, _ in groups]),
            ("site", "body_xyz"),
            {"site": sites, "body_xyz": xyz},
        ),
        "mic_site_possible": td.wrap(possible, ("mic", "site"), {"mic": mics, "site": sites}),
    }


PROVENANCE = {
    "source_url": "https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/PZAVUC",
    "doi": DOI,
    "version": "1.1 (Dataverse dataset 689, released 2022-08-29)",
    "license": "CC BY-NC-SA 4.0",
    "rights_note": "Deposit licence (Dataverse metadata) is CC-BY-NC-SA-4.0, no extra "
    "terms of use/access: non-commercial use, attribution, share-alike for adaptations.",
    "citation": "Tengan E., Dietzen T., Ruiz S., Alkmim M., Cardenuto J., van Waterschoot T., "
    "'Speech enhancement using ego-noise references with a microphone array embedded "
    "in an unmanned aerial vehicle', Proc. 24th ICA, Gyeongju 2022 (arXiv:2211.02690); "
    "data: Replication Data, KU Leuven RDR, doi:10.48804/PZAVUC.",
    "collection_method": "UAV rigidly mounted on a stand in a semi-anechoic chamber, "
    "throttle fixed (simulated hover), loudspeaker speech below the UAV",
    "equipment": "16 electret microphones + on-board sound card/minicomputer; external "
    "reference microphone; laser rpm probe (run average only published)",
    "observation_type": "uav_embedded_array",
    "sample_rate": 44100,
    "channels": N_CHANNELS,
    "description": "3 whole runs (4000/5000/6000 rpm; ~32.5 s each). One labeled 17-channel "
    "recording: 01-12 under-body main array, 13-16 under-rotor noise references, "
    "17 external speech reference. Do not use channel 17 as an onboard noise channel.",
    "geometry": GEOMETRY,
}


def _verify(root: Path) -> None:
    """Fail closed unless every pinned file is present with its size and md5."""
    for rel, (_, size, md5) in FILES.items():
        path = root / rel
        if not path.is_file() or path.stat().st_size != size:
            raise FileNotFoundError(f"{DATASET}: missing or truncated {rel} under {root}")
        digest = hashlib.md5()
        with open(path, "rb") as fh:
            while block := fh.read(1 << 20):
                digest.update(block)
        if digest.hexdigest() != md5:
            raise ValueError(f"{DATASET}: md5 mismatch for {rel}")


def build(root: Path) -> Iterator[tuple[str, td.Frame]]:
    import soundfile as sf

    from data_processing.sources._common import meta_frame

    root = Path(root)
    _verify(root)
    mics = td.LabelIndex(CHANNEL_NAMES)
    geometry = _channel_geometry(mics)
    for rpm in RPMS:
        rels = [f"rpm{rpm}/RPM{rpm}_Channel_{ch:02d}.wav" for ch in range(1, N_ONBOARD + 1)]
        rels.append(f"rpm{rpm}/RPM{rpm}_Speech_Ref.wav")
        infos = [sf.info(str(root / rel)) for rel in rels]
        sr, n_total, subtype = infos[0].samplerate, infos[0].frames, infos[0].subtype
        for rel, info in zip(rels, infos, strict=True):
            if (info.samplerate, info.frames, info.channels, info.subtype) != (
                sr,
                n_total,
                1,
                "PCM_16",
            ):
                raise ValueError(f"{DATASET}: {rel} is not mono PCM_16 {sr} Hz x {n_total}")
        block = np.empty((N_CHANNELS, n_total), dtype=np.float32)
        for row, rel in enumerate(rels):
            with sf.SoundFile(str(root / rel)) as fh:
                fh.read(n_total, dtype="float32", out=block[row])
        meta = meta_frame(
            f"RPM{rpm}",
            DATASET,
            system={
                "category": "multirotor_uav",
                "platform": "MikroKopter MK EASY Quadro V3 (custom bottom frame)",
                "rig": "kuleuven_mk_easy_16ch",
                "n_rotors": 4,
                "rotor_configuration": "+",
                "blades_per_rotor": 2,
                "propeller_diameter_m": 0.2748,
            },
            observation={
                "mic_placement": "onboard_and_external",
                "type": "uav_embedded_array_with_external_reference",
                "array": "Ch01-12 under-body main array; Ch13-16 under-rotor ego-noise references",
                "speech_reference": "Ch17: external mic 0.2 m above the speech loudspeaker",
                "environment": "semi-anechoic chamber, Siemens DISW Leuven",
                "source_motion": "static",
                "relative_trajectory": "none",
            },
            operating={
                "condition": "bench",
                "throttle": "fixed (joystick not spring-loaded); simulated hover",
                "rpm_mean": rpm,
                "rpm_units": "rev/min",
                "rpm_source": "published run average (laser probe under one blade); "
                "no time series released",
                "n_active_rotors": 4,
                "contains_rotor_noise": True,
                "external_source": "speech",
                "external_source_detail": "male VCTK utterance from a loudspeaker "
                "2 m from the stand base; first ~2 s treated as noise-only by the "
                "authors (thesis Sec. 6.4.1)",
            },
            label={
                "subcategory": f"stand-hover_speech_{rpm}rpm",
                "rig": "MK EASY Quadro V3 16-ch",
                "health": "normal",
                "mixed": "acoustic (speech played live during recording)",
            },
            extra={
                "source_kind": "original_recording",
                "doi": DOI,
                "original_files": rels,
                "channel_names": list(CHANNEL_NAMES),
                "channel_notes": [""] * N_ONBOARD + [_REFERENCE_CLOCK_NOTE],
                "original_subtype": subtype,
                "recording_n_samples": n_total,
                "recording_duration_s": n_total / sr,
                "reference_alignment": {
                    "basis": "files as released; same rate, length and speech event. Channel 17 "
                    "is suitable for speech timing, not a verified phase-synchronous spatial "
                    "microphone. No delay removal, interpolation or clock correction.",
                    "onboard_minus_reference_speech_lag_samples_early_mid_late": _SPEECH_LAG[rpm],
                    "interpretation": "6.1–8.0 ms across microphones includes real propagation "
                    "and possible device latency; small within-run changes might be clock "
                    "drift or acoustic effects. The acquisition clock relation is undocumented.",
                },
                "geometry": GEOMETRY,
            },
        )
        frame = td.Frame(
            {
                "audio": td.Series(
                    block, ("mic", "time"), {"mic": mics, "time": td.GridIndex(sr, n_total)}
                ),
                **geometry,
                "meta": meta,
            }
        )
        yield f"RPM{rpm}", frame
        del block, frame
