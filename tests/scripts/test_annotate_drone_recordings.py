"""An acquisition label is not a measurement of the shafts."""

import pytest
from annotate_drone_recordings import finalize

from data_processing.recording_annotations.derived import droneaudioset_path_facts


@pytest.mark.parametrize("basis", ["telemetry", "acoustic"])
def test_fixed_throttle_keeps_independent_evidence_of_speed_change(basis):
    rec = droneaudioset_path_facts(
        "drone-only-recordings/drone1-only/mic-dist-25cm/throttle-high/mic2_8array-down-File2.wav"
    )
    if basis == "telemetry":
        observed = {"source": "shaft tachometer", "suggested": "variable", "strong": True}
        measured = {"native_regime": observed}
    else:
        windows = [
            {
                "start_s": 2 * i,
                "end_s": 2 * i + 2,
                "accepted": True,
                "rates_rev_s": [base + j for j in range(4)],
                "quality": {"acceptance_basis": "independent_acoustic_comb"},
            }
            for i, base in enumerate([60, 61, 82, 83])
        ]
        measured = {
            "tach": {
                "status": "resolved",
                "n_expected": 4,
                "publication_state": "corroborated",
                "regime_evidence": {"suggested": "variable", "analyzed_s": 8},
                "windows": windows,
            }
        }
    out, _, _, observations, _ = finalize(rec, {"has_audio": True}, {}, measured)
    assert out["speed_regime"] == "constant" and out["speed_supplied"] == "no"
    if basis == "telemetry":
        assert observations["native_regime"]["suggested"] == "variable"
    else:
        assert out["speed_regime_acoustic_disagreement"]["complete_windows"] == 4
