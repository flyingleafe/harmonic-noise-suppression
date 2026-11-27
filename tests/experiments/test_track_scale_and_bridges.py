import numpy as np

from experiments.static_rig.bracket_tracks import fill_tracks
from experiments.static_rig.scale_tracks import calibrate


def _case():
    times = np.arange(10) + 0.5
    data = np.stack((40.0 + 0.01 * times, 90.0 + 0.03 * times), axis=1)
    return {
        "times_s": times.tolist(),
        "rates_rps": data.tolist(),
        "track_id": [[1, 2] for _ in times],
        "uncertainty_rps": [[0.03, 0.05] for _ in times],
        "provenance": [["measured"] * 2 for _ in times],
        "order_support": [[[2, 4, 6], [1, 2, 3]] for _ in times],
        "diagnostics": {"active_spans": [[0, 10]]},
    }


def test_one_order_per_track_no_timepoint_oracle():
    raw = _case()
    times = raw["times_s"]
    rates = np.array(raw["rates_rps"])
    true = rates * [2, 1]
    anchors = [{"time_s": times[i], "rates_rps": true[i].tolist()} for i in (0, 3, 6, 9)]
    out = calibrate(raw, anchors, written_for="test")
    np.testing.assert_allclose(out["rates_rps"], true)
    assert out["order_support"][5][0] == [1, 2, 3]
    assert out["provenance"][5] == ["track_supported", "measured"]
    unchanged = calibrate(raw, [], written_for="blind")
    np.testing.assert_array_equal(unchanged["rates_rps"], rates)
    assert (
        unchanged["diagnostics"]["scale_calibration"]["tracks"][0]["basis"]
        == "blind_no_scale_corroboration"
    )


def test_bridge_is_local_and_never_overwrites_partial_witness():
    raw = _case()
    truth = raw["rates_rps"][4][0]
    for name in ["rates_rps", "uncertainty_rps", "track_id"]:
        raw[name][4][0] = None
    raw["provenance"][4][0] = "unresolved"
    raw["order_support"][4][0] = []
    raw["rates_rps"][4][1] += 0.2
    out = fill_tracks(raw)
    assert abs(out["rates_rps"][4][0] - truth) < 1e-8
    assert out["rates_rps"][4][1] == raw["rates_rps"][4][1]
    assert out["provenance"][4] == ["interpolated", "measured"]
    raw["diagnostics"]["active_spans"] = [[0.0, 4.0], [4.9, 10.0]]
    assert fill_tracks(raw)["rates_rps"][4][0] is None
    raw["diagnostics"]["active_spans"] = [[0.0, 10.0]]
    raw["rates_rps"][4][1] += 5.0
    assert fill_tracks(raw)["rates_rps"][4][0] is None
