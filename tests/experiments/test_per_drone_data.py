import csv
import json

import numpy as np
import pytest

from experiments.rps_traj import per_drone_data as pdd
from experiments.rps_traj.data import Flight
from experiments.rps_traj.spectral_validation import CruiseMetric


def _flight(name, rng, *, ground=True, hover_s=60.0, level=80.0):
    """A 100 Hz 4-rotor track: off, standby, take-off, hover, landing, off (or hover only)."""
    parts = []
    if ground:
        parts += [np.zeros(200), np.full(300, 15.0), np.linspace(15.0, level, 200)]
    parts.append(level + np.cumsum(rng.normal(0, 0.05, int(hover_s * 100))).clip(-5, 5))
    if ground:
        parts += [np.linspace(level, 0.0, 150), np.zeros(300)]
    m = np.concatenate(parts)
    rps = np.maximum(0.0, m + rng.normal(0, 0.3, (4, m.size)) * (m > 1))
    return Flight("toy", name, 100.0, 0.0, rps, "toy-frames:rps")


def _pieces(rig, names, rng, **kw):
    out = []
    for n in names:
        f = _flight(n, rng, **kw)
        out.append(pdd.Piece(n, n, f, (0, f.rps.shape[1])))
    return out


def test_split_rules_and_guards():
    assert [
        len(pdd.seeded_validation("r", [str(i) for i in range(n)])) for n in (1, 2, 4, 68, 95)
    ] == [0, 1, 1, 14, 19]
    assert pdd.seeded_validation("r", list("abcd")) == pdd.seeded_validation("r", list("dcba"))
    a, b = pdd.within_flight_cut(10_000)
    assert a % 4 == 0 and b - a == 1600
    rng = np.random.default_rng(0)
    (p,) = _pieces("toy", ["x"], rng)
    with pytest.raises(AssertionError):
        pdd.assert_disjoint("toy", [p], [p])


@pytest.mark.parametrize("ground", [True, False])
def test_pack_and_score_roundtrip(tmp_path, ground):
    rng = np.random.default_rng(1)
    rig = "toy"
    train = _pieces(rig, ["a", "b", "c"], rng, ground=ground)
    valid = _pieces(rig, ["v"], rng, ground=ground)
    pdd.assert_disjoint(rig, train, valid)
    waves = rng.normal(80, 1, (4, 4, pdd.CRUISE_N))
    rows = [
        {"rig": r, "flight": f, "calibration": True}
        for r, f in (("p", "1"), ("p", "2"), ("q", "3"), ("q", "4"))
    ]
    metric = CruiseMetric.fit(waves, rows)
    pdd._write_rig(tmp_path, rig, train, valid, [], {"rule": "toy"}, {})
    p = pdd.paths(tmp_path, rig)
    tmeta = json.loads(p["train_meta"].read_text())
    assert [r["id"] for r in tmeta["train"]] == ["a", "b", "c"]
    (target,) = tmeta["targets"]
    assert set(target["marks"]) >= {"n", "takeoff", "landing"} and "hover" not in target["marks"]
    with np.load(p["train"]) as z:
        assert z["x"].shape[0] == 4 and z["offsets"][-1] == z["x"].shape[1]
    vmeta = json.loads(p["valid_meta"].read_text())
    phases = {c["phase"] for c in vmeta["targets"][0]["cells"]}
    absent = {c["phase"] for c in vmeta["cell_exclusions"] if c["kind"] == "absent"}
    assert "cruise" in phases
    if not ground:
        assert (
            absent == {"startup", "standby", "takeoff", "landing"}
            and target["events"]["takeoff"] is None
        )
    manifest = {
        "protocol": {"rigs": [rig]},
        "metric": {
            rig: {
                "band_power": metric.band_power.tolist(),
                "level_scale_rps": metric.level_scale_rps,
                "spectrum_scale": metric.spectrum_scale,
            }
        },
    }
    (tmp_path / "pack" / "manifest.json").write_text(json.dumps(manifest))
    with np.load(p["valid"]) as z:
        real = z["t0"]
    for arm, sha in (("template", tmeta["train_sha256"]), ("phasevar2", "stale")):
        rp = pdd.run_paths(tmp_path, rig, arm)
        rp["fit"].parent.mkdir(parents=True, exist_ok=True)
        rp["fit"].write_text(
            json.dumps(
                {
                    "rig": rig,
                    "arm": arm,
                    "train_sha256": sha,
                    "n_params": 3,
                    "train_ids": ["a", "b", "c"],
                }
            )
        )
        draws = real[None] + rng.normal(0, 1, (pdd.N_DRAWS, *real.shape))
        np.savez(rp["draws"], t0=draws.astype(np.float32))
    pdd.score(tmp_path)
    out = json.loads((tmp_path / "scores.json").read_text())
    by_arm = {(r["arm"], r["phase"]): r for r in out["rows"]}
    assert by_arm[("template", "cruise")]["status"] == "ok"
    assert by_arm[("phasevar2", "cruise")]["status"].startswith("failed: fit provenance")
    assert {m["arm"] for m in out["missing_runs"]} == {
        "regulated_bridge",
        "phasevar4",
        "msvar3",
        "hsmm3",
    }
    with (tmp_path / "phase_scores.csv").open() as fh:
        table = list(csv.DictReader(fh))
    assert table[0].keys() == {"rig", "phase", "n_valid", *pdd.ARMS}
    assert table[-1]["phase"] == "balanced"
