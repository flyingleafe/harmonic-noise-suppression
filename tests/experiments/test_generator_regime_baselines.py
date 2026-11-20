"""Standby/cruise-matched external baselines: supports, splicing, frozen states, blend."""

from __future__ import annotations

import hashlib
import json
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from experiments.generator_paper import cases
from experiments.generator_paper import eone_baseline as E
from experiments.generator_paper import regime_baselines as rb
from experiments.generator_paper.cases import FlightClip
from experiments.generator_paper.eone_baseline import EoneSettings
from experiments.stochastic_fit import clips as C

SR = 16000
STANDBY = [("FLY125", 2.0, 4.0), ("FLY125", 6.0, 4.0), ("FLY125", 10.0, 4.0)]
CRUISE = [("FLY125", 16.0, 8.0), ("FLY125", 32.0, 8.0)]


# ── synthetic inputs ───────────────────────────────────────────────────────


def _spec(rec: str, start: float, dur: float) -> tuple[str, str]:
    return (
        f"flight_michaels:{rec}:{start!r}:{dur!r}:rps_refined",
        f"flight_michaels_{rec}@{start:.3f}+{dur:g}_rps_refined",
    )


def _write_inputs(root: Path, supports: dict[str, list[tuple[str, float, float]]]) -> Path:
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    records = []
    for regime, windows in supports.items():
        specs, members = zip(*(_spec(*w) for w in windows), strict=True)
        path = inputs / f"md2_{regime}.json"
        path.write_text(json.dumps({"pool": {"specs": list(specs), "members": list(members)}}))
        records.append(
            {
                "regime": regime,
                "source": f"r2://x/{path.name}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "local_path": f"results/noise_generator_paper/inputs/{path.name}",
            }
        )
    (root / "input_provenance.json").write_text(
        json.dumps(
            {
                "accepted_md2_fits": records,
                "dataset_pins": {"michaels-frames": "pin0"},
            }
        )
    )
    return inputs


_LONG_S = 48.0
_RNG = np.random.default_rng(7)
_AUDIO = _RNG.standard_normal((1, int(_LONG_S * SR))).astype(np.float32).astype(np.float64)


def _rps_at(t: np.ndarray) -> np.ndarray:
    speed = np.where(t < 14.0, 30.0, 80.0)
    return np.stack([speed, speed + 2.0])


def _clip(start: float, dur: float, fit_regime: str) -> FlightClip:
    j0, n = int(start * SR), int(dur * SR)
    rps = _rps_at((j0 + np.arange(n)) / SR)
    audio = _AUDIO[:, j0 : j0 + n]
    prov = {
        "dataset": "michaels-frames",
        "dataset_version": "pin0",
        "dataset_pin": "michaels-frames@pin0",
        "recording_id": "FLY125",
        "label_key": "rps_refined",
        "split": "calibration",
        "absolute_start_s": start,
        "absolute_end_s": start + dur,
        "relative_start_s": start,
        "relative_end_s": start + dur,
        "native_start_sample": j0,
        "native_n_samples": n,
        "start_sample": j0,
        "n_samples": n,
        "rotor_min_rps": rps.min(axis=1).tolist(),
        "rotor_max_rps": rps.max(axis=1).tolist(),
        "regime_rule": {"standby_max_rps": 45.0, "cruise_min_rps": 65.0},
        "fit_regime": fit_regime,
    }
    return FlightClip(
        key=f"md2:FLY125@{start:.3f}+{dur:g}",
        rig="md2",
        recording_id="FLY125",
        start_s=start,
        sr=SR,
        audio=audio,
        rps=rps,
        native_sr=SR,
        native_audio=audio,
        regime=cases.regime_of(rps.min(axis=1), rps.max(axis=1)),
        provenance=prov,
    )


def _synthetic_supports() -> dict[str, list[FlightClip]]:
    return {
        "standby": [_clip(s, d, "standby") for _, s, d in STANDBY],
        "cruise": [_clip(s, d, "cruise") for _, s, d in CRUISE],
    }


# ── supports ───────────────────────────────────────────────────────────────


def test_supports_refuse_a_modified_accepted_fit(tmp_path):
    inputs = _write_inputs(tmp_path, {"standby": STANDBY, "cruise": CRUISE})
    windows, _ = rb.support_windows(inputs)
    assert [w["fit_regime"] for w in windows["cruise"]] == ["cruise", "cruise"]
    path = inputs / "md2_cruise.json"
    path.write_text(path.read_text().replace("32.0", "40.0"))
    with pytest.raises(ValueError, match="hash mismatch"):
        rb.support_windows(inputs)


def _fake_recording(seconds: float = 20.0) -> C.Recording:
    native_sr = 44100
    rng = np.random.default_rng(3)
    audio = rng.standard_normal((8, int(seconds * native_sr))).astype(np.float32)
    rps_t = np.arange(0.5, seconds - 0.4, 0.02)
    rps = np.stack([30.0 + 2.0 * r + 40.0 * (rps_t > 12.0) for r in range(4)])
    return C.Recording(
        "michaels-frames",
        cases._MD2.version,
        "FLY125",
        "fly125",
        audio,
        float(native_sr),
        0.0,
        rps,
        rps_t,
        "rps_refined",
        {"recording_id": "FLY125"},
    )


def test_explicit_support_windows_are_bit_identical_to_grid_windows(monkeypatch):
    rec = _fake_recording()
    monkeypatch.setattr(cases, "_decode", lambda spec, ids: {rid: (rec, {}) for rid in ids})
    grid = cases.load_flight_clips("md2", "calibration", duration_s=4.0)
    target = grid[1]
    (got,) = cases.load_support_clips(
        "md2", "calibration", [("FLY125", target.start_s, 4.0)], labels=[{"fit_regime": "standby"}]
    )
    assert got.key == target.key and got.regime == target.regime
    np.testing.assert_array_equal(got.audio, target.audio)
    np.testing.assert_array_equal(got.native_audio, target.native_audio)
    np.testing.assert_array_equal(got.rps, target.rps)
    for name in (
        "absolute_start_s",
        "native_start_sample",
        "start_sample",
        "n_samples",
        "rotor_min_rps",
        "rotor_max_rps",
        "dataset_pin",
    ):
        assert got.provenance[name] == target.provenance[name]
    assert got.provenance["fit_regime"] == "standby"
    with pytest.raises(ValueError, match="grid"):
        cases.load_support_clips("md2", "calibration", [("FLY125", 2.005, 4.0)])
    with pytest.raises(ValueError, match="coverage"):
        cases.load_support_clips("md2", "calibration", [("FLY125", 18.0, 4.0)])
    with pytest.raises(ValueError, match="FLY124"):
        cases.load_support_clips("md2", "calibration", [("FLY124", 2.0, 4.0)])


# ── splicing ───────────────────────────────────────────────────────────────


def test_contiguous_merge_is_the_exact_span_with_every_constituent_recorded():
    parts = _synthetic_supports()["standby"]
    merged = rb.merge_contiguous(parts)
    assert merged.key == "md2:FLY125@2.000+12"
    np.testing.assert_array_equal(merged.native_audio, _AUDIO[:, 2 * SR : 14 * SR])
    assert [m["key"] for m in merged.provenance["merged_from"]] == [p.key for p in parts]
    assert merged.provenance["n_samples"] == 12 * SR and merged.regime == "standby"
    with pytest.raises(ValueError, match="sample-adjacent"):
        rb.merge_contiguous([parts[0], parts[2]])


# ── blend ──────────────────────────────────────────────────────────────────


def _noise_expert(scale: float):
    def render(rps, *, sr, rng):
        return (scale * rng.standard_normal((2, rps.shape[-1]))).astype(np.float32)

    return render


def test_blend_interpolates_power_not_amplitude_and_keeps_streams_independent():
    experts = {"standby": _noise_expert(1.0), "cruise": _noise_expert(3.0)}
    for speed, expected_power in ((45.0, 1.0), (55.0, 5.0), (65.0, 9.0)):
        rps = np.full((2, 4 * SR), speed)
        mixed = rb.blend_experts(experts, rps, sr=SR, rng=np.random.default_rng(6))
        repeated = rb.blend_experts(experts, rps, sr=SR, rng=np.random.default_rng(6))
        np.testing.assert_array_equal(mixed, repeated)
        assert np.mean(mixed.astype(np.float64) ** 2) == pytest.approx(expected_power, rel=0.025)


# ── frozen fits ────────────────────────────────────────────────────────────


def _prepare(tmp_path, monkeypatch, methods, eone_device=None):
    inputs = _write_inputs(tmp_path / "data", {"standby": STANDBY, "cruise": CRUISE})
    supports = _synthetic_supports()
    monkeypatch.setattr(
        rb, "load_regime_calibration", lambda i: (supports, rb.support_windows(i)[1])
    )
    out = tmp_path / "out"
    return (
        out,
        inputs,
        rb.prepare_regime_fits(out, inputs, methods, seed=0, eone_device=eone_device),
    )


EONE_SMOKE = EoneSettings(
    piece_s=0.25,
    n_orders=8,
    n_bands=4,
    n_knots=5,
    hop=64,
    noise_loop=4096,
    epochs=3,
    max_lr=0.05,
    mrstft_max_fft=1024,
    mrstft_min_fft=64,
    harmonic_n_fft=4096,
    harmonic_win=1024,
    harmonic_hop=64,
)


def paired_eone_smoke(root: Path, monkeypatch, *, device: str) -> dict:
    """Throwaway smoke of the paired EONE path: fit both experts on ``device``, freeze, reload, blend.

    Tiny :data:`EONE_SMOKE` settings on the synthetic supports; patches ``fit_eone``'s settings
    and lifts the reload's production-settings refusal for this call only (``monkeypatch``).
    A CUDA job can run it outside pytest: ``runpy.run_path(<this file>)["paired_eone_smoke"]``
    inside ``pytest.MonkeyPatch.context()`` with ``device="cuda"``.
    """
    monkeypatch.setattr(E, "fit_eone", partial(E.fit_eone, settings=EONE_SMOKE))
    monkeypatch.setattr(rb, "_require_production", lambda report, label: None)
    _, _, info = _prepare(
        Path(root), monkeypatch, ("doerfler2026_eone_regimes",), eone_device=device
    )
    render, report = rb.regime_render("doerfler2026_eone_regimes", Path(info["path"]))
    t = np.arange(4 * SR) / SR
    rps = np.stack([30.0 + 15.0 * t, 32.0 + 15.0 * t])  # slowest rotor crosses 45-65 rev/s
    audio = render(rps, sr=SR, rng=np.random.default_rng(3))
    relocated = Path(root) / "relocated_fits"
    shutil.move(str(Path(info["path"]).parent), relocated)
    shutil.rmtree(Path(root) / "data")
    again = rb.regime_render("doerfler2026_eone_regimes", relocated / Path(info["path"]).name)[0](
        rps, sr=SR, rng=np.random.default_rng(3)
    )
    return {"info": info, "report": report, "rps": rps, "audio": audio, "reload_audio": again}


def test_paired_eone_smoke_fits_freezes_and_blends_both_experts(tmp_path, monkeypatch):
    result = paired_eone_smoke(tmp_path, monkeypatch, device="cpu")
    report = result["report"]
    assert report["base_method"] == "doerfler2026_eone" and report["n_mics"] == 1
    for regime in ("standby", "cruise"):
        expert = report["regimes"][regime]
        assert expert["fit_device"] == "cpu"
        assert expert["fit"]["training"]["completed_epochs"] == EONE_SMOKE.epochs
        assert expert["fit"]["calibration_clip_keys"] == [
            w["key"] for w in report["calibration_windows"] if w["fit_regime"] == regime
        ]
    audio = result["audio"]
    assert audio.dtype == np.float32 and audio.shape == (1, 4 * SR) and np.isfinite(audio).all()
    np.testing.assert_array_equal(audio, result["reload_audio"])


def test_frozen_experts_reload_render_and_refuse_tampering(tmp_path, monkeypatch):
    methods = ("heutschi2020_regimes", "yang2025_regimes")
    out, inputs, info = _prepare(tmp_path, monkeypatch, methods)
    manifest_path = Path(info["path"])
    assert manifest_path == (out / rb.MANIFEST).resolve() and info["methods"] == list(methods)
    manifest = json.loads(manifest_path.read_text())
    for family in manifest["families"].values():
        for expert in family["experts"].values():
            assert not Path(expert["state"]["path"]).is_absolute()

    render, report = rb.regime_render("heutschi2020_regimes", manifest_path)
    assert report["fitting_treatment"] == "standby_cruise_matched"
    assert report["base_method"] == "heutschi2020" and report["n_mics"] == 1
    assert [(w["key"], w["fit_regime"]) for w in report["calibration_windows"]] == (
        [(f"md2:FLY125@{s:.3f}+4", "standby") for _, s, _ in STANDBY]
        + [(f"md2:FLY125@{s:.3f}+8", "cruise") for _, s, _ in CRUISE]
    )
    standby = report["regimes"]["standby"]
    assert standby["fit"]["reference_clip"] == "md2:FLY125@2.000+12"
    assert [m["key"] for m in standby["fit_inputs"][0]["provenance"]["merged_from"]] == [
        w["key"] for w in report["calibration_windows"][:3]
    ]
    supports = _synthetic_supports()
    original_experts = {
        "standby": partial(
            rb.render_heutschi, rb.fit_heutschi([rb.merge_contiguous(supports["standby"])], seed=0)
        ),
        "cruise": partial(rb.render_heutschi, rb.fit_heutschi(supports["cruise"], seed=0)),
    }
    for speed in (31.0, 100.0):
        rps = np.full((2, 4 * SR), speed)
        actual = render(rps, sr=SR, rng=np.random.default_rng(1))
        expected = rb.blend_experts(original_experts, rps, sr=SR, rng=np.random.default_rng(1))
        assert actual.dtype == np.float32 and actual.shape == (1, 4 * SR)
        np.testing.assert_array_equal(actual, expected)

    yang_render, yang_report = rb.regime_render("yang2025_regimes", manifest_path)
    assert {
        r: yang_report["regimes"][r]["fit"]["calibration_clip_keys"] for r in ("standby", "cruise")
    } == {
        "standby": [w["key"] for w in yang_report["calibration_windows"][:3]],
        "cruise": [w["key"] for w in yang_report["calibration_windows"][3:]],
    }
    original_yang = {
        r: partial(rb.render_yang, rb.fit_yang(clips, seed=0)) for r, clips in supports.items()
    }
    rps = np.full((2, 4 * SR), 55.0)
    np.testing.assert_array_equal(
        yang_render(rps, sr=SR, rng=np.random.default_rng(2)),
        rb.blend_experts(original_yang, rps, sr=SR, rng=np.random.default_rng(2)),
    )

    with pytest.raises(ValueError, match="never extended"):
        rb.prepare_regime_fits(out, inputs, ("yang2025_phase_regimes",), seed=0, eone_device=None)
    with pytest.raises(ValueError, match="holds"):
        rb.regime_render("yang2025_phase_regimes", manifest_path)

    state = (
        manifest_path.parent
        / manifest["families"]["yang2025"]["experts"]["cruise"]["state"]["path"]
    )
    data = bytearray(state.read_bytes())
    data[-1] ^= 1
    state.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="sha256"):
        rb.regime_render("yang2025_regimes", manifest_path)


def test_interrupted_preparation_reuses_completed_experts_without_refitting(tmp_path, monkeypatch):
    out, inputs, info = _prepare(tmp_path, monkeypatch, ("yang2025_regimes",))
    manifest_path = Path(info["path"])
    states = {p.name: p.read_bytes() for p in (manifest_path.parent / rb.STATE_DIR).glob("*.npz")}
    manifest_path.unlink()

    def refit(*args, **kwargs):
        raise AssertionError("a completed expert was refitted")

    monkeypatch.setattr(rb, "_fit_expert", refit)
    again = rb.prepare_regime_fits(out, inputs, ("yang2025_regimes",), seed=0, eone_device=None)
    assert again["methods"] == ["yang2025_regimes"]
    assert {
        p.name: p.read_bytes() for p in (manifest_path.parent / rb.STATE_DIR).glob("*.npz")
    } == states
    with pytest.raises(ValueError, match="eone_device"):
        rb.prepare_regime_fits(
            tmp_path / "other", inputs, ("doerfler2026_eone_regimes",), seed=0, eone_device=None
        )
