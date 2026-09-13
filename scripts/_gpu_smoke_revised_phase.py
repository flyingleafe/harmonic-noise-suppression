"""Temporary GPU smoke + CPU planted-control harness for revised phase.

GPU smoke (default):
  NFFT=16384, hop=1024, work grid 64 kHz, K=230 orders, R=4 rotors, M=8 mics,
  frame_chunk=1, harmonic_chunk=8, on one planted 2 s clip.
  Runs three WARM forward+backward repeats (no K reduction), records each
  repeat's elapsed time and peak allocated memory, prints summary statistics,
  import paths and exits 0 on success / nonzero on failure.

CPU planted control (--cpu-planted-control):
  Plants shared-shaft and independent-per-order clips at real-like rotor rates
  (standby/ramp/cruise) and runs the production moment stage
  (window=4096, hop=256, lags [1,2,4,8,16,32,64], preregistered 10 dB gate).
  Reports whether dynamics are identified; if not, exits non-zero and writes a
  diagnostic JSON.  Designed for uni-cpu.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

# In shared omnirun worktrees the script may be unpacked into a fresh tree
# while PYTHONPATH still points at a stale cached src tree.  Force imports
# from this script's own repository root first.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "src"))

import numpy as np
import torch

from experiments.stochastic_fit import revised_phase as RP
from experiments.stochastic_fit.data import Clip

SR = 16000
N_FFT = 16384
HOP = 1024
SAMPLE_RATE_WORK = 64000
N_ROTORS = 4
N_MICS = 8
K_CAP = 230
DURATION_S = 2.0


def _planted_clip() -> Clip:
    n = int(SR * DURATION_S)
    rng = np.random.default_rng(7)
    rps = np.full((N_ROTORS, n), 75.0)  # cruise-like
    # simple audio: not used by predict_spectrum, but Clip wants a real array
    audio = rng.standard_normal((N_MICS, n)).astype(np.float32) * 1e-4
    return Clip("smoke", "synthetic", audio, rps, SR)


def _tiny_export() -> dict[str, object]:
    profile = [-6.0] * K_CAP
    return dict(
        schema_version=RP.SCHEMA_VERSION,
        model_family=RP.MODEL_FAMILY,
        rig_id="smoke",
        parameters=dict(
            lam=6.0,
            sigma=6.0,
            d_scalar=0.5,
            delay_s={str(r): 0.0 for r in range(N_ROTORS)},
            bias_hz={},
            bias_mean_hz=[0.5] * N_ROTORS,
            bias_prior_std_hz=0.5,
            profile_db=[profile],
            amp_exp=0.0,
            floor_mean_db=-60.0,
            floor_shape_db=[0.0] * RP.FLOOR_SHAPE_N_CTRL,
            floor_ctrl_hz=np.geomspace(30.0, 8000.0, RP.FLOOR_SHAPE_N_CTRL).tolist(),
            floor_tilt_db_oct=0.0,
            floor_exp=0.0,
            floor_static_rel=0.0,
            mic_gain_db=[[0.0]] * N_MICS,
            gain_all_db=[0.0] * N_MICS,
            mic_floor_db=[0.0] * N_MICS,
            k_cap=K_CAP,
            n_mics=N_MICS,
            n_rotors=N_ROTORS,
        ),
        training_provenance=dict(
            manifest_path=None,
            manifest_sha256=None,
            clips=[],
            front_end=dict(
                n_fft=N_FFT,
                hop=HOP,
                sr=SR,
                band_hz=[30.0, 7900.0],
                sample_rate_work=SAMPLE_RATE_WORK,
            ),
            model_grid_hz=SAMPLE_RATE_WORK,
            analysis_grid_hz=SR,
            state_rate_hz=1000.0,
            optimizer=dict(
                iters=1, lr=0.05, frame_chunk=1, frames_per_step=None, atom_dtype="float32"
            ),
            seed=0,
            composite_temperature=1.0,
            priors=dict(bias_std_hz=0.5, log_d_mean=0.0, log_d_std=2.0),
        ),
        diagnostics=dict(identified=True, map_state={}),
    )


def _build_smoke_model(device: torch.device) -> tuple[RP._RevisedModel, dict[str, object]]:
    clip = _planted_clip()
    export = _tiny_export()
    params: dict[str, object] = export["parameters"]  # type: ignore[assignment]

    cfg = RP._config_from_export(export, n_fft=N_FFT, hop=HOP)
    cfg = RP.FitConfig(
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop=cfg.hop,
        sample_rate_work=SAMPLE_RATE_WORK,
        band_hz=cfg.band_hz,
        state_rate_hz=cfg.state_rate_hz,
        k_cap=cfg.k_cap,
        delay_s=cfg.delay_s,
        frame_chunk=1,
        harmonic_chunk=8,
        temperature=1.0,
        atom_dtype="float32",
    )
    dynamics = RP._dynamics_from_export(export)

    model = RP._RevisedModel(
        [(clip.clip_id, clip)],
        dynamics=dynamics,
        config=cfg,
        device=str(device),
        observe=True,
    )
    model.load_parameters(params)
    model.train()
    return model, params


def _run_gpu_smoke(out: Path | None = None) -> int:
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available")
        return 1

    device = torch.device("cuda:0")
    print(f"device: {torch.cuda.get_device_name(device)}")

    # import provenance
    print(f"revised_phase: {RP.__file__}")
    print(f"cwd: {Path.cwd().resolve()}")

    # build model once; each repeat reuses it but resets peak memory
    model, _params = _build_smoke_model(device)
    ci = 0
    chunk = np.arange(model.clips[0].starts.size)

    times: list[float] = []
    peaks_gb: list[float] = []
    for repeat in range(3):
        torch.cuda.reset_peak_memory_stats(device)
        # zero gradients between repeats so the backward graph is fresh
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        t0 = time.time()
        loss = model.chunk_risk(ci, chunk, scale=1.0)
        forward_obj = float(loss.detach())
        if not math.isfinite(forward_obj):
            print(f"FAIL repeat {repeat}: non-finite objective {forward_obj}")
            return 1
        loss.backward()

        finite_grads = 0
        total_grads = 0
        for p in model.parameters():
            if p.grad is not None:
                total_grads += 1
                if torch.isfinite(p.grad).all():
                    finite_grads += 1
        if finite_grads != total_grads or total_grads == 0:
            print(f"FAIL repeat {repeat}: missing or non-finite gradients ({finite_grads}/{total_grads})")
            return 1

        elapsed = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
        times.append(elapsed)
        peaks_gb.append(peak_gb)
        print(
            f"repeat {repeat}: objective={forward_obj:.6f} elapsed_s={elapsed:.3f} "
            f"peak_allocated_gb={peak_gb:.3f}"
        )

    arr_t = np.asarray(times)
    arr_p = np.asarray(peaks_gb)
    summary = dict(
        repeats=len(times),
        elapsed_s=dict(
            mean=float(arr_t.mean()),
            std=float(arr_t.std()),
            min=float(arr_t.min()),
            max=float(arr_t.max()),
            per_repeat=[float(t) for t in times],
        ),
        peak_allocated_gb=dict(
            mean=float(arr_p.mean()),
            std=float(arr_p.std()),
            min=float(arr_p.min()),
            max=float(arr_p.max()),
            per_repeat=[float(p) for p in peaks_gb],
        ),
        device=str(torch.cuda.get_device_name(device)),
        torch_version=str(torch.__version__),
    )
    print("summary:")
    print(f"  repeats: {len(times)}")
    print(f"  elapsed_s: mean={arr_t.mean():.3f} std={arr_t.std():.3f} min={arr_t.min():.3f} max={arr_t.max():.3f}")
    print(f"  peak_allocated_gb: mean={arr_p.mean():.3f} std={arr_p.std():.3f} min={arr_p.min():.3f} max={arr_p.max():.3f}")
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))
        print(f"wrote {out}")
    print("PASS")
    return 0


def _production_moment_config() -> RP.FitConfig:
    """The production moment front end and gate, independent of spectral fit params."""
    return RP.FitConfig(
        n_fft=N_FFT,
        hop=HOP,
        sr=SR,
        sample_rate_work=SAMPLE_RATE_WORK,
        band_hz=(30.0, 7900.0),
        state_rate_hz=500.0,
        k_cap=130,
        delay_s=(0.0, 0.0, 0.0, 0.0),
        iters=1,
        lr=0.01,
        seed=0,
        frame_chunk=1,
        frames_per_step=None,
        temperature=1.0,
        bias_std_hz=0.5,
        log_d_mean=0.0,
        log_d_std=2.0,
        atom_dtype="float32",
        harmonic_chunk=8,
        moments=RP.MomentConfig(
            window=4096,
            hop=256,
            lags=(1, 2, 4, 8, 16, 32, 64),
        ),
        gate=RP.MomentGate(),
        provenance=dict(
            rig="bench",
            dataset="planted",
            bench_diagnostic_only=True,
            scored_arm=False,
        ),
    )


def _synthetic_harmonic_clip(
    rps: np.ndarray,
    thetas: dict[tuple[int, int], np.ndarray],
    d: float,
    *,
    n_mics: int = 8,
    seed: int = 0,
    clip_id: str = "planted",
) -> Clip:
    """A clip whose audio is a sum of harmonics with optional phase noise.

    ``thetas`` maps ``(rotor_index, order_k)`` to a shaft-phase path (radians);
    the same path across orders of one rotor gives a shared shaft, different
    paths give independent per-order motions.  ``d`` is the independent
    acoustic diffusion in rad^2/s, added as a Wiener phase per harmonic.
    """
    rng = np.random.default_rng(seed)
    R, T = rps.shape
    sr = float(SR)
    audio = np.zeros((n_mics, T), dtype=np.float64)
    phase_raw = 2.0 * math.pi * np.cumsum(rps, axis=1) / sr
    for (r, k), theta in thetas.items():
        eps = np.zeros(T, dtype=np.float64)
        if d > 0.0:
            eps = np.cumsum(rng.normal(0.0, np.sqrt(2.0 * d / sr), size=T))
        for m in range(n_mics):
            alpha = rng.uniform(0.0, 2.0 * math.pi)
            audio[m] += np.cos(k * phase_raw[r] + k * theta + eps + alpha)
    return Clip(clip_id, "synthetic", audio.astype(np.float32), rps, SR)


def _real_like_rps(duration_s: float = 48.0, seed: int = 0) -> np.ndarray:
    """Four-rotor RPS trajectories: 8 s standby @20, 8 s ramp to 80, 32 s cruise @80."""
    rng = np.random.default_rng(seed)
    T = int(duration_s * SR)
    base = np.empty(T, dtype=np.float64)
    standby_samples = int(8.0 * SR)
    ramp_samples = int(16.0 * SR)
    base[:standby_samples] = 20.0
    base[standby_samples:ramp_samples] = np.linspace(
        20.0, 80.0, ramp_samples - standby_samples
    )
    base[ramp_samples:] = 80.0
    # small rotor-to-rotor offsets plus telemetry-like jitter
    rps = np.stack(
        [base + rng.normal(0.0, 0.1, size=T) + offset for offset in [0.0, 0.2, -0.1, 0.15]]
    )
    rps = np.clip(rps, 20.0, None)
    return rps


def _ou_path(T: int, lam: float, sigma: float, seed: int = 0) -> np.ndarray:
    """One shared-shaft OU phase path on the audio sample grid."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / float(SR)
    innov = rng.standard_normal((T, 2))
    theta, _nu = RP.simulate_state(innov, lam=lam, sigma=sigma, dt=dt)
    return np.asarray(theta[0])


def _run_cpu_planted_control(out: Path | None) -> int:
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"revised_phase: {RP.__file__}")
    print(f"cwd: {Path.cwd().resolve()}")

    cfg = _production_moment_config()
    rps = _real_like_rps(duration_s=48.0, seed=11)
    T = rps.shape[1]
    lam_true, sigma_true, d_true = 6.0, 6.0, 0.05

    # shared shaft: all orders of each rotor feel the same rotor theta
    shared_orders: dict[tuple[int, int], np.ndarray] = {}
    for r in range(rps.shape[0]):
        theta_shared = _ou_path(T, lam=lam_true, sigma=sigma_true, seed=21 + r)
        for k in range(2, 41):
            shared_orders[(r, k)] = theta_shared
    clip_shared = _synthetic_harmonic_clip(
        rps, shared_orders, d=d_true, n_mics=N_MICS, seed=31, clip_id="shared_shaft"
    )

    # independent per-order: each (rotor, order) gets its own OU path with the SAME marginals
    independent_orders: dict[tuple[int, int], np.ndarray] = {}
    for r in range(rps.shape[0]):
        for k in range(2, 41):
            independent_orders[(r, k)] = _ou_path(
                T, lam=lam_true, sigma=sigma_true, seed=100 * r + k
            )
    clip_indep = _synthetic_harmonic_clip(
        rps, independent_orders, d=d_true, n_mics=N_MICS, seed=41, clip_id="independent_per_order"
    )

    results: list[dict[str, object]] = []
    for label, clip in [("shared_shaft", clip_shared), ("independent_per_order", clip_indep)]:
        print(f"\nmoment stage: {label}")
        dynamics = RP.estimate_shaft_dynamics([(clip.clip_id, clip)], gate=cfg.gate, config=cfg)
        print(
            f"  lam {dynamics.lam:.4g} 1/s   sigma {dynamics.sigma:.4g} rad/s   "
            f"D_init {dynamics.d_init:.4g} rad^2/s   identified {dynamics.identified}"
        )
        reason = dynamics.diagnostics.get("unidentifiable_reason")
        if reason:
            print(f"  reason: {reason}")
        results.append(
            dict(
                label=label,
                lam=float(dynamics.lam),
                sigma=float(dynamics.sigma),
                d_init=float(dynamics.d_init),
                identified=bool(dynamics.identified),
                unidentifiable_reason=str(reason) if reason else None,
                diagnostics=dynamics.diagnostics,
            )
        )
    # Shared shaft must be identified; independent per-order should report low shared sigma.
    shared = next(r for r in results if r["label"] == "shared_shaft")
    indep = next(r for r in results if r["label"] == "independent_per_order")
    shared_ok = bool(shared["identified"])
    shared_sigma = float(shared["sigma"])
    indep_sigma = float(indep["sigma"])
    indep_ok = indep_sigma < 0.5 * shared_sigma if shared_ok else False

    summary = dict(
        status="identified" if (shared_ok and indep_ok) else "flagged",
        shared_shaft_identified=shared_ok,
        independent_control_low_sigma=indep_ok,
        planted_lam=lam_true,
        planted_sigma=sigma_true,
        planted_d=d_true,
        results=results,
        config=dict(
            window=cfg.moments.window,
            hop=cfg.moments.hop,
            lags=list(cfg.moments.lags),
            gate=cfg.gate.as_dict(),
        ),
    )

    if out is not None:
        out.write_text(json.dumps(summary, indent=2, default=str))
        print(f"\nwrote diagnostic summary to {out}")

    print("\nsummary:")
    print(f"  shared shaft identified: {shared_ok}")
    print(f"  independent control low sigma: {indep_ok}")
    if shared_ok and indep_ok:
        print("PASS: production moment config distinguishes shared shaft from independent per-order motion")
        return 0
    else:
        print("FLAGGED: production moment config did not cleanly separate shared vs independent motion")
        return 2


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument(
        "--cpu-planted-control",
        action="store_true",
        help="run the CPU planted shared-vs-independent moment control instead of the GPU smoke",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional JSON path to write the planted-control diagnostic summary",
    )
    args = ap.parse_args()

    if args.cpu_planted_control:
        return _run_cpu_planted_control(args.out)
    return _run_gpu_smoke(args.out)


if __name__ == "__main__":
    sys.exit(main())
