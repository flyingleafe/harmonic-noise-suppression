"""Temporary GPU smoke for revised-phase forward/backward memory.

NFFT=16384, hop=1024, work grid 64 kHz, K=230 orders, R=4 rotors, M=8 mics,
frame_chunk=1, harmonic_chunk=32, on one planted 1 s clip.
Prints CUDA device, finite objective/gradients, peak allocated GB, elapsed time
and import paths, then exits 0 on success / nonzero on failure.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

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


def main() -> int:
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

    clip = _planted_clip()
    export = _tiny_export()
    params: dict[str, object] = dict(export["parameters"])

    cfg = RP._config_from_export(export, n_fft=N_FFT, hop=HOP)
    cfg = RP.FitConfig(
        sr=cfg.sr,
        n_fft=cfg.n_fft,
        hop=cfg.hop,
        sample_rate_work=SAMPLE_RATE_WORK,
        band_hz=cfg.band_hz,
        state_rate_hz=cfg.state_rate_hz,
        k_cap=cfg.k_cap,
        frame_chunk=1,
        harmonic_chunk=32,
        temperature=1.0,
        atom_dtype="float32",
    )
    dynamics = RP._dynamics_from_export(export)

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()

    model = RP._RevisedModel(
        [(clip.clip_id, clip)],
        dynamics=dynamics,
        config=cfg,
        device=str(device),
        observe=True,
    )
    model.load_parameters(params)
    model.train()

    ci = 0
    chunk = np.arange(model.clips[0].starts.size)
    loss = model.chunk_risk(ci, chunk, scale=1.0)
    print(f"forward_objective: {float(loss.detach()):.6f}")
    if not math.isfinite(float(loss.detach())):
        print("FAIL: non-finite objective")
        return 1

    loss.backward()

    finite_grads = 0
    total_grads = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grads += 1
            if torch.isfinite(p.grad).all():
                finite_grads += 1
    print(f"gradients: {finite_grads}/{total_grads} finite")
    if finite_grads != total_grads or total_grads == 0:
        print("FAIL: missing or non-finite gradients")
        return 1

    elapsed = time.time() - t0
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"elapsed_s: {elapsed:.3f}")
    print(f"peak_allocated_gb: {peak_gb:.3f}")
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
