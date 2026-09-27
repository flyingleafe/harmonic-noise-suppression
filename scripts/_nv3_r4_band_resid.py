"""Throwaway: per-band residual (observed mean periodogram / model expectation) of a
v3 fit on ITS OWN pool, mic mean, latents zero. Usage: python /tmp/r4_band_resid.py FIT.json SET [above_500|full]"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import noise_v3_latent_runaway_figs as F  # noqa: E402

from experiments.noise_model import fit as FT  # noqa: E402
from experiments.noise_model import model as MD  # noqa: E402
from experiments.noise_model import supports as SUP  # noqa: E402

fit_path, set_name, band = sys.argv[1], sys.argv[2], sys.argv[3]
fit = F.load(Path(fit_path))
name = str(fit["support"])
rig = "dregon" if name.startswith("dregon") else "michaels"
specs = {s.name: s for s in SUP.support_set(set_name)}
sups = [SUP.load_support(specs[n]) for n in fit["supports"]]
members = [
    (
        s.name,
        np.asarray(s.power, dtype=np.float64),
        np.asarray(s.carrier_rev_s_audio, dtype=np.float64),
        np.asarray(s.frame_starts, dtype=np.int64),
    )
    for s in sups
]
batch = MD.flight_batch(
    name=name,
    members=members,
    sr=int(sups[0].sr),
    n_fft=int(sups[0].n_fft),
    hop=int(sups[0].hop),
    k_cap=F.K_CAP,
    frame_stride=F.FRAME_STRIDE,
    channel_gains=FT.load_channel_gains(F.CHANNEL_GAINS, rig=rig, band=band),
    device="cpu",
    chunk_frames=F.CHUNK_FRAMES,
    sr_work=F.V3_SR_WORK,
)
batch = MD.with_blocks(batch, float(fit["latents"]["block_s"]))
params = MD.params_from_dict(fit["params"])
with torch.no_grad():
    m = MD.forward(batch, params, unit_autocorr=True).numpy()  # (M, N, F)
obs = batch.power.numpy()  # (M, N, F)
freqs = np.fft.rfftfreq(int(sups[0].n_fft), 1.0 / int(sups[0].sr))
bands = [
    (100, 300),
    (300, 700),
    (700, 1500),
    (1500, 3000),
    (3000, 5000),
    (5000, 7000),
    (7000, 7900),
]
print(fit_path, "frames", obs.shape[1], "mics", obs.shape[0])
print("band     obs-model dB (mic mean, frame mean)   | per mic")
for lo, hi in bands:
    sel = (freqs >= lo) & (freqs < hi)
    o = obs[:, :, sel].mean(axis=(1, 2))
    mm = m[:, :, sel].mean(axis=(1, 2))
    print(
        f"{lo:5d}-{hi:<5d} {10 * np.log10(o.mean() / mm.mean()):+6.2f}  | "
        + " ".join(f"{v:+5.1f}" for v in 10 * np.log10(o / mm))
    )
# lines vs floor share of the model in each band
off = MD.params_from_dict(fit["params"])
off = off.__class__(**{**off.__dict__, "profile_db": torch.full_like(off.profile_db, -300.0)})
with torch.no_grad():
    fl = MD.forward(batch, off, unit_autocorr=True).numpy()
print("band     model line share (1 - floor/model), obs/floor dB")
for lo, hi in bands:
    sel = (freqs >= lo) & (freqs < hi)
    mm = m[:, :, sel].mean()
    ff = fl[:, :, sel].mean()
    oo = obs[:, :, sel].mean()
    print(f"{lo:5d}-{hi:<5d} {1 - ff / mm:5.2f}  {10 * np.log10(oo / ff):+6.2f}")
