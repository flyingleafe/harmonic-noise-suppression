import json
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "/tmp")
from dregon_simple import strict_span

from experiments.static_rig import spectra as S

units = {v["key"]: v for v in json.load(open("/tmp/sr_dregon/units.json"))}
prior = {
    r["uid"].split("__")[-1]: r["full"]["whitened"]["speed"]
    for r in json.load(open("/tmp/bretthorst_dregon.json"))
}
WIND = {"Motor1": [6], "Motor2": [0], "Motor3": [1, 2], "Motor4": [4]}
R = 400
HB = 40.0  # envelope rate, demod half-band (Hz)
LAGS = [0.1, 0.3, 1.0, 3.0]
print(
    "phase wander: rms of [phi(t+tau) - phi(t) - 2*pi*k*s_mean*tau] / (2*pi*k), in revolutions (and degrees for tau=1 s); median over good mics; k = 2 and 4"
)
print(
    "key              s_mean    "
    + "".join(f"  tau={l:<4g}" for l in LAGS)
    + "   1s(deg)  k4/k2 ratio@1s  sqrt(t) check: r(1)/r(0.1)"
)
for key in sys.argv[1:] or [
    f"motor_Motor{m}_{t}" for m in (1, 2, 3, 4) for t in (50, 60, 70, 80, 90)
]:
    u = units[key]
    fs = u["fs"]
    x0 = np.load(u["audio"], mmap_mode="r")
    a, b = S.motor_on_span(np.asarray(x0), fs, S.Params())
    x = np.asarray(x0[:, a:b], dtype=np.float64)
    s0 = prior[key]
    a2, b2 = strict_span(x, fs, s0)
    x = x[:, a2:b2]
    good = [c for c in range(x.shape[0]) if c not in WIND[key.split("_")[1]]]
    nb = fs // R
    n_blk = x.shape[1] // nb
    t = np.arange(n_blk * nb) / fs
    res = {}
    for k in (2, 4):
        car = np.exp(-2j * np.pi * k * s0 * t)
        y = (x[good, : n_blk * nb] * car).reshape(len(good), n_blk, nb).mean(axis=2)
        L = int(round(0.75 / HB * R)) | 1
        w = np.hanning(L + 2)[1:-1]
        w /= w.sum()
        env = np.stack([np.convolve(y[c], w, mode="same") for c in range(len(good))])[:, L:-L]
        ph = np.unwrap(np.angle(env), axis=1)
        tt = np.arange(ph.shape[1]) / R
        # remove the mean speed (linear fit of the phase) -> residual phase in revolutions of the shaft
        rev = np.stack([(p - np.polyval(np.polyfit(tt, p, 1), tt)) / (2 * np.pi * k) for p in ph])
        s_mean = s0 + np.median([np.polyfit(tt, p, 1)[0] / (2 * np.pi * k) for p in ph])
        r = []
        for lag in LAGS:
            m = int(lag * R)
            d = rev[:, m:] - rev[:, :-m]
            r.append(np.median(np.sqrt(np.mean(d**2, axis=1))))
        res[k] = (s_mean, np.array(r))
    s_mean, r2 = res[2]
    r4 = res[4][1]
    print(
        f"{key:16s} {s_mean:8.3f}  "
        + "".join(f"  {v:8.4f}" for v in r2)
        + f"   {360 * r2[2]:6.1f}     {r4[2] / r2[2]:5.2f}          {r2[2] / r2[0]:5.2f} (sqrt(10)={np.sqrt(10):.2f})"
    )
