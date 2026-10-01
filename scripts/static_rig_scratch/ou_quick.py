"""Quick OU-shaft profile per DREGON single: (σ_ν, λ) from the free per-order
widths in width_law.json, amplitudes read at the peaks with the OU line
shape (non-floor), stored in results/static_rig/single_rotor/ou/<key>.json."""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "notebooks")
import single_rotor_lab as L  # noqa: E402
from experiments.static_rig import single_rotor as SR  # noqa: E402

OUT = Path("results/static_rig/single_rotor/ou")
OUT.mkdir(parents=True, exist_ok=True)
wl = json.loads(Path("results/static_rig/single_rotor/width_law.json").read_text())
for key in sys.argv[1:] or list(L.TABLE):
    t0 = time.time()
    s, _ = L.TABLE[key]
    x, fs, model = L._analysed(key)
    a, b = model.span
    x = x[:, a:b].astype(np.float64)
    r = wl[key]
    gam, ok = np.array(r["gamma_k"]), np.array(r["ok"], bool)
    orders = np.arange(1, gam.size + 1)
    sn, lm = SR.fit_width_law(orders[ok], gam[ok])
    fit = SR.line_powers_ou(x, fs, s, sn, lm)
    h = fit.gamma
    (OUT / f"{key}.json").write_text(
        json.dumps(
            {
                "s": s,
                "sigma_nu": sn,
                "lam": lm,
                "orders": fit.orders.tolist(),
                "amp2": fit.amp2.tolist(),
                "T": fit.T,
                "width_orders_used": int(ok.sum()),
            }
        )
    )
    resid = np.log(h[ok]) - np.log(gam[ok])
    print(
        f"{key}: σ_ν {sn:.4f} rev/s ({100 * sn / s:.2f} %), λ {lm:.2f} 1/s (memory {1 / lm:.2f} s), D=2σ²/λ {2 * sn**2 / lm:.1e}; "
        f"HWHM k=10/40/80: {h[9]:.2f}/{h[39]:.2f}/{h[79]:.2f} Hz; width residual sd {resid.std():.2f} (ln) over {ok.sum()} orders; {time.time() - t0:.0f} s",
        flush=True,
    )
