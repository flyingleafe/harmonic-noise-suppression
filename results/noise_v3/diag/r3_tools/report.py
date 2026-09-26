"""Throwaway: round-3 fit tables (per restart and selected) from the fit JSONs."""

import json
import math
import sys
from pathlib import Path

import numpy as np

POOLS = ("dregon_room2_floor", "michaels_fly125_cruise", "michaels_fly125_standby")
TARGET = {
    "dregon_room2_floor": -18043573.7,
    "michaels_fly125_cruise": -24805567.3,
    "michaels_fly125_standby": -6943448.2,
}
R2_MEAS = json.loads(Path("results/noise_v3/diag/reprice_measured.json").read_text())


def load(p):
    return json.loads(Path(p).read_text())


def decompose(x):
    pool = x.mean(axis=(0, -1), keepdims=True)
    ms = float(np.mean(x**2))
    static = float(np.mean(np.broadcast_to(pool, x.shape) ** 2))
    return static / ms if ms > 0 else float("nan"), math.sqrt(ms)


def fams(fit):
    wins = sorted(fit["latents"]["windows"], key=lambda w: int(w["window"]))
    lat = {
        n: np.stack([np.asarray(w[n], dtype=np.float64) for w in wins])
        for n in ("d", "v", "u", "uj")
    }
    out = {"d": lat["d"], "u": lat["u"], "u_j": lat["uj"]}
    for g, (k0, k1) in {"v k9-24": (8, 24), "v k25-60": (24, 60), "v k61+": (60, None)}.items():
        out[g] = lat["v"][:, :, k0:k1]
    return {k: decompose(v) for k, v in out.items()}


def rig_move(a, b):
    pa = np.asarray(a["params"]["profile"]["profile_db"])
    pb = np.asarray(b["params"]["profile"]["profile_db"])
    k = min(pa.shape[1], pb.shape[1])
    dp = pa[:, :k] - pb[:, :k]
    ga, gb = np.asarray(a["params"]["gamma_hz"]), np.asarray(b["params"]["gamma_hz"])
    dg = np.log(ga[:, :k]) - np.log(gb[:, :k])
    return dict(
        profile_rms=float(np.sqrt((dp**2).mean())),
        profile_max=float(np.abs(dp).max()),
        profile_mean=float(dp.mean()),
        log_gamma_rms=float(np.sqrt((dg**2).mean())),
    )


def main(fdir):
    fdir = Path(fdir)
    rows = {}
    for pool in POOLS:
        sel_path = fdir / f"{pool}__flight_v3.json"
        if not sel_path.exists():
            print("missing", sel_path)
            continue
        sel = load(sel_path)
        r2 = load(f"results/noise_v3/fits_r2/{pool}__flight_v3.json")
        o = sel["objective"]
        print(f"\n## {pool}: selected s{sel['restarts']['selected_seed']}")
        for p in sorted((fdir / "restarts").glob(f"{pool}__flight_v3__s*.json")):
            f = load(p)
            op = f["optimiser"]
            h = op["history"]
            moves = [x.get("move_per_cell") for x in h[1:]]
            iters = [(x.get("rig_lbfgs_iters"), x.get("rig_lbfgs_restart_iters")) for x in h]
            print(
                f"  {p.name}: rounds {op['rounds_run']} conv {op['converged']} alt {op['alternation_converged']} "
                f"rig {op['rig_converged']} which {op.get('which_converged')} wall {op['wall_s']:.0f}s "
                f"total {f['objective']['total_nats']:.1f} W {f['objective']['whittle_nats']:.1f} "
                f"rig {f['objective']['rig_neg_log_prior_nats']:.1f} ou {f['objective']['ou_neg_log_prior_nats']:.1f} "
                f"sigma_nu {f['params']['sigma_nu']:.3f}"
            )
            print(f"     moves {[f'{m:.2g}' for m in moves]}")
            print(
                f"     rig iters {iters[:4]}...{iters[-2:]} polish {(op.get('polish') or {}).get('rig_lbfgs_iters')}+{(op.get('polish') or {}).get('rig_lbfgs_restart_iters')} gain/cell {(op.get('polish') or {}).get('gain_per_cell')}"
            )
        rm = rig_move(sel, r2)
        tot = o["total_nats"]
        r2m = R2_MEAS[f"r2/{pool}"]["total_measured"]
        g = np.asarray(sel["params"]["gamma_hz"])
        rows[pool] = dict(
            total=tot,
            whittle=o["whittle_nats"],
            rig=o["rig_neg_log_prior_nats"],
            ou=o["ou_neg_log_prior_nats"],
            vs_target=tot - TARGET[pool],
            vs_r2_measured=tot - r2m,
            n_cells=o["n_cells"],
            rig_move_vs_r2=rm,
            shares=fams(sel),
            shares_r2=fams(r2),
            sigma_nu=sel["params"]["sigma_nu"],
            gamma_max=float(g.max()),
            gamma_median=float(np.median(g)),
            restarts=dict(
                best_minus_worst_per_cell=sel["restarts"]["best_minus_worst_per_cell"],
                best_minus_median_per_cell=sel["restarts"]["best_minus_median_per_cell"],
            ),
        )
        print(
            f"  total {tot:.1f} (target {TARGET[pool]:.1f}: {tot - TARGET[pool]:+.1f}; r2 under measured {r2m:.1f}: {tot - r2m:+.1f})"
        )
        print(
            f"  W {o['whittle_nats']:.1f} (r2 {r2['objective']['whittle_nats']:.1f}: {o['whittle_nats'] - r2['objective']['whittle_nats']:+.1f}) rig {o['rig_neg_log_prior_nats']:.1f} ou {o['ou_neg_log_prior_nats']:.1f}"
        )
        print(f"  rig move vs r2: {rm}")
        print(
            "  static share r2 -> r3 (rms r2 -> r3):",
            {
                k: f"{rows[pool]['shares_r2'][k][0]:.2f}->{v[0]:.2f} ({rows[pool]['shares_r2'][k][1]:.2f}->{v[1]:.2f})"
                for k, v in rows[pool]["shares"].items()
            },
        )
        print(
            f"  sigma_nu {sel['params']['sigma_nu']:.3f} (r2 {r2['params']['sigma_nu']:.3f}) gamma max {g.max():.2f} median {np.median(g):.3f} spread/cell {rows[pool]['restarts']}"
        )
    Path(fdir / "report.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
