"""Throwaway: findings §2 rows (rig to convergence; ridge + alternation) from a diag-A JSON.

Usage: python results/noise_v3/diag/r3_tools/diag_a_rows.py results/noise_v3/diag/rig/A/<pool>.json
"""

import json
import sys
from pathlib import Path


def main(path: str) -> None:
    d = json.loads(Path(path).read_text())
    rc = d["rig_converged"]
    op, ch = rc["optimiser"], rc["change"]
    rp = ch["rig_prior_nats"]
    print(
        f"rig converged: iters {op['iters']} stop {op['stop']} wall {op['wall_s']:.0f} s "
        f"s/eval {op['s_per_eval']:.2f} gain {-op['gain_nats']:+.1f} total {ch['total_nats']:+.1f} "
        f"W {ch['whittle_nats']:+.1f} gamma {rp['gamma_hz']:+.1f} profile {rp['profile_db']:+.1f} "
        f"z {rp['floor_shape_z']:+.1f} sigma_nu {rp['sigma_nu']:+.1f}"
    )
    print("  bands", {k: round(v, 1) for k, v in ch["whittle_band_nats"].items()})
    g0 = d["fit_state"]["rig"]["gamma_hz"]
    g1 = rc["rig"]["gamma_hz"]
    mx = lambda g: max(max(r) for r in g) if isinstance(g[0], list) else max(g)  # noqa: E731
    print(f"  gamma max {mx(g0):.1f} -> {mx(g1):.1f} Hz")
    print(f"ridge: total {d['ridge']['change']['total_nats']:+.1f}")
    for r in d["alt"]["rounds"]:
        c = r["change"]
        st = r["price"]["static"]
        print(
            f"alt round {r['round']}: total {c['total_nats']:+.1f} W {c['whittle_nats']:+.1f} "
            f"rig {c['rig_prior_nats']['total']:+.1f} OU {c['ou_prior_nats']['total']:+.1f} "
            f"rig iters {r['rig']['iters']} ({r['rig']['stop']}) "
            f"static d/v/u/uj {st['d']['static_share']:.2f} / {st['v']['static_share']:.2f} / "
            f"{st['u']['static_share']:.2f} / {st['uj']['static_share']:.2f}"
        )
        print("  bands", {k: round(v, 1) for k, v in c["whittle_band_nats"].items()})
    print(f"wall {d.get('wall_s')}")


if __name__ == "__main__":
    main(sys.argv[1])
