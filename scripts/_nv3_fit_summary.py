"""Throwaway: one-line compact summary of a flight_v3 fit JSON.

The noise-v3 fit job wrappers (`scripts/_nv3_mkjobs_r3.py`) ship it base64 and
print one `FIT_SUMMARY {json}` line per fit into the job log after every seed
(objective, alternation history, rig L-BFGS use, widths, sigma_nu).
Usage: `_nv3_fit_summary.py <fit.json> [...]`."""

import json
import sys
from pathlib import Path

import numpy as np

KEYS = (
    "round",
    "whittle_nats",
    "whittle_move_per_cell",
    "rig_converged",
    "adam_s_per_step",
    "rig_lbfgs_evals",
    "rig_lbfgs_iters",
    "rig_lbfgs_restart_iters",
    "rig_s_per_eval",
    "latent_evals",
    "latent_s_per_eval",
    "latent_cache_s",
    "latent_wall_s",
    "peak_mem_mb",
    "latent_peak_mem_mb",
    "wall_s",
)
RIG = (
    "adam_steps",
    "lbfgs_frames",
    "lbfgs_frames_used",
    "lbfgs_stratified",
    "adam_wall_s",
    "lbfgs_wall_s",
    "lbfgs_evals",
    "lbfgs_restart_evals",
    "lbfgs_iters_used",
    "lbfgs_restart_iters_used",
    "lbfgs_loss_before",
    "lbfgs_loss_first_pass",
    "lbfgs_loss_after_restart",
    "lbfgs_restart_gain_per_cell",
    "converged",
    "peak_mem_mb",
)


def summary(path):
    d = json.loads(Path(path).read_text())
    o, obj, p = d["optimiser"], d["objective"], d["params"]
    g = np.asarray(p["gamma_hz"], dtype=np.float64)
    fr, lr = o.get("first_rig", {}), o.get("last_rig", {})
    diag = d.get("diagnostics") or {}
    init = diag.get("init_from") or {}
    return dict(
        file=path,
        git=d.get("git"),
        device=o.get("device"),
        line_kernel=o.get("line_kernel"),
        chunk_frames=o.get("chunk_frames"),
        lbfgs_rtol=(o.get("rig") or {}).get("lbfgs_rtol"),
        wall_s=o.get("wall_s"),
        peak_mem_mb=o.get("peak_mem_mb"),
        converged=o.get("converged"),
        which_converged=o.get("which_converged"),
        alternation_converged=o.get("alternation_converged"),
        rig_converged=o.get("rig_converged"),
        rounds_run=o.get("rounds_run"),
        total_nats=obj.get("total_nats"),
        whittle_nats=obj.get("whittle_nats"),
        whittle_latents_zero_nats=obj.get("whittle_latents_zero_nats"),
        rig_neg_log_prior_nats=obj.get("rig_neg_log_prior_nats"),
        ou_neg_log_prior_nats=obj.get("ou_neg_log_prior_nats"),
        n_cells=obj.get("n_cells"),
        total_per_cell=(obj["total_nats"] / obj["n_cells"]) if obj.get("n_cells") else None,
        first_rig={k: fr.get(k) for k in RIG},
        last_rig={k: lr.get(k) for k in RIG},
        history=[{k: h.get(k) for k in KEYS if k in h} for h in o.get("history", [])],
        polish=o.get("polish"),
        front_end=d.get("front_end"),
        latent_fit=(d.get("latents") or {}).get("fit"),
        sigma_nu=p.get("sigma_nu"),
        gamma_median=float(np.median(g)),
        gamma_median_per_rotor=np.median(g, axis=1).round(4).tolist(),
        gamma_max_per_rotor=g.max(axis=1).round(3).tolist(),
        gamma_argmax_k_per_rotor=(g.argmax(axis=1) + 1).tolist(),
        gamma_over_50hz=int((g > 50.0).sum()),
        gamma_shape=list(g.shape),
        amp_exp=(p.get("profile") or {}).get("amp_exp"),
        span_pins=(diag.get("span_pins") or {}).get("pinned"),
        speed_span=(diag.get("span_pins") or {}).get("speed_span"),
        init_from={k: v for k, v in init.items() if k != "v2_floor_ctrl_db"},
        batch=diag.get("batch"),
    )


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print("FIT_SUMMARY", json.dumps(summary(path), default=str), flush=True)
