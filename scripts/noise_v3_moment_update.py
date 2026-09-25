#!/usr/bin/env python
"""Noise model v3: one moment-matching (EM) update of the wander hyperparameters.

``docs/explainers/noise-model-v3-wander.qmd`` §3.3a: under a correctly-shrunk
fit, the fitted latents' mean square plus their posterior variance equals the
OU's sigma^2, and the lag-1 product plus the posterior lag-1 covariance equals
rho sigma^2. One M-step therefore sets, per latent family,

    sigma_hat^2 = ms_plus_post_var_db2
    rho_hat     = clip(lag1_product_plus_post_cov_db2 / sigma_hat^2, 0.05, 0.99)
    tau_hat     = -block_s / ln(rho_hat)

from the pooled sums ``scripts/noise_v3_checks.py latents`` already records
(``<checks>/latents/latents.json``, per fit key and family). A family whose
sigma_hat is 0 (a ``v`` order group held at sigma 0: its latents are pinned)
keeps sigma 0 and the tau it was fitted under.

    PYTHONPATH=src python scripts/noise_v3_moment_update.py --tag mm1 \\
        --latents results/noise_v3/checks/latents/latents.json

writes ``results/noise_v3/wander/<rig>_<tag>.json`` (``noise-v3-wander/2``, the
measured record's keys with the updated sigma / tau) from the pool each rig's
fit reads (DREGON: ``dregon``; Michael's: ``michaels_cruise``), and a side-by-side
table of every pool (Michael's standby included) in ``--md``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

WANDER_DIR = Path("results/noise_v3/wander")
SCHEMA = "noise-v3-wander/2"
RHO_CLIP = (0.05, 0.99)
#: the pool whose update becomes the rig file the fits read, per rig
RIG_POOL = {"dregon": "dregon", "michaels": "michaels_cruise"}
#: pool key -> rig
POOL_RIG = {"dregon": "dregon", "michaels_cruise": "michaels", "michaels_standby": "michaels"}
#: scalar families: (latents.json family, sigma key, tau key)
SCALARS = (
    ("d", "sigma_d_db", "tau_d_s"),
    ("u", "sigma_u_db", "tau_u_s"),
    ("uj", "sigma_uj_db", "tau_uj_s"),
    ("v", "sigma_v_db", "tau_v_s"),
)


def moment(fam: dict[str, Any], block_s: float, tau_old: float) -> dict[str, float]:
    """sigma_hat, rho_hat, tau_hat of one family (``tau_old`` kept at sigma_hat 0)."""
    s2 = float(fam.get("ms_plus_post_var_db2", 0.0)) if fam.get("n_values") else 0.0
    if not s2 > 0.0:
        return dict(sigma_db=0.0, rho=math.exp(-block_s / tau_old), tau_s=tau_old, rho_raw=math.nan)
    rho_raw = float(fam["lag1_product_plus_post_cov_db2"]) / s2
    rho = min(max(rho_raw, RHO_CLIP[0]), RHO_CLIP[1])
    return dict(sigma_db=math.sqrt(s2), rho=rho, tau_s=-block_s / math.log(rho), rho_raw=rho_raw)


def update_pool(res: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The updated ``Wander`` fields of one pool and its old/new table rows."""
    w = res["wander"]
    bs = float(w["block_s"])
    fams = res["families"]
    new: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    def row(name: str, s_old: float, t_old: float, m: dict[str, float]) -> None:
        rows.append(
            dict(
                family=name,
                n_values=int(fams.get(name, {}).get("n_values", 0)),
                sigma_old_db=s_old,
                sigma_new_db=m["sigma_db"],
                tau_old_s=t_old,
                tau_new_s=m["tau_s"],
                rho_old=math.exp(-bs / t_old) if t_old > 0 else math.nan,
                rho_new=m["rho"],
                rho_raw=m["rho_raw"],
            )
        )

    for name, s_key, t_key in SCALARS:
        s_old, t_old = float(w[s_key]), float(w[t_key])
        m = moment(fams.get(name, {}), bs, t_old)
        new[s_key], new[t_key] = m["sigma_db"], m["tau_s"]
        row(name, s_old, t_old, m)
    by_s, by_t = w.get("sigma_v_db_by_order"), w.get("tau_v_s_by_order")
    if by_s:
        edges = [int(k) for k in by_s["k_edges"]]
        t_old_g = by_t["tau_s"] if by_t else [float(w["tau_v_s"])] * (len(edges) - 1)
        groups = list(res["v_order_groups"])  # "v k1-2", ..., in edge order
        if len(groups) != len(edges) - 1:
            raise ValueError(f"v order groups {groups} do not match k_edges {edges}")
        sig, tau, src = [], [], []
        for g, s_old, t_old in zip(groups, by_s["sigma_db"], t_old_g, strict=True):
            m = moment(fams.get(g, {}), bs, float(t_old))
            sig.append(m["sigma_db"])
            tau.append(m["tau_s"])
            src.append("moment" if m["sigma_db"] > 0.0 else "kept (sigma 0)")
            row(g, float(s_old), float(t_old), m)
        new["sigma_v_db_by_order"] = dict(k_edges=edges, sigma_db=sig)
        new["tau_v_s_by_order"] = dict(k_edges=edges, tau_s=tau, tau_source=src)
    return new, rows


def _f(x: float, fmt: str = ".3f") -> str:
    return "—" if not math.isfinite(x) else format(x, fmt)


def md_table(key: str, res: dict[str, Any], rows: list[dict[str, Any]], tag: str) -> list[str]:
    out = [
        f"### {key} (`{res['fit']}`, fitted under `{res.get('wander_path', '?')}`)",
        "",
        f"| family | n values | σ old dB | σ {tag} dB | σ {tag}/old | τ old s | τ {tag} s "
        f"| ρ old | ρ {tag} (raw) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        ratio = r["sigma_new_db"] / r["sigma_old_db"] if r["sigma_old_db"] > 0 else math.nan
        out.append(
            f"| {r['family']} | {r['n_values']} | {_f(r['sigma_old_db'])} | "
            f"{_f(r['sigma_new_db'])} | {_f(ratio, '.2f')} | {_f(r['tau_old_s'])} | "
            f"{_f(r['tau_new_s'])} | {_f(r['rho_old'])} | {_f(r['rho_new'])} "
            f"({_f(r['rho_raw'])}) |"
        )
    return out + [""]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="one moment-matching update of the v3 wander hyperparameters"
    )
    ap.add_argument("--latents", type=Path, required=True, help="noise_v3_checks.py latents JSON")
    ap.add_argument("--tag", required=True, help="output tag: <rig>_<tag>.json (e.g. mm1)")
    ap.add_argument("--wander-dir", type=Path, default=WANDER_DIR)
    ap.add_argument("--md", type=Path, default=None, help="default <wander-dir>/moment_update.md")
    args = ap.parse_args(argv)
    lat = json.loads(args.latents.read_text())
    tag = str(args.tag)
    md_path = args.md or args.wander_dir / "moment_update.md"
    lines = [
        f"# Wander moment update `{tag}` (explainer §3.3a, one M-step)",
        "",
        f"From `{args.latents}` (check (e) sums, git `{lat.get('git', '?')}`): "
        "σ̂² = fitted mean square + OU-smoother posterior variance, ρ̂ = (lag-1 product + "
        f"posterior lag-1 covariance) / σ̂², clipped to {RHO_CLIP}, τ̂ = −block_s / ln ρ̂. "
        "'old' = the law the fit was fitted under. A σ-0 group keeps σ 0 and its τ. "
        "Rig files: " + ", ".join(f"`{r}_{tag}.json` ← `{p}`" for r, p in RIG_POOL.items()) + ".",
        "",
    ]
    for key, res in lat["fits"].items():
        fit = json.loads(Path(res["fit"]).read_text())
        res = dict(res, wander_path=fit["priors"].get("wander", {}).get("path", "?"))
        new, rows = update_pool(res)
        lines += md_table(key, res, rows, tag)
        rig = POOL_RIG[key]
        if RIG_POOL[rig] != key:
            continue
        base = json.loads((args.wander_dir / f"{rig}.json").read_text())
        rec = copy.deepcopy(base)
        rec["schema"] = SCHEMA
        rec.update(new)
        prov = dict(base.get("provenance", {}))
        prov.update(
            generated_by="scripts/noise_v3_moment_update.py",
            method=f"moment_update_{tag.removeprefix('mm')} from {res['fit']}",
            fitted_under=res["wander_path"],
            latents=str(args.latents),
            latents_git=lat.get("git"),
            rho_clip=list(RHO_CLIP),
            rows=rows,
        )
        rec["provenance"] = prov
        out = args.wander_dir / f"{rig}_{tag}.json"
        out.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"wrote {out}")
    md_path.write_text("\n".join(lines))
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
