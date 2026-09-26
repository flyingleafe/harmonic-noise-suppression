"""Throwaway: the round-3 verdict tables, r2 / r3a / r3b side by side.

Reads, per round TAG: the reduced fits and restarts (`results/noise_v3/fits_<tag>/`, r2:
`fits_r2`), the check set (`results/noise_v3/checks_<tag>/`: `findings.md` summary,
`latents/latents.json`, `heldout/heldout.json`, `parity/arm_*_v3.json`, `proxy_seeds/`),
the reprice under the measured wander (`results/noise_v3/diag/reprice_r3.json`) and the
figure data of `noise_v3_latent_runaway_figs.py --round TAG`
(`docs/explainers/noise-model-v3-latent-runaway/figdata_<tag>.json`: rig views, § Listen).
Writes `results/noise_v3/diag/verdict_r3.json` and prints markdown tables.

(e) ratios are against the MEASURED sigma for every round: the lag-0 sum (fitted mean
square + posterior variance, as each round's check computed it under its own law) over
the measured sigma^2 of `checks_r3a/latents/latents.json` (r3a's law is the measured one).
"""

import json
import math
from pathlib import Path

import numpy as np

ROUNDS = ("r2", "r3a", "r3b")
FITS = {
    "r2": "results/noise_v3/fits_r2",
    "r3a": "results/noise_v3/fits_r3a",
    "r3b": "results/noise_v3/fits_r3b",
}
CHECKS = {r: f"results/noise_v3/checks_{r}" for r in ROUNDS}
POOLS = {
    "dregon": "dregon_room2_floor",
    "michaels_cruise": "michaels_fly125_cruise",
    "michaels_standby": "michaels_fly125_standby",
}
TARGET = {
    "dregon_room2_floor": -18043573.7,
    "michaels_fly125_cruise": -24805567.3,
    "michaels_fly125_standby": -6943448.2,
}
FIGDIR = Path("docs/explainers/noise-model-v3-latent-runaway")
PROXY_GROUP = {"dregon": "dregon_cruise", "michaels": "michaels_cruise"}
GROUPS = ("1-2", "3-8", "9-24", "25-60", "61+")
FAMS_E = ("d", "u", "uj", "v", "v k1-2", "v k3-8", "v k9-24", "v k25-60", "v k61+")


def load(p):
    return json.loads(Path(p).read_text())


def decompose(x):
    pool = x.mean(axis=(0, -1), keepdims=True)
    ms = float(np.mean(x**2))
    static = float(np.mean(np.broadcast_to(pool, x.shape) ** 2))
    return static / ms if ms > 0 else float("nan")


def shares(fit):
    wins = sorted(fit["latents"]["windows"], key=lambda w: int(w["window"]))
    lat = {
        n: np.stack([np.asarray(w[n], dtype=np.float64) for w in wins])
        for n in ("d", "v", "u", "uj")
    }
    out = {"d": lat["d"], "u": lat["u"], "u_j": lat["uj"], "v": lat["v"]}
    for g, (k0, k1) in {
        "v k1-8": (0, 8),
        "v k9-24": (8, 24),
        "v k25-60": (24, 60),
        "v k61+": (60, None),
    }.items():
        out[g] = lat["v"][:, :, k0:k1]
    return {k: decompose(v) for k, v in out.items()}


def rig_iters(fit):
    h = fit["optimiser"]["history"]
    n = sum(
        int(x.get("rig_lbfgs_iters") or 0) + int(x.get("rig_lbfgs_restart_iters") or 0) for x in h
    )
    pol = fit["optimiser"].get("polish") or {}
    return n, int(pol.get("rig_lbfgs_iters") or 0) + int(pol.get("rig_lbfgs_restart_iters") or 0)


def objective(tag, reprice):
    out = {}
    for pool in TARGET:
        sel = load(f"{FITS[tag]}/{pool}__flight_v3.json")
        o = sel["objective"]
        rp = reprice.get(f"{tag}/{pool}", {})
        meas = rp.get("total_measured", o["total_nats"])
        per = []
        for p in sorted(Path(FITS[tag], "restarts").glob(f"{pool}__flight_v3__s*.json")):
            f = load(p)
            op = f["optimiser"]
            it, pol = rig_iters(f)
            per.append(
                dict(
                    seed=int(p.stem.rsplit("__s", 1)[1]),
                    rounds=int(op["rounds_run"]),
                    converged=bool(op.get("converged")),
                    rig_iters=it,
                    polish_iters=pol,
                    total=float(f["objective"]["total_nats"]),
                )
            )
        g = np.asarray(sel["params"]["gamma_hz"])
        out[pool] = dict(
            selected_seed=sel["restarts"]["selected_seed"],
            total_own=float(o["total_nats"]),
            total_measured=float(meas),
            vs_target=float(meas) - TARGET[pool],
            whittle=float(o["whittle_nats"]),
            rig=float(o["rig_neg_log_prior_nats"]),
            ou_own=float(o["ou_neg_log_prior_nats"]),
            ou_measured=float(rp.get("ou_measured", float("nan"))),
            n_cells=int(o["n_cells"]),
            restarts=per,
            sigma_nu=float(sel["params"]["sigma_nu"]),
            gamma_max=float(g.max()),
            shares=shares(sel),
        )
    return out


def ratios_e(tag, meas_lat):
    lat = load(f"{CHECKS[tag]}/latents/latents.json")["fits"]
    out = {}
    for key in POOLS:
        fam, mfam = lat[key]["families"], meas_lat[key]["families"]
        row = {}
        for f in FAMS_E:
            if f not in fam or f not in mfam:
                continue
            s2 = float(mfam[f]["measured_sigma2_db2"])
            r2 = float(mfam[f]["measured_rho_sigma2_db2"])
            l0 = float(fam[f]["ms_plus_post_var_db2"])
            l1 = float(fam[f]["lag1_product_plus_post_cov_db2"])
            row[f] = dict(
                lag0=l0 / s2 if s2 > 0 else None,
                lag1=l1 / r2 if r2 > 0 else None,
                sigma_meas_db=math.sqrt(s2),
            )
        out[key] = row
    return out


def disappearance(tag):
    h = load(f"{CHECKS[tag]}/heldout/heldout.json")["rigs"]
    out = {}
    for rig in ("dregon", "michaels"):
        out[rig] = {}
        for arm in ("real", "v3"):
            gs = h[rig][arm]["intermittency_by_order_group"]
            out[rig][arm] = {
                g: dict(
                    rate=gs[g]["intermittency_boot"]["frac_dropout_visible"][1],
                    boot=gs[g]["intermittency_boot"]["frac_dropout_visible"],
                    n_visible=gs[g]["intermittency"]["n_visible"],
                    appear_disappear=gs[g]["intermittency_boot"]["frac_intermittent"][1],
                )
                for g in GROUPS
            }
    return out


def md_table(text, header_start):
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(header_start):
            rows = []
            for r in lines[i + 2 :]:
                if not r.startswith("|"):
                    break
                rows.append([c.strip() for c in r.strip("|").split("|")])
            return rows
    return []


def tonality(tag):
    txt = Path(f"{CHECKS[tag]}/findings.md").read_text()
    clips = {}
    for r in md_table(txt, "| fit | pattern | arm | audit"):
        clips.setdefault(r[1], {})[r[2]] = r[3]
    expect = {}
    for r in md_table(txt, "| pattern | payload |"):
        if r[1].startswith("v3"):
            expect[r[0]] = r[3]
    return dict(clips=clips, expectation=expect)


def parity(tag):
    out = {}
    for rig in ("dregon", "michaels"):
        a = load(f"{CHECKS[tag]}/parity/arm_{rig}_v3.json")
        b = a["bars"][rig]
        grp = a["gates"]["proxy"]["groups"][PROXY_GROUP[rig]]
        out[rig] = dict(
            mean=b.get("mean_rev_s"),
            upper=b.get("interval_upper_rev_s"),
            ratio=b.get("ratio"),
            bar=b["parity"]["bar_rev_s"],
            margin=b["parity"]["margin_rev_s"],
            within=b["parity"]["within"],
            stretch_within=(b.get("stretch") or {}).get("within"),
            proxy=grp["mean_ltas_abs_db"],
            proxy_gate=grp["gate_db"],
        )
        if rig == "michaels":
            per = {}
            for m in a["measurements"].values():
                per.setdefault(
                    m["support"].get("regime", "?") if isinstance(m["support"], dict) else "?", []
                ).append(m["pit_mae"])
            out[rig]["per_regime"] = {k: float(np.mean(v)) for k, v in per.items()}
    return out


def proxy_seeds(tag):
    out = {}
    d = Path(f"{CHECKS[tag]}/proxy_seeds")
    for rig in ("dregon", "michaels"):
        vals = {}
        for p in sorted(d.glob(f"arm_{rig}_{tag}_s*.json")):
            a = load(p)
            groups = a["gates"]["proxy"]["groups"]
            vals[int(p.stem.rsplit("_s", 1)[1])] = groups[PROXY_GROUP[rig]]["mean_ltas_abs_db"]
        if vals:
            v = np.array(list(vals.values()))
            out[rig] = dict(seeds=vals, mean=float(v.mean()), spread=float(v.max() - v.min()))
    return out


def figviews(tag):
    p = FIGDIR / f"figdata_{tag}.json"
    if not p.exists():
        return None
    d = load(p)
    rigs = {}
    for rnd in ("r2", tag):
        for k, row in d["rigs"][f"v3_{rnd}"].items():
            rigs.setdefault(rnd, {})[k] = dict(
                over_k1_db=row.get("over_k1_db"),
                over_k2_db=row.get("over_k2_db"),
                n_orders_over=row.get("n_orders_over"),
                n_orders=row.get("n_orders_shown"),
                gamma_max_hz=row.get("gamma_max_hz"),
                floor_at_db=row.get("floor_at_db"),
            )
    listen = {}
    for rig, r in d["listen"]["rigs"].items():
        listen[rig] = {arm: c.get("ltas_mean_abs_db") for arm, c in r["clips"].items()}
    ss = {
        k: {f: v["static_share"] for f, v in x["families"].items()}
        for k, x in d["static_shares"].items()
    }
    return dict(rigs=rigs, listen=listen, static_shares=ss)


def main():
    rp_path = Path("results/noise_v3/diag/reprice_r3.json")
    reprice = load(rp_path) if rp_path.exists() else {}
    r2m = load("results/noise_v3/diag/reprice_measured.json")
    reprice.update({k: v for k, v in r2m.items() if k.startswith("r2/")})
    meas_lat = load(f"{CHECKS['r3a']}/latents/latents.json")["fits"]
    res = {}
    for tag in ROUNDS:
        row = dict(objective=objective(tag, reprice))
        if Path(CHECKS[tag], "latents/latents.json").exists():
            row.update(
                e=ratios_e(tag, meas_lat),
                disappearance=disappearance(tag),
                tonality=tonality(tag),
                parity=parity(tag),
                proxy_seeds=proxy_seeds(tag),
            )
        res[tag] = row
    res["views"] = {tag: figviews(tag) for tag in ("r3a", "r3b")}
    Path("results/noise_v3/diag/verdict_r3.json").write_text(
        json.dumps(res, indent=1, default=float)
    )
    pr(res)
    Path("results/noise_v3/diag/verdict_r3.md").write_text(md(res))


def sp(x, nd=1):
    """Signed, thin-space thousands, '—' for None/NaN."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+,.{nd}f}".replace(",", " ").replace("-", "−")


def md(res):
    """The doc's § Round 3 Results tables, r2 / r3a / r3b."""
    have = [t for t in ROUNDS if "e" in res[t]]
    out = []
    pn = {
        "dregon_room2_floor": "DREGON",
        "michaels_fly125_cruise": "cruise",
        "michaels_fly125_standby": "standby",
    }
    out += [
        "**Objective under the measured wander** (nats; r3b re-priced by "
        "`_nv3_reprice.py`, its own-law total in brackets; Δ = total − target):",
        "",
        "| pool | target | r2 (Δ) | r3a (Δ) | r3b (Δ) [own law] | Whittle r2 / r3a / r3b |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for pool, name in pn.items():
        o = {t: res[t]["objective"][pool] for t in ROUNDS}
        out.append(
            f"| {name} | {f1(TARGET[pool])} | {f1(o['r2']['total_measured'])} ({sp(o['r2']['vs_target'])}) | "
            f"{f1(o['r3a']['total_measured'])} ({sp(o['r3a']['vs_target'])}) | "
            f"{f1(o['r3b']['total_measured'])} ({sp(o['r3b']['vs_target'])}) [{f1(o['r3b']['total_own'])}] | "
            f"{f1(o['r2']['whittle'])} / {f1(o['r3a']['whittle'])} / {f1(o['r3b']['whittle'])} |"
        )
    out += [
        "",
        "**Rounds and rig iterations** (per restart s0 / s1 / s2 / s3; rig L-BFGS iterations summed "
        "over the alternation, first pass + restart, then the all-frames polish):",
        "",
        "| pool | round | selected | rounds run | rig iterations | polish | σ_ν rad/s | max γ Hz |",
        "|---|---|---:|---|---|---|---:|---:|",
    ]
    for pool, name in pn.items():
        for t in ROUNDS:
            o = res[t]["objective"][pool]
            rs = o["restarts"]
            out.append(
                f"| {name} | {t} | s{o['selected_seed']} | {' / '.join(str(r['rounds']) for r in rs)} | "
                f"{' / '.join(f'{r["rig_iters"]:,}'.replace(',', ' ') for r in rs)} | "
                f"{' / '.join(str(r['polish_iters']) for r in rs)} | {o['sigma_nu']:.2f} | {o['gamma_max']:.1f} |"
            )
    fams = ("d", "u", "u_j", "v k9-24", "v k25-60", "v k61+")
    out += [
        "",
        "**Static share** of each family's mean square (pool mean per track), r2 → r3a → r3b:",
        "",
        "| pool | "
        + " | ".join(f.replace("k61+", "k ≥ 61").replace("-", "–") for f in fams)
        + " |",
        "|---|" + "---|" * len(fams),
    ]
    for pool, name in pn.items():
        cells = [
            " → ".join(f"{res[t]['objective'][pool]['shares'][f]:.2f}" for t in ROUNDS)
            for f in fams
        ]
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    efams = ("d", "u", "uj", "v k1-2", "v k3-8", "v k9-24", "v k25-60", "v k61+")
    out += [
        "",
        "**(e) Lag-0 sum / measured σ²** (fitted mean square + posterior variance over the "
        "measured σ² of `results/noise_v3/wander/<rig>.json`; r2 against the measured σ too; "
        "lag-1 ratio in brackets), " + " → ".join(have) + ":",
        "",
        "| pool | "
        + " | ".join(
            f.replace("uj", "u_j").replace("k61+", "k ≥ 61").replace("-", "–") for f in efams
        )
        + " |",
        "|---|" + "---|" * len(efams),
    ]
    for key, name in (
        ("dregon", "DREGON"),
        ("michaels_cruise", "cruise"),
        ("michaels_standby", "standby"),
    ):
        cells = []
        for f in efams:
            xs = [res[t]["e"][key].get(f) for t in have]
            if not all(xs) or any(x["lag0"] is None for x in xs):
                cells.append("σ 0")
                continue
            cells.append(
                " → ".join(f"{x['lag0']:.2f}" for x in xs)
                + " ("
                + " → ".join(f"{x['lag1']:.2f}" for x in xs)
                + ")"
            )
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    out += [
        "",
        "**(b) Disappearance rate per order group** (share of a visible line's blocks under "
        "6 dB, window-bootstrap median [5 %, 95 %], visible lines in brackets; a+d = lines that "
        "appear and disappear, %):",
        "",
        "| rig | orders | real | " + " | ".join(have) + " | a+d real / " + " / ".join(have) + " |",
        "|---|---|---|" + "---|" * len(have) + "---|",
    ]
    for rig, rname in (("dregon", "DREGON"), ("michaels", "Michael's")):
        for g in GROUPS:
            real = res["r3a"]["disappearance"][rig]["real"][g]
            cells = [
                f"{pc(real['rate'])} % ({real['n_visible']})"
                if real["n_visible"]
                else "no visible line"
            ]
            ads = [pc(real["appear_disappear"])]
            for t in have:
                x = res[t]["disappearance"][rig]["v3"][g]
                b = x["boot"]
                cells.append(
                    f"{pc(x['rate'])} % [{pc(b[0])}, {pc(b[-1])}] ({x['n_visible']})"
                    if x["n_visible"]
                    else "none"
                )
                ads.append(pc(x["appear_disappear"]))
            out.append(
                f"| {rname} | {g.replace('61+', '≥ 61').replace('-', '–')} | "
                + " | ".join(cells)
                + f" | {' / '.join(ads)} |"
            )
    out += [
        "",
        "**(c) Tonality** (audit counts, orders ≥ 3 / 6 / 10 dB per rotor; clips = median of 4 "
        "render seeds; expectation = `noise_v2_tonality_audit.py --fit`):",
        "",
        "| pattern | real clips | "
        + " | ".join(f"{t} clips" for t in have)
        + " | "
        + " | ".join(f"{t} expectation" for t in have)
        + " |",
        "|---|---|" + "---|" * (2 * len(have)),
    ]
    for pat in res["r3a"]["tonality"]["clips"]:
        real = res["r3a"]["tonality"]["clips"][pat].get("real")
        cl = [res[t]["tonality"]["clips"].get(pat, {}).get("v3") for t in have]
        ex = [res[t]["tonality"]["expectation"].get(pat) for t in have]
        nm = pat.replace("michaels_", "").replace("dregon_cruise_", "DREGON ").replace("_", " ")
        out.append(f"| {nm} | {real} | " + " | ".join(cl) + " | " + " | ".join(ex) + " |")
    out += [
        "",
        "**(d) Parity** (HPPNet PIT MAE, rev/s, frozen supports and seeds) and the LTAS proxy "
        "(`ltas_abs_db`, dB; selected fit, then the mean over the four restarts):",
        "",
        "| rig | "
        + " | ".join(have)
        + " | bar | proxy "
        + " / ".join(have)
        + " | proxy seed mean "
        + " / ".join(have)
        + " | gate |",
        "|---|" + "---|" * len(have) + "---|---|---|---|",
    ]
    for rig, rname in (("dregon", "DREGON"), ("michaels", "Michael's")):
        cells = []
        for t in have:
            p = res[t]["parity"][rig]
            s = f"{p['mean']:.6f}"
            s += (
                f", upper {p['upper']:.6f}"
                if p["upper"] is not None
                else f", ratio {p['ratio']:.3f}"
            )
            s += ", PASS" if p["within"] else ", FAIL"
            s += f" ({sp(p['margin'], 3)})"
            cells.append(s)
        p0 = res["r3a"]["parity"][rig]
        prox = " / ".join(f"{res[t]['parity'][rig]['proxy']:.4f}" for t in have)
        seedm = " / ".join(
            f"{res[t]['proxy_seeds'][rig]['mean']:.4f}" if rig in res[t]["proxy_seeds"] else "—"
            for t in have
        )
        out.append(
            f"| {rname} | "
            + " | ".join(cells)
            + f" | {p0['bar']:.6f} | {prox} | {seedm} | {p0['proxy_gate']:.4f} |"
        )
    reg = " | ".join(
        " / ".join(
            f"{res[t]['parity']['michaels']['per_regime'][k]:.3f}"
            for k in ("standby", "ramp", "cruise")
        )
        for t in have
    )
    out.append(f"| Michael's standby / ramp / cruise | {reg} | real 0.344 / 3.171 / 0.588 | | | |")
    out += [
        "",
        "**Listen** (LTAS distance to the real clip, dB: mean |Δ| over the six bands 100 Hz–7 kHz, "
        "same windows and seed as the explainer's § Listen):",
        "",
        "| rig | v2 | r2 | r3a | r3b |",
        "|---|---:|---:|---:|---:|",
    ]
    va, vb = res["views"]["r3a"], res["views"]["r3b"]
    for rig, rname in (("dregon", "DREGON"), ("michaels", "Michael's")):
        la, lb = va["listen"][rig], vb["listen"][rig]
        out.append(
            f"| {rname} | {la['v2']:.2f} | {la['v3']:.2f} | {la['v3_r3a']:.2f} | {lb['v3_r3b']:.2f} |"
        )
    out += [
        "",
        "**Rig views** (latents at zero; DREGON and cruise at 80 rev/s, standby at 30.6; line over "
        "floor dB at k = 1 / 2; orders > 3 dB of the orders ≤ 8 kHz; floor part dB):",
        "",
        "| pool | round | k = 1 | k = 2 | orders > 3 dB | max γ Hz | floor 100 Hz | 1 kHz | 4 kHz |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k, name in (("dregon", "DREGON"), ("cruise", "cruise"), ("standby", "standby")):
        for t, v in (
            ("r2", va["rigs"]["r2"][k]),
            ("r3a", va["rigs"]["r3a"][k]),
            ("r3b", vb["rigs"]["r3b"][k]),
        ):
            fl = v["floor_at_db"]
            out.append(
                f"| {name} | {t} | {v['over_k1_db']:.1f} | {v['over_k2_db']:.1f} | "
                f"{v['n_orders_over']}/{v['n_orders']} | {v['gamma_max_hz']:.1f} | "
                f"{fl['100']:.1f} | {fl['1000']:.1f} | {fl['4000']:.1f} |".replace("-", "−")
            )
    bias = {}
    for t, p in (
        ("r2", "ltas_bias_r2.json"),
        ("r3a", "ltas_bias_r3a.json"),
        ("r3b", "ltas_bias_r3b.json"),
        ("folded r3", "ltas_bias_folded_r3.json"),
    ):
        q = Path("results/noise_v3/diag") / p
        if q.exists():
            bias[t] = load(q)
    out += [
        "",
        "**Render − fit LTAS** (dB per band 100–300 / 300–700 / 700–1.5k / 1.5–3k / 3–5k / 5–7k Hz; "
        "`noise_v3_diag.py ltas-bias`, render expectation against the fit's own block-mean spectrum):",
        "",
        "| pool | " + " | ".join(bias) + " |",
        "|---|" + "---|" * len(bias),
    ]
    for pool, name in pn.items():
        cells = []
        for b in bias.values():
            r = b.get(pool)
            cells.append(
                " / ".join(f"{x:+.2f}".replace("-", "−") for x in r["render_minus_fit"]["abs_db"])
                if r
                else "—"
            )
        out.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def f1(x, nd=1):
    return (
        "—"
        if x is None or (isinstance(x, float) and math.isnan(x))
        else f"{x:,.{nd}f}".replace(",", " ").replace("-", "−")
    )


def pc(x):
    return "—" if x is None else f"{100 * x:.1f}"


def pr(res):
    print("## objective (under the measured wander) vs target")
    for pool in TARGET:
        cells = []
        for tag in ROUNDS:
            o = res[tag]["objective"][pool]
            rs = o["restarts"]
            cells.append(
                f"{tag}: {f1(o['total_measured'])} ({o['vs_target']:+,.1f}) own {f1(o['total_own'])} "
                f"W {f1(o['whittle'])} rig {f1(o['rig'])} OU {f1(o['ou_own'])}/{f1(o['ou_measured'])} "
                f"s{o['selected_seed']} rounds {[r['rounds'] for r in rs]} rig iters {[r['rig_iters'] for r in rs]} "
                f"polish {[r['polish_iters'] for r in rs]} gmax {o['gamma_max']:.1f} snu {o['sigma_nu']:.2f}"
            )
        print(pool, f"target {f1(TARGET[pool])}")
        for c in cells:
            print("  ", c)
    print("\n## static shares")
    for pool in TARGET:
        for tag in ROUNDS:
            s = res[tag]["objective"][pool]["shares"]
            print(pool, tag, " ".join(f"{k} {v:.2f}" for k, v in s.items()))
    if "e" not in res["r3a"]:
        return
    print("\n## (e) lag-0 / lag-1 vs measured sigma")
    for key in POOLS:
        for f in FAMS_E:
            cs = []
            for tag in ROUNDS:
                x = res[tag].get("e", {}).get(key, {}).get(f)
                cs.append("—" if not x else f"{f1(x['lag0'], 2)} / {f1(x['lag1'], 2)}")
            print(key, f, " | ".join(cs))
    print("\n## disappearance per order group (rate [boot] (visible); a+d %)")
    for rig in ("dregon", "michaels"):
        for g in GROUPS:
            real = res["r3a"]["disappearance"][rig]["real"][g]
            cs = [
                f"real {pc(real['rate'])} ({real['n_visible']}) ad {pc(real['appear_disappear'])}"
            ]
            for tag in ROUNDS:
                x = res[tag].get("disappearance", {}).get(rig, {}).get("v3", {}).get(g)
                if x:
                    b = x["boot"]
                    cs.append(
                        f"{tag} {pc(x['rate'])} [{pc(b[0])}, {pc(b[-1])}] ({x['n_visible']}) "
                        f"ad {pc(x['appear_disappear'])}"
                        if x["n_visible"]
                        else f"{tag} none ad {pc(x['appear_disappear'])}"
                    )
            print(rig, g, " | ".join(cs))
    print("\n## tonality (clips median; expectation)")
    for pat in res["r3a"]["tonality"]["clips"]:
        cs = [f"real {res['r3a']['tonality']['clips'][pat].get('real')}"]
        for tag in ROUNDS:
            t = res[tag].get("tonality")
            if t:
                cs.append(
                    f"{tag} {t['clips'].get(pat, {}).get('v3')} exp {t['expectation'].get(pat)}"
                )
        print(pat, " | ".join(cs))
    print("\n## parity")
    for rig in ("dregon", "michaels"):
        for tag in ROUNDS:
            p = res[tag].get("parity", {}).get(rig)
            ps = res[tag].get("proxy_seeds", {}).get(rig)
            print(rig, tag, p, ps)
    print("\n## views")
    for tag in ("r3a", "r3b"):
        v = res["views"].get(tag)
        if v:
            print(tag, json.dumps(v["rigs"]), json.dumps(v["listen"]))


if __name__ == "__main__":
    main()
