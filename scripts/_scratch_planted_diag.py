"""Scratch (to be removed): why does the planted fit's state score differently
when reloaded? Same rig; model A = load .fit.json then the saved state;
model B = load the truth payload (as cmd_planted's fit_rig did) then the saved
state. Compare objectives, per-recording parts and every non-flat attribute."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "scripts")
import noise_v4_bench as NB  # noqa: E402

from experiments.noise_model import bench_eta as E  # noqa: E402
from experiments.noise_model import bench_v4 as B  # noqa: E402

R = Path("results/noise_v4/bench/planted/dregon__kappa0__r5__s2__c6__kappa0__r1")
rep = json.loads(R.with_suffix(".json").read_text())
fit = json.loads(R.with_suffix(".fit.json").read_text())
truth_pay = json.loads(Path(rep["truth"]).read_text())
_, rig, _ = NB._planted(
    rep["truth"],
    floor_below=None,
    render_seed=int(rep["render_seed"]),
    kappa=float(rep["kappa"]),
    device="cuda",
)
seed = int(fit["optimiser"]["seed"])
out = {}
models = {}
m0 = B.RigModel(rig, device="cuda", seed=seed, kappa=float(rep["kappa"]))
m0.load_payload(fit)
x = E.get_flat(m0)
# the fit's own path: fit_rig(start=truth, resume=True) built RigModel and loaded the
# truth, then optimised; and the same through fit_rig with zero passes
_, rec = B.fit_rig(
    rig,
    seed=seed,
    kappa=float(rep["kappa"]),
    device="cuda",
    start=truth_pay,
    resume=True,
    deadline_s=1.0,
)
print("fit_rig(start=truth, resume) objective at the truth state:", rec["objective"], flush=True)
for tag, pay in (("A_fitjson", fit), ("B_truth", truth_pay)):
    m = B.RigModel(rig, device="cuda", seed=seed, kappa=float(rep["kappa"]))
    m.load_payload(pay)
    E.set_flat(m, x.to(m.params[next(iter(m.params))].device))
    f, parts = m.objective()
    out[tag] = dict(objective=float(f), parts=[float(p) for p in parts])
    models[tag] = m
    print(tag, f"{f:.1f}", flush=True)
a, b = models["A_fitjson"], models["B_truth"]
print(
    "per-recording parts diff (A - B):",
    np.round(np.array(out["A_fitjson"]["parts"]) - np.array(out["B_truth"]["parts"]), 1).tolist(),
)
for k in a.params:
    d = float((a.params[k] - b.params[k]).abs().max())
    if d > 0:
        print("param differs:", k, d)
free = {n for n, _ in E.free_params(a)}
print("params not in the flat vector:", sorted(set(a.params) - free))
for name in dir(a):
    if name.startswith("__"):
        continue
    va, vb = getattr(a, name, None), getattr(b, name, None)
    if isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor) and va.shape == vb.shape:
        d = float((va.double() - vb.double()).abs().max())
        if d > 0:
            print("attr differs:", name, d)
for j, (fa, fb) in enumerate(zip(a.frozen, b.frozen, strict=True)):
    for name in vars(fa):
        va, vb = getattr(fa, name), getattr(fb, name)
        if isinstance(va, torch.Tensor) and va.shape == vb.shape:
            d = float((va.double() - vb.double()).abs().max())
            if d > 0:
                print(f"frozen[{j}].{name} differs: {d}")
Path("results/noise_v4/bench/planted/_diag.json").write_text(json.dumps(out, indent=1))
