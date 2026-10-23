"""Is the planted rig reproducible, and does the planted .fit.json reload to its own objective?"""

import json
import sys

sys.path.insert(0, "scripts")
import noise_v4_bench as NB  # noqa: E402
from experiments.noise_model import bench_v4 as B  # noqa: E402

R = "results/noise_v4/bench/planted/dregon__kappa0__r5__s2__c6__kappa0__r1"
rep = json.load(open(R + ".json"))
fit = json.load(open(R + ".fit.json"))
for trial in (1, 2):
    truth, rig, _ = NB._planted(rep["truth"], floor_below=None, render_seed=int(rep["render_seed"]), kappa=0.0, device="cuda")
    p0 = [float(s.power.sum()) for s in rig.settings[:3]]
    m = B.RigModel(rig, device="cuda", seed=int(fit["optimiser"]["seed"]), kappa=0.0)
    m.load_payload(fit)
    f, _ = m.objective()
    mt = B.RigModel(rig, device="cuda", seed=0, kappa=0.0)
    mt.load_payload(json.load(open(rep["truth"])))
    ft, _ = mt.objective()
    print(f"trial {trial}: power checksums {p0}  fit objective {f:.1f} (recorded {fit['objective']['total_nats']:.1f})  truth objective {ft:.1f}", flush=True)
