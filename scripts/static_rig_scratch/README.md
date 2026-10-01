# static_rig_scratch — exploration scripts behind round 3 of the campaign

Throwaway analyses kept for provenance (`docs/experiments/static-rig-profiles.md`
§ Round 3). They read the audio caches written by `scripts/static_rig.py fetch`
(`/tmp/sr_avq`, `/tmp/sr_dregon`: `units.json` + `audio/*.npy`) and the joint
search outputs (`results/static_rig/joint/`); run with `PYTHONPATH=src`.
Durable logic lives in `src/experiments/static_rig/{joint_speeds,single_rotor}.py`.

| Script | What it produced |
|---|---|
| `avq_amp.py` | AVQ per-window band vs demod line powers, envelope/parametric rebuilds, Welch overlays |
| `avq_width.py` | in-window line widths vs order (AVQ D rotor) |
| `avq_high*.py` | full-band scans: D k42 per window, prominent lines 1.5–6.5 kHz |
| `avq_refine.py`, `avq_pct.py`, `avq_profplot.py` | block-refined tracks (0.5 s), line/noise per order to 10 kHz, profile figure |
| `avq_geom.py` | first inter-mic delay / localisation attempt (superseded) |
| `dregon_simple.py`, `dregon_core.py` | DREGON singles: strict span, wind mics, centroid/core-peak speeds, rms widths |
| `dregon_wander.py` | shaft-phase drift r(τ) per recording (the TABLE in `notebooks/single_rotor_lab.py`) |
| `dregon_pedestal.py` | pedestal level vs offset and order (the AM/FM separation) |
| `fit_D.py`, `fit_D2.py` | D by line-shape fit and by core HWHM |
