"""Assemble only previously fitted and scored outputs; do not train at render time."""

from pathlib import Path

from plots.per_drone_flights import export, tables

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / "results/rps_traj/per_drone"

if __name__ == "__main__":
    payload = export(ROOT, HERE / "assets")
    tables(ROOT, HERE / "assets", payload)
