#!/usr/bin/env python
"""Assets for the wrap-up paper walkthrough deck, taken from the paper itself.

Figures: the paper's ``figures/*.png`` (already 300 dpi) are copied; Fig. 1
(the TikZ architecture) is compiled standalone with tectonic and rasterized.
Tables: the generated ``src/tables/*.tex`` bodies are converted to Markdown
and written to ``assets/*.md`` for ``{{< include >}}``. The slide-3 table is
the headline table with the L2 rows of the adaptation table appended.

    python prepare.py
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PAPER = ROOT / "writing/papers/2026-08_wrapup"
ASSETS = HERE / "assets"

FIGURES = ["ladder_r1_r3", "freq_probe_panels", "qual_overlays"]


def tex_cell(c: str) -> str:
    c = c.strip()
    c = re.sub(r"\\multirow\{\d+\}\{\*\}\{(.*?)\}$", r"\1", c)
    c = re.sub(r"\\textbf\{(.*?)\}", r"**\1**", c)
    c = c.replace(r"$\sim$", "~").replace(r"$<$", "<").replace(r"$^\dagger$", "†")
    c = c.replace(r"$\to$", "→").replace("\\", "")
    return c


def tex_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    """Header and body rows of a booktabs ``tabular`` body; rules become empty rows."""
    header: list[str] = []
    rows: list[list[str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("\\begin", "\\end", "\\toprule", "\\bottomrule")):
            continue
        if line == "\\midrule":
            continue
        cells = [tex_cell(c) for c in line.rstrip("\\").split("&")]
        if not header:
            header = cells
        else:
            rows.append(cells)
    return header, rows


#: PIT MAE (rev/s) buckets -> colour; lower is better.
BUCKETS = ((3.0, "#1a7f37"), (6.0, "#8a7a00"), (15.0, "#c05a00"), (float("inf"), "#b02020"))


def colour(cell: str, fmt: str) -> str:
    m = re.fullmatch(r"(\*\*)?(\d+(?:\.\d+)?)(\*\*)?", cell)
    if not m:
        return cell
    v = float(m.group(2))
    hexc = next(c for lim, c in BUCKETS if v < lim)
    if fmt == "html":
        return f'<span style="color:{hexc};font-weight:600">{cell}</span>'
    text = f"\\textbf{{{m.group(2)}}}" if m.group(1) else m.group(2)
    return f"\\textcolor[HTML]{{{hexc[1:]}}}{{{text}}}"


def markdown(header, rows, drop=frozenset(), fmt="html") -> str:
    keep = [i for i in range(len(header)) if i not in drop]
    align = ["|:--" if i < 2 else "|--:" for i in keep]
    out = ["|" + "|".join(header[i] for i in keep) + "|", "".join(align) + "|"]
    for r in rows:
        out.append("|" + "|".join(colour(r[i], fmt) if i < len(r) else "" for i in keep) + "|")
    return "\n".join(out) + "\n"


def write_both(name: str, header, rows, drop=frozenset()) -> None:
    (ASSETS / f"{name}.md").write_text(markdown(header, rows, drop, "html"))
    (ASSETS / f"{name}_tex.md").write_text(markdown(header, rows, drop, "tex"))


def architecture_png() -> None:
    src = (PAPER / "src/journal.tex").read_text()
    tikz = re.search(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", src, re.S).group(0)
    doc = (
        "\\documentclass[border=4pt]{standalone}\n\\usepackage{tikz}\n"
        "\\usetikzlibrary{positioning,arrows.meta,fit,calc}\n\\usepackage{amsmath}\n"
        "\\begin{document}\n" + tikz + "\n\\end{document}\n"
    )
    with tempfile.TemporaryDirectory() as td:
        tex = Path(td) / "arch.tex"
        tex.write_text(doc)
        subprocess.run(["tectonic", str(tex)], check=True, capture_output=True)
        subprocess.run(
            [
                "pdftoppm",
                "-png",
                "-r",
                "400",
                "-singlefile",
                str(Path(td) / "arch.pdf"),
                str(ASSETS / "arch"),
            ],
            check=True,
        )


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    for f in FIGURES:
        shutil.copy(PAPER / "figures" / f"{f}.png", ASSETS / f"{f}.png")
    architecture_png()

    h, rows = tex_rows(PAPER / "src/tables/headline_r3.tex")
    _, adapt = tex_rows(PAPER / "src/tables/adaptation.tex")
    l2 = []
    model = ""
    for r in adapt:
        model = r[0] or model
        if r[1] == "L2":
            l2.append(["Multi-pitch L2, R4 (warm-up)" if not l2 else "", model] + r[2:8] + [""])
    # slide variant: without the greedy single-pitch trackers and the R4 regressors
    compact, skip = [], False
    for r in rows:
        if r[0]:
            skip = r[0].startswith(("Classical (greedy", "Regression, R4"))
        if not skip:
            compact.append(r)
    write_both("headline_slide", h, compact + l2, drop={8})
    (ASSETS / "adaptation.md").write_text(
        markdown(*tex_rows(PAPER / "src/tables/adaptation.tex"), drop={8})
    )
    write_both("crossval", *tex_rows(PAPER / "src/tables/crossval.tex"))
    print("wrote", sorted(p.name for p in ASSETS.iterdir()))


if __name__ == "__main__":
    main()
