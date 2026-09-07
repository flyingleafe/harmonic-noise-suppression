# Writing — papers, reports, slides

| Kind | Source | Where |
|---|---|---|
| Papers | LaTeX (`journal.tex`, `main.tex`; tectonic) | `writing/papers/<slug>/` — see `writing/papers/AGENTS.md` |
| **New** reports, explainers, slide decks | **Quarto** (`.qmd`; user-wide `quarto` skill) | `writing/reports/<date>_<title>/`, `writing/slides/<date>_<title>/` |
| Existing Typst reports/slides (≤ 2026-08) | Typst (Touying for slides) | same directories; still build with their `Makefile` |

`<date>` is `YYYY-MM-DD`, `<title>` a kebab-case slug: `2026-06-02_rps-progress/`.

## Quarto artifacts (default for anything new)

- One directory per artifact: `slides.qmd` (or `index.qmd`), a `styles.scss`
  if the deck needs one, a `prepare.py` that produces every figure/table into
  `assets/` (gitignored PNG/PDF, regenerated), and a `Makefile` target that
  renders. Rendered HTML with `embed-resources: true` is several MB — keep it
  out of git unless it is the deliverable of record; `make slides.html`
  rebuilds it. Interactive explainers use `{ojs}` cells fed by `ojs_define()`
  or a versioned JSON in `assets/`.
- Python execution: `QUARTO_PYTHON=$PWD/.venv/bin/python` (jupyter is in the
  venv); heavy analysis belongs in `prepare.py`, not in executed cells.
- Verify by rendering and opening the output in a browser (revealjs: check
  navigation and overflow on every slide), not by reading the source.
- Figures come from `eval.py` + `src/plots` (publication comparisons) or
  `plots.dwym` (quick looks); `prepare.py` only arranges them.

## Typst artifacts (legacy, keep buildable)

Templates: `writing/templates/typst/{report,slides}.typ` — import with a
root-absolute path (`#import "/writing/templates/typst/report.typ": ...`) and
compile with `typst compile --root $(git rev-parse --show-toplevel)`. Each
directory's `Makefile` has `all` / `figures` / `watch` / `check` (`pdftoppm`
page PNGs for visual review). Do not add per-artifact template copies.

## Principles

1. **Self-contained.** Every artifact owns its `prepare.py`; duplication beats
   cross-artifact coupling.
2. **Visually verified.** Nothing is done until the rendered pages/slides were
   looked at.
3. **Honest inventory first.** A deck or report starts from what actually
   happened since the last artifact (`docs/experiments/`, git log, results),
   not from a narrative; `writing/slides/NEXT-DECK-experiment-inventory.md`
   is the running inventory for the next supervisor deck.
