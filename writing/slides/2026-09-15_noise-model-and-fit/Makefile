ROOT := $(shell git rev-parse --show-toplevel)
PY := PYTHONPATH="$(ROOT)/src:$(ROOT)/scripts" $(ROOT)/.venv/bin/python
QUARTO := QUARTO_PYTHON="$(ROOT)/.venv/bin/python" quarto

all: slides.html slides.pdf

assets:
	cd $(ROOT) && $(PY) writing/slides/2026-09-15_noise-model-and-fit/prepare.py

slides.html: slides.qmd styles.scss assets
	$(QUARTO) render slides.qmd --to revealjs

slides.pdf: slides.qmd assets
	$(QUARTO) render slides.qmd --to beamer

check: slides.pdf
	mkdir -p check && rm -f check/page-*.png
	pdftoppm -png -r 80 slides.pdf check/page

.PHONY: all assets check
