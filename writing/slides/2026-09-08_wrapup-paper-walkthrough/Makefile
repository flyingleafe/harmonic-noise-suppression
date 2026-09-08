ROOT := $(shell git rev-parse --show-toplevel)

all: assets slides.html slides.pdf

assets:
	python prepare.py

slides.html: slides.qmd styles.scss
	quarto render slides.qmd --to revealjs

slides.pdf: slides.qmd
	quarto render slides.qmd --to beamer --pdf-engine tectonic -M output-file:slides.pdf

check: slides.pdf
	mkdir -p check && rm -f check/page-*.png
	pdftoppm -png -r 80 $< check/page

.PHONY: all assets check
