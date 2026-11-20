ROOT := ../../..
PYTHON := $(ROOT)/.venv/bin/python
EVIDENCE := $(ROOT)/results/noise_generator_paper/neural_extension_evidence.json
BRAVE := $(shell command -v brave)

.PHONY: all prepare
all: prepare
	QUARTO_CHROMIUM="$(BRAVE)" quarto render index.qmd --to html

prepare: $(EVIDENCE)
	MPLBACKEND=Agg $(PYTHON) prepare.py
