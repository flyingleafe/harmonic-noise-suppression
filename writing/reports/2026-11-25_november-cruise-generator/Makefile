ROOT := ../../..
PYTHON := $(ROOT)/.venv/bin/python
RESULTS := $(ROOT)/results/november_cruise_generator

.PHONY: all prepare
all: prepare
	quarto render index.qmd --to html

prepare:
	MPLBACKEND=Agg $(PYTHON) prepare.py --results $(RESULTS)
