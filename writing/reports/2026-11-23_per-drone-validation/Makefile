ROOT := $(abspath ../../..)
.PHONY: all
all:
	PYTHONPATH=$(ROOT)/src $(ROOT)/.venv/bin/python prepare.py
	quarto render index.qmd --to html
