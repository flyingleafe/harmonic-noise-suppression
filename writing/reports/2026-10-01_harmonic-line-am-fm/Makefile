all: index.html

assets/pedestal.png: prepare.py
	../../../.venv/bin/python prepare.py

index.html: index.qmd assets/pedestal.png
	quarto render index.qmd --to html

.PHONY: all
