# Build the manuscript. `make` produces main.pdf with references resolved.
# latexmk handles the bibtex and rerun logic; the explicit sequence is kept
# below as `make slow` for environments without latexmk.

.PHONY: all slow clean watch
all:
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

slow:
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode main.tex

clean:
	latexmk -C
	rm -f *.bbl *.blg *.aux *.log *.out *.toc
