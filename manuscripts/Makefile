# Build the three-paper program.
#
#   make            all three papers
#   make paper1     one paper (likewise paper2, paper3)
#   make watch1     continuous rebuild of paper 1 (likewise watch2, watch3)
#   make clean      remove build artifacts everywhere
#
# Each paper is built from inside its own directory so that the relative
# paths to ../preamble.tex and ../shared.bib resolve for both latexmk and
# bibtex. latexmk handles the bibtex and rerun logic; `make slow` under an
# individual paper is not provided, because latexmk is a hard dependency
# of this layout anyway.

PAPERS := paper1 paper2 paper3
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all $(PAPERS) watch1 watch2 watch3 clean

all: $(PAPERS)

$(PAPERS):
	cd $@ && $(LATEXMK) main.tex


watch1:
	cd paper1 && latexmk -pdf -pvc -interaction=nonstopmode main.tex

watch2:
	cd paper2 && latexmk -pdf -pvc -interaction=nonstopmode main.tex

watch3:
	cd paper3 && latexmk -pdf -pvc -interaction=nonstopmode main.tex

clean:
	for d in $(PAPERS); do (cd $$d && latexmk -C && rm -f ./*.bbl ./*.blg); done
