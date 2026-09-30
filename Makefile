PYTHON ?= python

.PHONY: install test lint typecheck verify boundary check example proposal proposal-summary proposal-all clean-proposal docs docs-serve docs-clean

install:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest tests/

lint:
	$(PYTHON) -m ruff check src tests examples

typecheck:
	$(PYTHON) -m mypy

# Fails if the package has grown a dependency on controlled code.
# Part of `check` deliberately: this repository is published, so the scope
# boundary is a build-breaking condition, not a lint.
boundary:
	$(PYTHON) tools/check_boundary.py

verify:
	$(PYTHON) -m aether.verification --output results

check: boundary lint typecheck test verify

example:
	$(PYTHON) -m examples.artemis1.entry

# Documentation targets
docs:
	$(PYTHON) -m sphinx -b html docs docs/_build/html

docs-serve:
	$(PYTHON) -m sphinx -b livehtml docs docs/_build/html

docs-clean:
	rm -rf docs/_build

# ---------------------------------------------------------------- proposal
# Two deliverables, both LuaLaTeX + Biber via latexmk:
#   proposal          -> main.pdf          full copy, with the dynamics appendix
#   proposal-summary  -> main-summary.pdf  body-only teaser for cold emails
# The summary predefines \summarymode, which main.tex uses to drop the
# appendix and blank the parenthetical pointers into it, so no reference
# dangles. References are kept in both (the body's inline citations need them).
PROPOSAL_DIR := manuscript/proposal

proposal:
	cd $(PROPOSAL_DIR) && latexmk -pdf -lualatex main.tex

proposal-summary:
	cd $(PROPOSAL_DIR) && latexmk -pdf -lualatex -jobname=main-summary \
	  -usepretex="\def\summarymode{}" main.tex

proposal-all: proposal proposal-summary

clean-proposal:
	cd $(PROPOSAL_DIR) && latexmk -c main.tex && \
	  latexmk -c -jobname=main-summary main.tex
