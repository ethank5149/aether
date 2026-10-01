PYTHON ?= python

.PHONY: install test lint typecheck verify derivations boundary check example proposal proposal-summary proposal-all proposal-dist clean-proposal docs docs-serve docs-clean

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

# Reduce every derivation the manuscript relies on to zero, symbolically, and
# exit non-zero if one does not. Seconds, not minutes, so it can be run after
# any edit to an equation. `verify` runs the same checks as task S1, together
# with the numerics generated from them; this is the quick form.
derivations:
	$(PYTHON) -m aether.symbolic

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
#   proposal          -> main.pdf          full copy, with the three appendices
#   proposal-summary  -> main-summary.pdf  body-only teaser for cold emails
# The summary predefines \summarymode, which main.tex uses to drop the
# appendices and blank the parenthetical pointers into them, so no reference
# dangles. References are kept in both (the body's inline citations need them).
#
#   proposal-dist     -> dist/<name>.pdf   the two PDFs under names fit to attach
# `main-summary.pdf` is a build name. An attachment is read by its filename
# before it is opened, so the copies that are sent are named for what they are.
PROPOSAL_DIR := manuscript/proposal
PROPOSAL_NAME ?= Knox-PhD-proposal

proposal:
	cd $(PROPOSAL_DIR) && latexmk -pdf -lualatex main.tex

proposal-summary:
	cd $(PROPOSAL_DIR) && latexmk -pdf -lualatex -jobname=main-summary \
	  -usepretex="\def\summarymode{}" main.tex

proposal-all: proposal proposal-summary

proposal-dist: proposal-all
	mkdir -p $(PROPOSAL_DIR)/dist
	cp $(PROPOSAL_DIR)/main.pdf $(PROPOSAL_DIR)/dist/$(PROPOSAL_NAME).pdf
	cp $(PROPOSAL_DIR)/main-summary.pdf $(PROPOSAL_DIR)/dist/$(PROPOSAL_NAME)-summary.pdf

clean-proposal:
	cd $(PROPOSAL_DIR) && latexmk -c main.tex && \
	  latexmk -c -jobname=main-summary main.tex
