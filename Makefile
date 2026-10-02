PYTHON ?= python

.PHONY: install test lint typecheck verify derivations boundary check example proposal proposal-precis proposal-figures proposal-all proposal-dist clean-proposal docs docs-serve docs-clean

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
# Two documents, both LuaLaTeX + Biber via latexmk:
#   proposal          -> main.pdf          the proposal
#   proposal-precis   -> precis.pdf        two pages, for a first email
# The precis is its own source, precis.tex, and links to the proposal.
#
#   proposal-figures  -> figures/*.pdf     regenerated from the example
# The figures are tracked beside the sources, so the targets above need no
# Python. Both are illustrations. Run this one when the example changes.
#
#   proposal-dist     -> dist/<name>.pdf   the PDFs under names fit to attach
# An attachment is read by its filename before it is opened, so the copies
# that are sent are named for what they are.
PROPOSAL_DIR := manuscript/proposal
PROPOSAL_NAME ?= Knox-PhD-proposal

proposal:
	cd $(PROPOSAL_DIR) && latexmk -pdf -lualatex main.tex

proposal-precis:
	cd $(PROPOSAL_DIR) && latexmk -pdf -lualatex precis.tex

proposal-figures:
	$(PYTHON) tools/gen_proposal_figures.py

proposal-all: proposal proposal-precis

proposal-dist: proposal-all
	mkdir -p $(PROPOSAL_DIR)/dist
	cp $(PROPOSAL_DIR)/main.pdf $(PROPOSAL_DIR)/dist/$(PROPOSAL_NAME).pdf
	cp $(PROPOSAL_DIR)/precis.pdf $(PROPOSAL_DIR)/dist/$(PROPOSAL_NAME)-precis.pdf

clean-proposal:
	cd $(PROPOSAL_DIR) && latexmk -c main.tex && latexmk -c precis.tex
