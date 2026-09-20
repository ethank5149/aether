PYTHON ?= python

.PHONY: install test lint typecheck verify boundary check example proposal clean-proposal

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

# ---------------------------------------------------------------- proposal
proposal:
	cd manuscript/proposal && latexmk -pdf main.tex

clean-proposal:
	cd manuscript/proposal && latexmk -c main.tex
