PYTHON ?= python

.PHONY: install test lint typecheck verify boundary check manuscript clean-manuscript

install:
	$(PYTHON) -m pip install -e .[dev]

test:
	$(PYTHON) -m pytest tests/

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy

verify:
	$(PYTHON) -m aether.verification --output results

# Fails if the public kernel has grown a dependency on controlled code.
# Part of `check` deliberately: this repository is published, so the scope
# boundary is a build-breaking condition, not a lint.
boundary:
	$(PYTHON) tools/check_boundary.py

check: boundary lint typecheck test verify

# ---------------------------------------------------------------- papers
# Three papers plus the working notes; see manuscripts/Makefile.
manuscript:
	$(MAKE) -C manuscripts

clean-manuscript:
	$(MAKE) -C manuscripts clean
