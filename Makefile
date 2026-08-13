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

# ------------------------------------------------------------------ paper
manuscript:
	$(MAKE) -C manuscript

clean-manuscript:
	$(MAKE) -C manuscript clean
