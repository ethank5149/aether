#!/usr/bin/env bash
# Build the body-only summary copy of the proposal -> main-summary.pdf
#
# The full copy (main.pdf) is built by the VS Code LaTeX Workshop extension as
# usual. This script produces the cold-email summary alongside it, without
# touching the main build: it predefines \summarymode, which main.tex uses to
# drop the appendix, neutralise the appendix cross-references, and compress the
# reference list (maxbibnames=3).
#
# Usage:  ./build-summary.sh        (run from anywhere; it cd's to its own dir)
set -euo pipefail
cd "$(dirname "$0")"

latexmk -pdf -lualatex -jobname=main-summary \
        -usepretex='\def\summarymode{}' main.tex
