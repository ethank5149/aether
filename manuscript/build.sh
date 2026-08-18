#!/bin/sh
# Build the manuscript. `make` is not installed in every environment this
# repository is used from, so this script is the portable entry point and
# the Makefile is a convenience where make exists.
set -e
cd "$(dirname "$0")"
case "${1:-}" in
  clean) latexmk -C; rm -f ./*.bbl ./*.blg; echo "cleaned" ;;
  watch) latexmk -pdf -pvc -interaction=nonstopmode main.tex ;;
  *)     latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
         echo "built main.pdf" ;;
esac
