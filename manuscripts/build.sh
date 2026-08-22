#!/bin/sh
# Build the three-paper program. `make` is not installed in every
# environment this repository is used from, so this script is the portable
# entry point and the Makefile is a convenience where make exists.
#
#   ./build.sh              all three papers
#   ./build.sh paper2       one paper
#   ./build.sh watch paper1 continuous rebuild
#   ./build.sh clean        remove build artifacts everywhere
#
# Papers are built from inside their own directory so the relative paths
# to ../preamble.tex and ../shared.bib resolve for bibtex as well as for
# latexmk.
set -e
cd "$(dirname "$0")"
root=$(pwd)

build_one() {
  dir=$1; tex=$2
  printf '=== %s ===\n' "$dir"
  (cd "$root/$dir" && latexmk -pdf -interaction=nonstopmode -halt-on-error "$tex")
}

case "${1:-}" in
  clean)
    for d in paper1 paper2 paper3; do
      (cd "$root/$d" && latexmk -C >/dev/null 2>&1 || true; rm -f ./*.bbl ./*.blg)
    done
    echo "cleaned"
    ;;
  watch)
    d=${2:-paper1}
    (cd "$root/$d" && latexmk -pdf -pvc -interaction=nonstopmode main.tex)
    ;;
  paper1|paper2|paper3) build_one "$1" main.tex ;;
  *)
    build_one paper1 main.tex
    build_one paper2 main.tex
    build_one paper3 main.tex
    echo "built paper1/main.pdf, paper2/main.pdf, paper3/main.pdf"
    ;;
esac
