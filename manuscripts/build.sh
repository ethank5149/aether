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

# latexmk reports a remembered failure as "gave an error in previous
# invocation" and says nothing about the cause, so on failure pull the real
# error out of the log rather than leaving it to be dug up by hand.
build_one() {
  dir=$1; tex=$2
  printf '=== %s ===\n' "$dir"
  # Remove stale dependency cache so latexmk re-checks all inputs.
  rm -f "$root/$dir/build/$(basename "$tex" .tex).fdb_latexmk"
  if (cd "$root/$dir" && latexmk -pdf -interaction=nonstopmode -halt-on-error "$tex"); then
    return 0
  fi
  log="$root/$dir/$(basename "$tex" .tex).log"
  if [ -f "$log" ]; then
    printf '\n--- %s: first errors in %s ---\n' "$dir" "$log"
    grep -n -E '^!|^l\.[0-9]+|\.tex:[0-9]+:' "$log" | head -8
    printf -- '--- if you have already fixed this, the log is stale: ./build.sh clean ---\n'
  fi
  return 1
}

case "${1:-}" in
  clean)
    for d in paper1 paper2 paper3; do
      (cd "$root/$d" && latexmk -C)
      rm -rf "$root/$d"/build
      rm -f "$root/$d"/*.aux "$root/$d"/*.bbl "$root/$d"/*.blg "$root/$d"/*.log "$root/$d"/*.out "$root/$d"/*.toc "$root/$d"/*.fls "$root/$d"/*.fdb_latexmk "$root/$d"/*.synctex.gz "$root/$d"/*.tdo "$root/$d"/*.pdf
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
