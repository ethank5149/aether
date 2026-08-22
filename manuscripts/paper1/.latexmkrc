# Read by every latexmk invocation from this directory, so the editor's build
# button and ../build.sh agree without either needing extra flags.

# Intermediates (aux, log, fls, fdb, toc, out, bbl, synctex) go to build/;
# only main.pdf is written beside the source, which is the file the repo
# tracks. One ignore rule then covers every artifact: manuscripts/**/build/.
$aux_dir = 'build';
$out_dir = '.';

# The shared bibliography lives one level up, but bibtex runs from whichever
# directory holds the .aux -- build/, not here. An absolute path resolved when
# this file is read is correct regardless of where bibtex is invoked from.
use Cwd ();
$ENV{BIBINPUTS} = Cwd::abs_path('..') . ':' . ($ENV{BIBINPUTS} // '');
