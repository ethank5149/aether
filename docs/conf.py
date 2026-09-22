"""Sphinx configuration for AETHER documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

from aether import __version__

project = "AETHER"
author = "Ethan Knox"
copyright = "2026, Ethan Knox"
release = __version__
version = __version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_design",
]

html_theme = "furo"
html_static_path = ["_static"]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

autodoc_mock_imports = [
    "gmsh",
    "cantera",
    "pymsis",
    "xarray",
    "h5py",
    "cupy",
    "flint",
    "rasterio",
    "matplotlib",
    "tqdm",
    "dymos",
    "openmdao",
]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

mathjax4_config = "TeX-AMS-MML_HTMLorMML"
mathjax4_options = {"chtml": True}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

autodoc_typehints = "description"

autosummary_generate = True

nitpicky = False