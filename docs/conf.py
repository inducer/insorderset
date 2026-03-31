# Configuration file for the Sphinx documentation builder.

import sys
from pathlib import Path

# Make the package importable from the source tree when building docs locally
# (not needed in CI where the package is installed via pip).
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# -- Project information -----------------------------------------------------

project = "insorderset"
author = "insorderset contributors"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

# Autodoc: show members in source order; include type hints in signatures.
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

suppress_warnings = ["intersphinx.fetch_warning"]

# -- Options for HTML output -------------------------------------------------

html_theme = "furo"
html_title = "insorderset"
