# -*- coding: utf-8 -*-
#
# Ministry of Random Walks documentation build configuration file, created by
# sphinx-quickstart on Mon Jun 17 13:52:34 2019.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.imgmath",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "contents"

# General information about the project.
project = u"Ministry of Random Walks"
copyright = u"2019, Lennart Schüler"
author = u"Lennart Schüler"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = u"0.0.1"
# The full version, including alpha/beta/rc tags.
release = u"0.0.1"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_html_theme_options = {
    #    'canonical_url': '',
    #    'analytics_id': '',
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "top",
    #    'style_external_links': False,
    #    'vcs_pageview_mode': '',
    # Toc options
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "MinistryofRandomWalksdoc"


# -- Options for LaTeX output ---------------------------------------------
latex_logo = "pics/walks_150.png"

# latex_show_urls = 'footnote'
# http://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-latex-output
latex_elements = {
    "preamble": r"""
\setcounter{secnumdepth}{1}
\setcounter{tocdepth}{2}
\pagestyle{fancy}
""",
    "pointsize": "10pt",
    "papersize": "a4paper",
    "fncychap": "\\usepackage[Glenn]{fncychap}",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        "MinistryofRandomWalks.tex",
        u"Ministry of Random Walks Documentation",
        u"Lennart Schüler",
        "manual",
    )
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (
        master_doc,
        "ministryofrandomwalks",
        u"Ministry of Random Walks Documentation",
        [author],
        1,
    )
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "MinistryofRandomWalks",
        u"Ministry of Random Walks Documentation",
        author,
        "MinistryofRandomWalks",
        "One line description of project.",
        "Miscellaneous",
    )
]

intersphinx_mapping = {
    "Python 3.7": ("https://docs.python.org/3.6", None),
    "Python": ("https://docs.python.org/", None),
    "NumPy": ("http://docs.scipy.org/doc/numpy/", None),
    "SciPy": ("http://docs.scipy.org/doc/scipy/reference", None),
    "matplotlib": ("http://matplotlib.org", None),
    "Sphinx": ("http://www.sphinx-doc.org/en/stable/", None),
}

suppress_warnings = [
    "image.nonlocal_uri",
    #    'app.add_directive',  # this evtl. suppresses the numpydoc induced warning
]
