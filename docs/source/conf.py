# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# Copyright xmuspeech (Author: Leo 2024-01)

from __future__ import annotations

import os
import sys

import lightning  # autodoc import member need this

_PATH_HERE = os.path.abspath(os.path.dirname(__file__))
_PATH_ROOT = os.path.join(_PATH_HERE, "..", "..")
sys.path.insert(0, _PATH_ROOT)

try:
    import egrecho
except Exception:
    raise ImportError

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "egrecho"
copyright = "2024, Dexin Liao"
author = "Dexin Liao"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "myst_parser",
    # "sphinx.ext.autosummary",
    # "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    # "sphinx_paramlinks",  #  "¶" for params
    "sphinx.ext.linkcode",
    "sphinx.ext.githubpages",
    # "sphinxext-opengraph", # url meta tag
    "sphinx_inline_tabs",
    "sphinx_togglebutton",
]

napoleon_use_admonition_for_examples = False
# maximum_signature_line_length = 80
copybutton_prompt_text = ">>> "
copybutton_prompt_text1 = "... "
copybutton_exclude = ".linenos"

# templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    # "python": ("https://docs.python.org/3", None),
    # "torch": ("https://pytorch.org/docs/stable/", None),
}

# The master toctree document.
master_doc = "index"

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
    # ".ipynb": "nbsphinx",
}

# -- Options for myst-------------------------------------------------
# myst options
myst_disable_syntax = [
    "colon_fence",
    "myst_block_break",
    "myst_line_comment",
    "math_block",
]
myst_enable_extensions = [
    # "amsmath",
    # "attrs_inline",
    "deflist",
    # "dollarmath",
    # "fieldlist",
    # "html_admonition",
    # "html_image",
    # "replacements",
    # "strikethrough",
    # "substitution",
    # "tasklist",
]

# Disable the conversion of dashes so that long options like "--find-links" won't
# render as "-find-links" if included in the text.The default of "qDe" converts normal
# quote characters ('"' and "'"), en and em dashes ("--" and "---"), and ellipses "..."
smartquotes_action = "qe"

# myst-parser, forcing to parse all html pages with mathjax
# https://github.com/executablebooks/MyST-Parser/issues/394
myst_update_mathjax = False
myst_heading_anchors = 3


# -- Options for autodoc-------------------------------------------------
autodoc_default_options = {
    "members": True,
    "methods": True,
    "member-order": "bysource",
    "special-members": "__call__,__init__",
    "exclude-members": "groups,__dict__,__weakref__",
    # "autosummary": True,
    # "autosummary-imported-members": False,
    "show-inheritance": True,
    # "imported_mebmbers": False,
}
# autodoc: Don't inherit docstrings (e.g. for nn.Module.forward)
autodoc_inherit_docstrings = False
# autosummary_generate = True
typehints_fully_qualified = False

nitpicky = False
nitpick_ignore = [("type", "py:class")]
# nitpick_ignore = [('py:class', 'type')]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Sphinx will add “permalinks” for each heading and description environment as paragraph signs that
#  become visible when the mouse hovers over them.
# This value determines the text for the permalink; it defaults to "¶".
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_add_permalinks
html_permalinks = True
html_permalinks_icon = "#"

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"  # igor is also good
# pygments_style = "sphinx"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
html_theme = "furo"

html_title = "Documentation"
# Disable the generation of the various indexes
html_use_modindex = False

# If false, no index is generated.
html_use_index = True


# html_logo = "logo.svg"
html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
]
html_theme_options = {
    "top_of_page_button": "edit",
    "source_repository": "https://github.com/wangers/subtools2",
    "source_branch": "master",
    "source_directory": "docs/source",
    "announcement": "<em>Feedback welcomed 🎉 at "
    "<a href='http://xxx'>dummy url</a></em>",
    # "sidebar_hide_name": True,  # hide if we have a logo
    # "light_css_variables": {
    #     "font-stack": "Inter,sans-serif",
    #     "font-stack--monospace": "BerkeleyMono, MonoLisa, ui-monospace, "
    #     "SFMono-Regular, Menlo, Consolas, Liberation Mono, monospace",
    # },
    # "footer_icons": [
    #     {
    #         "name": "GitHub",
    #         "url": "https://github.com/pradyunsg/furo",
    #         "html": "",
    #         "class": "fa-brands fa-solid fa-github fa-2x",
    #     },
    # ],
}


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "Egrecho",
        "Documentation for Egrecho",
        author,
        "Egrecho",
        "xxx",
        "xxx",
    )
]


def package_list_from_file(file):
    """List up package name (not containing version and extras) from a package list file."""
    mocked_packages = []
    with open(file) as fp:
        for ln in fp.readlines():
            # Example: `tqdm>=4.41.0` => `tqdm`
            # `[` is for package with extras
            found = [ln.index(ch) for ch in list(",=<>#[") if ch in ln]
            pkg = ln[: min(found)] if found else ln
            if pkg.rstrip():
                mocked_packages.append(pkg.rstrip())
    return mocked_packages


# define mapping from PyPI names to python imports
PACKAGE_MAPPING = {"pytorch": "torch"}
PACKAGE_MAPPING = {"lightning": "lightning"}
MOCK_PACKAGES = []

_path_require = lambda fname: os.path.join(_PATH_ROOT, "requirements", fname)  # noqa
# mock also base packages when we are on RTD since we don't install them there
MOCK_PACKAGES += package_list_from_file(_path_require("base.txt"))
MOCK_PACKAGES = [PACKAGE_MAPPING.get(pkg, pkg) for pkg in MOCK_PACKAGES]
autodoc_mock_imports = MOCK_PACKAGES


def linkcode_resolve(domain, info):
    # https://github.com/facebookresearch/ParlAI/blob/main/docs/source/conf.py
    # Resolve function for the linkcode extension.
    # Stolen shamelessly from Lasagne! Thanks Lasagne!
    # https://github.com/Lasagne/Lasagne/blob/5d3c63cb315c50b1cbd27a6bc8664b406f34dd99/docs/conf.py#L114-L135
    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/main/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(egrecho.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "egrecho/%s#L%d-L%d" % find_source()
        # tag = git.Git().rev_parse("HEAD")
        # return "https://github.com/wangers/subtools2/tree/master/%s/%s" % (
        #     tag,
        #     filename,
        # )
        return "https://github.com/wangers/subtools2/tree/master/%s" % (filename,)
    except Exception:
        return None
