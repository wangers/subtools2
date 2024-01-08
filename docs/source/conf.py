# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

import lightning  # import member need this

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
    "sphinx.ext.githubpages",
    "sphinx.ext.todo",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    # "sphinx.ext.autosummary",
    # "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_paramlinks",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
]
# maximum_signature_line_length = 80
copybutton_prompt_text = ">>> "
copybutton_prompt_text1 = "... "
copybutton_exclude = ".linenos"

# templates_path = ["_templates"]
exclude_patterns = []

# The master toctree document.
master_doc = "index"

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
    ".ipynb": "nbsphinx",
}

# myst options
myst_enable_extensions = [
    # "amsmath",
    # "attrs_inline",
    # "colon_fence",
    "deflist",
    # "dollarmath",
    "fieldlist",
    # "html_admonition",
    # "html_image",
    # "replacements",
    # "smartquotes",
    # "strikethrough",
    # "substitution",
    # "tasklist",
]
# myst-parser, forcing to parse all html pages with mathjax
# https://github.com/executablebooks/MyST-Parser/issues/394
myst_update_mathjax = False
# myst_heading_anchors = 3

nitpicky = False
nitpick_ignore = [("type", "py:class")]
# nitpick_ignore = [('py:class', 'type')]

# autodoc_default_options = {
#     "members": True,
#     "exclude-members": "groups,__dict__,__weakref__",
#     "member-order": "bysource",
#     "show-inheritance": False,
#     "autosummary": True,
#     "autosummary-imported-members": False,
#     "undoc-members": True,
#     "special-members": "__init__,__call__",
# }
autodoc_default_options = {
    "members": True,
    # "methods": True,
    "member-order": "bysource",
    "special-members": "__call__,__init__",
    "exclude-members": "groups,__dict__,__weakref__",
    "show-inheritance": False,
}
# autodoc: Don't inherit docstrings (e.g. for nn.Module.forward)
# autodoc_inherit_docstrings = False
# autosummary_generate = True
typehints_fully_qualified = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "friendly"  # igor is also good

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
}
html_show_sourcelink = False
toc_object_entries = True
toc_object_entries_show_parents = "hide"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ["_static"]
# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    # "python": ("https://docs.python.org/3", None),
    # "torch": ("https://pytorch.org/docs/stable/", None),
}


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
PACKAGE_MAPPING = {"torch": "torch"}
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
