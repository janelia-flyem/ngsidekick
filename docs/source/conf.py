# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import inspect

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ngsidekick'
copyright = '2025, HHMI'
author = 'Stuart Berg'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

# Support both .rst and .md files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# MyST-Parser configuration
myst_enable_extensions = [
    "colon_fence",      # ::: fences for directives
    "deflist",          # Definition lists
    "fieldlist",        # Field lists
    "html_image",       # HTML images
    "linkify",          # Auto-convert URLs to links
    "replacements",     # Text replacements
    "smartquotes",      # Smart quotes
    "tasklist",         # Task lists with [ ] and [x]
]

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']


# -- linkcode extension configuration ----------------------------------------
# Generate links to GitHub source code

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to a Python object.
    """
    if domain != 'py':
        return None
    
    modname = info['module']
    fullname = info['fullname']
    
    # Import the module
    try:
        import importlib
        module = importlib.import_module(modname)
    except Exception:
        return None
    
    # Get the object
    obj = module
    for part in fullname.split('.'):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
    
    # Unwrap decorators
    try:
        obj = inspect.unwrap(obj)
    except Exception:
        pass
    
    # Get the source file
    try:
        source_file = inspect.getsourcefile(obj)
        if source_file is None:
            return None
    except Exception:
        return None
    
    # Get line numbers
    try:
        source, lineno = inspect.getsourcelines(obj)
        linespec = f"#L{lineno}-L{lineno + len(source) - 1}"
    except Exception:
        linespec = ""
    
    # Convert absolute path to relative path within the repo
    # The source files are in src/ngsidekick/
    try:
        # Find the 'src' directory in the path
        parts = source_file.split(os.sep)
        if 'src' in parts:
            idx = parts.index('src')
            relpath = os.path.join(*parts[idx:])
        else:
            return None
    except Exception:
        return None
    
    # Construct GitHub URL
    github_url = f"https://github.com/janelia-flyem/ngsidekick/blob/main/{relpath}{linespec}"
    return github_url
