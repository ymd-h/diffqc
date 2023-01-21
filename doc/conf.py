project = "diffqc"
author = "Hiroyuki Yamada"
copyright = "2023, Hiroyuki Yamada"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    'sphinx.ext.napoleon',
    'myst_parser'
]

html_theme = "sphinx_rtd_theme"
html_logo = ""
html_favicon = ""
html_show_sourcelink = False

napoleon_include_init_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True

autosummary_generate = True
autodoc_default_options = {
    'member-order': 'bysource',
    'class-doc-from':'both',
    'exclude-members': '__dict__, __weakref__, __module__, __new__, __reduce__, __setstate__'
}
