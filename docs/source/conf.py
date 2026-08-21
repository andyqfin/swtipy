# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = "swtipy"
author = "Man Chung Yuen"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",   # ← 加上這行
]

autosummary_generate = True  # 自動生成 API 條目

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Important: set master doc -----------------------------------------------
master_doc = "index"
