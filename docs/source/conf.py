# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = "swtipy"
author = "Man Chung Yuen"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",      # 自動讀取 Python docstring
    "sphinx.ext.napoleon",     # 支援 Google/NumPy style docstring
    "sphinx.ext.viewcode",     # 在文件裡顯示原始碼連結
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Important: set master doc -----------------------------------------------
master_doc = "index"
