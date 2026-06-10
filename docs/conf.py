project = "areno"
author = "Areno contributors"
copyright = "2026, Areno contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "shibuya"
html_title = "areno"
html_static_path = ["_static"]
html_css_files = ["areno.css"]
html_show_sourcelink = False

html_theme_options = {
    "accent_color": "violet",
    "github_url": "",
    "nav_links": [
        {"title": "Training", "url": "cli/training"},
        {"title": "Inference", "url": "cli/inference"},
        {"title": "SDK", "url": "sdk/trainer"},
        {"title": "Models", "url": "models/supported"},
    ],
}
