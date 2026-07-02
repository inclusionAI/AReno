project = "AReno"
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
html_title = "AReno docs"
html_static_path = ["_static"]
html_css_files = ["areno.css"]
html_js_files = ["areno-sidebar.js"]
html_show_sourcelink = False

html_theme_options = {
    "accent_color": "gray",
    "light_logo": "_static/areno-logo.svg",
    "dark_logo": "_static/areno-logo-dark.svg",
    "ethical_ads_publisher": "",
    "github_url": "",
    "nav_links": [
        {"title": "Get Started", "url": "getting-started/welcome"},
        {"title": "Concepts", "url": "concepts/training-loop"},
        {"title": "Cookbook", "url": "cookbook/math-rlvr"},
        {"title": "Reference", "url": "reference/cli"},
        {"title": "Troubleshooting", "url": "troubleshooting/index"},
    ],
}
