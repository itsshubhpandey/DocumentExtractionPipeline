"""System-preference-aware dark theme for the /docs Swagger UI.

fastapi_swagger_dark ships a CSS file that always forces the dark theme,
regardless of the OS/browser setting. This wraps that same CSS in a
`prefers-color-scheme: dark` media query instead, so Swagger UI follows the
viewer's system theme: dark when the OS is dark, and FastAPI's normal light
Swagger UI (untouched) otherwise.
"""

from __future__ import annotations

from pathlib import Path

import fastapi_swagger_dark
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import PlainTextResponse

_DARK_CSS_PATH = Path(fastapi_swagger_dark.__file__).parent / "swagger_ui_dark.min.css"
_DARK_CSS = _DARK_CSS_PATH.read_text(encoding="utf-8")

# Nested @media queries already inside this CSS (responsive breakpoints)
# still work correctly nested one level deeper -- a browser applies them
# only when both the outer (dark mode) and inner (e.g. max-width) condition
# are true, which is exactly what's wanted here.
AUTO_DARK_SWAGGER_CSS = f"@media (prefers-color-scheme: dark) {{\n{_DARK_CSS}\n}}"


def install_system_theme_docs(app: FastAPI, path: str = "/docs") -> None:
    """Register a /docs route whose Swagger UI follows the system theme.

    Call this with `docs_url=None` passed to FastAPI(...) -- otherwise
    FastAPI's own built-in /docs route is registered first and always wins.
    """

    @app.get(path, include_in_schema=False)
    def swagger_docs():
        return get_swagger_ui_html(
            openapi_url=app.openapi_url or "/openapi.json",
            title=f"{app.title} - Swagger UI",
            swagger_css_url="/dark_theme.css",
        )

    @app.get("/dark_theme.css", include_in_schema=False)
    def swagger_theme_css() -> PlainTextResponse:
        return PlainTextResponse(AUTO_DARK_SWAGGER_CSS, media_type="text/css")
