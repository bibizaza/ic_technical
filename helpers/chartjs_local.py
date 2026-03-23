"""Local Chart.js loader to avoid CDN dependency during Playwright rendering.

During the assemble pipeline, Playwright launches multiple Chromium instances
in rapid succession.  Loading Chart.js from the jsdelivr CDN can fail
intermittently because Chromium flags the cross-site script as
parser-blocking and may block the network request on subsequent page loads.

This module loads Chart.js (and the annotation plugin) from local assets
and provides inline ``<script>`` tags that can be substituted into HTML
templates before rendering.
"""

from __future__ import annotations

import pathlib

_ASSETS_DIR = pathlib.Path(__file__).resolve().parents[1] / "assets"

_CDN_CHARTJS = '<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>'
_CDN_ANNOTATION = '<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation"></script>'


def _load_asset(filename: str) -> str | None:
    path = _ASSETS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


_chartjs_src = _load_asset("chart.umd.min.js")
_annotation_src = _load_asset("chartjs-plugin-annotation.min.js")

INLINE_CHARTJS = f"<script>{_chartjs_src}</script>" if _chartjs_src else _CDN_CHARTJS
INLINE_ANNOTATION = f"<script>{_annotation_src}</script>" if _annotation_src else _CDN_ANNOTATION


def patch_cdn(html: str) -> str:
    """Replace CDN ``<script>`` tags with inline local Chart.js code.

    Safe to call on any HTML string — returns unchanged if no CDN tags found.
    """
    html = html.replace(_CDN_CHARTJS, INLINE_CHARTJS)
    html = html.replace(_CDN_ANNOTATION, INLINE_ANNOTATION)
    return html
