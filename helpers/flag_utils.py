"""Platform-specific flag rendering utilities.

On Mac (darwin): Uses emoji flags for better appearance
On Windows/Linux: Uses image flags from flagcdn.com (emoji don't render well)
"""

import sys

# Codes that represent emerging markets / global (get globe icon instead of flag)
EM_CODES = {'un', 'em', 'emerging', 'world', 'global'}

# Herculis brand colors for EM globe icon
HERCULIS_NAVY = "#0d1b40"
HERCULIS_GOLD = "#c5a258"

# SVG globe icon for emerging markets (navy with gold accents)
EM_GLOBE_SVG = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="22" height="22">
  <circle cx="12" cy="12" r="10" fill="{HERCULIS_NAVY}" stroke="{HERCULIS_GOLD}" stroke-width="1.5"/>
  <ellipse cx="12" cy="12" rx="4" ry="10" fill="none" stroke="{HERCULIS_GOLD}" stroke-width="0.8"/>
  <line x1="2" y1="12" x2="22" y2="12" stroke="{HERCULIS_GOLD}" stroke-width="0.8"/>
  <path d="M3.5 7.5 Q12 9 20.5 7.5" fill="none" stroke="{HERCULIS_GOLD}" stroke-width="0.6"/>
  <path d="M3.5 16.5 Q12 15 20.5 16.5" fill="none" stroke="{HERCULIS_GOLD}" stroke-width="0.6"/>
</svg>'''

# Emoji flags mapping (country code -> emoji)
FLAG_EMOJI = {
    'us': '🇺🇸',
    'jp': '🇯🇵',
    'de': '🇩🇪',
    'ch': '🇨🇭',
    'br': '🇧🇷',
    'mx': '🇲🇽',
    'cn': '🇨🇳',
    'in': '🇮🇳',
    'sa': '🇸🇦',
    'gb': '🇬🇧',
    'eu': '🇪🇺',
    'au': '🇦🇺',
    'ca': '🇨🇦',
    'fr': '🇫🇷',
    'it': '🇮🇹',
    'es': '🇪🇸',
    'kr': '🇰🇷',
    'hk': '🇭🇰',
    'sg': '🇸🇬',
    'se': '🇸🇪',
    'no': '🇳🇴',
    'dk': '🇩🇰',
    'nl': '🇳🇱',
    'be': '🇧🇪',
    'at': '🇦🇹',
    'pl': '🇵🇱',
    'ru': '🇷🇺',
    'za': '🇿🇦',
    'tr': '🇹🇷',
    'il': '🇮🇱',
    'ae': '🇦🇪',
    'th': '🇹🇭',
    'id': '🇮🇩',
    'my': '🇲🇾',
    'ph': '🇵🇭',
    'vn': '🇻🇳',
    'tw': '🇹🇼',
    'nz': '🇳🇿',
    'ar': '🇦🇷',
    'cl': '🇨🇱',
    'co': '🇨🇴',
    'pe': '🇵🇪',
}


def get_flag_html(country_code: str, size: int = 22) -> str:
    """Return flag HTML - emoji on Mac, PNG on Windows/Linux.

    Note: Size is controlled by CSS in the templates (scaled appropriately).
    Special handling: EM/global codes return a globe icon with Herculis branding.

    Args:
        country_code: ISO 3166-1 alpha-2 country code (e.g., 'us', 'jp')
                      or EM code ('un', 'em', 'emerging', 'world', 'global')
        size: Not used - kept for backward compatibility

    Returns:
        HTML string for the flag or globe icon
    """
    code = country_code.lower()

    # Check for emerging market / global codes - return globe icon
    if code in EM_CODES:
        return f'<span class="flag em-globe" style="display:inline-flex; align-items:center;">{EM_GLOBE_SVG}</span>'

    if sys.platform == 'darwin':  # Mac - use emoji flags
        emoji = FLAG_EMOJI.get(code, '🏳️')
        return f'<span class="flag">{emoji}</span>'
    else:  # Windows/Linux - use PNG images
        return f'<img class="flag-img" src="https://flagcdn.com/w40/{code}.png" style="vertical-align:middle; flex-shrink:0;">'


def get_flag_css() -> str:
    """Return CSS needed for flags (only needed on Mac for emoji sizing)."""
    if sys.platform == 'darwin':
        return ''
    else:
        return ''  # No extra CSS needed for images


def is_mac() -> bool:
    """Check if running on Mac."""
    return sys.platform == 'darwin'
