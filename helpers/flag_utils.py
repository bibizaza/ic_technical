"""Platform-specific flag rendering utilities.

On Mac (darwin): Uses emoji flags for better appearance
On Windows/Linux: Uses image flags from flagcdn.com (emoji don't render well)
"""

import sys

# Codes that represent emerging markets / global (get globe icon instead of flag)
EM_CODES = {'un', 'em', 'emerging', 'world', 'global', ''}

# Herculis brand colors for EM globe icon
HERCULIS_NAVY = "#1B3A5A"
HERCULIS_GOLD = "#C9A227"
HERCULIS_LIGHT_BLUE = "#E8F0F8"


def get_em_globe_svg(size: int = 22) -> str:
    """Return Africa/Globe SVG icon for Emerging Markets at specified size.

    Uses viewBox="0 0 24 24" to match output size directly, with explicit
    size styles to prevent CSS from shrinking the icon.
    """
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{size}" height="{size}" style="width: {size}px; height: {size}px; min-width: {size}px; min-height: {size}px; vertical-align: middle; flex-shrink: 0;">
  <circle cx="12" cy="12" r="10" fill="{HERCULIS_LIGHT_BLUE}" stroke="{HERCULIS_NAVY}" stroke-width="1"/>
  <ellipse cx="12" cy="12" rx="4" ry="10" fill="none" stroke="{HERCULIS_NAVY}" stroke-width="0.5" opacity="0.4"/>
  <line x1="2" y1="12" x2="22" y2="12" stroke="{HERCULIS_NAVY}" stroke-width="0.5" opacity="0.4"/>
  <path d="M13 4 L14.5 5 L15 4 L16 5 L17 7 L17.5 9 L17 12 L18 14 L17 17 L15 19 L12.5 20 L11 19 L10 17 L9 14 L9.5 12 L9 10 L10 8 L11.5 6 L12.5 5 Z" fill="{HERCULIS_NAVY}" stroke="{HERCULIS_GOLD}" stroke-width="0.5"/>
</svg>'''


# Legacy constant for backward compatibility (uses default size)
EM_GLOBE_SVG = get_em_globe_svg(22)

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

    Note: Size parameter now properly controls EM globe icon size.
    Special handling: EM/global codes return a globe icon with Africa silhouette.

    Args:
        country_code: ISO 3166-1 alpha-2 country code (e.g., 'us', 'jp')
                      or EM code ('un', 'em', 'emerging', 'world', 'global')
        size: Icon size in pixels (default 22px to match flag emojis)

    Returns:
        HTML string for the flag or globe icon
    """
    # Handle None or empty - return EM globe
    if not country_code:
        return f'<span class="flag em-globe" style="display:inline-flex; align-items:center; width:{size}px; height:{size}px;">{get_em_globe_svg(size)}</span>'

    code = country_code.lower().strip()

    # Check for emerging market / global codes - return Africa/globe icon
    if code in EM_CODES:
        return f'<span class="flag em-globe" style="display:inline-flex; align-items:center; width:{size}px; height:{size}px;">{get_em_globe_svg(size)}</span>'

    if sys.platform == 'darwin':  # Mac - use emoji flags
        emoji = FLAG_EMOJI.get(code, '🏳️')
        return f'<span class="flag">{emoji}</span>'
    else:  # Windows/Linux - use PNG images
        return f'<img class="flag-img" src="https://flagcdn.com/w40/{code}.png" style="width:{size}px; height:auto; vertical-align:middle; flex-shrink:0;">'


def get_flag_css() -> str:
    """Return CSS needed for flags (only needed on Mac for emoji sizing)."""
    if sys.platform == 'darwin':
        return ''
    else:
        return ''  # No extra CSS needed for images


def is_mac() -> bool:
    """Check if running on Mac."""
    return sys.platform == 'darwin'
