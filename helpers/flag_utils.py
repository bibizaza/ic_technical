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

# SVG globe icon with Africa silhouette for emerging markets
EM_GLOBE_SVG = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="22" height="22" style="vertical-align: middle;">
  <!-- Globe outline -->
  <circle cx="50" cy="50" r="45" fill="{HERCULIS_LIGHT_BLUE}" stroke="{HERCULIS_NAVY}" stroke-width="2"/>

  <!-- Globe grid lines -->
  <ellipse cx="50" cy="50" rx="20" ry="45" fill="none" stroke="{HERCULIS_NAVY}" stroke-width="0.5" opacity="0.4"/>
  <line x1="5" y1="50" x2="95" y2="50" stroke="{HERCULIS_NAVY}" stroke-width="0.5" opacity="0.4"/>

  <!-- Africa silhouette (simplified) -->
  <path d="M55 18 L60 20 L62 16 L66 20 L70 28 L72 38 L70 48 L74 56 L70 66 L62 78 L52 84 L46 80 L42 70 L38 58 L40 48 L38 40 L42 32 L48 24 L52 20 Z"
        fill="{HERCULIS_NAVY}" stroke="{HERCULIS_GOLD}" stroke-width="1.5"/>

  <!-- Europe hint (small) -->
  <path d="M46 22 L52 18 L56 20 L54 26 L48 26 Z"
        fill="{HERCULIS_NAVY}" opacity="0.5"/>
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
    Special handling: EM/global codes return a globe icon with Africa silhouette.

    Args:
        country_code: ISO 3166-1 alpha-2 country code (e.g., 'us', 'jp')
                      or EM code ('un', 'em', 'emerging', 'world', 'global')
        size: Not used - kept for backward compatibility

    Returns:
        HTML string for the flag or globe icon
    """
    # Handle None or empty
    if not country_code:
        return f'<span class="flag em-globe" style="display:inline-flex; align-items:center;">{EM_GLOBE_SVG}</span>'

    code = country_code.lower().strip()

    # Check for emerging market / global codes - return Africa/globe icon
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
