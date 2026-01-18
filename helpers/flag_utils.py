"""Platform-specific flag rendering utilities.

On Mac (darwin): Uses emoji flags for better appearance
On Windows/Linux: Uses image flags from flagcdn.com (emoji don't render well)
"""

import sys

# Codes that represent emerging markets / global (get globe emoji/image instead of flag)
EM_CODES = {'un', 'em', 'emerging', 'world', 'global', ''}

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

    Uses simple PNG images (like country flags) to avoid layout issues.
    EM/global codes use globe emoji (Mac) or UN flag PNG (Windows/Linux).

    Args:
        country_code: ISO 3166-1 alpha-2 country code (e.g., 'us', 'jp')
                      or EM code ('un', 'em', 'emerging', 'world', 'global')
        size: Icon size in pixels (default 22px to match flag emojis)

    Returns:
        HTML string for the flag or globe icon
    """
    # Handle None or empty - treat as EM
    if not country_code:
        country_code = "em"

    code = country_code.lower().strip()

    # Emerging Markets - use globe emoji (Mac) or UN flag PNG (Windows/Linux)
    if code in EM_CODES:
        if sys.platform == 'darwin':
            return f'<span class="flag" style="font-size:{size}px; line-height:1;">🌍</span>'
        else:
            return f'<img class="flag-img" src="https://flagcdn.com/w40/un.png" style="width:{size}px; height:auto; vertical-align:middle; flex-shrink:0;">'

    # Regular country flags
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
