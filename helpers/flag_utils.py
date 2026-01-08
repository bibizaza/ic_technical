"""Platform-specific flag rendering utilities.

On Mac (darwin): Uses emoji flags for better appearance
On Windows/Linux: Uses image flags from flagcdn.com (emoji don't render well)
"""

import sys

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


def get_flag_html(country_code: str, size: int = 20) -> str:
    """Return flag HTML - emoji on Mac, image on Windows/Linux.

    Args:
        country_code: ISO 3166-1 alpha-2 country code (e.g., 'us', 'jp')
        size: Size in pixels

    Returns:
        HTML string for the flag
    """
    code = country_code.lower()

    if sys.platform == 'darwin':  # Mac
        # Use emoji flags
        emoji = FLAG_EMOJI.get(code, '🏳️')
        return f'<span style="font-size:{size}px; line-height:1;">{emoji}</span>'
    else:  # Windows/Linux
        # Use image flags from CDN
        return f'<img src="https://flagcdn.com/24x18/{code}.png" style="width:{size}px; height:auto; vertical-align:middle;">'


def get_flag_css() -> str:
    """Return CSS needed for flags (only needed on Mac for emoji sizing)."""
    if sys.platform == 'darwin':
        return ''
    else:
        return ''  # No extra CSS needed for images


def is_mac() -> bool:
    """Check if running on Mac."""
    return sys.platform == 'darwin'
