"""Platform-specific flag rendering utilities.

On Mac (darwin): Uses emoji flags for better appearance
On Windows/Linux: Uses image flags from flagcdn.com (emoji don't render well)
"""

import os
import sys

# Path to crypto logos
CRYPTO_ASSETS_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'crypto')

# Crypto logo mapping (all 10 cryptos)
CRYPTO_LOGOS = {
    'btc': 'btc.png',
    'eth': 'eth.png',
    'xrp': 'xrp.png',
    'sol': 'sol.png',
    'bnb': 'bnb.png',
    'dot': 'dot.png',
    'aave': 'aave.png',
    'ton': 'ton.png',
    'hyper': 'hyper.png',
    'bloomberg': 'bloomberg.jpeg',
}

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

    # Check for crypto logos FIRST
    if code in CRYPTO_LOGOS:
        logo_file = CRYPTO_LOGOS[code]
        logo_path = os.path.join(CRYPTO_ASSETS_PATH, logo_file)
        # Double the size to match country flags visually
        img_size = int(size * 2)
        return f'<img class="flag-img" src="file://{logo_path}" style="width:{img_size}px; height:{img_size}px; border-radius:50%; vertical-align:middle; flex-shrink:0; object-fit:cover;">'

    # Emerging Markets - use globe emoji (Mac) or UN flag PNG (Windows/Linux)
    if code in EM_CODES:
        if sys.platform == 'darwin':
            # Emojis need larger font-size to match PNG flag dimensions
            emoji_size = int(size * 3.0)
            return f'<span class="flag" style="font-size:{emoji_size}px; line-height:1; vertical-align:middle;">🌍</span>'
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
